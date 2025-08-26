from __future__ import annotations
from pathlib import Path
import json
from typing import Iterable, Dict, List

import numpy as np
import pandas as pd
import yaml

#  helpers 

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def load_paths() -> Dict[str, Path]:
    cfg = yaml.safe_load((project_root() / "configs/paths.yaml").read_text())
    return {k: (project_root() / v) for k, v in cfg.items()}

def load_feature_cfg() -> Dict:
    p = project_root() / "configs/features.yaml"
    if p.exists():
        return yaml.safe_load(p.read_text())
    return {"windows_days": [7, 30, 90]}

def _read_any(base: Path, stem: str) -> pd.DataFrame:
    pq = base / f"{stem}.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    csv = base / f"{stem}.csv"
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Expected {pq} or {csv}")

def _coerce_dt(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_datetime(df[c])
    return df

def _window_mask(ts: pd.Series, as_of: pd.Timestamp, days: int) -> pd.Series:
    start = as_of - pd.Timedelta(days=days)
    return (ts > start) & (ts <= as_of)

#  core feature logic
def build_features() -> pd.DataFrame:
    paths = load_paths()
    raw_dir = paths["raw_dir"]
    processed_dir = paths["processed_dir"]
    artifacts_dir = paths["artifacts_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Load tables
    orders   = _read_any(raw_dir, "orders")
    shipments = _read_any(raw_dir, "shipments")
    reviews  = _read_any(raw_dir, "reviews")
    sellers  = _read_any(raw_dir, "sellers")

    # Coerce datetimes
    orders   = _coerce_dt(orders, ["timestamp"])
    shipments = _coerce_dt(shipments, ["ship_date", "promised_date", "delivered_date"])
    reviews  = _coerce_dt(reviews, ["created_at"])
    sellers  = sellers.copy()

    # Join order-centric view for feature calc
    od = orders.merge(
        shipments, on="order_id", how="left", validate="one_to_one"
    ).merge(
        reviews[["order_id", "stars"]], on="order_id", how="left", validate="one_to_one"
    )

    # Sanity: derive lateness & latencies
    od["late"] = (od["delivered_date"] > od["promised_date"]).astype(int)
    od["time_to_ship_days"] = (od["ship_date"] - od["timestamp"]).dt.total_seconds() / 86400.0
    od["transit_days"] = (od["delivered_date"] - od["ship_date"]).dt.total_seconds() / 86400.0

    as_of = pd.to_datetime(
        max(od["timestamp"].max(), od["delivered_date"].max(), reviews["created_at"].max())
    )

    cfg = load_feature_cfg()
    windows: List[int] = list(cfg.get("windows_days", [7, 30, 90]))
    if 180 not in windows:
        windows.append(180)
    windows = sorted(set(int(d) for d in windows))

    last30_mask = _window_mask(od["timestamp"], as_of, 30)
    od_last30 = od.loc[last30_mask].copy()
    if not od_last30.empty:
        od_last30["day"] = od_last30["timestamp"].dt.floor("D")
        daily = od_last30.groupby(["seller_id", "day"], as_index=False).size()
        last_day = as_of.floor("D")
        daily_last = daily[daily["day"].eq(last_day)][["seller_id", "size"]].rename(columns={"size": "orders_last_day"})
        stats = (daily.groupby("seller_id", as_index=False)["size"]
                 .agg(orders_daily_mean_30d="mean", orders_daily_std_30d="std"))
        spike = stats.merge(daily_last, on="seller_id", how="left")
        spike["orders_last_day"] = spike["orders_last_day"].fillna(0.0)
        spike["orders_z_last_30d"] = (spike["orders_last_day"] - spike["orders_daily_mean_30d"]) / spike["orders_daily_std_30d"]
        spike["orders_z_last_30d"] = spike["orders_z_last_30d"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    else:
        spike = pd.DataFrame(columns=["seller_id", "orders_daily_mean_30d", "orders_daily_std_30d", "orders_last_day", "orders_z_last_30d"])

    # Window aggregations (with Laplace-smoothed rates)
    feats_by_window: List[pd.DataFrame] = []
    for d in windows:
        mask = _window_mask(od["timestamp"], as_of, d)
        w = od.loc[mask].copy()
        if w.empty:
            agg = pd.DataFrame({"seller_id": sellers["seller_id"]}).assign(
                **{
                    f"orders_{d}d": 0,
                    f"amount_sum_{d}d": 0.0,
                    f"refund_sum_{d}d": 0.0,
                    f"late_sum_{d}d": 0.0,
                    f"avg_review_{d}d": np.nan,
                    f"time_to_ship_avg_{d}d": np.nan,
                    f"transit_avg_{d}d": np.nan,
                }
            )
        else:
            grp = w.groupby("seller_id", as_index=False)
            agg = grp.agg(
                **{
                    f"orders_{d}d": ("order_id", "size"),
                    f"amount_sum_{d}d": ("amount", "sum"),
                    f"refund_sum_{d}d": ("refund_flag", "sum"),
                    f"late_sum_{d}d": ("late", "sum"),
                    f"avg_review_{d}d": ("stars", "mean"),
                    f"time_to_ship_avg_{d}d": ("time_to_ship_days", "mean"),
                    f"transit_avg_{d}d": ("transit_days", "mean"),
                }
            )

        # classical (unsmoothed) rates (keep for continuity)
        orders_col = f"orders_{d}d"
        agg[f"refund_rate_{d}d"] = np.where(agg[orders_col] > 0, agg[f"refund_sum_{d}d"] / agg[orders_col], 0.0)
        agg[f"late_rate_{d}d"]   = np.where(agg[orders_col] > 0, agg[f"late_sum_{d}d"]   / agg[orders_col], 0.0)

        # Laplace-smoothed rates: (sum + 1) / (orders + 2)
        agg[f"refund_rate_{d}d_sm"] = (agg[f"refund_sum_{d}d"] + 1.0) / (agg[orders_col] + 2.0)
        agg[f"late_rate_{d}d_sm"]   = (agg[f"late_sum_{d}d"]   + 1.0) / (agg[orders_col] + 2.0)

        # drop helper sums from final features
        agg = agg.drop(columns=[f"refund_sum_{d}d", f"late_sum_{d}d"])

        feats_by_window.append(agg)

    # Customer-mix features over 90d (distinct buyers, repeat rate)
    mask90 = _window_mask(od["timestamp"], as_of, 90)
    w90 = od.loc[mask90, ["seller_id", "buyer_id"]].copy()
    if w90.empty:
        mix = pd.DataFrame({"seller_id": sellers["seller_id"],
                            "distinct_buyers_90d": 0,
                            "repeat_rate_90d": 0.0})
    else:
        counts = (w90
                  .groupby(["seller_id", "buyer_id"], as_index=False)
                  .size()
                  .rename(columns={"size": "orders_per_buyer"}))
        buyers = counts.groupby("seller_id", as_index=False)["buyer_id"].nunique().rename(columns={"buyer_id": "distinct_buyers_90d"})
        repeaters = counts[counts["orders_per_buyer"] >= 2].groupby("seller_id", as_index=False)["buyer_id"].nunique().rename(columns={"buyer_id": "repeat_buyer_cnt"})
        mix = buyers.merge(repeaters, on="seller_id", how="left")
        mix["repeat_buyer_cnt"] = mix["repeat_buyer_cnt"].fillna(0)
        mix["repeat_rate_90d"] = (mix["repeat_buyer_cnt"] / mix["distinct_buyers_90d"].replace(0, np.nan)).fillna(0.0)
        mix = mix.drop(columns=["repeat_buyer_cnt"])

    # Merge all features onto full seller list
    features = sellers[["seller_id"]].copy()
    for f in feats_by_window:
        features = features.merge(f, on="seller_id", how="left")
    features = features.merge(spike, on="seller_id", how="left")
    features = features.merge(mix, on="seller_id", how="left")

    # Fill NA and ensure numeric dtype
    num_cols = [c for c in features.columns if c != "seller_id"]
    features[num_cols] = features[num_cols].fillna(0.0)
    # make everything float for model friendliness
    features[num_cols] = features[num_cols].astype(float)

    # Persist
    out_path_pq = project_root() / "data" / "processed" / "seller_features.parquet"
    try:
        features.to_parquet(out_path_pq, index=False)
        out_path = out_path_pq
    except Exception:
        out_csv = out_path_pq.with_suffix(".csv")
        features.to_csv(out_csv, index=False)
        out_path = out_csv

    # Feature list for serving (exclude id)
    feat_list = [c for c in features.columns if c != "seller_id"]
    (project_root() / "artifacts").mkdir(parents=True, exist_ok=True)
    (project_root() / "artifacts" / "feature_list.json").write_text(json.dumps(feat_list, indent=2))

    # Feature metadata (kind + window where applicable)
    kinds: Dict[str, str] = {}
    def _kind(name: str) -> str:
        if name.startswith("orders_") and name.endswith("d"): return "count"
        if name.startswith("amount_sum_"): return "sum"
        if name.endswith("_rate_7d") or name.endswith("_rate_30d") or name.endswith("_rate_90d") or name.endswith("_rate_180d"):
            return "rate"
        if name.endswith("_rate_7d_sm") or name.endswith("_rate_30d_sm") or name.endswith("_rate_90d_sm") or name.endswith("_rate_180d_sm"):
            return "rate_smoothed"
        if name.startswith("avg_review_"): return "avg"
        if name.startswith("time_to_ship_avg_") or name.startswith("transit_avg_"): return "avg"
        if name in {"orders_daily_mean_30d", "orders_daily_std_30d", "orders_last_day"}: return "daily_stat"
        if name == "orders_z_last_30d": return "spike"
        if name == "distinct_buyers_90d": return "distinct"
        if name == "repeat_rate_90d": return "rate"
        return "other"

    kinds = {f: _kind(f) for f in feat_list}
    meta = {
        "as_of": as_of.isoformat(),
        "windows_days": windows,
        "n_features": len(feat_list),
        "features": feat_list,
        "kinds": kinds,
    }
    (project_root() / "artifacts" / "feature_metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"✅ Built features as of {as_of.isoformat()}")
    print(f"   → {out_path}")
    print(f"   → artifacts/feature_list.json ({len(feat_list)} features)")
    print(f"   → artifacts/feature_metadata.json")
    return features

if __name__ == "__main__":
    build_features()
