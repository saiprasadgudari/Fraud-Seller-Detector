from __future__ import annotations
from pathlib import Path
import json, yaml, joblib
import numpy as np, pandas as pd

def root() -> Path: return Path(__file__).resolve().parents[2]
def load_paths():
    cfg = yaml.safe_load((root() / "configs/paths.yaml").read_text())
    return {k: (root() / v) for k, v in cfg.items()}

def _read_any(base: Path, stem: str) -> pd.DataFrame:
    pq = base / f"{stem}.parquet"
    if pq.exists(): return pd.read_parquet(pq)
    csv = base / f"{stem}.csv"
    if csv.exists(): return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing {stem} in {base}")

def first_order_ts_per_seller(orders: pd.DataFrame) -> pd.DataFrame:
    tmp = orders[["seller_id","timestamp"]].copy()
    tmp["timestamp"] = pd.to_datetime(tmp["timestamp"])
    return tmp.groupby("seller_id", as_index=False)["timestamp"].min().rename(columns={"timestamp":"first_order_ts"})

def build_labels(orders: pd.DataFrame, min_fraud_orders: int) -> pd.DataFrame:
    if "fraud_signal" not in orders.columns:
        raise RuntimeError("orders lacks fraud_signal; re-run src.data.make_sellers")
    s = (orders.groupby("seller_id", as_index=False)["fraud_signal"].sum()
               .rename(columns={"fraud_signal":"fraud_orders"}))
    s["label"] = (s["fraud_orders"] >= min_fraud_orders).astype(int)
    return s[["seller_id","label"]]

def main():
    paths = load_paths()
    cfg = yaml.safe_load((root() / "configs/model.yaml").read_text())

    features = _read_any(paths["processed_dir"], "seller_features")
    orders   = _read_any(paths["raw_dir"], "orders")
    orders["timestamp"] = pd.to_datetime(orders["timestamp"])

    labels = build_labels(orders, cfg["label_def"]["min_fraud_orders"])
    first  = first_order_ts_per_seller(orders)

    X = (features.merge(labels, on="seller_id", how="left")
                 .merge(first, on="seller_id", how="left"))
    X["label"] = X["label"].fillna(0).astype(int)

    with open(root() / "artifacts" / "feature_list.json") as f:
        feat_list = [c for c in json.load(f) if c in X.columns]

    cutoff = pd.to_datetime(cfg["split"]["cutoff_date"])
    time_mask = X["first_order_ts"] > cutoff
    # Fallback: if time-based test is empty, sample a random test set
    if time_mask.sum() == 0:
        rng = np.random.default_rng(42)
        time_mask = rng.random(len(X)) < 0.2  

    model = joblib.load(root() / "artifacts" / "model.pkl")
    Xt = X.loc[time_mask, feat_list]
    if Xt.empty or Xt.shape[1] == 0:
        raise RuntimeError("No features to score; check feature_list.json vs processed columns.")

    if hasattr(model, "predict_proba"):
        p = model.predict_proba(Xt)[:,1]
    else:
        z = model.decision_function(Xt)
        p = 1/(1+np.exp(-z))

    tv = X.loc[time_mask, ["seller_id","label"]].copy()
    tv["prob"] = p
    tv = tv.dropna(subset=["seller_id"])
    tv["seller_id"] = tv["seller_id"].astype(str)
    tv["label"] = tv["label"].astype(int)

    top20 = tv.sort_values("prob", ascending=False).head(20).to_dict(orient="records")
    out = root() / "artifacts" / "top20_test.json"
    out.write_text(json.dumps(top20, indent=2))
    print(f"✅ Wrote {out} with {len(top20)} rows")

if __name__ == "__main__":
    main()
