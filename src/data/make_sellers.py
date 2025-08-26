from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from datetime import timedelta

#  Helpers 

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def load_paths():
    cfg = yaml.safe_load((project_root() / "configs/paths.yaml").read_text())
    return {k: (project_root() / v) for k, v in cfg.items()}

def load_sim():
    return yaml.safe_load((project_root() / "configs/sim.yaml").read_text())

def write_table(df: pd.DataFrame, out: Path, prefer_parquet: bool = True) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    if prefer_parquet:
        try:
            df.to_parquet(out.with_suffix(".parquet"), index=False)
            return out.with_suffix(".parquet")
        except Exception:
            pass
    df.to_csv(out.with_suffix(".csv"), index=False)
    return out.with_suffix(".csv")

# Long-tail weights for sellers (power law-ish)
def long_tail_probs(n: int, alpha: float = 1.2, offset: float = 10.0) -> np.ndarray:
    ranks = np.arange(n)
    w = 1.0 / np.power(ranks + offset, alpha)
    w /= w.sum()
    return w

#  Core build 

def main():
    paths = load_paths()
    sim = load_sim()
    rng = np.random.default_rng(sim["seed"])

    raw = pd.read_csv(paths["raw_dir"] / "creditcard.csv")
    raw = raw.rename(columns=str)  
    # time column is seconds since first txn; anchor to a calendar date
    anchor = pd.to_datetime(sim["anchor_date"])
    order_time = anchor + pd.to_timedelta(raw["Time"], unit="s")

    N = len(raw)
    n_sellers = int(sim["n_sellers"])
    n_buyers  = int(sim["n_buyers"])

    seller_ids = np.array([f"S{idx:05d}" for idx in range(n_sellers)])
    buyer_ids  = np.array([f"B{idx:06d}" for idx in range(n_buyers)])
    order_ids  = np.array([f"O{idx:06d}" for idx in range(N)])

    # Assign sellers with a head-heavy distribution (marketplace reality)
    probs = long_tail_probs(n_sellers, alpha=1.25, offset=8.0)
    sellers_for_orders = rng.choice(seller_ids, size=N, p=probs)
    buyers_for_orders  = rng.choice(buyer_ids,  size=N, replace=True)

    # Payment methods
    pm_names, pm_weights = zip(*sim["payment_methods"])
    pm_names = np.array(pm_names)
    pm_weights = np.array(pm_weights, dtype=float)
    pm_weights = pm_weights / pm_weights.sum()
    payment_method = rng.choice(pm_names, size=N, p=pm_weights)

    # Amounts come from dataset (positive)
    amount = raw["Amount"].clip(lower=0.0).to_numpy()

    # Ship/Delivery times
    handling_days = rng.gamma(shape=2.0, scale=sim["handling_time_mean_days"]/2.0, size=N)
    promised_days = rng.poisson(lam=sim["promised_transit_mean_days"], size=N).astype(float).clip(min=1.0)
    actual_transit = rng.gamma(shape=2.0, scale=sim["promised_transit_mean_days"]/2.0, size=N)
    ship_date = order_time + pd.to_timedelta(handling_days, unit="D")
    delivered_date = ship_date + pd.to_timedelta(actual_transit, unit="D")
    promised_date  = ship_date + pd.to_timedelta(promised_days, unit="D")

    # Late flag derived from promised vs actual
    late = (delivered_date - promised_date).dt.total_seconds() > sim["late_threshold_days"] * 86400

    # Refund flag (boosted if original txn marked as fraud)
    fraud_txn = (raw["Class"] == 1).to_numpy()
    base_refund = sim["base_refund_rate"]
    refund_prob = np.full(N, base_refund, dtype=float)
    # a little more likely to refund when late
    refund_prob = refund_prob + late.astype(float) * 0.05
    # heavy boost if original dataset labeled it as fraud
    refund_prob = np.where(fraud_txn, np.clip(sim["fraud_refund_boost"], 0, 1), refund_prob)
    refund_flag = rng.random(N) < refund_prob

    # Carriers
    carriers = np.array(sim["carriers"])
    carrier_for_orders = rng.choice(carriers, size=N)

    # Reviews (stars & tiny text stub)
    noise = rng.normal(0.0, 0.3, size=N)
    stars = 4.4 + noise
    stars = stars - late.astype(float) * rng.uniform(0.5, 1.5, size=N)
    stars = stars - refund_flag.astype(float) * rng.uniform(1.0, 2.0, size=N)
    stars = np.clip(np.round(stars, 1), 1.0, 5.0)

    positive_text = ["Great product!", "As described.", "Fast delivery.", "Would buy again."]
    neutral_text  = ["Okay item.", "Average experience."]
    negative_text = ["Late delivery.", "Item not as described.", "Requested refund.", "Poor quality."]

    text = np.where(stars >= 4.0,
                    rng.choice(positive_text, size=N),
             np.where(stars >= 3.0,
                    rng.choice(neutral_text, size=N),
                    rng.choice(negative_text, size=N)))
    review_time = delivered_date + pd.to_timedelta(rng.integers(0, 5, size=N), unit="D")

    #  Build tables 
    orders = pd.DataFrame({
        "order_id": order_ids,
        "buyer_id": buyers_for_orders,
        "seller_id": sellers_for_orders,
        "timestamp": order_time,
        "amount": amount,
        "payment_method": payment_method,
        "refund_flag": refund_flag.astype(int),
        "fraud_signal": fraud_txn.astype(int)

    })

    shipments = pd.DataFrame({
        "order_id": order_ids,
        "ship_date": ship_date,
        "promised_date": promised_date,
        "delivered_date": delivered_date,
        "carrier": carrier_for_orders
    })

    reviews = pd.DataFrame({
        "order_id": order_ids,
        "stars": stars,
        "text": text,
        "created_at": review_time
    })

    # Seller master
    regions_cfg = sim["regions"]
    region_keys = list(regions_cfg.keys())
    seller_region = rng.choice(region_keys, size=n_sellers)
    seller_city = [rng.choice(regions_cfg[r]) for r in seller_region]
    categories = np.array(sim["categories"])
    seller_category = rng.choice(categories, size=n_sellers)

    # signup_date: sometime before their first observed order (0–180 days)
    seller_df = pd.DataFrame({
        "seller_id": seller_ids,
        "category": seller_category,
        "city": seller_city,
        "region": seller_region
    })
    # derive signup dates
    first_order_time = (
        orders[["seller_id", "timestamp"]]
        .groupby("seller_id", as_index=False)["timestamp"]
        .min()
        .rename(columns={"timestamp": "first_order_ts"})
    )
    seller_df = seller_df.merge(first_order_time, on="seller_id", how="left")
    offset_days = rng.integers(0, 180, size=len(seller_df))
    seller_df["signup_date"] = seller_df["first_order_ts"] - pd.to_timedelta(offset_days, unit="D")
    seller_df = seller_df.drop(columns=["first_order_ts"])

    #  Write out 
    out_dir = project_root() / "data" / "raw"
    orders_path    = write_table(orders,   out_dir / "orders")
    shipments_path = write_table(shipments,out_dir / "shipments")
    reviews_path   = write_table(reviews,  out_dir / "reviews")
    sellers_path   = write_table(seller_df,out_dir / "sellers")

    print("✅ Wrote:")
    for p in [orders_path, shipments_path, reviews_path, sellers_path]:
        print("   ", p)

if __name__ == "__main__":
    main()
