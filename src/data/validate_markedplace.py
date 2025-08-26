from pathlib import Path
import pandas as pd

def main():
    base = Path("data/raw")
    orders = pd.read_parquet(base/"orders.parquet") if (base/"orders.parquet").exists() else pd.read_csv(base/"orders.csv")
    shipments = pd.read_parquet(base/"shipments.parquet") if (base/"shipments.parquet").exists() else pd.read_csv(base/"shipments.csv")
    reviews = pd.read_parquet(base/"reviews.parquet") if (base/"reviews.parquet").exists() else pd.read_csv(base/"reviews.csv")
    sellers = pd.read_parquet(base/"sellers.parquet") if (base/"sellers.parquet").exists() else pd.read_csv(base/"sellers.csv")

    assert orders["order_id"].is_unique, "order_id must be unique in orders"
    assert set(orders["order_id"]) == set(shipments["order_id"]) == set(reviews["order_id"]), "order_id mismatch"
    assert orders["timestamp"].min() >= pd.Timestamp("2024-01-01"), "unexpected timestamp floor"
    assert set(orders["seller_id"]).issubset(set(sellers["seller_id"])), "unknown sellers in orders"
    assert (shipments["delivered_date"] >= shipments["ship_date"]).all(), "delivery before ship_date?"
    print("✅ Marketplace tables look sane.")

if __name__ == "__main__":
    main()
