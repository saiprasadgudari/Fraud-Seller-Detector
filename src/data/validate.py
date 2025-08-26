from pathlib import Path
import pandas as pd

EXPECTED = {"Time", "Amount", "Class"} | {f"V{i}" for i in range(1, 29)}

def validate_creditcard_csv(path: Path) -> None:
    df = pd.read_csv(path)
    missing = EXPECTED.difference(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")
    # tiny spot-checks
    assert df["Class"].isin([0,1]).all(), "Class must be 0/1."
    assert df.shape[0] > 200_000, "Row count unexpectedly small."
