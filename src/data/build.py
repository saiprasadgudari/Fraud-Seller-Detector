from pathlib import Path
from src.data.fetch_raw import fetch_creditcard_csv
from src.data.validate import validate_creditcard_csv

def main():
    csv_path: Path = fetch_creditcard_csv()
    validate_creditcard_csv(csv_path)
    print("✅ Raw dataset is ready:", csv_path)

if __name__ == "__main__":
    main()
