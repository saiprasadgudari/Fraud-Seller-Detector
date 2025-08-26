from pathlib import Path
import yaml
from src.utils.io import ensure_dir, download_file

ZENODO_URL = "https://zenodo.org/records/7395559/files/creditcard.csv?download=1"
ZENODO_MD5 = "e90efcb83d69faf99fcab8b0255024de"  # from Zenodo page

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def load_paths():
    cfg = yaml.safe_load((project_root() / "configs/paths.yaml").read_text())
    return {k: (project_root() / v) for k, v in cfg.items()}

def fetch_creditcard_csv() -> Path:
    paths = load_paths()
    raw_dir = paths["raw_dir"]
    ensure_dir(raw_dir)
    dest = raw_dir / "creditcard.csv"
    if dest.exists():
        # Re-verify checksum (cheap + reliable)
        from src.utils.io import md5sum
        if md5sum(dest).lower() == ZENODO_MD5:
            return dest
        else:
            dest.unlink()  # remove bad copy and re-download
    return download_file(ZENODO_URL, dest, expected_md5=ZENODO_MD5)

if __name__ == "__main__":
    p = fetch_creditcard_csv()
    print(f"Downloaded: {p} ({p.stat().st_size/1_048_576:.1f} MB)")
