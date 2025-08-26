from pathlib import Path
import hashlib
import requests
import time

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def md5sum(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def download_file(url: str, dest: Path, expected_md5: str | None = None, retries: int = 3) -> Path:
    ensure_dir(dest.parent)
    for attempt in range(1, retries + 1):
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            with tmp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
            tmp.replace(dest)
        if expected_md5:
            got = md5sum(dest)
            if got.lower() != expected_md5.lower():
                if attempt == retries:
                    raise ValueError(f"MD5 mismatch for {dest.name}: got {got}, expected {expected_md5}")
                time.sleep(2)
                continue
        break
    return dest
