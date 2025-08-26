from __future__ import annotations
from pathlib import Path
import json, math, datetime as dt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[2]

def load_paths():
    cfg = yaml.safe_load((ROOT / "configs/paths.yaml").read_text())
    return {k: (ROOT / v) for k, v in cfg.items()}

def read_any(base: Path, stem: str) -> pd.DataFrame:
    pq = base / f"{stem}.parquet"
    if pq.exists(): return pd.read_parquet(pq)
    csv = base / f"{stem}.csv"
    if csv.exists(): return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing {stem}.parquet/.csv in {base}")

def psi(a: np.ndarray, b: np.ndarray, bins: int = 10) -> float:
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2: return 0.0
    qs = np.quantile(a, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    pa, pb = [], []
    for i in range(bins):
        pa.append(((a > qs[i]) & (a <= qs[i+1])).mean() or 1e-12)
        pb.append(((b > qs[i]) & (b <= qs[i+1])).mean() or 1e-12)
    pa, pb = np.array(pa), np.array(pb)
    return float(np.sum((pa - pb) * np.log(pa / pb)))

def main():
    paths = load_paths()
    reports_dir = ROOT / "reports"
    artifacts_dir = ROOT / "artifacts"
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    feat = read_any(paths["processed_dir"], "seller_features")
    feat_list_path = artifacts_dir / "feature_list.json"
    if feat_list_path.exists():
        feat_cols = json.loads(feat_list_path.read_text())
    else:
        feat_cols = [c for c in feat.columns if c != "seller_id"]

    df = feat[["seller_id"] + feat_cols].copy()
    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    ref_path = artifacts_dir / "features_reference.parquet"
    now = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    if not ref_path.exists():
        df.to_parquet(ref_path, index=False)
        summary = {
            "created_reference": True,
            "ts": now,
            "n_features": len(feat_cols),
            "n_rows_reference": int(df.shape[0]),
            "message": "No reference found; saved current as reference. No drift computed."
        }
        (reports_dir / "drift.json").write_text(json.dumps(summary, indent=2))
        (reports_dir / "drift.md").write_text(
            f"# Drift Report\n\nCreated reference snapshot at {now}. Run again after new data to compute drift.\n"
        )
        print("📌 Created reference snapshot → artifacts/features_reference.parquet")
        return

    ref = pd.read_parquet(ref_path)
    # inner join on feature columns only (ignore ids)
    drift_rows = []
    n_flags_psi, n_flags_ks = 0, 0
    for c in feat_cols:
        a = pd.to_numeric(ref[c], errors="coerce").to_numpy()
        b = pd.to_numeric(df[c], errors="coerce").to_numpy()
        val_psi = psi(a, b, bins=10)
        ks = ks_2samp(a[~np.isnan(a)], b[~np.isnan(b)], alternative="two-sided", mode="asymp")
        ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)

        flag_psi = val_psi > 0.2         
        flag_ks  = ks_p < 0.01
        n_flags_psi += int(flag_psi)
        n_flags_ks  += int(flag_ks)

        drift_rows.append({
            "feature": c,
            "psi": round(val_psi, 6),
            "ks_stat": round(ks_stat, 6),
            "ks_pvalue": "{:.3e}".format(ks_p),
            "flag_psi": bool(flag_psi),
            "flag_ks": bool(flag_ks),
            "ref_mean": float(np.nanmean(a)),
            "cur_mean": float(np.nanmean(b)),
        })

    n = len(feat_cols)
    overall_flag = (n_flags_psi >= max(3, math.ceil(0.1 * n))) or (n_flags_ks >= max(3, math.ceil(0.1 * n)))
    summary = {
        "ts": now,
        "n_features": n,
        "flags_psi": n_flags_psi,
        "flags_ks": n_flags_ks,
        "overall_drift": bool(overall_flag),
    }

    out_json = {
        "summary": summary,
        "details": drift_rows,
    }
    (reports_dir / "drift.json").write_text(json.dumps(out_json, indent=2))

    import pandas as pd
    md = ["# Drift Report",
          f"- Timestamp: **{now}**",
          f"- Features checked: **{n}**",
          f"- PSI flags: **{n_flags_psi}** | KS flags: **{n_flags_ks}**",
          f"- Overall drift: **{overall_flag}**",
          "",
          "| feature | psi | ks_stat | ks_pvalue | ref_mean | cur_mean | flags |",
          "|---|---:|---:|---:|---:|---:|---|"]
    for r in drift_rows:
        flags = []
        if r["flag_psi"]: flags.append("PSI")
        if r["flag_ks"]: flags.append("KS")
        md.append(f"| {r['feature']} | {r['psi']:.3f} | {r['ks_stat']:.3f} | {r['ks_pvalue']} | {r['ref_mean']:.3f} | {r['cur_mean']:.3f} | {','.join(flags) or '-'} |")
    (reports_dir / "drift.md").write_text("\n".join(md))
    print(f"🧭 Drift report → reports/drift.json & reports/drift.md  (overall_drift={overall_flag})")

if __name__ == "__main__":
    main()
