from __future__ import annotations
from pathlib import Path
import json
import yaml
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, confusion_matrix
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#  utils 

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def load_paths():
    cfg = yaml.safe_load((project_root() / "configs/paths.yaml").read_text())
    return {k: (project_root() / v) for k, v in cfg.items()}

def _read_any(base: Path, stem: str) -> pd.DataFrame:
    pq = base / f"{stem}.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    csv = base / f"{stem}.csv"
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing {stem}.parquet/.csv in {base}")

def load_cfg() -> dict:
    return yaml.safe_load((project_root() / "configs/model.yaml").read_text())

def pick_supervised_model(model_type: str, params: dict):
    if model_type == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(**params)
        except Exception:
            print("⚠️ LightGBM not available, falling back to RandomForest. Install with: pip install lightgbm")
            return RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1, class_weight="balanced_subsample")
    elif model_type == "rf":
        return RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1, class_weight="balanced_subsample")
    else:
        return LogisticRegression(max_iter=500, class_weight="balanced", n_jobs=None)

#  label building 

def build_labels(orders: pd.DataFrame, min_fraud_orders: int) -> pd.DataFrame:
    if "fraud_signal" not in orders.columns:
        raise RuntimeError("orders table lacks 'fraud_signal'. Edit src/data/make_sellers.py and re-run it.")
    seller_fraud_counts = (orders
                           .groupby("seller_id", as_index=False)["fraud_signal"]
                           .sum()
                           .rename(columns={"fraud_signal": "fraud_orders"}))
    seller_labels = seller_fraud_counts.assign(label=(seller_fraud_counts["fraud_orders"] >= min_fraud_orders).astype(int))
    return seller_labels[["seller_id", "label"]]

def first_order_ts_per_seller(orders: pd.DataFrame) -> pd.DataFrame:
    tmp = orders[["seller_id", "timestamp"]].copy()
    tmp["timestamp"] = pd.to_datetime(tmp["timestamp"])
    return tmp.groupby("seller_id", as_index=False)["timestamp"].min().rename(columns={"timestamp": "first_order_ts"})

#  feedback integration 

def load_feedback_labels() -> pd.DataFrame:
    """Read latest labels from data/labels/feedback.jsonl (if present).
       Keep the most recent label per seller_id."""
    fb_path = project_root() / "data" / "labels" / "feedback.jsonl"
    if not fb_path.exists():
        return pd.DataFrame(columns=["seller_id", "label", "ts"])
    rows = []
    with fb_path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                rows.append(rec)
            except Exception:
                continue
    if not rows:
        return pd.DataFrame(columns=["seller_id", "label", "ts"])
    df = pd.DataFrame(rows)
    if "seller_id" not in df.columns or "label" not in df.columns:
        return pd.DataFrame(columns=["seller_id", "label", "ts"])
    # coerce dtypes
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        df = df.sort_values("ts")
        df = df.dropna(subset=["seller_id"]).groupby("seller_id", as_index=False).last()
    else:
        df = df.dropna(subset=["seller_id"]).groupby("seller_id", as_index=False).last()
    return df[["seller_id", "label"]]

def apply_feedback_overrides(labels: pd.DataFrame) -> pd.DataFrame:
    """Override rule-based labels with human feedback when available."""
    fb = load_feedback_labels()
    if fb.empty:
        print("ℹ️ No feedback labels found.")
        return labels
    merged = labels.merge(fb, on="seller_id", how="outer", suffixes=("", "_fb"))
    # if feedback provided, use it; otherwise keep original (fillna to 0)
    before = labels.set_index("seller_id")["label"]
    merged["label_final"] = merged["label_fb"].where(~merged["label_fb"].isna(), merged["label"]).fillna(0).astype(int)
    after = merged[["seller_id", "label_final"]].rename(columns={"label_final": "label"})
    overrides = (before.reindex(after["seller_id"])
                      .ne(after.set_index("seller_id")["label"])
                      .sum())
    print(f"🔁 Feedback overrides applied to {int(overrides)} sellers")
    return after

#  metrics 

def compute_metrics(y_true, y_prob, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    roc = roc_auc_score(y_true, y_prob)
    prauc = average_precision_score(y_true, y_prob)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "roc_auc": float(roc),
        "pr_auc": float(prauc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "positives": int(np.array(y_true).sum()),
        "n": int(len(y_true)),
        "threshold": threshold,
    }

#  main train 

def main():
    paths = load_paths()
    cfg = load_cfg()

    raw_dir = paths["raw_dir"]
    processed_dir = paths["processed_dir"]
    artifacts_dir = paths["artifacts_dir"]
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    features = _read_any(processed_dir, "seller_features")
    orders   = _read_any(raw_dir, "orders")

    # Build labels (rule-based) then apply human feedback overrides
    orders["timestamp"] = pd.to_datetime(orders["timestamp"])
    labels = build_labels(orders, cfg["label_def"]["min_fraud_orders"])
    labels = apply_feedback_overrides(labels)  

    # Merge X + y
    X = features.merge(labels, on="seller_id", how="left")
    X["label"] = X["label"].fillna(0).astype(int)

    # Split sellers (time-based by FIRST observed order)
    first_ts = first_order_ts_per_seller(orders)
    X = X.merge(first_ts, on="seller_id", how="left")

    if cfg["split"]["strategy"] == "time":
        cutoff = pd.to_datetime(cfg["split"]["cutoff_date"])
        train_mask = X["first_order_ts"] <= cutoff
        test_mask  = X["first_order_ts"] > cutoff
    else:
        rng = np.random.default_rng(42)
        mask = rng.random(len(X)) >= cfg["split"].get("random_test_size", 0.2)
        train_mask, test_mask = mask, ~mask

    id_col = "seller_id"
    target_col = "label"

    with open(project_root() / "artifacts" / "feature_list.json", "r") as f:
        feat_list = json.load(f)
    feat_list = [c for c in feat_list if c in X.columns]

    X_train = X.loc[train_mask, feat_list].copy()
    y_train = X.loc[train_mask, target_col].astype(int).to_numpy()
    X_test  = X.loc[test_mask,  feat_list].copy()
    y_test  = X.loc[test_mask,  target_col].astype(int).to_numpy()

    if X_test.empty:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X.loc[:, feat_list],
            X[target_col].astype(int),
            test_size=0.2,
            random_state=42,
            stratify=X[target_col].astype(int)
        )

    # Train supervised model
    model_type = cfg["model"]["type"]
    params = cfg["model"]["params"] or {}
    clf = pick_supervised_model(model_type, params)
    clf.fit(X_train, y_train)

    # Supervised probabilities
    if hasattr(clf, "predict_proba"):
        p_sup_train = clf.predict_proba(X_train)[:, 1]
        p_sup_test  = clf.predict_proba(X_test)[:, 1]
    else:
        d_train = clf.decision_function(X_train)
        d_test  = clf.decision_function(X_test)
        p_sup_train = 1 / (1 + np.exp(-d_train))
        p_sup_test  = 1 / (1 + np.exp(-d_test))

    # Optional anomaly score via IsolationForest
    cfg_combo = cfg.get("combine_anomaly", {"enabled": False})
    if cfg_combo.get("enabled", False):
        iso = IsolationForest(random_state=42, contamination="auto")
        iso.fit(X_train)
        an_train = -iso.score_samples(X_train)
        an_test  = -iso.score_samples(X_test)
        def minmax(a):
            lo, hi = float(np.min(a)), float(np.max(a))
            return (a - lo) / (hi - lo + 1e-9)
        an_train = minmax(an_train)
        an_test  = minmax(an_test)
        w_sup = float(cfg_combo.get("weight_supervised", 1.0))
        w_iso = float(cfg_combo.get("weight_isoforest", 0.0))
        p_train = w_sup * p_sup_train + w_iso * an_train
        p_test  = w_sup * p_sup_test  + w_iso * an_test
    else:
        p_train, p_test = p_sup_train, p_sup_test

    # Metrics
    thr = float(cfg["threshold"])
    train_metrics = compute_metrics(y_train, p_train, thr)
    test_metrics  = compute_metrics(y_test,  p_test,  thr)

    # Persist artifacts
    joblib.dump(clf, artifacts_dir / "model.pkl")
    (artifacts_dir / "metrics.json").write_text(json.dumps({
        "train": train_metrics,
        "test": test_metrics,
        "n_features": len(feat_list),
        "features": feat_list,
        "split": cfg["split"],
        "label_def": cfg["label_def"],
        "model_type": model_type
    }, indent=2))

    # Top-20 preview from test set
    test_view = X.loc[test_mask, [id_col, target_col]].copy()
    test_view["prob"] = p_test
    top20 = (test_view.sort_values("prob", ascending=False)
                      .head(20)
                      .to_dict(orient="records"))
    (artifacts_dir / "top20_test.json").write_text(json.dumps(top20, indent=2))

    print("✅ Training done.")
    print(f"Train ROC-AUC: {train_metrics['roc_auc']:.3f} | PR-AUC: {train_metrics['pr_auc']:.3f}")
    print(f" Test ROC-AUC: {test_metrics['roc_auc']:.3f} | PR-AUC: {test_metrics['pr_auc']:.3f}")
    print(f"Artifacts saved in {artifacts_dir}/ (model.pkl, metrics.json, top20_test.json)")

if __name__ == "__main__":
    main()
