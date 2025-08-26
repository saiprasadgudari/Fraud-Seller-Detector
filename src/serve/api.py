from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import os
import json
import time
import datetime as dt

import joblib
import pandas as pd
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST


ROOT = Path(__file__).resolve().parents[2]

def load_paths() -> Dict[str, Path]:
    cfg = yaml.safe_load((ROOT / "configs/paths.yaml").read_text())
    # expected keys in paths.yaml: raw_dir, processed_dir, artifacts_dir
    out = {k: (ROOT / v) for k, v in cfg.items()}
    # Add a default labels dir if not present
    out.setdefault("labels_dir", ROOT / "data" / "labels")
    return out

def read_any(base: Path, stem: str) -> pd.DataFrame:
    pq = base / f"{stem}.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    csv = base / f"{stem}.csv"
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing {stem}.parquet/.csv in {base}")

paths = load_paths()
paths["labels_dir"].mkdir(parents=True, exist_ok=True)
paths["artifacts_dir"].mkdir(parents=True, exist_ok=True)

# Load features
features_df = read_any(paths["processed_dir"], "seller_features")

feat_list_path = paths["artifacts_dir"] / "feature_list.json"
feat_list = json.loads(feat_list_path.read_text()) if feat_list_path.exists() else [
    c for c in features_df.columns if c != "seller_id"
]
feat_list = [c for c in feat_list if c in features_df.columns]

# In-memory feature matrix keyed by seller_id for quick lookups
feature_matrix: pd.DataFrame = features_df.set_index("seller_id")[feat_list].astype(float)
feature_means: np.ndarray = feature_matrix.mean(axis=0).to_numpy(dtype=float)

# Load model
model = joblib.load(paths["artifacts_dir"] / "model.pkl")

THRESH_PATH = paths["artifacts_dir"] / "threshold.json"
if THRESH_PATH.exists():
    try:
        THRESHOLD = float(json.loads(THRESH_PATH.read_text()).get("threshold", 0.5))
    except Exception:
        THRESHOLD = 0.5
else:
    THRESHOLD = 0.5

app = FastAPI(title="Fraud Seller Scoring API", version="0.3.0")

REQS = Counter("fraud_api_requests_total", "Requests", ["route"])
LAT = Histogram("fraud_api_latency_seconds", "Latency", ["route"])


class SellerReq(BaseModel):
    seller_id: str

class BatchReq(BaseModel):
    seller_ids: List[str]

class ExplainReq(BaseModel):
    seller_id: str
    top_k: int = 10

class FeedbackReq(BaseModel):
    seller_id: str
    label: int
    source: str = "ui"
    note: str | None = None

def _score_numpy(X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1].astype(float)
    # fallback: decision_function → sigmoid
    z = model.decision_function(X).astype(float)
    return 1.0 / (1.0 + np.exp(-z))

def _safe_explain(x_row: pd.Series, top_k: int) -> Dict:
    """
    Try to compute SHAP values; if SHAP or model-specific explainers are not available,
    fall back to a simple importance-weighted deviation heuristic.
    Returns: {"baseline": float, "items": [{"feature","value","contribution"}...]}
    """
    x = x_row.to_numpy(dtype=float).reshape(1, -1)
    feature_names = list(x_row.index)

    # Try SHAP (TreeExplainer for LightGBM / RF / XGB, Linear for LogisticRegression)
    shap_values = None
    baseline = 0.0
    try:
        import shap  # type: ignore
        name = type(model).__name__.lower()
        if any(t in name for t in ["lgbm", "randomforest", "xgb", "decisiontree", "extratrees"]):
            explainer = shap.TreeExplainer(model)
            res = explainer(x)
            # shap >=0.46 returns Explanation
            if hasattr(res, "values"):
                vals = np.array(res.values)
                if vals.ndim == 3:  # (1, n_features, n_outputs)
                    vals = vals[:, :, -1]
                shap_values = vals.reshape(1, -1)[0]
                base = getattr(res, "base_values", 0.0)
                baseline = float(np.array(base).reshape(-1)[-1]) if np.size(base) else 0.0
            else:
                vals = explainer.shap_values(x)
                if isinstance(vals, list):
                    shap_values = np.asarray(vals[-1]).reshape(1, -1)[0]
                else:
                    shap_values = np.asarray(vals).reshape(1, -1)[0]
                baseline = 0.0
        elif "logistic" in name:
            explainer = shap.LinearExplainer(model, feature_matrix.to_numpy(dtype=float))
            vals = explainer.shap_values(x)
            shap_values = np.asarray(vals).reshape(1, -1)[0]
            base = getattr(explainer, "expected_value", 0.0)
            baseline = float(np.array(base).reshape(-1)[-1]) if np.size(base) else 0.0
    except Exception:
        shap_values = None

    # Fallback: importance-weighted deviation from mean
    if shap_values is None:
        if hasattr(model, "feature_importances_"):
            w = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            w = np.abs(np.asarray(model.coef_).reshape(-1))
        else:
            w = np.ones(len(feature_names), dtype=float)
        w = w / (w.sum() + 1e-9)
        contrib = w * (x.reshape(-1) - feature_means)
        shap_values = contrib
        baseline = float(_score_numpy(feature_means.reshape(1, -1))[0])

    # Build top-k by absolute contribution
    order = np.argsort(np.abs(shap_values))[::-1][:int(top_k)]
    items = []
    xv = x.reshape(-1)
    for i in order:
        items.append({
            "feature": feature_names[i],
            "value": float(xv[i]),
            "contribution": float(shap_values[i]),
        })
    return {"baseline": baseline, "items": items}

def _feedback_files():
    jl = paths["labels_dir"] / "feedback.jsonl"
    overrides = paths["artifacts_dir"] / "feedback_overrides.json"
    return jl, overrides

#END Points
@app.get("/health")
def health():
    t0 = time.perf_counter(); REQS.labels("/health").inc()
    try:
        return {
            "ok": True,
            "n_sellers": int(feature_matrix.shape[0]),
            "n_features": int(feature_matrix.shape[1]),
            "model": type(model).__name__,
            "threshold": THRESHOLD,
        }
    finally:
        LAT.labels("/health").observe(time.perf_counter() - t0)

@app.post("/score")
def score(req: SellerReq):
    t0 = time.perf_counter(); REQS.labels("/score").inc()
    try:
        sid = req.seller_id
        if sid not in feature_matrix.index:
            raise HTTPException(status_code=404, detail=f"seller_id {sid} not found")
        x = feature_matrix.loc[sid].to_numpy(dtype=float).reshape(1, -1)
        prob = float(_score_numpy(x)[0])
        label = int(prob >= THRESHOLD)
        return {"seller_id": sid, "fraud_prob": prob, "threshold": THRESHOLD, "label": label}
    finally:
        LAT.labels("/score").observe(time.perf_counter() - t0)

@app.post("/batch_score")
def batch_score(req: BatchReq):
    t0 = time.perf_counter(); REQS.labels("/batch_score").inc()
    try:
        found = [sid for sid in req.seller_ids if sid in feature_matrix.index]
        if not found:
            raise HTTPException(status_code=404, detail="None of the seller_ids were found")
        X = feature_matrix.loc[found].to_numpy(dtype=float)
        probs = _score_numpy(X)
        return [
            {"seller_id": sid, "fraud_prob": float(p), "threshold": THRESHOLD, "label": int(p >= THRESHOLD)}
            for sid, p in zip(found, probs)
        ]
    finally:
        LAT.labels("/batch_score").observe(time.perf_counter() - t0)

@app.post("/explain")
def explain(req: ExplainReq):
    t0 = time.perf_counter(); REQS.labels("/explain").inc()
    try:
        sid = req.seller_id
        if sid not in feature_matrix.index:
            raise HTTPException(status_code=404, detail=f"seller_id {sid} not found")
        x_row = feature_matrix.loc[sid]
        out = _safe_explain(x_row, req.top_k)
        prob = float(_score_numpy(x_row.to_numpy(dtype=float).reshape(1, -1))[0])
        return {
            "seller_id": sid,
            "fraud_prob": prob,
            "threshold": THRESHOLD,
            "label": int(prob >= THRESHOLD),
            "baseline": out["baseline"],
            "features": out["items"],
        }
    finally:
        LAT.labels("/explain").observe(time.perf_counter() - t0)

@app.post("/feedback")
def feedback(req: FeedbackReq):
    t0 = time.perf_counter(); REQS.labels("/feedback").inc()
    try:
        sid = req.seller_id
        known = bool(sid in feature_matrix.index)

        record = {
            "ts": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "seller_id": sid,
            "label": int(req.label),
            "source": req.source,
            "known_seller": known,
            **({"note": req.note} if req.note else {}),
        }
        jl, overrides = _feedback_files()

        # append JSONL
        with open(jl, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # update overrides map {seller_id: label}
        try:
            if overrides.exists():
                ov = json.loads(overrides.read_text())
            else:
                ov = {}
            ov[sid] = int(req.label)
            overrides.write_text(json.dumps(ov, indent=2))
        except Exception:
            pass  
        return {"ok": True, "written": str(jl), "overrides": str(overrides)}
    finally:
        LAT.labels("/feedback").observe(time.perf_counter() - t0)

@app.get("/metrics")
def metrics():
    # Prometheus exposition format
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
