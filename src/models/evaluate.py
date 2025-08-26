from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
import yaml
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, confusion_matrix

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def load_paths():
    cfg = yaml.safe_load((project_root() / "configs/paths.yaml").read_text())
    return {k: (project_root() / v) for k, v in cfg.items()}

def _read_any(base: Path, stem: str) -> pd.DataFrame:
    pq = base / f"{stem}.parquet"
    if pq.exists(): return pd.read_parquet(pq)
    csv = base / f"{stem}.csv"
    if csv.exists(): return pd.read_csv(csv)
    raise FileNotFoundError

def main():
    paths = load_paths()
    X = _read_any(paths["processed_dir"], "seller_features")
    with open(project_root() / "artifacts" / "feature_list.json") as f:
        feat_list = json.load(f)
    model = joblib.load(project_root() / "artifacts" / "model.pkl")
    probs = (model.predict_proba(X[feat_list])[:,1] 
             if hasattr(model,"predict_proba") 
             else 1/(1+np.exp(-model.decision_function(X[feat_list]))))
    preview = pd.DataFrame({"seller_id": X["seller_id"], "prob": probs}).sort_values("prob", ascending=False).head(20)
    print(preview.to_string(index=False))

if __name__ == "__main__":
    main()
