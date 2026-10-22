"""Load and apply the trained laptop model."""
from pathlib import Path
import json, joblib, pandas as pd

def model_status(root="."):
    root=Path(root)
    for meta in (root/"results/development_model_metadata.json", root/"models/model_metadata.json"):
        if meta.exists(): return json.loads(meta.read_text(encoding="utf-8"))
    return None

def predict_trial(row, root="."):
    root=Path(root); dev_model=root/"models/decision_tree_dev60.joblib"; dev_names=root/"models/feature_names_dev60.json"
    model_path=dev_model if dev_model.exists() else root/"models/decision_tree.joblib"
    names_path=dev_names if dev_names.exists() else root/"models/feature_names.json"
    if not model_path.exists() or not names_path.exists(): raise FileNotFoundError("Train a model first")
    features=json.loads(names_path.read_text(encoding="utf-8")); model=joblib.load(model_path)
    x=pd.DataFrame([[float(row[f]) for f in features]],columns=features); pred=int(model.predict(x)[0]); probability=float(model.predict_proba(x)[0][list(model.classes_).index(1)]) if hasattr(model,"predict_proba") else None
    return pred,probability
