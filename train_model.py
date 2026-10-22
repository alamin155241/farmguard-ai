"""Leakage-safe, reproducible FarmGuard Decision Tree training pipeline."""
from __future__ import annotations
import argparse, json
from datetime import datetime
from pathlib import Path
import joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from dataset_service import MODEL_A_FEATURES
from protocol_60_20 import load_development_protocol, sha256_file

SEED=42

def metric_dict(y, pred):
    p,r,f,_=precision_recall_fscore_support(y,pred,average="binary",zero_division=0); cm=confusion_matrix(y,pred,labels=[0,1])
    return {"accuracy":float(accuracy_score(y,pred)),"precision":float(p),"recall":float(r),"f1":float(f),"tn":int(cm[0,0]),"fp":int(cm[0,1]),"fn":int(cm[1,0]),"tp":int(cm[1,1]),"confusion_matrix":cm.tolist()}

def _fit_best(x_train,y_train):
    best=None
    for depth in (2,3,4):
        for leaf in (1,2,3):
            m=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=leaf,class_weight="balanced",random_state=SEED).fit(x_train,y_train)
            score=metric_dict(y_train,m.predict(x_train))["f1"]
            candidate=(score,-depth,-leaf,m)
            if best is None or candidate[:3]>best[:3]: best=candidate
    return best[3]


def _development_model_selection(x, y):
    """Select tree parameters using stratified CV on development rows only."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    candidates = []
    for depth in (2, 3, 4):
        for leaf in (1, 2, 3):
            model = DecisionTreeClassifier(
                max_depth=depth,
                min_samples_leaf=leaf,
                class_weight="balanced",
                random_state=SEED,
            )
            scores = cross_val_score(model, x, y, cv=cv, scoring="f1")
            candidates.append({
                "max_depth": depth,
                "min_samples_leaf": leaf,
                "fold_f1": [float(value) for value in scores],
                "cv_f1_mean": float(scores.mean()),
                "cv_f1_std": float(scores.std()),
            })
    selected = sorted(
        candidates,
        key=lambda item: (
            -item["cv_f1_mean"], item["cv_f1_std"],
            item["max_depth"], -item["min_samples_leaf"],
        ),
    )[0]
    return selected, candidates, cv


def train_development(output_root="."):
    """Train the post-exploratory model using only the frozen 60-row development set."""
    root = Path(output_root)
    df, protocol = load_development_protocol(root)
    required = set(MODEL_A_FEATURES + ["label", "trial_id"])
    if not required.issubset(df):
        raise ValueError(f"Missing development columns: {sorted(required - set(df.columns))}")
    if df[MODEL_A_FEATURES + ["label"]].isna().any().any():
        raise ValueError("Development data contains missing model values")
    features = MODEL_A_FEATURES
    x = df[features].astype(float)
    y = pd.to_numeric(df["label"], errors="raise").astype(int)
    selected, candidates, cv = _development_model_selection(x, y)
    model = DecisionTreeClassifier(
        max_depth=selected["max_depth"],
        min_samples_leaf=selected["min_samples_leaf"],
        class_weight="balanced",
        random_state=SEED,
    )
    oof = cross_val_predict(model, x, y, cv=cv, method="predict")
    oof_metrics = metric_dict(y, oof)
    model.fit(x, y)

    results = root / "results"
    models = root / "models"
    results.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    model_path = models / "decision_tree_dev60.joblib"
    features_path = models / "feature_names_dev60.json"
    metrics_path = results / "development_cv_metrics.json"
    metadata_path = results / "development_model_metadata.json"
    predictions_path = results / "development_predictions.csv"
    joblib.dump(model, model_path)
    features_path.write_text(json.dumps(features, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame({
        "trial_id": df["trial_id"],
        "actual": y,
        "out_of_fold_prediction": oof,
        "evaluation": "development_5_fold_oof",
    }).to_csv(predictions_path, index=False)
    metrics = {
        "protocol": protocol["protocol"],
        "scientific_disclosure": protocol["scientific_disclosure"],
        "development_samples": int(len(df)),
        "class_counts": {"0": int((y == 0).sum()), "1": int((y == 1).sum())},
        "features": features,
        "selection_metric": "mean stratified 5-fold F1",
        "selected_parameters": {
            "max_depth": selected["max_depth"],
            "min_samples_leaf": selected["min_samples_leaf"],
            "class_weight": "balanced",
            "random_state": SEED,
        },
        "selected_fold_f1": selected["fold_f1"],
        "cv_f1_mean": selected["cv_f1_mean"],
        "cv_f1_std": selected["cv_f1_std"],
        "development_oof_metrics": oof_metrics,
        "all_candidates": candidates,
        "final_test_evaluated": False,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "training_timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_scope": "60 development trials only",
        "development_trial_ids": df["trial_id"].tolist(),
        "selected_parameters": metrics["selected_parameters"],
        "features": features,
        "protocol_metadata_sha256": sha256_file(root / "data/protocol_60_20/split_metadata.json"),
        "split_manifest_sha256": protocol["sha256"]["manifest"],
        "development_dataset_sha256": protocol["sha256"]["development"],
        "frozen_dataset_sha256": protocol["sha256"]["frozen_dataset"],
        "final_test_dataset_sha256": protocol["sha256"]["final_test"],
        "final_test_rows_loaded_during_training": False,
        "model_sha256": sha256_file(model_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metrics

def _cpp_tree(tree, features, node=0, indent="  "):
    left,right=tree.children_left[node],tree.children_right[node]
    if left==right:
        return f"{indent}return {int(np.argmax(tree.value[node][0]))};\n"
    name=features[tree.feature[node]]; threshold=tree.threshold[node]
    return f"{indent}if ({name} <= {threshold:.6f}f) {{\n"+_cpp_tree(tree,features,left,indent+"  ")+f"{indent}}} else {{\n"+_cpp_tree(tree,features,right,indent+"  ")+f"{indent}}}\n"

def train(dataset="data/ml_dataset.csv", output_root="."):
    df=pd.read_csv(dataset); required=set(MODEL_A_FEATURES+["label"])
    if not required.issubset(df): raise ValueError(f"Missing columns: {sorted(required-set(df.columns))}")
    df=df.dropna(subset=MODEL_A_FEATURES+["label"]); df["label"]=pd.to_numeric(df["label"],errors="coerce")
    df=df[df.label.isin([0,1])]
    counts=df.label.value_counts()
    if len(counts)<2 or counts.get(0,0)<10 or counts.get(1,0)<10:
        raise ValueError("Insufficient labeled data: need at least 10 NORMAL and 10 INTRUSION trials")
    root=Path(output_root); results=root/"results"; models=root/"models"; results.mkdir(parents=True,exist_ok=True); models.mkdir(parents=True,exist_ok=True)
    train_idx,test_idx=train_test_split(np.arange(len(df)),test_size=.25,stratify=df.label,random_state=SEED)
    y_train=df.label.iloc[train_idx].astype(int); y_test=df.label.iloc[test_idx].astype(int)
    features=MODEL_A_FEATURES
    model=_fit_best(df[features].iloc[train_idx],y_train); pred=model.predict(df[features].iloc[test_idx]); selected_metrics=metric_dict(y_test,pred)
    folds=min(5,int(counts.min())); selected_metrics["cv_f1_mean"]=float(cross_val_score(model,df[features],df.label.astype(int),cv=folds,scoring="f1").mean())
    evaluations={"model_a":selected_metrics}; chosen="model_a"
    # Refit deployment model on all data only after untouched-test metrics are saved.
    deployment=_fit_best(df[features],df.label.astype(int)); joblib.dump(deployment,models/"decision_tree.joblib")
    (models/"feature_names.json").write_text(json.dumps(features,indent=2),encoding="utf-8")
    baselines=[]
    pir=(df.iloc[test_idx].pir_detected.astype(int)).to_numpy(); rule=((df.iloc[test_idx].pir_detected.astype(int)==1)&(df.iloc[test_idx].beam_break_detected.astype(int)==1)).astype(int).to_numpy()
    for name,preds in (("A0_PIR_only",pir),("A1_PIR_and_beam_rule",rule),("A2_Decision_Tree_ML",pred)):
        baselines.append({"approach":name,**{k:v for k,v in metric_dict(y_test,preds).items() if k!="confusion_matrix"}})
    pd.DataFrame(baselines).to_csv(results/"baseline_comparison.csv",index=False)
    pd.DataFrame({"trial_id":df.iloc[test_idx].trial_id.to_numpy(),"actual":y_test.to_numpy(),"predicted":pred,"split":"held_out_test"}).to_csv(results/"predictions.csv",index=False)
    metrics={"warning":"Preliminary proof-of-concept metrics from a small dataset; do not claim strong generalization.","training_samples":len(df),"train_samples":len(train_idx),"held_out_test_samples":len(test_idx),"class_counts":{str(k):int(v) for k,v in counts.items()},"random_seed":SEED,"chosen_model":chosen,"features":features,"evaluation":evaluations}
    (results/"model_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    metadata={"training_timestamp":datetime.now().astimezone().isoformat(timespec="seconds"),"training_samples":len(df),"class_counts":metrics["class_counts"],"features":features,"parameters":deployment.get_params(),"held_out_metrics":evaluations[chosen],"random_seed":SEED}
    (models/"model_metadata.json").write_text(json.dumps(metadata,indent=2,default=str),encoding="utf-8")
    (results/"model_rules.txt").write_text(export_text(deployment,feature_names=features),encoding="utf-8")
    cm=np.array(evaluations[chosen]["confusion_matrix"]); fig,ax=plt.subplots(figsize=(4,4)); ax.imshow(cm,cmap="Greens"); ax.set(xticks=[0,1],yticks=[0,1],xlabel="Predicted",ylabel="Actual",title="Held-out confusion matrix"); [ax.text(j,i,cm[i,j],ha="center",va="center") for i in range(2) for j in range(2)]; fig.tight_layout(); fig.savefig(results/"confusion_matrix.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); ax.barh(features,deployment.feature_importances_); ax.set_title("Feature importance"); fig.tight_layout(); fig.savefig(results/"feature_importance.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(14,7)); plot_tree(deployment,feature_names=features,class_names=["NORMAL","INTRUSION"],filled=True,rounded=True,ax=ax); fig.tight_layout(); fig.savefig(results/"decision_tree.png",dpi=160); plt.close(fig)
    args=",\n    ".join(f"float {f}" for f in features); header="#pragma once\n\n// Generated from the trained FarmGuard Decision Tree.\ninline int predictIntrusion(\n    "+args+"\n) {\n"+_cpp_tree(deployment.tree_,features)+"}\n"
    (root/"esp32_ml_model.h").write_text(header,encoding="utf-8")
    return metrics

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--dataset",default="data/ml_dataset.csv"); p.add_argument("--output-root",default="."); p.add_argument("--protocol-60-20",action="store_true"); a=p.parse_args()
    try: print(json.dumps(train_development(a.output_root) if a.protocol_60_20 else train(a.dataset,a.output_root),indent=2))
    except (ValueError,FileNotFoundError) as e: print(f"Training refused: {e}"); raise SystemExit(2)
