"""Guarded one-time evaluation of the frozen FarmGuard 20-row final-test set.

This command never trains or refits a model.  It refuses to run unless the user
passes the explicit confirmation flag, and it refuses to overwrite an existing
final evaluation.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from protocol_60_20 import load_and_validate_protocol, sha256_file
from train_model import metric_dict

CONFIRMATION = "RUN_FINAL_20_ONCE"


def evaluate_final_test(root=".", confirmation=""):
    if confirmation != CONFIRMATION:
        raise PermissionError(
            f"Final evaluation is locked. Pass --confirm {CONFIRMATION} only when ready."
        )
    root = Path(root)
    results = root / "results"
    metrics_path = results / "final_test_metrics.json"
    predictions_path = results / "final_test_predictions.csv"
    figure_path = results / "final_test_confusion_matrix.png"
    if any(path.exists() for path in (metrics_path, predictions_path, figure_path)):
        raise FileExistsError("Final-test output already exists; refusing to rerun or overwrite it")

    development, final_test, protocol = load_and_validate_protocol(root)
    metadata_path = results / "development_model_metadata.json"
    model_path = root / "models/decision_tree_dev60.joblib"
    feature_path = root / "models/feature_names_dev60.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    features = json.loads(feature_path.read_text(encoding="utf-8"))
    if set(development.trial_id) & set(final_test.trial_id):
        raise ValueError("Development/final-test overlap detected")
    checks = {
        "model_sha256": sha256_file(model_path),
        "protocol_metadata_sha256": sha256_file(root / "data/protocol_60_20/split_metadata.json"),
        "split_manifest_sha256": protocol["sha256"]["manifest"],
        "development_dataset_sha256": protocol["sha256"]["development"],
        "final_test_dataset_sha256": protocol["sha256"]["final_test"],
        "frozen_dataset_sha256": protocol["sha256"]["frozen_dataset"],
    }
    for key, actual in checks.items():
        if metadata.get(key) != actual:
            raise ValueError(f"Development model/protocol metadata mismatch: {key}")
    if set(metadata["development_trial_ids"]) != set(development.trial_id):
        raise ValueError("Saved model development IDs do not match the frozen development set")
    if final_test[features + ["label"]].isna().any().any():
        raise ValueError("Final-test data contains missing model values")

    model = joblib.load(model_path)
    x = final_test[features].astype(float)
    y = pd.to_numeric(final_test["label"], errors="raise").astype(int)
    predicted = model.predict(x).astype(int)
    probabilities = model.predict_proba(x)[:, list(model.classes_).index(1)]
    metrics = metric_dict(y, predicted)
    report = {
        "protocol": protocol["protocol"],
        "scientific_disclosure": protocol["scientific_disclosure"],
        "evaluated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "evaluation": "one-time frozen 20-row final test; no refit performed",
        "final_test_samples": int(len(final_test)),
        "class_counts": {"0": int((y == 0).sum()), "1": int((y == 1).sum())},
        "metrics": metrics,
        "verification": checks,
    }
    results.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "trial_id": final_test["trial_id"],
        "actual": y,
        "predicted": predicted,
        "intrusion_probability": probabilities,
        "split": "final_test",
    }).to_csv(predictions_path, index=False)
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted", ylabel="Actual", title="Frozen final-test confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()
    try:
        print(json.dumps(evaluate_final_test(args.root, args.confirm), indent=2))
    except (PermissionError, FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"Final evaluation refused: {exc}")
        raise SystemExit(2)
