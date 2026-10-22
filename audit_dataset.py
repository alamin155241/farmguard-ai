"""Conservative audit for FarmGuard physical-trial data."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from dataset_service import ML_FIELDS, MODEL_B_FEATURES

def audit_dataset(path="data/ml_dataset.csv", clean_path="data/ml_dataset_clean.csv", write_clean=True):
    path = Path(path)
    if not path.exists(): raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    missing_columns = [c for c in ML_FIELDS if c not in df.columns]
    report = {"rows": len(df), "missing_columns": missing_columns}
    if missing_columns:
        report["cleaned_file_written"] = False
        return report
    labels = pd.to_numeric(df["label"], errors="coerce")
    samples = pd.to_numeric(df["sample_count"], errors="coerce")
    report.update({
        "class_counts": {str(k): int(v) for k,v in labels.value_counts(dropna=False).items()},
        "missing_values": {k:int(v) for k,v in df.isna().sum().items() if v},
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_trial_ids": int(df["trial_id"].duplicated().sum()),
        "invalid_labels": int((~labels.isin([0,1])).sum()),
        "zero_sample_trials": int((samples == 0).sum()),
        "very_low_sample_trials": int(((samples > 0) & (samples < 8)).sum()),
        "constant_columns": [c for c in df.columns if df[c].nunique(dropna=False) <= 1],
    })
    numeric = [c for c in MODEL_B_FEATURES + ["sample_count", "trial_duration_sec"] if c in df]
    report["feature_ranges"] = {c:{"min": float(pd.to_numeric(df[c],errors="coerce").min()), "max":float(pd.to_numeric(df[c],errors="coerce").max())} for c in numeric if pd.to_numeric(df[c],errors="coerce").notna().any()}
    suspicious = {}
    for c in ("pir_detected", "beam_break_detected"):
        values = pd.to_numeric(df[c], errors="coerce"); suspicious[f"{c}_outside_0_or_1"] = int((~values.isin([0, 1]) & values.notna()).sum())
    for c in ("pir_ratio", "beam_break_ratio"):
        values = pd.to_numeric(df[c], errors="coerce"); suspicious[f"{c}_outside_0_to_1"] = int(((values < 0) | (values > 1)).sum())
    for c in ("min_ldr", "max_ldr", "avg_ldr", "min_gas", "max_gas", "avg_gas"):
        values = pd.to_numeric(df[c], errors="coerce"); suspicious[f"{c}_negative"] = int((values < 0).sum())
    report["suspicious_values"] = {k:v for k,v in suspicious.items() if v}
    report["sensor_activity_by_class"] = df.assign(label=labels).groupby("label")[["pir_detected","beam_break_detected","pir_ratio","beam_break_ratio"]].mean().round(3).to_dict(orient="index")
    impossible = (~labels.isin([0,1])) | samples.isna() | (samples <= 0)
    clean = df.loc[~impossible].drop_duplicates().drop_duplicates(subset=["trial_id"], keep="first")
    if write_clean and (len(clean) != len(df)):
        Path(clean_path).parent.mkdir(parents=True, exist_ok=True); clean.to_csv(clean_path,index=False)
        report["cleaned_file_written"] = True; report["cleaned_rows"] = len(clean)
    else: report["cleaned_file_written"] = False
    return report

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--input",default="data/ml_dataset.csv"); p.add_argument("--output",default="data/ml_dataset_clean.csv"); a=p.parse_args()
    print(json.dumps(audit_dataset(a.input,a.output),indent=2,default=str))
