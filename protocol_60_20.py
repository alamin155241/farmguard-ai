"""Create and validate the post-exploratory FarmGuard 60/20 protocol.

This module never rewrites the source ``data/ml_dataset.csv``.  It freezes that
file byte-for-byte, records checksums, and creates deterministic, stratified
development/final-test artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
SOURCE_DATASET = Path("data/ml_dataset.csv")
FROZEN_DATASET = Path("data/frozen/ml_dataset_80_frozen.csv")
FROZEN_CHECKSUM = Path("data/frozen/ml_dataset_80_frozen.sha256")
FROZEN_RECORD = Path("data/frozen/dataset_record.json")
SPLIT_DIR = Path("data/protocol_60_20")
MANIFEST_PATH = SPLIT_DIR / "split_manifest_seed42.csv"
SPLIT_METADATA_PATH = SPLIT_DIR / "split_metadata.json"
DEVELOPMENT_PATH = SPLIT_DIR / "development_60.csv"
FINAL_TEST_PATH = SPLIT_DIR / "final_test_20.csv"
QUALITY_REPORT_PATH = Path("results/protocol_60_20_dataset_quality.json")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_source(df: pd.DataFrame) -> dict:
    required = {"trial_id", "timestamp", "trial_type", "label", "sample_count"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Source dataset is missing columns: {missing}")
    labels = pd.to_numeric(df["label"], errors="coerce")
    counts = labels.value_counts().to_dict()
    checks = {
        "rows": int(len(df)),
        "normal": int(counts.get(0, 0)),
        "intrusion": int(counts.get(1, 0)),
        "unique_trial_ids": int(df["trial_id"].nunique()),
        "missing_labels": int(labels.isna().sum()),
        "duplicate_trial_ids": int(df["trial_id"].duplicated().sum()),
    }
    expected = {
        "rows": 80, "normal": 40, "intrusion": 40,
        "unique_trial_ids": 80, "missing_labels": 0, "duplicate_trial_ids": 0,
    }
    if checks != expected:
        raise ValueError(f"Source dataset failed frozen-protocol checks: {checks}")
    return checks


def validate_split(development: pd.DataFrame, final_test: pd.DataFrame) -> dict:
    dev_ids = set(development["trial_id"])
    final_ids = set(final_test["trial_id"])
    dev_labels = pd.to_numeric(development["label"], errors="raise")
    final_labels = pd.to_numeric(final_test["label"], errors="raise")
    summary = {
        "development_rows": int(len(development)),
        "development_normal": int((dev_labels == 0).sum()),
        "development_intrusion": int((dev_labels == 1).sum()),
        "final_test_rows": int(len(final_test)),
        "final_test_normal": int((final_labels == 0).sum()),
        "final_test_intrusion": int((final_labels == 1).sum()),
        "overlap_count": int(len(dev_ids & final_ids)),
        "combined_unique_trial_ids": int(len(dev_ids | final_ids)),
    }
    expected = {
        "development_rows": 60, "development_normal": 30,
        "development_intrusion": 30, "final_test_rows": 20,
        "final_test_normal": 10, "final_test_intrusion": 10,
        "overlap_count": 0, "combined_unique_trial_ids": 80,
    }
    if summary != expected:
        raise ValueError(f"60/20 split validation failed: {summary}")
    return summary


def load_and_validate_protocol(root: str | Path = ".") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    root = Path(root)
    metadata_path = root / SPLIT_METADATA_PATH
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    paths = metadata["paths"]
    frozen = root / paths["frozen_dataset"]
    manifest_path = root / paths["manifest"]
    development_path = root / paths["development"]
    final_path = root / paths["final_test"]
    expected_hashes = metadata["sha256"]
    for name, path in (
        ("frozen_dataset", frozen), ("manifest", manifest_path),
        ("development", development_path), ("final_test", final_path),
    ):
        actual = sha256_file(path)
        if actual != expected_hashes[name]:
            raise ValueError(f"Checksum mismatch for {name}: {path}")
    development = pd.read_csv(development_path)
    final_test = pd.read_csv(final_path)
    validate_split(development, final_test)
    manifest = pd.read_csv(manifest_path)
    if set(manifest.loc[manifest.split == "development", "trial_id"]) != set(development.trial_id):
        raise ValueError("Development IDs do not match the frozen split manifest")
    if set(manifest.loc[manifest.split == "final_test", "trial_id"]) != set(final_test.trial_id):
        raise ValueError("Final-test IDs do not match the frozen split manifest")
    return development, final_test, metadata


def load_development_protocol(root: str | Path = ".") -> tuple[pd.DataFrame, dict]:
    """Load development rows without reading final-test feature data."""
    root = Path(root)
    metadata = json.loads((root / SPLIT_METADATA_PATH).read_text(encoding="utf-8"))
    paths = metadata["paths"]
    expected_hashes = metadata["sha256"]
    for name in ("frozen_dataset", "manifest", "development", "final_test"):
        path = root / paths[name]
        if sha256_file(path) != expected_hashes[name]:
            raise ValueError(f"Checksum mismatch for {name}: {path}")
    development = pd.read_csv(root / paths["development"])
    manifest = pd.read_csv(root / paths["manifest"])
    dev_manifest_ids = set(manifest.loc[manifest.split == "development", "trial_id"])
    final_manifest_ids = set(manifest.loc[manifest.split == "final_test", "trial_id"])
    if set(development.trial_id) != dev_manifest_ids:
        raise ValueError("Development IDs do not match the frozen split manifest")
    if set(development.trial_id) & final_manifest_ids:
        raise ValueError("Development data overlaps final-test IDs")
    labels = pd.to_numeric(development["label"], errors="raise")
    if len(development) != 60 or int((labels == 0).sum()) != 30 or int((labels == 1).sum()) != 30:
        raise ValueError("Development data must contain 30 NORMAL and 30 INTRUSION trials")
    return development, metadata


def create_protocol(root: str | Path = ".") -> dict:
    root = Path(root)
    source = root / SOURCE_DATASET
    frozen = root / FROZEN_DATASET
    checksum_path = root / FROZEN_CHECKSUM
    record_path = root / FROZEN_RECORD
    split_dir = root / SPLIT_DIR
    source_bytes = source.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    source_df = pd.read_csv(source)
    source_checks = validate_source(source_df)

    existing_metadata_path = root / SPLIT_METADATA_PATH
    if existing_metadata_path.exists():
        existing = json.loads(existing_metadata_path.read_text(encoding="utf-8"))
        if existing.get("sha256", {}).get("source_dataset") != source_hash:
            raise ValueError("Source dataset changed after the 60/20 protocol was frozen")
        load_and_validate_protocol(root)
        return existing

    frozen.parent.mkdir(parents=True, exist_ok=True)
    if frozen.exists() and frozen.read_bytes() != source_bytes:
        raise FileExistsError(f"Refusing to replace a different frozen dataset: {frozen}")
    if not frozen.exists():
        shutil.copyfile(source, frozen)
    if frozen.read_bytes() != source_bytes:
        raise RuntimeError("Frozen dataset is not byte-for-byte identical to the source")
    checksum_path.write_text(f"{source_hash}  {FROZEN_DATASET.as_posix()}\n", encoding="utf-8")

    indices = list(range(len(source_df)))
    dev_idx, final_idx = train_test_split(
        indices, test_size=20, stratify=source_df["label"], random_state=SEED,
    )
    development = source_df.iloc[sorted(dev_idx)].copy()
    final_test = source_df.iloc[sorted(final_idx)].copy()
    split_summary = validate_split(development, final_test)

    split_by_id = {trial_id: "development" for trial_id in development.trial_id}
    split_by_id.update({trial_id: "final_test" for trial_id in final_test.trial_id})
    manifest = source_df[["trial_id", "label", "timestamp"]].copy()
    manifest.insert(1, "split", manifest["trial_id"].map(split_by_id))

    split_dir.mkdir(parents=True, exist_ok=True)
    development.to_csv(root / DEVELOPMENT_PATH, index=False)
    final_test.to_csv(root / FINAL_TEST_PATH, index=False)
    manifest.to_csv(root / MANIFEST_PATH, index=False)

    hashes = {
        "source_dataset": source_hash,
        "frozen_dataset": sha256_file(frozen),
        "manifest": sha256_file(root / MANIFEST_PATH),
        "development": sha256_file(root / DEVELOPMENT_PATH),
        "final_test": sha256_file(root / FINAL_TEST_PATH),
    }
    metadata = {
        "protocol": "post-exploratory deterministic 60/20 evaluation",
        "scientific_disclosure": (
            "This protocol was created after earlier exploratory analysis used all 80 trials. "
            "The final-test subset is protected from the new protocol's model development, but "
            "must not be described as historically untouched before this protocol was created."
        ),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "random_seed": SEED,
        "source_verification": source_checks,
        "split_summary": split_summary,
        "paths": {
            "source_dataset": SOURCE_DATASET.as_posix(),
            "frozen_dataset": FROZEN_DATASET.as_posix(),
            "manifest": MANIFEST_PATH.as_posix(),
            "development": DEVELOPMENT_PATH.as_posix(),
            "final_test": FINAL_TEST_PATH.as_posix(),
        },
        "sha256": hashes,
    }
    (root / SPLIT_METADATA_PATH).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    record = {
        "description": "Byte-for-byte frozen FarmGuard 80-trial dataset",
        "source": SOURCE_DATASET.as_posix(),
        "frozen_copy": FROZEN_DATASET.as_posix(),
        "sha256": source_hash,
        "verification": source_checks,
        "protocol_metadata": SPLIT_METADATA_PATH.as_posix(),
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    samples = pd.to_numeric(source_df["sample_count"], errors="coerce")
    quality = {
        "dataset_sha256": source_hash,
        "rows": int(len(source_df)),
        "sample_count_min": int(samples.min()),
        "sample_count_max": int(samples.max()),
        "sample_count_mean": float(samples.mean()),
        "trials_below_8_samples": int((samples < 8).sum()),
        "intended_trial_duration_seconds": 8,
        "firmware_mqtt_interval_ms": 500,
        "expected_periodic_samples_at_500ms": 16,
        "warning": "Low-sample trials are flagged but retained unchanged; no historical rows were deleted or regenerated.",
    }
    quality_path = root / QUALITY_REPORT_PATH
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    quality_path.write_text(json.dumps(quality, indent=2) + "\n", encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(json.dumps(create_protocol(args.root), indent=2))
