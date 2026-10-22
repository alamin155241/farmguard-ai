import json
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import protocol_60_20
from evaluate_final_test import evaluate_final_test
from protocol_60_20 import (
    FINAL_TEST_PATH,
    FROZEN_DATASET,
    MANIFEST_PATH,
    SOURCE_DATASET,
    SPLIT_METADATA_PATH,
    load_development_protocol,
    sha256_file,
    validate_source,
    validate_split,
)


class Protocol6020Tests(unittest.TestCase):
    def test_original_and_frozen_dataset_are_identical(self):
        self.assertEqual(Path(SOURCE_DATASET).read_bytes(), Path(FROZEN_DATASET).read_bytes())
        self.assertEqual(sha256_file(SOURCE_DATASET), sha256_file(FROZEN_DATASET))
        self.assertEqual(validate_source(pd.read_csv(SOURCE_DATASET))["rows"], 80)

    def test_frozen_split_counts_and_no_overlap(self):
        metadata = json.loads(Path(SPLIT_METADATA_PATH).read_text(encoding="utf-8"))
        development = pd.read_csv(metadata["paths"]["development"])
        final_test = pd.read_csv(metadata["paths"]["final_test"])
        summary = validate_split(development, final_test)
        self.assertEqual(summary["development_rows"], 60)
        self.assertEqual(summary["final_test_rows"], 20)
        self.assertEqual(summary["overlap_count"], 0)

    def test_development_loader_never_reads_final_test_rows(self):
        calls = []
        original = protocol_60_20.pd.read_csv

        def recording_read_csv(path, *args, **kwargs):
            calls.append(Path(path))
            return original(path, *args, **kwargs)

        with patch.object(protocol_60_20.pd, "read_csv", side_effect=recording_read_csv):
            development, _ = load_development_protocol()
        self.assertEqual(len(development), 60)
        self.assertNotIn(Path(FINAL_TEST_PATH), calls)
        self.assertIn(Path(MANIFEST_PATH), calls)

    def test_development_outputs_declare_final_test_not_evaluated(self):
        metrics = json.loads(Path("results/development_cv_metrics.json").read_text(encoding="utf-8"))
        metadata = json.loads(Path("results/development_model_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["development_samples"], 60)
        self.assertFalse(metrics["final_test_evaluated"])
        self.assertFalse(metadata["final_test_rows_loaded_during_training"])
        self.assertEqual(len(metadata["development_trial_ids"]), 60)

    def test_final_evaluation_requires_confirmation_and_preserves_completed_outputs(self):
        with self.assertRaises(PermissionError):
            evaluate_final_test(confirmation="")
        self.assertTrue(Path("results/final_test_metrics.json").exists())
        self.assertTrue(Path("results/final_test_predictions.csv").exists())
        self.assertTrue(Path("results/final_test_confusion_matrix.png").exists())

    def test_live_results_are_outside_offline_datasets(self):
        self.assertNotEqual(Path("results/live_test_results.csv"), Path(SOURCE_DATASET))
        self.assertNotEqual(Path("results/live_test_results.csv"), Path(FINAL_TEST_PATH))


if __name__ == "__main__":
    unittest.main()
