"""Ground-truth trial aggregation and append-only ML dataset storage."""

from __future__ import annotations

import csv
import math
import uuid
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

ML_FIELDS = [
    "trial_id", "timestamp", "trial_type", "pir_detected", "min_ldr", "max_ldr",
    "avg_ldr", "beam_break_detected", "beam_break_sample_count", "pir_sample_count",
    "pir_ratio", "beam_break_ratio", "avg_gas", "min_gas", "max_gas",
    "avg_temperature", "sample_count", "trial_duration_sec", "label",
]
MODEL_A_FEATURES = [
    "pir_detected", "min_ldr", "max_ldr", "avg_ldr", "beam_break_detected",
    "pir_sample_count", "beam_break_sample_count", "pir_ratio", "beam_break_ratio",
]
MODEL_B_FEATURES = MODEL_A_FEATURES + ["avg_gas", "min_gas", "max_gas", "avg_temperature"]


def aggregate_trial(samples: list[dict[str, Any]], label: int, duration: float = 8.0) -> dict[str, Any]:
    if label not in (0, 1):
        raise ValueError("Ground-truth label must be 0 or 1")
    valid = [s for s in samples if all(k in s for k in ("pir", "ldr", "beam_blocked", "gas", "temperature"))]
    if not valid:
        raise ValueError("NO MQTT DATA RECEIVED")
    count = len(valid)
    pirs = [int(s["pir"]) for s in valid]
    beams = [int(s["beam_blocked"]) for s in valid]
    ldrs = [float(s["ldr"]) for s in valid]
    gases = [float(s["gas"]) for s in valid]
    temperatures = [float(s["temperature"]) for s in valid if math.isfinite(float(s["temperature"]))]
    return {
        "trial_id": f"FG-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6].upper()}",
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "trial_type": "NORMAL" if label == 0 else "INTRUSION",
        "pir_detected": int(any(pirs)), "min_ldr": min(ldrs), "max_ldr": max(ldrs),
        "avg_ldr": round(mean(ldrs), 3), "beam_break_detected": int(any(beams)),
        "beam_break_sample_count": sum(beams), "pir_sample_count": sum(pirs),
        "pir_ratio": round(sum(pirs) / count, 4), "beam_break_ratio": round(sum(beams) / count, 4),
        "avg_gas": round(mean(gases), 3), "min_gas": min(gases), "max_gas": max(gases),
        "avg_temperature": round(mean(temperatures), 3) if temperatures else "",
        "sample_count": count, "trial_duration_sec": round(duration, 2), "label": label,
    }


def quality_warnings(row: dict[str, Any]) -> list[str]:
    warnings = []
    if int(row["sample_count"]) < 8:
        warnings.append("TOO FEW MQTT SAMPLES")
    if int(row["label"]) == 0:
        if float(row["pir_ratio"]) >= 0.5: warnings.append("NORMAL trial contained substantial motion")
        if int(row["beam_break_detected"]): warnings.append("NORMAL trial contained a laser break")
    elif not int(row["pir_detected"]) and not int(row["beam_break_detected"]):
        warnings.append("INTRUSION trial contained no PIR or laser evidence")
    return warnings


class DatasetStore:
    def __init__(self, path: str | Path = "data/ml_dataset.csv"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists() or self.path.stat().st_size == 0:
            with self.path.open("a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=ML_FIELDS).writeheader()

    def append(self, row: dict[str, Any]) -> None:
        if int(row["label"]) not in (0, 1): raise ValueError("Invalid ground-truth label")
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=ML_FIELDS).writerow({k: row.get(k, "") for k in ML_FIELDS})
