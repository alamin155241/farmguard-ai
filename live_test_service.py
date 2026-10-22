"""Separate append-only logging for physical LIVE ML TEST experiments."""
from __future__ import annotations
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, MutableMapping

FIELDS = ["timestamp", "ground_truth", "rule_state", "ml_prediction", "ml_probability", "camera_result", "final_result", "correct"]

def append_live_result(row, path="results/live_test_results.csv"):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); new=not path.exists() or path.stat().st_size==0
    with path.open("a",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=FIELDS)
        if new: writer.writeheader()
        writer.writerow({"timestamp":datetime.now().astimezone().isoformat(timespec="seconds"),**{k:row.get(k,"") for k in FIELDS if k!="timestamp"}})


def finalize_live_test(sensor_result: dict[str, Any], camera_snapshot: dict[str, Any]) -> dict[str, Any] | None:
    """Combine completed sensor/ML and camera work into one immutable result."""
    if not sensor_result or camera_snapshot.get("running"):
        return None
    camera_result = camera_snapshot.get("result")
    if camera_result is None:
        return None
    if camera_result.error:
        camera_name = "ERROR"
    else:
        camera_name = "PERSON DETECTED" if camera_result.person_detected else "NO PERSON"
    ground_truth = sensor_result["truth"]
    rule_state = sensor_result["rule"]
    ml_prediction = sensor_result["prediction"]
    correct = ml_prediction == ground_truth
    final_result = (
        "VERIFIED INTRUSION"
        if rule_state == "CONFIRMED"
        and ml_prediction == "INTRUSION"
        and camera_name == "PERSON DETECTED"
        else "NOT VERIFIED"
    )
    return {
        "trial_id": sensor_result.get("row", {}).get("trial_id", sensor_result.get("trial_id", "")),
        "ground_truth": ground_truth,
        "rule_state": rule_state,
        "ml_prediction": ml_prediction,
        "ml_probability": sensor_result.get("probability"),
        "camera_result": camera_name,
        "correctness": "CORRECT" if correct else "INCORRECT",
        "correct": correct,
        "final_result": final_result,
        "warnings": list(sensor_result.get("warnings", [])),
        "camera_frame": camera_snapshot.get("frame"),
        "camera_detector": camera_snapshot.get("detector", ""),
        "camera_frames_processed": camera_snapshot.get("frames_processed", 0),
        "camera_error": camera_result.error,
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def persist_completed_live_test(
    session_state: MutableMapping[str, Any],
    sensor_result: dict[str, Any],
    camera_snapshot: dict[str, Any],
    path="results/live_test_results.csv",
) -> dict[str, Any] | None:
    """Finalize once, persist in session state, and append exactly one CSV row."""
    completed = finalize_live_test(sensor_result, camera_snapshot)
    if completed is None:
        return None
    previous = session_state.get("last_live_test_result")
    if previous and previous.get("trial_id") == completed.get("trial_id"):
        return previous
    session_state["last_live_test_result"] = completed
    append_live_result(completed, path)
    return completed
