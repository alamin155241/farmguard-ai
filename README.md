# FarmGuard-AI

## Public source snapshot

ESP32 sensor fusion with MQTT telemetry, a Streamlit dashboard, decision-tree experiments, and optional camera verification. Copy `sketch_feb15a/secrets.example.h` to `secrets.h` and configure your own Wi-Fi before uploading firmware.

This snapshot excludes recorded sensor logs, controlled trial datasets, trained model binaries, camera files, reports, and presentations. Collect your own trials using the dashboard; historical 60/20 protocol commands below require the original controlled dataset and are not runnable from a clean clone. Model-dependent features require training or supplying a compatible model. YOLO weights may download when camera verification is enabled. Dataset-dependent protocol tests require the excluded artifacts. The experimental protocol below is retained for transparency, not as a claim of reproducibility without its data.

Firmware needs the ESP32 board package plus PubSubClient and Adafruit DHT libraries. No hardware or camera was activated for publication.


Local Windows MQTT monitoring plus a ground-truth physical-trial ML workflow.
The ESP32 publishes to `farmguard/XXXX/data` through `broker.emqx.io:1883`.
The laptop subscribes to `farmguard/+/data`.

## Frozen post-exploratory 60/20 protocol

The original `data/ml_dataset.csv` contains 80 controlled trials (40 per
class) and is preserved unchanged. A deterministic seed-42 manifest under
`data/protocol_60_20/` assigns 60 trials (30 per class) to development and 20
trials (10 per class) to a locked final test. This protocol was created after
the previous exploratory pipeline had already used all 80 trials; consequently,
the final-test subset is protected from the new development pipeline but must
not be described as historically untouched before this protocol was created.

Create/verify the frozen protocol with `python protocol_60_20.py`. Train and
select the new development model with:

```powershell
python train_model.py --protocol-60-20
```

Training reads only `development_60.csv`; stratified five-fold CV and parameter
selection never load final-test feature rows. The resulting model is saved as
`models/decision_tree_dev60.joblib`, separately from the legacy model.

The one-time final evaluation is intentionally locked. After the experimental
pipeline is frozen and final evaluation is authorized, run exactly:

```powershell
python evaluate_final_test.py --confirm RUN_FINAL_20_ONCE
```

The evaluator validates checksums and non-overlap, never refits, and refuses to
overwrite existing final-test outputs. Do not run it during development.

LIVE ML TEST remains a separate functional demonstration and writes only to
`results/live_test_results.csv`. Live Monitoring currently verifies a person
with `(CONFIRMED OR ML intrusion)`, whereas completed LIVE ML TEST requires a
person, `CONFIRMED`, and ML intrusion. This known inconsistency is documented
and has not been silently changed.

## Install and start

```powershell
python -m pip install -r requirements.txt
python -m streamlit run dashboard.py
```

You can also double-click `run_dashboard.bat`. Open `http://localhost:8501` if
the browser does not open automatically.

## Exact experimental workflow

1. Power the ESP32 and sensors.
2. Start the dashboard and verify **MQTT Connection: Connected**.
3. Open **ML Dataset Collection**.
4. For a no-intrusion experiment, click **START NORMAL TRIAL**, prepare during
   the three-second countdown, and leave the monitored area undisturbed for the
   eight-second recording.
5. For a real intrusion experiment, click **START INTRUSION TRIAL**, prepare,
   then physically perform the intrusion during the eight-second recording.
6. Collect a target of 10 genuine trials per class. Each trial appends exactly
   one row to `data/ml_dataset.csv`. The selected button is the ground truth;
   sensors and rule state never relabel it.
7. Audit trial data from the dashboard or run `python audit_dataset.py`.
8. Train from the dashboard or run `python train_model.py`.
9. Review held-out metrics, plots, baselines, and exported tree rules.
10. Open **LIVE ML TEST**, select the true action, and perform a new physical
    test. Live tests are not saved into training data.

Twenty trials are enough only for a first working proof of concept. Any
resulting metrics are preliminary and are not sufficient for strong
generalization claims.

## Data separation

- `data/farmguard_events.csv` is a rolling MQTT/system history capped at the
  newest 1,000 readings so it cannot grow indefinitely.
- `data/confirmed_intrusions.csv` records transitions into CONFIRMED.
- `data/ml_dataset.csv` contains one row per controlled physical trial.
- Rule-generated `state`, timestamps, IDs, and trial text are never predictors.
- The audit writes `data/ml_dataset_clean.csv` only when conservative removal
  of invalid/zero-sample/duplicate software rows is justified.

## Training outputs

Training refuses fewer than 10 NORMAL or 10 INTRUSION trials. With sufficient
real data it makes a reproducible stratified held-out
split and compares PIR-only, PIR+beam rule, and a compact Decision Tree. It
saves models under `models/`, reports and plots under `results/`, and generates
`esp32_ml_model.h` from the actual fitted tree. Reported metrics are held-out
test metrics, not training accuracy. The deployment model is then refit on all
valid trials after evaluation.

The generated ML model remains a parallel laptop output. It does not replace
the ESP32 rule-based NORMAL/WATCH/CONFIRMED fusion system.

## Raw continuous-log audit

Run:

```powershell
python raw_sensor_audit.py
```

The read-only report is written to `results/raw_sensor_audit.txt`. The
continuous MQTT log is used only for sensor ranges, calibration, plots, and
debugging. Its rule-generated `state` is never treated as independent ML
ground truth.

## Laptop camera verification

The **Live Monitoring** tab shows rule state, sensor ML, camera result, and the
final verification status together. Enable **Auto-check webcam on WATCH /
CONFIRMED** for immediate triggered checks, or click **CHECK WEBCAM NOW**. The
camera processes multiple frames for about four seconds and treats detection in
any frame as positive. It uses the pretrained YOLO11 nano COCO person class on
CPU, with OpenCV HOG as a crash-safe fallback. It was not trained on the
FarmGuard dataset. LIVE ML TEST starts camera verification concurrently with
the eight-second sensor capture. Its experiment records are kept separately in
`results/live_test_results.csv`; `results/predictions.csv` remains reserved for
held-out Decision Tree evaluation.

A result becomes **VERIFIED INTRUSION** when a person is detected and either
the ESP32 state is CONFIRMED or the sensor Decision Tree predicts INTRUSION.

## Firmware note

Only `MQTT_INTERVAL` was changed from 2000 ms to 500 ms so an eight-second trial
captures enough samples. Pins, payload names, sensor reading, three-second
fusion window, alert hold, buzzer logic, Wi-Fi, broker, and topic are unchanged.
Re-upload the sketch to the ESP32 to use the improved sampling interval.

The public MQTT broker is appropriate for a prototype/demo; never publish
credentials or sensitive information to it.

## Publication checks

The 13 tests in test_camera_service, test_mqtt_service and test_ml_workflow passed using mocks and temporary training data. Dataset-dependent protocol tests were not run because their original trial data is excluded. No live MQTT connection or physical camera was used for verification.
