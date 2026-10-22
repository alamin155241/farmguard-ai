"""Thread-safe MQTT ingestion and CSV logging for FarmGuard-AI."""

from __future__ import annotations

import csv
import importlib
import json
import subprocess
import sys
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import paho.mqtt.client as mqtt
except ModuleNotFoundError as exc:
    # Streamlit is sometimes launched by a different Windows Python than pip.
    # Install into the interpreter that is actually executing this file.
    if exc.name != "paho":
        raise
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "paho-mqtt>=2.1,<3"],
        timeout=180,
    )
    importlib.invalidate_caches()
    import paho.mqtt.client as mqtt

BROKER = "broker.emqx.io"
PORT = 1883
TOPIC = "farmguard/+/data"
VALID_STATES = {"NORMAL", "WATCH", "CONFIRMED"}
EVENT_FIELDS = [
    "timestamp", "topic", "pir", "ldr", "beam_blocked", "gas",
    "temperature", "state",
]
CONFIRMED_FIELDS = EVENT_FIELDS.copy()
MAX_EVENT_ROWS = 1000


def parse_sensor_payload(payload: str | bytes) -> dict[str, Any] | None:
    """Return a normalized sensor reading, or None for status/invalid messages."""
    try:
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        data = json.loads(payload)
        if not isinstance(data, dict) or "status" in data:
            return None
        required = {"pir", "ldr", "beam_blocked", "gas", "temperature", "state"}
        if not required.issubset(data):
            return None
        state = str(data["state"]).upper()
        if state not in VALID_STATES:
            return None
        return {
            "pir": int(data["pir"]),
            "ldr": int(data["ldr"]),
            "beam_blocked": int(data["beam_blocked"]),
            "gas": int(data["gas"]),
            "temperature": float(data["temperature"]),
            "state": state,
        }
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


class MQTTService:
    """Owns the background MQTT client and exposes snapshots to Streamlit."""

    def __init__(self, data_dir: str | Path = "data", history_size: int = 500):
        self._lock = threading.RLock()
        self._history: deque[dict[str, Any]] = deque(maxlen=history_size)
        self._connected = False
        self._last_message: str | None = None
        self._latest: dict[str, Any] | None = None
        self._device_status: str | None = None
        self._last_error: str | None = None
        self._last_state_by_topic: dict[str, str] = {}
        self._sequence = 0
        self.data_dir = Path(data_dir)
        self.events_path = self.data_dir / "farmguard_events.csv"
        self.confirmed_path = self.data_dir / "confirmed_intrusions.csv"
        self._ensure_csv(self.events_path, EVENT_FIELDS)
        self._ensure_csv(self.confirmed_path, CONFIRMED_FIELDS)
        with self.events_path.open("r", encoding="utf-8") as handle:
            self._event_row_count = max(0, sum(1 for _ in handle) - 1)
        if self._event_row_count > MAX_EVENT_ROWS:
            self._trim_csv(self.events_path, EVENT_FIELDS, MAX_EVENT_ROWS)
            self._event_row_count = MAX_EVENT_ROWS

        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except AttributeError:  # paho-mqtt 1.x compatibility
            self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.reconnect_delay_set(min_delay=1, max_delay=30)

    @staticmethod
    def _ensure_csv(path: Path, fields: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.stat().st_size == 0:
            with path.open("a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=fields).writeheader()

    @staticmethod
    def _append_csv(
        path: Path,
        fields: list[str],
        row: dict[str, Any],
    ) -> None:
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writerow({key: row.get(key, "") for key in fields})
    @staticmethod
    def _trim_csv(path: Path, fields: list[str], max_rows: int) -> None:
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = deque(csv.DictReader(handle), maxlen=max_rows)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        with temp_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temp_path.replace(path)

    def start(self) -> None:
        """Start non-blocking MQTT networking with automatic reconnect."""
        try:
            self.client.connect_async(BROKER, PORT, keepalive=30)
            self.client.loop_start()
        except Exception as exc:  # Dashboard remains usable while offline.
            with self._lock:
                self._last_error = str(exc)

    def stop(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        # paho 2.x supplies a ReasonCode object; paho 1.x supplies an int.
        success = not reason_code.is_failure if hasattr(reason_code, "is_failure") else reason_code == 0
        with self._lock:
            self._connected = success
            self._last_error = None if success else f"MQTT connection refused: {reason_code}"
        if success:
            client.subscribe(TOPIC, qos=0)

    def _on_disconnect(self, client, userdata, *args):
        with self._lock:
            self._connected = False

    def _on_message(self, client, userdata, message):
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        try:
            decoded = json.loads(message.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            with self._lock:
                self._last_error = "Ignored an invalid MQTT message"
            return

        if isinstance(decoded, dict) and "status" in decoded:
            with self._lock:
                self._last_message = now
                self._device_status = str(decoded["status"])
            return

        reading = parse_sensor_payload(message.payload)
        if reading is None:
            with self._lock:
                self._last_error = "Ignored an incomplete sensor message"
            return

        row = {"timestamp": now, "topic": message.topic, **reading}
        with self._lock:
            self._sequence += 1
            row["sequence"] = self._sequence
            previous_state = self._last_state_by_topic.get(message.topic)
            self._last_state_by_topic[message.topic] = reading["state"]
            self._history.append(row)
            self._latest = row
            self._last_message = now
            self._last_error = None

            # Trial capture is an in-memory operation. Continuous CSV logging is
            # supporting diagnostics and must never prevent samples_after() from
            # seeing an otherwise valid MQTT reading (for example, when Windows
            # temporarily locks the rolling CSV during replacement).
            logging_errors = []
            try:
                self._append_csv(self.events_path, EVENT_FIELDS, row)
                self._event_row_count += 1
                if self._event_row_count > MAX_EVENT_ROWS:
                    self._trim_csv(self.events_path, EVENT_FIELDS, MAX_EVENT_ROWS)
                    self._event_row_count = MAX_EVENT_ROWS
            except OSError as exc:
                logging_errors.append(f"Continuous event log unavailable: {exc}")
            if reading["state"] == "CONFIRMED" and previous_state != "CONFIRMED":
                try:
                    self._append_csv(self.confirmed_path, CONFIRMED_FIELDS, row)
                except OSError as exc:
                    logging_errors.append(f"Confirmed-event log unavailable: {exc}")
            if logging_errors:
                self._last_error = " | ".join(logging_errors)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "connected": self._connected,
                "last_message": self._last_message,
                "latest": dict(self._latest) if self._latest else None,
                "history": [dict(row) for row in self._history],
                "device_status": self._device_status,
                "last_error": self._last_error,
                "sequence": self._sequence,
            }

    def samples_after(self, sequence: int) -> list[dict[str, Any]]:
        """Return buffered readings received after a trial marker."""
        with self._lock:
            return [dict(row) for row in self._history if row.get("sequence", 0) > sequence]
