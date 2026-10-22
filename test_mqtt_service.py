import csv
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

from mqtt_service import EVENT_FIELDS, MQTTService, parse_sensor_payload


def payload(state="NORMAL"):
    return json.dumps({
        "pir": 1, "ldr": 86, "beam_blocked": 0, "gas": 212,
        "temperature": 33.30, "state": state,
    }).encode()


class FarmGuardTests(unittest.TestCase):
    def test_parse_all_states_and_online_status(self):
        for state in ("NORMAL", "WATCH", "CONFIRMED"):
            self.assertEqual(parse_sensor_payload(payload(state))["state"], state)
        self.assertIsNone(parse_sensor_payload(b'{"status":"FarmGuard-AI ONLINE"}'))

    def test_csv_append_and_confirmed_transition_deduplication(self):
        with tempfile.TemporaryDirectory() as folder:
            service = MQTTService(folder)
            topic = "farmguard/ABCD/data"
            for state in ("NORMAL", "CONFIRMED", "CONFIRMED", "WATCH", "CONFIRMED"):
                service._on_message(None, None, SimpleNamespace(topic=topic, payload=payload(state)))
            with Path(service.events_path).open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 5)
            with Path(service.confirmed_path).open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)

    def test_event_csv_keeps_only_newest_1000_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "events.csv"
            MQTTService._ensure_csv(path, EVENT_FIELDS)
            for index in range(1005):
                row = {key: index for key in EVENT_FIELDS}
                MQTTService._append_csv(path, EVENT_FIELDS, row)
            MQTTService._trim_csv(path, EVENT_FIELDS, max_rows=1000)
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1000)
            self.assertEqual(rows[0]["timestamp"], "5")
            self.assertEqual(rows[-1]["timestamp"], "1004")

    def test_trim_failure_does_not_interrupt_in_memory_ingestion(self):
        with tempfile.TemporaryDirectory() as folder:
            service = MQTTService(folder)
            service._event_row_count = 1000
            marker = service.snapshot()["sequence"]
            message = SimpleNamespace(topic="farmguard/ABCD/data", payload=payload("NORMAL"))
            with patch.object(service, "_trim_csv", side_effect=PermissionError("file locked")):
                service._on_message(None, None, message)
                first = service.snapshot()
                self.assertEqual(first["sequence"], marker + 1)
                self.assertEqual(len(first["history"]), 1)
                self.assertEqual(len(service.samples_after(marker)), 1)
                self.assertIn("Continuous event log unavailable", first["last_error"])

                # A later message must still be accepted after the same disk failure.
                service._on_message(None, None, message)
                second = service.snapshot()
                self.assertEqual(second["sequence"], marker + 2)
                self.assertEqual(len(second["history"]), 2)
                self.assertEqual(len(service.samples_after(marker)), 2)

    def test_append_failure_does_not_interrupt_in_memory_ingestion(self):
        with tempfile.TemporaryDirectory() as folder:
            service = MQTTService(folder)
            marker = service.snapshot()["sequence"]
            message = SimpleNamespace(topic="farmguard/ABCD/data", payload=payload("WATCH"))
            with patch.object(service, "_append_csv", side_effect=OSError("disk unavailable")):
                service._on_message(None, None, message)
                service._on_message(None, None, message)
            snap = service.snapshot()
            self.assertEqual(snap["sequence"], marker + 2)
            self.assertEqual(len(snap["history"]), 2)
            self.assertEqual(len(service.samples_after(marker)), 2)
            self.assertIn("Continuous event log unavailable", snap["last_error"])


if __name__ == "__main__":
    unittest.main()
