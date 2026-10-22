"""Near-real-time, on-demand laptop webcam person verification."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class CameraResult:
    person_detected: bool
    frame: Any | None
    checked_at: str
    error: str | None = None
    detector: str = "uninitialized"
    confidence: float | None = None
    frames_processed: int = 0


class CameraVerifier:
    """Runs a multi-frame verification window without blocking Streamlit."""

    def __init__(self, camera_index: int = 0, duration: float = 4.0, target_fps: float = 7.0, prefer_yolo: bool = True):
        self.camera_index = camera_index
        self.duration = duration
        self.target_fps = target_fps
        self._lock = threading.RLock()
        self._worker: threading.Thread | None = None
        self._running = False
        self._latest_frame = None
        self._frames_processed = 0
        self._detected_any = False
        self._best_confidence: float | None = None
        self._error: str | None = None
        self._detector_name = "uninitialized"
        self._positive_until = 0.0
        self.last_result: CameraResult | None = None
        self.last_trigger_state = "NORMAL"
        self._yolo = None
        self._hog = None
        self._init_detector() if prefer_yolo else self._init_hog()

    def _init_detector(self) -> None:
        try:
            from ultralytics import YOLO
            self._yolo = YOLO("yolo11n.pt")
            self._detector_name = "YOLO11n COCO (CPU)"
            # Pay the one-time PyTorch setup cost while the dashboard starts,
            # not after WATCH/CONFIRMED triggers the four-second camera window.
            self._yolo.predict(
                np.zeros((320, 320, 3), dtype=np.uint8),
                classes=[0], conf=0.25, imgsz=320, device="cpu", verbose=False,
            )
        except Exception as exc:
            self._error = f"YOLO unavailable; using OpenCV HOG fallback: {exc}"
            self._init_hog()

    def _init_hog(self) -> None:
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self._detector_name = "OpenCV HOG fallback"

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    def start(self, duration: float | None = None) -> bool:
        """Start immediately; return False when a verification is already active."""
        with self._lock:
            if self._running:
                return False
            self._running = True
            self._latest_frame = None
            self._frames_processed = 0
            self._detected_any = False
            self._best_confidence = None
            self.last_result = None
            self._worker = threading.Thread(
                target=self._verify_window,
                args=(duration or self.duration,),
                daemon=True,
                name="farmguard-camera-verifier",
            )
            self._worker.start()
            return True

    def _detect(self, frame):
        annotated = frame.copy()
        detections: list[tuple[int, int, int, int, float]] = []
        if self._yolo is not None:
            try:
                result = self._yolo.predict(frame, classes=[0], conf=0.25, imgsz=640, device="cpu", verbose=False)[0]
                for box in result.boxes:
                    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                    detections.append((x1, y1, x2, y2, float(box.conf[0])))
            except Exception as exc:
                self._yolo = None
                self._error = f"YOLO inference failed; using OpenCV HOG fallback: {exc}"
                if self._hog is None:
                    self._init_hog()
        if self._yolo is None and self._hog is not None:
            boxes, weights = self._hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
            for (x, y, w, h), weight in zip(boxes, weights):
                detections.append((x, y, x + w, y + h, float(weight)))
        for x1, y1, x2, y2, confidence in detections:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(annotated, f"Person {confidence:.0%}", (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return annotated, detections

    def _open_camera(self):
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.camera_index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _verify_window(self, duration: float) -> None:
        cap = self._open_camera()
        if not cap.isOpened():
            cap.release()
            with self._lock:
                self._running = False
                self._error = "Could not open laptop webcam"
                self.last_result = CameraResult(False, None, time.strftime("%H:%M:%S"), self._error, self._detector_name)
            return
        started = time.monotonic()
        interval = 1.0 / self.target_fps
        next_frame = started
        try:
            while time.monotonic() - started < duration:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.03)
                    continue
                annotated, detections = self._detect(frame)
                now = time.monotonic()
                with self._lock:
                    self._frames_processed += 1
                    if detections:
                        self._detected_any = True
                        confidence = max(d[4] for d in detections)
                        self._best_confidence = max(self._best_confidence or 0.0, confidence)
                        self._positive_until = now + 2.0
                    if self._detected_any and now <= self._positive_until and not detections:
                        cv2.putText(annotated, "PERSON DETECTED (latched)", (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    self._latest_frame = annotated
                next_frame += interval
                time.sleep(max(0.0, next_frame - time.monotonic()))
        finally:
            cap.release()
        with self._lock:
            self._running = False
            self.last_result = CameraResult(
                self._detected_any,
                self._latest_frame,
                time.strftime("%H:%M:%S"),
                None if self._frames_processed else "Webcam returned no valid frames",
                self._detector_name,
                self._best_confidence,
                self._frames_processed,
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            latched = self._detected_any or time.monotonic() <= self._positive_until
            return {
                "running": self._running,
                "frame": None if self._latest_frame is None else self._latest_frame.copy(),
                "person_detected": latched,
                "frames_processed": self._frames_processed,
                "confidence": self._best_confidence,
                "detector": self._detector_name,
                "error": self._error,
                "result": self.last_result,
            }

    def should_auto_verify(self, state: str) -> bool:
        trigger = state in {"WATCH", "CONFIRMED"} and self.last_trigger_state not in {"WATCH", "CONFIRMED"}
        # Escalating WATCH -> CONFIRMED also deserves an immediate new window.
        trigger = trigger or (state == "CONFIRMED" and self.last_trigger_state != "CONFIRMED")
        self.last_trigger_state = state
        return trigger and not self.running

    def verify(self, duration: float | None = None) -> CameraResult:
        """Synchronous compatibility wrapper used by tests."""
        self.start(duration)
        worker = self._worker
        if worker:
            worker.join((duration or self.duration) + 3.0)
        return self.last_result or CameraResult(False, None, time.strftime("%H:%M:%S"), "Verification did not finish", self._detector_name)
