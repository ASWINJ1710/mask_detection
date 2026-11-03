"""Video processing for the Streamlit-WebRTC dashboard.

recv() runs in a separate thread, so it must not call any st.* function.
Stats are kept behind a lock and read from the main script via get_stats().
"""
import threading
import time

import av
import cv2
from streamlit_webrtc import VideoProcessorBase

from src.face_detector import FaceDetector
from src.mask_classifier import MaskClassifier

GREEN = (0, 200, 0)  # BGR
RED = (0, 0, 255)  # BGR
FONT = cv2.FONT_HERSHEY_SIMPLEX

_models_lock = threading.Lock()
_models = {}


def get_models():
    """Load the detector and classifier once and share them (single local user)."""
    with _models_lock:
        if not _models:
            _models["detector"] = FaceDetector()
            _models["classifier"] = MaskClassifier()
        return _models["detector"], _models["classifier"]


class MaskVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector, self.classifier = get_models()
        self.threshold = 0.5
        self._lock = threading.Lock()
        self._stats = {"faces": 0, "with_mask": 0, "without_mask": 0,
                       "fps": 0.0, "frames": 0}
        self._last_t = time.time()
        self._fps = 0.0

    def set_threshold(self, value):
        self.threshold = float(value)

    def get_stats(self):
        with self._lock:
            return dict(self._stats)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        boxes = self.detector.detect(img)
        self.classifier.threshold = self.threshold
        preds = self.classifier.predict_faces(img, boxes)

        with_mask = without_mask = 0
        for (x, y, w, h), pred in zip(boxes, preds):
            if pred is None:
                continue
            label, conf = pred
            if label == "with_mask":
                with_mask += 1
                color, text = GREEN, f"Mask {conf * 100:.1f}%"
            else:
                without_mask += 1
                color, text = RED, f"No Mask {conf * 100:.1f}%"
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            (tw, th), base = cv2.getTextSize(text, FONT, 0.6, 2)
            top = max(0, y - th - base - 6)
            cv2.rectangle(img, (x, top), (x + tw + 6, top + th + base + 6), color, -1)
            cv2.putText(img, text, (x + 3, top + th + 3), FONT, 0.6,
                        (255, 255, 255), 2)

        now = time.time()
        dt = now - self._last_t
        self._last_t = now
        if dt > 0:
            inst = 1.0 / dt
            self._fps = inst if self._fps == 0 else 0.9 * self._fps + 0.1 * inst

        with self._lock:
            self._stats["faces"] = with_mask + without_mask
            self._stats["with_mask"] = with_mask
            self._stats["without_mask"] = without_mask
            self._stats["fps"] = round(self._fps, 1)
            self._stats["frames"] += 1

        return av.VideoFrame.from_ndarray(img, format="bgr24")
