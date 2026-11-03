"""Stage 2: mask / no-mask classification of a cropped face.

Preprocessing must match training exactly:
  BGR -> RGB, resize to 224x224, float32 pixels in 0..255.
The saved model already scales pixels to -1..1 internally, so do NOT scale here.
The model output is the probability of "without_mask".

Sanity check on the test set (should be close to the training-time accuracy):
    python -m src.mask_classifier --check
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "face_mask_model.keras"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["with_mask", "without_mask"]


class MaskClassifier:
    def __init__(self, model_path=MODEL_PATH, threshold=0.5, margin=0.15):
        from tensorflow import keras

        self.model = keras.models.load_model(str(model_path))
        self.threshold = threshold
        self.margin = margin

    @staticmethod
    def _prepare(face_bgr):
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        return rgb.astype("float32")

    def crop(self, frame_bgr, box):
        """Crop a face box (x, y, w, h) with a margin, clamped to the frame."""
        x, y, w, h = box
        H, W = frame_bgr.shape[:2]
        mx, my = int(w * self.margin), int(h * self.margin)
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1, y1 = min(W, x + w + mx), min(H, y + h + my)
        return frame_bgr[y0:y1, x0:x1]

    def predict_crops(self, crops_bgr):
        """Return a list of (label, confidence) for a list of BGR face crops."""
        if not crops_bgr:
            return []
        batch = np.stack([self._prepare(c) for c in crops_bgr])
        probs = np.asarray(self.model(batch, training=False)).ravel()
        results = []
        for p in probs:
            p = float(p)
            if p >= self.threshold:
                results.append(("without_mask", p))
            else:
                results.append(("with_mask", 1.0 - p))
        return results

    def predict_faces(self, frame_bgr, boxes):
        """Return one (label, confidence) per box, aligned with the boxes list."""
        crops, index = [], []
        for i, box in enumerate(boxes):
            c = self.crop(frame_bgr, box)
            if c.size > 0:
                crops.append(c)
                index.append(i)
        preds = self.predict_crops(crops)
        out = [None] * len(boxes)
        for i, pr in zip(index, preds):
            out[i] = pr
        return out


def check_on_test_set():
    test_dir = ROOT / "dataset" / "test"
    clf = MaskClassifier()
    cm = np.zeros((2, 2), dtype=int)
    for true_idx, cls in enumerate(CLASS_NAMES):
        files = sorted(p for p in (test_dir / cls).iterdir() if p.is_file())
        for start in range(0, len(files), 64):
            imgs = []
            for p in files[start:start + 64]:
                im = cv2.imread(str(p))
                if im is not None:
                    imgs.append(im)
            for label, _ in clf.predict_crops(imgs):
                cm[true_idx, CLASS_NAMES.index(label)] += 1
    total = int(cm.sum())
    acc = (cm[0, 0] + cm[1, 1]) / total
    print(f"Images checked: {total}   Accuracy (live code path): {acc:.4f}")
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(cm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="run the test set through this module")
    args = ap.parse_args()
    if args.check:
        check_on_test_set()
    else:
        print("Use --check to verify against the test set.")


if __name__ == "__main__":
    main()
