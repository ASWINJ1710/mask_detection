"""Stage 1: face detection with an OpenCV Haar Cascade.

The rest of the project only calls FaceDetector.detect(), so this class can
later be replaced (for example by YuNet) without changing other code.

Demo from the project root:
    python -m src.face_detector              (webcam, press q to quit)
    python -m src.face_detector --image path/to/photo.jpg
"""
import argparse

import cv2


class FaceDetector:
    def __init__(self, scale_factor=1.1, min_neighbors=5, min_size=(60, 60)):
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(path)
        if self._cascade.empty():
            raise RuntimeError(f"Could not load Haar cascade: {path}")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, frame_bgr):
        """Return a list of (x, y, w, h) integer boxes for faces in a BGR frame."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def _draw(frame, boxes):
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
    cv2.putText(frame, f"faces: {len(boxes)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="test on one image instead of the webcam")
    args = ap.parse_args()
    detector = FaceDetector()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"Could not read image: {args.image}")
        boxes = detector.detect(frame)
        print(f"Faces found: {len(boxes)}  boxes: {boxes}")
        cv2.imshow("face_detector", _draw(frame, boxes))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        boxes = detector.detect(frame)
        cv2.imshow("face_detector (q to quit)", _draw(frame, boxes))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
