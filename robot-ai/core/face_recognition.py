"""
face_recognition.py — Real-time face detection + recognition for Bento Robot.

Two modes:
  DETECT  → Find any face (using OpenCV DNN or YOLOv8-face)
  RECOGNIZE → Match face against a known-people database

Use cases for the robot:
  • Identify known operators → auto-start/unlock
  • Detect unknown persons → alert security
  • Count passengers (for delivery robots)
  • Wave / acknowledge a recognised face

Database: a folder of labelled face images (one subfolder per person).
  known_faces/
    yash/    face1.jpg  face2.jpg
    saras/   face1.jpg

Install:  pip install deepface opencv-python
"""

import os
import cv2
import numpy as np
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import time


@dataclass
class FaceResult:
    """One detected / recognised face."""
    bbox:        Tuple[int, int, int, int] = (0, 0, 0, 0)  # x1,y1,x2,y2
    confidence:  float  = 0.0
    name:        str    = "unknown"
    is_known:    bool   = False
    emotion:     str    = ""    # optional — deepface emotion


@dataclass
class FaceFrame:
    """All faces found in one frame."""
    faces:        List[FaceResult] = field(default_factory=list)
    known_count:  int   = 0
    unknown_count: int  = 0
    timestamp:    float = 0.0

    @property
    def any_face(self) -> bool:
        return len(self.faces) > 0

    @property
    def known_names(self) -> List[str]:
        return [f.name for f in self.faces if f.is_known]


class FaceSystem:
    """
    Real-time face detection + optional recognition.

    Usage:
        fs = FaceSystem(db_path="known_faces/", mode="recognize")
        # In your camera loop:
        result = fs.process(frame)
        if result.any_face:
            print(result.known_names)
        annotated = fs.draw(frame, result)
    """

    # OpenCV DNN face detector model URLs
    DNN_PROTOTXT = "third_party/face_detector/deploy.prototxt"
    DNN_MODEL    = "third_party/face_detector/res10_300x300_ssd_iter_140000.caffemodel"

    def __init__(
        self,
        db_path: str  = "known_faces/",
        mode:    str  = "detect",       # "detect" or "recognize"
        threshold: float = 0.55,        # face detection confidence threshold
    ):
        self.mode      = mode
        self.db_path   = db_path
        self.threshold = threshold
        self._deepface = None
        self._net      = None

        # Load OpenCV DNN face detector
        self._load_opencv_detector()

        # Load deepface if recognition mode
        if mode == "recognize":
            self._load_deepface()

    def _load_opencv_detector(self):
        """Load OpenCV's DNN-based face detector (fast, no GPU needed)."""
        # Download model files if they don't exist
        os.makedirs("third_party/face_detector", exist_ok=True)

        proto = self.DNN_PROTOTXT
        model = self.DNN_MODEL

        if not os.path.exists(proto) or not os.path.exists(model):
            print("📥 Downloading face detector model...")
            import urllib.request
            base_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/"
            try:
                urllib.request.urlretrieve(base_url + "deploy.prototxt", proto)
                urllib.request.urlretrieve(
                    "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/"
                    "res10_300x300_ssd_iter_140000.caffemodel",
                    model
                )
                print("✅ Face detector downloaded")
            except Exception as e:
                print(f"⚠️  Could not download face model: {e}")
                print("   Using Haar cascade fallback")
                self._net = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
                return

        self._net = cv2.dnn.readNetFromCaffe(proto, model)
        print("✅ Face detector loaded (OpenCV DNN ResNet-SSD)")

    def _load_deepface(self):
        """Load deepface for face recognition."""
        try:
            import deepface
            from deepface import DeepFace
            self._deepface = DeepFace
            print(f"✅ DeepFace loaded — database: {self.db_path}")
        except ImportError:
            print("⚠️  deepface not installed — pip install deepface")
            print("   Falling back to detect-only mode")
            self.mode = "detect"

    def process(self, frame: np.ndarray) -> FaceFrame:
        """
        Detect (and optionally recognise) faces in a BGR frame.
        Returns a FaceFrame with all results.
        """
        result = FaceFrame(timestamp=time.time())
        h, w   = frame.shape[:2]

        # ── 1. Detect face bounding boxes ──────────────────────────
        raw_faces = self._detect_faces(frame, w, h)

        for (x1, y1, x2, y2, conf) in raw_faces:
            face = FaceResult(bbox=(x1, y1, x2, y2), confidence=conf)

            # ── 2. Optional: Recognise ──────────────────────────────
            if self.mode == "recognize" and self._deepface is not None:
                face_crop = frame[max(0,y1):y2, max(0,x1):x2]
                name, is_known = self._recognize_face(face_crop)
                face.name     = name
                face.is_known = is_known

            result.faces.append(face)

        result.known_count   = sum(1 for f in result.faces if f.is_known)
        result.unknown_count = len(result.faces) - result.known_count
        return result

    def _detect_faces(self, frame, w, h):
        """Returns list of (x1,y1,x2,y2,confidence)."""
        faces = []

        # DNN detector
        if isinstance(self._net, cv2.dnn_Net):
            blob    = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 117, 123))
            self._net.setInput(blob)
            detections = self._net.forward()

            for i in range(detections.shape[2]):
                conf = float(detections[0, 0, i, 2])
                if conf < self.threshold:
                    continue
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                faces.append((x1, y1, x2, y2, conf))

        # Haar cascade fallback
        elif isinstance(self._net, cv2.CascadeClassifier):
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self._net.detectMultiScale(gray, 1.1, 5)
            for (x, y, fw, fh) in rects:
                faces.append((x, y, x + fw, y + fh, 0.9))

        return faces

    def _recognize_face(self, face_crop: np.ndarray):
        """Match a face crop against the known_faces/ database."""
        if face_crop.size == 0 or not os.path.exists(self.db_path):
            return "unknown", False
        try:
            results = self._deepface.find(
                img_path    = face_crop,
                db_path     = self.db_path,
                model_name  = "Facenet512",
                enforce_detection = False,
                silent      = True,
            )
            if results and len(results[0]) > 0:
                best   = results[0].iloc[0]
                # Extract person name from path
                person = os.path.basename(os.path.dirname(best["identity"]))
                return person, True
        except Exception:
            pass
        return "unknown", False

    def draw(self, frame: np.ndarray, result: FaceFrame) -> np.ndarray:
        """Draw bounding boxes and labels on the frame."""
        out = frame.copy()
        for face in result.faces:
            x1, y1, x2, y2 = face.bbox
            color = (0, 200, 80) if face.is_known else (0, 80, 220)
            label = f"{face.name} ({face.confidence:.2f})"

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, label, (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Summary
        summary = f"Faces: {len(result.faces)} | Known: {result.known_count}"
        cv2.putText(out, summary, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return out

    def add_person(self, name: str, image_path: str):
        """Add a new person to the known_faces database."""
        person_dir = os.path.join(self.db_path, name)
        os.makedirs(person_dir, exist_ok=True)
        import shutil
        dst = os.path.join(person_dir, os.path.basename(image_path))
        shutil.copy2(image_path, dst)
        print(f"✅ Added {name} → {dst}")
