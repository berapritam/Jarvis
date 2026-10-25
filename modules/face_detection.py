"""
face_detection.py
------------------
Lightweight face detection using OpenCV's built-in Haar cascade (no external
model download needed, works offline). Accepts a base64-encoded image (e.g.
a webcam frame posted from the dashboard) and returns bounding boxes.

Install:
    pip install opencv-python-headless numpy

Wire into Flask (ollama_service2.py):

    from modules.face_detection import detect_faces_b64

    @app.post("/api/vision/detect-faces")
    def api_detect_faces():
        payload = request.get_json(silent=True) or {}
        image_b64 = payload.get("image_base64", "")
        if not image_b64:
            return jsonify({"error": "'image_base64' is required"}), 400
        result = detect_faces_b64(image_b64)
        return jsonify(result)
"""

from __future__ import annotations

import base64
import logging

import cv2
import numpy as np

log = logging.getLogger("sentinelai.vision")

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_faces_b64(image_base64: str, min_neighbors: int = 5) -> dict:
    """image_base64 may include a data: URL prefix; it is stripped automatically."""
    if "," in image_base64 and image_base64.strip().startswith("data:"):
        image_base64 = image_base64.split(",", 1)[1]

    try:
        raw = base64.b64decode(image_base64)
    except (ValueError, TypeError) as exc:
        return {"error": f"Invalid base64 image: {exc}"}

    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Could not decode image"}

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=min_neighbors, minSize=(40, 40)
    )

    boxes = [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)} for (x, y, w, h) in faces]
    log.info("Detected %d face(s)", len(boxes))
    return {
        "face_count": len(boxes),
        "faces": boxes,
        "image_size": {"width": frame.shape[1], "height": frame.shape[0]},
    }
