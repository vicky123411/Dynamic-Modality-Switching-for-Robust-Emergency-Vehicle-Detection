from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


def detect_context(image_path: Path) -> dict:
    img = cv2.imread(str(image_path))
    if img is None:
        return {"lighting": "unknown", "weather": "unknown", "traffic_density": "low", "blur": 0.0}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    lighting = "night" if brightness < 100 else "day"

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap_var < 60:
        weather = "fog"
    elif lap_var < 120:
        weather = "rain"
    else:
        weather = "clear"

    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = len(contours)
    if count > 300:
        traffic = "high"
    elif count > 120:
        traffic = "medium"
    else:
        traffic = "low"

    return {
        "lighting": lighting,
        "weather": weather,
        "traffic_density": traffic,
        "blur": lap_var,
        "brightness": brightness,
    }
