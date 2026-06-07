import numpy as np
import cv2
from typing import Optional, Tuple
from PIL import Image

def estimate_horizon_angle(gray: np.ndarray) -> Optional[float]:
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)
    if lines is None:
        return None
    angles, weights = [], []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = np.arctan2(y2 - y1, x2 - x1)
        deg = np.rad2deg(angle)
        if -20 <= deg <= 20:
            length = float(np.hypot(x2 - x1, y2 - y1))
            angles.append(angle)
            weights.append(length)
    if not angles:
        return None
    return float(np.rad2deg(np.average(angles, weights=weights)))

def compute_image_quality_metrics(np_img: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    brightness = float(np.mean(np_img))
    median_brightness = float(np.median(np_img))
    dark_fraction = float(np.sum(np_img < 80) / np_img.size)
    contrast = float(np.std(np_img))
    sharpness = float(cv2.Laplacian(np_img, cv2.CV_64F).var())
    blurred = cv2.medianBlur(np_img, 3)
    noise = float(np.mean(np.abs(np_img.astype(np.float32) - blurred.astype(np.float32))))
    return brightness, median_brightness, dark_fraction, contrast, sharpness, noise

def resize_to_smallest(img1_pil: Image.Image, img2_pil: Image.Image):
    w1, h1 = img1_pil.size
    w2, h2 = img2_pil.size
    a1, a2 = w1 * h1, w2 * h2
    target = (w1, h1) if a1 <= a2 else (w2, h2)
    return img1_pil.resize(target), img2_pil.resize(target), target
