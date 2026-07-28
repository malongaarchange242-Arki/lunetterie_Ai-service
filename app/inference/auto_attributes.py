from __future__ import annotations

import cv2
import numpy as np


COLOR_REFERENCE = {
    "black": np.array([20, 20, 20], dtype=np.float32),
    "gray": np.array([128, 128, 128], dtype=np.float32),
    "white": np.array([235, 235, 235], dtype=np.float32),
    "brown": np.array([105, 65, 35], dtype=np.float32),
    "green": np.array([45, 130, 70], dtype=np.float32),
    "blue": np.array([45, 90, 170], dtype=np.float32),
    "red": np.array([170, 45, 45], dtype=np.float32),
    "gold": np.array([205, 165, 55], dtype=np.float32),
    "silver": np.array([190, 190, 185], dtype=np.float32),
}


def build_frame_mask(image: np.ndarray) -> np.ndarray:
    if image is None:
        return np.zeros((0, 0), dtype=bool)

    img = np.array(image)
    if img.size == 0:
        return np.zeros((0, 0), dtype=bool)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    rgb = img[:, :, :3]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    non_background = (value < 235) & ((saturation > 25) | (gray < 205))
    very_dark = gray < 95
    edges = cv2.Canny(gray, 40, 120) > 0
    mask = non_background | very_dark | edges

    kernel = np.ones((3, 3), np.uint8)
    mask_uint8 = (mask.astype(np.uint8) * 255)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask_uint8 > 0


def _dominant_rgb_from_mask(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray | None, float]:
    if mask.shape != rgb.shape[:2]:
        mask = np.zeros(rgb.shape[:2], dtype=bool)

    pixels = rgb[mask]
    coverage = float(len(pixels)) / float(max(1, rgb.shape[0] * rgb.shape[1]))
    if len(pixels) < 20:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        fallback = gray < 230
        pixels = rgb[fallback]
        coverage = float(len(pixels)) / float(max(1, rgb.shape[0] * rgb.shape[1]))

    if len(pixels) == 0:
        return None, 0.0

    if len(pixels) > 5000:
        step = max(1, len(pixels) // 5000)
        pixels = pixels[::step]

    pixels_float = np.float32(pixels)
    clusters = min(3, len(pixels))
    if clusters <= 1:
        return pixels_float.mean(axis=0), coverage

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _compactness, labels, centers = cv2.kmeans(
        pixels_float,
        clusters,
        None,
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS,
    )
    counts = np.bincount(labels.flatten(), minlength=clusters)
    dominant = centers[int(np.argmax(counts))]
    return dominant, coverage


def _nearest_color(rgb_color: np.ndarray) -> tuple[str, float]:
    names = list(COLOR_REFERENCE)
    references = np.array([COLOR_REFERENCE[name] for name in names], dtype=np.float32)
    distances = np.linalg.norm(references - rgb_color.astype(np.float32), axis=1)
    best_idx = int(np.argmin(distances))
    confidence = max(0.35, 1.0 - float(distances[best_idx]) / 255.0)
    return names[best_idx], confidence


def get_color(image: np.ndarray, mask: np.ndarray | None = None) -> tuple[str, float]:
    if image is None:
        return "unknown", 0.0

    if isinstance(image, np.ndarray):
        img = image
    else:
        img = np.array(image)

    if img.size == 0:
        return "unknown", 0.0

    if img.ndim == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = img[:, :, :3]

    frame_mask = mask if mask is not None else build_frame_mask(rgb)
    dominant_rgb, coverage = _dominant_rgb_from_mask(rgb, frame_mask)
    if dominant_rgb is not None:
        color_name, confidence = _nearest_color(dominant_rgb)
        coverage_factor = min(1.0, max(0.55, coverage * 8.0))
        return color_name, confidence * coverage_factor

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mean_value = float(gray.mean())
    if mean_value < 80:
        return "black", 0.9
    if mean_value > 200:
        return "white", 0.9
    return "gray", 0.7


def get_material(image: np.ndarray) -> tuple[str, float]:
    if image is None:
        return "unknown", 0.0
    return "plastic", 0.65


def get_mount_type(image: np.ndarray, mask: np.ndarray | None = None) -> tuple[str, float]:
    if image is None:
        return "unknown", 0.0

    if isinstance(image, np.ndarray):
        img = image
    else:
        img = np.array(image)

    if img.ndim == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if mask is not None and mask.shape == gray.shape and np.count_nonzero(mask) > 0:
        dark_pixels = np.count_nonzero(mask)
    else:
        dark_pixels = np.count_nonzero(gray < 80)
    total_pixels = gray.size
    dark_ratio = dark_pixels / total_pixels

    if dark_ratio > 0.12:
        return "Pleine", 0.90
    if dark_ratio > 0.05:
        return "Semi-cerclée", 0.80
    return "Percée", 0.75
