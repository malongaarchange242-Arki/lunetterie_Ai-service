from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Any


class ShapeEstimator:
    """Estimate a coarse frame shape from an image using a heuristic."""

    def __init__(self) -> None:
        self.labels = ["round", "square", "rectangle", "oval", "aviator", "butterfly"]

    def estimate(self, image_path: str | Path) -> dict[str, Any]:
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            return {"shape": "unknown", "confidence": 0.0, "reason": "image_not_found"}

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        _, mask = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        if np.count_nonzero(mask) < 200:
            _, mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        dark_pixels = int(np.count_nonzero(mask > 0))
        total_pixels = mask.shape[0] * mask.shape[1]

        if dark_pixels < 200:
            return {"shape": "unknown", "confidence": 0.0, "reason": "insufficient_structure"}

        ys, xs = np.where(mask > 0)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())

        width = max(1, x1 - x0 + 1)
        height = max(1, y1 - y0 + 1)
        aspect_ratio = width / height
        occupancy = dark_pixels / total_pixels
        fill_ratio = dark_pixels / (width * height)

        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        image_cx = mask.shape[1] / 2.0
        image_cy = mask.shape[0] / 2.0
        horizontal_shift = abs(cx - image_cx) / max(1.0, image_cx)
        vertical_shift = abs(cy - image_cy) / max(1.0, image_cy)

        score = {label: 0.0 for label in self.labels}

        if 0.85 <= aspect_ratio <= 1.15 and fill_ratio > 0.35:
            score["round"] += 0.35
        if 0.95 <= aspect_ratio <= 1.35 and fill_ratio > 0.22:
            score["oval"] += 0.25

        if 0.90 <= aspect_ratio <= 1.10 and occupancy < 0.30:
            score["square"] += 0.45
        if 0.90 <= aspect_ratio <= 1.10:
            score["square"] += 0.45
        if 0.90 <= aspect_ratio <= 1.10 and fill_ratio > 0.18:
            score["square"] += 0.10

        if aspect_ratio > 1.25:
            score["rectangle"] += 0.55
        if aspect_ratio > 1.50:
            score["aviator"] += 0.20

        if aspect_ratio > 1.8:
            score["aviator"] += 0.60

        if 0.65 <= aspect_ratio <= 1.05 and occupancy > 0.18:
            score["butterfly"] += 0.35
        if horizontal_shift > 0.08 or vertical_shift > 0.08:
            score["butterfly"] += 0.10

        if fill_ratio < 0.15:
            score["butterfly"] += 0.10

        best_label = max(score, key=score.get)
        confidence = round(min(max(score.values()), 0.95), 2)

        return {
            "shape": best_label,
            "confidence": confidence,
            "aspect_ratio": round(aspect_ratio, 3),
            "occupancy": round(occupancy, 3),
            "fill_ratio": round(fill_ratio, 3),
        }
