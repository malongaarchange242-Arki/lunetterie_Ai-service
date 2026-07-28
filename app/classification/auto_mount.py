from pathlib import Path

import numpy as np
from PIL import Image


def estimate_mount_type(image_path: str) -> str:
    """Estimate a coarse mount-type category from the crop geometry."""
    image = Image.open(image_path).convert("L")
    arr = np.array(image)

    if arr.size == 0:
        return "Inconnu"

    edge_pixels = np.count_nonzero(arr < 80)
    total_pixels = arr.size
    dark_ratio = edge_pixels / total_pixels

    if dark_ratio > 0.25:
        return "Pleine"
    if dark_ratio > 0.12:
        return "Semi-cerclée"
    return "Percée"
