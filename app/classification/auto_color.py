from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image


def estimate_color(image_path: str) -> str:
    """Estimate a coarse color label from the dominant pixels in the crop."""
    image = Image.open(image_path).convert("RGB")
    pixels = np.array(image).reshape(-1, 3)

    if len(pixels) == 0:
        return "Inconnu"

    r = pixels[:, 0].mean()
    g = pixels[:, 1].mean()
    b = pixels[:, 2].mean()

    if r < 80 and g < 80 and b < 80:
        return "Noir"
    if r > 200 and g > 200 and b > 200:
        return "Blanc"
    if max(r, g, b) - min(r, g, b) < 40:
        return "Gris"
    if r > g and r > b:
        return "Rouge"
    if g > r and g > b:
        return "Vert"
    if b > r and b > g:
        return "Bleu"
    return "Brun"
