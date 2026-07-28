from typing import Any

from PIL import Image

from app.classification.auto_color import estimate_color
from app.classification.auto_mount import estimate_mount_type


class GlassesClassifier:
    def __init__(self) -> None:
        self.frame_shapes = ["round", "square", "oval", "rectangle", "unknown"]
        self.colors = ["black", "brown", "blue", "red", "green", "gold", "silver", "unknown"]
        self.materials = ["metal", "plastic", "titanium", "acetate", "unknown"]

    def classify(self, image_path: str) -> dict[str, Any]:
        with Image.open(image_path).convert("RGB") as image:
            pixels = image.load()
            width, height = image.size
            sample = [pixels[x, y] for y in range(0, height, max(1, height // 4)) for x in range(0, width, max(1, width // 4))]
            dominant_color = self._dominant_color(sample)
            aspect_ratio = width / max(height, 1)
            frame_shape = self._infer_frame_shape(aspect_ratio)

        color_label = estimate_color(image_path)
        mount_type = estimate_mount_type(image_path)

        return {
            "frame_shape": frame_shape,
            "color": color_label,
            "material": "unknown",
            "has_branches": True,
            "mount_type": mount_type,
        }

    def _infer_frame_shape(self, aspect_ratio: float) -> str:
        if aspect_ratio > 1.2:
            return "rectangle"
        if aspect_ratio > 0.9:
            return "round"
        return "oval"

    def _dominant_color(self, pixels: list[tuple[int, int, int]]) -> str:
        if not pixels:
            return "unknown"
        r = sum(p[0] for p in pixels) // len(pixels)
        g = sum(p[1] for p in pixels) // len(pixels)
        b = sum(p[2] for p in pixels) // len(pixels)
        if r < 80 and g < 80 and b < 80:
            return "black"
        if r > 180 and g > 180 and b > 180:
            return "white"
        if r < 140 and g < 140 and b < 140:
            return "black"
        if abs(r - g) < 25 and abs(g - b) < 25:
            return "gray"
        return "unknown"
