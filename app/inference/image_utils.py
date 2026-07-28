from pathlib import Path
from typing import Any

from PIL import Image


def crop_image(image_path: str, bbox: list[float]) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return image.crop((x1, y1, x2, y2))


def resize_image(image: Image.Image, size: tuple[int, int] = (224, 224)) -> Image.Image:
    return image.resize(size)


def save_image(image: Image.Image, output_path: str) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output
