from pathlib import Path

import cv2
import numpy as np

from app.ai.auto_color import ColorAnnotator
from app.inference.auto_attributes import build_frame_mask, get_color


def test_black_image_is_labeled_black(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(image_path), image)

    annotator = ColorAnnotator()
    result = annotator.process_image(image_path)

    assert result["color"] == "Noir"
    assert result["confidence"] >= 0.8


def test_get_color_ignores_white_background_and_focuses_on_object() -> None:
    image = np.full((120, 120, 3), 255, dtype=np.uint8)
    image[40:80, 40:80] = [20, 20, 20]

    color, confidence = get_color(image)

    assert color == "black"
    assert confidence >= 0.5


def test_get_color_uses_frame_mask_for_colored_mounture() -> None:
    image = np.full((160, 220, 3), 255, dtype=np.uint8)
    image[55:65, 35:185] = [30, 140, 65]
    image[95:105, 35:185] = [30, 140, 65]
    image[55:105, 35:45] = [30, 140, 65]
    image[55:105, 175:185] = [30, 140, 65]

    mask = build_frame_mask(image)
    color, confidence = get_color(image, mask)

    assert color == "green"
    assert confidence >= 0.5
