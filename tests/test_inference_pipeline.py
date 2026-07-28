import numpy as np

from app.inference.pipeline import _extract_glasses_crop


def test_extract_glasses_crop_returns_detected_region_not_full_image() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[:, :] = [255, 255, 255]
    image[30:70, 60:140] = [0, 128, 0]
    boxes = [{"confidence": 0.95, "bbox": [60, 30, 140, 70]}]

    rgb_crop, bgr_crop, best_box = _extract_glasses_crop(image, boxes)

    assert best_box == boxes[0]
    assert rgb_crop.shape[0] < image.shape[0]
    assert rgb_crop.shape[1] < image.shape[1]
    assert bgr_crop.shape[:2] == rgb_crop.shape[:2]
