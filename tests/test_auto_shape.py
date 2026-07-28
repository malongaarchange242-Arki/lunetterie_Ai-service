import cv2
import numpy as np

from app.ai.auto_shape import ShapeAnnotator


def test_circle_is_classified_as_round() -> None:
    annotator = ShapeAnnotator()
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    cv2.circle(image, (60, 60), 35, (255, 255, 255), thickness=-1)

    contour = annotator.extract_contour(image)
    assert contour is not None

    features = annotator.compute_shape_features(contour)
    shape_name, confidence = annotator.classify_shape(features)

    assert shape_name == "Ronde"
    assert confidence >= 0.8
