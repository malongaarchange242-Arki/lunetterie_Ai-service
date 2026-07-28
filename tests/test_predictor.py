from pathlib import Path

from PIL import Image

from app.ai.predictor import GlassesPredictor


def test_predictor_uses_square_shape_for_square_frame(tmp_path: Path) -> None:
    predictor = GlassesPredictor()
    image_path = tmp_path / "square_frame.png"

    image = Image.new("RGB", (300, 300), color=(255, 255, 255))
    for x in range(300):
        for y in range(300):
            if x in range(40, 80) or x in range(220, 260) or y in range(40, 80) or y in range(220, 260):
                image.putpixel((x, y), (0, 0, 0))
    image.save(image_path)

    result = predictor.predict_image(str(image_path))

    assert result["frame_shape"] == "square"


def test_predictor_returns_consistent_product_payload(tmp_path: Path) -> None:
    predictor = GlassesPredictor()
    image_path = tmp_path / "frame.png"

    image = Image.new("RGB", (400, 200), color=(255, 255, 255))
    image.save(image_path)

    result = predictor.predict_image(str(image_path))

    # detector may be a no-op when ultralytics isn't installed; accept either bool
    assert isinstance(result["detected"], bool)
    assert result["frame_shape"] in {"round", "square", "oval", "rectangle", "unknown"}
    assert result["color"] in {"black", "brown", "blue", "red", "green", "gold", "silver", "white", "gray", "unknown"}
    assert isinstance(result["confidence"], float)
    assert result["confidence"] >= 0.0
