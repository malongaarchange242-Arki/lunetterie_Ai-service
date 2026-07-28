from pathlib import Path

from PIL import Image

from app.ai.classifier import GlassesClassifier


def test_classifier_uses_image_content_for_shape_and_color(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    image = Image.new("RGB", (200, 100), color="white")
    image.paste((0, 0, 0), (40, 20, 160, 80))
    image.save(image_path)

    result = GlassesClassifier().classify(str(image_path))

    assert result["frame_shape"] == "rectangle"
    assert result["color"] == "black"
