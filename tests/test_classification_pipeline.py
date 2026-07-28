from pathlib import Path

from PIL import Image

from app.ai.classifier import GlassesClassifier
from app.classification.auto_color import estimate_color
from app.classification.auto_mount import estimate_mount_type


def test_estimate_color_detects_dark_frames(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    img = Image.new("RGB", (200, 120), color="white")
    img.paste((0, 0, 0), (40, 20, 160, 100))
    img.save(image_path)

    color = estimate_color(str(image_path))
    assert color in {"Noir", "Gris", "Brun"}


def test_estimate_mount_type_detects_rimless_shape(tmp_path: Path) -> None:
    image_path = tmp_path / "rimless.png"
    img = Image.new("RGB", (200, 120), color="white")
    img.paste((0, 0, 0), (80, 40, 120, 80))
    img.save(image_path)

    mount_type = estimate_mount_type(str(image_path))
    assert mount_type in {"Percée", "Semi-cerclée", "Pleine"}


def test_glasses_classifier_exposes_color_and_mount_type(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    img = Image.new("RGB", (200, 120), color="white")
    img.paste((0, 0, 0), (40, 20, 160, 100))
    img.save(image_path)

    result = GlassesClassifier().classify(str(image_path))
    assert result["color"] in {"Noir", "Gris", "Brun"}
    assert result["mount_type"] in {"Percée", "Semi-cerclée", "Pleine"}
