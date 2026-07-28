from pathlib import Path

from PIL import Image

from app.ai.auto_mount import MountAnnotator


def test_full_rim_is_detected_from_dark_border(tmp_path: Path) -> None:
    image_path = tmp_path / "mount.png"
    image = Image.new("RGB", (200, 200), color="white")
    image = image.convert("RGB")
    for x in range(20, 180):
        for y in range(20, 180):
            if x < 30 or x > 170 or y < 30 or y > 170:
                image.putpixel((x, y), (0, 0, 0))
    image.save(image_path)

    annotator = MountAnnotator()
    result = annotator.process_image(image_path)

    assert result["mount_type"] == "Pleine"
    assert result["confidence"] >= 0.8
