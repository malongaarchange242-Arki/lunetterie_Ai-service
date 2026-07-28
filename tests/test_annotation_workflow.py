from pathlib import Path

from PIL import Image

from app.annotation.csv_manager import CSVManager
from app.annotation.dataset_manager import copy_to_class_folders


def test_copy_to_class_folders_and_csv(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (10, 10), color="red").save(image_path)

    root_dir = tmp_path / "classification_dataset"
    csv_path = tmp_path / "annotations.csv"

    labels = {
        "forme": "ronde",
        "couleur": "noir",
        "materiau": "plastique",
        "type": "cerclée",
        "genre": "homme",
    }

    copy_to_class_folders(image_path, labels, root_dir=root_dir)

    csv_manager = CSVManager(csv_path)
    csv_manager.save_annotation(image_path.name, labels)

    assert (root_dir / "forme" / "train" / "ronde" / image_path.name).exists()
    assert (root_dir / "couleur" / "train" / "noir" / image_path.name).exists()
    assert (root_dir / "materiau" / "train" / "plastique" / image_path.name).exists()
    assert (root_dir / "type_monture" / "train" / "cerclée" / image_path.name).exists()
    assert (root_dir / "genre" / "train" / "homme" / image_path.name).exists()

    rows = csv_manager.read_rows()
    assert rows[0]["image"] == image_path.name
    assert rows[0]["forme"] == "ronde"
    assert rows[0]["genre"] == "homme"
