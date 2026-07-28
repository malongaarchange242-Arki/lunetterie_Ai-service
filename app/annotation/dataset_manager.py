import shutil
from pathlib import Path
from typing import Dict

from app.annotation.labels import COULEURS, FORMES, GENRES, MATERIAUX, TYPE_MONTURE

ROOT = Path("classification_dataset")


def create_structure(root_dir: Path | None = None) -> None:
    target_root = root_dir or ROOT
    datasets = {
        "forme": FORMES,
        "couleur": COULEURS,
        "materiau": MATERIAUX,
        "type_monture": TYPE_MONTURE,
        "genre": GENRES,
    }

    for dataset_name, classes in datasets.items():
        for split in ["train", "valid", "test"]:
            for classe in classes:
                path = target_root / dataset_name / split / classe
                path.mkdir(parents=True, exist_ok=True)

    (target_root / "crops").mkdir(parents=True, exist_ok=True)
    print("Structure créée avec succès.")


def copy_to_class_folders(image_path: Path, labels: Dict[str, str], root_dir: Path | None = None) -> None:
    target_root = root_dir or ROOT
    create_structure(target_root)

    mapping = {
        "forme": "forme",
        "couleur": "couleur",
        "materiau": "materiau",
        "type": "type_monture",
        "genre": "genre",
    }

    for source_key, target_folder in mapping.items():
        label_value = labels.get(source_key)
        if not label_value:
            continue
        destination_dir = target_root / target_folder / "train" / label_value
        destination_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, destination_dir / image_path.name)
