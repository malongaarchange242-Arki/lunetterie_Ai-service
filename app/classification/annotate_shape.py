import csv
import shutil
import sys
from pathlib import Path
from typing import Dict, List

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.annotation.labels import FORMES

IMAGE_DIR = ROOT / "classification_dataset" / "crops"
TARGET_ROOT = ROOT / "classification_dataset"
CSV_PATH = TARGET_ROOT / "annotations_shape.csv"
LABELS_DIR = TARGET_ROOT / "forme"


def ensure_structure() -> None:
    for split in ["train", "valid", "test"]:
        for label in FORMES:
            (LABELS_DIR / split / label).mkdir(parents=True, exist_ok=True)


def collect_images() -> List[Path]:
    if not IMAGE_DIR.exists():
        return []
    return sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )


def write_csv_row(image_name: str, label: str) -> None:
    fieldnames = ["image", "forme"]
    file_exists = CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({"image": image_name, "forme": label})


def copy_image(image_path: Path, label: str) -> None:
    destination_dir = LABELS_DIR / "train" / label
    destination_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, destination_dir / image_path.name)


def prompt_label(image_path: Path) -> str:
    print(f"\nImage: {image_path.name}")
    print("Choisissez une forme :")
    for index, label in enumerate(FORMES, start=1):
        print(f"  {index}. {label}")
    print("  0. Passer")

    while True:
        answer = input("Votre choix : ").strip()
        if answer == "0":
            return ""
        if answer.isdigit():
            idx = int(answer) - 1
            if 0 <= idx < len(FORMES):
                return FORMES[idx]
        print("Choix invalide.")


def main() -> None:
    ensure_structure()
    images = collect_images()
    if not images:
        print("Aucune image à annoter dans le dossier crops.")
        return

    for image_path in images:
        label = prompt_label(image_path)
        if not label:
            print("Image ignorée.")
            continue
        copy_image(image_path, label)
        write_csv_row(image_path.name, label)
        print(f"Enregistré : {label}")

    print("Annotation terminée.")


if __name__ == "__main__":
    main()
