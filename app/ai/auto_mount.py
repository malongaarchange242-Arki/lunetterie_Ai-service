from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class MountAnnotator:
    """Annotate image crops with a coarse mount-type using simple image heuristics."""

    def __init__(self, crop_dir: str = "classification_dataset/crops", output_file: str = "classification_dataset/annotations/annotations_mount.csv") -> None:
        self.crop_dir = Path(crop_dir)
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

    def process_image(self, image_path: Path) -> dict[str, object]:
        image = cv2.imread(str(image_path))
        if image is None:
            return {"error": "Impossible de lire l'image"}

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if gray.size == 0:
            return {"error": "Image vide"}

        dark_pixels = np.count_nonzero(gray < 80)
        total_pixels = gray.size
        dark_ratio = dark_pixels / total_pixels

        if dark_ratio > 0.12:
            mount_type = "Pleine"
            confidence = 0.90
        elif dark_ratio > 0.05:
            mount_type = "Semi-cerclée"
            confidence = 0.80
        else:
            mount_type = "Percée"
            confidence = 0.75

        return {
            "filename": image_path.name,
            "mount_type": mount_type,
            "confidence": confidence,
            "dark_ratio": float(dark_ratio),
        }

    def process_all_images(self, max_images: Optional[int] = None) -> list[dict[str, object]]:
        image_files = sorted(list(self.crop_dir.glob("*.jpg")) + list(self.crop_dir.glob("*.png")))
        if max_images:
            image_files = image_files[:max_images]

        results: list[dict[str, object]] = []
        errors: list[tuple[str, str]] = []
        for image_path in image_files:
            result = self.process_image(image_path)
            if "error" in result:
                errors.append((image_path.name, str(result["error"])))
            else:
                results.append(result)

        if results:
            print("\n📊 Statistiques des types de monture:")
            counts: dict[str, int] = {}
            for record in results:
                mount_type = str(record["mount_type"])
                counts[mount_type] = counts.get(mount_type, 0) + 1
            for mount_type, count in counts.items():
                print(f"  {mount_type}: {count} images ({count / len(results) * 100:.1f}%)")
        if errors:
            print(f"\n⚠️ {len(errors)} images en erreur:")
            for filename, error in errors[:5]:
                print(f"  {filename}: {error}")
        return results

    def save_annotations(self, records: list[dict[str, object]]) -> None:
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["filename", "mount_type", "confidence", "dark_ratio"]
        with self.output_file.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                row = {key: record.get(key, "") for key in fieldnames}
                writer.writerow(row)
        print(f"\n✅ Annotations sauvegardées dans {self.output_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop_dir", type=str, default="classification_dataset/crops", help="Dossier contenant les crops")
    parser.add_argument("--output", type=str, default="classification_dataset/annotations/annotations_mount.csv", help="Fichier CSV de sortie")
    parser.add_argument("--max_images", type=int, default=None, help="Nombre max d'images à traiter (pour test)")
    args = parser.parse_args()

    annotator = MountAnnotator(args.crop_dir, args.output)
    records = annotator.process_all_images(args.max_images)
    annotator.save_annotations(records)


if __name__ == "__main__":
    main()
