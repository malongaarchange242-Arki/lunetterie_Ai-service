from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class ColorAnnotator:
    """Assign a coarse color label to a crop using a small reference palette and dominant color estimation."""

    COLOR_REFERENCE = {
        "Noir": (0, 0, 0),
        "Gris": (128, 128, 128),
        "Blanc": (255, 255, 255),
        "Argent": (192, 192, 192),
        "Doré": (215, 180, 50),
        "Rouge": (200, 50, 50),
        "Bleu": (50, 50, 200),
        "Vert": (50, 200, 50),
        "Marron": (101, 67, 33),
        "Tortoise": (139, 90, 43),
        "Rose": (230, 150, 180),
        "Violet": (150, 50, 200),
    }

    def __init__(self, crop_dir: str = "classification_dataset/crops", output_file: str = "classification_dataset/annotations/annotations_color.csv") -> None:
        self.crop_dir = Path(crop_dir)
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.reference_colors = np.array(list(self.COLOR_REFERENCE.values()), dtype=np.float32)
        self.reference_names = list(self.COLOR_REFERENCE.keys())

    def extract_dominant_color(self, image: np.ndarray) -> np.ndarray:
        if image.shape[0] > 200 or image.shape[1] > 200:
            image = cv2.resize(image, (200, 200))

        pixels = image.reshape(-1, 3)
        if len(pixels) > 10000:
            pixels = pixels[np.linspace(0, len(pixels) - 1, 10000, dtype=int)]

        r = pixels[:, 0].mean()
        g = pixels[:, 1].mean()
        b = pixels[:, 2].mean()
        return np.array([b, g, r], dtype=np.float32)

    def get_color_name(self, bgr_color: np.ndarray) -> tuple[str, float]:
        rgb_color = np.array([bgr_color[2], bgr_color[1], bgr_color[0]], dtype=np.float32)
        distances = np.linalg.norm(self.reference_colors - rgb_color, axis=1)
        min_idx = int(np.argmin(distances))
        max_distance = 255 * np.sqrt(3)
        confidence = max(0.0, 1.0 - (distances[min_idx] / max_distance))
        return self.reference_names[min_idx], confidence

    def process_image(self, image_path: Path) -> dict[str, object]:
        image = cv2.imread(str(image_path))
        if image is None:
            return {"error": "Impossible de lire l'image"}

        dominant_color = self.extract_dominant_color(image)
        color_name, confidence = self.get_color_name(dominant_color)
        return {
            "filename": image_path.name,
            "color": color_name,
            "confidence": round(float(confidence), 3),
            "dominant_bgr": tuple(int(v) for v in dominant_color),
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
            print("\n📊 Statistiques des couleurs:")
            counts: dict[str, int] = {}
            for record in results:
                color = str(record["color"])
                counts[color] = counts.get(color, 0) + 1
            for color, count in counts.items():
                print(f"  {color}: {count} images ({count / len(results) * 100:.1f}%)")
        if errors:
            print(f"\n⚠️ {len(errors)} images en erreur:")
            for filename, error in errors[:5]:
                print(f"  {filename}: {error}")
        return results

    def save_annotations(self, records: list[dict[str, object]]) -> None:
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["filename", "color", "confidence", "dominant_bgr"]
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
    parser.add_argument("--output", type=str, default="classification_dataset/annotations/annotations_color.csv", help="Fichier CSV de sortie")
    parser.add_argument("--max_images", type=int, default=None, help="Nombre max d'images à traiter (pour test)")
    args = parser.parse_args()

    annotator = ColorAnnotator(args.crop_dir, args.output)
    records = annotator.process_all_images(args.max_images)
    annotator.save_annotations(records)


if __name__ == "__main__":
    main()
