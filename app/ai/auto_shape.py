from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class ShapeAnnotator:
    """Annotate image crops with a coarse frame shape using geometric heuristics."""

    def __init__(self, crop_dir: str = "classification_dataset/crops", output_file: str = "classification_dataset/annotations/annotations_shape.csv") -> None:
        self.crop_dir = Path(crop_dir)
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

    def extract_contour(self, image: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        return max(contours, key=cv2.contourArea)

    def compute_shape_features(self, contour: np.ndarray) -> dict[str, float]:
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0.0
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0.0
        rect_area = w * h
        rectangularity = area / rect_area if rect_area > 0 else 0.0
        roundness = (4 * area) / (np.pi * (max(w, h) ** 2)) if max(w, h) > 0 else 0.0

        return {
            "perimeter": float(perimeter),
            "area": float(area),
            "aspect_ratio": float(aspect_ratio),
            "width": float(w),
            "height": float(h),
            "circularity": float(circularity),
            "convexity": float(convexity),
            "rectangularity": float(rectangularity),
            "solidity": float(area / hull_area) if hull_area > 0 else 0.0,
            "roundness": float(roundness),
        }

    def classify_shape(self, features: dict[str, float]) -> tuple[str, float]:
        aspect_ratio = features["aspect_ratio"]
        circularity = features["circularity"]
        rectangularity = features["rectangularity"]
        roundness = features["roundness"]

        if circularity > 0.5 or roundness > 0.72:
            return ("Ronde", 0.90)

        if aspect_ratio > 1.8:
            return ("Pilote", 0.85) if circularity < 0.15 else ("Rectangulaire", 0.80)
        if aspect_ratio > 1.4:
            return ("Ovale", 0.75) if circularity > 0.2 else ("Rectangulaire", 0.70)
        if aspect_ratio > 0.8:
            if rectangularity > 0.7:
                return ("Carrée", 0.80)
            return ("Papillon", 0.65)
        if circularity > 0.4:
            return ("Ovale", 0.70)
        return ("Papillon", 0.60)

    def is_valid_shape(self, contour: np.ndarray) -> bool:
        return cv2.contourArea(contour) > 500

    def process_image(self, image_path: Path) -> dict[str, object]:
        image = cv2.imread(str(image_path))
        if image is None:
            return {"error": "Impossible de lire l'image"}

        contour = self.extract_contour(image)
        if contour is None or not self.is_valid_shape(contour):
            return {"error": "Contour non détecté ou trop petit"}

        features = self.compute_shape_features(contour)
        shape_name, confidence = self.classify_shape(features)
        return {"filename": image_path.name, "shape": shape_name, "confidence": confidence, **features}

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
            print("\n📊 Statistiques des formes:")
            counts: dict[str, int] = {}
            for record in results:
                shape = str(record["shape"])
                counts[shape] = counts.get(shape, 0) + 1
            for shape, count in counts.items():
                print(f"  {shape}: {count} images ({count / len(results) * 100:.1f}%)")
        if errors:
            print(f"\n⚠️ {len(errors)} images en erreur:")
            for filename, error in errors[:5]:
                print(f"  {filename}: {error}")
        return results

    def save_annotations(self, records: list[dict[str, object]]) -> None:
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["filename", "shape", "confidence", "aspect_ratio", "circularity", "width", "height"]
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
    parser.add_argument("--output", type=str, default="classification_dataset/annotations/annotations_shape.csv", help="Fichier CSV de sortie")
    parser.add_argument("--max_images", type=int, default=None, help="Nombre max d'images à traiter (pour test)")
    args = parser.parse_args()

    annotator = ShapeAnnotator(args.crop_dir, args.output)
    records = annotator.process_all_images(args.max_images)
    annotator.save_annotations(records)


if __name__ == "__main__":
    main()
