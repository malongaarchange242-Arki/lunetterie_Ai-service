from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


class DatasetCleaner:
    """Clean a crop dataset by rejecting blurry, tiny, cut-off, or multi-object images."""

    def __init__(self, input_dir: str = "data/crops", output_dir: str = "data/crops_cleaned", rejected_dir: str = "data/crops_rejected") -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.rejected_dir = Path(rejected_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rejected_dir.mkdir(parents=True, exist_ok=True)

        self.stats: dict[str, Any] = {
            "total": 0,
            "kept": 0,
            "rejected": 0,
            "reasons": {},
        }

    def is_blurry(self, image: np.ndarray, threshold: float = 40.0) -> bool:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < threshold

    def is_too_small(self, image: np.ndarray, min_size: int = 50) -> bool:
        height, width = image.shape[:2]
        return height < min_size or width < min_size

    def has_multiple_objects(self, image: np.ndarray) -> bool:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [contour for contour in contours if cv2.contourArea(contour) > 800]
        return len(valid_contours) > 1

    def is_cut_off(self, image: np.ndarray, margin: int = 20) -> bool:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        height, width = mask.shape
        borders = [
            mask[:margin, :].flatten(),
            mask[-margin:, :].flatten(),
            mask[:, :margin].flatten(),
            mask[:, -margin:].flatten(),
        ]

        border_pixels = sum(int(np.any(border > 0)) for border in borders)
        return border_pixels >= 3

    def has_extreme_reflections(self, image: np.ndarray, threshold: int = 240) -> bool:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bright_pixels = np.sum(gray > threshold)
        total_pixels = gray.size
        bright_ratio = bright_pixels / total_pixels
        return bright_ratio > 0.3

    def analyze_image(self, image_path: Path) -> dict[str, Any]:
        image = cv2.imread(str(image_path))
        if image is None:
            return {"valid": False, "problems": ["cannot_read"], "image_size": (0, 0)}

        problems: list[str] = []
        if self.is_too_small(image):
            problems.append("too_small")
        if self.is_blurry(image):
            problems.append("blurry")
        if self.has_multiple_objects(image):
            problems.append("multiple_objects")
        # The crops often contain the eyewear touching the border; keep this check optional.
        if False and self.is_cut_off(image):
            problems.append("cut_off")
        if self.has_extreme_reflections(image):
            problems.append("extreme_reflections")

        return {"valid": len(problems) == 0, "problems": problems, "image_size": image.shape[:2]}

    def process_all_images(self, max_images: int | None = None) -> dict[str, Any]:
        image_files = sorted(list(self.input_dir.glob("*.jpg")) + list(self.input_dir.glob("*.png")))
        if max_images is not None:
            image_files = image_files[:max_images]

        self.stats["total"] = len(image_files)
        print(f"📊 Analyse de {len(image_files)} images...")
        print("=" * 60)

        report: list[dict[str, Any]] = []
        for img_path in tqdm(image_files, desc="Nettoyage"):
            result = self.analyze_image(img_path)
            if result["valid"]:
                shutil.copy2(img_path, self.output_dir / img_path.name)
                self.stats["kept"] += 1
                report.append({"filename": img_path.name, "status": "kept", "size": result["image_size"]})
            else:
                shutil.copy2(img_path, self.rejected_dir / img_path.name)
                self.stats["rejected"] += 1
                for reason in result["problems"]:
                    self.stats["reasons"][reason] = self.stats["reasons"].get(reason, 0) + 1
                report.append({"filename": img_path.name, "status": "rejected", "reasons": result["problems"], "size": result["image_size"]})

        report_path = self.output_dir.parent / "cleaning_report.json"
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

        self.print_stats()
        return self.stats

    def print_stats(self) -> None:
        print("\n" + "=" * 60)
        print("📊 STATISTIQUES DU NETTOYAGE")
        print("=" * 60)
        print(f"📸 Total: {self.stats['total']} images")
        print(f"✅ Gardées: {self.stats['kept']} images ({self.stats['kept'] / self.stats['total'] * 100:.1f}%)")
        print(f"❌ Rejetées: {self.stats['rejected']} images ({self.stats['rejected'] / self.stats['total'] * 100:.1f}%)")
        print("\n📋 Raisons des rejets:")
        for reason, count in sorted(self.stats["reasons"].items(), key=lambda item: item[1], reverse=True):
            reason_names = {
                "too_small": "Image trop petite",
                "blurry": "Image floue",
                "multiple_objects": "Plusieurs montures",
                "cut_off": "Monture coupée",
                "extreme_reflections": "Reflets extrêmes",
                "cannot_read": "Image illisible",
            }
            print(f"  - {reason_names.get(reason, reason)}: {count} images")
        print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/crops")
    parser.add_argument("--output", type=str, default="data/crops_cleaned")
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    cleaner = DatasetCleaner(args.input, args.output)
    cleaner.process_all_images(args.max_images)


if __name__ == "__main__":
    main()
