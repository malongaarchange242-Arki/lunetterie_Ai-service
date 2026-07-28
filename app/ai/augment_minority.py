from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


class MinorityAugmenter:
    """Generate synthetic variations for minority classes in a balanced dataset."""

    def __init__(self, dataset_dir: str = "classification_dataset/balanced_controlled", target_count: int = 200) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.target_count = target_count

    def augment_image(self, image: np.ndarray) -> list[np.ndarray]:
        variations: list[np.ndarray] = []

        for angle in [-10, -5, 5, 10]:
            height, width = image.shape[:2]
            matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
            rotated = cv2.warpAffine(image, matrix, (width, height))
            variations.append(rotated)

        for scale in [0.95, 1.05]:
            height, width = image.shape[:2]
            matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 0, scale)
            scaled = cv2.warpAffine(image, matrix, (width, height))
            variations.append(scaled)

        variations.append(cv2.flip(image, 1))

        noise = np.random.normal(0, 5, image.shape).astype(np.uint8)
        variations.append(cv2.add(image, noise))

        for brightness in [0.9, 1.1]:
            bright = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
            variations.append(bright)

        return variations

    def augment_class(self, class_name: str) -> None:
        class_dir = self.dataset_dir / class_name
        if not class_dir.exists():
            print(f"❌ Dossier {class_name} non trouvé")
            return

        existing = [path for path in list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) if path.is_file()]
        if not existing:
            fallback_dir = Path("classification_dataset/crops")
            fallback_files = [path for path in list(fallback_dir.glob("*.jpg")) + list(fallback_dir.glob("*.png")) if path.is_file()]
            existing = fallback_files[:max(1, self.target_count - 1)]
        current_count = len(existing)

        if current_count >= self.target_count:
            print(f"✅ {class_name}: déjà {current_count} images, pas besoin d'augmentation")
            return

        print(f"🔄 {class_name}: {current_count} → {self.target_count} images")

        source_dir = self.dataset_dir / class_name
        aug_dir = self.dataset_dir / class_name / "augmented"
        aug_dir.mkdir(parents=True, exist_ok=True)

        generated = 0
        for img_path in existing:
            if generated >= self.target_count - current_count:
                break
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            variations = self.augment_image(image)
            for index, variation in enumerate(variations):
                if generated >= self.target_count - current_count:
                    break
                safe_stem = "".join(ch if ch.isalnum() else "_" for ch in img_path.stem)
                output_path = aug_dir / f"{safe_stem[:80]}_aug_{index}{img_path.suffix}"
                if cv2.imwrite(str(output_path), variation):
                    generated += 1

        for aug_path in sorted(aug_dir.glob("*")):
            if aug_path.is_file():
                shutil.move(str(aug_path), str(source_dir / aug_path.name))
        if aug_dir.exists():
            try:
                aug_dir.rmdir()
            except OSError:
                pass
        print(f"  ✅ Ajout de {generated} images augmentées")

    def augment_all_minority(self) -> None:
        print("🔄 Augmentation des classes minoritaires...")
        print(f"  Objectif: {self.target_count} images par classe")
        print("-" * 40)
        for class_name in ["Carrée", "Ovale", "Pilote"]:
            self.augment_class(class_name)
        print("-" * 40)
        print("✅ Augmentation terminée")
        print("\n📊 Nouvelles statistiques:")
        for class_dir in self.dataset_dir.iterdir():
            if class_dir.is_dir():
                count = len(list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")))
                print(f"  {class_dir.name:15} {count:4} images")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="classification_dataset/balanced_controlled")
    parser.add_argument("--target", type=int, default=200)
    args = parser.parse_args()

    augmenter = MinorityAugmenter(args.dataset, args.target)
    augmenter.augment_all_minority()


if __name__ == "__main__":
    main()
