"""
Script de rééquilibrage du dataset de formes de montures
- Analyse la distribution actuelle
- Augmente les classes sous-représentées
- Génère un dataset équilibré pour l'entraînement
"""

import shutil
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

try:
    import albumentations as A
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "albumentations n'est pas installé. Installez-le avec: pip install albumentations tqdm opencv-python numpy scikit-learn"
    ) from exc


DATASET_PATH = Path("classification_dataset/forme")
TRAIN_PATH = DATASET_PATH / "train"
OUTPUT_PATH = DATASET_PATH / "balanced_train"
TARGET_COUNT = 400


def get_augmentation_pipeline() -> A.Compose:
    """Définit les transformations d'augmentation."""
    return A.Compose(
        [
            A.Rotate(limit=15, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.CLAHE(clip_limit=2.0, p=0.2),
            A.Sharpen(alpha=(0.2, 0.5), p=0.3),
            A.Emboss(alpha=(0.2, 0.5), p=0.1),
            A.CropAndPad(percent=(-0.1, 0.1), pad_mode=cv2.BORDER_CONSTANT, p=0.3),
        ]
    )


def augment_image(image_path: Path, output_dir: Path, num_augment: int, transform: A.Compose) -> int:
    """Augmente une image et sauvegarde les variantes."""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"⚠️  Impossible de lire: {image_path}")
        return 0

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    base_name = image_path.stem
    count = 0

    for i in range(num_augment):
        try:
            augmented = transform(image=image)
            aug_image = augmented["image"]
            output_name = f"{base_name}_aug{i:03d}.jpg"
            output_path = output_dir / output_name
            aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), aug_image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            count += 1
        except Exception as exc:
            print(f"⚠️  Erreur augmentation {image_path.name} #{i}: {exc}")

    return count


def analyze_dataset(train_path: Path) -> dict[str, dict[str, Any]]:
    """Analyse la distribution actuelle du dataset."""
    print("\n" + "=" * 60)
    print("  📊 ANALYSE DU DATASET ACTUEL")
    print("=" * 60)

    classes: dict[str, dict[str, Any]] = {}
    total = 0

    for class_dir in sorted(train_path.iterdir()):
        if class_dir.is_dir():
            images = list(class_dir.glob("*.[jp][pn][g]*"))
            count = len(images)
            classes[class_dir.name] = {"count": count, "images": images}
            total += count

    print(f"\n  Total images: {total}")
    print(f"  Classes: {len(classes)}\n")

    for class_name, info in sorted(classes.items(), key=lambda item: item[1]["count"], reverse=True):
        count = info["count"]
        bar = "█" * (count // 20)
        print(f"  {class_name:<20} {count:>5}  {bar}")

    return classes


def balance_dataset(train_path: Path, output_path: Path, target_count: int = TARGET_COUNT) -> dict[str, dict[str, int]]:
    """Équilibre le dataset en copiant les originales et en augmentant les classes sous-représentées."""
    print("\n" + "=" * 60)
    print("  ⚖️  RÉÉQUILIBRAGE DU DATASET")
    print("=" * 60)

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    classes = analyze_dataset(train_path)
    transform = get_augmentation_pipeline()

    total_generated = 0
    stats: dict[str, dict[str, int]] = {}

    for class_name, info in tqdm(classes.items(), desc="Progression"):
        class_output = output_path / class_name
        class_output.mkdir(exist_ok=True)

        images = info["images"]
        copied = 0
        for img_path in images:
            dest = class_output / img_path.name
            shutil.copy2(img_path, dest)
            copied += 1

        needed = max(0, target_count - len(images))
        generated = 0
        if needed > 0 and len(images) > 0:
            aug_per_image = max(1, needed // len(images))
            for img_path in images:
                if generated >= needed:
                    break
                n = min(aug_per_image, needed - generated)
                generated += augment_image(img_path, class_output, n, transform)

        total_generated += generated
        stats[class_name] = {"original": copied, "generated": generated, "total": copied + generated}

        status = "✅" if (copied + generated) >= target_count else "⚠️"
        print(f"\n  {status} {class_name}: {copied} orig + {generated} aug = {copied + generated} total")

    print("\n" + "=" * 60)
    print("  📊 RÉSUMÉ FINAL")
    print("=" * 60)

    for class_name, info in sorted(stats.items(), key=lambda item: item[1]["total"], reverse=True):
        total = info["total"]
        bar = "█" * (total // 20)
        print(f"  {class_name:<20} {total:>5}  {bar}")

    print(f"\n  ✅ Total images générées: {total_generated}")
    print(f"  📁 Dataset équilibré: {output_path}")
    return stats


def create_validation_split(output_path: Path, val_ratio: float = 0.2) -> None:
    """Crée un split train/validation à partir du dataset équilibré."""
    print("\n" + "=" * 60)
    print("  🔀 CRÉATION SPLIT TRAIN/VALIDATION")
    print("=" * 60)

    train_path = output_path.parent / "train_balanced"
    val_path = output_path.parent / "val_balanced"

    for p in [train_path, val_path]:
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True)

    for class_dir in output_path.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            train_class = train_path / class_name
            val_class = val_path / class_name
            train_class.mkdir(exist_ok=True)
            val_class.mkdir(exist_ok=True)

            images = list(class_dir.glob("*.[jp][pn][g]*"))
            random.shuffle(images)
            val_count = int(len(images) * val_ratio)
            val_images = images[:val_count]
            train_images = images[val_count:]

            for img in train_images:
                shutil.copy2(img, train_class / img.name)
            for img in val_images:
                shutil.copy2(img, val_class / img.name)

            print(f"  {class_name:<20} train: {len(train_images):>4}  val: {len(val_images):>4}")

    print(f"\n  ✅ Split créé:")
    print(f"     Train: {train_path}")
    print(f"     Val:   {val_path}")


def main() -> None:
    """Point d'entrée principal."""
    print("\n🕶️  RÉÉQUILIBRAGE DATASET MONTURES")
    print("=" * 60)

    if not TRAIN_PATH.exists():
        print(f"\n❌ Dataset source introuvable: {TRAIN_PATH}")
        print("   Vérifiez le chemin et réessayez.")
        return

    analyze_dataset(TRAIN_PATH)

    print(f"\n📋 Configuration:")
    print(f"   Source: {TRAIN_PATH}")
    print(f"   Sortie: {OUTPUT_PATH}")
    print(f"   Cible: {TARGET_COUNT} images/classe")

    response = input("\n👉 Lancer le rééquilibrage ? (o/n): ").strip().lower()
    if response not in ["o", "oui", "y", "yes"]:
        print("   Annulé.")
        return

    balance_dataset(TRAIN_PATH, OUTPUT_PATH, TARGET_COUNT)

    response = input("\n👉 Créer le split train/validation ? (o/n): ").strip().lower()
    if response in ["o", "oui", "y", "yes"]:
        create_validation_split(OUTPUT_PATH, val_ratio=0.2)

    print("\n✅ Terminé !")
    print(f"\n📁 Fichiers créés:")
    print(f"   Dataset équilibré: {OUTPUT_PATH}")
    print(f"   Train équilibré:  {OUTPUT_PATH.parent / 'train_balanced'}")
    print(f"   Val équilibré:    {OUTPUT_PATH.parent / 'val_balanced'}")


if __name__ == "__main__":
    main()
