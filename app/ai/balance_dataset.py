from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


class DatasetBalancer:
    """Organize and balance shape-classification images by class."""

    def __init__(self, csv_path: str = "classification_dataset/annotations/annotations_complete.csv", images_dir: str = "classification_dataset/crops_cleaned", output_dir: str = "classification_dataset/balanced") -> None:
        self.csv_path = Path(csv_path)
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fallback_images_dir = Path("classification_dataset/crops")

        self.df = pd.read_csv(csv_path)
        print(f"📊 Dataset chargé: {len(self.df)} images")
        print(f"📊 Classes: {self.df['shape'].unique().tolist()}")

    def create_class_folders(self) -> int:
        for shape in self.df["shape"].unique():
            (self.output_dir / str(shape)).mkdir(parents=True, exist_ok=True)

        print("\n📁 Organisation des images par classe...")
        copied = 0
        missing = 0

        for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
            src = self.images_dir / str(row["filename"])
            dst = self.output_dir / str(row["shape"]) / str(row["filename"])
            if src.exists():
                shutil.copy2(src, dst)
                copied += 1
            else:
                missing += 1

        print(f"\n✅ {copied} images copiées")
        if missing > 0:
            print(f"⚠️ {missing} images manquantes")
        return copied

    def get_class_stats(self) -> None:
        print("\n📊 Statistiques par classe:")
        print("-" * 40)
        for shape in sorted(self.df["shape"].unique()):
            count = int((self.df["shape"] == shape).sum())
            pct = count / len(self.df) * 100
            print(f"  {shape:15} {count:4} images ({pct:5.1f}%)")
        print("-" * 40)
        print(f"  {'Total':15} {len(self.df):4} images")

    def analyze_balance(self) -> int:
        counts = self.df["shape"].value_counts()
        max_count = int(counts.max())
        min_count = int(counts.min())

        print("\n⚖️ Analyse du déséquilibre:")
        print(f"  Classe majoritaire: {max_count} images")
        print(f"  Classe minoritaire: {min_count} images")
        print(f"  Ratio: {max_count / max(1, min_count):.1f}x")

        target = int(max_count * 0.8)
        print(f"\n  Nombre cible par classe: {target}")
        return target

    def balance_dataset(self, samples_per_class: int | None = None) -> Path:
        if samples_per_class is None:
            samples_per_class = self.analyze_balance()

        balanced_dir = self.output_dir.parent / "balanced_controlled"
        balanced_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n⚖️ Création d'un dataset équilibré ({samples_per_class} images/classe)...")

        for shape in self.df["shape"].unique():
            class_df = self.df[self.df["shape"] == shape]
            if len(class_df) >= samples_per_class:
                sampled = class_df.sample(n=samples_per_class, random_state=42)
            else:
                sampled = class_df.sample(n=samples_per_class, replace=True, random_state=42)

            (balanced_dir / str(shape)).mkdir(parents=True, exist_ok=True)
            for _, row in sampled.iterrows():
                src = self.images_dir / str(row["filename"])
                if not src.exists() and self.fallback_images_dir.exists():
                    src = self.fallback_images_dir / str(row["filename"])
                if not src.exists():
                    continue
                src_name = str(row["filename"])
                suffix = Path(src_name).suffix
                stem = Path(src_name).stem
                safe_name = f"{stem[:80]}_{abs(hash(src_name)) % 100000}{suffix}"
                dst = balanced_dir / str(shape) / safe_name
                if dst.exists():
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

        print(f"✅ Dataset équilibré créé dans {balanced_dir}")
        print("\n📊 Distribution des classes (équilibrée):")
        for shape in sorted(self.df["shape"].unique()):
            count = len(list((balanced_dir / str(shape)).glob("*")))
            print(f"  {shape:15} {count:4} images")

        return balanced_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="classification_dataset/annotations/annotations_complete.csv")
    parser.add_argument("--images", type=str, default="classification_dataset/crops_cleaned")
    parser.add_argument("--samples", type=int, default=None)
    args = parser.parse_args()

    balancer = DatasetBalancer(args.csv, args.images)
    balancer.get_class_stats()
    balancer.balance_dataset(args.samples)


if __name__ == "__main__":
    main()
