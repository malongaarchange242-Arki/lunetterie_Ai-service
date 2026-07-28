from pathlib import Path

import pandas as pd
import torch
from torchvision import datasets


def check_class_mapping() -> None:
    print("🔍 VÉRIFICATION DU MAPPING DES CLASSES")
    print("=" * 60)

    train_path = Path("classification_dataset/forme/train")
    if train_path.exists():
        valid_subdirs = []
        for d in sorted(train_path.iterdir()):
            if not d.is_dir():
                continue
            image_files = list(d.glob("*.jpg")) + list(d.glob("*.jpeg")) + list(d.glob("*.png")) + list(d.glob("*.bmp"))
            if image_files:
                valid_subdirs.append(d.name)

        if valid_subdirs:
            train_classes = valid_subdirs
            print(f"\n📊 Classes valides dans forme/train:")
            for i, cls in enumerate(train_classes):
                print(f"  {i}: {cls}")
        else:
            print("\n⚠️ Aucun sous-dossier contenant des images trouvé dans train")
            train_classes = []
    else:
        print("\n⚠️ Dossier train non trouvé")
        train_classes = None

    model_path = Path("best_shape_model.pth")
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "classes" in checkpoint:
            model_classes = checkpoint["classes"]
            print(f"\n📊 Classes dans le checkpoint:")
            for i, cls in enumerate(model_classes):
                print(f"  {i}: {cls}")
        else:
            model_classes = None
            print("\n⚠️ Pas d'information de classe dans le checkpoint")
    else:
        model_classes = None
        print("\n⚠️ Modèle non trouvé")

    csv_path = Path("classification_dataset/annotations/annotations_complete.csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        csv_classes = sorted(df["shape"].dropna().unique())
        print(f"\n📊 Classes dans le CSV:")
        for cls in csv_classes:
            count = int((df["shape"] == cls).sum())
            print(f"  {cls}: {count} images")
    else:
        print("\n⚠️ CSV annotations_complete.csv non trouvé")
        csv_classes = None

    print("\n" + "=" * 60)
    print("🔍 COMPARAISON")
    print("=" * 60)

    if train_classes is not None and model_classes is not None:
        if train_classes == model_classes:
            print("✅ Les classes correspondent entre ImageFolder et le checkpoint")
        else:
            print("❌ LES CLASSES NE CORRESPONDENT PAS !")
            print("\nDifférences:")
            for i in range(max(len(train_classes), len(model_classes))):
                train = train_classes[i] if i < len(train_classes) else "N/A"
                model = model_classes[i] if i < len(model_classes) else "N/A"
                if train != model:
                    print(f"  Position {i}: {train} (train) vs {model} (checkpoint)")

    if train_classes is not None and model_classes is not None and train_classes != model_classes:
        print("\n💡 SOLUTION:")
        print("1. Ré-entraîner le modèle en sauvegardant les classes exactes")
        print("2. Ou corriger le mapping dans predict_shape()")


if __name__ == "__main__":
    check_class_mapping()
