"""
Script de réentraînement du classifieur de formes
sur le dataset équilibré avec EfficientNet-B0.
"""

import json
import shutil
import sys
import time
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from tqdm import tqdm


DATASET_PATH = Path("classification_dataset/forme/balanced_train")
FALLBACK_DATASET_PATHS = [
    Path("classification_dataset/forme/train"),
    Path("classification_dataset/forme"),
]
MODEL_OUTPUT = Path("app/models/classification/best_shape_model_balanced.pth")
REPORT_OUTPUT = Path("training_report.json")

BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 2e-4
VAL_SPLIT = 0.2
IMAGE_SIZE = 224
NUM_WORKERS = 0
LABEL_SMOOTHING = 0.1
EARLY_STOPPING_PATIENCE = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📟 Device: {DEVICE}")

train_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def prepare_dataset_for_training(source_path: Path, output_path: Path | None = None) -> Path:
    """Prépare un dossier d'entraînement en supprimant les classes vides et en copiant les images valides."""
    if output_path is None:
        output_path = source_path.parent / f"{source_path.name}_prepared"

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    valid_class_dirs = []
    for class_dir in sorted(source_path.iterdir()):
        if not class_dir.is_dir():
            continue
        image_files = [
            image_path
            for image_path in class_dir.iterdir()
            if image_path.is_file() and image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}
        ]
        if not image_files:
            continue
        valid_class_dirs.append((class_dir.name, image_files))

    if not valid_class_dirs:
        raise FileNotFoundError(f"Aucun dataset valide trouvé dans {source_path}")

    for class_name, image_files in valid_class_dirs:
        target_class_dir = output_path / class_name
        target_class_dir.mkdir(parents=True, exist_ok=True)
        for image_path in image_files:
            shutil.copy2(image_path, target_class_dir / image_path.name)

    return output_path


def resolve_dataset_path() -> Path | None:
    """Choisit le premier dossier de dataset contenant au moins une classe avec des images valides."""
    candidates = [DATASET_PATH, *FALLBACK_DATASET_PATHS]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if not candidate.exists():
            continue
        try:
            prepared_path = prepare_dataset_for_training(candidate)
            return prepared_path
        except FileNotFoundError as exc:
            print(f"⚠️  {candidate} ignoré: {exc}")
    return None


def load_dataset():
    """Charge le dataset équilibré ou un fallback valide."""
    print("\n📂 Chargement du dataset...")

    dataset_path = resolve_dataset_path()
    if dataset_path is None:
        print(f"❌ Aucun dataset valide trouvé. Vérifiez l'un des chemins suivants :")
        for candidate in [DATASET_PATH, *FALLBACK_DATASET_PATHS]:
            print(f"   - {candidate}")
        print("   Lancez d'abord: python balance_dataset.py")
        return None, None, None, None

    print(f"   Dataset utilisé: {dataset_path}")
    full_dataset = datasets.ImageFolder(str(dataset_path), transform=train_transform)
    class_names = full_dataset.classes
    num_classes = len(class_names)

    print(f"   Classes: {num_classes} → {class_names}")
    print(f"   Images totales: {len(full_dataset)}")

    class_counts = {}
    for _, label in full_dataset:
        name = class_names[label]
        class_counts[name] = class_counts.get(name, 0) + 1

    print("\n   Distribution:")
    for name, count in sorted(class_counts.items()):
        bar = "█" * (count // 10)
        print(f"   {name:<20} {count:>5}  {bar}")

    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"\n   Train: {train_size} images ({len(train_loader)} batches)")
    print(f"   Val:   {val_size} images ({len(val_loader)} batches)")

    return train_loader, val_loader, class_names, num_classes


def create_model(num_classes: int):
    """Crée un modèle EfficientNet-B0 pré-entraîné."""
    print("\n🏗️  Création du modèle EfficientNet-B0...")

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    for param in model.features[:5].parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(256, num_classes),
    )

    model = model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"   Paramètres totaux: {total_params:,}")
    print(f"   Paramètres entraînables: {trainable_params:,}")

    return model


def train_epoch(model, loader, criterion, optimizer, epoch, num_epochs):
    """Une époque d'entraînement."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")

    for inputs, labels in pbar:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({"loss": f"{running_loss / len(loader):.3f}", "acc": f"{100. * correct / total:.1f}%"})

    return running_loss / len(loader), 100.0 * correct / total


def validate_epoch(model, loader, criterion):
    """Une époque de validation."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="[Val]  "):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / len(loader), 100.0 * correct / total, all_preds, all_labels


def train_model(model, train_loader, val_loader, class_names, num_epochs=NUM_EPOCHS):
    """Boucle d'entraînement complète."""
    print("\n🚀 Début de l'entraînement...")
    print("=" * 70)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "best_acc": 0,
        "best_epoch": 0,
    }

    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch, num_epochs)
        val_loss, val_acc, val_preds, val_labels = validate_epoch(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > history["best_acc"]:
            history["best_acc"] = val_acc
            history["best_epoch"] = epoch + 1
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "class_names": class_names,
                },
                MODEL_OUTPUT,
            )
            print(f"   💾 Meilleur modèle sauvegardé (acc: {val_acc:.1f}%)")
        else:
            epochs_without_improvement += 1

        epoch_time = time.time() - start_time
        print(f"   ⏱️  Temps: {epoch_time:.1f}s")
        print()

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"   ⏹️  Early stopping: pas d'amélioration depuis {EARLY_STOPPING_PATIENCE} epochs (meilleur: {history['best_acc']:.1f}% à l'epoch {history['best_epoch']})\n")
            break

    return history


def evaluate_model(model, val_loader, class_names):
    """Évalue le modèle et génère les métriques."""
    print("\n📊 Évaluation du modèle...")
    print("=" * 70)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Évaluation"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    print("\n📋 Rapport de classification:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    results = {
        "class_names": class_names,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "accuracy": float(np.mean(np.array(all_preds) == np.array(all_labels))),
    }

    with REPORT_OUTPUT.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    print(f"\n📁 Rapport sauvegardé: {REPORT_OUTPUT}")
    return results


def plot_training_history(history):
    """Affiche les courbes d'entraînement."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["train_loss"], label="Train", linewidth=2)
    ax1.plot(history["val_loss"], label="Validation", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Courbe de perte")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["train_acc"], label="Train", linewidth=2)
    ax2.plot(history["val_acc"], label="Validation", linewidth=2)
    ax2.scatter(
        history["best_epoch"] - 1,
        history["best_acc"],
        color="red",
        s=100,
        zorder=5,
        label=f'Best ({history["best_acc"]:.1f}%)',
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Courbe de précision")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()
    print("\n📈 Courbes sauvegardées: training_curves.png")


def main() -> None:
    print("\n🕶️  RÉENTRAÎNEMENT CLASSIFIEUR DE FORMES")
    print("=" * 70)

    train_loader, val_loader, class_names, num_classes = load_dataset()
    if train_loader is None:
        return

    model = create_model(num_classes)
    history = train_model(model, train_loader, val_loader, class_names)

    print(f"\n🔄 Rechargement du meilleur modèle (epoch {history['best_epoch']}, acc: {history['best_acc']:.1f}%) pour l'évaluation finale...")
    checkpoint = torch.load(MODEL_OUTPUT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    evaluate_model(model, val_loader, class_names)
    plot_training_history(history)

    print("\n" + "=" * 70)
    print("  ✅ ENTRAÎNEMENT TERMINÉ")
    print("=" * 70)
    print(f"  Meilleure précision: {history['best_acc']:.2f}% (Epoch {history['best_epoch']})")
    print(f"  Modèle: {MODEL_OUTPUT}")
    print(f"  Rapport: {REPORT_OUTPUT}")


if __name__ == "__main__":
    main()
