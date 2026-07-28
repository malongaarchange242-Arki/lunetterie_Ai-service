from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

from app.ai.shape_estimator import ShapeEstimator

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms

MODEL_CLASS_DISPLAY_NAMES = {
    "aviateur": "Pilote",
    "carrée": "Carrée",
    "ovale": "Ovale",
    "papillon": "Papillon",
    "rectangulaire": "Rectangulaire",
    "ronde": "Ronde",
}
IGNORED_SHAPE_LABELS = {"", "inconnu", "unknown", "class_0"}


def build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class ShapeDataset(Dataset):
    def __init__(self, csv_path: str, image_dir: str, transform: Any | None = None, max_samples: int | None = None) -> None:
        self.image_dir = Path(image_dir)
        self.transform = transform or build_transform()

        self.rows: list[tuple[str, str, Path | None]] = []
        with open(csv_path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                filename = (row.get("filename") or "").strip()
                shape = (row.get("shape") or "").strip()
                path_value = (row.get("path") or "").strip()
                if filename and shape and shape.lower() not in IGNORED_SHAPE_LABELS:
                    image_path = Path(path_value) if path_value else None
                    self.rows.append((filename, shape, image_path))

        if max_samples is not None:
            self.rows = self._sample_rows_by_class(self.rows, max_samples)

        self.classes = sorted({shape for _, shape, _ in self.rows})
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

    def _sample_rows_by_class(self, rows: list[tuple[str, str, Path | None]], max_samples: int) -> list[tuple[str, str, Path | None]]:
        if max_samples <= 0 or len(rows) <= max_samples:
            return rows

        by_class: dict[str, list[tuple[str, str, Path | None]]] = {}
        for row in rows:
            by_class.setdefault(row[1], []).append(row)

        rng = random.Random(42)
        sampled: list[tuple[str, str, Path | None]] = []
        per_class = max(1, math.ceil(max_samples / max(1, len(by_class))))
        for class_name in sorted(by_class):
            class_rows = by_class[class_name][:]
            rng.shuffle(class_rows)
            sampled.extend(class_rows[:per_class])

        rng.shuffle(sampled)
        return sampled[:max_samples]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        filename, shape, path = self.rows[idx]
        if path is not None:
            image_path = path if path.is_absolute() else self.image_dir / path
        else:
            image_path = self.image_dir / filename
        image = Image.open(image_path).convert("RGB") if image_path.exists() else Image.new("RGB", (224, 224), "black")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.class_to_idx[shape]


class ShapeClassifierTrainer:
    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        epochs: int = 3,
        batch_size: int = 16,
        max_samples: int | None = 128,
        output_path: str = "app/models/classification/best_shape_model.pth",
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = ShapeDataset(csv_path, image_dir, max_samples=max_samples)
        if len(self.dataset.classes) < 2:
            raise ValueError(f"Training requires at least 2 shape classes, found: {self.dataset.classes}")
        self.epochs = epochs
        self.batch_size = batch_size
        self.output_path = Path(output_path)
        self.model = models.efficientnet_b0(weights=None)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, len(self.dataset.classes))
        self.model.to(self.device)
        self.shape_estimator = ShapeEstimator()

    def load_state(self, model_path: str = "best_shape_model.pth") -> None:
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def train(self) -> None:
        if len(self.dataset) == 0:
            print("Aucune image disponible pour l'entraînement.")
            return

        train_idx = list(range(len(self.dataset)))
        train_dataset = Subset(self.dataset, train_idx)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        labels = [self.dataset.rows[idx][1] for idx in train_idx]
        class_weights = compute_class_weights(labels, class_names=self.dataset.classes)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}/{self.epochs} - loss: {total_loss / max(1, len(train_loader)):.4f}")

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "classes": self.dataset.classes,
            "num_classes": len(self.dataset.classes),
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, self.output_path)
        torch.save(checkpoint, "best_shape_model.pth")
        print(f"Modele et mapping de classes sauvegardes dans {self.output_path}")


def compute_class_weights(labels: list[str], class_names: list[str] | None = None) -> torch.Tensor:
    classes = class_names or sorted(set(labels))
    counts = Counter(labels)
    total = sum(counts.values())
    weights = [total / max(1, counts.get(label, 1)) for label in classes]
    return torch.tensor(weights, dtype=torch.float32)


def normalize_shape_label(label: str) -> str:
    mapping = {
        "round": "Ronde",
        "square": "Carrée",
        "rectangle": "Rectangulaire",
        "oval": "Ovale",
        "aviator": "Pilote",
        "butterfly": "Papillon",
    }
    normalized = label.strip().lower()
    return mapping.get(normalized, label)


def merge_shape_prediction(model_probabilities: dict[str, float], heuristic_result: dict[str, Any]) -> dict[str, Any]:
    heuristic_label = normalize_shape_label(str(heuristic_result.get("shape", "unknown")))
    heuristic_conf = float(heuristic_result.get("confidence", 0.0))

    if heuristic_label == "unknown":
        return {
            "shape": max(model_probabilities, key=model_probabilities.get),
            "confidence": round(max(model_probabilities.values()) * 100.0, 2),
            "probabilities": {name: round(float(prob) * 100.0, 2) for name, prob in model_probabilities.items()},
        }

    boosted: dict[str, float] = {}
    for name, prob in model_probabilities.items():
        base = float(prob)
        if name == heuristic_label:
            boost = 0.35 + (heuristic_conf * 0.45)
            base = max(base, 0.08) + boost
        else:
            base *= 0.65 if heuristic_label in {"Pilote", "Papillon"} else 0.9
        boosted[name] = base

    total = sum(boosted.values())
    if total <= 0:
        total = 1.0
    normalized = {name: value / total for name, value in boosted.items()}
    best_name = max(normalized, key=normalized.get)
    confidence = max(normalized.values()) * 100.0
    if heuristic_label in {"Pilote", "Papillon"}:
        confidence = max(confidence, 80.0 + heuristic_conf * 10.0)
    return {
        "shape": best_name,
        "confidence": round(min(confidence, 99.99), 2),
        "probabilities": {name: round(float(prob) * 100.0, 2) for name, prob in normalized.items()},
    }


def predict_shape(image_path: str, model_path: str = "best_shape_model.pth", class_names: list[str] | None = None) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_b0(weights=None)

    # Load checkpoint (supports either raw state_dict or a dict with metadata)
    checkpoint = torch.load(model_path, map_location=device)
    checkpoint_classes = None
    checkpoint_num_classes = None
    if isinstance(checkpoint, dict):
        # Prefer explicit model_state_dict when present
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}
        checkpoint_classes = checkpoint.get("classes")
        checkpoint_num_classes = checkpoint.get("num_classes")
    else:
        state_dict = checkpoint

    # Determine number of output classes
    num_classes = None
    if checkpoint_num_classes is not None:
        num_classes = int(checkpoint_num_classes)
    else:
        for k, v in state_dict.items():
            if k.endswith("classifier.1.weight") or k.endswith("classifier.weight"):
                try:
                    num_classes = int(v.shape[0])
                    break
                except Exception:
                    continue
    if num_classes is None:
        raise RuntimeError("Unable to determine number of classes from checkpoint")

    # Recreate classifier head and load weights
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.to(device)
    try:
        model.load_state_dict(state_dict, strict=False)
    except RuntimeError:
        # tolerate prefixed keys like 'model.' by stripping a common prefix
        stripped = {k.split("model.", 1)[-1]: v for k, v in state_dict.items()}
        model.load_state_dict(stripped, strict=False)
    model.eval()

    # Resolve class name mapping: checkpoint -> explicit arg -> ImageFolder fallback -> autogenerated
    if checkpoint_classes:
        class_names = list(checkpoint_classes)
    if class_names is None:
        folder = Path("classification_dataset/forme/train")
        if folder.exists():
            dataset = datasets.ImageFolder(str(folder))
            class_names = dataset.classes
        else:
            class_names = [f"class_{i}" for i in range(num_classes)]

    # Ensure the class list length matches the model output
    if len(class_names) != num_classes:
        if len(class_names) > num_classes:
            class_names = class_names[:num_classes]
        else:
            class_names = class_names + [f"class_{i}" for i in range(len(class_names), num_classes)]

    display_names = [MODEL_CLASS_DISPLAY_NAMES.get(cls, cls) for cls in class_names]

    # Run model and compute probabilities
    with Image.open(image_path).convert("RGB") as image:
        tensor = build_transform()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    if len(display_names) != len(probs):
        display_names = display_names[: len(probs)]

    model_probabilities = {display_names[i]: float(probs[i]) for i in range(len(display_names))}
    heuristic_result = ShapeEstimator().estimate(str(image_path))
    merged = merge_shape_prediction(model_probabilities, heuristic_result)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="classification_dataset/annotations/annotations_complete.csv")
    parser.add_argument("--image-dir", default="classification_dataset/crops")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--output", default="app/models/classification/best_shape_model.pth")
    args = parser.parse_args()

    max_samples = None if args.max_samples <= 0 else args.max_samples
    trainer = ShapeClassifierTrainer(
        args.csv,
        args.image_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=max_samples,
        output_path=args.output,
    )
    trainer.train()


if __name__ == "__main__":
    main()
