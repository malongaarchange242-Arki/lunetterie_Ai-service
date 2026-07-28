from __future__ import annotations

import csv
from pathlib import Path
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

from app.ai.shape_estimator import ShapeEstimator
from app.ai.train_shape_classifier import build_transform, merge_shape_prediction


def load_class_names(csv_path: Path):
    if csv_path.exists():
        with csv_path.open('r', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            labels = sorted({(row.get('shape') or '').strip() for row in reader if (row.get('shape') or '').strip()})
        return labels
    return ['Carrée', 'Ovale', 'Papillon', 'Pilote', 'Rectangulaire', 'Ronde']


def debug_predict(image_path: str, model_path: str = 'best_shape_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    num_classes = next(v.shape[0] for k, v in state_dict.items() if k.endswith('classifier.1.weight'))

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    csv_path = Path('classification_dataset/annotations/annotations_complete.csv')
    class_names = load_class_names(csv_path)
    class_names = class_names[:num_classes]

    with Image.open(image_path).convert('RGB') as image:
        tensor = build_transform()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    model_probabilities = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

    print('Model probabilities (raw):')
    for k, v in model_probabilities.items():
        print(f'  {k}: {v:.6f}')

    heuristic = ShapeEstimator().estimate(image_path)
    print('\nHeuristic result:')
    for k, v in heuristic.items():
        print(f'  {k}: {v}')

    merged = merge_shape_prediction(model_probabilities, heuristic)
    print('\nMerged result:')
    print(merged)


if __name__ == '__main__':
    import sys
    img = sys.argv[1] if len(sys.argv) > 1 else None
    if img is None or not Path(img).exists():
        # try to find the file by pattern if direct path fails (handles encoding issues)
        candidates = list(Path('.').rglob('*154225*.png'))
        if candidates:
            img = str(candidates[0].resolve())
        else:
            raise FileNotFoundError('Target image not found and no argument provided')
    debug_predict(img)
