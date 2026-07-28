import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent))

from app.ai.detector import GlassesDetector


def test_detection(image_path: str):
    print("🔍 TEST DE DÉTECTION ET CLASSIFICATION")
    print("=" * 60)
    print(f"📸 Image: {Path(image_path).name}")
    print()

    # 1. Détecter les lunettes avec YOLO
    print("1️⃣ Détection YOLO...")
    detector = GlassesDetector()
    detections = detector.detect(image_path)

    if not detections:
        print("   ❌ Aucune lunette détectée")
        return {"detected": False}

    best = max(detections, key=lambda det: det.get("confidence", 0.0))
    print(f"   ✅ Lunettes détectées avec confiance: {best['confidence']:.2%}")
    print(f"   📍 Bounding box: {tuple(int(coord) for coord in best['bbox'])}")

    # 2. Classifier la forme (importer dynamiquement pour éviter l'échec si torch absent)
    print("\n2️⃣ Classification de la forme...")
    try:
        from app.ai.train_shape_classifier import predict_shape
        shape_result = predict_shape(image_path)
        print(f"   📊 Forme prédite: {shape_result['shape']}")
        print(f"   📊 Confiance: {shape_result['confidence']:.2f}%")
    except Exception as exc:
        print("   ⚠️ Impossible de charger le classifieur (PyTorch absent ou erreur).")
        print("   --> Exécutez avec l'environnement virtuel du projet: .venv\\Scripts\\python.exe test_detection.py")
        print(f"   Détail erreur: {exc}")
        # Fallback minimal result
        shape_result = {"shape": "Inconnu", "confidence": 0.0, "probabilities": {"Inconnu": 100.0}}

    print("\n📈 Probabilités par classe:")
    print("-" * 40)
    for shape, prob in sorted(shape_result["probabilities"].items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob * 0.5 / 100)
        print(f"  {shape:15} {bar} {prob:.1f}%")

    display_result(image_path, best, shape_result)
    return {
        "detected": True,
        "shape": shape_result["shape"],
        "confidence": shape_result["confidence"],
        "bbox": tuple(int(coord) for coord in best["bbox"]),
    }


def display_result(image_path: str, detection: dict, shape_result: dict):
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Impossible de lire l'image {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 8))

    plt.subplot(1, 2, 1)
    x1, y1, x2, y2 = [int(coord) for coord in detection["bbox"]]
    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 3)
    label = f"{shape_result['shape']} ({detection['confidence']:.2%})"
    cv2.putText(img_rgb, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    plt.imshow(img_rgb)
    plt.title("Détection YOLO + Classification")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    shapes = list(shape_result["probabilities"].keys())
    probs = list(shape_result["probabilities"].values())
    colors = [('#2ecc71' if s == shape_result['shape'] else '#95a5a6') for s in shapes]
    plt.barh(shapes, probs, color=colors)
    plt.xlabel('Probabilité')
    plt.title('Distribution des probabilités')
    plt.xlim(0, 1)

    plt.tight_layout()
    plt.savefig('detection_result.png', dpi=150)
    plt.show()
    print(f"\n💾 Résultat sauvegardé dans detection_result.png")


if __name__ == "__main__":
    target = Path('D:\\Pojet la lunetterie\\Capture d’écran 2026-07-26 194736.png')
    if not target.exists():
        raise FileNotFoundError(f"Image non trouvée : {target}")
    test_detection(str(target))
