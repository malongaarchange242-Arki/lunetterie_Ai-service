from pathlib import Path
from ultralytics import YOLO
import cv2

# ==========================
# CONFIGURATION
# ==========================

MODEL_PATH = r"runs/detect/train-3/weights/best.pt"
# ou :
# MODEL_PATH = r"yolov8n.pt"

DATASET_DIR = Path("dataset")
OUTPUT_DIR = Path("classification_dataset/crops")

CONFIDENCE = 0.35
MARGIN = 0.15

# ==========================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO(MODEL_PATH)

# Recherche toutes les images
image_files = []

for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
    image_files.extend(DATASET_DIR.rglob(ext))

total = len(image_files)

saved = 0
ignored = 0

print(f"\nImages trouvées : {total}\n")

for i, image_path in enumerate(image_files, start=1):

    img = cv2.imread(str(image_path))

    if img is None:
        ignored += 1
        continue

    h, w = img.shape[:2]

    results = model.predict(
        source=img,
        conf=CONFIDENCE,
        verbose=False
    )

    if len(results) == 0 or len(results[0].boxes) == 0:
        ignored += 1
        continue

    # on prend la détection la plus confiante
    boxes = results[0].boxes

    best_box = max(boxes, key=lambda b: float(b.conf))

    x1, y1, x2, y2 = best_box.xyxy[0].tolist()

    bw = x2 - x1
    bh = y2 - y1

    mx = bw * MARGIN
    my = bh * MARGIN

    x1 = max(0, int(x1 - mx))
    y1 = max(0, int(y1 - my))
    x2 = min(w, int(x2 + mx))
    y2 = min(h, int(y2 + my))

    crop = img[y1:y2, x1:x2]

    output_name = image_path.stem + ".jpg"

    cv2.imwrite(
        str(OUTPUT_DIR / output_name),
        crop
    )

    saved += 1

    print(
        f"[{i}/{total}] ✓ {image_path.name}",
        end="\r"
    )

print("\n")
print("=" * 40)
print(f"Images analysées : {total}")
print(f"Montures enregistrées : {saved}")
print(f"Images ignorées : {ignored}")
print(f"Dossier : {OUTPUT_DIR.resolve()}")
print("=" * 40)
