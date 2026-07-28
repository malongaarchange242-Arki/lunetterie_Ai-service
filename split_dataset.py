import random
import shutil
from pathlib import Path

# ==============================
# Configuration
# ==============================

DATASET = Path("dataset")

SOURCE_IMAGES = DATASET / "train" / "images"
SOURCE_LABELS = DATASET / "train" / "labels"

TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1

SEED = 42

# ==============================
# Création des dossiers
# ==============================

for split in ["train", "valid", "test"]:
    (DATASET / split / "images").mkdir(parents=True, exist_ok=True)
    (DATASET / split / "labels").mkdir(parents=True, exist_ok=True)

# ==============================
# Liste des images
# ==============================

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

images = [
    f for f in SOURCE_IMAGES.iterdir()
    if f.suffix.lower() in extensions
]

random.seed(SEED)
random.shuffle(images)

total = len(images)

train_end = int(total * TRAIN_RATIO)
valid_end = train_end + int(total * VALID_RATIO)

train_images = images[:train_end]
valid_images = images[train_end:valid_end]
test_images = images[valid_end:]

print(f"Images trouvées : {total}")
print(f"Train : {len(train_images)}")
print(f"Valid : {len(valid_images)}")
print(f"Test  : {len(test_images)}")


def move_files(files, split):
    for img in files:

        label = SOURCE_LABELS / f"{img.stem}.txt"

        shutil.move(
            str(img),
            DATASET / split / "images" / img.name
        )

        if label.exists():
            shutil.move(
                str(label),
                DATASET / split / "labels" / label.name
            )


# Déplacer uniquement valid et test
# Les images train restent en place

move_files(valid_images, "valid")
move_files(test_images, "test")

print("\nDataset réparti avec succès !")