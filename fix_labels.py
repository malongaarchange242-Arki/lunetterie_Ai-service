# script de nettoyage pour corriger les classes dans les fichiers de labels YOLO

from pathlib import Path

DATASET = Path("dataset")

splits = ["train", "valid", "test"]

files_fixed = 0
labels_fixed = 0

for split in splits:

    labels_dir = DATASET / split / "labels"

    if not labels_dir.exists():
        continue

    for txt_file in labels_dir.glob("*.txt"):

        new_lines = []
        modified = False

        with open(txt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 5:
                continue

            old_class = parts[0]

            if old_class != "0":
                modified = True
                labels_fixed += 1

            parts[0] = "0"

            new_lines.append(" ".join(parts))

        if modified:
            files_fixed += 1

            with open(txt_file, "w", encoding="utf-8") as f:
                f.write("\n".join(new_lines) + "\n")

print("=" * 50)
print("Nettoyage terminé")
print(f"Fichiers corrigés : {files_fixed}")
print(f"Labels modifiés   : {labels_fixed}")
print("=" * 50)