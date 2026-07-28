from __future__ import annotations

import cv2
import numpy as np
import shutil
from pathlib import Path
import pandas as pd
import re


def _is_image_valid(file_path: Path) -> bool:
    try:
        data = file_path.read_bytes()
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        return image is not None and image.size > 0
    except Exception:
        return False


def restore_and_validate_carre() -> int:
    """
    Restaure les fichiers depuis Carrée_backup, valide chaque image,
    et ne garde que les fichiers lisibles avec des noms propres.
    """
    print("🔄 RESTAURATION ET VALIDATION DE LA CLASSE CARRÉE")
    print("=" * 60)

    backup_dir = Path("classification_dataset/balanced_controlled/Carrée_backup")
    carre_dir = Path("classification_dataset/balanced_controlled/Carrée")
    csv_path = Path("classification_dataset/annotations/annotations_complete.csv")

    if not backup_dir.exists():
        print("❌ Le dossier Carrée_backup n'existe pas !")
        return 0

    carre_dir.mkdir(parents=True, exist_ok=True)

    backup_files = [path for path in sorted(backup_dir.iterdir()) if path.is_file()]
    print(f"📸 {len(backup_files)} fichiers trouvés dans le backup")

    if not backup_files:
        print("❌ Aucun fichier dans le backup")
        return 0

    print("\n🔍 Validation des fichiers...")
    valid_files: list[tuple[Path, str]] = []
    invalid_files: list[Path] = []

    for file_path in backup_files:
        if file_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            invalid_files.append(file_path)
            continue

        if _is_image_valid(file_path):
            original_name = file_path.name
            clean_name = re.sub(r'[^a-zA-Z0-9._-]', '_', original_name)
            if len(clean_name) > 100:
                name_parts = clean_name.split('.')
                if len(name_parts) > 1:
                    clean_name = name_parts[0][:90] + '.' + name_parts[-1]
                else:
                    clean_name = clean_name[:100]
            valid_files.append((file_path, clean_name))
        else:
            invalid_files.append(file_path)

    print(f"\n✅ {len(valid_files)} fichiers valides")
    print(f"🗑️ {len(invalid_files)} fichiers invalides ignorés")

    if not valid_files:
        print("❌ Aucun fichier valide trouvé")
        return 0

    print("\n📁 Copie des fichiers valides...")
    copied_files: list[str] = []

    for src_path, clean_name in valid_files:
        dst_path = carre_dir / clean_name
        counter = 1
        while dst_path.exists():
            name_parts = clean_name.split('.')
            if len(name_parts) > 1:
                new_name = f"{name_parts[0]}_{counter}.{name_parts[-1]}"
            else:
                new_name = f"{clean_name}_{counter}"
            dst_path = carre_dir / new_name
            counter += 1
        shutil.copy2(src_path, dst_path)
        copied_files.append(dst_path.name)

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"\n📊 Dataset actuel: {len(df)} images")

        df = df[df['shape'] != 'Carrée']

        new_rows = [
            {
                'filename': filename,
                'shape': 'Carrée',
                'mount_type': 'Pleine'
            }
            for filename in copied_files
        ]

        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(csv_path, index=False)

        print(f"✅ CSV mis à jour: {len(df)} entrées")
        print(f"📊 Carrée dans le CSV: {len(df[df['shape'] == 'Carrée'])}")
    else:
        print(f"❌ CSV introuvable: {csv_path}")

    final_count = len(copied_files)
    print("\n" + "=" * 60)
    print(f"✅ {final_count} images carrées restaurées et validées")
    print(f"📁 Dossier: {carre_dir}")

    return final_count


if __name__ == "__main__":
    restore_and_validate_carre()
