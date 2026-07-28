from __future__ import annotations

import cv2
import numpy as np
import shutil
from pathlib import Path
import pandas as pd
import re
from tqdm import tqdm


def _is_image_valid(file_path: Path) -> bool:
    try:
        data = file_path.read_bytes()
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        return image is not None and image.size > 0
    except Exception:
        return False


def recreate_carre() -> int:
    """
    Recrée la classe Carrée à partir du dossier data/carré.
    """
    print("🔄 RECRÉATION DE LA CLASSE CARRÉE")
    print("=" * 60)

    source_dir = Path("data/carré")
    carre_dir = Path("classification_dataset/balanced_controlled/Carrée")
    csv_path = Path("classification_dataset/annotations/annotations_complete.csv")

    if carre_dir.exists():
        print("🗑️ Suppression du dossier Carrée existant...")
        for file in carre_dir.glob("*"):
            if file.is_file():
                file.unlink()
    else:
        carre_dir.mkdir(parents=True, exist_ok=True)

    images = [img for img in sorted(source_dir.glob("*")) if img.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    print(f"📸 {len(images)} images trouvées dans {source_dir}")

    if not images:
        print("❌ Aucune image trouvée dans data/carré")
        return 0

    print("\n📁 Copie des images...")
    copied = []

    for i, img_path in enumerate(tqdm(images, desc="Copie"), 1):
        new_name = f"carre_{i:04d}{img_path.suffix.lower()}"
        dst_path = carre_dir / new_name

        try:
            data = img_path.read_bytes()
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            image = None

        if image is None or image.size == 0:
            print(f"⚠️ Image corrompue ignorée: {img_path.name}")
            continue

        shutil.copy2(img_path, dst_path)
        copied.append(new_name)

    print(f"✅ {len(copied)} images copiées")

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
            for filename in copied
        ]

        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(csv_path, index=False)

        print(f"✅ CSV mis à jour: {len(df)} entrées")
        print(f"📊 Carrée dans le CSV: {len(df[df['shape'] == 'Carrée'])}")
    else:
        print(f"❌ CSV introuvable: {csv_path}")

    final_count = len(copied)
    print("\n" + "=" * 60)
    print(f"✅ {final_count} images carrées recréées")
    print(f"📁 Dossier: {carre_dir}")

    return final_count


if __name__ == "__main__":
    recreate_carre()
