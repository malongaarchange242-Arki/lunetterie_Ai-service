import shutil
from pathlib import Path
import pandas as pd
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kw):
        return x


def import_carre_properly():
    """Importe proprement les images carrées depuis data/datas."""
    print("🔍 RECHERCHE D'IMAGES CARRÉES DANS DATA/DATAS")
    print("=" * 60)

    source_dir = Path("data/datas")
    target_dir = Path("classification_dataset/balanced_controlled/Carrée")
    csv_path = Path("classification_dataset/annotations/annotations_complete.csv")

    target_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for pattern in ['*carre*', '*square*', '*carré*', '*carr?*']:
        images.extend(list(source_dir.rglob(pattern + ".jpg")))
        images.extend(list(source_dir.rglob(pattern + ".png")))
        images.extend(list(source_dir.rglob(pattern + ".jpeg")))

    images = sorted(set(images), key=lambda p: str(p))
    print(f"📸 {len(images)} images trouvées avec 'carre' dans le nom")

    if len(images) < 50:
        print("📸 Pas assez d'images carrées trouvées, analyse des images de lunettes...")
        all_glasses = list(source_dir.rglob("*.jpg")) + list(source_dir.rglob("*.png")) + list(source_dir.rglob("*.jpeg"))
        existing_names = {img.name for img in images}

        for img_path in tqdm(all_glasses, desc="Analyse"):
            if img_path.name in existing_names:
                continue
            try:
                import cv2
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                h, w = img.shape[:2]
                aspect = w / h
                if 0.7 < aspect < 1.3:
                    images.append(img_path)
                    existing_names.add(img_path.name)
            except Exception:
                continue

        images = sorted(set(images), key=lambda p: str(p))
        print(f"📸 {len(images)} images après analyse")

    if len(images) > 200:
        images = images[:200]

    print("\n📁 Copie des images...")

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=['filename', 'shape', 'mount_type'])

    existing_carre = len(df[df['shape'] == 'Carrée'])
    print(f"📊 Images Carrée déjà présentes: {existing_carre}")

    copied = 0
    new_rows = []

    for img_path in tqdm(images, desc="Copie"):
        base_name = img_path.stem
        suffix = img_path.suffix
        clean_name = "".join(c for c in base_name if c.isalnum() or c in '._-')
        if not clean_name:
            clean_name = f"carre_{copied+1:04d}"
        new_name = f"carre_import_{clean_name}{suffix}"

        counter = 1
        dst_path = target_dir / new_name
        while dst_path.exists():
            dst_path = target_dir / f"carre_import_{clean_name}_{counter}{suffix}"
            counter += 1

        try:
            shutil.copy2(img_path, dst_path)
            new_rows.append({'filename': dst_path.name, 'shape': 'Carrée', 'mount_type': 'Pleine'})
            copied += 1
        except Exception as e:
            print(f"⚠️ Erreur sur {img_path.name}: {e}")

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(csv_path, index=False)

    print(f"\n✅ {copied} images carrées ajoutées")
    print(f"📊 Total Carrée: {len(df[df['shape'] == 'Carrée'])} images")
    print(f"📊 Dataset total: {len(df)} images")


if __name__ == "__main__":
    import_carre_properly()
