import shutil
from pathlib import Path
import pandas as pd
import json
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kw):
        return x


class SquareFolderImporter:
    """Importe les images du dossier data/carré dans le dataset."""

    def __init__(self,
                 source_dir: str = "data/carré",
                 target_dir: str = "classification_dataset/balanced_controlled/Carrée",
                 csv_path: str = "classification_dataset/annotations/annotations_complete.csv"):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.csv_path = Path(csv_path)

        self.target_dir.mkdir(parents=True, exist_ok=True)

        self.stats = {
            "found": 0,
            "copied": 0,
            "skipped": 0,
            "errors": 0
        }

    def get_images(self):
        images = []
        for ext in ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']:
            images.extend(list(self.source_dir.glob(ext)))
        images = [img for img in images if img.is_file()]
        self.stats["found"] = len(images)
        return images

    def import_images(self):
        print("📥 IMPORTATION DES IMAGES CARRÉES")
        print("=" * 60)

        images = self.get_images()
        print(f"📸 {len(images)} images trouvées dans {self.source_dir}")

        if not images:
            print("❌ Aucune image trouvée")
            return 0

        if self.csv_path.exists():
            df = pd.read_csv(self.csv_path)
            print(f"📊 Dataset existant: {len(df)} images")
        else:
            df = pd.DataFrame(columns=['filename', 'shape', 'mount_type'])
            print("📊 Nouveau dataset créé")

        existing_carre = len(df[df['shape'] == 'Carrée'])
        print(f"📊 Carrée existantes: {existing_carre}")

        print("\n📁 Copie des images...")

        new_rows = []

        for img_path in tqdm(images, desc="Importation"):
            try:
                base_name = img_path.stem
                suffix = img_path.suffix
                clean_name = "".join(c for c in base_name if c.isalnum() or c in '._-')
                if not clean_name:
                    clean_name = f"carre_{self.stats['copied']+1:04d}"
                new_name = f"carre_{clean_name}{suffix}"

                counter = 1
                while (self.target_dir / new_name).exists():
                    new_name = f"carre_{clean_name}_{counter}{suffix}"
                    counter += 1

                dst_path = self.target_dir / new_name
                shutil.copy2(img_path, dst_path)

                new_rows.append({
                    'filename': new_name,
                    'shape': 'Carrée',
                    'mount_type': 'Pleine'
                })
                self.stats["copied"] += 1
            except Exception as e:
                self.stats["errors"] += 1
                print(f"⚠️ Erreur sur {img_path.name}: {e}")

        if new_rows:
            new_df = pd.DataFrame(new_rows)
            df = pd.concat([df, new_df], ignore_index=True)
            df.to_csv(self.csv_path, index=False)

        final_carre = len(df[df['shape'] == 'Carrée'])
        print(f"\n✅ {self.stats['copied']} images importées")
        print(f"📊 Total Carrée: {final_carre}")
        print(f"📊 Dataset total: {len(df)}")

        report_path = self.csv_path.parent / "import_carre_report.json"
        with open(report_path, "w", encoding='utf8') as f:
            json.dump({
                "source": str(self.source_dir),
                "images_found": self.stats["found"],
                "images_copied": self.stats["copied"],
                "errors": self.stats["errors"],
                "total_carre": final_carre,
                "total_dataset": len(df)
            }, f, indent=2, ensure_ascii=False)

        return self.stats["copied"]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="data/carré")
    parser.add_argument("--target", type=str, default="classification_dataset/balanced_controlled/Carrée")
    args = parser.parse_args()

    importer = SquareFolderImporter(args.source, args.target)
    count = importer.import_images()
    print(f"\n🎯 {count} images carrées ajoutées au dataset!")


if __name__ == "__main__":
    main()
