from pathlib import Path
import shutil
import pandas as pd
import json
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kw):
        return x


class AllDataImporter:
    """Import all images from a source directory into the classification dataset."""

    def __init__(self, source_dir: str = "data/datas",
                 target_dir: str = "classification_dataset/balanced_controlled",
                 csv_path: str = "classification_dataset/annotations/annotations_complete.csv"):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.csv_path = Path(csv_path)

        self.target_dir.mkdir(parents=True, exist_ok=True)
        self.target_dir.parent.mkdir(parents=True, exist_ok=True)

        self.stats = {"total": 0, "copied": 0, "skipped": 0, "errors": 0, "by_source": {}}

    def get_all_images(self):
        images = []
        if not self.source_dir.exists():
            return images

        for subdir in sorted(self.source_dir.rglob("*")):
            if subdir.is_dir():
                jpg = list(subdir.glob("*.jpg")) + list(subdir.glob("*.jpeg"))
                png = list(subdir.glob("*.png"))
                imgs = jpg + png
                if imgs:
                    images.extend(imgs)
                    try:
                        rel = str(subdir.relative_to(self.source_dir))
                    except Exception:
                        rel = str(subdir)
                    self.stats["by_source"][rel] = len(imgs)

        self.stats["total"] = len(images)
        return images

    def determine_shape(self, filename: str) -> str:
        name_lower = filename.lower()
        if 'square' in name_lower or 'carre' in name_lower or 'carr' in name_lower:
            return 'Carrée'
        if 'round' in name_lower or 'ronde' in name_lower:
            return 'Ronde'
        if 'oval' in name_lower or 'ovale' in name_lower:
            return 'Ovale'
        if 'rect' in name_lower or 'rectangle' in name_lower:
            return 'Rectangulaire'
        if 'pilot' in name_lower or 'aviator' in name_lower or 'pilote' in name_lower:
            return 'Pilote'
        if 'butterfly' in name_lower or 'papillon' in name_lower:
            return 'Papillon'
        return 'Inconnu'

    def determine_mount_type(self, filename: str) -> str:
        return 'Pleine'

    def import_all(self):
        print("📥 IMPORTATION DE TOUTES LES IMAGES")
        print("=" * 60)

        images = self.get_all_images()
        print(f"📸 {len(images)} images trouvées dans {self.source_dir}")

        print("\n📂 Répartition par dossier:")
        for folder, count in self.stats["by_source"].items():
            print(f"  {folder}: {count} images")

        print("\n" + "=" * 60)

        if self.csv_path.exists():
            try:
                df = pd.read_csv(self.csv_path)
                print(f"📊 Dataset existant: {len(df)} images")
            except Exception:
                df = pd.DataFrame(columns=['filename', 'shape', 'mount_type'])
                print("📊 Dataset existant illisible — nouveau dataset créé")
        else:
            df = pd.DataFrame(columns=['filename', 'shape', 'mount_type'])
            print("📊 Nouveau dataset créé")

        print("\n📁 Copie des images...")
        new_rows = []
        copied_count = 0

        for img_path in tqdm(images, desc="Importation"):
            try:
                base_name = img_path.stem
                suffix = img_path.suffix
                clean_name = "".join(c for c in base_name if c.isalnum() or c in '._-')
                if not clean_name:
                    clean_name = f"image_{copied_count+1:04d}"
                new_name = f"{clean_name}{suffix}"
                counter = 1
                while (self.target_dir / new_name).exists():
                    new_name = f"{clean_name}_{counter}{suffix}"
                    counter += 1
                dst_path = self.target_dir / new_name
                shutil.copy2(img_path, dst_path)

                shape = self.determine_shape(img_path.name)
                mount_type = self.determine_mount_type(img_path.name)

                new_rows.append({'filename': new_name, 'shape': shape, 'mount_type': mount_type})
                copied_count += 1
            except Exception as e:
                self.stats["errors"] += 1
                print(f"⚠️ Erreur sur {img_path}: {e}")

        new_df = pd.DataFrame(new_rows)
        if not new_df.empty:
            df = pd.concat([df, new_df], ignore_index=True)

        try:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.csv_path, index=False)
        except Exception as e:
            print(f"❌ Impossible d'écrire le CSV {self.csv_path}: {e}")

        self.stats["copied"] = copied_count
        self.stats["skipped"] = self.stats["total"] - copied_count - self.stats["errors"]

        self.print_stats(df)
        return copied_count

    def print_stats(self, df):
        print("\n" + "=" * 60)
        print("📊 STATISTIQUES FINALES")
        print("=" * 60)
        print(f"📸 Images dans {self.source_dir}: {self.stats['total']}")
        print(f"✅ Images copiées: {self.stats['copied']}")
        print(f"⏭️ Images ignorées: {self.stats['skipped']}")
        print(f"❌ Erreurs: {self.stats['errors']}")

        print(f"\n📊 Dataset total: {len(df)} images")

        print("\n📊 Distribution par forme:")
        if 'shape' in df.columns and not df.empty:
            shape_counts = df['shape'].value_counts()
            for shape, count in shape_counts.items():
                print(f"  {shape:15} {count:4} images")

        report_path = self.target_dir.parent / "import_report.json"
        try:
            with open(report_path, "w", encoding='utf8') as f:
                json.dump(self.stats, f, indent=2, ensure_ascii=False)
            print(f"\n📄 Rapport sauvegardé: {report_path}")
        except Exception as e:
            print(f"❌ Impossible d'écrire le rapport: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="data/datas")
    parser.add_argument("--target", type=str, default="classification_dataset/balanced_controlled")
    parser.add_argument("--csv", type=str, default="classification_dataset/annotations/annotations_complete.csv")
    args = parser.parse_args()

    importer = AllDataImporter(args.source, args.target, args.csv)
    count = importer.import_all()

    print(f"\n✅ Importation terminée: {count} images ajoutées!")


if __name__ == "__main__":
    main()
