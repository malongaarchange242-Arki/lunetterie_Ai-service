import pandas as pd
from pathlib import Path


def sync_carre_csv():
    """Synchronise le CSV avec les fichiers réellement présents dans le dossier Carrée."""
    print("🔄 SYNCHRONISATION DU CSV AVEC LE DOSSIER CARRÉE")
    print("=" * 60)

    carre_dir = Path("classification_dataset/balanced_controlled/Carrée")
    csv_path = Path("classification_dataset/annotations/annotations_complete.csv")

    real_files = set()
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        real_files.update({f.name for f in carre_dir.glob(ext)})

    print(f"📸 Fichiers réels dans Carrée: {len(real_files)}")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV non trouvé : {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"📊 Entrées totales dans le CSV: {len(df)}")

    carre_in_csv = set(df[df['shape'] == 'Carrée']['filename'])
    print(f"📊 Entrées Carrée dans le CSV: {len(carre_in_csv)}")

    missing = sorted(real_files - carre_in_csv)
    print(f"\n🔍 Fichiers manquants dans le CSV: {len(missing)}")

    if missing:
        print("\n📝 Ajout des fichiers manquants:")
        new_rows = []
        for filename in missing:
            print(f"  - {filename}")
            new_rows.append({
                'filename': filename,
                'shape': 'Carrée',
                'mount_type': 'Pleine'
            })
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(csv_path, index=False)
        print(f"\n✅ {len(missing)} fichiers ajoutés au CSV")
    else:
        print("\n✅ Aucun fichier manquant à ajouter.")

    final_carre = len(df[df['shape'] == 'Carrée'])
    print(f"\n📊 Nouvelles stats:")
    print(f"  Carrée dans le CSV: {final_carre}")
    print(f"  Fichiers réels: {len(real_files)}")
    print(f"  Dataset total: {len(df)}")


if __name__ == "__main__":
    sync_carre_csv()
