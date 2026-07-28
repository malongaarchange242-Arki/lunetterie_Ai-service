import pandas as pd
from pathlib import Path


def clean_carre_folder():
    """Nettoie le dossier Carrée en supprimant les entrées CSV sans fichiers."""
    print("🔧 NETTOYAGE DU DOSSIER CARRÉE")
    print("=" * 60)

    carre_dir = Path("classification_dataset/balanced_controlled/Carrée")
    csv_path = Path("classification_dataset/annotations/annotations_complete.csv")

    print(f"📁 Vérification du dossier {carre_dir}")

    if not carre_dir.exists():
        print("❌ Le dossier Carrée n'existe pas !")
        return

    real_files = set()
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        real_files.update({f.name for f in carre_dir.glob(ext)})

    print(f"📸 Fichiers réels trouvés: {len(real_files)}")

    if not csv_path.exists():
        print("❌ CSV non trouvé !")
        return

    df = pd.read_csv(csv_path)
    print(f"📊 Entrées dans le CSV: {len(df)}")

    carre_df = df[df['shape'] == 'Carrée']
    print(f"📊 Entrées Carrée dans le CSV: {len(carre_df)}")

    df_cleaned = df[~((df['shape'] == 'Carrée') & (~df['filename'].isin(real_files)))]

    removed = len(df) - len(df_cleaned)
    print(f"\n🗑️ {removed} entrées orphelines supprimées du CSV")

    df_cleaned.to_csv(csv_path, index=False)
    print(f"✅ CSV mis à jour: {len(df_cleaned)} entrées")

    final_carre = df_cleaned[df_cleaned['shape'] == 'Carrée']
    print(f"\n📊 Nouvelles stats Carrée:")
    print(f"  Fichiers réels: {len(real_files)}")
    print(f"  Entrées CSV: {len(final_carre)}")

    return len(real_files)


if __name__ == "__main__":
    clean_carre_folder()
