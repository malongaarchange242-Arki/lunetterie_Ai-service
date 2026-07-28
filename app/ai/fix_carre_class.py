from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def _windows_long_path(path: Path) -> str:
    path_str = os.path.abspath(str(path))
    if os.name == 'nt' and not path_str.startswith('\\\\?\\'):
        return '\\?\\' + path_str
    return path_str


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(_windows_long_path(src), 'rb') as fsrc, open(_windows_long_path(dst), 'wb') as fdst:
        shutil.copyfileobj(fsrc, fdst)


def _is_image_valid(file_path: Path) -> bool:
    try:
        with open(str(file_path), 'rb') as f:
            data = f.read()
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        return image is not None and image.size > 0
    except Exception:
        return False


def fix_carre_class(dataset_dir: str = "classification_dataset/balanced_controlled",
                    csv_path: str = "classification_dataset/annotations/annotations_complete.csv") -> int:
    """Nettoie la classe Carrée : supprime les fichiers illisibles et remet le CSV en cohérence."""
    print("🔧 RÉPARATION DE LA CLASSE CARRÉE")
    print("=" * 60)

    carre_dir = Path(dataset_dir) / "Carrée"
    backup_dir = Path(dataset_dir) / "Carrée_backup"
    csv_path = Path(csv_path)

    if not carre_dir.exists():
        print(f"❌ Le dossier {carre_dir} n'existe pas !")
        return 0

    # 1. Sauvegarder le dossier actuel
    print(f"📁 Sauvegarde du dossier Carrée dans {backup_dir}")
    for file_path in sorted(carre_dir.iterdir()):
        if not file_path.is_file():
            continue
        try:
            _copy_file(file_path, backup_dir / file_path.name)
        except Exception as exc:
            print(f"⚠️ Impossible de copier {file_path.name}: {exc}")

    # 2. Vérifier chaque fichier
    print("\n🔍 Vérification des fichiers...")
    valid_files: list[Path] = []
    corrupted_files: list[Path] = []

    for file_path in sorted(carre_dir.iterdir()):
        if not file_path.is_file():
            continue
        if _is_image_valid(file_path):
            valid_files.append(file_path)
        else:
            corrupted_files.append(file_path)
            try:
                file_path.unlink()
            except OSError:
                print(f"⚠️ Impossible de supprimer {file_path.name}")

    print(f"✅ {len(valid_files)} fichiers valides")
    print(f"🗑️ {len(corrupted_files)} fichiers corrompus supprimés")

    # 3. Mettre à jour le CSV
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        valid_names = {path.name for path in valid_files}

        df_filtered = df[~((df["shape"] == "Carrée") & (~df["filename"].isin(valid_names)))]

        missing_in_csv = valid_names - set(df_filtered[df_filtered["shape"] == "Carrée"]["filename"])
        if missing_in_csv:
            print(f"\n📝 Ajout de {len(missing_in_csv)} fichiers manquants dans le CSV")
            new_rows = [
                {
                    "filename": name,
                    "shape": "Carrée",
                    "mount_type": "Pleine"
                }
                for name in sorted(missing_in_csv)
            ]
            df_filtered = pd.concat([df_filtered, pd.DataFrame(new_rows)], ignore_index=True)

        df_filtered.to_csv(csv_path, index=False)
        print(f"✅ CSV mis à jour: {len(df_filtered)} entrées")
    else:
        print(f"❌ CSV introuvable: {csv_path}")

    final_count = len(valid_files)
    print(f"\n📊 Classe Carrée: {final_count} images valides")
    return final_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Nettoie la classe Carrée et remet le CSV en cohérence.")
    parser.add_argument("--dataset", type=str, default="classification_dataset/balanced_controlled")
    parser.add_argument("--csv", type=str, default="classification_dataset/annotations/annotations_complete.csv")
    args = parser.parse_args()

    fix_carre_class(args.dataset, args.csv)


if __name__ == "__main__":
    main()
