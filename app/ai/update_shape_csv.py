from pathlib import Path
import pandas as pd


def update_shape_csv():
    train_dir = Path("classification_dataset/forme/train")
    csv_path = Path("classification_dataset/annotations/annotations_shape.csv")

    folder_to_shape = {
        "aviateur": "Pilote",
        "carrée": "Carrée",
        "ovale": "Ovale",
        "papillon": "Papillon",
        "rectangulaire": "Rectangulaire",
        "ronde": "Ronde"
    }

    rows = []
    counts = {}
    for folder in sorted(train_dir.iterdir()):
        if folder.is_dir() and folder.name in folder_to_shape:
            shape = folder_to_shape[folder.name]
            imgs = [p for p in folder.iterdir() if p.is_file()]
            counts[shape] = len(imgs)
            for img_path in imgs:
                relative_path = img_path.relative_to(train_dir.parent.parent)
                rows.append({
                    'filename': img_path.name,
                    'shape': shape,
                    'path': str(relative_path.as_posix())
                })

    df = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"✅ {len(df)} images enregistrées dans {csv_path}")
    print('\n📊 Distribution:')
    print(pd.Series(counts).sort_index())


if __name__ == "__main__":
    update_shape_csv()
