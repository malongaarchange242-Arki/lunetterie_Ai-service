from pathlib import Path
import pandas as pd

carre_dir = Path('classification_dataset/balanced_controlled/Carrée')
print('carre exists', carre_dir.exists())
files = sorted([p.name for p in carre_dir.iterdir() if p.is_file() and p.suffix.lower() in ['.jpg', '.png', '.jpeg']])
print('real count', len(files))
print(files[:20])

csv_path = Path('classification_dataset/annotations/annotations_complete.csv')
print('csv exists', csv_path.exists())
if csv_path.exists():
    df = pd.read_csv(csv_path)
    print('csv total', len(df))
    carre = df[df['shape'] == 'Carrée']
    print('csv carre count', len(carre))
    print(carre.head(20).to_string(index=False))
else:
    print('csv missing')
