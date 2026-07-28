from pathlib import Path
import pandas as pd
from collections import Counter

base = Path('classification_dataset/balanced_controlled')
print('base exists', base.exists())
files = [p for p in base.rglob('*') if p.is_file() and p.suffix.lower() in ['.jpg', '.png', '.jpeg']]
counts = Counter(p.parent.name for p in files)
print('file counts by class:')
for cls in ['Carrée', 'Ovale', 'Pilote', 'Papillon', 'Rectangulaire', 'Ronde']:
    print(f'  {cls}: {counts.get(cls,0)}')

csv_path = Path('classification_dataset/annotations/annotations_complete.csv')
print('csv exists', csv_path.exists())
if csv_path.exists():
    df = pd.read_csv(csv_path)
    print('csv total', len(df))
    print('csv Carrée', len(df[df['shape']=='Carrée']))
    print('csv Carrée sample:')
    print(df[df['shape']=='Carrée'].head(10).to_string(index=False))
else:
    print('csv missing')
