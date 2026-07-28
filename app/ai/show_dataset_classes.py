from pathlib import Path
import pandas as pd

csv_path = Path('classification_dataset/annotations/annotations_complete.csv')
print('CSV exists:', csv_path.exists())
if csv_path.exists():
    df = pd.read_csv(csv_path)
    shapes = df['shape'].fillna('').astype(str)
    counts = shapes.value_counts()
    print('\nUnique shapes in CSV and counts:')
    for name, cnt in counts.items():
        print(f'  {name}: {cnt}')
else:
    print('CSV not found')

print('\nDataset folders under classification_dataset/balanced_controlled:')
base = Path('classification_dataset/balanced_controlled')
if base.exists():
    for p in sorted(base.iterdir()):
        if p.is_dir():
            cnt = len(list(p.glob('*')))
            print(f'  {p.name}: {cnt}')
else:
    print('Dataset folder not found')
