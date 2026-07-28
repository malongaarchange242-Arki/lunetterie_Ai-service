from pathlib import Path
import pandas as pd

p = Path('classification_dataset/annotations/annotations_shape.csv')
if not p.exists():
    print('CSV not found:', p)
    raise SystemExit(1)

df = pd.read_csv(p)
res = df[df['shape'] == 'Pilote']
print('Found', len(res), 'rows for Pilote\n')
print(res.head(20).to_string(index=False))
