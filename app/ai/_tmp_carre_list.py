from pathlib import Path

carre_dir = Path('classification_dataset/balanced_controlled/Carrée')
print('carre_dir exists', carre_dir.exists())
print('carre_dir repr', repr(str(carre_dir)))
for i, p in enumerate(sorted(carre_dir.iterdir())):
    if i >= 10:
        break
    print(i, repr(p.name), len(p.name), repr(str(p)))
print('total', sum(1 for p in carre_dir.iterdir() if p.is_file()))
