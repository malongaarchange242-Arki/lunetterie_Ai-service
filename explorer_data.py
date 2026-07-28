from pathlib import Path

p = Path('data/datas')
print('📁', p)
print('='*60)
for item in sorted(p.iterdir()):
    if item.is_dir():
        imgs = list(item.glob('*.jpg')) + list(item.glob('*.png')) + list(item.glob('*.jpeg'))
        print(f'📂 {item.name}/ - {len(imgs)} images')
    else:
        print('📄', item.name)

print('\n' + '='*60)
print('📸 Exemples:')
for item in sorted(p.iterdir()):
    if item.is_dir():
        imgs = list(item.glob('*.jpg')) + list(item.glob('*.png'))
        if imgs:
            print(f'\n📂 {item.name}/')
            for img in imgs[:3]:
                print('  -', img.name)
            if len(imgs) > 3:
                print('  ... et', len(imgs)-3, 'autres')
