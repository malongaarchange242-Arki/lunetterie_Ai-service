"""Utilities to inspect and validate a YOLO-formatted dataset.

Usage:
    python -m app.dataset_utils data

It checks that images exist in `data/images/{train,val,test}` and that corresponding
labels exist in `data/labels/{train,val,test}` with normalized coordinates.
"""
from pathlib import Path
import sys
import re
from typing import Tuple


VALID_IMAGE_EXT = {'.jpg', '.jpeg', '.png'}


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return [p for p in folder.iterdir() if p.suffix.lower() in VALID_IMAGE_EXT]


def validate_label_file(label_path: Path) -> Tuple[bool, str]:
    # Each line: class x_center y_center width height (floats normalized 0..1)
    if not label_path.exists():
        return False, 'missing file'
    text = label_path.read_text(encoding='utf8').strip()
    if text == '':
        return False, 'empty file'
    for i, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            return False, f'line {i}: expected 5 values'
        cls = parts[0]
        if not re.fullmatch(r'\d+', cls):
            return False, f'line {i}: invalid class id'
        try:
            vals = [float(x) for x in parts[1:]]
        except ValueError:
            return False, f'line {i}: non-numeric values'
        if any(v < 0.0 or v > 1.0 for v in vals):
            return False, f'line {i}: coordinate(s) outside [0,1]'
    return True, 'ok'


def inspect(data_dir: Path) -> int:
    summary = {}
    errors = []
    for split in ('train', 'val', 'test'):
        imgs = list_images(data_dir / 'images' / split)
        summary[split] = {'images': len(imgs), 'labels_ok': 0, 'labels_missing': 0, 'labels_bad': 0}
        for img in imgs:
            label = (data_dir / 'labels' / split / img.with_suffix('.txt').name)
            ok, reason = validate_label_file(label)
            if ok:
                summary[split]['labels_ok'] += 1
            else:
                if reason == 'missing file':
                    summary[split]['labels_missing'] += 1
                else:
                    summary[split]['labels_bad'] += 1
                    errors.append(f'{split}/{img.name}: {reason}')

    # Print summary
    print('Dataset summary:')
    for split, stats in summary.items():
        print(f"  {split}: {stats['images']} images, {stats['labels_ok']} labels OK, {stats['labels_missing']} missing, {stats['labels_bad']} bad")
    if errors:
        print('\nExamples of annotation problems:')
        for e in errors[:20]:
            print('  -', e)
    return 0 if not errors and all(summary[s]['images'] == summary[s]['labels_ok'] for s in summary) else 2


def main(argv=None):
    argv = argv or sys.argv[1:]
    if not argv:
        print('Usage: python -m app.dataset_utils <data_dir>')
        return 1
    data_dir = Path(argv[0])
    if not data_dir.exists():
        print(f'Dataset folder not found: {data_dir}')
        return 1
    return inspect(data_dir)


if __name__ == '__main__':
    raise SystemExit(main())
