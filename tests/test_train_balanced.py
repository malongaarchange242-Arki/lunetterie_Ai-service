from pathlib import Path
import sys

from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train_balanced


def _write_image(path: Path) -> None:
    image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    image.save(path)


def test_create_dataset_tree_skips_empty_class_dirs(tmp_path):
    source_dir = tmp_path / "source"
    (source_dir / "class_a").mkdir(parents=True)
    (source_dir / "class_b").mkdir(parents=True)
    (source_dir / "empty").mkdir(parents=True)

    _write_image(source_dir / "class_a" / "a.png")
    _write_image(source_dir / "class_b" / "b.png")

    prepared_dir = train_balanced.prepare_dataset_for_training(source_dir)

    assert (prepared_dir / "class_a" / "a.png").exists()
    assert (prepared_dir / "class_b" / "b.png").exists()
    assert not (prepared_dir / "empty").exists()
