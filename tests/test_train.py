from pathlib import Path

from app.train import prepare_data_yaml


def test_prepare_data_yaml_writes_valid_yaml(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "images" / "train").mkdir(parents=True)
    (dataset_dir / "images" / "val").mkdir(parents=True)
    (dataset_dir / "labels" / "train").mkdir(parents=True)
    (dataset_dir / "labels" / "val").mkdir(parents=True)

    yaml_path = prepare_data_yaml(dataset_dir)

    assert yaml_path.exists()
    assert yaml_path.suffix == ".yaml"
    contents = yaml_path.read_text(encoding="utf-8")
    assert 'path:' in contents
    assert 'train:' in contents
    assert 'val:' in contents
    assert 'test:' in contents
