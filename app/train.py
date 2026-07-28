import argparse
import tempfile
from pathlib import Path

try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # graceful fallback when ultralytics isn't available
    YOLO = None  # type: ignore


def prepare_data_yaml(data_dir: Path, names: list[str] | None = None) -> Path:
    resolved_data_dir = data_dir.resolve()
    train_dir = resolved_data_dir / 'train' / 'images'
    val_dir = resolved_data_dir / 'valid' / 'images'
    test_dir = resolved_data_dir / 'test' / 'images'

    yaml_lines = [
        f'path: {resolved_data_dir}',
        '',
        f'train: {train_dir}',
        f'val: {val_dir}',
        f'test: {test_dir}',
        '',
        'nc: 1',
        '',
        'names:',
        '  0: glasses',
    ]
    yaml = '\n'.join(yaml_lines) + '\n'

    _, tmp_path = tempfile.mkstemp(suffix='.yaml')
    p = Path(tmp_path)
    p.write_text(yaml, encoding='utf-8')
    return p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', '-d', type=Path, default=Path('data'))
    parser.add_argument('--weights', '-w', default='yolov8n.pt')
    parser.add_argument('--epochs', '-e', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir.exists():
        print(f"Dataset introuvable: {data_dir.resolve()}\nVeuillez préparer le dataset avec la structure suivante:\n  {data_dir}/images/train  {data_dir}/images/val\n  {data_dir}/labels/train  {data_dir}/labels/val")
        return

    yaml_path = prepare_data_yaml(data_dir)
    print(f"Fichier dataset YAML généré: {yaml_path}")

    if YOLO is None:
        print('"ultralytics" non installé. Pour entraîner, installez ultralytics puis lancez:')
        print('  pip install ultralytics')
        print('Puis exécutez:')
        print(f"  python -m ultralytics train model={args.weights} data={yaml_path} epochs={args.epochs} imgsz={args.imgsz} batch={args.batch}")
        return

    # Run training
    model = YOLO(args.weights)
    print('Démarrage de l\'entraînement...')
    model.train(data=str(yaml_path), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)


if __name__ == '__main__':
    main()
