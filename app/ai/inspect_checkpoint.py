import torch
from pathlib import Path

model_path = Path('best_shape_model.pth')
if not model_path.exists():
    print('Checkpoint not found:', model_path)
    raise SystemExit(1)

ck = torch.load(model_path, map_location='cpu')
print('type:', type(ck))
if isinstance(ck, dict):
    print('keys:', list(ck.keys()))
    print('has classes:', 'classes' in ck)
    if 'classes' in ck:
        print('classes:', ck['classes'])
        print('num_classes:', len(ck['classes']))
    if 'model_state_dict' in ck:
        print('model_state_dict classifier weight shape:', next(v.shape for k, v in ck['model_state_dict'].items() if k.endswith('classifier.1.weight')))
else:
    print('checkpoint is not a dict')
