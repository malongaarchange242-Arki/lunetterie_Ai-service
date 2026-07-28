from __future__ import annotations

import torch
from pathlib import Path

MODEL = Path('best_shape_model.pth')

if not MODEL.exists():
    print('❌ Model not found:', MODEL)
    raise SystemExit(1)

ck = torch.load(MODEL, map_location='cpu')

print('Checkpoint type:', type(ck))

# If outer dict contains a state_dict, inspect it
state = None
if isinstance(ck, dict):
    keys = list(ck.keys())
    print('\nTop-level keys in checkpoint:')
    for k in keys:
        print('  ', k)
    if 'classes' in ck:
        print('\nClasses stored in checkpoint:')
        for i, c in enumerate(ck['classes']):
            print(f'  {i}: {c}')
    if 'state_dict' in ck:
        state = ck['state_dict']
    elif 'model_state_dict' in ck:
        state = ck['model_state_dict']
    else:
        # maybe the checkpoint is already a state_dict
        # check if values are tensors
        sample = next(iter(ck.values()))
        if hasattr(sample, 'shape'):
            state = ck

if state is None:
    print('\nNo state_dict found to inspect classifier weights.')
    raise SystemExit(0)

# Inspect classifier weight keys and shape
print('\nInspecting state_dict keys for classifier weights...')
classifier_keys = [k for k in state.keys() if 'classifier' in k and 'weight' in k]
if not classifier_keys:
    print('  No classifier weight keys found. Sample keys:')
    for k in list(state.keys())[:20]:
        print('   ', k)
else:
    for k in classifier_keys:
        print('  ', k, '->', getattr(state[k], 'shape', 'n/a'))

# Derive num classes from classifier weight if possible
if classifier_keys:
    w = state[classifier_keys[0]]
    try:
        num_classes = int(w.shape[0])
        print(f'\nDerived num_classes = {num_classes} from {classifier_keys[0]}')
    except Exception as e:
        print('Could not derive num_classes:', e)

print('\nDone')
