from __future__ import annotations

import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ai.train_shape_classifier import predict_shape

image = Path(r'D:\Pojet la lunetterie\test.png')
if not image.exists():
    print('MISSING IMAGE', image)
    raise SystemExit(1)

res = predict_shape(str(image))
print(json.dumps(res, ensure_ascii=False, indent=2))
