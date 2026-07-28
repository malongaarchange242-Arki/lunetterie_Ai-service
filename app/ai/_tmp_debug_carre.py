from pathlib import Path
import cv2
import numpy as np

p = Path('classification_dataset/balanced_controlled/Carrée/30_jpg.rf.t6cDM9GBEoqzLn10f5tw_24958.jpg')
print('exists', p.exists())
if p.exists():
    print('path', p)
    try:
        data = p.read_bytes()
        print('bytes', len(data))
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        print('img', img is None)
        if img is not None:
            print('shape', img.shape)
    except Exception as e:
        print('exception', type(e).__name__, e)
