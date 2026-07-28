from pathlib import Path
from typing import Any

try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # graceful fallback when ultralytics isn't installed
    YOLO = None  # type: ignore


class GlassesDetector:
    def __init__(self, model_path: str | None = None, model: Any | None = None) -> None:
        default_path = model_path or str(Path("runs/detect/train-3/weights/best.pt"))
        if Path(default_path).exists():
            resolved_path = str(Path(default_path).resolve())
        else:
            resolved_path = str(Path(default_path))

        if model is not None:
            self.model = model
            self.model_path = resolved_path
        elif YOLO is None:
            self.model = None
            self.model_path = resolved_path
        else:
            self.model = YOLO(resolved_path)
            self.model_path = resolved_path

    def detect(self, image_path: str) -> list[dict[str, Any]]:
        if self.model is None:
            return []

        results = self.model(image_path, stream=False, conf=0.25)
        detections: list[dict[str, Any]] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                xyxy = getattr(box, "xyxy", None)
                if xyxy is None:
                    continue
                coords = xyxy[0].tolist()
                x1, y1, x2, y2 = coords
                conf = float(getattr(box, "conf")[0])
                cls = int(getattr(box, "cls")[0])
                label = result.names.get(cls, str(cls)) if hasattr(result, "names") else str(cls)
                detections.append(
                    {
                        "class_id": cls,
                        "class_name": label,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2],
                    }
                )
        return detections
