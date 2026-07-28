from pathlib import Path
from typing import Any

from app.ai.classifier import GlassesClassifier
from app.ai.detector import GlassesDetector
from app.ai.image_utils import crop_image, resize_image, save_image
from app.ai.shape_estimator import ShapeEstimator


class GlassesPredictor:
    def __init__(
        self,
        detector: GlassesDetector | None = None,
        classifier: GlassesClassifier | None = None,
        shape_estimator: ShapeEstimator | None = None,
    ) -> None:
        self.detector = detector or GlassesDetector()
        self.classifier = classifier or GlassesClassifier()
        self.shape_estimator = shape_estimator or ShapeEstimator()

    def predict_image(self, image_path: str) -> dict[str, Any]:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image introuvable: {image_path}")

        detections = self.detector.detect(str(path))
        detected = bool(detections)
        crop_path: str | None = None
        if detected:
            bbox = detections[0]["bbox"]
            crop = crop_image(str(path), bbox)
            resized = resize_image(crop)
            temp_output = path.with_suffix(".crop.jpg")
            save_image(resized, str(temp_output))
            crop_path = str(temp_output)

        classification = self.classifier.classify(str(path))
        shape_result = self.shape_estimator.estimate(str(path))
        frame_shape = shape_result.get("shape", classification["frame_shape"])
        if frame_shape != "unknown":
            classification["frame_shape"] = frame_shape
        confidence = max((item["confidence"] for item in detections), default=0.0)
        confidence = max(confidence, shape_result.get("confidence", 0.0))

        return {
            "detected": detected,
            "confidence": round(confidence, 4),
            "frame_shape": classification["frame_shape"],
            "color": classification["color"],
            "material": classification["material"],
            "has_branches": classification["has_branches"],
            "mount_type": classification["mount_type"],
            "crop_path": crop_path,
            "product_fiche": {
                "name": "Monture à compléter",
                "description": "Fiche générée automatiquement à partir de l'analyse IA.",
                "brand": None,
                "reference": None,
            },
        }
