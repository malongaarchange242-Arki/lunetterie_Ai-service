import logging
from pathlib import Path
from typing import Any

from app.ai import claude_vision
from app.ai.classifier import GlassesClassifier
from app.ai.detector import GlassesDetector
from app.ai.image_utils import crop_image, resize_image, save_image
from app.ai.shape_estimator import ShapeEstimator

logger = logging.getLogger(__name__)


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

        result = {
            "detected": detected,
            "confidence": round(confidence, 4),
            "frame_shape": classification["frame_shape"],
            "color": classification["color"],
            "material": classification["material"],
            "has_branches": classification["has_branches"],
            "mount_type": classification["mount_type"],
            "gender": None,
            "crop_path": crop_path,
            "product_fiche": {
                "name": "Monture à compléter",
                "description": "Fiche générée automatiquement à partir de l'analyse IA.",
                "brand": None,
                "reference": None,
            },
        }

        # Claude vision : plus fiable que le pipeline YOLO local sur forme/couleur/matière/genre.
        # Repli silencieux sur les résultats locaux ci-dessus si la clé API est absente, l'appel
        # échoue, ou que Claude ne renvoie rien d'exploitable pour un champ donné.
        try:
            claude_result = claude_vision.analyze_monture(crop_path or str(path))
        except Exception as exc:  # défensif: ne doit jamais faire échouer l'analyse locale
            logger.warning("Analyse Claude monture indisponible: %s", exc)
            claude_result = None

        if claude_result:
            if claude_result.get("shape"):
                result["frame_shape"] = claude_result["shape"]
            if claude_result.get("color"):
                result["color"] = claude_result["color"]
            if claude_result.get("material"):
                result["material"] = claude_result["material"]
            if claude_result.get("gender"):
                result["gender"] = claude_result["gender"]
            result["confidence"] = max(result["confidence"], claude_result.get("confidence") or 0.0)

        # OCR marque sur le verre : appel séparé, purement OCR (voir docstring de
        # ocr_monture_brand) — plus fiable que de mélanger la marque à la classification ci-dessus.
        try:
            brand_result = claude_vision.ocr_monture_brand(crop_path or str(path))
        except Exception as exc:  # défensif: ne doit jamais faire échouer l'analyse locale
            logger.warning("OCR marque monture indisponible: %s", exc)
            brand_result = None

        if brand_result and brand_result.get("brand"):
            result["product_fiche"]["brand"] = brand_result["brand"]

        return result
