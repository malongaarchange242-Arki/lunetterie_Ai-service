import tempfile
import time
from typing import Any

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage

from app.core.config import settings
from app.core.model_loader import get_class_names, get_classification_model, get_detection_model
from app.inference.auto_attributes import build_frame_mask, get_color, get_material, get_mount_type
from app.inference.detector import GlassesDetector


def _extract_glasses_crop(image: np.ndarray, boxes: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, dict[str, Any] | None]:
    if not boxes:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return rgb_image, image, None

    best_box = max(boxes, key=lambda item: item.get("confidence", 0.0))
    bbox = best_box.get("bbox", [0, 0, image.shape[1], image.shape[0]])
    x1, y1, x2, y2 = [round(float(v)) for v in bbox]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))

    h = max(1, y2 - y1)
    w = max(1, x2 - x1)
    margin_x = int(w * 0.03)
    margin_y = int(h * 0.03)

    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(image.shape[1], x2 + margin_x)
    y2 = min(image.shape[0], y2 + margin_y)

    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        cropped = image

    rgb_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    return rgb_image, cropped, best_box


def _save_temp_rgb_image(rgb_image: np.ndarray) -> str:
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    temp_path = temp_file.name
    temp_file.close()
    PILImage.fromarray(rgb_image).save(temp_path)
    return temp_path


def analyze_image(image_bytes: bytes) -> dict[str, Any]:
    start_time = time.time()

    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        image = None

    if image is None:
        image = np.zeros((224, 224, 3), dtype=np.uint8)

    detection_model = get_detection_model()
    boxes = []
    if detection_model is not None:
        source_temp_path = ""
        try:
            source_temp_path = _save_temp_rgb_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            detector = GlassesDetector(settings.MODEL_PATH_DETECTION, model=detection_model)
            boxes = detector.detect(source_temp_path)
        except Exception:
            boxes = []
        finally:
            if source_temp_path:
                try:
                    Path(source_temp_path).unlink(missing_ok=True)
                except Exception:
                    pass

    rgb_image, cropped_image, best_box = _extract_glasses_crop(image, boxes)
    if not boxes:
        pil_image = PILImage.fromarray(rgb_image)
        width, height = pil_image.size
        if width > 0 and height > 0:
            aspect_ratio = width / max(height, 1)
            shape = "rectangle" if aspect_ratio > 1.2 else "round" if aspect_ratio > 0.9 else "oval"
        else:
            shape = "unknown"
    else:
        shape = "unknown"

    shape_confidence = 0.75
    if get_classification_model() is not None:
        temp_path = ""
        try:
            from app.ai.train_shape_classifier import predict_shape

            temp_path = _save_temp_rgb_image(rgb_image)
            prediction = predict_shape(temp_path, model_path=settings.MODEL_PATH_CLASSIFICATION, class_names=get_class_names())
            shape = prediction.get("shape", shape)
            shape_confidence = float(prediction.get("confidence", 0.75)) / 100.0
        except Exception:
            shape_confidence = 0.75
        finally:
            if temp_path:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except Exception:
                    pass

    frame_mask = build_frame_mask(rgb_image)
    color, color_conf = get_color(rgb_image, frame_mask)
    material, material_conf = get_material(rgb_image)
    mount_type, mount_conf = get_mount_type(rgb_image, frame_mask)

    processing_time = (time.time() - start_time) * 1000

    return {
        "shape": shape,
        "shape_confidence": round(shape_confidence, 2),
        "color": color,
        "color_confidence": round(color_conf, 2),
        "material": material,
        "material_confidence": round(material_conf, 2),
        "mount_type": mount_type,
        "mount_type_confidence": round(mount_conf, 2),
        "gender": None,
        "gender_confidence": None,
        "brand": None,
        "brand_confidence": None,
        "reference": None,
        "reference_confidence": None,
        "size": None,
        "model_version": settings.MODEL_VERSION,
        "processing_time_ms": round(processing_time, 2),
    }
