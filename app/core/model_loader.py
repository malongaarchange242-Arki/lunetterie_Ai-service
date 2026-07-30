try:
    import torch
except Exception:  # pragma: no cover - fallback if torch is unavailable
    torch = None

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - fallback if ultralytics is unavailable
    YOLO = None

from app.core.config import settings


detection_model = None
classification_model = None
class_names = None


def get_detection_model():
    global detection_model
    if detection_model is None:
        if YOLO is None:
            detection_model = None
        else:
            detection_model = YOLO(settings.MODEL_PATH_DETECTION)
    return detection_model


def get_classification_model():
    global classification_model, class_names
    if classification_model is None:
        if torch is None:
            classification_model = None
        else:
            try:
                from torchvision.models import efficientnet_b0
                from app.ai.train_shape_classifier import _rebuild_classifier_head

                model = efficientnet_b0(pretrained=False)
                checkpoint = torch.load(settings.MODEL_PATH_CLASSIFICATION, map_location=settings.DEVICE)
                if isinstance(checkpoint, dict):
                    state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
                    class_names = checkpoint.get("classes") or checkpoint.get("class_names") or class_names
                    if isinstance(state, dict) and all(isinstance(v, torch.Tensor) for v in state.values()):
                        try:
                            _rebuild_classifier_head(model, state)
                            model.load_state_dict(state, strict=False)
                        except Exception:
                            model = None
                    else:
                        model = None
                else:
                    model = None
                if model is not None:
                    model.eval()
                    classification_model = model
            except Exception:
                classification_model = None
    return classification_model


def get_class_names():
    global class_names
    if class_names is None:
        get_classification_model()
    return class_names or ["aviateur", "carree", "ovale", "papillon", "rectangulaire", "ronde"]
