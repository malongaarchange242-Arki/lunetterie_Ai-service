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

                model = efficientnet_b0(pretrained=False)
                checkpoint = torch.load(settings.MODEL_PATH_CLASSIFICATION, map_location=settings.DEVICE)
                if isinstance(checkpoint, dict):
                    state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
                    class_names = checkpoint.get("classes") or class_names
                    if isinstance(state, dict):
                        if all(isinstance(v, torch.Tensor) for v in state.values()):
                            num_classes = checkpoint.get("num_classes")
                            if num_classes is None:
                                for key, value in state.items():
                                    if key.endswith("classifier.1.weight") or key.endswith("classifier.weight"):
                                        num_classes = int(value.shape[0])
                                        break
                            if num_classes is None:
                                model = None
                            else:
                                model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, int(num_classes))
                                try:
                                    model.load_state_dict(state, strict=False)
                                except Exception:
                                    stripped = {key.split("model.", 1)[-1]: value for key, value in state.items()}
                                    model.load_state_dict(stripped, strict=False)
                        else:
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
    return class_names or ["Rectangle", "Rond", "Ovale", "Carré", "Papillon"]
