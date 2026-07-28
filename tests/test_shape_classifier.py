from app.ai.train_shape_classifier import compute_class_weights, merge_shape_prediction


def test_compute_class_weights_upweights_rare_classes() -> None:
    weights = compute_class_weights(["Ronde", "Ronde", "Ronde", "Pilote", "Pilote", "Ovale"], class_names=["Ronde", "Pilote", "Ovale"])

    assert weights[1] > weights[0]
    assert weights[2] > weights[0]


def test_merge_shape_prediction_prefers_heuristic_when_model_is_uncertain() -> None:
    model_probabilities = {
        "Carrée": 0.55,
        "Ovale": 0.05,
        "Papillon": 0.03,
        "Pilote": 0.20,
        "Rectangulaire": 0.17,
    }

    merged = merge_shape_prediction(model_probabilities, {"shape": "aviator", "confidence": 0.82})

    assert merged["shape"] == "Pilote"
    assert merged["confidence"] >= 80.0
