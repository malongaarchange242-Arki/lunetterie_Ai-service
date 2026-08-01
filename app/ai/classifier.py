# app/ai/classifier.py
from typing import Any
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np

from app.classification.auto_color import estimate_color
from app.classification.auto_mount import estimate_mount_type
from app.core.config import settings


class GlassesClassifier:
    def __init__(self, use_cnn: bool = True, model_path: str | None = None):
        self.use_cnn = use_cnn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Charger le modèle CNN
        if use_cnn:
            model_path = model_path or settings.MODEL_PATH_CLASSIFICATION
            self.model, self.classes = self._load_model(model_path)
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.model = None
            self.classes = []
        
        # Mapping des classes CNN vers les noms d'affichage (si besoin)
        self.display_names = {
            'Carrée': 'square',
            'Ovale': 'oval',
            'Papillon': 'papillon',
            'Pilote': 'aviator',
            'Rectangulaire': 'rectangle',
            'Ronde': 'round'
        }
    
    def _load_model(self, model_path):
        """
        Charge le modèle et les classes depuis le checkpoint.
        """
        path = Path(model_path)
        if not path.exists():
            print(f"⚠️ Modèle non trouvé: {model_path}, fallback sur heuristique")
            self.use_cnn = False
            return None, []
        
        checkpoint = torch.load(path, map_location=self.device)
        
        if 'classes' not in checkpoint:
            print("⚠️ Pas de classes dans le checkpoint")
            return None, []
        
        classes = checkpoint['classes']
        num_classes = len(classes)
        
        # Reconstruire le modèle
        import torchvision.models as models
        import torch.nn as nn
        
        model = models.efficientnet_b0()
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
        
        # Charger les poids
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        print(f"✅ Modèle chargé: {model_path}")
        print(f"📊 Classes: {classes}")
        
        return model, classes
    
    def _predict_cnn(self, image_path: str) -> dict:
        """
        Prédiction avec le modèle CNN.
        """
        if self.model is None:
            return None
        
        try:
            # Charger et transformer l'image
            image = Image.open(image_path).convert('RGB')
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Prédiction
            with torch.no_grad():
                outputs = self.model(tensor)
                probs = torch.softmax(outputs, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_idx].item()
            
            shape = self.classes[pred_idx]
            display_name = self.display_names.get(shape, shape.lower())
            
            # Toutes les probabilités
            all_probs = {
                self.display_names.get(cls, cls.lower()): probs[0, i].item()
                for i, cls in enumerate(self.classes)
            }
            
            return {
                "shape": display_name,
                "confidence": confidence,
                "probabilities": all_probs
            }
            
        except Exception as e:
            print(f"⚠️ Erreur CNN: {e}")
            return None
    
    def _predict_heuristic(self, image_path: str) -> dict:
        """
        Prédiction heuristique (fallback).
        """
        try:
            from PIL import Image
            img = Image.open(image_path)
            width, height = img.size
            aspect_ratio = width / max(height, 1)
            
            if aspect_ratio > 1.8:
                shape = "aviator"
            elif aspect_ratio > 1.2:
                shape = "rectangle"
            elif aspect_ratio > 0.9:
                shape = "round"
            else:
                shape = "oval"
            
            return {
                "shape": shape,
                "confidence": 0.5,
                "probabilities": {}
            }
        except:
            return {
                "shape": "unknown",
                "confidence": 0.0,
                "probabilities": {}
            }

    def classify(self, image_path: str) -> dict[str, Any]:
        """
        Analyse une image de lunettes.
        """
        # 1. Forme
        if self.use_cnn:
            shape_result = self._predict_cnn(image_path)
            if shape_result is None:
                shape_result = self._predict_heuristic(image_path)
        else:
            shape_result = self._predict_heuristic(image_path)
        
        # 2. Couleur
        try:
            color = estimate_color(image_path)
        except:
            color = "unknown"
        
        # 3. Type de monture
        try:
            mount_type = estimate_mount_type(image_path)
        except:
            mount_type = "unknown"
        
        return {
            "frame_shape": shape_result.get("shape", "unknown"),
            "shape_confidence": shape_result.get("confidence", 0.0),
            "color": color,
            "material": "unknown",
            "has_branches": True,
            "mount_type": mount_type,
            "probabilities": shape_result.get("probabilities", {})
        }