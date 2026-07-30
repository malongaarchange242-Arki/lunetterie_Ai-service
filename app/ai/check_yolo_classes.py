# app/ai/check_yolo_classes.py
from ultralytics import YOLO
from pathlib import Path

def check_yolo_model(model_path="ai-service/best.pt"):
    """
    Vérifie les classes du modèle YOLO.
    """
    print(f"🔍 Vérification du modèle: {model_path}")
    print("=" * 60)
    
    if not Path(model_path).exists():
        print("❌ Modèle non trouvé")
        return
    
    # Charger le modèle
    model = YOLO(model_path)
    
    # Afficher les classes
    if hasattr(model, 'names'):
        print(f"\n📊 Classes du modèle ({len(model.names)}):")
        for idx, name in model.names.items():
            print(f"  {idx}: {name}")
    else:
        print("⚠️ Impossible de lire les classes du modèle")
    
    # Informations sur le modèle
    print(f"\n📊 Informations:")
    print(f"  Modèle: {model_path}")
    print(f"  Taille: {Path(model_path).stat().st_size / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    # Vérifier best.pt
    check_yolo_model("ai-service/best.pt")
    
    print("\n" + "=" * 60)
    
    # Vérifier last.pt
    check_yolo_model("ai-service/last.pt")