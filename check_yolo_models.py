# check_yolo_models.py
# Placer ce fichier dans D:\Pojet la lunetterie\

from ultralytics import YOLO
from pathlib import Path
import torch

def check_model(model_path):
    print(f"\n🔍 Vérification: {model_path}")
    full_path = Path(model_path)
    
    if not full_path.exists():
        print(f"❌ Fichier non trouvé: {full_path}")
        print(f"   Vérifie le chemin: {full_path.absolute()}")
        return
    
    print(f"✅ Fichier trouvé: {full_path.name}")
    print(f"📦 Taille: {full_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    try:
        # Essayer de charger avec YOLO
        model = YOLO(str(full_path))
        print(f"📊 Classes ({len(model.names)}):")
        for idx, name in model.names.items():
            print(f"  {idx}: {name}")
    except Exception as e:
        print(f"⚠️ Erreur YOLO: {e}")
        
        # Essayer avec PyTorch
        try:
            checkpoint = torch.load(full_path, map_location='cpu')
            if 'model' in checkpoint:
                print("📊 Structure: checkpoint avec 'model'")
            elif 'model_state_dict' in checkpoint:
                print("📊 Structure: checkpoint avec 'model_state_dict'")
            else:
                print(f"📊 Clés disponibles: {list(checkpoint.keys())[:5]}")
        except Exception as e2:
            print(f"⚠️ Erreur PyTorch: {e2}")

if __name__ == "__main__":
    print("=" * 60)
    print("🔍 VÉRIFICATION DES MODÈLES YOLO")
    print("=" * 60)
    
    # Dossier où chercher les modèles
    base_dir = Path("ai-service")
    
    # Vérifier best.pt
    check_model(base_dir / "best.pt")
    
    print("\n" + "-" * 40)
    
    # Vérifier last.pt
    check_model(base_dir / "last.pt")
    
    print("\n" + "=" * 60)
    print("🔍 Vérification du dossier ai-service/")
    print("=" * 60)
    
    if base_dir.exists():
        files = list(base_dir.glob("*.pt"))
        print(f"📁 Fichiers .pt trouvés ({len(files)}):")
        for f in files:
            print(f"  - {f.name} ({f.stat().st_size / 1024 / 1024:.2f} MB)")
    else:
        print(f"❌ Dossier non trouvé: {base_dir}")