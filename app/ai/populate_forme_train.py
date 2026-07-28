from pathlib import Path
import shutil


def populate_forme_train():
    """
    Copie les images depuis balanced_controlled vers forme/train.
    """
    source_base = Path("classification_dataset/balanced_controlled")
    target_base = Path("classification_dataset/forme/train")
    
    # Mapping des classes
    class_mapping = {
        "Carrée": "carrée",
        "Ovale": "ovale",
        "Papillon": "papillon",
        "Pilote": "aviateur",  # Pilote → Aviateur
        "Rectangulaire": "rectangulaire",
        "Ronde": "ronde"
    }
    
    print("📁 Copie des images vers forme/train/")
    print("=" * 60)
    
    for source_class, target_class in class_mapping.items():
        source_dir = source_base / source_class
        target_dir = target_base / target_class
        
        if not source_dir.exists():
            print(f"⚠️ Source non trouvée: {source_dir}")
            continue
        
        target_dir.mkdir(parents=True, exist_ok=True)
        
        images = [p for p in source_dir.iterdir() if p.is_file()]
        copied = 0
        for img_path in images:
            try:
                shutil.copy2(img_path, target_dir / img_path.name)
                copied += 1
            except Exception as e:
                print(f"Erreur en copiant {img_path}: {e}")
        
        print(f"✅ {source_class} → {target_class}: {copied} images")


if __name__ == "__main__":
    populate_forme_train()
