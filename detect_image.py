"""
Script de test de détection d'image de monture
Usage: python detect_image.py <chemin_image>
Exemple: python detect_image.py "Capture d'écran 2026-07-26 194736.png"
"""

import sys
import requests
import json
import os
from pathlib import Path

# Configuration
API_URL = "http://localhost:8000"
ANALYZE_ENDPOINT = f"{API_URL}/analyze"
HEALTH_ENDPOINT = f"{API_URL}/health"

def check_health():
    """Vérifie que le service IA est en ligne"""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Service IA en ligne - Version: {data.get('model_version', 'N/A')}")
            return True
        else:
            print(f"❌ Service IA inaccessible - Status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Service IA non démarré. Lancez: uvicorn app.api.main:app --host 0.0.0.0 --port 8000")
        return False

def detect_glasses(image_path: str):
    """Envoie une image au service IA pour analyse"""
    
    # Vérifier que le fichier existe
    if not os.path.exists(image_path):
        print(f"❌ Fichier introuvable: {image_path}")
        return None
    
    # Vérifier l'extension
    ext = Path(image_path).suffix.lower()
    if ext not in ['.png', '.jpg', '.jpeg', '.webp', '.bmp']:
        print(f"⚠️  Extension non standard: {ext}. L'API accepte PNG, JPG, JPEG, WEBP, BMP")
    
    # Envoyer l'image
    print(f"\n📸 Analyse de: {os.path.basename(image_path)}")
    print(f"   Taille: {os.path.getsize(image_path) / 1024:.1f} Ko")
    print("   Envoi en cours...")
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, f'image/{ext[1:]}')}
            response = requests.post(ANALYZE_ENDPOINT, files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Analyse réussie en {result.get('processing_time_ms', 0):.0f} ms\n")
            return result
        else:
            print(f"   ❌ Erreur API: {response.status_code}")
            print(f"   {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("   ❌ Timeout: L'analyse a pris trop de temps")
        return None
    except Exception as e:
        print(f"   ❌ Erreur: {str(e)}")
        return None

def display_result(result: dict):
    """Affiche les résultats de manière lisible"""
    
    if not result:
        return
    
    print("=" * 50)
    print("  RÉSULTAT DE L'ANALYSE")
    print("=" * 50)
    
    # Caractéristiques principales
    fields = [
        ("Forme", "shape", "shape_confidence", "%"),
        ("Couleur", "color", "color_confidence", "%"),
        ("Matériau", "material", "material_confidence", "%"),
        ("Type monture", "mount_type", "mount_type_confidence", "%"),
        ("Genre", "gender", "gender_confidence", "%"),
        ("Marque", "brand", "brand_confidence", "%"),
        ("Référence", "reference", "reference_confidence", "%"),
    ]
    
    for label, key, conf_key, unit in fields:
        value = result.get(key)
        confidence = result.get(conf_key)
        if value:
            if confidence is not None and unit == "%" and confidence <= 1:
                confidence = confidence * 100
            conf_str = f"({confidence:.1f}{unit})" if confidence is not None else ""
            print(f"  📍 {label:<15}: {value:<20} {conf_str}")
    
    print("-" * 50)
    print(f"  ⏱️  Temps de traitement: {result.get('processing_time_ms', 0):.0f} ms")
    print(f"  🤖 Version modèle: {result.get('model_version', 'N/A')}")
    print("=" * 50)

def batch_analyze(folder_path: str):
    """Analyse toutes les images d'un dossier"""
    
    if not os.path.isdir(folder_path):
        print(f"❌ Dossier introuvable: {folder_path}")
        return
    
    images = [f for f in os.listdir(folder_path) 
              if Path(f).suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp', '.bmp']]
    
    if not images:
        print(f"❌ Aucune image trouvée dans: {folder_path}")
        return
    
    print(f"\n📁 Analyse par lot de {len(images)} images dans: {folder_path}\n")
    
    results = []
    for i, image in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {image}")
        image_path = os.path.join(folder_path, image)
        result = detect_glasses(image_path)
        if result:
            results.append({'file': image, 'result': result})
        print()
    
    # Résumé
    print("=" * 50)
    print(f"  RÉSUMÉ: {len(results)}/{len(images)} analyses réussies")
    print("=" * 50)
    
    return results

def main():
    """Point d'entrée principal"""
    
    print("\n🕶️  Service IA - Détection de montures")
    print("-" * 40)
    
    # Vérifier le service
    if not check_health():
        sys.exit(1)
    
    # Analyser l'image passée en argument
    if len(sys.argv) > 1:
        path = sys.argv[1]
        
        if os.path.isdir(path):
            # Mode dossier
            batch_analyze(path)
        else:
            # Mode fichier unique
            result = detect_glasses(path)
            display_result(result)
    else:
        # Mode interactif
        print("\n📋 Utilisation:")
        print("  python detect_image.py <chemin_image>")
        print("  python detect_image.py <dossier_images>")
        print("\nExemples:")
        print('  python detect_image.py "Capture d\'écran 2026-07-26 194736.png"')
        print('  python detect_image.py "C:\\Users\\Photos\\montures"')
        
        # Chercher des images dans le dossier courant
        current_images = [f for f in os.listdir('.') 
                         if Path(f).suffix.lower() in ['.png', '.jpg', '.jpeg']]
        
        if current_images:
            print(f"\n📸 Images trouvées dans le dossier courant ({len(current_images)}):")
            for img in current_images[:5]:
                print(f"   - {img}")
            
            choix = input("\n👉 Voulez-vous analyser ces images ? (o/n): ").strip().lower()
            if choix in ['o', 'oui', 'y', 'yes']:
                for img in current_images[:5]:
                    result = detect_glasses(img)
                    display_result(result)
                    print()

if __name__ == "__main__":
    main()
