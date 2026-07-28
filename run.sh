#!/bin/bash
echo "Démarrage du service IA Lunetterie..."
cd "$(dirname "$0")"
python3 -m venv venv 2>/dev/null
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
