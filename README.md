# AI Service - Lunetterie

Service FastAPI pour l’analyse d’images de montures.

## Structure

- app/api/main.py : point d’entrée API
- app/inference/pipeline.py : pipeline d’analyse
- app/models : poids des modèles
- tests : tests rapides

## Lancer localement

```bash
pip install -r requirements.txt
uvicorn app.api.main:app --reload --port 8000
```
