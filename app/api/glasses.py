import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.ai import claude_vision
from app.ai.predictor import GlassesPredictor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/glasses", tags=["glasses"])
predictor = GlassesPredictor()


@router.post("/analyze")
async def analyze_glasses(file: UploadFile = File(...)) -> Any:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Le fichier est requis")

    upload_dir = Path("app/uploads/temp")
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / file.filename

    with destination.open("wb") as buffer:
        content = await file.read()
        buffer.write(content)

    result = predictor.predict_image(str(destination))
    return JSONResponse(status_code=200, content=result)


@router.post("/analyze-branche")
async def analyze_branche(file: UploadFile = File(...)) -> Any:
    """OCR de la branche (temple) : lit la référence et la marque gravées/imprimées dessus
    via Claude vision. Renvoie reference/brand à null si la clé API est absente, l'appel
    échoue, ou que le texte n'est pas lisible — jamais d'erreur bloquante côté appelant."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Le fichier est requis")

    upload_dir = Path("app/uploads/temp")
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / file.filename

    with destination.open("wb") as buffer:
        content = await file.read()
        buffer.write(content)

    try:
        result = claude_vision.ocr_branche(str(destination))
    except Exception as exc:
        logger.warning("OCR branche indisponible: %s", exc)
        result = None

    return JSONResponse(status_code=200, content=result or {"reference": None, "brand": None, "confidence": 0.0})
