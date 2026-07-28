from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.ai.predictor import GlassesPredictor

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
