import logging
from fastapi import FastAPI, File, HTTPException, UploadFile

from app.inference.pipeline import analyze_image
from app.schemas.analysis import AnalysisResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Service - Lunetterie",
    description="Service d'analyse de montures par IA",
    version="1.0.0",
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "healthy", "model_version": "1.0.0"}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...)) -> dict[str, object]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image")

    try:
        image_bytes = await file.read()
        result = analyze_image(image_bytes)
        if "error" in result:
            raise HTTPException(status_code=422, detail=result["error"])
        logger.info("Analyse réussie")
        return result
    except Exception as exc:  # pragma: no cover - defensive path
        logger.error("Erreur lors de l'analyse: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
