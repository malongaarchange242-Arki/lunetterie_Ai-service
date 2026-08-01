import logging

from fastapi import FastAPI
from app.api.glasses import router as glasses_router

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Glasses AI Registration")
app.include_router(glasses_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
