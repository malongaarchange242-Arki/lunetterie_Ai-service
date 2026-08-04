import logging

from fastapi import FastAPI
from app.api.chat import router as chat_router
from app.api.glasses import router as glasses_router

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Glasses AI Registration")
app.include_router(glasses_router)
app.include_router(chat_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
