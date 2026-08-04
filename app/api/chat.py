from fastapi import APIRouter, HTTPException

from app.ai.chat import chat_reply
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/assistant", tags=["assistant"])


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    try:
        reply = chat_reply(
            message=payload.message,
            history=[m.model_dump() for m in payload.history],
            context=payload.context,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:  # appel Claude en échec (réseau, quota, etc.)
        raise HTTPException(status_code=502, detail=f"Appel Claude échoué: {exc}")

    return ChatResponse(reply=reply)
