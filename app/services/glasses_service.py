from app.database.database import SessionLocal
from app.database.models import Glasses
from app.schemas.glasses import GlassesCreate


def create_glasses(payload: GlassesCreate) -> Glasses:
    db = SessionLocal()
    try:
        record = Glasses(**payload.model_dump())
        db.add(record)
        db.commit()
        db.refresh(record)
        return record
    finally:
        db.close()
