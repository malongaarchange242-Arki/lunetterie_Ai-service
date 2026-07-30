from typing import Optional

from pydantic import BaseModel


class AnalysisRequest(BaseModel):
    image_id: Optional[str] = None


class ShapeCandidate(BaseModel):
    shape: str
    confidence: float


class AnalysisResponse(BaseModel):
    shape: str
    shape_confidence: float
    shape_top3: Optional[list[ShapeCandidate]] = None
    color: str
    color_confidence: float
    material: str
    material_confidence: float
    mount_type: str
    mount_type_confidence: float
    gender: Optional[str] = None
    gender_confidence: Optional[float] = None
    brand: Optional[str] = None
    brand_confidence: Optional[float] = None
    reference: Optional[str] = None
    reference_confidence: Optional[float] = None
    size: Optional[str] = None
    model_version: str
    processing_time_ms: float
