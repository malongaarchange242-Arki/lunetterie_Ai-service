from pydantic import BaseModel


class GlassesCreate(BaseModel):
    name: str
    description: str | None = None
    brand: str | None = None
    reference: str | None = None
    frame_shape: str | None = None
    color: str | None = None
    material: str | None = None
    has_branches: bool | None = None
    mount_type: str | None = None
