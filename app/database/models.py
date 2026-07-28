from sqlalchemy import Boolean, Column, Integer, String

from app.database.database import Base


class Glasses(Base):
    __tablename__ = "glasses"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    brand = Column(String, nullable=True)
    reference = Column(String, nullable=True)
    frame_shape = Column(String, nullable=True)
    color = Column(String, nullable=True)
    material = Column(String, nullable=True)
    has_branches = Column(Boolean, nullable=True)
    mount_type = Column(String, nullable=True)
