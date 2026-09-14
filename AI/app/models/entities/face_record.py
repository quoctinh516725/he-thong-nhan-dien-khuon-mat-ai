import uuid

from app.database.db import Base
from sqlalchemy import Column, DateTime, ForeignKey, String
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from datetime import datetime, timezone
from sqlalchemy.dialects.postgresql import UUID


class FaceRecord(Base):
    __tablename__ = "face_records"

    id = Column(UUID(as_uuid=True), primary_key=True, index=True, default=uuid.uuid4, unique=True, nullable=False)
    person_id = Column(UUID(as_uuid=True), ForeignKey("persons.id"), index=True, nullable=False)
    image_path = Column(String, nullable=True)
    pose_label = Column(String, nullable=True, index=True)
    embedding = Column(Vector(512), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    person = relationship("Person", back_populates="face_records")
