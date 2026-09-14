from datetime import datetime, timezone
import uuid
from app.database.db import Base
from sqlalchemy import BigInteger, Column, DateTime, Identity, String, Integer
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID


class Person(Base):
    __tablename__ = "persons"

    id = Column(UUID(as_uuid=True), primary_key=True, index=True, default=uuid.uuid4, unique=True, nullable=False)
    person_code = Column(BigInteger, Identity(always=False), unique=True, index=True, nullable=False)
    name = Column(String,  nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    face_records = relationship("FaceRecord", back_populates="person", cascade="all, delete-orphan")
