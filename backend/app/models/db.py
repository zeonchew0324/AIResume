from sqlalchemy import Column, String, Text, DateTime
from sqlalchemy.dialects.postgresql import UUID
from app.db.base import Base
from datetime import datetime, timezone
import uuid

def utcnow():
    return datetime.now(timezone.utc)

class Resume(Base):
    __tablename__ = "resumes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    name = Column(String(255), nullable=False)
    resume_text = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow)
