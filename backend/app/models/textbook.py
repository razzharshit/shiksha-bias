from sqlalchemy import Column, String, Integer, Enum as SQLEnum, Text
from sqlalchemy.orm import relationship
from app.models.base import BaseModel
from app.models.enums import ProcessingStatus


class Textbook(BaseModel):
    __tablename__ = "textbooks"
    
    title = Column(String(255), nullable=False)
    subject = Column(String(100), nullable=False, default="General")
    target_grade = Column(Integer, nullable=False, default=6)
    board_or_curriculum = Column(String(100), nullable=False, default="NCERT")
    file_path = Column(String(500), nullable=False)
    file_size_bytes = Column(Integer, nullable=False, default=0)
    total_pages = Column(Integer, nullable=False, default=0)
    status = Column(SQLEnum(ProcessingStatus), nullable=False, default=ProcessingStatus.PENDING)
    error_message = Column(Text, nullable=True)
    
    # Relationships
    chapters = relationship("Chapter", back_populates="textbook", cascade="all, delete-orphan", order_by="Chapter.chapter_number")

