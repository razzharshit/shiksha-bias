from sqlalchemy import Column, String, Integer, ForeignKey
from sqlalchemy.orm import relationship
from app.models.base import BaseModel


class Chapter(BaseModel):
    __tablename__ = "chapters"
    
    textbook_id = Column(String(36), ForeignKey("textbooks.id", ondelete="CASCADE"), nullable=False, index=True)
    chapter_number = Column(Integer, nullable=False)
    title = Column(String(255), nullable=False)
    start_page = Column(Integer, nullable=False)
    end_page = Column(Integer, nullable=False)
    male_mentions = Column(Integer, default=0)
    female_mentions = Column(Integer, default=0)
    
    # Relationships
    textbook = relationship("Textbook", back_populates="chapters")
    chunks = relationship("Chunk", back_populates="chapter", cascade="all, delete-orphan", order_by="Chunk.chunk_index")

