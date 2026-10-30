from sqlalchemy import Column, String, Integer, Text, JSON, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from app.models.base import BaseModel


class Chunk(BaseModel):
    __tablename__ = "chunks"
    
    chapter_id = Column(String(36), ForeignKey("chapters.id", ondelete="CASCADE"), nullable=False, index=True)
    page_number = Column(Integer, nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    cleaned_content = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=False, default=0)
    bounding_boxes = Column(JSON, nullable=True)  # List of [x0, y0, x1, y1] box coordinates
    
    # Model input optimization (Max 256 tokens) vs Context chunking
    is_model_input = Column(Boolean, default=False, index=True)
    parent_chunk_id = Column(String(36), ForeignKey("chunks.id", ondelete="SET NULL"), nullable=True)
    
    # Relationships
    chapter = relationship("Chapter", back_populates="chunks")
    parent_chunk = relationship("Chunk", remote_side="Chunk.id", backref="sub_chunks")
    detections = relationship("Detection", back_populates="chunk", cascade="all, delete-orphan")

