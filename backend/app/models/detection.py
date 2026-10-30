from sqlalchemy import Column, String, Float, Text, Boolean, Integer, JSON, Enum as SQLEnum, ForeignKey
from sqlalchemy.orm import relationship
from app.models.base import BaseModel
from app.models.enums import BinaryClassification, TaxonomyCategory, ContextualOutcome


class Detection(BaseModel):
    __tablename__ = "detections"
    
    chunk_id = Column(String(36), ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Dimension 1: Binary Classification
    classification = Column(SQLEnum(BinaryClassification), nullable=False, default=BinaryClassification.STEREOTYPE, index=True)
    
    # Dimension 2: Granular Taxonomy (Nullable, predicted or annotated separately)
    taxonomy_category = Column(SQLEnum(TaxonomyCategory), nullable=True, index=True)
    
    # Contextual Evaluation Outcome (Historical context, Counter-narrative, etc.)
    contextual_outcome = Column(SQLEnum(ContextualOutcome), nullable=True)
    
    confidence = Column(Float, nullable=False, default=0.5)
    evidence_span = Column(Text, nullable=False)
    start_char = Column(Integer, nullable=True)
    end_char = Column(Integer, nullable=True)
    surrounding_context = Column(Text, nullable=True)
    detection_method = Column(String(50), nullable=False, default="HYBRID")
    is_suppressed = Column(Boolean, nullable=False, default=False)
    audit_metadata = Column(JSON, nullable=True)
    
    # Relationships
    chunk = relationship("Chunk", back_populates="detections")

