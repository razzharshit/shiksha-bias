from sqlalchemy import Column, String, Integer, Float, Text, JSON, Boolean, Enum as SQLEnum
from app.models.base import BaseModel
from app.models.enums import BinaryClassification, TaxonomyCategory


class BenchmarkItem(BaseModel):
    __tablename__ = "benchmark_items"
    
    dataset_source = Column(String(50), nullable=False, default="IndiCASA", index=True)
    context_id = Column(Integer, nullable=False, index=True)  # Used for grouping to prevent data leakage
    sentence = Column(Text, nullable=False)
    
    # Classification dimension
    classification = Column(SQLEnum(BinaryClassification), nullable=False, index=True)
    
    # Raw annotator scores preserved exactly as reported
    annotator_1_score = Column(Float, nullable=True)
    annotator_2_score = Column(Float, nullable=True)
    
    # Score validation flags for out-of-range detection (e.g. 0, 9, 44)
    is_annotator_1_valid = Column(Boolean, default=True, index=True)
    is_annotator_2_valid = Column(Boolean, default=True, index=True)
    is_pair_valid = Column(Boolean, default=True, index=True)
    validation_flag = Column(String(50), nullable=True)
    
    # Optional taxonomy category (left unassigned until annotated separately)
    taxonomy_category = Column(SQLEnum(TaxonomyCategory), nullable=True, index=True)
    
    # Dataset split assigned at the context_id group level ("train", "val", "test")
    split_group = Column(String(20), nullable=False, default="train", index=True)
    
    # 5-fold GroupKFold assignment for leak-free cross-validation
    group_kfold = Column(Integer, nullable=False, default=0, index=True)
    
    raw_metadata = Column(JSON, nullable=True)

