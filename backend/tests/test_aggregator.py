import os
import sys
import uuid
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

try:
    from app.models.base import Base
    from app.models.enums import BinaryClassification, TaxonomyCategory, ContextualOutcome
    from app.models.textbook import Textbook
    from app.models.chapter import Chapter
    from app.models.chunk import Chunk
    from app.models.detection import Detection

    from app.services.detection.aggregator import HybridDetectionAggregator, AggregatedDetection
    from app.services.detection.transformer import TransformerClassifier, DetectionResult
    from app.services.detection.statistical import StatisticalRepresentationAnalyzer
except ImportError:
    from backend.app.models.base import Base
    from backend.app.models.enums import BinaryClassification, TaxonomyCategory, ContextualOutcome
    from backend.app.models.textbook import Textbook
    from backend.app.models.chapter import Chapter
    from backend.app.models.chunk import Chunk
    from backend.app.models.detection import Detection

    from backend.app.services.detection.aggregator import HybridDetectionAggregator, AggregatedDetection
    from backend.app.services.detection.transformer import TransformerClassifier, DetectionResult
    from backend.app.services.detection.statistical import StatisticalRepresentationAnalyzer


@pytest.fixture
def mock_classifier():
    class MockTransformerClassifier(TransformerClassifier):
        def __init__(self):
            self.is_loaded = True

        def predict(self, text: str, surrounding_context: str = "") -> DetectionResult:
            if "cooking" in text.lower() or "busy sweeping" in text.lower():
                return DetectionResult(
                    classification="stereotype",
                    confidence=0.85,
                    evidence_span="busy sweeping",
                    start_char=text.find("busy sweeping") if "busy sweeping" in text else 0,
                    end_char=text.find("busy sweeping") + 13 if "busy sweeping" in text else 13,
                    surrounding_context=surrounding_context or text,
                    tokens=text.split()
                )
            elif "despite being a girl" in text.lower() or "counter" in text.lower():
                return DetectionResult(
                    classification="anti_stereotype",
                    confidence=0.82,
                    evidence_span="despite being a girl",
                    start_char=0,
                    end_char=20,
                    surrounding_context=surrounding_context or text,
                    tokens=text.split()
                )
            else:
                return DetectionResult(
                    classification="neutral",
                    confidence=0.50,
                    evidence_span=text[:20],
                    start_char=0,
                    end_char=20,
                    surrounding_context=surrounding_context or text,
                    tokens=text.split()
                )

    return MockTransformerClassifier()


@pytest.fixture
def aggregator(mock_classifier):
    return HybridDetectionAggregator(
        transformer_classifier=mock_classifier,
        statistical_analyzer=StatisticalRepresentationAnalyzer(),
        weight_neural=0.7,
        weight_stat=0.3,
        confidence_threshold=0.65
    )


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Create dummy textbook and chunk for foreign key consistency
    tb = Textbook(
        id=str(uuid.uuid4()),
        title="Class 7 Social Studies",
        subject="Social Studies",
        target_grade=7,
        board_or_curriculum="NCERT",
        file_path="/data/sample_ncert.pdf"
    )
    session.add(tb)
    session.flush()

    ch = Chapter(
        id=str(uuid.uuid4()),
        textbook_id=tb.id,
        chapter_number=1,
        title="Gender and Society",
        start_page=1,
        end_page=15
    )
    session.add(ch)
    session.flush()

    chunk = Chunk(
        id=str(uuid.uuid4()),
        chapter_id=ch.id,
        page_number=1,
        chunk_index=0,
        content="Mother was busy sweeping while her daughter was washing the laundry.",
        cleaned_content="Mother was busy sweeping while her daughter was washing the laundry.",
        token_count=12
    )
    session.add(chunk)
    session.commit()

    yield session, chunk.id
    session.close()


def test_high_confidence_stereotype_detection(aggregator):
    text = "The mother was busy sweeping the floor while the girl was cooking in the kitchen."
    result = aggregator.analyze_chunk(text=text, chunk_id="chunk_test_1")

    assert result.classification == BinaryClassification.STEREOTYPE
    assert result.confidence >= 0.65
    assert result.is_suppressed is False
    assert result.taxonomy_category == TaxonomyCategory.DOMESTIC_ROLES
    assert "sweeping" in result.evidence_span or "mother" in result.evidence_span


def test_low_confidence_suppression(aggregator):
    text = "The student walked to school and sat down quietly at the desk."
    result = aggregator.analyze_chunk(text=text, chunk_id="chunk_test_2")

    assert result.classification == BinaryClassification.NEUTRAL
    assert result.is_suppressed is True
    assert result.contextual_outcome == ContextualOutcome.GENDER_BALANCED


def test_anti_stereotype_detection(aggregator):
    text = "Despite being a girl, she outperformed her peers in mathematics and counter narratives."
    result = aggregator.analyze_chunk(text=text, chunk_id="chunk_test_3")

    assert result.classification == BinaryClassification.ANTI_STEREOTYPE
    assert result.contextual_outcome == ContextualOutcome.COUNTER_NARRATIVE
    assert result.is_suppressed is False


def test_representation_imbalance_mapping(aggregator):
    text = (
        "The king spoke with his sons, his brother, his uncle, and his grandfather. "
        "The boy listened carefully as the men planned the journey."
    )
    result = aggregator.analyze_chunk(text=text, chunk_id="chunk_test_4")

    # Has male references > 0.75 with >= 4 mentions
    assert result.taxonomy_category == TaxonomyCategory.REPRESENTATIONAL_IMBALANCE


def test_database_persistence(aggregator, db_session):
    session, chunk_id = db_session
    text = "The mother was busy sweeping the house all day."
    detection_res = aggregator.analyze_chunk(text=text, chunk_id=chunk_id)

    db_item = aggregator.persist_to_database(session, detection_res)

    assert db_item.id is not None
    assert db_item.chunk_id == chunk_id
    assert db_item.classification == BinaryClassification.STEREOTYPE
    assert db_item.detection_method == "HYBRID"

    # Query from database to verify persistence
    saved = session.query(Detection).filter_by(id=db_item.id).first()
    assert saved is not None
    assert saved.confidence == detection_res.confidence
