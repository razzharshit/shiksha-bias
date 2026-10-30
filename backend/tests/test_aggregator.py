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
            t = text.lower()
            if "cooking" in t or "busy sweeping" in t or "chess too complex" in t:
                return DetectionResult(
                    classification="stereotype",
                    confidence=0.85,
                    evidence_span="busy sweeping" if "busy sweeping" in t else "cooking",
                    start_char=10,
                    end_char=23,
                    surrounding_context=surrounding_context or text,
                    tokens=text.split(),
                    prob_stereotype=0.85,
                    prob_anti_stereotype=0.15
                )
            elif "despite being a girl" in t or "counter" in t:
                return DetectionResult(
                    classification="anti_stereotype",
                    confidence=0.82,
                    evidence_span="despite being a girl",
                    start_char=0,
                    end_char=20,
                    surrounding_context=surrounding_context or text,
                    tokens=text.split(),
                    prob_stereotype=0.18,
                    prob_anti_stereotype=0.82
                )
            else:
                return DetectionResult(
                    classification="neutral",
                    confidence=0.50,
                    evidence_span=text[:20],
                    start_char=0,
                    end_char=min(20, len(text)),
                    surrounding_context=surrounding_context or text,
                    tokens=text.split(),
                    prob_stereotype=0.50,
                    prob_anti_stereotype=0.50
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


# -----------------------------------------------------------------------------
# Targeted Reliability Tests (Tasks 1-7)
# -----------------------------------------------------------------------------

def test_1_severe_representation_imbalance_alone_does_not_produce_stereotype(aggregator):
    """
    Test 1: Severe representation imbalance alone does not produce a stereotype classification.
    A passage mentioning only males (ratio = 1.0) without stereotypical role association
    must NOT be classified as STEREOTYPE.
    """
    text = (
        "The king summoned his sons, his brother, his uncle, and his grandfather. "
        "The boy listened carefully as the men discussed the harvest festival."
    )
    result = aggregator.analyze_chunk(text=text, chunk_id="chunk_skew_1")

    # Representation is skewed
    assert result.is_representation_imbalanced is True
    assert result.male_ratio == 1.0
    # Must NOT produce a stereotype classification
    assert result.classification != BinaryClassification.STEREOTYPE
    assert result.classification == BinaryClassification.NEUTRAL
    assert result.is_suppressed is True
    # Human review flagged for representation imbalance, but not as stereotype
    assert result.review_recommendation.requires_human_review is True
    assert "REPRESENTATION_ANALYSIS" in result.review_recommendation.contributing_sources


def test_2_low_heuristic_score_does_not_produce_gender_balanced(aggregator):
    """
    Test 2: A low heuristic score does not produce GENDER_BALANCED.
    Suppression due to lack of evidence must keep contextual_outcome as None.
    """
    text = "The committee held an annual review of the school syllabus."
    result = aggregator.analyze_chunk(text=text, chunk_id="chunk_low_1")

    assert result.heuristic_fusion_score < 0.65
    assert result.is_suppressed is True
    # Must NOT claim GENDER_BALANCED
    assert result.contextual_outcome is None


def test_3_insufficient_statistical_evidence_distinguishable_from_balanced(aggregator):
    """
    Test 3: Insufficient statistical evidence remains distinguishable from balanced representation.
    """
    # Text with 1 mention (insufficient data)
    sparse_text = "The boy walked home."
    sparse_result = aggregator.analyze_chunk(text=sparse_text)
    assert sparse_result.representation_imbalance_status == "INSUFFICIENT_DATA"

    # Text with balanced representation across 4+ mentions
    balanced_text = (
        "The boy and the girl studied together. "
        "Her mother and his father supported their education equally."
    )
    balanced_result = aggregator.analyze_chunk(text=balanced_text)
    assert balanced_result.representation_imbalance_status == "BALANCED"
    assert balanced_result.is_representation_imbalanced is False
    assert 0.25 <= balanced_result.male_ratio <= 0.75


def test_4_gender_neutral_occupation_does_not_trigger_occupational_bias(aggregator):
    """
    Test 4: A gender-neutral occupation does not trigger occupational bias.
    """
    text = "Modern software engineering and scientific inquiry require dedicated practice."
    result = aggregator.analyze_chunk(text=text)

    assert len(result.co_occurrences) == 0
    assert TaxonomyCategory.OCCUPATIONAL_BIAS not in result.candidate_taxonomy_categories
    assert result.taxonomy_category is None


def test_5_ambiguous_lexical_proximity_not_treated_as_confirmed_attribution(aggregator):
    """
    Test 5: Ambiguous lexical proximity is not treated as confirmed subject attribution.
    """
    text = "The girl watched as the father was cooking dinner."
    result = aggregator.analyze_chunk(text=text)

    assert len(result.evidence_items) > 0
    stat_items = [e for e in result.evidence_items if e.source == "STATISTICAL_ANALYZER"]
    assert len(stat_items) > 0
    # Provenance warning must explicitly state ambiguous proximity
    assert all("Ambiguous lexical proximity" in e.limitations_or_warnings for e in stat_items)


def test_6_neural_and_statistical_evidence_retain_separate_provenance(aggregator):
    """
    Test 6: Neural and statistical evidence retain separate provenance.
    """
    text = "The mother was busy sweeping the floor while cooking."
    result = aggregator.analyze_chunk(text=text)

    sources = [e.source for e in result.evidence_items]
    assert "NEURAL_MODEL" in sources
    assert "STATISTICAL_ANALYZER" in sources

    neural_item = next(e for e in result.evidence_items if e.source == "NEURAL_MODEL")
    stat_item = next(e for e in result.evidence_items if e.source == "STATISTICAL_ANALYZER")

    assert "MODEL_ATTRIBUTION_SPAN" in neural_item.associated_signal
    assert "POTENTIAL_STEREOTYPICAL_ASSOCIATION" in stat_item.associated_signal
    assert "attention weights" in neural_item.limitations_or_warnings
    assert "dependency parsing" in stat_item.limitations_or_warnings


def test_7_unsupported_taxonomy_categories_not_assigned(aggregator):
    """
    Test 7: Unsupported taxonomy categories are not assigned.
    ABILITY_INTELLECTUAL_BIAS must not be defaulted without affirmative evidence.
    """
    text = "The mother was cooking in the kitchen."
    result = aggregator.analyze_chunk(text=text)

    # Domestic role evidence exists, but NO ability bias evidence exists
    assert result.taxonomy_category == TaxonomyCategory.DOMESTIC_ROLES
    assert TaxonomyCategory.ABILITY_INTELLECTUAL_BIAS not in result.candidate_taxonomy_categories

    # Neutral text with no signals must have NO taxonomy category
    neutral_text = "The students gathered in the central auditorium."
    neutral_result = aggregator.analyze_chunk(text=neutral_text)
    assert neutral_result.taxonomy_category is None
    assert len(neutral_result.candidate_taxonomy_categories) == 0


def test_8_neural_probabilities_remain_unchanged_by_representation_statistics(aggregator):
    """
    Test 8: Neural probabilities remain unchanged by representation statistics.
    """
    neutral_text_unskewed = "The boy and girl walked to school."
    neutral_text_skewed = "The king summoned his brother, son, uncle, and father to the castle."

    res_unskewed = aggregator.analyze_chunk(text=neutral_text_unskewed)
    res_skewed = aggregator.analyze_chunk(text=neutral_text_skewed)

    # In both cases, mock neural model predicts neutral with p_stereo=0.50
    # Representation statistics must NOT alter neural_prob_stereotype
    assert res_unskewed.neural_prob_stereotype == 0.50
    assert res_skewed.neural_prob_stereotype == 0.50
    assert res_unskewed.neural_prob_anti_stereotype == 0.50
    assert res_skewed.neural_prob_anti_stereotype == 0.50


def test_9_repeated_processing_prevents_duplicate_records(aggregator, db_session):
    """
    Test 9: Repeated processing follows the intended database duplicate behavior (idempotent update).
    """
    session, chunk_id = db_session
    text = "Mother was busy sweeping the floor."
    detection_res = aggregator.analyze_chunk(text=text, chunk_id=chunk_id)

    # First persistence
    item_1 = aggregator.persist_to_database(session, detection_res)
    first_id = item_1.id

    # Second persistence with the exact same chunk_id and span
    item_2 = aggregator.persist_to_database(session, detection_res)
    second_id = item_2.id

    # Must update existing record and NOT create duplicate
    assert first_id == second_id
    total_count = session.query(Detection).filter_by(chunk_id=chunk_id).count()
    assert total_count == 1

    # Verify audit metadata was persisted
    saved = session.query(Detection).filter_by(id=first_id).first()
    assert saved.audit_metadata is not None
    assert "heuristic_fusion_score" in saved.audit_metadata


def test_10_high_confidence_stereotype_and_anti_stereotype_preservation(aggregator):
    """
    Test 10: Preserves high-confidence stereotype and anti-stereotype predictions with human review recommendations.
    """
    # Stereotype case
    stereo_text = "Mother was busy sweeping while cooking dinner."
    s_res = aggregator.analyze_chunk(text=stereo_text)
    assert s_res.classification == BinaryClassification.STEREOTYPE
    assert s_res.heuristic_fusion_score >= 0.65
    assert s_res.review_recommendation.requires_human_review is True
    assert s_res.review_recommendation.priority in ["HIGH", "MEDIUM"]

    # Anti-stereotype case
    anti_text = "Despite being a girl, she outperformed in the counter competition."
    a_res = aggregator.analyze_chunk(text=anti_text)
    assert a_res.classification == BinaryClassification.ANTI_STEREOTYPE
    assert a_res.neural_prob_anti_stereotype >= 0.65
    assert a_res.contextual_outcome == ContextualOutcome.COUNTER_NARRATIVE
