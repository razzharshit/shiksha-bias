import os
import sys
import json
import pytest
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings
from app.core.database import SessionLocal, init_db, engine
from app.models.enums import BinaryClassification, TaxonomyCategory, ContextualOutcome, ProcessingStatus
from app.models.textbook import Textbook
from app.models.chapter import Chapter
from app.models.chunk import Chunk
from app.models.detection import Detection
from app.models.benchmark_item import BenchmarkItem
from app.services.ingestion.pdf_parser import PDFParser, PDFParserConfig
from app.services.ingestion.chunker import ContextAwareChunker


@pytest.fixture(scope="module")
def db_session():
    init_db()
    db = SessionLocal()
    yield db
    db.close()


# ============================================================================
# P1-01: Core Configuration & Database Setup
# ============================================================================
def test_p1_01_core_config_and_database(db_session):
    assert settings.PROJECT_NAME is not None
    assert settings.MODEL_MAX_SEQUENCE_LENGTH == 256
    assert settings.DATASET_DIR.exists()
    assert settings.BENCHMARKS_DIR.exists()
    
    # Verify engine can execute query
    result = db_session.execute(Textbook.__table__.select()).fetchall()
    assert isinstance(result, list)


# ============================================================================
# P1-02: Essential Data Models
# ============================================================================
def test_p1_02_essential_models_and_enums(db_session):
    # Test Enums
    assert BinaryClassification.STEREOTYPE.value == "stereotype"
    assert BinaryClassification.ANTI_STEREOTYPE.value == "anti_stereotype"
    assert TaxonomyCategory.DOMESTIC_ROLES.value == "DOMESTIC_ROLES"
    assert ContextualOutcome.HISTORICAL_CONTEXT.value == "HISTORICAL_CONTEXT"

    # Create and persist a sample textbook hierarchy
    tb = Textbook(
        id="test_ncert_tb_1",
        title="Grade 6 Social Science",
        subject="Social Science",
        target_grade=6,
        board_or_curriculum="NCERT",
        file_path="dataset/India/Grade 1/English/aeen101.pdf",
        total_pages=15,
        status=ProcessingStatus.COMPLETED
    )
    db_session.merge(tb)
    db_session.commit()

    ch = Chapter(
        id="test_ch_1",
        textbook_id="test_ncert_tb_1",
        chapter_number=1,
        title="Diversity and Discrimination",
        start_page=1,
        end_page=5
    )
    db_session.merge(ch)
    db_session.commit()

    chunk = Chunk(
        id="test_chunk_1",
        chapter_id="test_ch_1",
        page_number=1,
        chunk_index=0,
        content="While the mother prepared food, the father read the newspaper.",
        cleaned_content="While the mother prepared food, the father read the newspaper.",
        token_count=14,
        is_model_input=True
    )
    db_session.merge(chunk)
    db_session.commit()

    det = Detection(
        id="test_det_1",
        chunk_id="test_chunk_1",
        classification=BinaryClassification.STEREOTYPE,
        taxonomy_category=TaxonomyCategory.DOMESTIC_ROLES,
        confidence=0.91,
        evidence_span="mother prepared food"
    )
    db_session.merge(det)
    db_session.commit()

    # Query back
    queried_tb = db_session.query(Textbook).filter(Textbook.id == "test_ncert_tb_1").first()
    assert queried_tb is not None
    assert len(queried_tb.chapters) == 1
    assert queried_tb.chapters[0].chunks[0].detections[0].evidence_span == "mother prepared food"


# ============================================================================
# P1-03: PDF Extraction Pipeline & Extraction Validation
# ============================================================================
def test_p1_03_pdf_parser_extraction_and_validation():
    sample_pdf = settings.BASE_DIR / "SDG_5_Gender_Stereotype_AI_Project.pdf"
    if not sample_pdf.exists():
        pytest.skip("Proposal PDF not found.")

    parser = PDFParser(PDFParserConfig(filter_headers_footers=True, header_margin_pct=0.05))
    pages = parser.parse(str(sample_pdf))

    assert len(pages) > 0
    # Validate reading order and blocks
    first_page = pages[0]
    assert first_page.page_number == 1
    assert first_page.word_count > 50
    assert first_page.printable_ratio > 0.90

    # Comprehensive validation
    report = parser.validate_extraction(pages)
    assert report.is_valid is True
    assert report.page_coverage_pct == 100.0
    assert report.total_words > 500


# ============================================================================
# P1-04: Model-Compatible Chunking (Strictly <= 256 tokens)
# ============================================================================
def test_p1_04_chunker_token_and_sentence_constraints():
    sample_pdf = settings.BASE_DIR / "SDG_5_Gender_Stereotype_AI_Project.pdf"
    if not sample_pdf.exists():
        pytest.skip("Proposal PDF not found.")

    parser = PDFParser()
    pages = parser.parse(str(sample_pdf))

    chunker = ContextAwareChunker(model_max_tokens=256, context_target_tokens=400)
    chunks = chunker.chunk_pages(pages)

    assert len(chunks) > 0

    # Extract all model-input sub-chunks
    model_chunks = [sc for c in chunks for sc in c.sub_chunks]
    assert len(model_chunks) > 0

    for mc in model_chunks:
        assert mc.is_model_input is True
        # Enforce that model input chunk never exceeds 256 tokens
        assert mc.token_count <= 256, f"Chunk exceeded 256 tokens: {mc.token_count}"
        # Ensure non-empty text
        assert len(mc.cleaned_content.strip()) > 0


# ============================================================================
# P1-05: Taxonomy & Group-Aware IndiCASA Dataset Ingestion
# ============================================================================
def test_p1_05_indicasa_group_split_and_taxonomy(db_session):
    benchmark_file = settings.BENCHMARKS_DIR / "indicasa_split_benchmark.json"
    taxonomy_file = settings.BENCHMARKS_DIR / "taxonomy.json"

    assert taxonomy_file.exists(), "Taxonomy JSON missing"
    assert benchmark_file.exists(), "IndiCASA benchmark JSON missing"

    with open(taxonomy_file, "r", encoding="utf-8") as f:
        tax = json.load(f)
        assert "dimension_1_classification" in tax["label_dimensions"]
        assert "dimension_2_taxonomy_categories" in tax["label_dimensions"]
        assert len(tax["label_dimensions"]["dimension_2_taxonomy_categories"]["categories"]) == 6

    with open(benchmark_file, "r", encoding="utf-8") as f:
        bench = json.load(f)
        assert bench["total_sentences"] == 853
        assert bench["total_contexts"] == 37

        train_ctx = set(bench["splits_70_15_15"]["train"]["context_ids"])
        val_ctx = set(bench["splits_70_15_15"]["val"]["context_ids"])
        test_ctx = set(bench["splits_70_15_15"]["test"]["context_ids"])

        # ZERO DATA LEAKAGE: All contexts must be mutually exclusive
        assert train_ctx.isdisjoint(val_ctx), "Context leakage between train and val"
        assert train_ctx.isdisjoint(test_ctx), "Context leakage between train and test"
        assert val_ctx.isdisjoint(test_ctx), "Context leakage between val and test"

        # Verify 5-Fold GroupKFold presence
        assert "group_kfold_5" in bench
        assert len(bench["group_kfold_5"]) == 5

        # Check raw annotator scores preservation & validation flags
        first_item = bench["records"][0]
        assert "annotator_1_score" in first_item
        assert "annotator_2_score" in first_item
        assert "is_annotator_1_valid" in first_item
        assert "is_annotator_2_valid" in first_item
        assert "group_kfold" in first_item
        assert first_item["taxonomy_category"] is None, "Taxonomy category should not be guessed"

    # Check database persistence
    items_count = db_session.query(BenchmarkItem).count()
    assert items_count == 853

