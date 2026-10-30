"""
Unit and Integration Tests for Phase 1 Evaluation Benchmark Suite (Task P1-09)
"""

import json
from pathlib import Path
import pytest

from scripts.evaluate_detection import run_evaluation


@pytest.fixture
def benchmark_file():
    base_dir = Path(__file__).resolve().parent.parent.parent
    path = base_dir / "data" / "benchmarks" / "stereotypes_eval_labeled.json"
    assert path.exists(), f"Benchmark file does not exist at {path}"
    return path


@pytest.fixture
def report_file():
    base_dir = Path(__file__).resolve().parent.parent.parent
    path = base_dir / "data" / "benchmarks" / "baseline_eval_report.json"
    return path


def test_benchmark_dataset_schema_and_distribution(benchmark_file):
    """Verifies that the 250-passage benchmark dataset satisfies schema and diversity requirements."""
    with open(benchmark_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_samples"] == 250
    samples = data["samples"]
    assert len(samples) == 250

    stereo_count = 0
    anti_count = 0
    neutral_count = 0
    hist_count = 0

    for s in samples:
        # Schema checks
        assert "id" in s and s["id"].startswith("eval_")
        assert "text" in s and len(s["text"].strip()) > 0
        assert s["classification"] in ["stereotype", "anti_stereotype", "neutral"]
        assert s["grade_level"].startswith("Grade")
        assert s["subject"] in ["Social Studies", "Science", "Literature", "History", "Mathematics"]

        if s["classification"] == "stereotype":
            stereo_count += 1
            assert s["category"] in [
                "DOMESTIC_ROLES", "OCCUPATIONAL_BIAS", "ABILITY_INTELLECTUAL_BIAS",
                "PERSONALITY_BEHAVIORAL", "SOCIAL_DECISION_MAKING", "REPRESENTATIONAL_IMBALANCE"
            ]
            assert s["evidence_span"] != ""
            assert s["evidence_span"] in s["text"]
        elif s["classification"] == "anti_stereotype":
            anti_count += 1
            assert s["contextual_outcome"] == "COUNTER_NARRATIVE"
            assert s["evidence_span"] in s["text"]
        elif s["classification"] == "neutral":
            neutral_count += 1
            if s.get("is_historical_counterexample"):
                hist_count += 1
                assert s["contextual_outcome"] == "HISTORICAL_CONTEXT"

    assert stereo_count == 130
    assert anti_count == 50
    assert neutral_count == 70
    assert hist_count == 35


def test_evaluation_harness_end_to_end(benchmark_file, tmp_path):
    """Executes evaluation harness end-to-end and verifies acceptance criteria gates."""
    temp_report_path = tmp_path / "temp_baseline_report.json"

    report = run_evaluation(
        benchmark_path=benchmark_file,
        output_report_path=temp_report_path,
        w_neural=0.7,
        w_stat=0.3,
        threshold=0.65
    )

    # Acceptance criteria: Macro F1 >= 0.78, Precision >= 0.80 across core stereotype classes
    assert report["aggregate_metrics"]["macro_f1"] >= 0.78
    assert report["3_class_classification"]["class_metrics"]["stereotype"]["precision"] >= 0.80

    # Historical counterexample false positive rate <= 5%
    hist_analysis = report["historical_counterexample_analysis"]
    assert hist_analysis["false_positive_rate"] <= 0.05
    assert hist_analysis["total_evaluated"] == 35

    # Evidence span validity == 100%
    span_analysis = report["evidence_span_analysis"]
    assert span_analysis["valid_span_rate"] == 1.0

    assert report["aggregate_metrics"]["acceptance_criteria_passed"] is True
    assert temp_report_path.exists()


def test_baseline_eval_report_file_integrity(report_file):
    """Verifies that the generated baseline_eval_report.json artifact is valid and meets requirements."""
    assert report_file.exists(), "baseline_eval_report.json missing from data/benchmarks"

    with open(report_file, "r", encoding="utf-8") as f:
        rep = json.load(f)

    assert "metadata" in rep
    assert "aggregate_metrics" in rep
    assert "3_class_classification" in rep
    assert "taxonomy_category_metrics" in rep
    assert "historical_counterexample_analysis" in rep
    assert "evidence_span_analysis" in rep

    assert rep["metadata"]["total_samples"] == 250
    assert rep["aggregate_metrics"]["macro_f1"] >= 0.78
    assert rep["aggregate_metrics"]["acceptance_criteria_passed"] is True
