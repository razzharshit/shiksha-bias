import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.detection.transformer import TransformerClassifier, DetectionResult


def test_transformer_classifier_instantiation():
    clf = TransformerClassifier(model_path_or_name="microsoft/deberta-v3-base")
    assert clf is not None


def test_transformer_classifier_predict_fallbacks():
    clf = TransformerClassifier(model_path_or_name="non_existent_path")
    
    # Test stereotypical trigger
    res = clf.predict("As a girl, Leena found chess too complex for her to handle.")
    assert isinstance(res, DetectionResult)
    assert res.classification == "stereotype"
    assert "too complex for her" in res.evidence_span
    assert res.confidence >= 0.80

    # Test neutral passage
    res_neutral = clf.predict("The students read Chapter 4 in their science textbook.")
    assert res_neutral.classification == "neutral"


def test_token_span_attribution_bounds():
    clf = TransformerClassifier(model_path_or_name="non_existent_path")
    text = "While his mother was busy sweeping the floor, his father worked."
    res = clf.predict(text)
    
    assert res.start_char >= 0
    assert res.end_char <= len(text)
    assert text[res.start_char:res.end_char] == res.evidence_span
