import os
import sys
import re
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field

try:
    from app.models.enums import BinaryClassification, TaxonomyCategory, ContextualOutcome
    from app.services.detection.transformer import TransformerClassifier, DetectionResult
    from app.services.detection.statistical import StatisticalRepresentationAnalyzer, StatisticalAnalysisResult, CoOccurrence
except ImportError:
    from backend.app.models.enums import BinaryClassification, TaxonomyCategory, ContextualOutcome
    from backend.app.services.detection.transformer import TransformerClassifier, DetectionResult
    from backend.app.services.detection.statistical import StatisticalRepresentationAnalyzer, StatisticalAnalysisResult, CoOccurrence


class AggregatedDetection(BaseModel):
    """Calibrated hybrid detection result combining transformer classification and statistical analysis."""
    chunk_id: Optional[str] = None
    classification: BinaryClassification = BinaryClassification.NEUTRAL
    taxonomy_category: Optional[TaxonomyCategory] = None
    contextual_outcome: Optional[ContextualOutcome] = None
    confidence: float = Field(ge=0.0, le=1.0, description="Calibrated fused confidence score")
    neural_confidence: float = Field(ge=0.0, le=1.0)
    statistical_score: float = Field(ge=0.0, le=1.0)
    evidence_span: str = ""
    start_char: int = 0
    end_char: int = 0
    surrounding_context: str = ""
    detection_method: str = "HYBRID"
    is_suppressed: bool = False
    reasons_for_decision: List[str] = Field(default_factory=list)


class HybridDetectionAggregator:
    """
    Fuses transformer neural representations with deterministic statistical signals.
    
    Calibration & Fusion Formula:
        Confidence_fused = w_neural * P_neural + w_stat * S_stat
        where default weights are w_neural = 0.7, w_stat = 0.3.
    
    Threshold Gating:
        Detections with fused confidence < confidence_threshold (default 0.65)
        are marked as suppressed or balanced to prevent false alarms.
    
    Evidence Span Merging:
        Merges overlapping or contiguous sub-spans from neural attention
        and statistical co-occurrence window tokens.
    """

    def __init__(
        self,
        transformer_classifier: Optional[TransformerClassifier] = None,
        statistical_analyzer: Optional[StatisticalRepresentationAnalyzer] = None,
        weight_neural: float = 0.7,
        weight_stat: float = 0.3,
        confidence_threshold: float = 0.65
    ):
        self.classifier = transformer_classifier or TransformerClassifier()
        self.statistical_analyzer = statistical_analyzer or StatisticalRepresentationAnalyzer()
        self.w_neural = weight_neural
        self.w_stat = weight_stat
        self.confidence_threshold = confidence_threshold

    def analyze_chunk(
        self,
        text: str,
        chunk_id: Optional[str] = None,
        surrounding_context: str = ""
    ) -> AggregatedDetection:
        """
        Executes hybrid detection over a chunk:
        1. Runs neural sequence classifier with token attribution span.
        2. Runs deterministic statistical representation analyzer.
        3. Calibrates confidence score via weighted fusion.
        4. Maps to granular taxonomy dimensions.
        5. Deduplicates evidence spans.
        """
        clean_text = text.strip() if text else ""
        if not clean_text:
            return AggregatedDetection(
                chunk_id=chunk_id,
                classification=BinaryClassification.NEUTRAL,
                contextual_outcome=ContextualOutcome.GENDER_BALANCED,
                confidence=0.0,
                neural_confidence=0.0,
                statistical_score=0.5,
                is_suppressed=True,
                reasons_for_decision=["Empty input passage"]
            )

        # 1. Neural classification
        neural_res: DetectionResult = self.classifier.predict(clean_text, surrounding_context=surrounding_context)
        p_neural = neural_res.confidence if neural_res.classification == "stereotype" else (1.0 - neural_res.confidence)

        # 2. Statistical analysis
        stat_res: StatisticalAnalysisResult = self.statistical_analyzer.analyze(clean_text)
        s_stat = self._compute_statistical_score(stat_res)

        # 3. Weighted Score Fusion
        fused_conf = (self.w_neural * p_neural) + (self.w_stat * s_stat)
        fused_conf = round(max(0.0, min(1.0, fused_conf)), 4)

        # 4. Evidence Span Fusion & Deduplication
        merged_span, start_c, end_c = self._merge_evidence_spans(clean_text, neural_res, stat_res)

        # 5. Threshold Gating & Taxonomy Attribution
        reasons = []
        is_suppressed = False
        contextual_outcome = None
        taxonomy_category = None

        if fused_conf >= self.confidence_threshold:
            classification = BinaryClassification.STEREOTYPE
            reasons.append(f"Fused confidence ({fused_conf:.3f}) exceeds threshold ({self.confidence_threshold})")
            
            # Map granular taxonomy dimension based on statistical co-occurrences
            taxonomy_category = self._assign_taxonomy_category(stat_res)
        elif neural_res.classification == "anti_stereotype" and (1.0 - fused_conf) >= self.confidence_threshold:
            classification = BinaryClassification.ANTI_STEREOTYPE
            contextual_outcome = ContextualOutcome.COUNTER_NARRATIVE
            reasons.append(f"Anti-stereotype probability ({1.0 - fused_conf:.3f}) exceeds threshold ({self.confidence_threshold})")
        else:
            classification = BinaryClassification.NEUTRAL
            contextual_outcome = ContextualOutcome.GENDER_BALANCED
            is_suppressed = True
            reasons.append(f"Fused confidence ({fused_conf:.3f}) below threshold ({self.confidence_threshold}); suppressed as neutral")

        # If representation is heavily skewed, flag taxonomy category
        if stat_res.is_imbalanced and taxonomy_category is None:
            taxonomy_category = TaxonomyCategory.REPRESENTATIONAL_IMBALANCE
            reasons.append(f"Marked as representation imbalance due to skewed reference ratio ({stat_res.male_ratio:.2f})")

        return AggregatedDetection(
            chunk_id=chunk_id,
            classification=classification,
            taxonomy_category=taxonomy_category,
            contextual_outcome=contextual_outcome,
            confidence=fused_conf if classification == BinaryClassification.STEREOTYPE else round(1.0 - fused_conf, 4),
            neural_confidence=round(p_neural, 4),
            statistical_score=round(s_stat, 4),
            evidence_span=merged_span,
            start_char=start_c,
            end_char=end_c,
            surrounding_context=surrounding_context or clean_text,
            detection_method="HYBRID",
            is_suppressed=is_suppressed,
            reasons_for_decision=reasons
        )

    def _compute_statistical_score(self, stat_res: StatisticalAnalysisResult) -> float:
        """
        Converts deterministic signals into a normalized continuous score S_stat in [0.0, 1.0].
        Higher values favor stereotypical interpretation.
        """
        if stat_res.signal_type == "POTENTIAL_STEREOTYPICAL_ASSOCIATION":
            # Scale with number of stereotypical co-occurrence candidates
            stereo_count = sum(1 for c in stat_res.co_occurrences if c.evidence_indicator == "POTENTIAL_STEREOTYPICAL_ASSOCIATION")
            return min(0.90, 0.70 + (stereo_count * 0.05))
        elif stat_res.signal_type == "POTENTIAL_COUNTER_STEREOTYPICAL_ASSOCIATION":
            return 0.20
        elif stat_res.signal_type == "MIXED_SIGNALS":
            return 0.50
        elif stat_res.is_imbalanced:
            # Moderate signal if text has extreme quantitative reference skew
            return 0.60
        else:
            return 0.50

    def _assign_taxonomy_category(self, stat_res: StatisticalAnalysisResult) -> Optional[TaxonomyCategory]:
        """Maps statistical signals to taxonomy categories."""
        for c in stat_res.co_occurrences:
            if c.category == "DOMESTIC_ROLE":
                return TaxonomyCategory.DOMESTIC_ROLES
            elif c.category == "OCCUPATIONAL_BIAS":
                return TaxonomyCategory.OCCUPATIONAL_BIAS

        if stat_res.is_imbalanced:
            return TaxonomyCategory.REPRESENTATIONAL_IMBALANCE

        # Default category if unassigned
        return TaxonomyCategory.ABILITY_INTELLECTUAL_BIAS

    def _merge_evidence_spans(
        self,
        text: str,
        neural_res: DetectionResult,
        stat_res: StatisticalAnalysisResult
    ) -> Tuple[str, int, int]:
        """
        Deduplicates and merges overlapping character offset spans between
        neural attention attribution and statistical co-occurrences.
        """
        spans: List[Tuple[int, int]] = []

        if neural_res.evidence_span and neural_res.start_char is not None and neural_res.end_char is not None:
            spans.append((neural_res.start_char, neural_res.end_char))

        # Add co-occurrence snippets
        for co in stat_res.co_occurrences:
            m = re.search(re.escape(co.context_snippet), text, re.IGNORECASE)
            if m:
                spans.append((m.start(), m.end()))

        if not spans:
            return text[:60], 0, min(60, len(text))

        # Sort spans by start position
        spans.sort(key=lambda s: s[0])

        # Merge overlapping or contiguous spans
        merged_start, merged_end = spans[0]
        for s, e in spans[1:]:
            if s <= merged_end + 3:  # within 3 characters
                merged_end = max(merged_end, e)
            else:
                break

        merged_start = max(0, min(merged_start, len(text)))
        merged_end = max(merged_start, min(merged_end, len(text)))
        merged_span = text[merged_start:merged_end].strip()

        return merged_span, merged_start, merged_end

    def persist_to_database(
        self,
        db_session: Any,
        detection: AggregatedDetection
    ) -> Any:
        """Persists the aggregated detection record into the database 'detections' table."""
        try:
            from app.models.detection import Detection
        except ImportError:
            from backend.app.models.detection import Detection

        db_item = Detection(
            chunk_id=detection.chunk_id or "00000000-0000-0000-0000-000000000000",
            classification=detection.classification,
            taxonomy_category=detection.taxonomy_category,
            contextual_outcome=detection.contextual_outcome,
            confidence=detection.confidence,
            evidence_span=detection.evidence_span or "No span identified",
            start_char=detection.start_char,
            end_char=detection.end_char,
            surrounding_context=detection.surrounding_context,
            detection_method=detection.detection_method,
            is_suppressed=detection.is_suppressed
        )
        db_session.add(db_item)
        db_session.commit()
        db_session.refresh(db_item)
        return db_item
