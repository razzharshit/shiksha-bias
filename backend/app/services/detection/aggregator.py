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


class EvidenceItem(BaseModel):
    """Structured evidence item preserving provenance, location, and attribution warnings."""
    source: str = Field(description="'NEURAL_MODEL' or 'STATISTICAL_ANALYZER'")
    matched_text: str
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    associated_signal: str
    limitations_or_warnings: str


class ReviewRecommendation(BaseModel):
    """Human-review recommendation detailing whether, why, and which sources flagged a passage."""
    requires_human_review: bool = False
    priority: str = Field(default="NONE", description="'HIGH', 'MEDIUM', 'LOW', or 'NONE'")
    flagged_reasons: List[str] = Field(default_factory=list)
    contributing_sources: List[str] = Field(default_factory=list)


class AggregatedDetection(BaseModel):
    """
    Multi-dimensional detection result keeping neural classification, statistical evidence,
    representation analysis, and human-review recommendations strictly independent.
    """
    chunk_id: Optional[str] = None

    # Dimension 1: Neural Binary Classification (Preserves both class probabilities)
    neural_classification: BinaryClassification = BinaryClassification.NEUTRAL
    neural_prob_stereotype: float = Field(default=0.5, ge=0.0, le=1.0)
    neural_prob_anti_stereotype: float = Field(default=0.5, ge=0.0, le=1.0)
    classification: BinaryClassification = BinaryClassification.NEUTRAL  # Overall output classification

    # Dimension 2: Statistical Lexical Evidence
    statistical_signal_type: str = "NO_STRONG_LEXICAL_SIGNAL"
    statistical_score: float = Field(default=0.5, ge=0.0, le=1.0)
    co_occurrences: List[CoOccurrence] = Field(default_factory=list)

    # Dimension 3: Representation Analysis (Strictly separated from stereotype score)
    gendered_reference_frequency: Dict[str, int] = Field(default_factory=dict)
    male_ratio: float = 0.5
    female_ratio: float = 0.5
    is_representation_imbalanced: bool = False
    representation_imbalance_status: str = "INSUFFICIENT_DATA"

    # Dimension 4: Candidate Taxonomy & Contextual Outcomes (Evidence-dependent)
    candidate_taxonomy_categories: List[TaxonomyCategory] = Field(default_factory=list)
    taxonomy_category: Optional[TaxonomyCategory] = None  # Suggested category only if evidence exists
    contextual_outcome: Optional[ContextualOutcome] = None

    # Heuristic Score Fusion (Explicitly NOT claimed as calibrated probability)
    heuristic_fusion_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Heuristic weighted combination of neural probability and lexical signals. Not a calibrated probability."
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Preserved for database schema compatibility; equals heuristic_fusion_score"
    )

    # Evidence Provenance (Preserved separately by source)
    evidence_items: List[EvidenceItem] = Field(default_factory=list)
    evidence_span: str = Field(default="", description="Primary attribution span for database/UI compatibility")
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    surrounding_context: str = ""

    # Dimension 5: Human-Review Recommendation
    review_recommendation: ReviewRecommendation = Field(default_factory=ReviewRecommendation)
    is_suppressed: bool = False
    detection_method: str = "HYBRID"

    def get_audit_metadata(self) -> Dict[str, Any]:
        """Serializes full multi-dimensional detection state for database audit trail."""
        return {
            "neural_classification": str(self.neural_classification.value if hasattr(self.neural_classification, 'value') else self.neural_classification),
            "neural_prob_stereotype": self.neural_prob_stereotype,
            "neural_prob_anti_stereotype": self.neural_prob_anti_stereotype,
            "statistical_signal_type": self.statistical_signal_type,
            "statistical_score": self.statistical_score,
            "heuristic_fusion_score": self.heuristic_fusion_score,
            "gendered_reference_frequency": self.gendered_reference_frequency,
            "representation_ratio": {"male": self.male_ratio, "female": self.female_ratio},
            "is_representation_imbalanced": self.is_representation_imbalanced,
            "candidate_taxonomy_categories": [c.value if hasattr(c, 'value') else str(c) for c in self.candidate_taxonomy_categories],
            "review_recommendation": self.review_recommendation.model_dump() if hasattr(self.review_recommendation, 'model_dump') else self.review_recommendation.dict(),
            "evidence_items": [e.model_dump() if hasattr(e, 'model_dump') else e.dict() for e in self.evidence_items]
        }


class HybridDetectionAggregator:
    """
    Combines neural transformer representations with deterministic statistical signals.
    
    CRITICAL METHODOLOGICAL GUARANTEES:
    1. Score Terminology:
       The fusion formula:
           heuristic_fusion_score = w_neural * P_neural_stereotype + w_stat * S_stat
       is an experimental heuristic score, NOT a calibrated real-world probability.
    2. Representation Imbalance Independence:
       Reference frequency skew NEVER increases S_stat or the stereotype classification score.
       Representation statistics are preserved as an independent descriptive dimension.
    3. Evidence-Dependent Taxonomy:
       Taxonomy categories are suggested ONLY when matching lexical or representation evidence is present.
       ABILITY_INTELLECTUAL_BIAS is never defaulted without affirmative evidence.
    4. Threshold & Suppression:
       Scores below threshold suppress bias alerts, but do NOT assert GENDER_BALANCED.
       Insufficient evidence is strictly distinguished from verified balanced content.
    5. Provenance Preservation:
       Neural model attribution and statistical co-occurrence items maintain separate provenance.
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
        Executes multi-dimensional hybrid analysis on a text chunk.
        """
        clean_text = text.strip() if text else ""
        if not clean_text:
            return AggregatedDetection(
                chunk_id=chunk_id,
                classification=BinaryClassification.NEUTRAL,
                neural_classification=BinaryClassification.NEUTRAL,
                contextual_outcome=None,
                confidence=0.0,
                heuristic_fusion_score=0.0,
                neural_prob_stereotype=0.0,
                neural_prob_anti_stereotype=0.0,
                statistical_score=0.5,
                is_suppressed=True,
                review_recommendation=ReviewRecommendation(
                    requires_human_review=False,
                    priority="NONE",
                    flagged_reasons=["Empty input passage"],
                    contributing_sources=[]
                )
            )

        # 1. Neural Classification & Both Probabilities
        neural_res: DetectionResult = self.classifier.predict(clean_text, surrounding_context=surrounding_context)
        p_stereo = getattr(neural_res, 'prob_stereotype', 0.88 if neural_res.classification == "stereotype" else 0.12)
        p_anti = getattr(neural_res, 'prob_anti_stereotype', 0.88 if neural_res.classification == "anti_stereotype" else 0.12)

        if neural_res.classification == "stereotype":
            neural_class = BinaryClassification.STEREOTYPE
        elif neural_res.classification == "anti_stereotype":
            neural_class = BinaryClassification.ANTI_STEREOTYPE
        else:
            neural_class = BinaryClassification.NEUTRAL

        # Neural Evidence Item
        evidence_items: List[EvidenceItem] = []
        if neural_res.evidence_span:
            evidence_items.append(EvidenceItem(
                source="NEURAL_MODEL",
                matched_text=neural_res.evidence_span,
                start_char=neural_res.start_char,
                end_char=neural_res.end_char,
                associated_signal=f"MODEL_ATTRIBUTION_SPAN (pred={neural_res.classification}, conf={neural_res.confidence:.2f})",
                limitations_or_warnings="Attribution derived from attention weights; represents model focus, not verified causal explanation."
            ))

        # 2. Statistical Analysis & Provenance
        stat_res: StatisticalAnalysisResult = self.statistical_analyzer.analyze(clean_text)

        for co in stat_res.co_occurrences:
            evidence_items.append(EvidenceItem(
                source="STATISTICAL_ANALYZER",
                matched_text=f"{co.gender_term} ... {co.associated_term}",
                associated_signal=co.evidence_indicator,
                limitations_or_warnings="Ambiguous lexical proximity; syntactic dependency parsing required to verify true subject-verb attribution."
            ))

        # 3. Statistical Score S_stat (Exclusively based on co-occurrence, NEVER on representation skew)
        s_stat = self._compute_statistical_score(stat_res)

        # 4. Heuristic Fusion Score (Documented as heuristic, not calibrated probability)
        heuristic_score = (self.w_neural * p_stereo) + (self.w_stat * s_stat)
        heuristic_score = round(max(0.0, min(1.0, heuristic_score)), 4)

        # 5. Evidence-Dependent Candidate Taxonomy Categories
        candidate_taxonomies: List[TaxonomyCategory] = []
        has_domestic = any(c.category == "DOMESTIC_ROLE" for c in stat_res.co_occurrences)
        has_occupational = any(c.category == "OCCUPATIONAL_BIAS" for c in stat_res.co_occurrences)

        if has_domestic:
            candidate_taxonomies.append(TaxonomyCategory.DOMESTIC_ROLES)
        if has_occupational:
            candidate_taxonomies.append(TaxonomyCategory.OCCUPATIONAL_BIAS)
        if stat_res.is_imbalanced:
            candidate_taxonomies.append(TaxonomyCategory.REPRESENTATIONAL_IMBALANCE)

        # Suggested primary taxonomy category (only if supported by candidate evidence)
        suggested_category: Optional[TaxonomyCategory] = None
        if TaxonomyCategory.DOMESTIC_ROLES in candidate_taxonomies:
            suggested_category = TaxonomyCategory.DOMESTIC_ROLES
        elif TaxonomyCategory.OCCUPATIONAL_BIAS in candidate_taxonomies:
            suggested_category = TaxonomyCategory.OCCUPATIONAL_BIAS
        elif TaxonomyCategory.REPRESENTATIONAL_IMBALANCE in candidate_taxonomies:
            suggested_category = TaxonomyCategory.REPRESENTATIONAL_IMBALANCE
        # ABILITY_INTELLECTUAL_BIAS is NEVER defaulted without explicit evidence

        # 6. Classification & Threshold Evaluation
        flagged_reasons: List[str] = []
        contributing_sources: List[str] = []
        contextual_outcome: Optional[ContextualOutcome] = None

        if heuristic_score >= self.confidence_threshold:
            classification = BinaryClassification.STEREOTYPE
            is_suppressed = False
            flagged_reasons.append(f"Heuristic fusion score ({heuristic_score:.2f}) meets stereotype threshold ({self.confidence_threshold:.2f}).")
            contributing_sources.append("NEURAL_MODEL")
            if stat_res.signal_type == "POTENTIAL_STEREOTYPICAL_ASSOCIATION":
                contributing_sources.append("STATISTICAL_ANALYZER")
        elif p_anti >= self.confidence_threshold and heuristic_score < (1.0 - self.confidence_threshold):
            classification = BinaryClassification.ANTI_STEREOTYPE
            contextual_outcome = ContextualOutcome.COUNTER_NARRATIVE
            is_suppressed = False
            flagged_reasons.append(f"Neural model indicates counter-stereotypical narrative (p={p_anti:.2f}).")
            contributing_sources.append("NEURAL_MODEL")
        else:
            classification = BinaryClassification.NEUTRAL
            is_suppressed = True
            # CRITICAL: Do NOT assign GENDER_BALANCED simply because score is below threshold
            contextual_outcome = None
            flagged_reasons.append("Insufficient evidence to establish stereotype; suppressed as neutral.")

        # 7. Human-Review Recommendation
        requires_review = False
        priority = "NONE"

        if classification == BinaryClassification.STEREOTYPE:
            requires_review = True
            priority = "HIGH" if heuristic_score >= 0.80 else "MEDIUM"
        elif stat_res.is_imbalanced:
            # Flag representation imbalance separately for human review
            requires_review = True
            if priority == "NONE":
                priority = "LOW"
            flagged_reasons.append(f"Representation reference skew detected (male ratio: {stat_res.male_ratio:.2f}).")
            contributing_sources.append("REPRESENTATION_ANALYSIS")

        review_rec = ReviewRecommendation(
            requires_human_review=requires_review,
            priority=priority,
            flagged_reasons=flagged_reasons,
            contributing_sources=list(set(contributing_sources))
        )

        # Primary evidence span for UI/Database compatibility
        primary_span = neural_res.evidence_span
        start_char = neural_res.start_char
        end_char = neural_res.end_char

        if not primary_span and stat_res.co_occurrences:
            primary_span = stat_res.co_occurrences[0].context_snippet
            m = re.search(re.escape(primary_span), clean_text)
            if m:
                start_char, end_char = m.start(), m.end()

        return AggregatedDetection(
            chunk_id=chunk_id,
            neural_classification=neural_class,
            neural_prob_stereotype=p_stereo,
            neural_prob_anti_stereotype=p_anti,
            classification=classification,
            statistical_signal_type=stat_res.signal_type,
            statistical_score=s_stat,
            co_occurrences=stat_res.co_occurrences,
            gendered_reference_frequency=stat_res.gendered_reference_frequency,
            male_ratio=stat_res.male_ratio,
            female_ratio=stat_res.female_ratio,
            is_representation_imbalanced=stat_res.is_imbalanced,
            representation_imbalance_status=stat_res.imbalance_status,
            candidate_taxonomy_categories=candidate_taxonomies,
            taxonomy_category=suggested_category,
            contextual_outcome=contextual_outcome,
            heuristic_fusion_score=heuristic_score,
            confidence=heuristic_score,
            evidence_items=evidence_items,
            evidence_span=primary_span or clean_text[:60],
            start_char=start_char,
            end_char=end_char,
            surrounding_context=surrounding_context or clean_text,
            review_recommendation=review_rec,
            is_suppressed=is_suppressed,
            detection_method="HYBRID"
        )

    def _compute_statistical_score(self, stat_res: StatisticalAnalysisResult) -> float:
        """
        Converts deterministic signals into S_stat in [0.0, 1.0].
        CRITICAL: Skewed representation ratio NEVER increases this score.
        """
        if stat_res.signal_type == "POTENTIAL_STEREOTYPICAL_ASSOCIATION":
            stereo_count = sum(1 for c in stat_res.co_occurrences if c.evidence_indicator == "POTENTIAL_STEREOTYPICAL_ASSOCIATION")
            return min(0.90, 0.70 + (stereo_count * 0.05))
        elif stat_res.signal_type == "POTENTIAL_COUNTER_STEREOTYPICAL_ASSOCIATION":
            return 0.20
        elif stat_res.signal_type == "MIXED_SIGNALS":
            return 0.50
        else:
            return 0.50

    def persist_to_database(
        self,
        db_session: Any,
        detection: AggregatedDetection
    ) -> Any:
        """
        Persists the aggregated detection into the database 'detections' table.
        Avoids duplicates: updates existing record if (chunk_id, evidence_span) already exists.
        """
        try:
            from app.models.detection import Detection
        except ImportError:
            from backend.app.models.detection import Detection

        cid = detection.chunk_id or "00000000-0000-0000-0000-000000000000"
        span = detection.evidence_span or "No span identified"

        # Check for existing record to avoid duplicate entries on re-processing
        existing = db_session.query(Detection).filter_by(
            chunk_id=cid,
            evidence_span=span
        ).first()

        audit_data = detection.get_audit_metadata()

        if existing:
            existing.classification = detection.classification
            existing.taxonomy_category = detection.taxonomy_category
            existing.contextual_outcome = detection.contextual_outcome
            existing.confidence = detection.confidence
            existing.start_char = detection.start_char
            existing.end_char = detection.end_char
            existing.surrounding_context = detection.surrounding_context
            existing.detection_method = detection.detection_method
            existing.is_suppressed = detection.is_suppressed
            if hasattr(existing, 'audit_metadata'):
                existing.audit_metadata = audit_data

            db_session.commit()
            db_session.refresh(existing)
            return existing

        db_item = Detection(
            chunk_id=cid,
            classification=detection.classification,
            taxonomy_category=detection.taxonomy_category,
            contextual_outcome=detection.contextual_outcome,
            confidence=detection.confidence,
            evidence_span=span,
            start_char=detection.start_char,
            end_char=detection.end_char,
            surrounding_context=detection.surrounding_context,
            detection_method=detection.detection_method,
            is_suppressed=detection.is_suppressed,
            audit_metadata=audit_data
        )
        db_session.add(db_item)
        db_session.commit()
        db_session.refresh(db_item)
        return db_item
