import re
import time
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field


class CoOccurrence(BaseModel):
    """
    Represents a surface lexical co-occurrence of a gender term with a role/activity keyword
    within a sliding token proximity window.
    
    LIMITATION NOTE:
    Lexical proximity does NOT establish grammatical subject-action dependency or semantic endorsement.
    It indicates proximity only.
    """
    gender_term: str
    gender: str  # "male" or "female"
    associated_term: str
    category: str  # "DOMESTIC_ROLE", "OCCUPATIONAL_BIAS"
    evidence_indicator: str = Field(
        description="POTENTIAL_STEREOTYPICAL_ASSOCIATION, POTENTIAL_COUNTER_STEREOTYPICAL_ASSOCIATION, or INSUFFICIENT_CONTEXT"
    )
    attribution_confidence: str = Field(
        default="AMBIGUOUS_LEXICAL_PROXIMITY",
        description="AMBIGUOUS_LEXICAL_PROXIMITY: Syntax/dependency parsing required to verify true subject attribution."
    )
    context_snippet: str
    token_distance: int


class StatisticalAnalysisResult(BaseModel):
    """
    Structured deterministic gender reference frequency and lexical co-occurrence analysis.
    
    CRITICAL METHODOLOGICAL BOUNDARIES:
    1. Reference Frequency != Unique Characters:
       This metric counts occurrences of gendered lexical tokens (pronouns and kinship nouns).
       Repeated references to the same individual (e.g. 'he... he... his...') are counted multiple times.
       Unique character count requires entity coreference resolution.
    2. Representation Imbalance != Stereotype:
       A skewed gender reference frequency ratio does not establish stereotype presence.
       It indicates quantitative asymmetry in lexical mentions.
    3. Lexical Co-occurrence != Confirmed Stereotype:
       Surface keyword proximity within a window cannot confirm gender-role attribution.
       It provides initial candidate evidence indicators for subsequent neural/syntactic verification.
    """
    # Reference counts (Token frequency, NOT unique character count)
    male_count: int = Field(default=0, description="Total count of male lexical indicators")
    female_count: int = Field(default=0, description="Total count of female lexical indicators")
    male_pronoun_count: int = Field(default=0)
    female_pronoun_count: int = Field(default=0)
    male_noun_count: int = Field(default=0)
    female_noun_count: int = Field(default=0)
    total_gender_count: int = Field(default=0)

    # Renamed explicit metric dictionary to prevent confusion with unique entity count
    gendered_reference_frequency: Dict[str, int] = Field(
        default_factory=dict,
        description="Frequencies of gendered token mentions: {'male': N, 'female': N, 'total': N}"
    )

    # Statistical representation ratios
    male_ratio: float = Field(default=0.5, description="R_gender = N_male / (N_male + N_female)")
    female_ratio: float = Field(default=0.5)

    # Representation skew status (Configurable heuristic thresholds, NOT universal ground truth)
    imbalance_status: str = Field(
        default="INSUFFICIENT_DATA",
        description="BALANCED, MALE_SKEWED, FEMALE_SKEWED, or INSUFFICIENT_DATA"
    )
    is_imbalanced: bool = Field(
        default=False,
        description="Heuristic flag: True if mentions >= min_mentions and ratio exceeds upper/lower thresholds"
    )

    # Co-occurrence signals (Potential evidence indicators only)
    signal_type: str = Field(
        default="NO_STRONG_LEXICAL_SIGNAL",
        description="POTENTIAL_STEREOTYPICAL_ASSOCIATION, POTENTIAL_COUNTER_STEREOTYPICAL_ASSOCIATION, MIXED_SIGNALS, INSUFFICIENT_CONTEXT, or NO_STRONG_LEXICAL_SIGNAL"
    )
    co_occurrences: List[CoOccurrence] = Field(default_factory=list)
    male_terms_found: List[str] = Field(default_factory=list)
    female_terms_found: List[str] = Field(default_factory=list)

    # Audit & limitation metadata
    warnings: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(
        default_factory=lambda: [
            "Measures token reference frequency, not unique character counts (no coreference resolution).",
            "Co-occurrences are based solely on token proximity window, not syntactic dependency parse.",
            "Lexical matching does not detect negation, hypothetical modality, or meta-discussions rejecting bias.",
            "Representation imbalance is strictly quantitative and does not independently prove stereotype presence."
        ]
    )
    execution_time_ms: float = Field(default=0.0)


class StatisticalRepresentationAnalyzer:
    """
    Deterministic Lightweight NLP Analyzer for Textbook Gender Reference Frequency.
    
    Computes:
    - Gendered token reference frequencies (pronouns and kinship nouns)
    - Male-to-Female Reference Ratio:
        R_gender = N_male / (N_male + N_female)
    - Threshold-gated Reference Imbalance (Initial heuristic thresholds: default 0.75 / 0.25, min 4 mentions)
    - Proximity-based candidate co-occurrence indicators with explicit attribution limitations
    """

    MALE_PRONOUNS = {"he", "him", "his", "himself"}
    FEMALE_PRONOUNS = {"she", "her", "hers", "herself"}

    MALE_NOUNS = {
        "father", "dad", "brother", "son", "boy", "boys", "man", "men",
        "grandfather", "grandpa", "uncle", "husband", "king", "prince",
        "monk", "gentleman", "male", "sir", "nephew"
    }

    FEMALE_NOUNS = {
        "mother", "mom", "sister", "daughter", "girl", "girls", "woman", "women",
        "grandmother", "grandma", "aunt", "wife", "queen", "princess",
        "nun", "lady", "female", "madam", "niece"
    }

    # Traditional domestic & caregiving terms
    DOMESTIC_CARE_TERMS = {
        "cook", "cooking", "cooked", "clean", "cleaning", "cleaned",
        "sweep", "sweeping", "swept", "wash", "washing", "washed",
        "iron", "ironing", "ironed", "mop", "mopping", "mopped",
        "sew", "sewing", "sewed", "serve", "serving", "served",
        "dishes", "laundry", "kitchen", "household", "chores",
        "caregiver", "babysitter", "maid", "housewife", "nurse", "nursing"
    }

    # Traditional leadership, STEM, and high-prestige occupations
    LEADERSHIP_STEM_TERMS = {
        "engineer", "engineering", "pilot", "doctor", "scientist", "scientific",
        "mechanic", "driver", "soldier", "army", "officer", "captain",
        "carpenter", "technician", "politician", "president", "minister",
        "judge", "architect", "manager", "director", "chief", "executive", "boss"
    }

    def __init__(
        self,
        skew_upper_threshold: float = 0.75,
        skew_lower_threshold: float = 0.25,
        min_mentions_for_skew: int = 4,
        cooccurrence_window: int = 5
    ):
        self.skew_upper_threshold = skew_upper_threshold
        self.skew_lower_threshold = skew_lower_threshold
        self.min_mentions_for_skew = min_mentions_for_skew
        self.cooccurrence_window = cooccurrence_window

        # Precompile token matching regex for rapid execution (<5 ms)
        self._word_re = re.compile(r"\b[A-Za-z]+(?:'[A-Za-z]+)?\b")

    def analyze(self, text: str) -> StatisticalAnalysisResult:
        """
        Analyzes text for deterministic gender reference frequencies and proximity co-occurrences.
        """
        start_time = time.perf_counter()

        if not text or not text.strip():
            return StatisticalAnalysisResult(
                signal_type="INSUFFICIENT_CONTEXT",
                execution_time_ms=0.0
            )

        raw_tokens = self._word_re.findall(text)
        lower_tokens = [t.lower() for t in raw_tokens]

        male_pronouns_found = []
        female_pronouns_found = []
        male_nouns_found = []
        female_nouns_found = []

        token_gender_map: List[Tuple[int, str, str]] = []  # (index, token, gender)

        for idx, token in enumerate(lower_tokens):
            if token in self.MALE_PRONOUNS:
                male_pronouns_found.append(token)
                token_gender_map.append((idx, token, "male"))
            elif token in self.FEMALE_PRONOUNS:
                female_pronouns_found.append(token)
                token_gender_map.append((idx, token, "female"))
            elif token in self.MALE_NOUNS:
                male_nouns_found.append(token)
                token_gender_map.append((idx, token, "male"))
            elif token in self.FEMALE_NOUNS:
                female_nouns_found.append(token)
                token_gender_map.append((idx, token, "female"))

        male_count = len(male_pronouns_found) + len(male_nouns_found)
        female_count = len(female_pronouns_found) + len(female_nouns_found)
        total_gender_count = male_count + female_count

        if total_gender_count > 0:
            male_ratio = round(male_count / total_gender_count, 4)
            female_ratio = round(female_count / total_gender_count, 4)
        else:
            male_ratio = 0.5
            female_ratio = 0.5

        # Determine reference imbalance status
        warnings = []
        if total_gender_count < self.min_mentions_for_skew:
            imbalance_status = "INSUFFICIENT_DATA"
            is_imbalanced = False
            if total_gender_count > 0:
                warnings.append(
                    f"Total gender references ({total_gender_count}) below minimum threshold ({self.min_mentions_for_skew}); ratio is statistically unreliable."
                )
        elif male_ratio > self.skew_upper_threshold:
            imbalance_status = "MALE_SKEWED"
            is_imbalanced = True
        elif male_ratio < self.skew_lower_threshold:
            imbalance_status = "FEMALE_SKEWED"
            is_imbalanced = True
        else:
            imbalance_status = "BALANCED"
            is_imbalanced = False

        # Compute proximity-based co-occurrences
        co_occurrences = self._detect_co_occurrences(lower_tokens, raw_tokens, token_gender_map)

        if len(co_occurrences) > 0:
            warnings.append(
                "Co-occurrences reflect window proximity and are classified as AMBIGUOUS_LEXICAL_PROXIMITY. Grammatical subject attribution is not verified."
            )

        # Derive potential evidence signal type
        if total_gender_count == 0:
            signal_type = "NO_STRONG_LEXICAL_SIGNAL"
        elif len(co_occurrences) == 0:
            signal_type = "NO_STRONG_LEXICAL_SIGNAL"
        else:
            indicators = set(c.evidence_indicator for c in co_occurrences)
            if "POTENTIAL_STEREOTYPICAL_ASSOCIATION" in indicators and "POTENTIAL_COUNTER_STEREOTYPICAL_ASSOCIATION" in indicators:
                signal_type = "MIXED_SIGNALS"
            elif "POTENTIAL_STEREOTYPICAL_ASSOCIATION" in indicators:
                signal_type = "POTENTIAL_STEREOTYPICAL_ASSOCIATION"
            elif "POTENTIAL_COUNTER_STEREOTYPICAL_ASSOCIATION" in indicators:
                signal_type = "POTENTIAL_COUNTER_STEREOTYPICAL_ASSOCIATION"
            else:
                signal_type = "NO_STRONG_LEXICAL_SIGNAL"

        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 3)

        return StatisticalAnalysisResult(
            male_count=male_count,
            female_count=female_count,
            male_pronoun_count=len(male_pronouns_found),
            female_pronoun_count=len(female_pronouns_found),
            male_noun_count=len(male_nouns_found),
            female_noun_count=len(female_nouns_found),
            total_gender_count=total_gender_count,
            gendered_reference_frequency={
                "male": male_count,
                "female": female_count,
                "total": total_gender_count
            },
            male_ratio=male_ratio,
            female_ratio=female_ratio,
            imbalance_status=imbalance_status,
            is_imbalanced=is_imbalanced,
            signal_type=signal_type,
            co_occurrences=co_occurrences,
            male_terms_found=male_pronouns_found + male_nouns_found,
            female_terms_found=female_pronouns_found + female_nouns_found,
            warnings=warnings,
            execution_time_ms=elapsed_ms
        )

    def _detect_co_occurrences(
        self,
        lower_tokens: List[str],
        raw_tokens: List[str],
        token_gender_map: List[Tuple[int, str, str]]
    ) -> List[CoOccurrence]:
        """Identifies gender terms co-occurring within sliding proximity window with role keywords."""
        results: List[CoOccurrence] = []
        n_tokens = len(lower_tokens)

        for g_idx, g_term, gender in token_gender_map:
            start_w = max(0, g_idx - self.cooccurrence_window)
            end_w = min(n_tokens, g_idx + self.cooccurrence_window + 1)

            for target_idx in range(start_w, end_w):
                if target_idx == g_idx:
                    continue

                target_token = lower_tokens[target_idx]
                dist = abs(target_idx - g_idx)

                # Domestic role proximity
                if target_token in self.DOMESTIC_CARE_TERMS:
                    indicator = (
                        "POTENTIAL_STEREOTYPICAL_ASSOCIATION"
                        if gender == "female"
                        else "POTENTIAL_COUNTER_STEREOTYPICAL_ASSOCIATION"
                    )
                    snippet_start = max(0, min(g_idx, target_idx) - 2)
                    snippet_end = min(n_tokens, max(g_idx, target_idx) + 3)
                    snippet = " ".join(raw_tokens[snippet_start:snippet_end])

                    results.append(CoOccurrence(
                        gender_term=raw_tokens[g_idx],
                        gender=gender,
                        associated_term=raw_tokens[target_idx],
                        category="DOMESTIC_ROLE",
                        evidence_indicator=indicator,
                        attribution_confidence="AMBIGUOUS_LEXICAL_PROXIMITY",
                        context_snippet=snippet,
                        token_distance=dist
                    ))

                # Leadership / STEM / Occupational proximity
                elif target_token in self.LEADERSHIP_STEM_TERMS:
                    indicator = (
                        "POTENTIAL_STEREOTYPICAL_ASSOCIATION"
                        if gender == "male"
                        else "POTENTIAL_COUNTER_STEREOTYPICAL_ASSOCIATION"
                    )
                    snippet_start = max(0, min(g_idx, target_idx) - 2)
                    snippet_end = min(n_tokens, max(g_idx, target_idx) + 3)
                    snippet = " ".join(raw_tokens[snippet_start:snippet_end])

                    results.append(CoOccurrence(
                        gender_term=raw_tokens[g_idx],
                        gender=gender,
                        associated_term=raw_tokens[target_idx],
                        category="OCCUPATIONAL_BIAS",
                        evidence_indicator=indicator,
                        attribution_confidence="AMBIGUOUS_LEXICAL_PROXIMITY",
                        context_snippet=snippet,
                        token_distance=dist
                    ))

        return results
