import pytest
from backend.app.services.detection.statistical import StatisticalRepresentationAnalyzer


@pytest.fixture
def analyzer():
    return StatisticalRepresentationAnalyzer(
        skew_upper_threshold=0.75,
        skew_lower_threshold=0.25,
        min_mentions_for_skew=4,
        cooccurrence_window=5
    )


# -------------------------------------------------------------------------
# Core Baseline Tests (Preserved Functionality)
# -------------------------------------------------------------------------

def test_male_dominated_passage(analyzer):
    text = (
        "The king summoned his sons and his brother to the council. "
        "He told the boy that his father had been a brave soldier and an officer. "
        "He believed men should lead the kingdom."
    )
    result = analyzer.analyze(text)

    assert result.male_count >= 8
    assert result.female_count == 0
    assert result.total_gender_count >= 8
    assert result.gendered_reference_frequency["male"] == result.male_count
    assert result.gendered_reference_frequency["female"] == 0
    assert result.male_ratio == 1.0
    assert result.is_imbalanced is True
    assert result.imbalance_status == "MALE_SKEWED"


def test_female_dominated_passage(analyzer):
    text = (
        "The queen called her daughter and her sister. "
        "She praised the girl for her dedication. "
        "Her mother smiled as she watched women organize the event."
    )
    result = analyzer.analyze(text)

    assert result.female_count >= 7
    assert result.male_count == 0
    assert result.total_gender_count >= 7
    assert result.gendered_reference_frequency["female"] == result.female_count
    assert result.gendered_reference_frequency["male"] == 0
    assert result.female_ratio == 1.0
    assert result.male_ratio == 0.0
    assert result.is_imbalanced is True
    assert result.imbalance_status == "FEMALE_SKEWED"


def test_balanced_passage(analyzer):
    text = (
        "The boy and the girl studied together. "
        "He asked his sister for assistance, and she helped her brother willingly. "
        "Their mother and father were both proud."
    )
    result = analyzer.analyze(text)

    assert result.male_count > 0
    assert result.female_count > 0
    assert result.total_gender_count >= 4
    assert 0.25 <= result.male_ratio <= 0.75
    assert result.is_imbalanced is False
    assert result.imbalance_status == "BALANCED"


def test_insufficient_mentions_neutral_text(analyzer):
    text = "The committee held a meeting in the grand hall. The report was submitted promptly."
    result = analyzer.analyze(text)

    assert result.total_gender_count == 0
    assert result.male_ratio == 0.5
    assert result.is_imbalanced is False
    assert result.imbalance_status == "INSUFFICIENT_DATA"
    assert result.signal_type == "NO_STRONG_LEXICAL_SIGNAL"


def test_domestic_co_occurrences(analyzer):
    text = "The mother was cooking dinner in the kitchen while her daughter was washing dishes."
    result = analyzer.analyze(text)

    assert len(result.co_occurrences) > 0
    domestic_pairs = [c for c in result.co_occurrences if c.category == "DOMESTIC_ROLE"]
    assert len(domestic_pairs) > 0

    # Mother + cooking -> potential stereotypical domestic evidence indicator
    stereotypical = [c for c in domestic_pairs if c.evidence_indicator == "POTENTIAL_STEREOTYPICAL_ASSOCIATION"]
    assert len(stereotypical) > 0
    assert stereotypical[0].attribution_confidence == "AMBIGUOUS_LEXICAL_PROXIMITY"


def test_counter_stereotypical_co_occurrences(analyzer):
    text = "The father was cooking dinner while she worked as an engineer at the laboratory."
    result = analyzer.analyze(text)

    # Father + cooking -> potential counter-stereotypical
    father_cooking = [
        c for c in result.co_occurrences
        if c.gender_term.lower() == "father" and c.associated_term.lower() == "cooking"
    ]
    assert len(father_cooking) > 0
    assert father_cooking[0].evidence_indicator == "POTENTIAL_COUNTER_STEREOTYPICAL_ASSOCIATION"
    assert father_cooking[0].category == "DOMESTIC_ROLE"

    # She + engineer -> potential counter-stereotypical STEM
    she_engineer = [
        c for c in result.co_occurrences
        if c.gender_term.lower() == "she" and c.associated_term.lower() == "engineer"
    ]
    assert len(she_engineer) > 0
    assert she_engineer[0].evidence_indicator == "POTENTIAL_COUNTER_STEREOTYPICAL_ASSOCIATION"
    assert she_engineer[0].category == "OCCUPATIONAL_BIAS"


def test_analyzer_execution_speed(analyzer):
    long_text = (
        "The headmaster praised the students for their academic performance. "
        "He congratulated the boy who built the scientific device, and he commended the girl "
        "who led the school orchestra. Her mother and his father attended the ceremony. "
        "The women and men in the audience applauded enthusiastically. She spoke about her ambitions "
        "to become an engineer, while he expressed his desire to become a doctor. "
    ) * 3

    result = analyzer.analyze(long_text)
    assert result.execution_time_ms < 15.0
    assert result.total_gender_count > 10


# -------------------------------------------------------------------------
# Adversarial Unit Tests (Task 4)
# -------------------------------------------------------------------------

def test_adversarial_1_female_near_action_performed_by_male(analyzer):
    """
    Adversarial 1: A female character mentioned near an action performed by a male character.
    'The girl saw father cooking dinner in the kitchen.'
    Demonstrates that raw token proximity window flags female + cooking,
    which is why attribution_confidence is marked AMBIGUOUS_LEXICAL_PROXIMITY.
    """
    text = "The girl saw father cooking dinner in the kitchen."
    result = analyzer.analyze(text)

    # Both 'girl' and 'father' are within 5 tokens of 'cooking'
    girl_matches = [c for c in result.co_occurrences if c.gender_term.lower() == "girl"]
    father_matches = [c for c in result.co_occurrences if c.gender_term.lower() == "father"]

    assert len(girl_matches) > 0
    assert len(father_matches) > 0
    # Every co-occurrence must be flagged as ambiguous proximity, never confirmed
    assert all(c.attribution_confidence == "AMBIGUOUS_LEXICAL_PROXIMITY" for c in result.co_occurrences)
    assert any("AMBIGUOUS_LEXICAL_PROXIMITY" in w for w in result.warnings)


def test_adversarial_2_male_near_action_performed_by_female(analyzer):
    """
    Adversarial 2: A male character mentioned near an action performed by a female character.
    'The boy saw queen captain the ship.'
    """
    text = "The boy saw the queen captain the ship."
    result = analyzer.analyze(text)

    boy_matches = [c for c in result.co_occurrences if c.gender_term.lower() == "boy"]
    assert len(boy_matches) > 0
    assert all(c.attribution_confidence == "AMBIGUOUS_LEXICAL_PROXIMITY" for c in boy_matches)


def test_adversarial_3_explicitly_rejecting_gender_stereotypes(analyzer):
    """
    Adversarial 3: A sentence explicitly rejecting gender stereotypes.
    'It is false that only women should do cooking and cleaning in the household.'
    """
    text = "It is entirely false that only women should do cooking and cleaning in the household."
    result = analyzer.analyze(text)

    # Demonstrates lexical limitation: lexical analyzer flags potential match on 'women' + 'cooking',
    # but does NOT claim the text is stereotypical.
    assert "Measures token reference frequency" in result.limitations[0]
    assert "Lexical matching does not detect negation" in result.limitations[2]


def test_adversarial_4_occupation_without_gender_role(analyzer):
    """
    Adversarial 4: A sentence discussing an occupation without implying a gender role.
    'Engineering and scientific research require rigorous mathematics.'
    """
    text = "Engineering and scientific research require rigorous mathematics and problem-solving skills."
    result = analyzer.analyze(text)

    assert result.total_gender_count == 0
    assert result.signal_type == "NO_STRONG_LEXICAL_SIGNAL"
    assert result.is_imbalanced is False
    assert len(result.co_occurrences) == 0


def test_adversarial_5_repeated_pronouns_referring_to_one_character(analyzer):
    """
    Adversarial 5: Repeated pronouns referring to one character.
    'He woke up early. He put on his shoes, he checked his bag, and he walked to his school.'
    Confirms reference frequency != unique character count.
    """
    text = "He woke up early. He put on his shoes, he checked his bag, and he walked to his school."
    result = analyzer.analyze(text)

    # 7 references to male pronouns, but represents only 1 individual
    assert result.male_pronoun_count == 7
    assert result.gendered_reference_frequency["male"] == 7
    assert result.is_imbalanced is True  # Reference frequency is skewed
    # Check limitation documentation
    assert any("unique character" in lim for lim in result.limitations)


def test_adversarial_6_negation_and_hypothetical_sentences(analyzer):
    """
    Adversarial 6: Negation and hypothetical sentences.
    'She does not do any cooking, and if he were an engineer, he would disagree.'
    """
    text = "She does not do any cooking, and if he were an engineer, he would disagree."
    result = analyzer.analyze(text)

    # Lexical analyzer reports potential matches, but signals attribution limitations
    assert len(result.co_occurrences) > 0
    assert result.signal_type == "MIXED_SIGNALS"
    assert any("negation" in lim for lim in result.limitations)


def test_adversarial_7_multiple_gendered_characters_same_passage(analyzer):
    """
    Adversarial 7: Multiple gendered characters in the same passage.
    'The mother, father, brother, sister, and grandmother all attended the gathering.'
    """
    text = "The mother, father, brother, sister, and grandmother all attended the family gathering."
    result = analyzer.analyze(text)

    assert result.male_noun_count == 2   # father, brother
    assert result.female_noun_count == 3  # mother, sister, grandmother
    assert result.total_gender_count == 5
    assert result.gendered_reference_frequency["male"] == 2
    assert result.gendered_reference_frequency["female"] == 3
    # 2/5 = 0.40 male ratio -> Balanced (within [0.25, 0.75])
    assert result.imbalance_status == "BALANCED"
    assert result.is_imbalanced is False


def test_adversarial_8_insufficient_or_ambiguous_gender_references(analyzer):
    """
    Adversarial 8: Insufficient or ambiguous gender references.
    'The teacher gave the student an assignment. They completed it on time.'
    """
    text = "The teacher gave the student an assignment. They completed it on time."
    result = analyzer.analyze(text)

    # Epicene / non-binary or ungendered terms produce zero binary gender counts
    assert result.total_gender_count == 0
    assert result.imbalance_status == "INSUFFICIENT_DATA"
    assert result.signal_type == "NO_STRONG_LEXICAL_SIGNAL"
    assert len(result.warnings) == 0
