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
    # Check ratio is within [0.25, 0.75]
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

    # Only 2 mentions (below threshold of 4)
    short_text = "The boy saw his dog."
    short_result = analyzer.analyze(short_text)
    assert short_result.total_gender_count == 2
    assert short_result.imbalance_status == "INSUFFICIENT_DATA"
    assert short_result.is_imbalanced is False


def test_domestic_co_occurrences(analyzer):
    text = "The mother was cooking dinner in the kitchen while her daughter was washing dishes."
    result = analyzer.analyze(text)

    assert len(result.co_occurrences) > 0
    domestic_pairs = [c for c in result.co_occurrences if c.category == "DOMESTIC_ROLE"]
    assert len(domestic_pairs) > 0

    # Mother + cooking -> stereotypical domestic
    stereotypical = [c for c in domestic_pairs if c.direction == "STEREOTYPICAL"]
    assert len(stereotypical) > 0


def test_counter_stereotypical_co_occurrences(analyzer):
    text = "The father was cooking dinner while she worked as an engineer at the laboratory."
    result = analyzer.analyze(text)

    # Father + cooking -> counter-stereotypical
    father_cooking = [
        c for c in result.co_occurrences
        if c.gender_term.lower() == "father" and c.associated_term.lower() == "cooking"
    ]
    assert len(father_cooking) > 0
    assert father_cooking[0].direction == "COUNTER_STEREOTYPICAL"
    assert father_cooking[0].category == "DOMESTIC_ROLE"

    # She + engineer -> counter-stereotypical STEM
    she_engineer = [
        c for c in result.co_occurrences
        if c.gender_term.lower() == "she" and c.associated_term.lower() == "engineer"
    ]
    assert len(she_engineer) > 0
    assert she_engineer[0].direction == "COUNTER_STEREOTYPICAL"
    assert she_engineer[0].category == "OCCUPATIONAL_BIAS"


def test_analyzer_execution_speed(analyzer):
    # Standard 300-word paragraph
    long_text = (
        "The headmaster praised the students for their academic performance. "
        "He congratulated the boy who built the scientific device, and he commended the girl "
        "who led the school orchestra. Her mother and his father attended the ceremony. "
        "The women and men in the audience applauded enthusiastically. She spoke about her ambitions "
        "to become an engineer, while he expressed his desire to become a doctor. "
    ) * 3

    result = analyzer.analyze(long_text)
    assert result.execution_time_ms < 15.0  # Well within latency budget
    assert result.total_gender_count > 10
