"""
Unit tests for app.services.loan_service.calculate_risk_score.

These use real values from HDFC100125 (a rejected application in the actual
dataset) rather than invented numbers, so the expected risk_score here is
traceable back to agent_specifications.md's worked example.

Run with: pytest tests/test_loan_service.py -v
"""
from app.services.loan_service import calculate_risk_score


def make_record(**overrides) -> dict:
    """Baseline low-risk record; tests override only the fields they're probing."""
    base = {
        "financials": {
            "debt_to_income_ratio": 0.3,
            "loan_to_annual_income": 1.0,
        },
        "credit": {
            "cibil_score": 780,
            "credit_history": 1,
            "default_history_count": 0,
        },
        "employment": {
            "employment_status": "Salaried",
        },
    }
    for key, value in overrides.items():
        section, field = key.split(".")
        base[section][field] = value
    return base


def test_hdfc100125_matches_documented_example() -> None:
    """
    Real values from the dataset: CIBIL 616, credit_history 1, DTI 1.885,
    loan_to_annual_income 8.768, Unemployed, 0 defaults.
    Documented expected result: risk_score=80, HIGH, requires_manual_review=True.
    """
    record = {
        "financials": {"debt_to_income_ratio": 1.885, "loan_to_annual_income": 8.768},
        "credit": {"cibil_score": 616, "credit_history": 1, "default_history_count": 0},
        "employment": {"employment_status": "Unemployed"},
    }
    result = calculate_risk_score(record)

    assert result["risk_score"] == 80
    assert result["risk_level"] == "HIGH"
    assert result["requires_manual_review"] is True
    assert "Below-average CIBIL score (616)" in result["risk_factors"]
    assert "Very high debt-to-income ratio (1.89x)" in result["risk_factors"]
    assert "Applicant is currently unemployed" in result["risk_factors"]
    assert "Established credit history" in result["positive_factors"]
    assert "No prior loan defaults" in result["positive_factors"]


def test_clean_low_risk_application() -> None:
    record = make_record()
    result = calculate_risk_score(record)

    assert result["risk_score"] == 0
    assert result["risk_level"] == "LOW"
    assert result["requires_manual_review"] is False
    assert result["risk_factors"] == []


def test_default_history_forces_manual_review_regardless_of_score() -> None:
    """
    Even a single default should force manual review, per the spec's escalation
    intent -- a low aggregate score shouldn't hide a real default on record.
    """
    record = make_record(**{"credit.default_history_count": 1})
    result = calculate_risk_score(record)

    assert result["requires_manual_review"] is True
    assert "1 prior loan default(s) on record" in result["risk_factors"]


def test_default_count_is_capped_at_two_multiplier() -> None:
    """A large default count shouldn't single-handedly blow past 100 on its own."""
    record = make_record(**{"credit.default_history_count": 5})
    result = calculate_risk_score(record)

    # 5 defaults capped at 2x20=40 points, not 5x20=100
    assert result["risk_score"] == 40


def test_sanity_cap_prevents_divide_by_zero_row_from_breaking_score() -> None:
    """
    Simulates a row where Annual_Household_Income=0 produced a raw ratio of
    ~712 (see HDFC100008 in the real dataset). The repository layer caps this
    to 20.0 before it ever reaches scoring -- confirm the score stays sane.
    """
    record = make_record(**{"financials.debt_to_income_ratio": 20.0})
    result = calculate_risk_score(record)

    assert result["risk_score"] <= 100
    assert "Very high debt-to-income ratio (20.00x)" in result["risk_factors"]


def test_unemployed_with_high_dti_triggers_review_even_at_moderate_score() -> None:
    """The compound-risk escalation rule should fire independent of raw score."""
    record = make_record(
        **{
            "employment.employment_status": "Unemployed",
            "financials.debt_to_income_ratio": 0.85,
            "credit.cibil_score": 760,  # good CIBIL, shouldn't matter here
        }
    )
    result = calculate_risk_score(record)

    assert result["requires_manual_review"] is True
