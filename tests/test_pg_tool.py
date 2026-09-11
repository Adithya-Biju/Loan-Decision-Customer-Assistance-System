"""
Integration tests for PostgreSQL tools used by Agent 1.

These tests query the actual PostgreSQL database.

Run with:
    pytest tests/test_postgres_tools.py -v
"""

from app.tools.postgres_tools import (
    get_loan_details,
    get_customer_financials,
    get_credit_history,
    get_previous_loans,
    get_employment_details,
)


LOAN_ID = "HDFC100125"


def test_get_loan_details() -> None:
    result = get_loan_details.invoke({
        "loan_id": LOAN_ID
    })

    assert result["found"] is True
    assert result["loan_id"] == LOAN_ID

    # Values from the documented HDFC100125 example
    assert result["loan_amount"] == 1753165
    assert result["loan_term_months"] == 360
    assert result["purpose_of_loan"] == "Business"
    assert result["loan_status"] == "Rejected"
    assert result["property_area"] == "Semiurban"
    assert result["region_branch"] == "MUM-002"


def test_get_customer_financials() -> None:
    result = get_customer_financials.invoke({
        "loan_id": LOAN_ID
    })

    assert result["found"] is True
    assert result["loan_id"] == LOAN_ID

    assert result["applicant_income"] == 13036
    assert result["coapplicant_income"] == 3627
    assert result["annual_household_income"] == 199956

    assert result["debt_to_income_ratio"] == 1.885
    assert result["loan_to_annual_income"] == 8.768

    assert result["existing_emis"] == 31406
    assert result["asset_value"] == 501634


def test_get_credit_history() -> None:
    result = get_credit_history.invoke({
        "loan_id": LOAN_ID
    })

    assert result["found"] is True
    assert result["loan_id"] == LOAN_ID

    assert result["cibil_score"] == 616
    assert result["credit_history"] == 1
    assert result["default_history_count"] == 0


def test_get_previous_loans() -> None:
    result = get_previous_loans.invoke({
        "loan_id": LOAN_ID
    })

    assert result["found"] is True
    assert result["loan_id"] == LOAN_ID

    assert result["number_of_previous_loans"] == 1


def test_get_employment_details() -> None:
    result = get_employment_details.invoke({
        "loan_id": LOAN_ID
    })

    assert result["found"] is True
    assert result["loan_id"] == LOAN_ID

    assert result["employment_status"] == "Unemployed"
    assert result["employment_length_years"] == 5
    assert result["business_type"] == None


def test_invalid_loan_returns_not_found() -> None:
    result = get_loan_details.invoke({
        "loan_id": "DOES_NOT_EXIST"
    })

    assert result["found"] is False
    assert result["loan_id"] == "DOES_NOT_EXIST"