"""
One-time ingestion: HDFC CSV -> normalized Postgres schema.

Usage:
    python scripts/ingest_postgres.py --csv data/raw/hdfc_loan_dataset.csv

What this does, per row:
  1. Casts Aadhaar_Synthetic and Phone_Number to TEXT explicitly (the source
     CSV stores them as numeric, which already lost precision on Aadhaar --
     casting via str() at least stops further corruption at our layer).
  2. Applies the sanity cap (settings.max_ratio_sanity_cap, default 20.0) to
     Debt_to_Income_Ratio and Loan_to_Annual_Income, storing both the raw and
     capped values -- see db_models.py docstring for why this cap must stay
     high enough not to erase genuine high-risk signal.
  3. Splits the 47 flat columns across the 8 normalized tables.
  4. Generates a synthetic customer_ref per row (there is no reliable
     customer identity in the source data -- see db_models.py docstring).

This script is idempotent-ish: it creates tables if they don't exist, but
does NOT upsert -- re-running against a populated DB will raise a primary
key violation. Use --reset to drop and recreate first (destructive, local
dev only).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings 
from app.core.database import Base, SessionLocal, engine 
from app.models.db_models import ( 
    ApplicationNote,
    Customer,
    CustomerFeedback,
    EmploymentDetail,
    FinancialDetail,
    Loan,
    LoanHistory,
    Verification,
)

settings = get_settings()


def cap_ratio(value: float) -> float:
    return min(value, settings.max_ratio_sanity_cap)


def row_to_orm_objects(row: pd.Series) -> list:
    loan_id = row["Loan_ID"]

    loan = Loan(
        loan_id=loan_id,
        bank=row["Bank"],
        loan_amount=int(row["Loan_Amount"]),
        loan_term_months=int(row["Loan_Term_Months"]),
        purpose_of_loan=row["Purpose_of_Loan"],
        loan_status=row["Loan_Status"],
        property_area=row["Property_Area"],
        region_branch=row["Region_Branch"],
    )

    customer = Customer(
        loan_id=loan_id,
        customer_ref=f"CUST-{loan_id}",
        customer_name=row["Customer_Name"],
        gender=row["Gender"],
        religion=row["Religion"],
        married=row["Married"],
        dependents=int(row["Dependents"]),
        education=row["Education"],
        age=int(row["Age"]),
    )

    dti_raw = float(row["Debt_to_Income_Ratio"])
    lti_raw = float(row["Loan_to_Annual_Income"])
    financials = FinancialDetail(
        loan_id=loan_id,
        applicant_income=int(row["Applicant_Income"]),
        coapplicant_income=int(row["Coapplicant_Income"]),
        annual_household_income=int(row["Annual_Household_Income"]),
        debt_to_income_ratio_raw=dti_raw,
        debt_to_income_ratio_capped=cap_ratio(dti_raw),
        loan_to_annual_income_raw=lti_raw,
        loan_to_annual_income_capped=cap_ratio(lti_raw),
        existing_emis=int(row["Existing_EMIs"]),
        monthly_expense=int(row["Monthly_Expense"]),
        asset_value=int(row["Asset_Value"]),
    )

    employment = EmploymentDetail(
        loan_id=loan_id,
        employment_status=row["Employment_Status"],
        employment_length_years=int(row["Employment_Length_Years"]),
        organization_type=row["Organization_Type"],
        business_type=None if pd.isna(row["Business_Type"]) else row["Business_Type"],
        occupation=row["Occupation"],
    )

    history = LoanHistory(
        loan_id=loan_id,
        credit_history=int(row["Credit_History"]),
        cibil_score=int(row["CIBIL_Score"]),
        number_of_previous_loans=int(row["Number_of_Previous_Loans"]),
        default_history_count=int(row["Default_History_Count"]),
        guarantor=row["Guarantor"],
        co_signer_relationship=(
            None if pd.isna(row["Co-signer_Relationship"]) else row["Co-signer_Relationship"]
        ),
    )

    feedback = CustomerFeedback(
        loan_id=loan_id,
        customer_feedback=row["Customer_Feedback"],
        customer_sentiment=row["Customer_Sentiment"],
    )

    notes = ApplicationNote(
        loan_id=loan_id,
        application_text=row["Application_Text"],
        agent_notes=row["Agent_Notes"],
        institutional_relationships=(
            None if pd.isna(row["Institutional_Relationships"]) else row["Institutional_Relationships"]
        ),
    )

    verification = Verification(
        loan_id=loan_id,
        mobile_verified=row["Mobile_Verified"],
        email_verified=row["Email_Verified"],
        aadhaar_synthetic=str(int(row["Aadhaar_Synthetic"])),
        phone_number=str(int(row["Phone_Number"])),
        email=row["Email"],
        pin_code=str(int(row["PIN_Code"])),
        state=row["State"],
        city=row["City"],
    )

    return [loan, customer, financials, employment, history, feedback, notes, verification]


def ingest(csv_path: str, reset: bool = False) -> None:
    if reset:
        print("--reset passed: dropping all tables...")
        Base.metadata.drop_all(engine)

    print("Creating tables if they don't exist...")
    Base.metadata.create_all(engine)

    print(f"Reading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

    session = SessionLocal()
    try:
        inserted = 0
        for _, row in df.iterrows():
            objects = row_to_orm_objects(row)
            session.add_all(objects)
            inserted += 1
            if inserted % 200 == 0:
                session.commit()
                print(f"  ...committed {inserted} rows")
        session.commit()
        print(f"Done. Ingested {inserted} loan applications across 8 tables.")
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest HDFC loan CSV into Postgres.")
    parser.add_argument("--csv", required=True, help="Path to the HDFC loan dataset CSV.")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate all tables first (destructive).")
    args = parser.parse_args()
    ingest(args.csv, reset=args.reset)
