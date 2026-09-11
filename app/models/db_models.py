"""
SQLAlchemy ORM models.

Design decisions carried over from the dataset profiling step:

1. There is NO reliable customer identity in the source data (Phone_Number is
   100% unique across all 1000 rows, Email is templated off a 41-name pool,
   Aadhaar duplicates are float-rounding artifacts, not real repeats). So
   `loans.loan_id` is the true grain of this dataset, not a customer_id.
   The `customers` table below stores a 1:1 snapshot per loan, not a
   deduplicated customer master -- this is documented here rather than
   silently implying a customer graph that doesn't exist.

2. Aadhaar_Synthetic and Phone_Number are stored as TEXT, not numeric. The
   source CSV already lost precision on Aadhaar by storing it as a float
   (e.g. 694000000000.0) -- casting to TEXT preserves whatever precision
   remains and, more importantly, avoids ever re-introducing numeric
   coercion bugs (leading zeros, scientific notation) downstream.

3. Debt_to_Income_Ratio and Loan_to_Annual_Income are stored twice per row:
   the raw value (`*_raw`) for audit purposes, and a sanity-capped value
   (`*_capped`, capped at settings.max_ratio_sanity_cap = 20.0) that is what
   services/agents actually read. This is a SANITY cap, not a cosmetic
   display cap -- it only neutralizes the ~11 rows where
   Annual_Household_Income = 0 (raw ratio ~38,000, a divide-by-zero
   artifact). It is set well above any realistic-but-high ratio (e.g. 8.8x)
   so genuine risk signal reaches the scoring function unchanged. An earlier
   version of this cap was set to 3.0 for "display safety" and it silently
   zeroed out real high-risk signal before scoring ever saw it -- do not
   reintroduce a low cap here.

4. Sensitivity tiers (documented in models/schemas.py, enforced at the
   repository layer): PROTECTED (religion, gender), RESTRICTED (aadhaar,
   phone, email, pin_code, customer_name), INTERNAL (state, city, branch),
   SAFE (everything financial/analytical).
"""
from sqlalchemy import (
    CheckConstraint,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Loan(Base):
    """
    The root table -- one row per application, matching Loan_ID 1:1.
    Everything else joins to this via loan_id.
    """
    __tablename__ = "loans"

    loan_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    bank: Mapped[str] = mapped_column(String(50), default="HDFC Bank")
    loan_amount: Mapped[int] = mapped_column(Integer, nullable=False)
    loan_term_months: Mapped[int] = mapped_column(Integer, nullable=False)
    purpose_of_loan: Mapped[str] = mapped_column(String(50), nullable=False)
    loan_status: Mapped[str] = mapped_column(String(20), nullable=False)
    property_area: Mapped[str] = mapped_column(String(20), nullable=False)
    region_branch: Mapped[str] = mapped_column(String(20), nullable=False)

    customer = relationship("Customer", back_populates="loan", uselist=False, cascade="all, delete-orphan")
    financials = relationship("FinancialDetail", back_populates="loan", uselist=False, cascade="all, delete-orphan")
    employment = relationship("EmploymentDetail", back_populates="loan", uselist=False, cascade="all, delete-orphan")
    credit = relationship("LoanHistory", back_populates="loan", uselist=False, cascade="all, delete-orphan")
    feedback = relationship("CustomerFeedback", back_populates="loan", uselist=False, cascade="all, delete-orphan")
    notes = relationship("ApplicationNote", back_populates="loan", uselist=False, cascade="all, delete-orphan")
    verification = relationship("Verification", back_populates="loan", uselist=False, cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint("loan_status IN ('Approved', 'Rejected')", name="ck_loan_status_valid"),
    )


class Customer(Base):
    """
    Per-application demographic snapshot. NOT a deduplicated customer master
    -- see module docstring. `customer_ref` is a synthetic surrogate key we
    generate at ingestion (loan_id itself, prefixed) purely so this table has
    its own PK independent of loans.loan_id, in case a real customer master
    is introduced later without reshaping this table.
    """
    __tablename__ = "customers"

    loan_id: Mapped[str] = mapped_column(String(20), ForeignKey("loans.loan_id"), primary_key=True)
    customer_ref: Mapped[str] = mapped_column(String(30), unique=True, nullable=False)

    # RESTRICTED
    customer_name: Mapped[str] = mapped_column(String(100), nullable=False)

    # PROTECTED -- never surfaced to LLM context or used in scoring
    gender: Mapped[str] = mapped_column(String(10), nullable=False)
    religion: Mapped[str] = mapped_column(String(20), nullable=False)

    married: Mapped[str] = mapped_column(String(5), nullable=False)
    dependents: Mapped[int] = mapped_column(Integer, nullable=False)
    education: Mapped[str] = mapped_column(String(20), nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=False)

    loan = relationship("Loan", back_populates="customer")


class FinancialDetail(Base):
    __tablename__ = "financial_details"

    loan_id: Mapped[str] = mapped_column(String(20), ForeignKey("loans.loan_id"), primary_key=True)
    applicant_income: Mapped[int] = mapped_column(Integer, nullable=False)
    coapplicant_income: Mapped[int] = mapped_column(Integer, nullable=False)
    annual_household_income: Mapped[int] = mapped_column(Integer, nullable=False)

    debt_to_income_ratio_raw: Mapped[float] = mapped_column(Float, nullable=False)
    debt_to_income_ratio_capped: Mapped[float] = mapped_column(Float, nullable=False)
    loan_to_annual_income_raw: Mapped[float] = mapped_column(Float, nullable=False)
    loan_to_annual_income_capped: Mapped[float] = mapped_column(Float, nullable=False)

    existing_emis: Mapped[int] = mapped_column(Integer, nullable=False)
    monthly_expense: Mapped[int] = mapped_column(Integer, nullable=False)
    asset_value: Mapped[int] = mapped_column(Integer, nullable=False)

    loan = relationship("Loan", back_populates="financials")


class EmploymentDetail(Base):
    __tablename__ = "employment_details"

    loan_id: Mapped[str] = mapped_column(String(20), ForeignKey("loans.loan_id"), primary_key=True)
    employment_status: Mapped[str] = mapped_column(String(30), nullable=False)
    employment_length_years: Mapped[int] = mapped_column(Integer, nullable=False)
    organization_type: Mapped[str] = mapped_column(String(30), nullable=False)
    business_type: Mapped[str | None] = mapped_column(String(50), nullable=True)  # null unless self-employed
    occupation: Mapped[str] = mapped_column(String(50), nullable=False)

    loan = relationship("Loan", back_populates="employment")


class LoanHistory(Base):
    """
    NOTE: number_of_previous_loans and default_history_count are attributes
    reported ON this application, not links to other real rows in this
    dataset (see module docstring, point 1). Tool docstrings that read this
    table must say so explicitly.
    """
    __tablename__ = "loan_history"

    loan_id: Mapped[str] = mapped_column(String(20), ForeignKey("loans.loan_id"), primary_key=True)
    credit_history: Mapped[int] = mapped_column(Integer, nullable=False)  # 0/1
    cibil_score: Mapped[int] = mapped_column(Integer, nullable=False)
    number_of_previous_loans: Mapped[int] = mapped_column(Integer, nullable=False)
    default_history_count: Mapped[int] = mapped_column(Integer, nullable=False)
    guarantor: Mapped[str] = mapped_column(String(5), nullable=False)
    co_signer_relationship: Mapped[str | None] = mapped_column(String(30), nullable=True)

    loan = relationship("Loan", back_populates="credit")

    __table_args__ = (
        CheckConstraint("credit_history IN (0, 1)", name="ck_credit_history_binary"),
        CheckConstraint("cibil_score BETWEEN 300 AND 900", name="ck_cibil_range"),
    )


class CustomerFeedback(Base):
    __tablename__ = "customer_feedback"

    loan_id: Mapped[str] = mapped_column(String(20), ForeignKey("loans.loan_id"), primary_key=True)
    customer_feedback: Mapped[str] = mapped_column(Text, nullable=False)
    customer_sentiment: Mapped[str] = mapped_column(String(20), nullable=False)

    loan = relationship("Loan", back_populates="feedback")


class ApplicationNote(Base):
    __tablename__ = "application_notes"

    loan_id: Mapped[str] = mapped_column(String(20), ForeignKey("loans.loan_id"), primary_key=True)
    application_text: Mapped[str] = mapped_column(Text, nullable=False)
    agent_notes: Mapped[str] = mapped_column(Text, nullable=False)
    institutional_relationships: Mapped[str | None] = mapped_column(Text, nullable=True)

    loan = relationship("Loan", back_populates="notes")


class Verification(Base):
    """
    Everything in this table is RESTRICTED tier. The repository layer
    excludes this table entirely from any query used to build LLM context;
    it exists only for KYC/audit workflows outside the chat path.
    """
    __tablename__ = "verification"

    loan_id: Mapped[str] = mapped_column(String(20), ForeignKey("loans.loan_id"), primary_key=True)
    mobile_verified: Mapped[str] = mapped_column(String(5), nullable=False)
    email_verified: Mapped[str] = mapped_column(String(5), nullable=False)

    # RESTRICTED -- stored as TEXT deliberately, see module docstring point 2
    aadhaar_synthetic: Mapped[str] = mapped_column(String(20), nullable=False)
    phone_number: Mapped[str] = mapped_column(String(20), nullable=False)
    email: Mapped[str] = mapped_column(String(100), nullable=False)
    pin_code: Mapped[str] = mapped_column(String(10), nullable=False)
    state: Mapped[str] = mapped_column(String(50), nullable=False)
    city: Mapped[str] = mapped_column(String(50), nullable=False)

    loan = relationship("Loan", back_populates="verification")
