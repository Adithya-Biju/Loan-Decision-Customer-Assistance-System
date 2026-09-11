# Debt-to-Income Ratio Guidelines

> Document type: Synthetic reference policy
> Policy domain: Financial Risk
> Version: 1.0
> Note: This document is fictional and created for the HDFC Loan Intelligence demonstration system.

## Definition

Debt-to-income ratio (DTI) measures the applicant's debt obligations relative to income.

For this reference system:

DTI = Existing Monthly EMI Obligations / Monthly Income

A higher DTI indicates that a larger portion of income is already committed to debt repayment.

## Interpretation

### DTI Below 0.40

DTI below 0.40 indicates relatively lower existing debt burden.

This is generally considered favorable from a repayment-capacity perspective.

### DTI Between 0.40 and 0.80

DTI between 0.40 and 0.80 indicates a moderate existing debt burden.

The application should be assessed together with income stability, credit history, requested loan amount, and loan tenure.

### DTI Above 0.80

DTI above 0.80 indicates a high existing debt burden.

Applications with DTI above 0.80 should receive additional scrutiny because repayment capacity may be constrained.

## Example

An applicant earns ₹100,000 per month and has ₹50,000 in existing EMI obligations.

DTI = ₹50,000 / ₹100,000 = 0.50

The applicant therefore has a DTI of 0.50, which falls within the moderate range.

## High DTI Review

A high DTI becomes more concerning when combined with:

- Low CIBIL score.
- Existing defaults.
- Unemployment or unstable income.
- High requested loan amount.
- High loan-to-income ratio.

## Missing Data

If income or EMI information is unavailable, DTI should not be estimated without supporting information.

The system should report that DTI cannot be reliably calculated.

## Important Limitation

DTI thresholds in this document are reference thresholds for the demonstration system and are not official lending policy.