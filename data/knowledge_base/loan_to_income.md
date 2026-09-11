# Loan-to-Income Guidelines

> Document type: Synthetic reference policy
> Policy domain: Financial Risk
> Version: 1.0
> Note: This document is fictional and created for the HDFC Loan Intelligence demonstration system.

## Definition

Loan-to-income (LTI) compares the requested loan amount with the applicant's annual household income.

For this reference system:

LTI = Requested Loan Amount / Annual Household Income

The ratio provides a simple indication of how large the requested loan is relative to annual income.

## Interpretation

### LTI Below 3

An LTI below 3 indicates that the requested loan is relatively moderate compared with annual household income.

### LTI Between 3 and 6

An LTI between 3 and 6 indicates a higher loan requirement relative to annual income.

Additional assessment may be appropriate.

### LTI Above 6

An LTI above 6 indicates a high loan amount relative to annual household income.

Applications with a high LTI should receive additional scrutiny, particularly when DTI or credit indicators are also unfavorable.

## Example

An applicant requests a loan of ₹3,000,000 and has annual household income of ₹1,000,000.

LTI = ₹3,000,000 / ₹1,000,000 = 3.0

The application falls into the 3-to-6 reference range.

## Combined Assessment

LTI should not be used as a standalone lending decision.

It should be considered together with:

- DTI.
- CIBIL score.
- Existing EMIs.
- Employment stability.
- Previous loan history.
- Loan tenure.

## Missing Income

If annual household income is unavailable, the system must not invent an LTI value.

The application should be flagged for incomplete financial information.

## Important Limitation

The LTI thresholds in this document are synthetic reference values created for this demonstration.