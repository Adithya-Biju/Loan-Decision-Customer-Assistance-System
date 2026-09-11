# Required Loan Documentation

> Document type: Synthetic reference policy
> Policy domain: Documentation
> Version: 1.0
> Note: This document is fictional and created for the HDFC Loan Intelligence demonstration system.

## Overview

Applicants must provide sufficient documentation to verify identity, income, employment, and other relevant application information.

The exact documents required may vary depending on the loan product and applicant profile.

## Identity Documentation

Acceptable identity documentation may include:

- Government-issued identity document.
- Valid address proof.
- Other approved identity verification documents.

Sensitive identity numbers should not be exposed in application responses unless specifically required for an authorized workflow.

## Income Documentation

Depending on employment type, income verification may include:

- Recent salary slips.
- Bank statements.
- Income tax documentation.
- Employer-issued income documentation.
- Business financial records for self-employed applicants.

## Employment Documentation

Salaried applicants may be asked to provide:

- Employment confirmation.
- Recent salary slips.
- Employer details.
- Bank statements showing salary credits where applicable.

Self-employed applicants may be asked for:

- Business registration information.
- Income tax returns.
- Business bank statements.
- Financial statements.

## Existing Loan Information

Applicants may need to provide information about:

- Existing loans.
- Current EMI obligations.
- Outstanding balances.
- Previous repayment history.

## Verification

Documentation may be subject to verification.

If information in submitted documentation conflicts with information in the application, the application should be escalated for review.

## Incomplete Documentation

Incomplete documentation should not automatically be treated as a rejection.

The reviewer should identify the missing documents and determine whether they are material to the eligibility assessment.

## Privacy

The system must avoid exposing sensitive personal information such as:

- Aadhaar numbers.
- Phone numbers.
- Email addresses.
- PIN codes.

Only the minimum information required for the task should be included in an LLM context.