"""
Streamlit frontend -- Phase 1 scope: a single "Analyze Loan" page that calls
the FastAPI backend's /api/v1/loan/analyze endpoint. This is intentionally
thin right now; Phase 2+ will add the chat interface (streaming from
/api/v1/assistant/chat), the knowledge-query page, and the analytics page,
each as a separate file under frontend/pages/ following Streamlit's
multi-page convention.

The frontend never talks to Postgres or Qdrant directly -- it only ever
calls the FastAPI backend. This keeps the backend as the single point of
guardrail enforcement (PII masking, SQL safety, etc.) regardless of which
client is calling it.
"""
import os

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="HDFC Loan Intelligence", page_icon="🏦", layout="centered")

st.title("🏦 HDFC Loan Intelligence System")
st.caption(
    "AI-assisted loan risk analysis for loan officer review. "
    "This is a recommendation tool, not an automated approval/rejection system."
)

st.subheader("Analyze a Loan Application")

loan_id = st.text_input("Loan ID", placeholder="e.g. HDFC100125")
analyze_clicked = st.button("Analyze", type="primary")

if analyze_clicked:
    if not loan_id.strip():
        st.warning("Enter a loan ID first.")
    else:
        with st.spinner(f"Analyzing {loan_id}..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/api/v1/loan/analyze",
                    json={"loan_id": loan_id.strip().upper()},
                    timeout=15,
                )
            except requests.exceptions.ConnectionError:
                st.error(
                    f"Could not reach the backend at {BACKEND_URL}. "
                    "Is the FastAPI service running?"
                )
                st.stop()

        if response.status_code == 200:
            data = response.json()

            risk_color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(data["financial_risk"], "⚪")

            col1, col2, col3 = st.columns(3)
            col1.metric("Risk Level", f"{risk_color} {data['financial_risk']}")
            col2.metric("Risk Score", f"{data['risk_score']} / 100")
            col3.metric("Manual Review", "Yes" if data["requires_manual_review"] else "No")

            if data["requires_manual_review"]:
                st.warning(
                    "⚠️ This application is flagged for manual loan officer review.",
                    icon="⚠️",
                )

            st.markdown("#### Risk Factors")
            if data["risk_factors"]:
                for factor in data["risk_factors"]:
                    st.markdown(f"- {factor}")
            else:
                st.markdown("_None identified._")

            st.markdown("#### Positive Factors")
            if data["positive_factors"]:
                for factor in data["positive_factors"]:
                    st.markdown(f"- {factor}")
            else:
                st.markdown("_None identified._")

            with st.expander("Raw response"):
                st.json(data)

        elif response.status_code == 404:
            st.error(f"No loan application found with ID '{loan_id}'.")
        elif response.status_code == 422:
            st.error(response.json().get("detail", "Invalid loan ID format."))
        else:
            st.error(f"Unexpected error ({response.status_code}): {response.text}")

st.divider()
st.caption(
    "This system produces an AI-assisted recommendation for human loan officer "
    "validation and does not make final lending decisions."
)
