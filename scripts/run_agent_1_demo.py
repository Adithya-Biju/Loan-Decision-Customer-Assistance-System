"""
Standalone demo/smoke-test for Agent 1, runnable once you have:
  - Postgres running and ingested (see scripts/ingest_postgres.py)
  - .env filled in with OPENROUTER_API_KEY
  - `pip install -r requirements.txt` done

Usage:
    python scripts/run_agent_1_demo.py "Analyze loan HDFC100125"
    python scripts/run_agent_1_demo.py "why is this loan risky" --loan-id HDFC100125
    python scripts/run_agent_1_demo.py "what's the CIBIL score on HDFC100125"

This is the first real end-to-end test of the agent -- it couldn't be run in
the sandbox this was built in (no network, no installed packages), so this
is the script to run first on your machine before wiring it into the API.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.loan_analysis_agent import run_loan_analysis 


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Agent 1 against a query.")
    parser.add_argument("query", help="Natural-language question for the agent.")
    parser.add_argument("--loan-id", default=None, help="Optional loan_id hint (e.g. from a known context).")
    args = parser.parse_args()

    print(f"Query: {args.query}")
    if args.loan_id:
        print(f"Loan ID hint: {args.loan_id}")
    print("-" * 60)

    result = run_loan_analysis(args.query, loan_id_hint=args.loan_id)

    print(f"Resolved loan_id : {result.loan_id}")
    print(f"Tools called     : {result.tools_called}")
    if result.gap_filled_tools:
        print(f"Gap-filled tools : {result.gap_filled_tools}  <-- LLM didn't call these itself, code filled in")
    print(f"Agent's message  : {result.agent_message}")

    if result.error:
        print(f"\nERROR: {result.error}")
        return

    if result.loan_analysis:
        print("\n--- LoanAnalysis (deterministic, never LLM-generated) ---")
        print(result.loan_analysis.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
