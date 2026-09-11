# HDFC Intelligent Loan Decision & Customer Assistance System

Multi-agent GenAI system for loan risk analysis, policy Q&A, and decision
review. Produces AI-assisted recommendations for human loan officer
validation — it does not make final lending decisions.

## Status: Phase 1

This phase delivers the layered backend architecture, the full Postgres
schema, real-data ingestion, and one complete vertical slice
(`POST /api/v1/loan/analyze`) end-to-end through router → controller →
service → repository, plus a minimal Streamlit page calling it.

**Not yet built** (upcoming phases): the 3 LangGraph agents, Qdrant RAG
pipeline, chat/streaming endpoint, NL→SQL analytics, guardrails module,
memory, and the critic/retry loop. `app/agents/`, `app/graph/`, `app/rag/`,
`app/tools/`, and `app/guardrails/` currently exist as empty packages
reserved for this.

## Architecture

```
Router          → HTTP only (FastAPI route + schema)
Controller      → validation, error shaping, orchestration
Service         → business logic (e.g. deterministic risk scoring)
Repository      → raw Postgres queries, sensitivity-tier enforcement
```

Repositories only ever SELECT fields classified SAFE or INTERNAL (see
`app/models/db_models.py` docstring). RESTRICTED fields (Aadhaar, phone,
email, PIN code, name) and PROTECTED fields (religion, gender) are never
queried by any function an agent or API response can reach — not filtered
after the fact, never selected in the first place.

## Local setup

1. Copy the dataset into `data/raw/`:
   ```
   cp <your-csv> data/raw/hdfc_loan_dataset.csv
   ```

2. Copy `.env.example` to `.env` and fill in an LLM API key (not required
   for Phase 1's endpoint, but needed once agents land in Phase 2).

3. Start Postgres and Qdrant:
   ```
   docker-compose up -d postgres qdrant
   ```

4. Install dependencies locally (for running ingestion/tests outside
   Docker):
   ```
   pip install -r requirements.txt
   ```

5. Ingest the dataset:
   ```
   python scripts/ingest_postgres.py --csv data/raw/hdfc_loan_dataset.csv
   ```

6. Run the backend:
   ```
   uvicorn app.main:app --reload --port 8000
   ```
   Swagger docs: http://localhost:8000/docs

7. Run the frontend:
   ```
   streamlit run frontend/streamlit_app.py
   ```

Or run everything together:
```
docker-compose up --build
```
(then run step 5's ingestion command once against the running Postgres
container — it isn't run automatically by compose, to avoid accidentally
re-ingesting on every restart).

## Tests

```
pytest tests/ -v
```

`tests/test_loan_service.py` validates the deterministic risk-scoring rubric
against real values from the dataset (HDFC100125), not invented numbers.

## Try it

```
curl -X POST http://localhost:8000/api/v1/loan/analyze \
  -H "Content-Type: application/json" \
  -d '{"loan_id": "HDFC100125"}'
```

Expected: `risk_score: 80`, `financial_risk: "HIGH"`,
`requires_manual_review: true`.
