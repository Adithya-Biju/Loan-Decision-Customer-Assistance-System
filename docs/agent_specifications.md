# HDFC Loan Intelligence System — Agent Specifications

This document defines exactly what each of the 3 agents is built from: their tools, their system prompts, and a worked example (using real rows from your dataset) showing the tool trace and the exact structured response each agent produces.

A note on design philosophy that shapes all three agents: **risk scoring is deterministic code, not an LLM guess.** The LLM's job is to decide *which tools to call* and *how to phrase the result* — the actual risk_score number comes from a fixed, documented rule function so the same loan_id always produces the same score. This is what lets the critic (Section 7 of the spec) check "did the model hallucinate a number" — if the number came from code, it can't hallucinate.

---

## Agent 1 — Loan Analysis Agent

### Built from
- A LangGraph node wrapping an LLM with **tool-calling** enabled (the LLM only orchestrates; it never computes the score itself)
- A set of read-only Postgres query functions, each scoped to one logical table
- One **deterministic risk-scoring function** (plain Python, no LLM involved) that the LLM must call rather than reason about numbers itself
- Output constrained to the `LoanAnalysis` Pydantic schema — the LLM cannot return free text here, only the schema fields

### Tools

| Tool | Input | Returns | Backing |
|---|---|---|---|
| `get_loan_details` | `loan_id` | loan amount, term, purpose, status, property area, branch | Postgres `loans` |
| `get_customer_financials` | `loan_id` | applicant/coapplicant income, household income, DTI (capped), loan-to-income (capped), EMIs, monthly expense, asset value | Postgres `financial_details` |
| `get_credit_history` | `loan_id` | CIBIL score, credit history flag, default history count | Postgres `loan_history` |
| `get_previous_loans` | `loan_id` | number of previous loans reported *on this application* | Postgres `loan_history` — see caveat below |
| `get_employment_details` | `loan_id` | employment status, length, organization type, business type | Postgres `employment_details` |
| `calculate_risk_score` | the combined record from the tools above | `risk_score` (0–100), `risk_level`, `risk_factors[]`, `positive_factors[]`, `requires_manual_review` | Deterministic rule function — **not an LLM call** |

**Caveat baked into the tool description itself:** `get_previous_loans` returns an attribute stored on this one application record, not a join across a customer's real loan history — the dataset has no reliable customer identity to join on (covered in the earlier data-profiling step). The tool's docstring says this explicitly so the LLM doesn't imply a linked history that doesn't exist.

### Risk scoring methodology (what `calculate_risk_score` actually does)

| Factor | Condition | Points |
|---|---|---|
| CIBIL Score | < 650 | +20 |
| | 650–749 | +8 |
| | ≥ 750 | 0 → positive factor |
| Credit History | 0 (none) | +15 |
| | 1 (established) | 0 → positive factor |
| Debt-to-Income Ratio | > 0.8 | +20 |
| | 0.4–0.8 | +8 |
| | < 0.4 | 0 |
| Employment Status | Unemployed | +25 |
| | Self-Employed | +8 |
| | Salaried / Retired | 0 |
| Loan-to-Income Ratio | > 6 | +15 |
| | 3–6 | +6 |
| | < 3 | 0 |
| Default History Count | ≥ 1 | +20 (capped at +40) |
| | 0 | 0 → positive factor |

`risk_score = sum of triggered points, capped at 100`
`risk_level = LOW (<40) / MEDIUM (40–70) / HIGH (>70)`
`requires_manual_review = risk_score ≥ 60 OR default_history_count ≥ 1 OR (Unemployed AND DTI > 0.8)`

This is a starting rubric — the spec explicitly leaves the methodology to you, but it needs to be this concrete and documented, not "the LLM decides."

### System prompt

```
You are the Loan Analysis Agent for HDFC's loan intelligence system.

Your job is to analyze the financial risk of ONE loan application at a time,
using ONLY data retrieved through your tools. You never see the full dataset
directly, and you never assume or invent values for fields you haven't retrieved.

Rules:
1. Always resolve the loan_id and call get_loan_details first to confirm it exists.
   If the loan_id is not found, return an ErrorResponse — do not guess or
   fabricate a plausible-looking loan.
2. Retrieve all fields you need through your tools (financials, credit history,
   previous loans, employment) before drawing any conclusion.
3. You must call calculate_risk_score to obtain risk_score, risk_level,
   risk_factors, and positive_factors. Never compute or state a risk score
   yourself — always use the tool's output verbatim.
4. Never include Aadhaar, phone number, email, PIN code, customer name,
   religion, or gender in your output or reasoning, even if a tool result
   contains them.
5. Your output must conform exactly to the LoanAnalysis schema. Do not add
   prose outside the schema fields.
6. You are producing a risk ASSESSMENT for human review, not a final lending
   decision. Never state that a loan "is approved" or "is rejected" as a
   result of your own analysis — only report requires_manual_review as
   returned by the tool.
```

### Example interaction

**User query:** "Analyze loan HDFC100125."

**Tool call trace:**
1. `get_loan_details("HDFC100125")` → loan_amount=1,753,165, term=360mo, purpose=Business, status=Rejected, property_area=Semiurban, branch=MUM-002
2. `get_customer_financials("HDFC100125")` → applicant_income=13,036, coapplicant_income=3,627, household_income=199,956, DTI=1.885, loan_to_income=8.768, existing_emis=31,406, asset_value=501,634
3. `get_credit_history("HDFC100125")` → CIBIL=616, credit_history=1, default_history_count=0
4. `get_previous_loans("HDFC100125")` → 1 previous loan reported
5. `get_employment_details("HDFC100125")` → status=Unemployed, length=5yrs, org_type=Startup
6. `calculate_risk_score(combined_record)` →
   - CIBIL 616 < 650 → +20
   - Credit history = 1 → 0 (positive)
   - DTI 1.885 > 0.8 → +20
   - Unemployed → +25
   - Loan-to-income 8.77 > 6 → +15
   - Default history 0 → 0 (positive)
   - **Total: 80 → HIGH → requires_manual_review: true**

**Final structured output (`LoanAnalysis`):**

```json
{
  "loan_id": "HDFC100125",
  "financial_risk": "HIGH",
  "risk_score": 80,
  "risk_factors": [
    "Applicant is currently unemployed",
    "Very high debt-to-income ratio (1.89x)",
    "High loan-to-income ratio (8.77x annual household income)",
    "Below-average CIBIL score (616)"
  ],
  "positive_factors": [
    "Established credit history",
    "No prior loan defaults"
  ],
  "requires_manual_review": true
}
```

---

## Agent 2 — Policy & Knowledge Agent

### Built from
- A LangGraph node wrapping an LLM for query rewriting and answer synthesis only — it never invents policy content
- A Qdrant collection of chunked policy documents (loan eligibility, CIBIL bands, DTI guidelines, documentation requirements, guarantor rules, FAQs) with metadata tags per chunk: `{doc_id, title, category, section}`
- A retrieval pipeline: rewrite → multi-query search → dedupe/rerank → compress → cite

### Tools

| Tool | Input | Returns | Backing |
|---|---|---|---|
| `rewrite_and_expand_query` | raw user question | 1 canonical + 2 alternate phrasings | LLM step |
| `qdrant_similarity_search` | query variant, optional `category` filter, `top_k` | ranked chunks with scores + metadata | Qdrant |
| `dedupe_and_rerank` | pooled chunks from all variants | top-N unique, reranked chunks | Reranker step |
| `compress_context` | top-N chunks | condensed, attribution-preserving context block | LLM/rule step |
| `format_citations` | selected chunks | citation strings (doc + section) | Formatting step |

### System prompt

```
You are the Policy & Knowledge Agent for HDFC's loan intelligence system.

You answer questions about loan policy, eligibility, and general lending
concepts using ONLY the content retrieved from the policy knowledge base via
your tools. You have no other source of policy truth — not your own training
knowledge, not assumptions about "how banks usually work."

Rules:
1. Always retrieve chunks before answering. Never answer a policy question
   from memory alone.
2. Every factual claim in your answer must be traceable to a specific
   retrieved chunk. Attach a citation to each claim.
3. If retrieval returns no chunks above a reasonable relevance threshold, or
   the chunks don't actually address the question, say plainly that the
   knowledge base doesn't cover this — do not fill the gap with invented
   policy.
4. Do not discuss or reference any specific customer's application here —
   that belongs to the Loan Analysis Agent. You only discuss general policy.
5. Output must conform to the RAGResponse schema.
```

### Example interaction

**User query:** "What CIBIL score is generally considered strong for a loan?"

**Tool call trace:**
1. `rewrite_and_expand_query` → variants: "strong CIBIL score for loan approval", "minimum CIBIL score good rating", "CIBIL score interpretation bands"
2. `qdrant_similarity_search` × 3 variants, filter `category=cibil` → 6 candidate chunks
3. `dedupe_and_rerank` → top 2: `CIBIL_Score_Interpretation.md §2` and `Loan_Eligibility_Guidelines.md §1`
4. `compress_context` → merged ~90-word context
5. `format_citations` → citation strings built

**Final structured output (`RAGResponse`):**

```json
{
  "query": "What CIBIL score is generally considered strong for a loan?",
  "answer": "A CIBIL score of 750 or above is generally treated as strong and improves approval odds and terms. Scores between 650–749 are considered moderate and may require additional documentation or a guarantor. Scores below 650 are treated as high-risk under current guidelines.",
  "citations": [
    {"source": "CIBIL_Score_Interpretation.md", "section": "Score bands"},
    {"source": "Loan_Eligibility_Guidelines.md", "section": "Credit criteria"}
  ],
  "coverage": "fully_supported"
}
```

---

## Agent 3 — Decision Review & Customer Agent

### Built from
- A LangGraph node that treats **Agent 1 and Agent 2 as callable sub-agents** (agent-as-tool pattern), not as raw DB/vector tools itself
- A rule-based escalation checker (deterministic, same philosophy as risk scoring)
- One additional Postgres tool for customer-facing scenarios
- Two output schemas depending on audience: `FinalRecommendation` (loan officer) and `CustomerGuidance` (customer-facing)

### Tools

| Tool | Input | Returns | Backing |
|---|---|---|---|
| `invoke_loan_analysis_agent` | `loan_id` | `LoanAnalysis` object | Agent 1 (sub-call) |
| `invoke_policy_agent` | question, optional category filter | `RAGResponse` object | Agent 2 (sub-call) |
| `get_customer_feedback` | `loan_id` | feedback text + sentiment | Postgres `customer_feedback` |
| `compare_against_policy` | `LoanAnalysis` + `RAGResponse` | grounded synthesis linking each risk factor to a policy citation | LLM step, strictly grounded |
| `determine_escalation` | `LoanAnalysis`, `RAGResponse`, conversation state | `requires_human_review`, `escalation_reason` | Deterministic rule function |
| `generate_final_response` | everything above | final schema-conformant output | LLM formatting step |

### Escalation rules (`determine_escalation`)

| Condition | Result |
|---|---|
| `risk_score ≥ 80` | Escalate — high risk |
| `default_history_count ≥ 1` and `risk_score ≥ 60` | Escalate |
| Policy coverage is `not_covered` or `partially_supported` | Escalate — can't ground the explanation |
| User message disputes the data (e.g. "that's not my CIBIL score") | Escalate |
| Contradictory fields (e.g. `credit_history=1` but `default_history_count≥2`) | Escalate |

### System prompt

```
You are the Decision Review & Customer Agent for HDFC's loan intelligence
system — the final layer a loan officer or customer actually talks to.

You never invent financial facts or policy claims yourself. Every factual
statement you make must come from either the Loan Analysis Agent's output or
the Policy Agent's output, which you retrieve by calling them. Your own job
is synthesis, tone, and escalation — not invention.

Rules:
1. For any question referencing a specific loan_id, call
   invoke_loan_analysis_agent before saying anything about that application.
2. For any question involving "why" or policy justification, also call
   invoke_policy_agent and connect it explicitly to the specific risk factors
   returned by the Loan Analysis Agent — don't state policy and financial
   facts as separate, disconnected paragraphs.
3. Always call determine_escalation and include its result. If it flags
   human review, say so plainly and explain why in one sentence.
4. Never expose Aadhaar, phone, email, PIN code, name, religion, or gender.
5. You are producing an AI-assisted recommendation for human validation, not
   a final decision. State this in every decision-review response.
6. Match your tone and schema to the audience: FinalRecommendation for
   loan-officer-style "why" questions, CustomerGuidance for "what can I
   improve" style questions.
```

### Example interaction 1 — Decision review

**User query:** "Why was HDFC100125 rejected?"

**Tool call trace:**
1. Router detects `loan_id=HDFC100125`, intent=`decision_review`
2. `invoke_loan_analysis_agent("HDFC100125")` → the `LoanAnalysis` object from Agent 1's example above (risk_score=80, HIGH, requires_manual_review=true)
3. `invoke_policy_agent("why do high DTI and unemployment increase rejection risk", filter=[dti, employment])` → `RAGResponse` citing `DTI_Guidelines.md` and `Employment_Eligibility.md`
4. `compare_against_policy` → maps each risk factor to its supporting policy citation
5. `determine_escalation` → risk_score 80 ≥ 80 → **escalate: true**
6. `generate_final_response` → `FinalRecommendation`

**Final structured output (`FinalRecommendation`):**

```json
{
  "loan_id": "HDFC100125",
  "audience": "loan_officer",
  "decision_summary": "Rejected — driven primarily by unemployment status combined with a debt-to-income ratio far above policy limits, alongside a below-average CIBIL score.",
  "supporting_factors": [
    {"factor": "Unemployed applicant", "policy_basis": "Employment_Eligibility.md — stable income source required for unsecured/high-ticket loans"},
    {"factor": "DTI of 1.89x", "policy_basis": "DTI_Guidelines.md — DTI above 0.8 classified as high risk"},
    {"factor": "CIBIL score 616", "policy_basis": "CIBIL_Score_Interpretation.md — below 650 treated as high-risk band"}
  ],
  "requires_human_review": true,
  "escalation_reason": "Computed risk score (80) meets the high-risk escalation threshold.",
  "disclaimer": "This is an AI-generated recommendation for loan officer validation, not a final lending decision."
}
```

### Example interaction 2 — Customer-facing improvement

**User query:** "What can I improve before applying again?" *(same loan, session already has `loan_id=HDFC100125` in memory — no ID re-stated)*

**Tool call trace:**
1. Router resolves `loan_id` from session state (memory, Section 11) since none was typed this turn
2. `invoke_loan_analysis_agent("HDFC100125")` → reuses/refetches the same `LoanAnalysis`
3. `get_customer_feedback("HDFC100125")` → "Very dissatisfied — long waits and inconsistent updates" / sentiment: Neutral
4. `invoke_policy_agent("how to improve DTI and employment status for loan eligibility")` → guidance chunks
5. `generate_final_response` → `CustomerGuidance` (no escalation check needed — this is advisory, not a decision)

**Final structured output (`CustomerGuidance`):**

```json
{
  "loan_id": "HDFC100125",
  "audience": "customer",
  "recommendations": [
    "Establish a steady, verifiable income source before reapplying — unemployment was the largest single factor in this outcome.",
    "Reduce existing monthly obligations relative to income; your debt-to-income ratio was well above the accepted range.",
    "Work on improving your CIBIL score before your next application, ideally into the 650+ range."
  ],
  "tone": "supportive",
  "disclaimer": "These are general suggestions based on this application's data, not a guarantee of future approval."
}
```

---

## How this ties back to the LangGraph state

All three agents read and write the same shared state object rather than passing data through any database join. The fields that matter most:

| State field | Set by | Read by |
|---|---|---|
| `current_loan_id` | Router / memory | All three agents |
| `loan_analysis` | Agent 1 | Agent 3 |
| `policy_response` | Agent 2 | Agent 3 |
| `requires_human_review` | Agent 3's `determine_escalation` | API layer, critic node |
| `conversation_history` | Every turn | Router (for "why?" follow-ups) |

Next natural step: design the LangGraph graph itself (nodes, conditional edges, the critic/retry loop from Section 7) using these three agents as the graph's nodes — want to do that next?
