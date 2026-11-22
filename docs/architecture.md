# InsureFlow AI — Architecture Notes

This document explains *why* the system is built the way it is. The README covers *what* it does and how to run it.

## 1. Layers

```
app/        Streamlit UI (pages, components, theme)         ─┐
api/        FastAPI routes, response schemas, error format    ├─ presentation
services/   claim_service: submit · process · review · ingest ─┘ application
workflows/  LangGraph state, nodes, routing, persistence       orchestration
agents/     six agents + Pydantic schemas + Gemini wrapper     domain logic
rag/        policy PDFs, loader, embeddings, FAISS retriever   retrieval
documents/  PDF generation, storage/validation, OCR            infrastructure
database/   models, repositories, catalogue, synthetic data    persistence
```

Rules that keep the layers clean:
- **Agents are pure.** They take Pydantic input and return Pydantic output. Only the Policy, Validation and Fraud agents read from the database (claim history); none of them writes.
- **The workflow is the only automated writer.** `workflows/persistence.py` saves a finished run in one place.
- **Humans are the only source of final statuses.** `ReviewRepository.record` is the only code that sets `HUMAN_APPROVED` or `HUMAN_REJECTED`, and `claim_service.record_review` enforces the role rules first.
- **One service layer.** The FastAPI routes and the Streamlit pages call the same service functions.

## 2. Request flows

### Submit and process a claim
```
Dashboard "Submit claim"
  └─ app/components/backend.submit_claim
       ├─ FastAPI online  → POST /api/v1/claims (multipart) → services.submit_claim → commit
       │                                                     → services.process  → workflows.graph.process_claim
       └─ FastAPI offline → the same service functions in-process (labelled in the UI)
```

### LangGraph run (one claim)
```
initial_state → intake → document_agent → policy_agent → (rag_agent) → validation_agent
             → (fraud_agent) → (escalate_high_risk) → decision_agent → human_review_queue → END
```
Each node returns a partial state update. `execution_history` and `error_information` use `operator.add` reducers, so each node appends rather than overwrites. Routers are pure functions of the state; the same function both drives the conditional edge and writes the human-readable `route_reason`.

## 3. Key design decisions

| Decision | Alternatives considered | Why |
|---|---|---|
| Deterministic rules for policy, validation and risk | Ask an LLM to judge the claim | Must be reproducible, auditable and explainable rule by rule. LLM judgement is not acceptable for coverage decisions or for anything that sounds like a fraud accusation. |
| LLM only for gap-filling, explanation and summary | LLM everywhere / no LLM | Uses Gemini where language ability helps. Every LLM output is validated (schema, citations, banned decision language) and falls back to rules. |
| Clause-level RAG chunks | Fixed 500-token windows | A reviewer needs a citable unit (clause code, title, page). Fixed windows split clauses and blur citations. |
| Lexical grounding for exclusions | Similarity threshold only | A short diagnosis matches a long examples list weakly. Requiring the diagnosis words in the clause avoids both false negatives and similar-sounding false positives. |
| Local hashing embeddings in demo mode | sentence-transformers | No 400 MB torch download and no network; it runs on any laptop. Clearly labelled as not neutral. |
| RapidOCR | Tesseract | pip-installable on Windows, macOS and Linux with no system binary. |
| Duplicate invoices compared only against *earlier* claims | Flag both claims | Mirrors real processing order; the original claimant is not penalised by a later resubmission. |
| Risk screening deferred for incomplete claims | Always run | Screening an incomplete claim gives misleading scores; the claim first goes back for information. |
| The technical-failure node still reaches the human queue | Abort the run | No claim can disappear silently; the reviewer sees what failed. |

## 4. Honesty and safety mechanisms

- `settings.demo_mode` is true when there is no key, and then `agents/llm.py` makes **no network call**.
- `used_llm` and `model_name` are recorded per agent and per execution row, taken from the actual `LLMCall` result.
- RAG: a Gemini explanation that cites any clause code not retrieved is discarded and a warning is recorded.
- Decision support: a Gemini summary matching `approv*|reject*|deny|fraudulent|...` is discarded.
- Fraud agent: fixed indicator table, neutral wording, an explicit `features_not_used` list, and a fairness unit test (changing the name does not change the score).
- `DecisionSupport.human_decision_required` is typed `Literal[True]`, so it cannot be false.

## 5. Data lifecycle

1. `database/synthetic.py` generates customers, policies and claims relative to a reference date (seeded RNG, reproducible).
2. `database/init_db.py` creates the tables, seeds data idempotently, generates the claim PDFs (some rasterised to force OCR), generates the policy wordings, and builds and syncs the FAISS index.
3. `workflows.graph --all` processes every claim, so the dashboard has real agent output.

## 6. Moving to production

- PostgreSQL via `DATABASE_URL` (schema already portable), with Alembic for migrations.
- Run the workflow in a worker queue. LangGraph checkpointers allow resuming after failures.
- Object storage for documents, malware scanning on upload, and signed URLs for downloads.
- Real identity (OIDC) with roles from claims, not a UI selector.
- Model monitoring: compare AI recommendations with human decisions (the snapshot is already stored).
