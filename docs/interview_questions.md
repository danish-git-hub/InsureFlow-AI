# InsureFlow AI — Interview Preparation

Likely questions about this project, with short answers based on the actual code.

### Architecture
**Q: Why LangGraph instead of calling the agents one after another?**
The workflow has real branches: unknown policy skips RAG, incomplete claims defer risk screening, high risk adds an escalation step, and any failure goes to a failure handler. LangGraph gives a typed shared state with reducers, explicit conditional edges that can be drawn (`python -m workflows.graph --mermaid`), and each node is testable on its own. Routers are pure functions, so routing is unit-tested without a database.

**Q: Where does the state live, and how is history kept?**
`workflows/state.py` defines `ClaimState` (a TypedDict). `execution_history` and `error_information` are `Annotated[list, operator.add]`, so each node's update is appended. At the end, `persistence.py` writes every record to `agent_executions`.

**Q: What happens if an agent throws an exception?**
The node wrapper catches it, records it in `error_information`, and sets `processing_status=FAILED`. The router sends the claim to `technical_failure`, which creates a `REVIEW_REQUIRED` recommendation explaining the failure, and the claim still reaches the human queue. This is tested with an injected failure (`fail_on={"policy_agent"}`).

### AI and RAG
**Q: How do you prevent hallucinated policy clauses?**
(1) Clause text is copied from the index, never generated. (2) Clauses below a similarity threshold are ignored. (3) An exclusion counts only if specific diagnosis words appear in the clause. (4) If Gemini's explanation cites any clause that was not retrieved, it is discarded. (5) Insufficient evidence gives `INSUFFICIENT_EVIDENCE` and routes the claim to a human.

**Q: Why chunk by clause?**
Citations must be meaningful: "IHS EXC-01 Cosmetic or aesthetic treatment, page 1". Fixed-size windows split clauses. Long clauses are still split by LangChain's recursive splitter, keeping their metadata.

**Q: What are local hashing embeddings, and why use them?**
A deterministic bag of words, word pairs and character 4-grams, hashed into 512 signed buckets and L2-normalised. It is offline, reproducible and lightweight. It is lexical only, which is why it is labelled as such and backed by thresholds and grounding.

**Q: Where is Gemini used, and how do you know it was called?**
For gap-filling in extraction, the RAG explanation, and the decision summary (and embeddings when a key is set). All calls go through `invoke_structured`, which returns an `LLMCall`; `used_llm` is set from that record, never assumed. In demo mode no call is made.

### Responsible AI
**Q: Why can't the system approve claims?**
Coverage decisions have legal and financial consequences. The decision agent's output type allows only four advisory values, `human_decision_required` is `Literal[True]`, and only `ReviewRepository.record` can set final human statuses, behind role checks.

**Q: How do you avoid a biased fraud model?**
There is no learned model. There are 11 transparent indicators about the claim and its documents, with fixed points. Personal attributes are never inputs (listed in `features_not_used`), and a unit test checks that changing the claimant's name does not change the score. The wording describes observations and the output is "refer for investigation".

**Q: Why is only the later claim flagged as a duplicate?**
Validation compares against claims submitted *before* the current one, which matches real processing order and does not penalise the original claim.

### Engineering
**Q: How do the dashboard and the API share logic?**
Both call `services/claim_service.py`. The dashboard sends writes through FastAPI when it is up and falls back to in-process calls otherwise, and the UI says which path was used.

**Q: How is file upload secured?**
Extension and `%PDF-` signature check, a size limit enforced while streaming, rejection of encrypted or empty PDFs, sanitised file names with a random prefix, all-or-nothing validation before anything is written, and SHA-256 to detect re-uploads.

**Q: How did you test without an API key?**
Fake chat models are injected (`llm_model=FakeLLM(...)`) to test valid output, invalid citations, decision language and API failures. Every scenario runs end to end on temporary databases. The API uses TestClient with dependency overrides, and the dashboard pages run with Streamlit AppTest.

**Q: What would you change for production?**
PostgreSQL with migrations, a task queue with LangGraph checkpointing, OIDC roles, object storage with malware scanning, calibrated Gemini embeddings with hybrid retrieval, and monitoring of AI recommendations against human decisions.
