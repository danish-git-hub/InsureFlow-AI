"""
Shared claim state for the LangGraph workflow.

Beginner note:
    In LangGraph every node receives the *current* state and returns a dict of
    the keys it wants to change. LangGraph merges that update into the state
    and passes it to the next node. Keys annotated with a *reducer*
    (`operator.add` below) are appended to instead of overwritten — that is how
    `execution_history` grows by one entry per agent.

All values are plain JSON-compatible data (agent results are stored with
`model_dump(mode="json")`), so the state can be shown in the dashboard,
returned by the API and saved to the database without conversion.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

AGENT_ORDER: tuple[str, ...] = (
    "document_agent", "policy_agent", "rag_agent", "validation_agent", "fraud_agent", "decision_agent",
)
AGENT_TITLES: dict[str, str] = {
    "intake": "Claim Intake",
    "document_agent": "Document Extraction Agent",
    "policy_agent": "Policy Verification Agent",
    "rag_agent": "Policy RAG Agent",
    "validation_agent": "Claim Validation Agent",
    "fraud_agent": "Fraud Risk Screening Agent",
    "escalate_high_risk": "High-Risk Escalation",
    "decision_agent": "Decision Support Agent",
    "technical_failure": "Technical Failure Handler",
    "human_review_queue": "Human Review Queue",
}


class ExecutionRecord(TypedDict, total=False):
    node: str
    title: str
    sequence: int
    status: str  # SUCCESS | WARNING | FAILED | SKIPPED
    started_at: str
    finished_at: str
    duration_ms: float
    used_llm: bool
    model_name: str | None
    input_summary: dict[str, Any]
    output_summary: dict[str, Any]
    next_step: str | None
    route_reason: str | None
    error: str | None


class ClaimState(TypedDict, total=False):
    # --- identity -------------------------------------------------------------
    run_id: str
    claim_id: int | None
    claim_number: str | None

    # --- the 13 shared fields required by the specification -------------------
    customer_information: dict[str, Any]
    uploaded_documents: list[dict[str, Any]]       # {path, document_type, filename}
    extracted_document_information: dict[str, Any] | None
    policy_information: dict[str, Any] | None      # policy snapshot (None when not found)
    policy_verification: dict[str, Any] | None
    retrieved_policy_clauses: dict[str, Any] | None
    claim_validation: dict[str, Any] | None
    risk_assessment: dict[str, Any] | None
    decision_support: dict[str, Any] | None
    processing_status: str                          # PENDING | RUNNING | COMPLETED | FAILED
    execution_history: Annotated[list[ExecutionRecord], operator.add]
    error_information: Annotated[list[dict[str, Any]], operator.add]

    # --- workflow bookkeeping -------------------------------------------------
    submitted_claim: dict[str, Any]                 # claim form values as submitted
    current_stage: str
    skipped_agents: Annotated[list[str], operator.add]
    escalation: dict[str, Any] | None
    started_at: str
    finished_at: str | None


def initial_state(run_id: str, claim_id: int | None = None, claim_number: str | None = None) -> ClaimState:
    return ClaimState(
        run_id=run_id, claim_id=claim_id, claim_number=claim_number, customer_information={},
        uploaded_documents=[], extracted_document_information=None, policy_information=None,
        policy_verification=None, retrieved_policy_clauses=None, claim_validation=None, risk_assessment=None,
        decision_support=None, processing_status="PENDING", execution_history=[], error_information=[],
        submitted_claim={}, current_stage="START", skipped_agents=[], escalation=None, started_at="",
        finished_at=None,
    )
