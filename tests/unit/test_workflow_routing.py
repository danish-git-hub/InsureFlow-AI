"""Phase 7 — routing rules and graph structure (no database needed)."""

from workflows.graph import build_graph, graph_mermaid
from workflows.nodes import (
    WorkflowContext,
    decide_after_documents,
    decide_after_fraud,
    decide_after_policy,
    decide_after_rag,
    decide_after_validation,
)
from workflows.state import ClaimState, initial_state

REQUIRED_STATE_FIELDS = {
    "claim_id", "customer_information", "uploaded_documents", "extracted_document_information",
    "policy_information", "policy_verification", "retrieved_policy_clauses", "claim_validation",
    "risk_assessment", "decision_support", "processing_status", "execution_history", "error_information",
}


def test_state_has_every_specified_field():
    assert REQUIRED_STATE_FIELDS <= set(ClaimState.__annotations__)
    assert REQUIRED_STATE_FIELDS <= set(initial_state("run-1", 1))


def test_graph_contains_all_agents_and_paths():
    graph = build_graph(WorkflowContext(session=None, persist=False)).get_graph()
    nodes = set(graph.nodes)
    assert {"intake", "document_agent", "policy_agent", "rag_agent", "validation_agent", "fraud_agent",
            "escalate_high_risk", "decision_agent", "technical_failure", "human_review_queue"} <= nodes
    edges = {(e.source, e.target) for e in graph.edges}
    assert ("policy_agent", "validation_agent") in edges        # unknown policy skips RAG
    assert ("validation_agent", "decision_agent") in edges      # incomplete claim skips risk screening
    assert ("fraud_agent", "escalate_high_risk") in edges       # high risk escalation
    assert ("technical_failure", "human_review_queue") in edges
    assert ("human_review_queue", "__end__") in edges
    assert "human_review_queue" in graph_mermaid()


def test_every_failure_goes_to_technical_failure():
    failed = {"processing_status": "FAILED"}
    for decide in (decide_after_documents, decide_after_policy, decide_after_rag, decide_after_validation,
                   decide_after_fraud):
        assert decide(failed)[0] == "technical_failure"


def test_unknown_policy_skips_rag():
    nxt, reason, skipped = decide_after_policy({"policy_verification": {"status": "POLICY_NOT_FOUND", "flags": {}}})
    assert nxt == "validation_agent" and skipped == ["rag_agent"] and "Unknown policy" in reason


def test_waiting_period_and_uncertain_coverage_are_explained():
    pv = {"status": "FAILED", "flags": {"waiting_period_incomplete": True}}
    assert "Waiting period" in decide_after_policy({"policy_verification": pv})[1]
    rag = {"status": "INSUFFICIENT_EVIDENCE"}
    assert "Uncertain policy coverage" in decide_after_rag({"retrieved_policy_clauses": rag})[1]


def test_missing_documents_and_invalid_dates_defer_risk_screening():
    for flag in ("missing_documents", "missing_fields", "invalid_dates"):
        nxt, _, skipped = decide_after_validation({"claim_validation": {"flags": {flag: True}}})
        assert nxt == "decision_agent" and skipped == ["fraud_agent"]
    assert decide_after_validation({"claim_validation": {"flags": {}}})[0] == "fraud_agent"


def test_high_risk_escalates():
    assert decide_after_fraud({"risk_assessment": {"risk_level": "HIGH", "risk_score": 55}})[0] == "escalate_high_risk"
    assert decide_after_fraud({"risk_assessment": {"risk_level": "MEDIUM"}})[0] == "decision_agent"
