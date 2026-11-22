"""Phase 7 — the LangGraph workflow end to end on a throw-away database."""

import pytest

from database.init_db import attach_documents
from database.repository import AgentExecutionRepository, ClaimRepository, LogRepository, RiskFlagRepository
from database.synthetic import SCENARIOS
from tests.conftest import REFERENCE_DATE
from workflows.graph import process_all, process_claim


@pytest.fixture
def run(seeded_session, synthetic_data, policy_index, tmp_path):
    """Attach PDFs for the chosen scenarios' claims, then return a runner."""
    def _run(scenario=None, **kwargs):
        rows = [c for c in synthetic_data["claims"] if scenario is None or c["scenario"] == scenario]
        attach_documents(seeded_session, {"claims": rows}, out_dir=tmp_path / "claims")
        kwargs.setdefault("rag_index", policy_index)
        kwargs.setdefault("today", REFERENCE_DATE)
        kwargs.setdefault("use_llm", False)
        if scenario is None:
            return process_all(seeded_session, **kwargs)
        claim = ClaimRepository(seeded_session).get_by_number(rows[0]["claim_number"])
        return process_claim(seeded_session, claim, **kwargs)
    return _run


def path(state):
    return [r["node"] for r in state["execution_history"]]


def test_all_scenarios_end_to_end(run, seeded_session):
    states = run()
    assert len(states) == 33
    repo = ClaimRepository(seeded_session)
    for state in states:
        claim = repo.get(state["claim_id"])
        expected = SCENARIOS[claim.scenario][1]
        assert state["decision_support"]["recommendation"] == expected, claim.claim_number
        assert state["processing_status"] == "COMPLETED" and state["error_information"] == []
        assert path(state)[0] == "intake" and path(state)[-1] == "human_review_queue"
        # persisted
        assert claim.claim_status == expected and claim.processing_status == "COMPLETED"
        assert claim.decision_support["recommendation"] == expected and claim.processed_at is not None
        assert AgentExecutionRepository(seeded_session).latest_run(claim.id)
        # demo mode: no record may claim an LLM call
        assert not any(r["used_llm"] for r in state["execution_history"])


def test_valid_claim_runs_every_agent(run, seeded_session):
    state = run("VALID")
    assert path(state) == ["intake", "document_agent", "policy_agent", "rag_agent", "validation_agent",
                           "fraud_agent", "decision_agent", "human_review_queue"]
    assert state["extracted_document_information"]["missing_fields"] == []
    assert state["policy_information"]["policy_number"]
    executions = AgentExecutionRepository(seeded_session).latest_run(state["claim_id"])
    assert [e.agent_name for e in executions] == path(state)[:-1]  # queue node persists the steps before it
    assert all(e.run_id == state["run_id"] and e.duration_ms is not None for e in executions)
    assert executions[1].output["next_step"] == "policy_agent"


def test_unknown_policy_skips_rag(run):
    state = run("UNKNOWN_POLICY")
    assert "rag_agent" not in path(state) and state["skipped_agents"] == ["rag_agent"]
    assert state["retrieved_policy_clauses"] is None
    assert state["decision_support"]["recommendation"] == "POLICY_NOT_FOUND"


def test_missing_documents_defer_risk_screening(run):
    state = run("MISSING_DOCUMENTS")
    assert "fraud_agent" not in path(state) and state["risk_assessment"] is None
    assert state["decision_support"]["recommendation"] == "ADDITIONAL_INFORMATION_REQUIRED"
    assert any("deferred" in n for n in state["decision_support"]["confidence_notes"])


def test_high_risk_is_escalated(run, seeded_session):
    state = run("DUPLICATE_INVOICE")
    assert state["risk_assessment"]["risk_level"] == "HIGH"
    assert "escalate_high_risk" in path(state) and state["escalation"]["referral_suggested"]
    assert state["decision_support"]["required_actions"][0].startswith("PRIORITY")
    flags = RiskFlagRepository(seeded_session).for_claim(state["claim_id"])
    assert "DUPLICATE_INVOICE" in {f.code for f in flags}


def test_technical_failure_routes_to_human(run, seeded_session):
    state = run("VALID", fail_on={"policy_agent"})
    assert path(state) == ["intake", "document_agent", "policy_agent", "technical_failure", "human_review_queue"]
    assert state["processing_status"] == "FAILED"
    assert state["error_information"][0]["node"] == "policy_agent"
    assert state["decision_support"]["recommendation"] == "REVIEW_REQUIRED"
    claim = ClaimRepository(seeded_session).get(state["claim_id"])
    assert claim.processing_status == "FAILED" and claim.claim_status == "REVIEW_REQUIRED"
    events = {log.event for log in LogRepository(seeded_session).recent(claim_id=claim.id)}
    assert "PROCESSING_FAILED" in events


def test_persist_false_leaves_database_untouched(run, seeded_session):
    state = run("VALID", persist=False, commit=False)
    claim = ClaimRepository(seeded_session).get(state["claim_id"])
    assert state["processing_status"] == "COMPLETED"
    assert claim.claim_status == "SUBMITTED" and claim.decision_support is None
    assert AgentExecutionRepository(seeded_session).for_claim(claim.id) == []


def test_unknown_claim_raises(seeded_session):
    with pytest.raises(LookupError):
        process_claim(seeded_session, "CLM-DOES-NOT-EXIST")
