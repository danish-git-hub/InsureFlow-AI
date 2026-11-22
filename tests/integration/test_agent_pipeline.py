"""Phase 6 — all agents (2-6) on every seeded scenario reach the expected recommendation."""

import pytest

from agents.decision_agent import FORBIDDEN_LANGUAGE, DecisionSupportAgent
from agents.fraud_agent import FraudRiskAgent
from agents.policy_agent import PolicyVerificationAgent, build_input
from agents.rag_agent import PolicyRAGAgent, build_rag_input
from agents.validation_agent import ValidationAgent, build_validation_input
from database.repository import ClaimRepository
from database.synthetic import SCENARIOS
from tests.conftest import REFERENCE_DATE


@pytest.fixture
def outcomes(seeded_session, synthetic_data, policy_index):
    docs_by_claim = {c["claim_number"]: c["documents"] for c in synthetic_data["claims"]}
    rag_agent = PolicyRAGAgent(index=policy_index, use_llm=False)
    decision_agent = DecisionSupportAgent(use_llm=False)
    out = []
    for claim in ClaimRepository(seeded_session).list():
        facts = build_input(claim)
        policy = PolicyVerificationAgent(seeded_session).run(facts)
        rag = rag_agent.run(build_rag_input(policy, facts))
        vin = build_validation_input(claim, document_types=docs_by_claim[claim.claim_number])
        validation = ValidationAgent(seeded_session, today=REFERENCE_DATE).run(vin)
        risk = FraudRiskAgent(seeded_session).run(vin, validation, policy)
        out.append((claim, validation, risk, decision_agent.run(policy, rag, validation, risk)))
    return out


def test_every_scenario_reaches_expected_recommendation(outcomes):
    wrong = [(c.claim_number, c.scenario, d.recommendation) for c, _, _, d in outcomes
             if d.recommendation != SCENARIOS[c.scenario][1]]
    assert wrong == []


def test_risk_indicators_on_expected_scenarios(outcomes):
    for claim, validation, risk, _ in outcomes:
        codes = {i.code for i in risk.indicators}
        if claim.scenario == "DUPLICATE_INVOICE":
            assert "DUPLICATE_INVOICE" in codes and risk.refer_for_investigation
        if claim.scenario == "INCONSISTENT_AMOUNT":
            assert "AMOUNT_ABOVE_BILL" in codes and risk.risk_level != "LOW"
        if claim.scenario == "VALID":
            assert risk.risk_level == "LOW" and validation.status == "VALID"


def test_no_output_ever_decides_or_accuses(outcomes):
    for _, _, risk, decision in outcomes:
        assert decision.human_decision_required is True
        assert not FORBIDDEN_LANGUAGE.search(decision.summary)
        assert not FORBIDDEN_LANGUAGE.search(risk.summary + " ".join(i.description for i in risk.indicators))
