"""Phase 5 — Agents 2 + 3 together on every seeded scenario."""

import pytest

from agents.policy_agent import PolicyVerificationAgent, build_input
from agents.rag_agent import PolicyRAGAgent, build_rag_input
from database.repository import ClaimRepository


@pytest.fixture
def results(seeded_session, policy_index):
    rag = PolicyRAGAgent(index=policy_index, use_llm=False)
    out = []
    for claim in ClaimRepository(seeded_session).list():
        facts = build_input(claim)
        verification = PolicyVerificationAgent(seeded_session).run(facts)
        out.append((claim, verification, rag.run(build_rag_input(verification, facts))))
    return out


def test_every_claim_gets_a_rag_result(results, synthetic_data):
    assert len(results) == len(synthetic_data["claims"])


def test_rag_outcome_per_scenario(results):
    for claim, verification, rag in results:
        if claim.scenario == "UNKNOWN_POLICY":
            assert rag.status == "POLICY_CONTEXT_UNAVAILABLE" and rag.clauses == []
        elif claim.scenario == "POLICY_EXCLUSION":
            assert rag.possible_exclusions, claim.diagnosis
            assert rag.status == "EVIDENCE_FOUND" and rag.requires_human_review
        else:
            assert rag.status == "EVIDENCE_FOUND", (claim.claim_number, rag.uncertainty_reasons)
            assert rag.possible_exclusions == [], (claim.claim_number, claim.diagnosis)


def test_rag_searches_only_the_verified_plan(results):
    for _, verification, rag in results:
        if verification.policy:
            assert {c.plan_code for c in rag.clauses} == {verification.policy.plan_code}


def test_rule_flags_and_wording_agree_on_exclusions(results):
    for _, verification, rag in results:
        assert verification.flags.get("possible_exclusion", False) == bool(rag.possible_exclusions)
