"""Integration: Policy Verification Agent over every synthetic scenario in a seeded database."""

import pytest

from agents.policy_agent import PolicyVerificationAgent, build_input
from database.repository import ClaimRepository

EXPECTED = {
    "VALID": ("VERIFIED", set()),
    "EXPIRED_POLICY": ("FAILED", {"policy_inactive"}),
    "UNKNOWN_POLICY": ("POLICY_NOT_FOUND", {"policy_not_found"}),
    "EXCEEDS_SUM_INSURED": ("FAILED", {"exceeds_sum_insured"}),
    "WAITING_PERIOD": ("FAILED", {"waiting_period_incomplete"}),
    "POLICY_EXCLUSION": ("FAILED", {"possible_exclusion"}),
    # Policy is fine in these — other agents (validation / fraud) catch them.
    "MISSING_DOCUMENTS": ("VERIFIED", set()),
    "DUPLICATE_INVOICE": ("VERIFIED", set()),
    "INCONSISTENT_AMOUNT": ("VERIFIED", set()),
}


@pytest.mark.parametrize("scenario", sorted(EXPECTED))
def test_scenario_outcomes(seeded_session, scenario):
    agent = PolicyVerificationAgent(seeded_session)
    claims = [c for c in ClaimRepository(seeded_session).list() if c.scenario == scenario]
    assert claims
    status, flags = EXPECTED[scenario]
    for claim in claims:
        result = agent.run(build_input(claim))
        raised = {k for k, v in result.flags.items() if v}
        assert result.status == status, (claim.claim_number, result.summary)
        assert raised == flags, (claim.claim_number, raised)


def test_every_seeded_claim_verifies_without_error(seeded_session):
    agent = PolicyVerificationAgent(seeded_session)
    for claim in ClaimRepository(seeded_session).list():
        result = agent.run(build_input(claim))
        assert result.summary and result.checks
