"""Unit tests for ORM models, constraints and repositories."""

from datetime import date

import pytest
from sqlalchemy.exc import IntegrityError

from database.models import ClaimStatus, ProcessingStatus
from database.repository import (
    AgentExecutionRepository,
    AnalyticsRepository,
    ClaimRepository,
    CustomerRepository,
    DocumentRepository,
    LogRepository,
    PolicyClauseRepository,
    PolicyRepository,
    ReviewRepository,
    RiskFlagRepository,
)


def _customer(session, code="CUST-9001"):
    return CustomerRepository(session).create(customer_code=code, full_name="Test Person")


def _policy(session, customer, number="IHS-2026-900001", **kw):
    fields = dict(policy_number=number, customer_id=customer.id, insurance_type="Individual Health",
                  plan_name="InsureFlow Health Secure", plan_code="IHS", sum_insured=500000, annual_premium=14000,
                  start_date=date(2026, 1, 1), end_date=date(2026, 12, 31), waiting_period_days=30)
    fields.update(kw)
    return PolicyRepository(session).create(**fields)


def _claim(session, customer, policy, **kw):
    fields = dict(customer_id=customer.id, policy_id=policy.id if policy else None,
                  customer_name_submitted=customer.full_name, policy_number_submitted="IHS-2026-900001",
                  insurance_type="Individual Health", claim_amount=50000, invoice_number="INV-T-1")
    fields.update(kw)
    return ClaimRepository(session).create(**fields)


def test_create_and_fetch_policy_with_customer(session):
    c = _customer(session)
    _policy(session, c)
    p = PolicyRepository(session).get_by_number("ihs-2026-900001")  # case-insensitive input
    assert p.customer.full_name == "Test Person"


def test_unique_policy_number_constraint(session):
    c = _customer(session)
    _policy(session, c)
    with pytest.raises(IntegrityError):
        _policy(session, c)


def test_check_constraint_rejects_invalid_status(session):
    c = _customer(session)
    with pytest.raises(IntegrityError):
        _policy(session, c, status="BOGUS")


def test_check_constraint_rejects_negative_sum_insured(session):
    c = _customer(session)
    with pytest.raises(IntegrityError):
        _policy(session, c, sum_insured=-1)


def test_foreign_key_enforced(session):
    with pytest.raises(IntegrityError):
        ClaimRepository(session).create(customer_id=99999, customer_name_submitted="x",
                                        policy_number_submitted="x", insurance_type="Individual Health",
                                        claim_amount=1)


def test_claim_number_generated_and_defaults(session):
    c = _customer(session)
    claim = _claim(session, c, _policy(session, c))
    assert claim.claim_number.startswith("CLM-")
    assert claim.claim_status == ClaimStatus.SUBMITTED
    assert claim.processing_status == ProcessingStatus.PENDING


def test_generated_claim_numbers_are_unique():
    assert len({ClaimRepository.generate_claim_number() for _ in range(500)}) == 500


def test_find_by_invoice_excludes_self(session):
    c = _customer(session)
    p = _policy(session, c)
    a = _claim(session, c, p)
    b = _claim(session, c, p, invoice_number="inv-t-1")
    repo = ClaimRepository(session)
    assert [x.id for x in repo.find_by_invoice("INV-T-1", exclude_claim_id=a.id)] == [b.id]


def test_save_agent_results_rejects_unknown_field(session):
    c = _customer(session)
    claim = _claim(session, c, None)
    repo = ClaimRepository(session)
    repo.save_agent_results(claim, risk_level="LOW", decision_support={"recommendation": "REVIEW_REQUIRED"})
    assert claim.risk_level == "LOW"
    with pytest.raises(ValueError):
        repo.save_agent_results(claim, password="nope")


def test_child_records_and_cascade_delete(session):
    c = _customer(session)
    claim = _claim(session, c, _policy(session, c))
    DocumentRepository(session).add(claim_id=claim.id, original_filename="bill.pdf", stored_filename="x.pdf",
                                    file_path="/tmp/x.pdf", file_size_bytes=10, sha256="ab" * 32)
    AgentExecutionRepository(session).record(claim_id=claim.id, run_id="r1", agent_name="document_agent",
                                             sequence=1, status="SUCCESS", duration_ms=12.5)
    RiskFlagRepository(session).replace_for_claim(claim.id, [{"code": "DUP", "severity": "HIGH", "description": "d"}])
    LogRepository(session).log("test", "EVT", "msg", claim_id=claim.id)
    session.flush()
    assert len(DocumentRepository(session).find_by_hash("ab" * 32)) == 1
    assert len(AgentExecutionRepository(session).latest_run(claim.id)) == 1
    session.delete(claim)
    session.flush()
    assert DocumentRepository(session).for_claim(claim.id) == []
    assert RiskFlagRepository(session).for_claim(claim.id) == []


def test_risk_flags_replaced_per_run(session):
    c = _customer(session)
    claim = _claim(session, c, None)
    repo = RiskFlagRepository(session)
    repo.replace_for_claim(claim.id, [{"code": "A", "severity": "LOW", "description": "a"}] * 2)
    repo.replace_for_claim(claim.id, [{"code": "B", "severity": "HIGH", "description": "b"}])
    assert [f.code for f in repo.for_claim(claim.id)] == ["B"]


def test_human_review_keeps_ai_snapshot_separate(session):
    c = _customer(session)
    claim = _claim(session, c, None, claim_status=ClaimStatus.ELIGIBLE_FOR_HUMAN_REVIEW.value,
                   decision_support={"recommendation": "ELIGIBLE_FOR_HUMAN_REVIEW"})
    review = ReviewRepository(session).record(claim, "Asha Reviewer", "APPROVED", "Docs verified")
    assert review.ai_recommendation_snapshot == "ELIGIBLE_FOR_HUMAN_REVIEW"
    assert claim.claim_status == ClaimStatus.HUMAN_APPROVED
    with pytest.raises(ValueError):
        ReviewRepository(session).record(claim, "Asha", "AUTO_APPROVE")


def test_policy_clauses_reindex(session):
    repo = PolicyClauseRepository(session)
    repo.replace_for_document("IHS.pdf", [{"category": "EXCLUSION", "text": "a", "page_number": 1}] * 3)
    repo.replace_for_document("IHS.pdf", [{"category": "COVERAGE", "text": "b", "page_number": 2}])
    assert repo.count() == 1


# ----------------------------- seeded database ----------------------------- #
def test_seed_counts_and_idempotency(seeded_session, synthetic_data):
    from database.init_db import seed

    counts = AnalyticsRepository(seeded_session).table_counts()
    assert counts["customers"] == len(synthetic_data["customers"])
    assert counts["policies"] == len(synthetic_data["policies"])
    assert counts["claims"] == len(synthetic_data["claims"])
    assert seed(seeded_session, synthetic_data) == {"customers": 0, "policies": 0, "claims": 0}


def test_unknown_policy_claims_have_no_policy_link(seeded_session):
    unknown = [c for c in ClaimRepository(seeded_session).list() if c.scenario == "UNKNOWN_POLICY"]
    assert unknown and all(c.policy_id is None for c in unknown)


def test_claim_filters(seeded_session):
    repo = ClaimRepository(seeded_session)
    assert len(repo.list(search="Rahul")) == 1
    assert all(c.insurance_type == "Family Floater" for c in repo.list(insurance_type="Family Floater"))
    assert len(repo.pending_processing()) == repo.count()
    assert repo.review_queue() == []


def test_analytics_from_real_records(seeded_session, synthetic_data):
    a = AnalyticsRepository(seeded_session)
    k = a.kpis()
    assert k["total_claims"] == len(synthetic_data["claims"])
    assert k["total_claim_amount"] == pytest.approx(sum(c["claim_amount"] for c in synthetic_data["claims"]))
    assert sum(m["claims"] for m in a.monthly_claims()) == k["total_claims"]
    assert sum(a.claims_by_type().values()) == k["total_claims"]
    assert a.risk_distribution() == {"UNASSESSED": k["total_claims"]}
