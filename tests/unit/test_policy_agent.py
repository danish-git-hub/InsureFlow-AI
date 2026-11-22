"""Unit tests for Agent 2 — Policy Verification Agent (one test per business rule + edge cases)."""

from datetime import date

import pytest

from agents.policy_agent import PolicyVerificationAgent, build_input, compare_names, matched_exclusions
from agents.schemas import ClaimFields, DocumentExtractionResult, PolicyVerificationInput
from database.repository import ClaimRepository, CustomerRepository, PolicyRepository

START, END = date(2026, 1, 1), date(2026, 12, 31)


@pytest.fixture
def world(session):
    """One customer with one policy per product we need."""
    cust = CustomerRepository(session).create(customer_code="CUST-0100", full_name="Meera Iyer")
    repo = PolicyRepository(session)

    def policy(number, plan_code="IHS", itype="Individual Health", **kw):
        fields = dict(policy_number=number, customer_id=cust.id, insurance_type=itype, plan_name=plan_code,
                      plan_code=plan_code, sum_insured=500_000, annual_premium=10_000, start_date=START,
                      end_date=END, waiting_period_days=30, status="ACTIVE",
                      exclusions=["Cosmetic or aesthetic treatment"])
        fields.update(kw)
        return repo.create(**fields)

    policy("IHS-2026-000001")
    policy("IHS-2025-000002", start_date=date(2025, 1, 1), end_date=date(2025, 12, 31), status="EXPIRED")
    policy("IHS-2026-000003", status="LAPSED")
    policy("PAG-2026-000004", plan_code="PAG", itype="Personal Accident", waiting_period_days=0)
    policy("SCP-2026-000005", plan_code="SCP", itype="Senior Citizen Health", waiting_period_days=90)
    policy("IHS-2026-000006", sum_insured=300_000)
    session.flush()
    return session, cust


def _run(session, **kw):
    base = dict(policy_number="IHS-2026-000001", customer_name="Meera Iyer", customer_id="CUST-0100",
                insurance_type="Individual Health", claim_amount=80_000, admission_date=date(2026, 6, 10),
                discharge_date=date(2026, 6, 13), diagnosis="Dengue fever")
    base.update(kw)
    return PolicyVerificationAgent(session).run(PolicyVerificationInput(**base))


def test_valid_claim_is_verified(world):
    r = _run(world[0])
    assert r.status == "VERIFIED" and not any(r.flags.values())
    assert r.used_llm is False and r.mode == "DETERMINISTIC_RULES"
    assert {c.code for c in r.checks} >= {"POLICY_FOUND", "POLICY_STATUS", "HOLDER_MATCH", "COVERAGE_PERIOD",
                                          "WAITING_PERIOD", "HOSPITALIZATION_COVER", "SUM_INSURED", "EXCLUSIONS"}


def test_unknown_policy_with_suggestions(world):
    r = _run(world[0], policy_number="IHS-2026-999999")
    assert r.status == "POLICY_NOT_FOUND" and r.flags["policy_not_found"] and r.policy is None
    assert any("IHS-2026-000001" in s for s in r.suggestions)


def test_missing_policy_number(world):
    r = _run(world[0], policy_number=None)
    assert r.status == "POLICY_NOT_FOUND"
    assert "No policy number" in r.check("POLICY_FOUND").message


def test_admission_after_expiry_fails(world):
    r = _run(world[0], policy_number="IHS-2025-000002", admission_date=date(2026, 2, 1))
    assert r.status == "FAILED" and r.flags["policy_inactive"]
    assert r.check("COVERAGE_PERIOD").status == "FAIL"


def test_expired_policy_but_admitted_during_term_passes(world):
    r = _run(world[0], policy_number="IHS-2025-000002", admission_date=date(2025, 7, 1),
             discharge_date=date(2025, 7, 3))
    assert r.check("POLICY_STATUS").status == "PASS"
    assert "in force on the admission date" in r.check("POLICY_STATUS").message
    assert r.status == "VERIFIED"


def test_lapsed_policy_fails(world):
    r = _run(world[0], policy_number="IHS-2026-000003")
    assert r.check("POLICY_STATUS").status == "FAIL" and r.flags["policy_inactive"]


def test_admission_before_start_fails(world):
    r = _run(world[0], admission_date=date(2025, 12, 20))
    assert "before the policy started" in r.check("COVERAGE_PERIOD").message


def test_waiting_period_incomplete(world):
    r = _run(world[0], admission_date=date(2026, 1, 15))
    assert r.check("WAITING_PERIOD").status == "FAIL" and r.flags["waiting_period_incomplete"]


def test_waiting_period_boundary_day_passes(world):
    assert _run(world[0], admission_date=date(2026, 1, 31)).check("WAITING_PERIOD").status == "PASS"
    assert _run(world[0], admission_date=date(2026, 1, 30)).check("WAITING_PERIOD").status == "FAIL"


def test_accident_exempt_from_waiting_period(world):
    r = _run(world[0], admission_date=date(2026, 1, 5), diagnosis="Fracture of left wrist after a fall")
    assert r.check("WAITING_PERIOD").status == "NOT_APPLICABLE"
    assert r.status == "VERIFIED"


def test_exceeds_sum_insured(world):
    r = _run(world[0], policy_number="IHS-2026-000006", claim_amount=350_000)
    assert r.check("SUM_INSURED").status == "FAIL" and r.flags["exceeds_sum_insured"]


def test_remaining_sum_insured_warns_after_other_claims(world):
    session, cust = world
    policy = PolicyRepository(session).get_by_number("IHS-2026-000006")
    ClaimRepository(session).create(claim_number="CLM-PRIOR", customer_id=cust.id, policy_id=policy.id,
                                    customer_name_submitted="Meera Iyer", policy_number_submitted="IHS-2026-000006",
                                    insurance_type="Individual Health", claim_amount=250_000)
    r = _run(session, policy_number="IHS-2026-000006", claim_amount=100_000)
    assert r.check("SUM_INSURED").status == "WARN"
    assert r.coverage.remaining_sum_insured == 50_000
    assert r.status == "VERIFIED_WITH_WARNINGS"
    # the claim itself is not counted against its own balance
    r2 = _run(session, policy_number="IHS-2026-000006", claim_amount=100_000, claim_number="CLM-PRIOR")
    assert r2.coverage.other_claims_on_policy == 0


def test_cosmetic_exclusion(world):
    r = _run(world[0], diagnosis="Cosmetic rhinoplasty (aesthetic nasal reshaping)")
    assert r.check("EXCLUSIONS").status == "FAIL" and r.flags["possible_exclusion"]
    assert "EXC-01" in r.check("EXCLUSIONS").message


def test_personal_accident_does_not_cover_illness(world):
    r = _run(world[0], policy_number="PAG-2026-000004", insurance_type="Personal Accident", diagnosis="Typhoid fever")
    assert r.check("EXCLUSIONS").status == "FAIL" and "EXC-PA" in r.check("EXCLUSIONS").message
    ok = _run(world[0], policy_number="PAG-2026-000004", insurance_type="Personal Accident",
              diagnosis="Road traffic accident - knee ligament tear")
    assert ok.check("EXCLUSIONS").status == "PASS" and ok.check("WAITING_PERIOD").status == "NOT_APPLICABLE"


def test_senior_plan_copay_in_indicative_amount(world):
    r = _run(world[0], policy_number="SCP-2026-000005", insurance_type="Senior Citizen Health",
             admission_date=date(2026, 6, 1), claim_amount=100_000)
    assert r.coverage.copay_percent == 10
    assert r.coverage.indicative_admissible_amount == 90_000


def test_holder_mismatch_and_partial_name(world):
    r = _run(world[0], customer_name="Someone Else", customer_id="CUST-0999")
    assert r.check("HOLDER_MATCH").status == "WARN" and r.flags["holder_mismatch"]
    assert r.status == "VERIFIED_WITH_WARNINGS"
    assert _run(world[0], customer_name="M. Iyer").check("HOLDER_MATCH").status == "WARN"


def test_insurance_type_mismatch_warns(world):
    assert _run(world[0], insurance_type="Critical Illness").check("INSURANCE_TYPE").status == "WARN"


def test_missing_admission_date_is_insufficient_data(world):
    r = _run(world[0], admission_date=None)
    assert r.status == "INSUFFICIENT_DATA" and r.flags["insufficient_data"]
    assert r.check("COVERAGE_PERIOD").status == "NOT_EVALUATED"


def test_discharge_after_expiry_is_informational(world):
    r = _run(world[0], admission_date=date(2026, 12, 29), discharge_date=date(2027, 1, 3))
    assert r.check("DISCHARGE_WITHIN_TERM").status == "WARN"
    assert r.check("COVERAGE_PERIOD").status == "PASS"


@pytest.mark.parametrize("a,b,expected", [("Rahul Sharma", "rahul  SHARMA", "EXACT"),
                                          ("Sharma Rahul", "Rahul Sharma", "EXACT"),
                                          ("R. Sharma", "Rahul Sharma", "PARTIAL"),
                                          ("Rohan Verma", "Rahul Sharma", "MISMATCH"),
                                          (None, "Rahul Sharma", "MISMATCH")])
def test_compare_names(a, b, expected):
    assert compare_names(a, b) == expected


def test_exclusion_matching_without_catalog_plan():
    hits = matched_exclusions("Dental implant", "UNKNOWN", ["Dental treatment (non-accidental)"], "Individual Health")
    assert [h.code for h in hits] == ["EXC-02"]


def test_build_input_prefers_documents_over_form(seeded_session):
    claim = ClaimRepository(seeded_session).list(search="Rahul")[0]
    extraction = DocumentExtractionResult(fields=ClaimFields(policy_number="IHS-2026-123456"))
    data = build_input(claim, extraction)
    assert data.policy_number == "IHS-2026-123456" and data.sources["policy_number"] == "documents"
    assert data.claim_amount == claim.claim_amount and data.sources["claim_amount"] == "claim_form_submission"
    assert build_input(claim).sources["policy_number"] == "claim_form_submission"
