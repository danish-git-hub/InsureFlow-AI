"""Phase 6 — Validation Agent rules."""

from datetime import date

import pytest

from agents.schemas import ClaimFields, FieldConflict, ValidationInput
from agents.validation_agent import ValidationAgent

TODAY = date(2026, 6, 15)
ALL_DOCS = ["CLAIM_FORM", "HOSPITAL_BILL", "DISCHARGE_SUMMARY"]


def make(**over) -> ValidationInput:
    submitted = dict(customer_name="Rahul Sharma", policy_number="IHS-2025-000001", claim_amount=85000,
                     hospital_name="Sunrise Multispeciality Hospital", invoice_number="INV-SMH-2026-0001",
                     admission_date=date(2026, 5, 20), discharge_date=date(2026, 5, 23),
                     diagnosis="Acute appendicitis")
    submitted.update(over.pop("submitted", {}))
    extracted = {**submitted, "billed_amount": 85000, **over.pop("extracted", {})}
    return ValidationInput(claim_number="CLM-T", submitted=ClaimFields(**submitted),
                           extracted=ClaimFields(**extracted), document_types=over.pop("docs", ALL_DOCS),
                           submitted_on=over.pop("submitted_on", date(2026, 6, 1)), **over)


@pytest.fixture
def agent():
    return ValidationAgent(today=TODAY)


def test_clean_claim_is_valid(agent):
    r = agent.run(make())
    assert r.status == "VALID" and not any(r.flags.values())
    assert r.mode == "DETERMINISTIC_RULES" and r.used_llm is False


def test_missing_documents_make_claim_incomplete(agent):
    r = agent.run(make(docs=["CLAIM_FORM"], extracted={"billed_amount": None}))
    assert r.status == "INCOMPLETE"
    assert r.missing_documents == ["HOSPITAL_BILL", "DISCHARGE_SUMMARY"]
    assert r.flags["missing_documents"] and r.flags["missing_fields"] and "billed_amount" in r.missing_fields


def test_discharge_before_admission(agent):
    r = agent.run(make(submitted={"discharge_date": date(2026, 5, 17)}, extracted={"discharge_date": date(2026, 5, 17)}))
    assert r.status == "INVALID" and r.flags["invalid_dates"]
    assert r.check("DATE_ORDER").status == "FAIL"


def test_admission_after_submission_is_future_date(agent):
    later = {"admission_date": date(2026, 6, 10), "discharge_date": date(2026, 6, 12)}
    r = agent.run(make(submitted=later, extracted=later, submitted_on=date(2026, 6, 1)))
    assert r.check("FUTURE_DATES").status == "FAIL" and r.flags["invalid_dates"]


def test_claim_above_bill_is_amount_mismatch(agent):
    r = agent.run(make(submitted={"claim_amount": 120000}, extracted={"claim_amount": 120000, "billed_amount": 85000}))
    check = r.check("AMOUNT_CONSISTENCY")
    assert check.status == "FAIL" and check.evidence["percent_above_bill"] == pytest.approx(41.2, abs=0.1)
    assert r.flags["amount_mismatch"]


def test_claim_below_bill_and_rounding_are_fine(agent):
    assert agent.run(make(submitted={"claim_amount": 60000}, extracted={"claim_amount": 60000})).status == "VALID"
    assert agent.run(make(extracted={"billed_amount": 84000})).check("AMOUNT_CONSISTENCY").status == "PASS"  # +1.2%


def test_bill_items_must_add_up(agent):
    r = agent.run(make(bill_items_total=80000))
    assert r.check("BILL_ITEMS_TOTAL").status == "WARN" and r.status == "VALID_WITH_WARNINGS"


def test_form_and_documents_disagree(agent):
    r = agent.run(make(extracted={"policy_number": "IHS-2025-000099"},
                       conflicts=[FieldConflict(field="admission_date", values={"a.pdf": "1", "b.pdf": "2"})]))
    c = r.check("FORM_VS_DOCUMENTS")
    assert c.status == "WARN" and "policy_number" in c.evidence and "documents_admission_date" in c.evidence
    assert r.flags["field_mismatch"]


def test_long_stay_and_late_submission_warn(agent):
    dates = {"admission_date": date(2026, 1, 1), "discharge_date": date(2026, 3, 15)}
    r = agent.run(make(submitted=dates, extracted=dates, submitted_on=date(2026, 6, 14)))
    assert r.check("LENGTH_OF_STAY").status == "WARN" and r.check("SUBMISSION_DELAY").status == "WARN"


def test_duplicate_check_needs_history(agent):
    assert agent.run(make()).check("DUPLICATE_INVOICE").status == "NOT_EVALUATED"


def test_duplicate_invoice_only_flags_later_claim(seeded_session):
    from agents.validation_agent import build_validation_input
    from database.repository import ClaimRepository

    claims = ClaimRepository(seeded_session).list()
    dupes = [c for c in claims if c.scenario == "DUPLICATE_INVOICE"]
    agent = ValidationAgent(seeded_session, today=TODAY)
    for dup in dupes:
        original = next(c for c in claims if c.invoice_number == dup.invoice_number and c.id != dup.id)
        docs = ALL_DOCS
        r_dup = agent.run(build_validation_input(dup, document_types=docs))
        r_orig = agent.run(build_validation_input(original, document_types=docs))
        assert r_dup.flags["duplicate_invoice"] and original.claim_number in r_dup.duplicate_claims
        assert not r_orig.flags["duplicate_invoice"]  # the earlier claim is not penalised
