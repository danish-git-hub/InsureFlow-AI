"""Phase 6 — Fraud Risk Agent: explainable, non-accusatory, non-discriminatory."""

import re
from datetime import date

import pytest

from agents.fraud_agent import DISCLAIMER, INDICATORS, FraudRiskAgent, risk_level, typical_cost_range
from agents.schemas import (
    ClaimFields,
    DocumentExtractionResult,
    ExtractedDocument,
    PolicySnapshot,
    PolicyVerificationResult,
    RuleCheck,
    ValidationInput,
    ValidationResult,
)


def facts(**over):
    sub = dict(customer_name="Rahul Sharma", policy_number="IHS-2025-000001", claim_amount=85000,
               admission_date=date(2026, 5, 20), discharge_date=date(2026, 5, 23),
               diagnosis="Acute appendicitis - laparoscopic appendectomy", invoice_number="INV-1")
    sub.update(over)
    return ValidationInput(submitted=ClaimFields(**sub), submitted_on=date(2026, 6, 1))


def policy(start=date(2025, 8, 1), si=500000, **flags):
    snap = PolicySnapshot(policy_number="IHS-2025-000001", holder_name="Rahul Sharma", customer_code="CUST-0001",
                          insurance_type="Individual Health", plan_name="x", plan_code="IHS", sum_insured=si,
                          start_date=start, end_date=date(2026, 7, 31), waiting_period_days=30, status="ACTIVE",
                          hospitalization_covered=True, exclusions=[])
    return PolicyVerificationResult(status="VERIFIED", policy=snap, flags=flags)


def validation(**flags):
    checks = [RuleCheck(code="AMOUNT_CONSISTENCY", name="a", status="PASS", message="ok"),
              RuleCheck(code="SUBMISSION_DELAY", name="d", status="PASS", message="ok")]
    return ValidationResult(status="VALID", checks=checks, flags=flags)


@pytest.fixture
def agent():
    return FraudRiskAgent()


def test_clean_claim_has_no_indicators(agent):
    r = agent.run(facts(), validation(), policy())
    assert r.risk_score == 0 and r.risk_level == "LOW" and not r.refer_for_investigation
    assert r.mode == "DETERMINISTIC_INDICATORS" and r.used_llm is False and r.disclaimer == DISCLAIMER


def test_duplicate_invoice_is_high_and_referred(agent):
    v = validation(duplicate_invoice=True)
    v.duplicate_claims = ["CLM-1"]
    r = agent.run(facts(), v, policy())
    assert [i.code for i in r.indicators] == ["DUPLICATE_INVOICE"] and r.refer_for_investigation
    assert r.indicators[0].evidence["earlier_claims"] == "CLM-1"


def test_score_is_sum_of_points_and_capped(agent):
    v = validation(duplicate_invoice=True, invalid_dates=True, field_mismatch=True)
    v.checks[0] = RuleCheck(code="AMOUNT_CONSISTENCY", name="a", status="FAIL", message="x", evidence={"claim_amount": 1})
    r = agent.run(facts(claim_amount=480000), v, policy(start=date(2026, 5, 10), holder_mismatch=True))
    assert r.risk_score == min(100, sum(i.points for i in r.indicators)) == 100
    assert r.risk_level == "HIGH"


def test_early_claim_but_not_for_accidents(agent):
    early = policy(start=date(2026, 5, 10))
    assert "EARLY_CLAIM" in {i.code for i in agent.run(facts(), validation(), early).indicators}
    accident = facts(diagnosis="Fracture of right radius - open reduction internal fixation")
    assert "EARLY_CLAIM" not in {i.code for i in agent.run(accident, validation(), early).indicators}


def test_cost_above_typical_range_and_share_of_sum_insured(agent):
    r = agent.run(facts(claim_amount=450000), validation(), policy(si=500000))
    codes = {i.code for i in r.indicators}
    assert {"COST_ABOVE_TYPICAL_RANGE", "HIGH_SHARE_OF_SUM_INSURED"} <= codes
    assert typical_cost_range("Typhoid fever") == (20000, 50000) and typical_cost_range("unknown") is None


def test_low_ocr_confidence_is_quality_not_intent(agent):
    doc = ExtractedDocument(filename="a.pdf", document_type="HOSPITAL_BILL", page_count=1, extraction_method="OCR",
                            ocr_confidence=0.6, characters=100)
    ex = DocumentExtractionResult(fields=ClaimFields(), documents=[doc])
    r = agent.run(facts(), validation(), policy(), ex)
    assert [i.code for i in r.indicators] == ["LOW_DOCUMENT_QUALITY"] and r.risk_level == "LOW"


def test_personal_attributes_do_not_change_the_score(agent):
    """Fairness: name, city, age etc. are not inputs — changing them changes nothing."""
    a = agent.run(facts(customer_name="Rahul Sharma"), validation(field_mismatch=True), policy())
    b = agent.run(facts(customer_name="Fatima Begum"), validation(field_mismatch=True), policy())
    assert a.risk_score == b.risk_score and [i.code for i in a.indicators] == [i.code for i in b.indicators]
    assert {"name", "age", "gender", "religion or caste"} <= set(a.features_not_used)


def test_language_never_accuses(agent):
    v = validation(duplicate_invoice=True, invalid_dates=True, field_mismatch=True)
    r = agent.run(facts(claim_amount=480000), v, policy(start=date(2026, 5, 10), holder_mismatch=True))
    text = " ".join([r.summary] + [i.description for i in r.indicators]).lower()
    assert not re.search(r"\bfraudulent\b|\bfraudster\b|\bdishonest|\bliar\b|\bcheat", text)
    assert "not evidence of wrongdoing" in r.disclaimer


def test_thresholds():
    assert risk_level(0) == "LOW" and risk_level(25) == "MEDIUM" and risk_level(50) == "HIGH"
    assert all(sev in ("LOW", "MEDIUM", "HIGH") and pts > 0 for sev, pts in INDICATORS.values())


def test_risk_flag_rows_match_db_columns(agent):
    v = validation(duplicate_invoice=True)
    rows = FraudRiskAgent.risk_flag_rows(agent.run(facts(), v, policy()))
    assert rows == [{"code": "DUPLICATE_INVOICE", "severity": "HIGH", "description": rows[0]["description"],
                     "evidence": rows[0]["evidence"]}]
