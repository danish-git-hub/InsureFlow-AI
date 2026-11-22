"""Phase 6 — Decision Support Agent: fixed precedence, never a decision."""

import pytest

from agents.decision_agent import FORBIDDEN_LANGUAGE, DecisionSupportAgent, _Summary
from agents.schemas import (
    CoverageSummary,
    PolicyRAGResult,
    PolicyVerificationResult,
    RetrievedClause,
    RiskAssessment,
    RiskIndicator,
    RuleCheck,
    ValidationResult,
)


def pol(status="VERIFIED", checks=(), suggestions=()):
    cov = CoverageSummary(sum_insured=500000, other_claims_on_policy=0, remaining_sum_insured=500000,
                          claim_amount=85000, copay_percent=0, indicative_admissible_amount=85000)
    return PolicyVerificationResult(status=status, checks=list(checks), suggestions=list(suggestions), coverage=cov,
                                    summary=f"policy {status}")


def rag(status="EVIDENCE_FOUND", exclusion=False):
    clauses = []
    if exclusion:
        clauses.append(RetrievedClause(topic="exclusions", clause_code="EXC-01", title="Cosmetic", category="EXCLUSION",
                                       plan_code="IHS", source_document="IHS_policy_wording.pdf", page_number=2,
                                       score=0.3, text="...", matched_terms=["cosmetic"]))
    return PolicyRAGResult(status=status, clauses=clauses, possible_exclusions=["EXC-01"] if exclusion else [],
                           uncertainty_reasons=["thin"] if status != "EVIDENCE_FOUND" else [])


def val(status="VALID", checks=(), missing=(), **flags):
    return ValidationResult(status=status, checks=list(checks), missing_documents=list(missing), flags=flags)


def risk(score=0, level="LOW", indicators=(), refer=False):
    return RiskAssessment(risk_score=score, risk_level=level, indicators=list(indicators),
                          refer_for_investigation=refer, summary=f"risk {score}")


@pytest.fixture
def agent():
    return DecisionSupportAgent(use_llm=False)


def test_clean_claim_is_eligible_for_human_review(agent):
    d = agent.run(pol(), rag(), val(), risk())
    assert d.recommendation == "ELIGIBLE_FOR_HUMAN_REVIEW" and d.human_decision_required is True
    assert d.indicative_admissible_amount == 85000 and "authorised claims officer" in d.disclaimer
    assert not FORBIDDEN_LANGUAGE.search(d.summary)


def test_policy_not_found_has_highest_precedence(agent):
    v = val(checks=[RuleCheck(code="REQUIRED_DOCUMENTS", name="d", status="FAIL", message="Missing bill.")],
            missing=["HOSPITAL_BILL"], missing_documents=True)
    d = agent.run(pol("POLICY_NOT_FOUND", suggestions=["IHS-2025-000005"]), rag("POLICY_CONTEXT_UNAVAILABLE"), v, risk())
    assert d.recommendation == "POLICY_NOT_FOUND"
    assert any("IHS-2025-000005" in a for a in d.required_actions)


def test_missing_documents_before_review(agent):
    v = val("INCOMPLETE", checks=[RuleCheck(code="REQUIRED_DOCUMENTS", name="d", status="FAIL", message="Missing bill.")],
            missing=["HOSPITAL_BILL"], missing_documents=True)
    d = agent.run(pol("FAILED", [RuleCheck(code="SUM_INSURED", name="SI", status="FAIL", message="too much")]),
                  rag(), v, risk(60, "HIGH", refer=True))
    assert d.recommendation == "ADDITIONAL_INFORMATION_REQUIRED"
    assert "Request the hospital bill from the claimant." in d.required_actions


def test_invalid_dates_need_information(agent):
    v = val("INVALID", checks=[RuleCheck(code="DATE_ORDER", name="o", status="FAIL", message="Discharge before admission.")],
            invalid_dates=True)
    assert agent.run(pol(), rag(), v, risk()).recommendation == "ADDITIONAL_INFORMATION_REQUIRED"


@pytest.mark.parametrize("case", ["policy_fail", "exclusion", "rag_thin", "medium_risk", "duplicate", "warning"])
def test_review_required_triggers(agent, case):
    p, r, v, k = pol(), rag(), val(), risk()
    if case == "policy_fail":
        p = pol("FAILED", [RuleCheck(code="WAITING_PERIOD", name="Waiting period", status="FAIL", message="too early")])
    elif case == "exclusion":
        r = rag(exclusion=True)
    elif case == "rag_thin":
        r = rag("INSUFFICIENT_EVIDENCE")
    elif case == "medium_risk":
        k = risk(30, "MEDIUM")
    elif case == "duplicate":
        v = val("INVALID", checks=[RuleCheck(code="DUPLICATE_INVOICE", name="d", status="FAIL", message="used before")],
                duplicate_invoice=True)
    elif case == "warning":
        v = val("VALID_WITH_WARNINGS", checks=[RuleCheck(code="LENGTH_OF_STAY", name="s", status="WARN", message="long")])
    d = agent.run(p, r, v, k)
    assert d.recommendation == "REVIEW_REQUIRED"
    assert any(reason.blocking for reason in d.reasons)


def test_low_risk_indicators_are_shown_but_do_not_route(agent):
    k = risk(10, "LOW", [RiskIndicator(code="HIGH_SHARE_OF_SUM_INSURED", severity="LOW", points=10, description="80%")])
    d = agent.run(pol(), rag(), val(), k)
    assert d.recommendation == "ELIGIBLE_FOR_HUMAN_REVIEW"
    assert any(r.code == "HIGH_SHARE_OF_SUM_INSURED" and not r.blocking for r in d.reasons)


def test_confidence_reflects_evidence_completeness(agent):
    full = agent.run(pol(), rag(), val(), risk())
    thin = agent.run(pol("INSUFFICIENT_DATA"), rag("INSUFFICIENT_EVIDENCE"), val(), risk())
    assert full.confidence > thin.confidence >= 0.3
    assert thin.confidence_notes


def test_outputs_are_only_the_four_recommendations():
    from typing import get_args

    from agents.schemas import Recommendation
    assert set(get_args(Recommendation)) == {"ELIGIBLE_FOR_HUMAN_REVIEW", "ADDITIONAL_INFORMATION_REQUIRED",
                                             "POLICY_NOT_FOUND", "REVIEW_REQUIRED"}


class _Fake:
    def __init__(self, result):
        self.result = result

    def with_structured_output(self, schema):
        return self

    def invoke(self, prompt):
        self.prompt = prompt
        return self.result


def test_gemini_summary_used_when_neutral():
    fake = _Fake(_Summary(summary="The claim is routed as ELIGIBLE_FOR_HUMAN_REVIEW; the officer decides."))
    d = DecisionSupportAgent(llm_model=fake).run(pol(), rag(), val(), risk())
    assert d.summary_source == "gemini" and d.used_llm and "Do not say the claim is approved" in fake.prompt


@pytest.mark.parametrize("bad", ["This claim is approved.", "The claim should be rejected.", "The customer is fraudulent."])
def test_gemini_summary_with_decision_language_is_discarded(bad):
    d = DecisionSupportAgent(llm_model=_Fake(_Summary(summary=bad))).run(pol(), rag(), val(), risk())
    assert d.summary_source == "template" and bad not in d.summary
    assert any("discarded" in w for w in d.warnings)
