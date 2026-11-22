"""
Agent 5 — Fraud Risk Agent (risk *indicators* for investigation).

Purpose:
    Point out observable patterns that a claims investigator may want to look
    at — a reused invoice, a claim larger than the bill, a claim very soon
    after the policy started — and combine them into a transparent 0–100 score.

What this agent must NEVER do:
    * declare that a claim or a customer is fraudulent,
    * use personal characteristics (name, age, gender, city, religion, ...),
    * use an LLM to "judge" honesty.

Every indicator has a fixed number of points, a neutral description of what
was observed, and the evidence behind it. The score is simply the sum, capped
at 100, so anyone can recompute it by hand.

    python -m agents.fraud_agent CLM-...
"""

from __future__ import annotations

from datetime import timedelta

from sqlalchemy.orm import Session

from agents.schemas import (
    DocumentExtractionResult,
    PolicyVerificationResult,
    RiskAssessment,
    RiskIndicator,
    ValidationInput,
    ValidationResult,
)
from app.config import get_logger
from database.catalog import DIAGNOSES, is_accidental
from database.repository import ClaimRepository, PolicyRepository

log = get_logger("fraud_agent")

# code: (severity, points). Kept in one table so the scoring is auditable.
INDICATORS: dict[str, tuple[str, int]] = {
    "DUPLICATE_INVOICE": ("HIGH", 40),
    "AMOUNT_ABOVE_BILL": ("HIGH", 30),
    "DOCUMENT_INCONSISTENCY": ("MEDIUM", 15),
    "EARLY_CLAIM": ("MEDIUM", 15),
    "COST_ABOVE_TYPICAL_RANGE": ("MEDIUM", 15),
    "HIGH_SHARE_OF_SUM_INSURED": ("LOW", 10),
    "FREQUENT_CLAIMS": ("MEDIUM", 15),
    "HOLDER_MISMATCH": ("LOW", 10),
    "DATE_INCONSISTENCY": ("LOW", 10),
    "LATE_SUBMISSION": ("LOW", 5),
    "LOW_DOCUMENT_QUALITY": ("LOW", 5),
}
MEDIUM_FROM, HIGH_FROM = 25, 50
EARLY_CLAIM_DAYS = 30
FREQUENT_WINDOW_DAYS, FREQUENT_COUNT = 180, 2  # 2+ other claims on the policy within 180 days
TYPICAL_COST_MULTIPLIER = 1.25
HIGH_SHARE = 0.8
LATE_DAYS = 90
MIN_OCR_CONFIDENCE = 0.8

FEATURES_USED = [
    "invoice history", "claimed vs billed amount", "document consistency", "days since policy start",
    "cost vs typical range for the diagnosis", "share of sum insured", "claims on the same policy",
    "policy-holder match", "date consistency", "submission delay", "document readability",
]
FEATURES_NOT_USED = [
    "name", "age", "gender", "city or state", "religion or caste", "income", "nationality", "health history",
]
DISCLAIMER = ("Risk indicators highlight patterns that may merit investigation. They are not evidence of "
              "wrongdoing and do not state that any person acted dishonestly. A human investigator decides.")


def typical_cost_range(diagnosis: str | None) -> tuple[int, int] | None:
    text = (diagnosis or "").lower()
    for dx in DIAGNOSES:
        if dx.name.lower() == text or dx.name.lower().split(" - ")[0] in text:
            return dx.min_cost, dx.max_cost
    return None


def risk_level(score: int) -> str:
    return "HIGH" if score >= HIGH_FROM else "MEDIUM" if score >= MEDIUM_FROM else "LOW"


class FraudRiskAgent:
    key = "fraud_agent"
    name = "Fraud Risk Agent"
    purpose = "Flags explainable risk indicators for investigation. Never labels anyone fraudulent."

    def __init__(self, session: Session | None = None):
        self.session = session

    def run(self, facts: ValidationInput, validation: ValidationResult, policy: PolicyVerificationResult,
            extraction: DocumentExtractionResult | None = None) -> RiskAssessment:
        found: list[RiskIndicator] = []

        def add(code: str, description: str, **evidence) -> None:
            severity, points = INDICATORS[code]
            found.append(RiskIndicator(code=code, severity=severity, points=points, description=description,
                                       evidence=evidence))

        # 1. Invoice already claimed earlier
        if validation.flags.get("duplicate_invoice"):
            add("DUPLICATE_INVOICE", "The invoice number also appears on an earlier claim.",
                invoice_number=facts.value("invoice_number"), earlier_claims=", ".join(validation.duplicate_claims))

        # 2. Claimed more than billed
        amount = validation.check("AMOUNT_CONSISTENCY")
        if amount and amount.status == "FAIL":
            add("AMOUNT_ABOVE_BILL", "The claimed amount is higher than the hospital bill total.", **amount.evidence)

        # 3. Documents disagree with each other or with the form
        if validation.flags.get("field_mismatch") or (extraction and extraction.conflicts):
            fields = sorted({c.field for c in extraction.conflicts} if extraction else set())
            add("DOCUMENT_INCONSISTENCY", "Some values differ between the claim form and the documents.",
                fields=", ".join(fields) or "see validation")

        # 4. Very early claim after policy start (accidents are naturally unpredictable: excluded)
        snap = policy.policy
        adm = facts.value("admission_date")
        if snap and adm:
            days = (adm - snap.start_date).days
            if 0 <= days < EARLY_CLAIM_DAYS and not is_accidental(facts.value("diagnosis")):
                add("EARLY_CLAIM", f"Hospitalisation began {days} day(s) after the policy started.",
                    policy_start=snap.start_date.isoformat(), admission_date=adm.isoformat(), days=days)

        # 5. Cost well above the typical range for the diagnosis
        claimed = facts.submitted.claim_amount
        typical = typical_cost_range(facts.value("diagnosis"))
        if claimed and typical and claimed > typical[1] * TYPICAL_COST_MULTIPLIER:
            add("COST_ABOVE_TYPICAL_RANGE",
                f"The claimed amount is above the typical range for this diagnosis "
                f"(INR {typical[0]:,} - {typical[1]:,}).",
                claim_amount=claimed, typical_max=typical[1])

        # 6. Claim uses most of the sum insured
        if claimed and snap and snap.sum_insured and claimed >= HIGH_SHARE * snap.sum_insured:
            add("HIGH_SHARE_OF_SUM_INSURED",
                f"The claim is {claimed / snap.sum_insured:.0%} of the sum insured.",
                claim_amount=claimed, sum_insured=snap.sum_insured)

        # 7. Several recent claims on the same policy
        recent = self._recent_claims(facts, snap.policy_number if snap else None)
        if len(recent) >= FREQUENT_COUNT:
            add("FREQUENT_CLAIMS", f"{len(recent)} other claim(s) on this policy in the previous "
                                   f"{FREQUENT_WINDOW_DAYS} days.", other_claims=", ".join(recent))

        # 8. Claimant differs from the policy holder (may be legitimate on family policies)
        if policy.flags.get("holder_mismatch"):
            add("HOLDER_MISMATCH", "The claimant details differ from the policy holder on record.")

        # 9. Impossible dates (often a data-entry error — low weight)
        if validation.flags.get("invalid_dates"):
            add("DATE_INCONSISTENCY", "Hospitalisation dates are inconsistent (see validation).")

        # 10. Late submission
        delay = validation.check("SUBMISSION_DELAY")
        if delay and delay.status == "WARN":
            add("LATE_SUBMISSION", delay.message, **delay.evidence)

        # 11. Hard-to-read documents (quality, not intent)
        if extraction:
            confs = [d.ocr_confidence for d in extraction.documents if d.ocr_confidence is not None]
            if confs and min(confs) < MIN_OCR_CONFIDENCE:
                add("LOW_DOCUMENT_QUALITY", "At least one scanned document was hard to read automatically.",
                    min_ocr_confidence=round(min(confs), 3))

        score = min(100, sum(i.points for i in found))
        level = risk_level(score)
        refer = level == "HIGH" or any(i.severity == "HIGH" for i in found)
        if not found:
            summary = "No risk indicators were observed."
        else:
            summary = (f"{len(found)} risk indicator(s), score {score}/100 ({level}): "
                       + "; ".join(i.code.replace('_', ' ').lower() for i in found) + ". "
                       + ("Referral to the investigation team is suggested." if refer
                          else "Standard review is sufficient."))
        return RiskAssessment(risk_score=score, risk_level=level, indicators=found, refer_for_investigation=refer,
                              summary=summary, features_used=FEATURES_USED, features_not_used=FEATURES_NOT_USED,
                              disclaimer=DISCLAIMER)

    def _recent_claims(self, facts: ValidationInput, policy_number: str | None) -> list[str]:
        if self.session is None or not policy_number:
            return []
        db_policy = PolicyRepository(self.session).get_by_number(policy_number)
        if db_policy is None:
            return []
        ref = facts.submitted_on
        out = []
        for c in ClaimRepository(self.session).for_policy(db_policy.id, exclude_claim_id=facts.claim_id):
            when = c.submitted_at.date()
            if ref is None or (ref - timedelta(days=FREQUENT_WINDOW_DAYS) <= when <= ref):
                out.append(c.claim_number)
        return sorted(out)

    @staticmethod
    def risk_flag_rows(assessment: RiskAssessment) -> list[dict]:
        """Rows for the risk_flags table."""
        return [{"code": i.code, "severity": i.severity, "description": i.description, "evidence": i.evidence}
                for i in assessment.indicators]
