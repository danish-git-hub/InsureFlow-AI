"""
Agent 6 — Decision Support Agent.

Purpose:
    Combine the five previous agents into ONE routing recommendation for a
    human claims officer, with the reasons, the actions needed and a checklist.

The only possible outputs (never an approval or a rejection):

    POLICY_NOT_FOUND                 the policy cannot be identified
    ADDITIONAL_INFORMATION_REQUIRED  documents, fields or valid dates are missing
    REVIEW_REQUIRED                  a rule failed, wording may exclude, or risk indicators exist
    ELIGIBLE_FOR_HUMAN_REVIEW        all automated checks passed; a human still decides

The precedence above is fixed and deterministic, so the recommendation can be
audited. `human_decision_required` is always True. Gemini (if configured) may
only rewrite the summary in plain English; an answer that uses approval /
rejection / accusation language is discarded.

    python -m agents.decision_agent CLM-...      # runs Agents 1-6 on a stored claim
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from agents.llm import invoke_structured, llm_available
from agents.schemas import (
    DecisionReason,
    DecisionSupport,
    DocumentExtractionResult,
    PolicyRAGResult,
    PolicyVerificationResult,
    RiskAssessment,
    ValidationResult,
)
from app.config import get_logger, settings

log = get_logger("decision_agent")

DISCLAIMER = ("Decision support only. This recommendation routes the claim to the right human review; it is not an "
              "approval, rejection or payment decision. An authorised claims officer makes the final decision.")
FORBIDDEN_LANGUAGE = re.compile(
    r"\b(approv\w*|reject\w*|den(?:y|ied|ial)|fraudulent|fraudster|dishonest\w*|guilty|liar|cheat\w*)\b",
    re.IGNORECASE,
)
BASE_CHECKLIST = [
    "Confirm the claimant is the insured person or a covered member of the policy.",
    "Confirm the treatment is medically necessary and consistent with the diagnosis.",
    "Confirm the admissible amount against the bill, sub-limits and co-payment.",
    "Record your decision and reasons in the Human Review Center.",
]


class _Summary(BaseModel):
    summary: str = Field(description="4-6 plain sentences for a claims officer, using only the facts provided")


def _label(code: str) -> str:
    return code.replace("_", " ").lower()


class DecisionSupportAgent:
    key = "decision_agent"
    name = "Decision Support Agent"
    purpose = "Combines all agent outputs into a routing recommendation for a human claims officer."

    def __init__(self, use_llm: bool | None = None, llm_model=None):
        self.llm_model = llm_model
        self.use_llm = (llm_model is not None or llm_available()) if use_llm is None else use_llm

    def run(self, policy: PolicyVerificationResult, rag: PolicyRAGResult, validation: ValidationResult,
            risk: RiskAssessment, extraction: DocumentExtractionResult | None = None) -> DecisionSupport:
        reasons: list[DecisionReason] = []
        actions: list[str] = []

        # ---- 1. policy not found ------------------------------------------------
        not_found = policy.status == "POLICY_NOT_FOUND"
        if not_found:
            reasons.append(DecisionReason(source="policy_agent", code="POLICY_NOT_FOUND",
                                          message=policy.summary or "No matching policy was found.", blocking=True))
            actions.append("Confirm the policy number with the claimant.")
            actions += [f"Check whether the claim belongs to {s}." for s in policy.suggestions]

        # ---- 2. information missing --------------------------------------------
        info = []
        if validation.flags.get("missing_documents"):
            info.append(DecisionReason(source="validation_agent", code="MISSING_DOCUMENTS",
                                       message=validation.check("REQUIRED_DOCUMENTS").message, blocking=True))
            actions += [f"Request the {_label(d)} from the claimant." for d in validation.missing_documents]
        if validation.flags.get("missing_fields"):
            info.append(DecisionReason(source="validation_agent", code="MISSING_FIELDS",
                                       message=validation.check("REQUIRED_FIELDS").message, blocking=True))
            actions.append("Obtain the missing claim details: " + ", ".join(_label(f) for f in validation.missing_fields) + ".")
        if validation.flags.get("invalid_dates"):
            for code in ("DATE_ORDER", "FUTURE_DATES"):
                c = validation.check(code)
                if c and c.status == "FAIL":
                    info.append(DecisionReason(source="validation_agent", code=code, message=c.message, blocking=True))
            actions.append("Ask the claimant or hospital to confirm the admission and discharge dates.")
        if policy.status == "INSUFFICIENT_DATA":
            info.append(DecisionReason(source="policy_agent", code="INSUFFICIENT_DATA",
                                       message="Policy rules could not be evaluated: " + "; ".join(policy.issues or [policy.summary]),
                                       blocking=True))
            actions.append("Collect the details needed to verify the policy (dates, amount).")
        reasons += info

        # ---- 3. needs a closer human review ------------------------------------
        review = []
        if policy.status in ("FAILED", "VERIFIED_WITH_WARNINGS"):
            for c in policy.checks:
                if c.status in ("FAIL", "WARN"):
                    review.append(DecisionReason(source="policy_agent", code=c.code, message=c.message,
                                                 blocking=c.status == "FAIL"))
                    if c.status == "FAIL":
                        actions.append(f"Review the policy rule '{c.name}': {c.message}")
        if validation.flags.get("duplicate_invoice"):
            review.append(DecisionReason(source="validation_agent", code="DUPLICATE_INVOICE",
                                         message=validation.check("DUPLICATE_INVOICE").message, blocking=True))
            actions.append("Compare this claim with " + ", ".join(validation.duplicate_claims)
                           + " before any assessment.")
        if validation.flags.get("amount_mismatch"):
            review.append(DecisionReason(source="validation_agent", code="AMOUNT_MISMATCH",
                                         message=validation.check("AMOUNT_CONSISTENCY").message, blocking=True))
            actions.append("Ask the claimant to explain the difference between the claimed and billed amounts.")
        for w in validation.checks:
            if w.status == "WARN":
                review.append(DecisionReason(source="validation_agent", code=w.code, message=w.message, blocking=True))
        for clause in (c for c in rag.clauses if c.topic == "exclusions"):
            review.append(DecisionReason(source="rag_agent", code="POSSIBLE_EXCLUSION", blocking=True,
                                         message=f"Exclusion wording mentions the diagnosis: {clause.citation}."))
            actions.append(f"Read {clause.plan_code} {clause.clause_code} ({clause.title}) and decide whether it applies.")
        if rag.status == "INSUFFICIENT_EVIDENCE":
            review.append(DecisionReason(source="rag_agent", code="INSUFFICIENT_EVIDENCE", blocking=True,
                                         message=" ".join(rag.uncertainty_reasons)))
            actions.append("Check the policy wording manually; automated retrieval found insufficient evidence.")
        if risk.risk_level in ("MEDIUM", "HIGH"):
            review.append(DecisionReason(source="fraud_agent", code=f"RISK_{risk.risk_level}", blocking=True,
                                         message=risk.summary))
        if risk.refer_for_investigation:
            actions.append("Consider referring the claim to the investigation team (indicators only, no finding).")
        # Low-risk indicators are shown but do not change the route.
        if risk.risk_level == "LOW":
            for i in risk.indicators:
                review.append(DecisionReason(source="fraud_agent", code=i.code, message=i.description, blocking=False))
        reasons += review

        # ---- recommendation (fixed precedence) ---------------------------------
        if not_found:
            rec = "POLICY_NOT_FOUND"
        elif info:
            rec = "ADDITIONAL_INFORMATION_REQUIRED"
        elif any(r.blocking for r in review):
            rec = "REVIEW_REQUIRED"
        else:
            rec = "ELIGIBLE_FOR_HUMAN_REVIEW"
            reasons.append(DecisionReason(source="decision_agent", code="ALL_CHECKS_PASSED",
                                          message="Policy, wording, validation and risk checks raised no blocking issue."))

        confidence, notes = self._confidence(policy, rag, risk, extraction)
        checklist = list(BASE_CHECKLIST)
        if rag.clauses:
            checklist.insert(2, "Read the cited policy clauses (Claim Details → Policy clauses).")
        result = DecisionSupport(
            recommendation=rec, confidence=confidence, confidence_notes=notes, reasons=reasons,
            required_actions=list(dict.fromkeys(actions)), reviewer_checklist=checklist,
            indicative_admissible_amount=policy.coverage.indicative_admissible_amount if policy.coverage else None,
            disclaimer=DISCLAIMER,
        )
        result.summary = self._template_summary(result, risk)
        if self.use_llm:
            self._llm_summary(result)
        return result

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _confidence(policy, rag, risk, extraction) -> tuple[float, list[str]]:
        """How complete the evidence behind the recommendation is (not a probability of payment)."""
        score, notes = 0.95, []
        if extraction is None:
            score -= 0.15
            notes.append("Documents were not read in this run; stored values were used.")
        else:
            if extraction.ocr_used:
                score -= 0.05
                notes.append("Some documents were read with OCR.")
            if extraction.conflicts:
                score -= 0.05
                notes.append("Documents disagree on some values.")
            if extraction.missing_fields:
                score -= 0.10
                notes.append("Some fields could not be extracted.")
        if rag.status != "EVIDENCE_FOUND":
            score -= 0.15
            notes.append("Policy wording evidence is incomplete.")
        if policy.status == "INSUFFICIENT_DATA":
            score -= 0.15
            notes.append("Some policy rules could not be evaluated.")
        return round(max(0.3, score), 2), notes or ["All evidence sources were available."]

    @staticmethod
    def _template_summary(r: DecisionSupport, risk: RiskAssessment) -> str:
        blocking = [x for x in r.reasons if x.blocking]
        head = {
            "ELIGIBLE_FOR_HUMAN_REVIEW": "All automated checks passed and the claim is ready for a claims officer.",
            "ADDITIONAL_INFORMATION_REQUIRED": "The claim cannot be assessed until more information is provided.",
            "POLICY_NOT_FOUND": "No matching policy was found for this claim.",
            "REVIEW_REQUIRED": "The claim needs a closer review by a claims officer.",
        }[r.recommendation]
        parts = [head]
        if blocking:
            parts.append("Main reason(s): " + " ".join(x.message for x in blocking[:3]))
        parts.append(f"Risk score {risk.risk_score}/100 ({risk.risk_level}).")
        if r.indicative_admissible_amount is not None:
            parts.append(f"Indicative admissible amount from policy rules: INR {r.indicative_admissible_amount:,.2f}.")
        parts.append("Final decision rests with an authorised claims officer.")
        return " ".join(parts)

    def _llm_summary(self, r: DecisionSupport) -> None:
        facts = "\n".join(f"- [{x.source}] {x.message}" for x in r.reasons)
        prompt = (
            "Write a short, neutral summary for an insurance claims officer. Use ONLY the facts below. "
            "Do not say the claim is approved, rejected, denied or fraudulent, and do not judge anyone's honesty. "
            f"State that the routing recommendation is {r.recommendation} and that the officer makes the final "
            f"decision.\n\nFacts:\n{facts}\n\nRequired actions:\n" + "\n".join(f"- {a}" for a in r.required_actions)
        )
        answer, call = invoke_structured(prompt, _Summary, model=self.llm_model)
        if answer is None:
            r.warnings.append(f"Gemini summary unavailable ({call.error}); template summary used.")
            return
        r.used_llm, r.llm_model = True, call.model or settings.gemini_model
        if FORBIDDEN_LANGUAGE.search(answer.summary):
            r.warnings.append("Gemini summary discarded: it used decision or accusation language; "
                              "template summary used.")
            return
        r.summary, r.summary_source = answer.summary.strip(), "gemini"


def main() -> None:
    import argparse

    from agents.document_agent import DocumentExtractionAgent, DocumentInput
    from agents.fraud_agent import FraudRiskAgent
    from agents.policy_agent import PolicyVerificationAgent, build_input
    from agents.rag_agent import PolicyRAGAgent, build_rag_input
    from agents.validation_agent import ValidationAgent, build_validation_input
    from app.config import PROJECT_ROOT
    from database.connection import SessionLocal
    from database.repository import ClaimRepository

    parser = argparse.ArgumentParser(description="Run all six agents on a stored claim (no database writes)")
    parser.add_argument("claim_number")
    args = parser.parse_args()
    with SessionLocal() as session:
        claim = ClaimRepository(session).get_by_number(args.claim_number)
        docs = [DocumentInput(PROJECT_ROOT / d.file_path, d.document_type, d.original_filename) for d in claim.documents]
        extraction = DocumentExtractionAgent().run(docs) if docs else None
        facts = build_input(claim, extraction)
        policy = PolicyVerificationAgent(session).run(facts)
        rag = PolicyRAGAgent().run(build_rag_input(policy, facts))
        vin = build_validation_input(claim, extraction)
        validation = ValidationAgent(session).run(vin)
        risk = FraudRiskAgent(session).run(vin, validation, policy, extraction)
        decision = DecisionSupportAgent().run(policy, rag, validation, risk, extraction)
    print(f"{claim.claim_number}  scenario={claim.scenario}")
    print(f"  policy={policy.status}  rag={rag.status}  validation={validation.status}  "
          f"risk={risk.risk_score} ({risk.risk_level})")
    print(f"  RECOMMENDATION: {decision.recommendation}  confidence={decision.confidence}")
    for a in decision.required_actions:
        print(f"   - {a}")
        print("\nSUMMARY:")
    print(decision.summary)

    print("\nGEMINI DIAGNOSTICS:")
    print("Used LLM:", decision.used_llm)
    print("LLM Model:", decision.llm_model)
    print("Summary Source:", decision.summary_source)

    print("\nWARNINGS:")
    for warning in decision.warnings:
        print("-", warning)

    if not decision.warnings:
        print("No warnings.")
if __name__ == "__main__":
    main()