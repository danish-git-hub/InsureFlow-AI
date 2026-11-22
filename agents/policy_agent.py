"""
Agent 2 — Policy Verification Agent
===================================

Responsibility: confirm that the policy quoted on a claim exists and was in
force for this hospitalisation, using DETERMINISTIC business rules.

Why no LLM here?
    Policy validity, dates, waiting periods and sum-insured limits are exact,
    auditable checks. They must give the same answer every time and be
    explainable line by line, so they are plain Python rules. (Agent 3 — RAG —
    is where policy *wording* is retrieved and explained.)

Rules (each produces a RuleCheck with PASS / WARN / FAIL / NOT_EVALUATED):
    POLICY_FOUND            policy number exists in the policy register
    POLICY_STATUS           not lapsed / cancelled
    HOLDER_MATCH            claimant matches the policy holder (name / customer ID)
    INSURANCE_TYPE          submitted type matches the policy product
    COVERAGE_PERIOD         admission date between policy start and end date
    WAITING_PERIOD          admission after the initial waiting period (accidents exempt)
    HOSPITALIZATION_COVER   product covers in-patient hospitalisation
    SUM_INSURED             claim within sum insured and remaining balance
    EXCLUSIONS              diagnosis does not match a listed exclusion
    DISCHARGE_WITHIN_TERM   informational: discharge after policy end

The agent never approves or rejects; it reports verification evidence.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any

from sqlalchemy.orm import Session

from agents.schemas import (
    CoverageSummary,
    DocumentExtractionResult,
    PolicySnapshot,
    PolicyVerificationInput,
    PolicyVerificationResult,
    RuleCheck,
)
from database.catalog import ALL_EXCLUSIONS, PLANS, Exclusion, is_accidental
from database.models import Claim, ClaimStatus, InsuranceType, Policy, PolicyStatus
from database.repository import ClaimRepository, PolicyRepository

CRITICAL_CHECKS = ("COVERAGE_PERIOD", "WAITING_PERIOD", "SUM_INSURED")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _tokens(name: str | None) -> list[str]:
    return [t for t in re.split(r"[^a-z]+", (name or "").lower()) if t]


def compare_names(submitted: str | None, holder: str | None) -> str:
    """EXACT, PARTIAL (same surname + first initial, e.g. 'R. Sharma') or MISMATCH."""
    a, b = _tokens(submitted), _tokens(holder)
    if not a or not b:
        return "MISMATCH"
    if sorted(a) == sorted(b):
        return "EXACT"
    if a[-1] == b[-1] and a[0][0] == b[0][0]:
        return "PARTIAL"
    return "MISMATCH"


def matched_exclusions(diagnosis: str | None, plan_code: str, policy_exclusions: list[str],
                       insurance_type: str) -> list[Exclusion]:
    """Return exclusions whose trigger keywords appear in the diagnosis."""
    if not diagnosis:
        return []
    plan = PLANS.get(plan_code)
    exclusions = list(plan.exclusions) if plan else [ALL_EXCLUSIONS[t] for t in policy_exclusions if t in ALL_EXCLUSIONS]
    text = diagnosis.lower()
    hits = []
    for exc in exclusions:
        if exc.code == "EXC-PA":
            # Personal Accident: anything that is NOT an accident is outside cover.
            if insurance_type == InsuranceType.ACCIDENT.value and not is_accidental(diagnosis):
                hits.append(exc)
        elif any(k in text for k in exc.keywords):
            hits.append(exc)
    return hits


def snapshot(policy: Policy) -> PolicySnapshot:
    plan = PLANS.get(policy.plan_code)
    return PolicySnapshot(
        policy_number=policy.policy_number, holder_name=policy.customer.full_name,
        customer_code=policy.customer.customer_code, insurance_type=policy.insurance_type,
        plan_name=policy.plan_name, plan_code=policy.plan_code, sum_insured=float(policy.sum_insured),
        start_date=policy.start_date, end_date=policy.end_date, waiting_period_days=policy.waiting_period_days,
        status=policy.status, hospitalization_covered=policy.hospitalization_covered,
        exclusions=list(policy.exclusions or []), copay_percent=plan.copay_percent if plan else 0.0,
    )


def build_input(claim: Claim, extraction: DocumentExtractionResult | None = None) -> PolicyVerificationInput:
    """Combine extracted document values with the submitted claim form.

    Document evidence is preferred; the submitted form fills any gap. The
    `sources` map shows which one was used for every field (explainability).
    """
    extracted = extraction.fields if extraction else None
    submitted: dict[str, Any] = {
        "policy_number": claim.policy_number_submitted,
        "customer_name": claim.customer_name_submitted,
        "customer_id": claim.customer.customer_code if claim.customer else None,
        "claim_amount": claim.claim_amount,
        "admission_date": claim.admission_date,
        "discharge_date": claim.discharge_date,
        "diagnosis": claim.diagnosis,
    }
    values: dict[str, Any] = {"insurance_type": claim.insurance_type, "claim_number": claim.claim_number}
    sources: dict[str, str] = {"insurance_type": "claim_form_submission"}
    for field, form_value in submitted.items():
        doc_value = getattr(extracted, field, None) if extracted else None
        if doc_value not in (None, ""):
            values[field], sources[field] = doc_value, "documents"
        elif form_value not in (None, ""):
            values[field], sources[field] = form_value, "claim_form_submission"
    return PolicyVerificationInput(**values, sources=sources)


# --------------------------------------------------------------------------- #
# The agent
# --------------------------------------------------------------------------- #
class PolicyVerificationAgent:
    key = "policy_agent"
    name = "Policy Verification Agent"
    purpose = "Verifies policy existence, status, validity, waiting period, sum insured and exclusions."

    def __init__(self, session: Session):
        self.policies = PolicyRepository(session)
        self.claims = ClaimRepository(session)

    def run(self, data: PolicyVerificationInput) -> PolicyVerificationResult:
        checks: list[RuleCheck] = []
        suggestions: list[str] = []

        # ---- Rule 1: policy exists --------------------------------------------
        policy = self.policies.get_by_number(data.policy_number) if data.policy_number else None
        if policy is None:
            if data.policy_number:
                msg = f"Policy number {data.policy_number} does not exist in the policy register."
            else:
                msg = "No policy number was found on the claim or its documents."
            checks.append(RuleCheck(code="POLICY_FOUND", name="Policy exists", status="FAIL", message=msg,
                                    evidence={"policy_number": data.policy_number}))
            suggestions += self._suggest_policies(data)
            return self._finish(checks, None, None, suggestions, data)

        snap = snapshot(policy)
        checks.append(RuleCheck(code="POLICY_FOUND", name="Policy exists", status="PASS",
                                message=f"Policy {snap.policy_number} ({snap.plan_name}) found.",
                                evidence={"policy_number": snap.policy_number, "plan": snap.plan_name}))
        adm, dis = data.admission_date, data.discharge_date

        # ---- Rule 2: status ---------------------------------------------------
        if snap.status in (PolicyStatus.LAPSED, PolicyStatus.CANCELLED):
            checks.append(RuleCheck(code="POLICY_STATUS", name="Policy status", status="FAIL",
                                    message=f"Policy is {snap.status}; it does not provide cover.",
                                    evidence={"status": snap.status}))
        elif snap.status == PolicyStatus.EXPIRED:
            in_term = adm is not None and snap.start_date <= adm <= snap.end_date
            checks.append(RuleCheck(
                code="POLICY_STATUS", name="Policy status", status="PASS" if in_term else "FAIL",
                message=("Policy has since expired but was in force on the admission date." if in_term
                         else f"Policy EXPIRED on {snap.end_date:%d %b %Y}."),
                evidence={"status": snap.status, "end_date": snap.end_date.isoformat()}))
        else:
            checks.append(RuleCheck(code="POLICY_STATUS", name="Policy status", status="PASS",
                                    message=f"Policy status is {snap.status}.", evidence={"status": snap.status}))

        # ---- Rule 3: holder ---------------------------------------------------
        name_result = compare_names(data.customer_name, snap.holder_name)
        id_ok = data.customer_id is None or data.customer_id.upper() == snap.customer_code
        if name_result == "EXACT" and id_ok:
            status, msg = "PASS", "Claimant matches the policy holder."
        elif name_result == "MISMATCH" or not id_ok:
            status = "WARN"
            msg = (f"Claimant '{data.customer_name}' / ID {data.customer_id} does not match policy holder "
                   f"'{snap.holder_name}' / {snap.customer_code}. Verify the insured person (e.g. floater member).")
        else:
            status, msg = "WARN", f"Claimant name '{data.customer_name}' only partially matches '{snap.holder_name}'."
        checks.append(RuleCheck(code="HOLDER_MATCH", name="Policy holder match", status=status, message=msg,
                                evidence={"claimant": data.customer_name, "holder": snap.holder_name,
                                          "customer_id": data.customer_id, "holder_id": snap.customer_code}))

        # ---- Rule 4: insurance type --------------------------------------------
        if data.insurance_type and data.insurance_type != snap.insurance_type:
            checks.append(RuleCheck(code="INSURANCE_TYPE", name="Insurance type", status="WARN",
                                    message=f"Claim filed as '{data.insurance_type}' but policy is "
                                            f"'{snap.insurance_type}'.",
                                    evidence={"submitted": data.insurance_type, "policy": snap.insurance_type}))
        else:
            checks.append(RuleCheck(code="INSURANCE_TYPE", name="Insurance type", status="PASS",
                                    message=f"Insurance type is {snap.insurance_type}."))

        # ---- Rule 5: coverage period -----------------------------------------
        if adm is None:
            checks.append(RuleCheck(code="COVERAGE_PERIOD", name="Policy in force on admission date",
                                    status="NOT_EVALUATED", message="Admission date unknown."))
        elif snap.start_date <= adm <= snap.end_date:
            checks.append(RuleCheck(code="COVERAGE_PERIOD", name="Policy in force on admission date", status="PASS",
                                    message=f"Admission {adm:%d %b %Y} is within the policy term "
                                            f"{snap.start_date:%d %b %Y} – {snap.end_date:%d %b %Y}.",
                                    evidence={"admission_date": adm.isoformat()}))
        else:
            when = "before the policy started" if adm < snap.start_date else "after the policy ended"
            checks.append(RuleCheck(code="COVERAGE_PERIOD", name="Policy in force on admission date", status="FAIL",
                                    message=f"Admission {adm:%d %b %Y} is {when} "
                                            f"({snap.start_date:%d %b %Y} – {snap.end_date:%d %b %Y}).",
                                    evidence={"admission_date": adm.isoformat(), "start_date": snap.start_date.isoformat(),
                                              "end_date": snap.end_date.isoformat()}))

        # ---- Rule 6: waiting period --------------------------------------------
        waiting_end = snap.start_date + timedelta(days=snap.waiting_period_days)
        if snap.waiting_period_days == 0:
            checks.append(RuleCheck(code="WAITING_PERIOD", name="Initial waiting period", status="NOT_APPLICABLE",
                                    message="This product has no initial waiting period."))
        elif is_accidental(data.diagnosis):
            checks.append(RuleCheck(code="WAITING_PERIOD", name="Initial waiting period", status="NOT_APPLICABLE",
                                    message="Accidental injury: initial waiting period does not apply."))
        elif adm is None:
            checks.append(RuleCheck(code="WAITING_PERIOD", name="Initial waiting period", status="NOT_EVALUATED",
                                    message="Admission date unknown."))
        elif adm < waiting_end:
            days_in = (adm - snap.start_date).days
            checks.append(RuleCheck(code="WAITING_PERIOD", name="Initial waiting period", status="FAIL",
                                    message=f"Admitted {days_in} days after policy start; the "
                                            f"{snap.waiting_period_days}-day waiting period ends {waiting_end:%d %b %Y}.",
                                    evidence={"days_since_start": days_in, "waiting_days": snap.waiting_period_days}))
        else:
            checks.append(RuleCheck(code="WAITING_PERIOD", name="Initial waiting period", status="PASS",
                                    message=f"{snap.waiting_period_days}-day waiting period completed on "
                                            f"{waiting_end:%d %b %Y}."))

        # ---- Rule 7: hospitalisation cover -----------------------------------
        checks.append(RuleCheck(
            code="HOSPITALIZATION_COVER", name="Hospitalisation covered",
            status="PASS" if snap.hospitalization_covered else "FAIL",
            message="Product covers in-patient hospitalisation." if snap.hospitalization_covered
            else "Product does not cover hospitalisation expenses."))

        # ---- Rule 8: sum insured -------------------------------------------------
        coverage = self._coverage(policy, snap, data)
        if data.claim_amount is None:
            checks.append(RuleCheck(code="SUM_INSURED", name="Within sum insured", status="NOT_EVALUATED",
                                    message="Claim amount unknown."))
        elif data.claim_amount > snap.sum_insured:
            checks.append(RuleCheck(code="SUM_INSURED", name="Within sum insured", status="FAIL",
                                    message=f"Claim INR {data.claim_amount:,.0f} exceeds the sum insured "
                                            f"INR {snap.sum_insured:,.0f}.",
                                    evidence={"claim_amount": data.claim_amount, "sum_insured": snap.sum_insured}))
        elif data.claim_amount > coverage.remaining_sum_insured:
            checks.append(RuleCheck(code="SUM_INSURED", name="Within sum insured", status="WARN",
                                    message=f"Claim INR {data.claim_amount:,.0f} exceeds the remaining balance "
                                            f"INR {coverage.remaining_sum_insured:,.0f} after other claims this term.",
                                    evidence={"remaining": coverage.remaining_sum_insured}))
        else:
            checks.append(RuleCheck(code="SUM_INSURED", name="Within sum insured", status="PASS",
                                    message=f"Claim INR {data.claim_amount:,.0f} is within the remaining sum insured "
                                            f"INR {coverage.remaining_sum_insured:,.0f}."))

        # ---- Rule 9: exclusions ----------------------------------------------------
        hits = matched_exclusions(data.diagnosis, snap.plan_code, snap.exclusions, snap.insurance_type)
        if data.diagnosis is None:
            checks.append(RuleCheck(code="EXCLUSIONS", name="Policy exclusions", status="NOT_EVALUATED",
                                    message="Diagnosis unknown."))
        elif hits:
            checks.append(RuleCheck(code="EXCLUSIONS", name="Policy exclusions", status="FAIL",
                                    message="Diagnosis matches exclusion(s): "
                                            + "; ".join(f"{h.code} {h.title}" for h in hits)
                                            + ". Confirm against the policy wording.",
                                    evidence={"diagnosis": data.diagnosis, "exclusion_codes": ", ".join(h.code for h in hits)}))
        else:
            checks.append(RuleCheck(code="EXCLUSIONS", name="Policy exclusions", status="PASS",
                                    message="No listed exclusion matches the diagnosis."))

        # ---- Rule 10: discharge after expiry (informational) -----------------------
        if adm and dis and snap.start_date <= adm <= snap.end_date < dis:
            checks.append(RuleCheck(code="DISCHARGE_WITHIN_TERM", name="Discharge within term", status="WARN",
                                    message="Discharge falls after policy end; admission date governs cover."))

        return self._finish(checks, snap, coverage, suggestions, data)

    # ------------------------------------------------------------------------ #
    def _coverage(self, policy: Policy, snap: PolicySnapshot, data: PolicyVerificationInput) -> CoverageSummary:
        others = [c for c in self.claims.for_policy(policy.id)
                  if c.claim_number != data.claim_number and c.claim_status != ClaimStatus.HUMAN_REJECTED]
        used = float(sum(c.claim_amount or 0 for c in others))
        remaining = max(snap.sum_insured - used, 0.0)
        admissible = None
        if data.claim_amount is not None:
            admissible = round(min(data.claim_amount, remaining) * (1 - snap.copay_percent / 100), 2)
        return CoverageSummary(sum_insured=snap.sum_insured, other_claims_on_policy=used,
                               remaining_sum_insured=remaining, claim_amount=data.claim_amount,
                               copay_percent=snap.copay_percent, indicative_admissible_amount=admissible)

    def _suggest_policies(self, data: PolicyVerificationInput) -> list[str]:
        """If we know the customer, list their real policies as a hint to the reviewer."""
        if not data.customer_id:
            return []
        from database.repository import CustomerRepository

        customer = CustomerRepository(self.policies.session).get_by_code(data.customer_id)
        if not customer:
            return []
        return [f"Customer {customer.customer_code} holds policy {p.policy_number} ({p.plan_name}, {p.status})."
                for p in self.policies.for_customer(customer.id)]

    @staticmethod
    def _finish(checks, snap, coverage, suggestions, data) -> PolicyVerificationResult:
        by_code = {c.code: c for c in checks}

        def failed(code: str) -> bool:
            return by_code.get(code) is not None and by_code[code].status == "FAIL"

        flags = {
            "policy_not_found": failed("POLICY_FOUND"),
            "policy_inactive": failed("POLICY_STATUS") or failed("COVERAGE_PERIOD"),
            "waiting_period_incomplete": failed("WAITING_PERIOD"),
            "exceeds_sum_insured": failed("SUM_INSURED"),
            "possible_exclusion": failed("EXCLUSIONS"),
            "holder_mismatch": by_code.get("HOLDER_MATCH") is not None and by_code["HOLDER_MATCH"].status == "WARN",
            "insufficient_data": any(by_code.get(c) is not None and by_code[c].status == "NOT_EVALUATED"
                                     for c in CRITICAL_CHECKS),
        }
        issues = [c.message for c in checks if c.status in ("FAIL", "WARN")]

        if flags["policy_not_found"]:
            status = "POLICY_NOT_FOUND"
        elif any(c.status == "FAIL" for c in checks):
            status = "FAILED"
        elif flags["insufficient_data"]:
            status = "INSUFFICIENT_DATA"
        elif any(c.status == "WARN" for c in checks):
            status = "VERIFIED_WITH_WARNINGS"
        else:
            status = "VERIFIED"

        passed = sum(c.status == "PASS" for c in checks)
        summary = (f"{status.replace('_', ' ').title()}: {passed}/{len(checks)} policy rules passed."
                   + (f" Issues: {' '.join(issues)}" if issues else ""))
        return PolicyVerificationResult(status=status, policy=snap, checks=checks, issues=issues, flags=flags,
                                        coverage=coverage, suggestions=suggestions, summary=summary)


# --------------------------------------------------------------------------- #
# CLI demo:  python -m agents.policy_agent CLM-202611-0025
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover - manual demo
    import sys

    from database.connection import session_scope

    number = sys.argv[1] if len(sys.argv) > 1 else None
    with session_scope() as s:
        repo = ClaimRepository(s)
        claim = repo.get_by_number(number) if number else repo.list(search="Rahul")[0]
        result = PolicyVerificationAgent(s).run(build_input(claim))
        print(f"{claim.claim_number}  scenario={claim.scenario}  ->  {result.status}")
        for c in result.checks:
            print(f"  [{c.status:<14}] {c.name}: {c.message}")
        if result.coverage:
            print(f"  Indicative admissible: INR {result.coverage.indicative_admissible_amount:,.2f}")
        for s_ in result.suggestions:
            print("  hint:", s_)
