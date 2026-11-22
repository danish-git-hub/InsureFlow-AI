"""
Agent 4 — Validation Agent.

Purpose:
    Check that the claim is complete and internally consistent *before* anyone
    spends time assessing it:

        * all required documents were provided,
        * all required fields could be read,
        * dates are possible (discharge after admission, nothing in the future),
        * the claimed amount is supported by the hospital bill,
        * the bill's line items add up,
        * the claim form agrees with the documents,
        * the invoice was not already used by an earlier claim.

Every check is a deterministic `RuleCheck` (no LLM): the same claim always
gives the same result, and each result carries its evidence.

    python -m agents.validation_agent CLM-...
"""

from __future__ import annotations

from datetime import date

from sqlalchemy.orm import Session

from agents.policy_agent import compare_names
from agents.schemas import (
    ClaimFields,
    DocumentExtractionResult,
    RuleCheck,
    ValidationInput,
    ValidationResult,
)
from app.config import get_logger
from database.catalog import REQUIRED_DOCUMENTS
from database.models import Claim
from database.repository import ClaimRepository

log = get_logger("validation_agent")

AMOUNT_TOLERANCE = 0.02  # claimed may exceed billed by up to 2% (rounding) before it is a mismatch
MAX_STAY_DAYS = 60
LATE_SUBMISSION_DAYS = 90
REQUIRED_SUBMITTED_FIELDS = ("customer_name", "policy_number", "claim_amount", "admission_date", "discharge_date",
                             "diagnosis", "hospital_name")
COMPARED_FIELDS = ("policy_number", "customer_name", "admission_date", "discharge_date", "invoice_number",
                   "claim_amount")


def build_validation_input(claim: Claim, extraction: DocumentExtractionResult | None = None,
                           document_types: list[str] | None = None) -> ValidationInput:
    """Collect the submitted form values, extracted values and document types for one claim."""
    submitted = ClaimFields(
        customer_name=claim.customer_name_submitted,
        customer_id=claim.customer.customer_code if claim.customer else None,
        policy_number=claim.policy_number_submitted,
        claim_amount=claim.claim_amount,
        billed_amount=None,  # the bill amount must come from the hospital bill, not the claimant
        hospital_name=claim.hospital_name,
        invoice_number=claim.invoice_number,
        admission_date=claim.admission_date,
        discharge_date=claim.discharge_date,
        diagnosis=claim.diagnosis,
    )
    if document_types is None:
        document_types = (extraction.document_types_found if extraction
                          else sorted({d.document_type for d in claim.documents}))
    if extraction is None and claim.billed_amount is not None and "HOSPITAL_BILL" in document_types:
        # Without reading the PDFs, the stored bill amount stands in for the extracted one.
        extracted = ClaimFields(billed_amount=claim.billed_amount, invoice_number=claim.invoice_number)
    else:
        extracted = extraction.fields if extraction else None
    return ValidationInput(
        claim_number=claim.claim_number,
        claim_id=claim.id,
        submitted=submitted,
        extracted=extracted,
        documents_read=extraction is not None,
        document_types=list(document_types),
        conflicts=list(extraction.conflicts) if extraction else [],
        extraction_missing_fields=list(extraction.missing_fields) if extraction else [],
        bill_items_total=round(sum(i.amount for i in extraction.bill_items), 2)
        if extraction and extraction.bill_items else None,
        submitted_on=claim.submitted_at.date() if claim.submitted_at else None,
    )


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"INR {v:,.2f}"
    return str(v)


class ValidationAgent:
    key = "validation_agent"
    name = "Validation Agent"
    purpose = "Checks documents, required fields, dates, amounts and duplicate invoices."

    def __init__(self, session: Session | None = None, today: date | None = None):
        self.session = session
        self.today = today or date.today()

    def run(self, data: ValidationInput) -> ValidationResult:
        checks = [
            self._documents(data),
            self._fields(data),
            self._date_order(data),
            self._future_dates(data),
            self._length_of_stay(data),
            self._amounts(data),
            self._bill_items(data),
            self._form_vs_documents(data),
            self._duplicate_invoice(data),
            self._submission_delay(data),
        ]
        return self._finish(checks, data)

    # ------------------------------------------------------------------ rules
    @staticmethod
    def _documents(d: ValidationInput) -> RuleCheck:
        missing = [t for t in REQUIRED_DOCUMENTS if t not in d.document_types]
        if missing:
            return RuleCheck(code="REQUIRED_DOCUMENTS", name="Required documents", status="FAIL",
                             message="Missing required document(s): " + ", ".join(m.replace("_", " ").title()
                                                                                   for m in missing) + ".",
                             evidence={"missing": ", ".join(missing), "provided": ", ".join(d.document_types)})
        return RuleCheck(code="REQUIRED_DOCUMENTS", name="Required documents", status="PASS",
                         message="Claim form, hospital bill and discharge summary are all provided.",
                         evidence={"provided": ", ".join(d.document_types)})

    @staticmethod
    def _fields(d: ValidationInput) -> RuleCheck:
        missing = [f for f in REQUIRED_SUBMITTED_FIELDS if getattr(d.submitted, f) in (None, "")]
        missing += [f for f in d.extraction_missing_fields if f not in missing and d.value(f) in (None, "")]
        if d.value("billed_amount") is None and "billed_amount" not in missing:
            missing.append("billed_amount")
        if missing:
            return RuleCheck(code="REQUIRED_FIELDS", name="Required fields", status="FAIL",
                             message="Could not find: " + ", ".join(f.replace("_", " ") for f in missing) + ".",
                             evidence={"missing": ", ".join(missing)})
        return RuleCheck(code="REQUIRED_FIELDS", name="Required fields", status="PASS",
                         message="All required claim fields are present.")

    @staticmethod
    def _date_order(d: ValidationInput) -> RuleCheck:
        adm, dis = d.value("admission_date"), d.value("discharge_date")
        if not adm or not dis:
            return RuleCheck(code="DATE_ORDER", name="Admission before discharge", status="NOT_EVALUATED",
                             message="Admission or discharge date missing.")
        ev = {"admission_date": adm.isoformat(), "discharge_date": dis.isoformat()}
        if dis < adm:
            return RuleCheck(code="DATE_ORDER", name="Admission before discharge", status="FAIL",
                             message=f"Discharge date {dis} is before admission date {adm}.", evidence=ev)
        return RuleCheck(code="DATE_ORDER", name="Admission before discharge", status="PASS",
                         message="Discharge date is on or after the admission date.", evidence=ev)

    def _future_dates(self, d: ValidationInput) -> RuleCheck:
        ref = min(x for x in (d.submitted_on, self.today) if x)
        future = {f: d.value(f) for f in ("admission_date", "discharge_date")
                  if d.value(f) and d.value(f) > ref}
        if future:
            return RuleCheck(code="FUTURE_DATES", name="Dates not in the future", status="FAIL",
                             message="Hospitalisation dates are after the claim submission date: "
                                     + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in future.items()) + ".",
                             evidence={"reference_date": ref.isoformat(), **{k: v.isoformat() for k, v in future.items()}})
        return RuleCheck(code="FUTURE_DATES", name="Dates not in the future", status="PASS",
                         message="Hospitalisation dates are before the submission date.",
                         evidence={"reference_date": ref.isoformat()})

    @staticmethod
    def _length_of_stay(d: ValidationInput) -> RuleCheck:
        adm, dis = d.value("admission_date"), d.value("discharge_date")
        if not adm or not dis or dis < adm:
            return RuleCheck(code="LENGTH_OF_STAY", name="Length of stay", status="NOT_EVALUATED",
                             message="Length of stay cannot be computed.")
        days = (dis - adm).days
        if days > MAX_STAY_DAYS:
            return RuleCheck(code="LENGTH_OF_STAY", name="Length of stay", status="WARN",
                             message=f"Unusually long stay of {days} days.", evidence={"days": days})
        return RuleCheck(code="LENGTH_OF_STAY", name="Length of stay", status="PASS",
                         message=f"Length of stay: {days} day(s).", evidence={"days": days})

    @staticmethod
    def _amounts(d: ValidationInput) -> RuleCheck:
        claimed, billed = d.submitted.claim_amount, d.value("billed_amount")
        if claimed is None or billed is None:
            return RuleCheck(code="AMOUNT_CONSISTENCY", name="Claimed vs billed amount", status="NOT_EVALUATED",
                             message="Claimed or billed amount missing; the hospital bill is needed.",
                             evidence={"claim_amount": claimed, "billed_amount": billed})
        ev = {"claim_amount": claimed, "billed_amount": billed, "difference": round(claimed - billed, 2)}
        if claimed > billed * (1 + AMOUNT_TOLERANCE):
            pct = (claimed / billed - 1) * 100 if billed else 100.0
            ev["percent_above_bill"] = round(pct, 1)
            return RuleCheck(code="AMOUNT_CONSISTENCY", name="Claimed vs billed amount", status="FAIL",
                             message=f"Claimed {_fmt(claimed)} is {pct:.0f}% above the hospital bill total "
                                     f"{_fmt(billed)}.", evidence=ev)
        return RuleCheck(code="AMOUNT_CONSISTENCY", name="Claimed vs billed amount", status="PASS",
                         message="The claimed amount is supported by the hospital bill.", evidence=ev)

    @staticmethod
    def _bill_items(d: ValidationInput) -> RuleCheck:
        billed = d.value("billed_amount")
        if d.bill_items_total is None or billed is None:
            return RuleCheck(code="BILL_ITEMS_TOTAL", name="Bill line items add up", status="NOT_EVALUATED",
                             message="Bill line items were not read.")
        ev = {"line_items_total": d.bill_items_total, "bill_total": billed}
        if abs(d.bill_items_total - billed) > 1:
            return RuleCheck(code="BILL_ITEMS_TOTAL", name="Bill line items add up", status="WARN",
                             message=f"Line items add up to {_fmt(d.bill_items_total)} but the bill total is "
                                     f"{_fmt(billed)}.", evidence=ev)
        return RuleCheck(code="BILL_ITEMS_TOTAL", name="Bill line items add up", status="PASS",
                         message="Bill line items add up to the bill total.", evidence=ev)

    @staticmethod
    def _form_vs_documents(d: ValidationInput) -> RuleCheck:
        if d.extracted is None or not d.documents_read:
            return RuleCheck(code="FORM_VS_DOCUMENTS", name="Claim form matches documents", status="NOT_EVALUATED",
                             message="Documents were not read; nothing to compare.")
        mismatches: dict[str, str] = {}
        for f in COMPARED_FIELDS:
            form, doc = getattr(d.submitted, f), getattr(d.extracted, f)
            if form in (None, "") or doc in (None, ""):
                continue
            if f == "customer_name":
                same = compare_names(str(form), str(doc)) != "MISMATCH"
            elif f == "claim_amount":
                same = abs(float(form) - float(doc)) <= 1
            else:
                same = str(form).strip().upper() == str(doc).strip().upper()
            if not same:
                mismatches[f] = f"form: {_fmt(form)} / documents: {_fmt(doc)}"
        for conflict in d.conflicts:
            mismatches.setdefault(f"documents_{conflict.field}", " vs ".join(conflict.values.values()))
        if mismatches:
            return RuleCheck(code="FORM_VS_DOCUMENTS", name="Claim form matches documents", status="WARN",
                             message="Values differ between the claim form and documents: "
                                     + "; ".join(f"{k.replace('_', ' ')} ({v})" for k, v in mismatches.items()) + ".",
                             evidence=mismatches)
        return RuleCheck(code="FORM_VS_DOCUMENTS", name="Claim form matches documents", status="PASS",
                         message="The claim form agrees with the documents.")

    def _duplicate_invoice(self, d: ValidationInput) -> RuleCheck:
        invoice = d.value("invoice_number")
        if not invoice:
            return RuleCheck(code="DUPLICATE_INVOICE", name="Invoice not previously claimed", status="NOT_EVALUATED",
                             message="No invoice number available.")
        if self.session is None:
            return RuleCheck(code="DUPLICATE_INVOICE", name="Invoice not previously claimed", status="NOT_EVALUATED",
                             message="Claim history not available.", evidence={"invoice_number": invoice})
        earlier = [c for c in ClaimRepository(self.session).find_by_invoice(invoice, exclude_claim_id=d.claim_id)
                   if d.submitted_on is None or c.submitted_at.date() <= d.submitted_on]
        if earlier:
            numbers = ", ".join(c.claim_number for c in earlier)
            return RuleCheck(code="DUPLICATE_INVOICE", name="Invoice not previously claimed", status="FAIL",
                             message=f"Invoice {invoice} was already used in earlier claim(s) {numbers}.",
                             evidence={"invoice_number": invoice, "earlier_claims": numbers})
        return RuleCheck(code="DUPLICATE_INVOICE", name="Invoice not previously claimed", status="PASS",
                         message=f"Invoice {invoice} has not been used in an earlier claim.",
                         evidence={"invoice_number": invoice})

    @staticmethod
    def _submission_delay(d: ValidationInput) -> RuleCheck:
        dis = d.value("discharge_date")
        if not dis or not d.submitted_on or d.submitted_on < dis:
            return RuleCheck(code="SUBMISSION_DELAY", name="Timely submission", status="NOT_EVALUATED",
                             message="Submission delay cannot be computed.")
        days = (d.submitted_on - dis).days
        if days > LATE_SUBMISSION_DAYS:
            return RuleCheck(code="SUBMISSION_DELAY", name="Timely submission", status="WARN",
                             message=f"Claim submitted {days} days after discharge.", evidence={"days": days})
        return RuleCheck(code="SUBMISSION_DELAY", name="Timely submission", status="PASS",
                         message=f"Submitted {days} day(s) after discharge.", evidence={"days": days})

    # ----------------------------------------------------------------- result
    @staticmethod
    def _finish(checks: list[RuleCheck], d: ValidationInput) -> ValidationResult:
        by = {c.code: c for c in checks}
        missing_docs = [t for t in REQUIRED_DOCUMENTS if t not in d.document_types]
        missing_fields = by["REQUIRED_FIELDS"].evidence.get("missing", "")
        missing_fields = missing_fields.split(", ") if missing_fields else []
        dup = by["DUPLICATE_INVOICE"].evidence.get("earlier_claims")
        flags = {
            "missing_documents": by["REQUIRED_DOCUMENTS"].status == "FAIL",
            "missing_fields": by["REQUIRED_FIELDS"].status == "FAIL",
            "invalid_dates": by["DATE_ORDER"].status == "FAIL" or by["FUTURE_DATES"].status == "FAIL",
            "amount_mismatch": by["AMOUNT_CONSISTENCY"].status == "FAIL",
            "duplicate_invoice": by["DUPLICATE_INVOICE"].status == "FAIL",
            "field_mismatch": by["FORM_VS_DOCUMENTS"].status == "WARN",
        }
        issues = [c.message for c in checks if c.status == "FAIL"]
        warnings = [c.message for c in checks if c.status == "WARN"]
        if flags["missing_documents"] or flags["missing_fields"]:
            status = "INCOMPLETE"
        elif issues:
            status = "INVALID"
        elif warnings:
            status = "VALID_WITH_WARNINGS"
        else:
            status = "VALID"
        summary = {
            "VALID": "All validation checks passed.",
            "VALID_WITH_WARNINGS": f"Valid, with {len(warnings)} warning(s) for the reviewer.",
            "INCOMPLETE": "The claim is incomplete: " + " ".join(issues),
            "INVALID": "Validation found inconsistencies: " + " ".join(issues),
        }[status]
        return ValidationResult(status=status, checks=checks, missing_documents=missing_docs,
                                missing_fields=missing_fields, issues=issues, warnings=warnings,
                                duplicate_claims=dup.split(", ") if dup else [], flags=flags, summary=summary)


def main() -> None:
    import argparse

    from database.connection import SessionLocal

    parser = argparse.ArgumentParser(description="Validate a stored claim (form + stored document list)")
    parser.add_argument("claim_number")
    args = parser.parse_args()
    with SessionLocal() as session:
        claim = ClaimRepository(session).get_by_number(args.claim_number)
        r = ValidationAgent(session).run(build_validation_input(claim))
    print(f"{claim.claim_number}  scenario={claim.scenario}  validation={r.status}")
    for c in r.checks:
        print(f"  {c.status:14} {c.code:20} {c.message}")


if __name__ == "__main__":
    main()
