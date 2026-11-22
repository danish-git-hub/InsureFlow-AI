"""
Agent 1 — Document Extraction Agent
===================================

Responsibility: turn uploaded claim PDFs into validated, structured claim data.

Pipeline for each PDF
    1. Open with PyMuPDF and read every page's text layer (words -> lines).
    2. If a page has (almost) no text, it is a scan -> render it and run OCR.
    3. Classify the document (claim form, hospital bill, discharge summary ...).
    4. Pull labelled fields with deterministic patterns ("Policy Number: ...").
    5. Parse values into proper types (dates, amounts, IDs).

Then across all PDFs
    6. Merge fields using a per-field document priority (e.g. dates are taken
       from the discharge summary first) and record any disagreements as
       `conflicts` — the Validation and Risk agents use these as evidence.
    7. OPTIONAL: if a Gemini key is configured and fields are still missing,
       ask Gemini to find them. Gemini can only FILL GAPS, never overwrite a
       rule-based value, and every field records whether it came from
       "rules" or "gemini".
    8. Validate everything with Pydantic and list the missing fields.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pymupdf
from pydantic import ValidationError

from agents.llm import invoke_structured, llm_available
from agents.schemas import (
    CLAIM_FIELDS,
    REQUIRED_CLAIM_FIELDS,
    BillItem,
    ClaimFields,
    DocumentExtractionResult,
    ExtractedDocument,
    FieldConflict,
)
from app.config import get_logger
from documents import ocr

log = get_logger("document_agent")

MIN_TEXT_CHARS = 25  # fewer native characters than this on a page => treat as scanned
LLM_TEXT_BUDGET = 8000  # characters sent to Gemini (token-conscious)

# --------------------------------------------------------------------------- #
# Document classification (keyword rules on the first part of the document)
# --------------------------------------------------------------------------- #
DOC_TYPE_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("CLAIM_FORM", ("CLAIM FORM",)),
    ("DISCHARGE_SUMMARY", ("DISCHARGE SUMMARY",)),
    ("HOSPITAL_BILL", ("HOSPITAL BILL", "TAX INVOICE", "FINAL BILL", "IN-PATIENT BILL")),
    ("DIAGNOSTIC_REPORT", ("DIAGNOSTIC REPORT", "LABORATORY REPORT", "RADIOLOGY REPORT", "PATHOLOGY REPORT")),
    ("MEDICAL_REPORT", ("MEDICAL REPORT", "CONSULTATION NOTES", "MEDICAL CERTIFICATE")),
    ("POLICY_DOCUMENT", ("POLICY WORDING", "POLICY SCHEDULE")),
    ("INVOICE", ("INVOICE",)),
]


def classify_document(text: str) -> str:
    head = text[:800].upper()
    for doc_type, keywords in DOC_TYPE_KEYWORDS:
        if any(k in head for k in keywords):
            return doc_type
    return "OTHER"


# --------------------------------------------------------------------------- #
# Field patterns. Each label must start a line or follow a wide gap, and be
# followed by ":" or "-". Order matters: more specific labels first.
# --------------------------------------------------------------------------- #
FIELD_LABELS: dict[str, tuple[str, ...]] = {
    "customer_name": (r"claimant name", r"patient name", r"name of (?:the )?insured", r"insured name"),
    "customer_id": (r"customer id", r"member id"),
    "policy_number": (r"policy (?:number|no\.?)",),
    "claim_amount": (r"claim amount", r"amount claimed", r"total claimed amount"),
    "billed_amount": (r"total amount", r"grand total", r"net payable", r"total bill amount"),
    "hospital_name": (r"hospital name", r"name of hospital", r"hospital"),
    "invoice_number": (r"hospital invoice (?:number|no\.?)", r"invoice (?:number|no\.?)", r"bill (?:number|no\.?)"),
    "admission_date": (r"date of admission", r"admission date", r"admitted on"),
    "discharge_date": (r"date of discharge", r"discharge date", r"discharged on"),
    "diagnosis": (r"final diagnosis", r"primary diagnosis", r"diagnosis"),
}
_LABEL_RE = {
    f: re.compile(rf"(?:^|\s{{2,}})(?:{'|'.join(labels)})\s*[:\-]\s*(?P<value>.+)$", re.IGNORECASE)
    for f, labels in FIELD_LABELS.items()
}

# Which document to trust first for each field.
DEFAULT_PRIORITY = ("CLAIM_FORM", "HOSPITAL_BILL", "INVOICE", "DISCHARGE_SUMMARY", "MEDICAL_REPORT",
                    "DIAGNOSTIC_REPORT", "OTHER")
FIELD_PRIORITY: dict[str, tuple[str, ...]] = {
    "billed_amount": ("HOSPITAL_BILL", "INVOICE"),
    "claim_amount": ("CLAIM_FORM",),
    "invoice_number": ("HOSPITAL_BILL", "INVOICE", "CLAIM_FORM"),
    "admission_date": ("DISCHARGE_SUMMARY", "HOSPITAL_BILL", "CLAIM_FORM", "MEDICAL_REPORT"),
    "discharge_date": ("DISCHARGE_SUMMARY", "HOSPITAL_BILL", "CLAIM_FORM"),
    "diagnosis": ("DISCHARGE_SUMMARY", "CLAIM_FORM", "MEDICAL_REPORT"),
    "hospital_name": ("CLAIM_FORM", "DISCHARGE_SUMMARY", "HOSPITAL_BILL", "INVOICE"),
}

DATE_FORMATS = ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d-%b-%Y", "%d %b %Y", "%d %B %Y", "%Y-%m-%d", "%b %d, %Y")
_AMOUNT_RE = re.compile(r"(?:INR|Rs\.?|₹)?\s*([0-9][0-9,]*(?:\.\d{1,2})?)", re.IGNORECASE)
_POLICY_RE = re.compile(r"[A-Z]{2,5}-\d{4}-\d{4,8}")
_INVOICE_RE = re.compile(r"[A-Z0-9][A-Z0-9/\-]{3,39}")
_ITEM_RE = re.compile(r"^(?P<desc>[A-Za-z][^:]{2,80}?)\s+(?:INR|Rs\.?|₹)\s*(?P<amount>[0-9][0-9,]*\.\d{2})$")
_HOSPITAL_WORDS = re.compile(r"\b(hospital|clinic|medical cent(?:re|er)|institute|nursing home)\b", re.I)


# --------------------------------------------------------------------------- #
# Value parsers
# --------------------------------------------------------------------------- #
def parse_amount(value: str) -> float | None:
    m = _AMOUNT_RE.search(value or "")
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def parse_date(value: str) -> date | None:
    value = re.sub(r"\s+", " ", (value or "").strip().rstrip(".,"))
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    # tolerate trailing text such as "27/10/2026 (10:30 AM)"
    m = re.match(r"(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}[ \-][A-Za-z]{3,9}[ \-]\d{4})", value)
    return parse_date(m.group(1)) if m and m.group(1) != value else None


def _clean_text(value: str) -> str | None:
    value = re.split(r"\s{2,}", value.strip())[0].strip(" .,:;-")
    return value or None


def parse_field(field: str, raw: str) -> str | float | date | None:
    if field in ("claim_amount", "billed_amount"):
        return parse_amount(raw)
    if field in ("admission_date", "discharge_date"):
        return parse_date(_clean_text(raw) or "")
    if field == "policy_number":
        compact = re.sub(r"\s*-\s*", "-", raw.upper())  # OCR sometimes adds spaces around hyphens
        m = _POLICY_RE.search(compact)
        return m.group(0) if m else _clean_text(raw)
    if field == "invoice_number":
        m = _INVOICE_RE.search(re.sub(r"\s*-\s*", "-", raw.upper()))
        return m.group(0) if m else None
    return _clean_text(raw)


# --------------------------------------------------------------------------- #
# Reading PDFs
# --------------------------------------------------------------------------- #
def _native_lines(page: pymupdf.Page, gap: float = 12.0) -> list[str]:
    """Rebuild visual lines from word boxes, so a label and a right-aligned
    amount on the same row end up on one line (wide gaps become 2 spaces)."""
    words = sorted(page.get_text("words"), key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    rows: list[tuple[float, list[tuple[float, float, str]]]] = []  # (mid_y, [(x0, x1, word)])
    for x0, y0, x1, y1, word, *_ in words:
        mid = (y0 + y1) / 2
        if rows and abs(mid - rows[-1][0]) <= 3:
            rows[-1][1].append((x0, x1, word))
        else:
            rows.append((mid, [(x0, x1, word)]))
    lines = []
    for _, cells in rows:
        cells.sort()
        parts, prev_x1 = [], None
        for x0, x1, word in cells:
            if prev_x1 is not None:
                parts.append("  " if x0 - prev_x1 > gap else " ")
            parts.append(word)
            prev_x1 = x1
        lines.append("".join(parts).strip())
    return [ln for ln in lines if ln]


@dataclass
class DocumentInput:
    path: Path
    declared_type: str | None = None  # type chosen by the user at upload time (a hint only)
    display_name: str | None = None


def read_pdf(path: Path) -> tuple[list[str], int, str, float | None, list[str]]:
    """Return (lines, page_count, method, ocr_confidence, warnings)."""
    warnings: list[str] = []
    lines: list[str] = []
    used_text = used_ocr = False
    confidences: list[float] = []
    with pymupdf.open(path) as doc:
        page_count = doc.page_count
        for number, page in enumerate(doc, start=1):
            page_lines = _native_lines(page)
            if sum(len(x) for x in page_lines) >= MIN_TEXT_CHARS:
                used_text = True
                lines.extend(page_lines)
                continue
            # Scanned page -> OCR fallback
            if not ocr.ocr_available():
                warnings.append(f"Page {number} appears scanned but OCR is not installed; text could not be read.")
                continue
            try:
                ocr_lines, conf = ocr.ocr_page(page)
                used_ocr = True
                confidences.append(conf)
                lines.extend(ocr_lines)
                if conf < 0.8:
                    warnings.append(f"Page {number}: low OCR confidence ({conf:.0%}); verify values manually.")
            except Exception as exc:
                warnings.append(f"Page {number}: OCR failed ({exc.__class__.__name__}).")
    method = "TEXT+OCR" if used_text and used_ocr else "OCR" if used_ocr else "TEXT" if used_text else "NONE"
    conf = round(sum(confidences) / len(confidences), 3) if confidences else None
    return lines, page_count, method, conf, warnings


def extract_fields(lines: list[str], doc_type: str) -> dict[str, str | float | date | None]:
    found: dict[str, str | float | date | None] = {}
    for line in lines:
        for field, pattern in _LABEL_RE.items():
            if field in found:
                continue
            m = pattern.search(line)
            if m:
                value = parse_field(field, m.group("value"))
                if value not in (None, ""):
                    found[field] = value
    # Bills often print the hospital only as a letterhead, not as "Hospital: ...".
    if "hospital_name" not in found and doc_type in ("HOSPITAL_BILL", "INVOICE", "DISCHARGE_SUMMARY"):
        for line in lines[:4]:
            if _HOSPITAL_WORDS.search(line) and ":" not in line:
                found["hospital_name"] = _clean_text(line)
                break
    return found


def extract_bill_items(lines: list[str]) -> list[BillItem]:
    items = []
    for line in lines:
        m = _ITEM_RE.match(line.strip())
        if m and "total" not in m.group("desc").lower():
            items.append(BillItem(description=m.group("desc").strip(), amount=parse_amount(m.group("amount")) or 0))
    return items


# --------------------------------------------------------------------------- #
# Merge helpers
# --------------------------------------------------------------------------- #
def _normalise(field: str, value) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).lower()
    if field == "hospital_name":
        text = text.split(",")[0]
    return re.sub(r"[^a-z0-9]", "", text)


def _display(value) -> str:
    return value.isoformat() if isinstance(value, date) else f"{value:,.2f}" if isinstance(value, float) else str(value)


# --------------------------------------------------------------------------- #
# The agent
# --------------------------------------------------------------------------- #
class DocumentExtractionAgent:
    key = "document_agent"
    name = "Document Extraction Agent"
    purpose = "Reads claim PDFs (with OCR fallback) and extracts structured, validated claim fields."

    def __init__(self, use_llm: bool | None = None, llm_model=None):
        # `llm_model` lets tests inject a fake model; otherwise Gemini is used only if a key exists.
        self.llm_model = llm_model
        self.use_llm = (llm_available() or llm_model is not None) if use_llm is None else use_llm

    # -- single document ------------------------------------------------------
    def extract_document(self, item: DocumentInput) -> tuple[ExtractedDocument, dict, list[BillItem], str]:
        """Returns (document summary, parsed fields, bill items, full text)."""
        path = Path(item.path)
        name = item.display_name or path.name
        try:
            lines, pages, method, conf, warnings = read_pdf(path)
        except Exception as exc:
            log.warning("Could not open %s: %s", name, exc)
            doc = ExtractedDocument(filename=name, document_type=item.declared_type or "OTHER",
                                    declared_type=item.declared_type, page_count=0, extraction_method="NONE",
                                    characters=0, warnings=[f"Could not open PDF ({exc.__class__.__name__})."])
            return doc, {}, [], ""

        text = "\n".join(lines)
        detected = classify_document(text)
        doc_type = detected if detected != "OTHER" else (item.declared_type or "OTHER")
        if item.declared_type and detected not in ("OTHER", item.declared_type):
            warnings.append(f"Uploaded as {item.declared_type} but content looks like {detected}.")
        fields = extract_fields(lines, doc_type)
        items = extract_bill_items(lines) if doc_type in ("HOSPITAL_BILL", "INVOICE") else []
        doc = ExtractedDocument(
            filename=name, document_type=doc_type, declared_type=item.declared_type, page_count=pages,
            extraction_method=method, ocr_confidence=conf, characters=len(text),
            fields={k: _display(v) for k, v in fields.items()}, bill_items=items,
            text_preview=text[:500], warnings=warnings,
        )
        return doc, fields, items, text

    # -- whole claim ------------------------------------------------------------
    def run(self, documents: list[DocumentInput | str | Path]) -> DocumentExtractionResult:
        inputs = [d if isinstance(d, DocumentInput) else DocumentInput(Path(d)) for d in documents]
        extracted = [self.extract_document(item) for item in inputs]

        merged, sources_doc, conflicts = self._merge(extracted)
        warnings = [f"{d.filename}: {w}" for d, *_ in extracted for w in d.warnings]

        bill_items: list[BillItem] = next((items for d, _, items, _ in extracted if items), [])
        if bill_items and merged.get("billed_amount") is not None:
            total = round(sum(i.amount for i in bill_items), 2)
            if abs(total - merged["billed_amount"]) > 1:
                warnings.append(f"Bill line items add up to {total:,.2f} but the bill total is "
                                f"{merged['billed_amount']:,.2f}.")

        field_sources = {f: "rules" for f in merged}
        result_kwargs = {}
        missing = [f for f in REQUIRED_CLAIM_FIELDS if merged.get(f) in (None, "")]

        # Optional Gemini gap-filling
        if self.use_llm and missing and any(t for *_, t in extracted):
            corpus = "\n\n".join(f"### {d.filename} ({d.document_type})\n{t}" for d, _, _, t in extracted if t)
            llm_fields, call = invoke_structured(self._prompt(corpus[:LLM_TEXT_BUDGET], missing), ClaimFields,
                                                 model=self.llm_model)
            if call.ok and llm_fields is not None:
                result_kwargs.update(used_llm=True, llm_model=call.model, extraction_mode="RULE_BASED+GEMINI")
                for f in missing:
                    value = getattr(llm_fields, f)
                    if value not in (None, ""):
                        merged[f] = value
                        field_sources[f] = "gemini"
                        sources_doc[f] = "Gemini (from document text)"
            else:
                warnings.append(f"Gemini extraction unavailable ({call.error}); rule-based results used.")

        try:
            fields = ClaimFields(**{f: merged.get(f) for f in CLAIM_FIELDS})
        except ValidationError as exc:
            # Drop only the invalid fields and keep the rest.
            bad = {e["loc"][0] for e in exc.errors()}
            warnings.append(f"Discarded invalid values for: {', '.join(sorted(bad))}.")
            fields = ClaimFields(**{f: merged.get(f) for f in CLAIM_FIELDS if f not in bad})

        missing = [f for f in REQUIRED_CLAIM_FIELDS if getattr(fields, f) in (None, "")]
        docs = [d for d, *_ in extracted]
        return DocumentExtractionResult(
            fields=fields,
            field_sources={k: v for k, v in field_sources.items() if getattr(fields, k, None) is not None},
            field_documents=sources_doc,
            bill_items=bill_items,
            documents=docs,
            document_types_found=sorted({d.document_type for d in docs}),
            missing_fields=missing,
            conflicts=conflicts,
            warnings=warnings,
            ocr_used=any(d.extraction_method in ("OCR", "TEXT+OCR") for d in docs),
            **result_kwargs,
        )

    @staticmethod
    def _merge(extracted) -> tuple[dict, dict, list[FieldConflict]]:
        merged: dict = {}
        sources: dict[str, str] = {}
        conflicts: list[FieldConflict] = []
        for field in CLAIM_FIELDS:
            priority = FIELD_PRIORITY.get(field, DEFAULT_PRIORITY)
            candidates = [(d, f[field]) for d, f, *_ in extracted if f.get(field) not in (None, "")]
            if field in FIELD_PRIORITY:  # restrict to trusted document types for this field
                candidates = [c for c in candidates if c[0].document_type in priority]
            if not candidates:
                continue
            rank = {t: i for i, t in enumerate(priority)}
            candidates.sort(key=lambda c: rank.get(c[0].document_type, len(rank)))
            merged[field], sources[field] = candidates[0][1], candidates[0][0].filename

            distinct = OrderedDict()
            for d, v in candidates:
                distinct.setdefault(_normalise(field, v), (d.filename, _display(v)))
            if field == "hospital_name":  # "X Hospital" vs "X Hospital, City" is not a conflict
                keys = list(distinct)
                if all(k.startswith(keys[0]) or keys[0].startswith(k) for k in keys):
                    continue
            if len(distinct) > 1:
                conflicts.append(FieldConflict(field=field, values={fn: val for fn, val in distinct.values()}))
        return merged, sources, conflicts

    @staticmethod
    def _prompt(text: str, missing: list[str]) -> str:
        return (
            "You are an insurance claims document extraction assistant.\n"
            "Extract the claim fields from the documents below. Rules:\n"
            "- Use ONLY values explicitly written in the documents. Never guess or infer.\n"
            "- If a value is not present, return null for that field.\n"
            "- Dates as YYYY-MM-DD. Amounts as plain numbers in INR.\n"
            f"- The fields currently missing are: {', '.join(missing)}.\n\n"
            f"DOCUMENTS:\n{text}"
        )
