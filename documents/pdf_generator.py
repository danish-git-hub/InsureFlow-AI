"""
Generates realistic, fictional claim documents as PDFs with PyMuPDF.

Each synthetic claim gets the document types listed in its `documents` field:
    CLAIM_FORM, HOSPITAL_BILL, DISCHARGE_SUMMARY, DIAGNOSTIC_REPORT, MEDICAL_REPORT

The documents deliberately use *different* label wording and date formats
(dd/mm/yyyy, 27-Oct-2026, 2026-10-27) so the extraction agent is tested on
realistic variation, not one fixed template.

`scanned=True` renders the page to an image and saves an image-only PDF —
there is no text layer, so the extraction agent must fall back to OCR.

Built-in PDF fonts cannot draw the rupee sign, so amounts are printed as "INR".
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pymupdf

NAVY = (16 / 255, 42 / 255, 67 / 255)
TEAL = (15 / 255, 118 / 255, 110 / 255)
GREY = (0.38, 0.45, 0.55)
A4 = pymupdf.paper_rect("a4")
MARGIN = 56


def inr(amount: float | int) -> str:
    return f"INR {amount:,.2f}"


def _d(value: str | date | None, fmt: str) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        value = date.fromisoformat(value[:10])
    return value.strftime(fmt)


class _Page:
    """Tiny layout helper that keeps a vertical cursor."""

    def __init__(self, doc: pymupdf.Document, org: str, title: str, subtitle: str = ""):
        self.page = doc.new_page(width=A4.width, height=A4.height)
        self.y = 0.0
        band = pymupdf.Rect(0, 0, A4.width, 92)
        self.page.draw_rect(band, color=NAVY, fill=NAVY)
        self.page.insert_text((MARGIN, 38), org, fontsize=15, fontname="hebo", color=(1, 1, 1))
        if subtitle:
            self.page.insert_text((MARGIN, 58), subtitle, fontsize=9, fontname="helv", color=(0.85, 0.9, 0.95))
        self.page.insert_text((MARGIN, 128), title, fontsize=16, fontname="hebo", color=TEAL)
        self.page.draw_line((MARGIN, 138), (A4.width - MARGIN, 138), color=TEAL, width=1.2)
        self.y = 166

    def field(self, label: str, value: Any) -> None:
        # Label and value in ONE text run so text extraction keeps them on one line.
        self.page.insert_text((MARGIN, self.y), f"{label}: {value}", fontsize=10.5, fontname="helv", color=NAVY)
        self.y += 19

    def heading(self, text: str) -> None:
        self.y += 8
        self.page.insert_text((MARGIN, self.y), text, fontsize=11.5, fontname="hebo", color=TEAL)
        self.y += 18

    def paragraph(self, text: str, size: float = 9.5) -> None:
        rect = pymupdf.Rect(MARGIN, self.y - 10, A4.width - MARGIN, self.y + 90)
        overflow = self.page.insert_textbox(rect, text, fontsize=size, fontname="helv", color=GREY)
        # insert_textbox returns the unused height (negative if the text did not fit)
        used = 90 - overflow if overflow >= 0 else 90
        self.y += used + 6

    def item(self, description: str, amount: float) -> None:
        self.page.insert_text((MARGIN + 6, self.y), description, fontsize=10, fontname="helv", color=NAVY)
        txt = inr(amount)
        width = pymupdf.get_text_length(txt, fontname="helv", fontsize=10)
        self.page.insert_text((A4.width - MARGIN - width, self.y), txt, fontsize=10, fontname="helv", color=NAVY)
        self.y += 17

    def rule(self) -> None:
        self.page.draw_line((MARGIN, self.y - 8), (A4.width - MARGIN, self.y - 8), color=GREY, width=0.5)
        self.y += 6

    def footer(self, text: str) -> None:
        self.page.insert_text((MARGIN, A4.height - 36), text, fontsize=7.5, fontname="helv", color=GREY)


FICTION = "Synthetic document generated for InsureFlow AI demonstration purposes only. All data is fictional."


def _hospital_parts(hospital: str) -> tuple[str, str]:
    name, _, city = hospital.partition(",")
    return name.strip(), city.strip()


# --------------------------------------------------------------------------- #
# Document builders
# --------------------------------------------------------------------------- #
def _claim_form(doc, c: dict) -> None:
    p = _Page(doc, "InsureFlow Insurance (Fictional)", "HEALTH INSURANCE CLAIM FORM", "Part A - To be filled by the insured")
    p.field("Claimant Name", c["customer_name"])
    p.field("Customer ID", c["customer_code"])
    p.field("Policy Number", c["policy_number"])
    p.field("Insurance Type", c["insurance_type"])
    p.heading("Hospitalisation details")
    p.field("Hospital Name", c["hospital_name"])
    p.field("Date of Admission", _d(c["admission_date"], "%d/%m/%Y"))
    p.field("Date of Discharge", _d(c["discharge_date"], "%d/%m/%Y"))
    p.field("Diagnosis", c["diagnosis"])
    p.field("Hospital Invoice Number", c["invoice_number"])
    p.heading("Claim")
    p.field("Claim Amount", inr(c["claim_amount"]))
    p.field("Date of Claim", _d(c["submitted_at"], "%d/%m/%Y"))
    p.heading("Declaration")
    p.paragraph("I declare that the information furnished in this claim form is true and correct to the best of my "
                "knowledge, and that I have not withheld any material fact. I authorise the insurer to obtain medical "
                "records relevant to this claim.")
    p.field("Signature", c["customer_name"])
    p.footer(FICTION)


def _hospital_bill(doc, c: dict) -> None:
    name, city = _hospital_parts(c["hospital_name"])
    p = _Page(doc, name, "FINAL HOSPITAL BILL / TAX INVOICE", f"{city} - NABH accredited (fictional)")
    p.field("Invoice No", c["invoice_number"])
    p.field("Bill Date", _d(c["discharge_date"], "%d-%b-%Y"))
    p.field("Patient Name", c["customer_name"])
    p.field("Policy Number", c["policy_number"])
    p.field("Admission Date", _d(c["admission_date"], "%d-%b-%Y"))
    p.field("Discharge Date", _d(c["discharge_date"], "%d-%b-%Y"))
    p.heading("Bill particulars")
    for item in c["bill_items"]:
        p.item(item["description"], item["amount"])
    p.rule()
    p.field("Total Amount", inr(c["billed_amount"]))
    p.paragraph("This is a computer generated invoice. Payment received from patient / TPA as applicable.")
    p.footer(FICTION)


def _discharge_summary(doc, c: dict) -> None:
    name, city = _hospital_parts(c["hospital_name"])
    p = _Page(doc, name, "DISCHARGE SUMMARY", city)
    p.field("Patient Name", c["customer_name"])
    p.field("Hospital", c["hospital_name"])
    p.field("Admission Date", _d(c["admission_date"], "%Y-%m-%d"))
    p.field("Discharge Date", _d(c["discharge_date"], "%Y-%m-%d"))
    p.field("Final Diagnosis", c["diagnosis"])
    p.heading("Course in hospital")
    p.paragraph(f"The patient was admitted with symptoms consistent with {c['diagnosis'].lower()}. "
                "Investigations were performed and appropriate treatment was administered. The patient responded "
                "well and is being discharged in a stable condition.")
    p.field("Condition at Discharge", "Stable")
    p.field("Treating Doctor", "Dr. A. Fictional, MS")
    p.footer(FICTION)


def _diagnostic_report(doc, c: dict) -> None:
    name, _ = _hospital_parts(c["hospital_name"])
    p = _Page(doc, f"{name} - Laboratory", "DIAGNOSTIC REPORT")
    p.field("Patient Name", c["customer_name"])
    p.field("Report Date", _d(c["admission_date"], "%d/%m/%Y"))
    p.field("Investigation", "Complete blood count, imaging as clinically indicated")
    p.field("Impression", f"Findings consistent with {c['diagnosis']}")
    p.footer(FICTION)


def _medical_report(doc, c: dict) -> None:
    name, _ = _hospital_parts(c["hospital_name"])
    p = _Page(doc, name, "MEDICAL REPORT", "Consultation notes")
    p.field("Patient Name", c["customer_name"])
    p.field("Consultation Date", _d(c["admission_date"], "%d/%m/%Y"))
    p.field("Diagnosis", c["diagnosis"])
    p.paragraph("Hospitalisation advised for further management. Patient counselled regarding the procedure and "
                "expected recovery.")
    p.footer(FICTION)


BUILDERS = {
    "CLAIM_FORM": _claim_form,
    "HOSPITAL_BILL": _hospital_bill,
    "DISCHARGE_SUMMARY": _discharge_summary,
    "DIAGNOSTIC_REPORT": _diagnostic_report,
    "MEDICAL_REPORT": _medical_report,
}


def _rasterise(doc: pymupdf.Document, dpi: int = 150) -> pymupdf.Document:
    """Simulate a scanner: replace every page with a picture of itself."""
    scanned = pymupdf.open()
    for page in doc:
        pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
        new = scanned.new_page(width=page.rect.width, height=page.rect.height)
        new.insert_image(new.rect, pixmap=pix)
    return scanned


def build_document(doc_type: str, claim: dict, scanned: bool = False) -> bytes:
    """Return PDF bytes for one document type of one claim."""
    if doc_type not in BUILDERS:
        raise ValueError(f"Unsupported document type: {doc_type}")
    doc = pymupdf.open()
    BUILDERS[doc_type](doc, claim)
    doc.set_metadata({"title": f"{doc_type} {claim['claim_number']}", "producer": "InsureFlow AI synthetic generator"})
    if scanned:
        doc = _rasterise(doc)
    return doc.tobytes(garbage=3, deflate=True)


def generate_claim_documents(claim: dict, out_dir: Path, scanned_types: set[str] | None = None) -> list[Path]:
    """Write all documents for one claim to `out_dir`. Returns the file paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for doc_type in claim["documents"]:
        scanned = doc_type in (scanned_types or set())
        path = out_dir / f"{claim['claim_number']}_{doc_type.lower()}{'_scanned' if scanned else ''}.pdf"
        path.write_bytes(build_document(doc_type, claim, scanned=scanned))
        paths.append(path)
    return paths

