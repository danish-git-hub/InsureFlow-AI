"""
Generates the fictional policy wording PDFs that the RAG pipeline indexes.

One PDF per plan in `database.catalog.PLANS`, written to data/policies/:

    IHS_policy_wording.pdf, FFS_policy_wording.pdf, SCP_..., PAG_..., CIP_...

Every clause starts with a heading line of the form

    COV-01  In-patient hospitalisation

so the loader can split the document clause by clause and keep the clause
code, title, section and page number as metadata (the citation shown to the
reviewer). The wording is derived from the catalogue, so the deterministic
rules (Agent 2) and the retrieved text (Agent 3) never contradict each other.

    python -m rag.policy_documents      # (re)generate the PDFs
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf

from app.config import settings
from database.catalog import PLANS, REQUIRED_DOCUMENTS, Plan
from database.models import ClauseCategory
from documents.pdf_generator import A4, GREY, MARGIN, NAVY, TEAL

FICTION = "Fictional policy wording generated for InsureFlow AI demonstration purposes only. Not a real insurance contract."

# Section titles in the PDF -> clause category stored with every chunk.
SECTIONS: dict[str, ClauseCategory] = {
    "Section 1 - Definitions": ClauseCategory.DEFINITION,
    "Section 2 - What is covered": ClauseCategory.COVERAGE,
    "Section 3 - Waiting periods": ClauseCategory.WAITING_PERIOD,
    "Section 4 - Exclusions": ClauseCategory.EXCLUSION,
    "Section 5 - Claim conditions": ClauseCategory.CONDITION,
}


@dataclass(frozen=True)
class Clause:
    code: str
    title: str
    text: str


def _doc_label(code: str) -> str:
    return code.replace("_", " ").lower()


def plan_clauses(plan: Plan) -> dict[str, list[Clause]]:
    """The clause content of one plan, grouped by section title."""
    accident_only = plan.code == "PAG"
    definitions = [
        Clause("DEF-01", "Hospital",
               "Hospital means an institution registered with the local authorities for in-patient care, with "
               "qualified nursing staff round the clock, at least 10 in-patient beds and a fully equipped "
               "operation theatre."),
        Clause("DEF-02", "Hospitalisation",
               "Hospitalisation means admission to a hospital for a minimum period of 24 consecutive in-patient "
               "care hours, except for listed day-care procedures."),
        Clause("DEF-03", "Accident",
               "Accident means a sudden, unforeseen and involuntary event caused by external, violent and visible "
               "means, such as a road traffic accident, a fall resulting in a fracture, or a burn injury."),
        Clause("DEF-04", "Sum insured",
               "Sum insured means the maximum amount stated in the policy schedule that the insurer will pay for "
               f"all admissible claims in one policy year. Sum insured options for this plan: "
               + ", ".join(f"INR {s:,}" for s in plan.sum_insured_options) + "."),
    ]

    coverage = [
        Clause("COV-01", "In-patient hospitalisation",
               f"Subject to the terms of this policy, the insurer covers: {plan.covered_summary} Admissible "
               "expenses include room rent, nursing, surgeon and anaesthetist fees, medicines, diagnostics and "
               "operation theatre charges incurred during hospitalisation."),
        Clause("COV-02", "Limit of cover",
               "The total of all admissible claims during the policy year shall not exceed the sum insured. Any "
               "claimed amount above the available sum insured is not payable and is reviewed by the claims "
               "assessor."),
    ]
    if plan.copay_percent:
        coverage.append(Clause(
            "COV-03", "Co-payment",
            f"A co-payment of {plan.copay_percent:g}% applies to every admissible claim. The insured person "
            f"bears {plan.copay_percent:g}% of the admissible amount and the insurer pays the balance."))
    if accident_only:
        coverage.append(Clause(
            "COV-04", "Accidental injury only",
            "Cover applies only to hospitalisation caused solely and directly by an accident, such as a "
            "fracture, ligament tear, dislocation, burn or road traffic injury."))
    if plan.code == "CIP":
        coverage.append(Clause(
            "COV-05", "Listed critical illnesses",
            "Cover applies to hospitalisation for the listed critical illnesses: coronary artery disease "
            "requiring angioplasty or bypass surgery, stroke, cancer of specified severity and end-stage "
            "kidney failure."))

    if plan.default_waiting_days:
        waiting = [Clause(
            "WP-01", "Initial waiting period",
            f"No claim is payable for hospitalisation that begins within the first {plan.default_waiting_days} "
            "days from the policy start date. This waiting period does not apply to hospitalisation caused by "
            "an accident.")]
    else:
        waiting = [Clause(
            "WP-01", "Initial waiting period",
            "There is no initial waiting period under this plan. Cover for accidental injury starts from the "
            "policy start date.")]

    exclusions = []
    for exc in plan.exclusions:
        examples = ", ".join(exc.keywords)
        exclusions.append(Clause(exc.code, exc.title, f"{exc.wording} Examples include: {examples}."))

    docs = ", ".join(_doc_label(d) for d in REQUIRED_DOCUMENTS)
    conditions = [
        Clause("CON-01", "Documents required",
               f"A claim must be supported by the following documents: {docs}. Diagnostic and medical reports "
               "may be requested. The insurer may ask for additional information before assessing a claim."),
        Clause("CON-02", "Policy in force",
               "A claim is admissible only if the policy was in force on the date of admission. Hospitalisation "
               "that begins after the policy end date, or while the policy is lapsed or cancelled, is not covered."),
        Clause("CON-03", "Duplicate and inconsistent claims",
               "The same expense cannot be claimed more than once. Where invoice numbers repeat or amounts differ "
               "between documents, the claim is referred for verification before assessment."),
        Clause("CON-04", "Claim assessment",
               "Every claim is assessed by an authorised claims officer, who makes the final decision on "
               "admissibility and the amount payable."),
    ]

    return dict(zip(SECTIONS, [definitions, coverage, waiting, exclusions, conditions]))


class _Writer:
    """Multi-page writer with a vertical cursor and automatic page breaks."""

    BOTTOM = A4.height - 60

    def __init__(self, doc: pymupdf.Document, plan: Plan):
        self.doc, self.plan = doc, plan
        self.page: pymupdf.Page | None = None
        self.y = 0.0
        self._new_page(first=True)

    def _new_page(self, first: bool = False) -> None:
        self.page = self.doc.new_page(width=A4.width, height=A4.height)
        band_h = 92 if first else 44
        self.page.draw_rect(pymupdf.Rect(0, 0, A4.width, band_h), color=NAVY, fill=NAVY)
        if first:
            self.page.insert_text((MARGIN, 38), "InsureFlow Insurance (Fictional)", fontsize=15, fontname="hebo",
                                  color=(1, 1, 1))
            self.page.insert_text((MARGIN, 58), f"Policy wording - {self.plan.name} ({self.plan.code})", fontsize=9,
                                  fontname="helv", color=(0.85, 0.9, 0.95))
            self.page.insert_text((MARGIN, 128), f"{self.plan.name} - Policy Wording", fontsize=16,
                                  fontname="hebo", color=TEAL)
            self.page.insert_text((MARGIN, 146), f"Plan code: {self.plan.code}   Product: {self.plan.insurance_type.value}",
                                  fontsize=9.5, fontname="helv", color=GREY)
            self.y = 176
        else:
            self.page.insert_text((MARGIN, 28), f"{self.plan.name} ({self.plan.code}) - continued", fontsize=9,
                                  fontname="helv", color=(0.85, 0.9, 0.95))
            self.y = 76
        self.page.insert_text((MARGIN, A4.height - 30), f"{FICTION}   Page {self.doc.page_count}", fontsize=7,
                              fontname="helv", color=GREY)

    def _ensure(self, height: float) -> None:
        if self.y + height > self.BOTTOM:
            self._new_page()

    def section(self, title: str) -> None:
        self._ensure(60)
        self.y += 6
        self.page.insert_text((MARGIN, self.y), title, fontsize=12.5, fontname="hebo", color=TEAL)
        self.page.draw_line((MARGIN, self.y + 6), (A4.width - MARGIN, self.y + 6), color=TEAL, width=0.8)
        self.y += 24

    def clause(self, clause: Clause) -> None:
        width = A4.width - 2 * MARGIN
        # Measure the body height on a scratch rect so a clause never splits across pages.
        lines = max(1, int(pymupdf.get_text_length(clause.text, fontname="helv", fontsize=9.5) / (width - 10)) + 1)
        body_h = lines * 13 + 14
        self._ensure(18 + body_h)
        self.page.insert_text((MARGIN, self.y), f"{clause.code}  {clause.title}", fontsize=10.5, fontname="hebo",
                              color=NAVY)
        self.y += 8
        rect = pymupdf.Rect(MARGIN, self.y, A4.width - MARGIN, self.y + body_h + 40)
        unused = self.page.insert_textbox(rect, clause.text, fontsize=9.5, fontname="helv", color=GREY)
        self.y += (rect.height - unused if unused >= 0 else rect.height) + 14


def build_policy_wording(plan: Plan) -> bytes:
    doc = pymupdf.open()
    writer = _Writer(doc, plan)
    for section, clauses in plan_clauses(plan).items():
        writer.section(section)
        for clause in clauses:
            writer.clause(clause)
    doc.set_metadata({"title": f"{plan.name} - Policy Wording", "subject": plan.code,
                      "creator": "InsureFlow AI (synthetic)"})
    data = doc.tobytes(garbage=3, deflate=True)
    doc.close()
    return data


def generate_policy_documents(out_dir: Path | None = None, overwrite: bool = False) -> list[Path]:
    """Write one wording PDF per plan; existing files are kept unless overwrite=True."""
    out_dir = out_dir or settings.policies_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for plan in PLANS.values():
        path = out_dir / plan.document_filename
        if overwrite or not path.exists():
            path.write_bytes(build_policy_wording(plan))
        paths.append(path)
    return paths


if __name__ == "__main__":
    for p in generate_policy_documents(overwrite=True):
        print(p)
