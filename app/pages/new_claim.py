"""New Claim Submission — validated form, PDF uploads, optional immediate AI processing."""

from __future__ import annotations

import random
from datetime import date, timedelta

import streamlit as st
from pydantic import ValidationError

from app.components import backend
from app.components.ui import badge, label, page_header, section, status_badge
from app.config import settings
from database.catalog import DIAGNOSES, HOSPITALS
from database.connection import SessionLocal
from database.models import DocumentType, InsuranceType, PolicyStatus
from database.repository import PolicyRepository
from services.claim_service import ServiceError
from services.schemas import ClaimSubmission, UploadedFile

page_header("New Claim", "Submit a health insurance claim with its documents. The six AI agents can process it "
                         "immediately; a claims officer always makes the final decision.")

FIELDS = ("customer_name", "customer_id", "policy_number", "insurance_type", "claim_amount", "hospital_name",
          "admission_date", "discharge_date", "diagnosis", "description")
TYPES = [t.value for t in InsuranceType]
DOC_TYPES = ["Auto-detect"] + [t.value for t in DocumentType if t.value != "POLICY_DOCUMENT"]


def _guess_type(filename: str) -> str:
    name = filename.lower()
    for key, value in (("claim", "CLAIM_FORM"), ("bill", "HOSPITAL_BILL"), ("invoice", "HOSPITAL_BILL"),
                       ("discharge", "DISCHARGE_SUMMARY"), ("diagnostic", "DIAGNOSTIC_REPORT"),
                       ("lab", "DIAGNOSTIC_REPORT"), ("medical", "MEDICAL_REPORT")):
        if key in name:
            return value
    return "Auto-detect"


def _fill_demo() -> None:
    """Pre-fill the form with a fictional, policy-consistent claim for a live demo."""
    with SessionLocal() as s:
        today = date.today()
        candidates = [p for p in PolicyRepository(s).list(status=PolicyStatus.ACTIVE.value)
                      if p.start_date + timedelta(days=p.waiting_period_days + 10) < today - timedelta(days=5)
                      and p.plan_code != "PAG"]
        policy = random.choice(candidates)
        dx = random.choice([d for d in DIAGNOSES if not d.critical])
        admission = today - timedelta(days=random.randint(5, 20))
        st.session_state.update({
            "customer_name": policy.customer.full_name, "customer_id": policy.customer.customer_code,
            "policy_number": policy.policy_number, "insurance_type": policy.insurance_type,
            "claim_amount": float(random.randint(dx.min_cost // 1000, dx.max_cost // 1000) * 1000),
            "hospital_name": random.choice(HOSPITALS), "admission_date": admission,
            "discharge_date": admission + timedelta(days=random.randint(2, 4)), "diagnosis": dx.name,
            "description": "Demo claim generated from fictional data.", "demo_docs": True,
        })


def _demo_documents(form: ClaimSubmission) -> list[UploadedFile]:
    """Generate claim form, hospital bill and discharge summary PDFs that match the form."""
    from documents.pdf_generator import build_document

    amount = form.claim_amount
    parts = [("Room and nursing charges", 0.28), ("Surgeon and consultant fees", 0.32), ("Medicines and consumables", 0.2),
             ("Diagnostics and laboratory", 0.12)]
    items = [{"description": d, "amount": round(amount * share, 2)} for d, share in parts]
    items.append({"description": "Operation theatre and other charges", "amount": round(amount - sum(i["amount"] for i in items), 2)})
    claim = {
        **form.model_dump(), "customer_code": form.customer_id, "insurance_type": form.insurance_type.value,
        "invoice_number": f"INV-DEMO-{date.today():%Y}-{random.randint(10000, 99999)}", "billed_amount": amount,
        "bill_items": items, "submitted_at": date.today().isoformat(), "claim_number": "NEW",
        "admission_date": form.admission_date.isoformat(), "discharge_date": form.discharge_date.isoformat(),
    }
    return [UploadedFile(filename=f"{t.lower()}.pdf", content=build_document(t, claim), declared_type=t)
            for t in ("CLAIM_FORM", "HOSPITAL_BILL", "DISCHARGE_SUMMARY")]


for _key in ("admission_date", "discharge_date"):
    st.session_state.setdefault(_key, None)

top = st.columns([4, 1])
with top[1]:
    st.button("Fill demo claim", icon=":material/auto_fix_high:", on_click=_fill_demo, width="stretch",
              help="Pre-fills a fictional claim for an active policy and generates matching PDFs.")

section("Claimant and policy")
c1, c2, c3 = st.columns(3)
c1.text_input("Customer name *", key="customer_name", placeholder="e.g. Rahul Sharma")
c2.text_input("Customer ID *", key="customer_id", placeholder="e.g. CUST-0001")
c3.text_input("Policy number *", key="policy_number", placeholder="e.g. IHS-2025-000001")
c1, c2, c3 = st.columns(3)
c1.selectbox("Insurance type *", TYPES, key="insurance_type")
c2.number_input("Claim amount (₹) *", min_value=0.0, step=1000.0, key="claim_amount", format="%.2f")
c3.text_input("Hospital name *", key="hospital_name")

section("Hospitalisation")
c1, c2 = st.columns(2)
c1.date_input("Admission date *", key="admission_date", max_value=date.today() + timedelta(days=365))
c2.date_input("Discharge date *", key="discharge_date", max_value=date.today() + timedelta(days=365))
st.text_input("Diagnosis *", key="diagnosis")
st.text_area("Claim description", key="description", height=80)

section("Documents", f"PDF only · max {settings.max_upload_mb} MB each · claim form, hospital bill and discharge "
                     "summary are required")
use_demo_docs = st.toggle("Generate matching demo documents instead of uploading", key="demo_docs",
                          help="Creates fictional PDFs from the form values (useful for a live demo).")
uploads = []
if not use_demo_docs:
    uploads = st.file_uploader("Upload claim documents", type=["pdf"], accept_multiple_files=True) or []
    for i, f in enumerate(uploads):
        cols = st.columns([3, 2, 1])
        cols[0].markdown(f"**{f.name}**")
        cols[1].selectbox("Document type", DOC_TYPES, index=DOC_TYPES.index(_guess_type(f.name)),
                          key=f"doctype_{i}_{f.name}", label_visibility="collapsed")
        cols[2].caption(f"{f.size / 1024:.0f} KB")

process_now = st.checkbox("Process immediately with the AI agents", value=True)
submitted = st.button("Submit claim", type="primary", icon=":material/send:")

if submitted:
    values = {k: st.session_state.get(k) for k in FIELDS}
    values["description"] = values["description"] or None
    errors: list[str] = []
    form = None
    try:
        form = ClaimSubmission(**values)
    except ValidationError as exc:
        for e in exc.errors():
            field = label(str(e["loc"][0])) if e["loc"] else "Form"
            errors.append(f"**{field}**: {e['msg'].removeprefix('Value error, ')}")
    files: list[UploadedFile] = []
    if form and use_demo_docs:
        files = _demo_documents(form)
    elif form:
        for i, f in enumerate(uploads):
            chosen = st.session_state.get(f"doctype_{i}_{f.name}", "Auto-detect")
            files.append(UploadedFile(filename=f.name, content=f.getvalue(),
                                      declared_type=None if chosen == "Auto-detect" else chosen))
        if not files:
            errors.append("**Documents**: upload at least one PDF (or use generated demo documents).")
    if errors:
        st.error("Please correct the following:\n\n" + "\n".join(f"- {e}" for e in errors), icon=":material/error:")
        st.stop()

    with st.status("Submitting claim…", expanded=True) as box:
        try:
            st.write(f"Uploading {len(files)} document(s) and validating…")
            result, via = backend.submit_claim(form, files, process_now)
        except ServiceError as exc:
            box.update(label="Submission failed", state="error")
            st.error(exc.message, icon=":material/error:")
            for d in exc.details or []:
                st.write(f"- {d}")
            st.stop()
        box.update(label=f"Claim {result['claim_number']} submitted ({via})", state="complete")

    st.success(f"Claim **{result['claim_number']}** was created with {len(files)} document(s). "
               f"Request handled {('by the FastAPI backend' if via == 'FastAPI' else 'in-process (API offline)')}.",
               icon=":material/check_circle:")
    for w in result.get("warnings", []):
        st.warning(w, icon=":material/warning:")
    proc = result.get("processing")
    if proc:
        st.markdown(
            f"AI workflow: {status_badge(proc['processing_status'])} &nbsp; Recommendation: "
            f"{status_badge(proc['recommendation'])} &nbsp; {badge('advisory — a claims officer decides', 'neutral')}",
            unsafe_allow_html=True,
        )
        st.caption("Path: " + " → ".join(proc["path"]))
    st.session_state["selected_claim"] = result["claim_number"]
    st.page_link("pages/claim_details.py", label="Open claim details", icon=":material/description:")
