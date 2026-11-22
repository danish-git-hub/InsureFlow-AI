"""Claim Details — everything known about one claim, from customer to agent history."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components import charts
from app.components.ui import (
    badge,
    df_height,
    inr,
    key_values,
    label,
    page_header,
    recommendation_banner,
    section,
    status_badge,
)
from app.config import PROJECT_ROOT
from database.connection import SessionLocal
from database.repository import AgentExecutionRepository, ClaimRepository, ReviewRepository

page_header("Claim Details", "Customer, policy, documents and every agent's output for a single claim.")

with SessionLocal() as s:
    numbers = [c.claim_number for c in sorted(ClaimRepository(s).list(), key=lambda c: c.submitted_at, reverse=True)]
if not numbers:
    st.info("No claims yet.")
    st.stop()

wanted = st.query_params.get("claim") or st.session_state.get("selected_claim")
number = st.selectbox("Claim", numbers, index=numbers.index(wanted) if wanted in numbers else 0)
st.session_state["selected_claim"] = number
st.query_params["claim"] = number

with SessionLocal() as s:
    claim = ClaimRepository(s).get_by_number(number)
    customer, policy = claim.customer, claim.policy
    cust = {"Customer ID": customer.customer_code, "Name": customer.full_name, "Date of birth": customer.date_of_birth,
            "Email": customer.email, "Phone": customer.phone, "City": f"{customer.city}, {customer.state}"}
    pol = ({"Policy number": policy.policy_number, "Plan": f"{policy.plan_name} ({policy.plan_code})",
            "Type": policy.insurance_type, "Sum insured": inr(policy.sum_insured),
            "Term": f"{policy.start_date} → {policy.end_date}", "Waiting period": f"{policy.waiting_period_days} days",
            "Status": policy.status} if policy else None)
    facts = {"Claim ID": claim.claim_number, "Submitted": claim.submitted_at.strftime("%Y-%m-%d %H:%M"),
             "Insurance type": claim.insurance_type, "Claim amount": inr(claim.claim_amount),
             "Hospital": claim.hospital_name, "Admission": claim.admission_date, "Discharge": claim.discharge_date,
             "Diagnosis": claim.diagnosis, "Invoice": claim.invoice_number, "Description": claim.description}
    docs = [{"Type": label(d.document_type), "File": d.original_filename, "Pages": d.page_count,
             "Size (KB)": round(d.file_size_bytes / 1024, 1), "SHA-256": d.sha256[:16] + "…", "path": d.file_path}
            for d in claim.documents]
    results = {k: getattr(claim, k) for k in ("extracted_data", "policy_verification", "retrieved_clauses",
                                              "validation_result", "risk_assessment", "decision_support")}
    status = (claim.claim_status, claim.processing_status, claim.risk_level, claim.processed_at)
    executions = [{"#": e.sequence, "Agent": e.agent_name, "Status": e.status, "Duration (ms)": e.duration_ms,
                   "LLM used": "Gemini" if e.used_llm else "No (rules)", "Next step": (e.output or {}).get("next_step"),
                   "Route reason": (e.output or {}).get("route_reason"), "Error": e.error_message}
                  for e in AgentExecutionRepository(s).latest_run(claim.id)]
    reviews = [{"When": r.reviewed_at.strftime("%Y-%m-%d %H:%M"), "Reviewer": r.reviewer_name, "Role": r.reviewer_role,
                "AI recommendation (snapshot)": label(r.ai_recommendation_snapshot), "Human decision": label(r.decision),
                "Comments": r.comments} for r in ReviewRepository(s).for_claim(claim.id)]

st.markdown(" ".join([status_badge(status[0]), status_badge(status[1]),
                      badge(f"Risk {status[2]}", {"LOW": "success", "MEDIUM": "warning", "HIGH": "danger"}
                            .get(status[2] or "", "neutral")) if status[2] else ""]), unsafe_allow_html=True)
st.write("")
recommendation_banner(results["decision_support"])

tabs = st.tabs(["Overview", "Documents", "Extraction", "Policy verification", "Policy clauses", "Validation",
                "Risk indicators", "Decision support", "Agent history", "Human reviews"])

with tabs[0]:
    c1, c2, c3 = st.columns(3)
    with c1:
        section("Claim")
        key_values(facts)
    with c2:
        section("Customer")
        key_values(cust)
    with c3:
        section("Policy")
        if pol:
            key_values(pol)
        else:
            st.error(f"Policy {claim.policy_number_submitted} is not on record.", icon=":material/policy:")

with tabs[1]:
    if not docs:
        st.info("No documents were uploaded with this claim.")
    for d in docs:
        c = st.columns([2, 3, 1, 1, 1.2])
        c[0].markdown(f"**{d['Type']}**")
        c[1].write(d["File"])
        c[2].write(f"{d['Pages']} page(s)")
        c[3].write(f"{d['Size (KB)']} KB")
        path = PROJECT_ROOT / d["path"] if not d["path"].startswith("/") else d["path"]
        try:
            c[4].download_button("Download", open(path, "rb").read(), d["File"], "application/pdf",
                                 key=f"dl_{d['File']}", icon=":material/download:")
        except OSError:
            c[4].caption("file missing")

with tabs[2]:
    ex = results["extracted_data"]
    if not ex:
        st.info("Documents have not been processed yet.")
    else:
        st.caption(f"Mode: **{ex['extraction_mode']}** · OCR used: **{ex['ocr_used']}** · LLM used: "
                   f"**{'Gemini (' + str(ex['llm_model']) + ')' if ex['used_llm'] else 'No'}**")
        rows = [{"Field": label(f), "Value": "—" if v is None else str(v), "Source": ex["field_sources"].get(f, "—"),
                 "From document": ex["field_documents"].get(f, "—")} for f, v in ex["fields"].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, height=df_height(len(rows)))
        if ex["missing_fields"]:
            st.warning("Missing fields: " + ", ".join(label(f) for f in ex["missing_fields"]))
        for c in ex["conflicts"]:
            st.warning(f"Documents disagree on **{label(c['field'])}**: "
                       + "; ".join(f"{k}: {v}" for k, v in c["values"].items()))
        with st.expander("Per-document details"):
            st.dataframe(pd.DataFrame([{"File": d["filename"], "Detected type": d["document_type"],
                                        "Method": d["extraction_method"], "OCR confidence": d["ocr_confidence"],
                                        "Characters": d["characters"], "Warnings": "; ".join(d["warnings"])}
                                       for d in ex["documents"]]), hide_index=True)

with tabs[3]:
    pv = results["policy_verification"]
    if not pv:
        st.info("Not processed yet.")
    else:
        st.markdown(f"{status_badge(pv['status'])} &nbsp; {pv['summary']}", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{"Rule": c["name"], "Result": c["status"], "Explanation": c["message"]}
                                   for c in pv["checks"]]), hide_index=True, height=df_height(len(pv["checks"])),
                     column_config={"Explanation": st.column_config.TextColumn(width="large")})
        if pv.get("coverage"):
            cov = pv["coverage"]
            key_values({"Sum insured": inr(cov["sum_insured"]), "Other claims this term": inr(cov["other_claims_on_policy"]),
                        "Remaining sum insured": inr(cov["remaining_sum_insured"]),
                        "Co-payment": f"{cov['copay_percent']:g}%",
                        "Indicative admissible amount": inr(cov["indicative_admissible_amount"]) + " (indicative only)"})
        for sgt in pv.get("suggestions", []):
            st.info(f"Customer's policies on record: {sgt}")

with tabs[4]:
    rag = results["retrieved_clauses"]
    if not rag:
        st.info("Policy RAG did not run for this claim (not processed, or skipped because the policy is unknown).")
    else:
        st.markdown(f"{status_badge(rag['status'])} &nbsp; Embeddings: `{rag['embedding_model']}` · retrieval "
                    f"{rag['retrieval_mode']} · explanation by **{rag['explanation_source']}**", unsafe_allow_html=True)
        st.info(rag["explanation"], icon=":material/menu_book:")
        for reason in rag["uncertainty_reasons"]:
            st.warning(reason, icon=":material/help:")
        for c in rag["clauses"]:
            excl = " · possible exclusion" if c["matched_terms"] else ""
            with st.expander(f"{c['plan_code']} {c['clause_code']} — {c['title']}  ({label(c['topic'])}{excl})"):
                st.markdown(f"> {c['text']}")
                st.caption(f"Source: {c['source_document']}, page {c['page_number']} · similarity {c['score']:.3f}"
                           + (f" · diagnosis terms found: {', '.join(c['matched_terms'])}" if c["matched_terms"] else ""))

with tabs[5]:
    v = results["validation_result"]
    if not v:
        st.info("Not processed yet.")
    else:
        st.markdown(f"{status_badge(v['status'])} &nbsp; {v['summary']}", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{"Check": c["name"], "Result": c["status"], "Explanation": c["message"]}
                                   for c in v["checks"]]), hide_index=True, height=df_height(len(v["checks"])),
                     column_config={"Explanation": st.column_config.TextColumn(width="large")})

with tabs[6]:
    r = results["risk_assessment"]
    if not r:
        st.info("Risk screening has not run (claim not processed, or deferred because it was incomplete).")
    else:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.plotly_chart(charts.gauge(r["risk_score"]))
            st.markdown(status_badge(r["risk_level"]), unsafe_allow_html=True)
        with c2:
            st.write(r["summary"])
            if r["indicators"]:
                st.dataframe(pd.DataFrame([{"Indicator": label(i["code"]), "Severity": i["severity"],
                                            "Points": i["points"], "What was observed": i["description"]}
                                           for i in r["indicators"]]), hide_index=True)
            st.caption(r["disclaimer"])
            st.caption("Not used in scoring: " + ", ".join(r["features_not_used"]))

with tabs[7]:
    d = results["decision_support"]
    if not d:
        st.info("Not processed yet.")
    else:
        st.write(d["summary"])
        c1, c2 = st.columns(2)
        with c1:
            section("Reasons")
            for reason in d["reasons"]:
                icon = "🔸" if reason["blocking"] else "▫️"
                st.markdown(f"{icon} **{label(reason['code'])}** ({reason['source']}): {reason['message']}")
        with c2:
            section("Required actions")
            for a in d["required_actions"] or ["None"]:
                st.markdown(f"- {a}")
            section("Reviewer checklist")
            for item in d["reviewer_checklist"]:
                st.checkbox(item, key=f"chk_{number}_{item[:30]}")
        st.caption(f"Evidence confidence {d['confidence']:.0%}: " + " ".join(d["confidence_notes"]))
        for w in d.get("warnings", []):
            st.warning(w)
        st.caption(d["disclaimer"])

with tabs[8]:
    if executions:
        st.dataframe(pd.DataFrame(executions), hide_index=True, height=df_height(len(executions)),
                     column_config={"Route reason": st.column_config.TextColumn(width="large")})
        st.page_link("pages/processing_center.py", label="Open in the AI Processing Center", icon=":material/hub:")
    else:
        st.info("No agent executions recorded for this claim yet.")

with tabs[9]:
    if reviews:
        st.dataframe(pd.DataFrame(reviews), hide_index=True)
    else:
        st.info("No human review recorded yet.")
    st.page_link("pages/human_review.py", label="Go to the Human Review Center", icon=":material/fact_check:")
