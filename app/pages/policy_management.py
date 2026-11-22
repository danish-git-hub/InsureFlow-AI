"""Policy Management — policy records, synthetic policy administration and the RAG document library."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from app.components import backend
from app.components.ui import df_height, inr, label, page_header, section
from database.catalog import PLANS
from database.connection import SessionLocal
from database.models import InsuranceType, PolicyStatus
from database.repository import CustomerRepository, LogRepository, PolicyClauseRepository, PolicyRepository
from services.claim_service import ServiceError

page_header("Policy Management", "Synthetic policy records and the policy wording indexed for retrieval (RAG).")

tab_records, tab_new, tab_rag = st.tabs(["Policy records", "Add synthetic policy", "Policy documents (RAG)"])

with tab_records:
    f1, f2, f3 = st.columns([2, 1.3, 1.3])
    search = f1.text_input("Search", placeholder="Policy number or customer name")
    status = f2.selectbox("Status", ["All"] + [s.value for s in PolicyStatus])
    ptype = f3.selectbox("Insurance type", ["All"] + [t.value for t in InsuranceType])
    with SessionLocal() as s:
        policies = PolicyRepository(s).list(status=None if status == "All" else status,
                                            insurance_type=None if ptype == "All" else ptype, search=search or None)
        rows = [{"Policy number": p.policy_number, "Customer": p.customer.full_name, "Customer ID": p.customer.customer_code,
                 "Insurance type": p.insurance_type, "Plan": p.plan_code, "Coverage (₹)": p.sum_insured,
                 "Start date": p.start_date, "End date": p.end_date, "Waiting (days)": p.waiting_period_days,
                 "Status": p.status} for p in policies]
    df = pd.DataFrame(rows)
    st.caption(f"{len(df)} policies")
    st.dataframe(df, hide_index=True, height=df_height(len(df), 14),
                 column_config={"Coverage (₹)": st.column_config.NumberColumn(format="₹%d")})
    if not df.empty:
        st.download_button("Export CSV", df.to_csv(index=False).encode(), "policies.csv", "text/csv",
                           icon=":material/download:")

        section("Change policy status", "synthetic administration — logged in the processing log")
        c1, c2, c3 = st.columns([2, 1.3, 1])
        target = c1.selectbox("Policy", df["Policy number"].tolist())
        new_status = c2.selectbox("New status", [s.value for s in PolicyStatus], format_func=label)
        if c3.button("Update status", icon=":material/edit:"):
            with backend.db_session() as s:
                pol = PolicyRepository(s).get_by_number(target)
                old = pol.status
                PolicyRepository(s).update_status(target, new_status)
                LogRepository(s).log("policy_management", "POLICY_STATUS_CHANGED", f"{target}: {old} → {new_status}")
            st.success(f"{target}: {label(old)} → {label(new_status)}")
            st.rerun()

with tab_new:
    with SessionLocal() as s:
        customers = {f"{c.customer_code} · {c.full_name}": c.customer_code for c in CustomerRepository(s).list()}
    with st.form("new_policy"):
        c1, c2 = st.columns(2)
        who = c1.selectbox("Customer", list(customers))
        plan_code = c2.selectbox("Plan", list(PLANS), format_func=lambda c: f"{PLANS[c].name} ({c})")
        c1, c2, c3 = st.columns(3)
        si = c1.selectbox("Sum insured (₹)", sorted({x for p in PLANS.values() for x in p.sum_insured_options}),
                          format_func=lambda x: f"₹{x:,}")
        start = c2.date_input("Start date", value=date.today())
        waiting = c3.number_input("Waiting period (days)", min_value=0, max_value=1460,
                                  value=PLANS[plan_code].default_waiting_days)
        ok = st.form_submit_button("Create policy", type="primary", icon=":material/add:")
    if ok:
        plan = PLANS[plan_code]
        try:
            with backend.db_session() as s:
                cust = CustomerRepository(s).get_by_code(customers[who])
                repo = PolicyRepository(s)
                seq = repo.count() + 1
                number = f"{plan_code}-{start:%Y}-{seq:06d}"
                while repo.get_by_number(number):
                    seq += 1
                    number = f"{plan_code}-{start:%Y}-{seq:06d}"
                repo.create(policy_number=number, customer_id=cust.id, insurance_type=plan.insurance_type.value,
                            plan_name=plan.name, plan_code=plan.code, sum_insured=si,
                            annual_premium=round(si * plan.premium_rate, 2), start_date=start,
                            end_date=start + timedelta(days=364), waiting_period_days=int(waiting),
                            status=PolicyStatus.ACTIVE.value, hospitalization_covered=True,
                            exclusions=[e.title for e in plan.exclusions], policy_document=plan.document_filename)
                LogRepository(s).log("policy_management", "POLICY_CREATED", f"{number} created for {cust.customer_code}")
            st.success(f"Policy **{number}** created ({plan.name}, sum insured {inr(si)}).")
        except Exception as exc:  # constraint violations etc.
            st.error(f"Could not create the policy: {exc}")

with tab_rag:
    with SessionLocal() as s:
        clauses = PolicyClauseRepository(s).list()
        lib = pd.DataFrame([{"Document": c.source_document, "Plan": c.plan_code, "Category": c.category}
                            for c in clauses])
    if lib.empty:
        st.warning("No policy wording is indexed yet. Run `python -m rag.retriever` or upload a policy PDF below.")
    else:
        summary = (lib.groupby(["Document", "Plan"]).size().reset_index(name="Clauses"))
        section("Indexed policy documents", f"{len(lib)} clauses in the FAISS index")
        st.dataframe(summary, hide_index=True)

        section("Try a retrieval query", "the same FAISS search the Policy RAG Agent uses")
        q1, q2 = st.columns([3, 1])
        query = q1.text_input("Query", placeholder="e.g. cosmetic surgery, waiting period, co-payment")
        plan = q2.selectbox("Plan", ["All"] + sorted(lib["Plan"].dropna().unique().tolist()))
        if query:
            from rag.retriever import get_policy_index

            for hit in get_policy_index().search(query, k=4, plan_code=None if plan == "All" else plan):
                c = hit.chunk
                with st.expander(f"{hit.score:.3f} · {c.plan_code} {c.clause_code or ''} {c.title or ''}"):
                    st.markdown(f"> {c.text}")
                    st.caption(f"{c.source_document}, page {c.page_number} · {label(c.category)}")

    section("Upload a policy PDF", "the index is rebuilt and clauses are synced to the database")
    up = st.file_uploader("Policy wording PDF", type=["pdf"], key="policy_pdf")
    if up and st.button("Index document", type="primary", icon=":material/upload_file:"):
        try:
            with st.spinner("Indexing…"):
                result, via = backend.upload_policy_document(up.name, up.getvalue())
            st.success(f"Indexed **{result['filename']}**: {result['chunks']} chunk(s) "
                       f"({'clause-structured' if result['structured'] else 'plain text, category GENERAL'}) via {via}.")
        except ServiceError as exc:
            st.error(exc.message)
