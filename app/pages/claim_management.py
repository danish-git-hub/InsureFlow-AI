"""Claim Management — search, filter and open claims."""

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from app.components.ui import df_height, label, page_header
from database.connection import SessionLocal
from database.models import ClaimStatus, InsuranceType, RiskLevel
from database.repository import ClaimRepository

page_header("Claim Management", "Search and filter every claim; select a row to open its details.")

f1, f2, f3, f4 = st.columns([2.2, 1.6, 1.4, 1.2])
search = f1.text_input("Search", placeholder="Claim ID, customer, policy, hospital or diagnosis")
statuses = f2.multiselect("Claim status", [s.value for s in ClaimStatus], format_func=label)
ins_type = f3.selectbox("Insurance type", ["All"] + [t.value for t in InsuranceType])
risk = f4.selectbox("Risk level", ["All"] + [r.value for r in RiskLevel])
d1, d2, _ = st.columns([1, 1, 3])
date_from = d1.date_input("Submitted from", value=date.today() - timedelta(days=365))
date_to = d2.date_input("Submitted to", value=date.today())

with SessionLocal() as s:
    claims = ClaimRepository(s).list(status=statuses or None, insurance_type=None if ins_type == "All" else ins_type,
                                     risk_level=None if risk == "All" else risk, search=search or None,
                                     date_from=date_from, date_to=date_to)
    rows = [{
        "Claim ID": c.claim_number, "Customer": c.customer_name_submitted, "Policy": c.policy_number_submitted,
        "Type": c.insurance_type, "Amount (₹)": c.claim_amount, "Claim date": c.submitted_at.date(),
        "Claim status": label(c.claim_status), "Risk": c.risk_level or "—",
        "Risk score": c.risk_score, "Processing": label(c.processing_status),
    } for c in sorted(claims, key=lambda c: c.submitted_at, reverse=True)]

df = pd.DataFrame(rows)
st.caption(f"{len(df)} claim(s) match the filters.")
if df.empty:
    st.info("No claims match these filters.")
    st.stop()

event = st.dataframe(
    df, hide_index=True, on_select="rerun", selection_mode="single-row", height=df_height(len(df), 16),
    column_config={
        "Amount (₹)": st.column_config.NumberColumn(format="₹%d"),
        "Risk score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
    },
)
selected = event.selection.rows
c1, c2, _ = st.columns([1.3, 1.3, 4])
if selected:
    number = df.iloc[selected[0]]["Claim ID"]
    st.session_state["selected_claim"] = number
    if c1.button(f"Open {number}", type="primary", icon=":material/open_in_new:"):
        st.switch_page("pages/claim_details.py")
else:
    c1.caption("Select a row to open the claim.")
c2.download_button("Export CSV", df.to_csv(index=False).encode(), "claims.csv", "text/csv", icon=":material/download:")
