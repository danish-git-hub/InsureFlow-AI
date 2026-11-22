"""Executive Dashboard — every number on this page is queried from the database."""

import pandas as pd
import streamlit as st

from app.components import charts
from app.components.ui import df_height, inr, kpi_card, label, page_header, section
from database.connection import SessionLocal
from database.repository import AnalyticsRepository, ClaimRepository, LogRepository
from workflows.state import AGENT_ORDER, AGENT_TITLES

page_header("Executive Dashboard", "Portfolio KPIs and trends, computed live from the claims database.")

with SessionLocal() as s:
    analytics = AnalyticsRepository(s)
    k = analytics.kpis()
    monthly = analytics.monthly_claims()
    status = analytics.status_distribution()
    by_type = analytics.claims_by_type()
    agents = analytics.agent_performance()
    recent = [
        {"Claim ID": c.claim_number, "Customer": c.customer_name_submitted, "Type": c.insurance_type,
         "Amount (₹)": c.claim_amount, "Submitted": c.submitted_at.date(), "Status": label(c.claim_status),
         "Risk": c.risk_level or "—"}
        for c in sorted(ClaimRepository(s).list(), key=lambda c: c.submitted_at, reverse=True)[:10]
    ]
    activity = [
        {"Time": l.created_at.strftime("%Y-%m-%d %H:%M"), "Source": l.source, "Event": label(l.event),
         "Message": l.message, "Level": l.level}
        for l in LogRepository(s).recent(10)
    ]

if not k["total_claims"]:
    st.warning("The database has no claims yet. Run `python -m database.init_db` to load the synthetic data.")
    st.stop()

cols = st.columns(6)
avg = k["avg_processing_seconds"]
cards = [
    ("Total Claims", f"{k['total_claims']}", f"{k['pending_processing']} awaiting AI processing"),
    ("Under Review", f"{k['claims_under_review']}", "in the human review queue"),
    ("Claims Processed", f"{k['claims_processed']}", "by the AI workflow"),
    ("Info Required", f"{k['additional_info_required']}", "waiting for the claimant"),
    ("Total Claim Amount", inr(k["total_claim_amount"]), "across all claims"),
    ("Avg Processing Time", f"{avg:.2f} s" if avg is not None else "—", "per claim, all six agents"),
]
for col, (title, value, sub) in zip(cols, cards):
    with col:
        kpi_card(title, value, sub)

left, right = st.columns([3, 2])
with left:
    section("Monthly claims trend", "count and amount by submission month")
    st.plotly_chart(charts.monthly_trend(monthly))
with right:
    section("Claim status distribution", "current status of every claim")
    st.plotly_chart(charts.donut(status, charts.STATUS_COLORS))

c1, c2, c3 = st.columns(3)
with c1:
    section("Claims by insurance type")
    st.plotly_chart(charts.hbar(by_type, charts.NAVY2, 300))
with c2:
    section("Monthly claim amount", "INR")
    st.plotly_chart(charts.amount_area(monthly, 300))
with c3:
    section("Agent processing performance", "average ms per run")
    if agents:
        perf = {AGENT_TITLES.get(a["agent"], a["agent"]): round(a["avg_ms"], 1) for a in agents
                if a["agent"] in AGENT_ORDER}
        st.plotly_chart(charts.hbar(perf, charts.TEAL, 300, "%{x:.1f}"))
    else:
        st.info("No agent executions recorded yet.")

left, right = st.columns([3, 2])
with left:
    section("Recent claims")
    df = pd.DataFrame(recent)
    st.dataframe(df, hide_index=True, height=df_height(len(df), 10),
                 column_config={"Amount (₹)": st.column_config.NumberColumn(format="₹%d")})
with right:
    section("Latest processing activity")
    st.dataframe(pd.DataFrame(activity), hide_index=True, height=df_height(len(activity), 10),
                 column_config={"Message": st.column_config.TextColumn(width="large")})
