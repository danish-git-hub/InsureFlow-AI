"""Reports & Analytics — trends, risk, agent performance and review workload, with CSV export."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components import charts
from app.components.ui import label, page_header, section
from database.connection import SessionLocal
from workflows.state import AGENT_TITLES
from database.repository import AgentExecutionRepository, AnalyticsRepository, ClaimRepository, ReviewRepository

page_header("Reports & Analytics", "Operational analytics from stored claims, agent executions and human reviews.")

with SessionLocal() as s:
    a = AnalyticsRepository(s)
    monthly, by_type, risk, perf, workload = (a.monthly_claims(), a.claims_by_type(), a.risk_distribution(),
                                              a.agent_performance(), a.review_workload())
    claims = pd.DataFrame([{
        "claim_number": c.claim_number, "customer": c.customer_name_submitted, "policy_number": c.policy_number_submitted,
        "insurance_type": c.insurance_type, "claim_amount": c.claim_amount, "submitted_at": c.submitted_at,
        "processed_at": c.processed_at, "processing_seconds": c.processing_time_seconds, "claim_status": c.claim_status,
        "processing_status": c.processing_status, "risk_level": c.risk_level, "risk_score": c.risk_score,
    } for c in ClaimRepository(s).list()])
    executions = pd.DataFrame([{
        "run_id": e.run_id, "claim_id": e.claim_id, "agent": e.agent_name, "sequence": e.sequence, "status": e.status,
        "duration_ms": e.duration_ms, "used_llm": e.used_llm, "model": e.model_name, "started_at": e.started_at,
    } for e in AgentExecutionRepository(s).recent(5000)])
    reviews = pd.DataFrame([{
        "claim_number": r.claim.claim_number, "reviewer": r.reviewer_name, "role": r.reviewer_role,
        "ai_recommendation": r.ai_recommendation_snapshot, "decision": r.decision, "reviewed_at": r.reviewed_at,
    } for r in ReviewRepository(s).recent(5000)])

if claims.empty:
    st.info("No data yet.")
    st.stop()

c1, c2 = st.columns(2)
with c1:
    section("Claim processing trend", "claims processed per day by the AI workflow")
    done = claims.dropna(subset=["processed_at"])
    if done.empty:
        st.caption("No processed claims yet.")
    else:
        daily = done.groupby(done["processed_at"].dt.date).size()
        fig = go.Figure(go.Bar(x=[str(x) for x in daily.index], y=daily.values, marker_color=charts.TEAL,
                               hovertemplate="%{x}: %{y} claims<extra></extra>"))
        fig.update_xaxes(type="category")
        st.plotly_chart(charts.style(fig, 300, legend=False))
with c2:
    section("Monthly claims", "by submission month")
    st.plotly_chart(charts.monthly_trend(monthly, 300))

c1, c2, c3 = st.columns(3)
with c1:
    section("Claim categories")
    st.plotly_chart(charts.donut(by_type, height=300))
with c2:
    section("Risk distribution")
    st.plotly_chart(charts.donut(risk, charts.RISK_COLORS, 300))
with c3:
    section("Review workload", "human decisions recorded")
    if workload:
        st.plotly_chart(charts.hbar({label(k): v for k, v in workload.items()}, charts.NAVY2, 300))
    else:
        queue = claims["claim_status"].isin(["ELIGIBLE_FOR_HUMAN_REVIEW", "REVIEW_REQUIRED",
                                             "ADDITIONAL_INFORMATION_REQUIRED", "POLICY_NOT_FOUND"]).sum()
        st.metric("Claims waiting for review", int(queue))
        st.caption("No human decisions recorded yet.")

c1, c2 = st.columns(2)
with c1:
    section("Agent performance", "average duration and runs per agent")
    if perf:
        dfp = pd.DataFrame(perf).sort_values("avg_ms")
        dfp["agent"] = dfp["agent"].map(lambda a: AGENT_TITLES.get(a, a))
        fig = go.Figure(go.Bar(x=dfp["avg_ms"], y=dfp["agent"], orientation="h", marker_color=charts.TEAL,
                               customdata=dfp[["runs", "failures"]],
                               hovertemplate="%{y}: %{x:.1f} ms avg · %{customdata[0]} runs · %{customdata[1]} failed<extra></extra>"))
        st.plotly_chart(charts.style(fig, 320, legend=False))
        st.dataframe(pd.DataFrame(perf).rename(columns={"agent": "Agent", "runs": "Runs", "avg_ms": "Avg ms",
                                                        "failures": "Failures"}), hide_index=True)
with c2:
    section("Processing time per claim", "seconds, full workflow")
    times = claims["processing_seconds"].dropna()
    if times.empty:
        st.caption("No processing times recorded yet.")
    else:
        fig = go.Figure(go.Histogram(x=times, nbinsx=20, marker_color=charts.BLUE))
        fig.update_xaxes(title="seconds")
        st.plotly_chart(charts.style(fig, 320, legend=False))
        st.caption(f"Median {times.median():.2f} s · max {times.max():.2f} s over {len(times)} claims")

section("Export data", "CSV files straight from the database")
e1, e2, e3 = st.columns(3)
e1.download_button("Claims CSV", claims.to_csv(index=False).encode(), "insureflow_claims.csv", "text/csv",
                   icon=":material/download:", width="stretch")
e2.download_button("Agent executions CSV", executions.to_csv(index=False).encode(), "insureflow_agent_executions.csv",
                   "text/csv", icon=":material/download:", width="stretch", disabled=executions.empty)
e3.download_button("Human reviews CSV", reviews.to_csv(index=False).encode(), "insureflow_reviews.csv", "text/csv",
                   icon=":material/download:", width="stretch", disabled=reviews.empty)
