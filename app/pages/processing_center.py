"""AI Processing Center — run the LangGraph workflow and inspect each agent's execution."""

from __future__ import annotations

import json

import streamlit as st

from app.components import backend
from app.components.ui import badge, label, page_header, section, status_badge, workflow_diagram
from app.config import settings
from database.connection import SessionLocal
from database.repository import AgentExecutionRepository, ClaimRepository
from services.claim_service import ServiceError
from workflows.state import AGENT_TITLES

PURPOSES = {
    "intake": "Loads the claim, customer and uploaded documents into the shared workflow state.",
    "document_agent": "Reads the PDFs (OCR fallback for scans) and extracts structured claim fields.",
    "policy_agent": "Checks policy existence, status, holder, coverage period, waiting period, sum insured and exclusions.",
    "rag_agent": "Retrieves cited policy wording from the FAISS index; reports uncertainty instead of guessing.",
    "validation_agent": "Checks documents, required fields, dates, amounts and duplicate invoices.",
    "fraud_agent": "Flags explainable risk indicators for investigation — never accusations.",
    "escalate_high_risk": "Prioritises high-risk claims in the review queue.",
    "decision_agent": "Combines all results into a routing recommendation for a human claims officer.",
    "technical_failure": "Routes the claim to manual review after a technical failure.",
    "human_review_queue": "Saves all results and places the claim in the human review queue.",
}

page_header("AI Processing Center", "Run the six-agent LangGraph workflow and inspect the actual execution of every node.")

with SessionLocal() as s:
    repo = ClaimRepository(s)
    claims = sorted(repo.list(), key=lambda c: (c.processed_at is not None, c.submitted_at), reverse=True)
    numbers = [c.claim_number for c in claims]
    pending = len(repo.pending_processing())
if not numbers:
    st.info("No claims yet.")
    st.stop()

top = st.columns([3, 1.2, 1.2])
wanted = st.session_state.get("selected_claim")
number = top[0].selectbox("Claim", numbers, index=numbers.index(wanted) if wanted in numbers else 0)
st.session_state["selected_claim"] = number
online = backend.api_online()
top[1].markdown("<br>" + (badge("FastAPI online", "success") if online else badge("FastAPI offline · in-process", "warning"))
                + " " + badge(settings.mode_label.split(" (")[0], "info" if not settings.demo_mode else "neutral"),
                unsafe_allow_html=True)
run = top[2].button("Run AI workflow", type="primary", icon=":material/play_arrow:", width="stretch")

if run:
    with st.status(f"Running the LangGraph workflow for {number}…", expanded=True) as box:
        try:
            result, via = backend.process_claim(number)
            box.update(label=f"Workflow finished: {label(result['recommendation'])} ({via})",
                       state="error" if result["processing_status"] == "FAILED" else "complete")
            st.write("Path: " + " → ".join(result["path"]))
        except ServiceError as exc:
            box.update(label="Workflow could not start", state="error")
            st.error(exc.message)

if pending:
    if st.button(f"Process all {pending} pending claim(s)", icon=":material/playlist_play:"):
        with st.spinner("Processing pending claims…"):
            with SessionLocal() as s:
                for c in ClaimRepository(s).pending_processing():
                    backend.process_claim(c.claim_number)
        st.rerun()

with SessionLocal() as s:
    claim = ClaimRepository(s).get_by_number(number)
    steps = [{"node": e.agent_name, "status": e.status, "duration_ms": e.duration_ms, "used_llm": e.used_llm,
              "model": e.model_name, "input": e.input_summary or {}, "output": e.output or {}, "error": e.error_message,
              "sequence": e.sequence, "run_id": e.run_id, "started": e.started_at}
             for e in AgentExecutionRepository(s).latest_run(claim.id)]
    decision = claim.decision_support or {}
    proc_status, processed_at, seconds = claim.processing_status, claim.processed_at, claim.processing_time_seconds

if not steps:
    st.info("This claim has not been processed yet. Click **Run AI workflow** to start the agents.")
    st.stop()

# The queue node records itself in the live state but persists the steps before it.
executed = {s["node"] for s in steps}
if decision:
    steps.append({"node": "human_review_queue", "status": "FAILED" if proc_status == "FAILED" else "SUCCESS",
                  "duration_ms": None, "used_llm": False, "model": None, "input": {},
                  "output": {"queued_as": decision.get("recommendation")}, "error": None,
                  "sequence": len(steps) + 1, "run_id": steps[0]["run_id"], "started": processed_at})
skipped = set()
for s_ in steps:
    if s_["node"] == "policy_agent" and "rag_agent" not in executed and s_["output"].get("next_step") == "validation_agent":
        skipped.add("rag_agent")
    if s_["node"] == "validation_agent" and s_["output"].get("next_step") == "decision_agent":
        skipped.add("fraud_agent")

section("Workflow execution", f"run {steps[0]['run_id'][:8]} · {proc_status.lower()} · "
                              f"{seconds:.2f} s total" if seconds else f"run {steps[0]['run_id'][:8]}")
workflow_diagram(steps, skipped)
c = st.columns(4)
c[0].markdown(f"**Processing**<br>{status_badge(proc_status)}", unsafe_allow_html=True)
c[1].markdown(f"**Current stage**<br>{label(steps[-1]['node'])}", unsafe_allow_html=True)
c[2].markdown(f"**AI recommendation (advisory)**<br>{status_badge(decision.get('recommendation'))}", unsafe_allow_html=True)
c[3].markdown("**Skipped by routing**<br>" + (", ".join(AGENT_TITLES[x] for x in skipped) or "none"),
              unsafe_allow_html=True)

section("Agent details", "inputs, outputs, routing decisions and errors as recorded in the database")
for step in steps:
    title = AGENT_TITLES.get(step["node"], step["node"])
    dur = f" · {step['duration_ms']:.1f} ms" if step["duration_ms"] is not None else ""
    with st.expander(f"{step['sequence']}. {title} — {label(step['status'])}{dur}",
                     expanded=step["status"] == "FAILED"):
        st.caption(PURPOSES.get(step["node"], ""))
        st.markdown(f"{status_badge(step['status'])} &nbsp; "
                    + (badge(f"LLM: {step['model']}", "info") if step["used_llm"] else badge("No LLM call (rules)", "neutral")),
                    unsafe_allow_html=True)
        out = dict(step["output"])
        nxt, why = out.pop("next_step", None), out.pop("route_reason", None)
        if nxt:
            st.markdown(f"**Next:** {AGENT_TITLES.get(nxt, nxt)} — {why}")
        if step["error"]:
            st.error(step["error"])
        a, b = st.columns(2)
        a.markdown("**Input**")
        a.code(json.dumps(step["input"], indent=2, default=str), language="json")
        b.markdown("**Output**")
        b.code(json.dumps(out, indent=2, default=str), language="json")

st.page_link("pages/claim_details.py", label="Open full claim details", icon=":material/description:")
