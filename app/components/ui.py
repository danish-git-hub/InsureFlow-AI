"""Reusable, branded UI building blocks for the Streamlit dashboard.

Keeping these helpers in one module means every page looks consistent and a
colour change only has to happen in `app/styles/theme.css`.
"""

from __future__ import annotations

import html
from pathlib import Path

import streamlit as st

from app.config import settings

_CSS_PATH = Path(__file__).resolve().parent.parent / "styles" / "theme.css"


def load_theme() -> None:
    """Inject the custom CSS once per page render."""
    st.markdown(f"<style>{_CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def sidebar_brand() -> None:
    """Brand block + current AI mode indicator at the top of the sidebar."""
    mode_cls = "demo" if settings.demo_mode else "live"
    with st.sidebar:
        st.markdown(
            f"""
            <div class="if-brand">
              <div class="if-brand-logo">IF</div>
              <div>
                <div class="if-brand-name">{settings.app_name}</div>
                <div class="if-brand-tag">{settings.tagline}</div>
              </div>
            </div>
            <div class="if-mode {mode_cls}"><b>AI mode:</b> {html.escape(settings.mode_label)}</div>
            """,
            unsafe_allow_html=True,
        )


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div class="if-header"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f"""<div class="if-card if-kpi">
              <div class="if-kpi-label">{html.escape(label)}</div>
              <div class="if-kpi-value">{html.escape(value)}</div>
              <div class="if-kpi-sub">{html.escape(sub)}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def info_card(title: str, body: str) -> None:
    st.markdown(
        f'<div class="if-card"><h4>{html.escape(title)}</h4><p>{html.escape(body)}</p></div>',
        unsafe_allow_html=True,
    )


def badge(text: str, tone: str = "neutral") -> str:
    """Return HTML for a coloured pill. tone: success|warning|danger|info|neutral."""
    return f'<span class="if-badge {tone}">{html.escape(text)}</span>'


def coming_soon(phase: int, what: str) -> None:
    """Honest placeholder for pages not built yet (no fake data)."""
    st.info(f"**Planned for Phase {phase}.** {what}", icon=":material/construction:")


# --------------------------------------------------------------------------- #
# Status helpers shared by the claim pages
# --------------------------------------------------------------------------- #
STATUS_TONE = {
    "ELIGIBLE_FOR_HUMAN_REVIEW": "success", "REVIEW_REQUIRED": "warning", "ADDITIONAL_INFORMATION_REQUIRED": "info",
    "POLICY_NOT_FOUND": "danger", "SUBMITTED": "neutral", "HUMAN_APPROVED": "success", "HUMAN_REJECTED": "danger",
    "HUMAN_INFO_REQUESTED": "info", "COMPLETED": "success", "RUNNING": "info", "PENDING": "neutral",
    "FAILED": "danger", "LOW": "success", "MEDIUM": "warning", "HIGH": "danger", "SUCCESS": "success",
    "WARNING": "warning", "SKIPPED": "neutral", "PASS": "success", "WARN": "warning", "FAIL": "danger",
    "NOT_EVALUATED": "neutral", "NOT_APPLICABLE": "neutral", "VERIFIED": "success",
    "VERIFIED_WITH_WARNINGS": "warning", "INSUFFICIENT_DATA": "warning", "EVIDENCE_FOUND": "success",
    "INSUFFICIENT_EVIDENCE": "warning", "POLICY_CONTEXT_UNAVAILABLE": "danger", "VALID": "success",
    "VALID_WITH_WARNINGS": "warning", "INCOMPLETE": "info", "INVALID": "danger", "ACTIVE": "success",
    "EXPIRED": "neutral", "LAPSED": "warning", "CANCELLED": "danger",
}


def label(code: str | None) -> str:
    return (code or "—").replace("_", " ").title()


def status_badge(code: str | None) -> str:
    return badge(label(code), STATUS_TONE.get(code or "", "neutral"))


def inr(amount: float | None) -> str:
    if amount is None:
        return "—"
    if amount >= 10_000_000:
        return f"₹{amount / 10_000_000:.2f} Cr"
    if amount >= 100_000:
        return f"₹{amount / 100_000:.2f} L"
    return f"₹{amount:,.0f}"


def section(title: str, caption: str = "") -> None:
    st.markdown(f'<div class="if-section"><h3>{html.escape(title)}</h3>'
                f'<span>{html.escape(caption)}</span></div>', unsafe_allow_html=True)


def recommendation_banner(decision: dict | None) -> None:
    """The AI recommendation, always labelled as advisory."""
    if not decision:
        st.info("This claim has not been processed by the AI agents yet.", icon=":material/hourglass_empty:")
        return
    rec = decision.get("recommendation")
    tone = STATUS_TONE.get(rec, "neutral")
    st.markdown(
        f"""<div class="if-rec {tone}">
              <div class="if-rec-label">AI recommendation · advisory only</div>
              <div class="if-rec-value">{html.escape(label(rec))}</div>
              <div class="if-rec-sub">Evidence confidence {decision.get('confidence', 0):.0%} ·
                summary by {html.escape(decision.get('summary_source', 'template'))} ·
                final decision by an authorised claims officer</div>
            </div>""",
        unsafe_allow_html=True,
    )


def workflow_diagram(steps: list[dict], skipped: set[str] | None = None) -> None:
    """Horizontal pipeline of the LangGraph run: one card per node, coloured by status."""
    from workflows.state import AGENT_TITLES

    order = ["intake", "document_agent", "policy_agent", "rag_agent", "validation_agent", "fraud_agent",
             "escalate_high_risk", "decision_agent", "technical_failure", "human_review_queue"]
    by_node = {s["node"]: s for s in steps}
    skipped = skipped or set()
    cards = []
    for node in order:
        step = by_node.get(node)
        if step is None and node in ("escalate_high_risk", "technical_failure"):
            continue  # optional branches: only shown when taken
        status = step["status"] if step else ("SKIPPED" if node in skipped else "NOT_RUN")
        ms = f"{step['duration_ms']:.0f} ms" if step and step.get("duration_ms") is not None else ""
        llm = " · Gemini" if step and step.get("used_llm") else ""
        cards.append(f'<div class="if-step {status.lower()}"><div class="if-step-title">'
                     f'{html.escape(AGENT_TITLES.get(node, node))}</div><div class="if-step-meta">'
                     f'{html.escape(label(status))}{" · " + ms if ms else ""}{llm}</div></div>')
    st.markdown('<div class="if-flow">' + '<div class="if-arrow">›</div>'.join(cards) + "</div>",
                unsafe_allow_html=True)


def df_height(rows: int, max_rows: int = 15) -> int:
    return 38 + 35 * min(max(rows, 1), max_rows)


def key_values(pairs: dict) -> None:
    rows = "".join(f"<div>{html.escape(str(k))}</div><div>{html.escape(str(v) if v not in (None, '') else '—')}</div>"
                   for k, v in pairs.items())
    st.markdown(f'<div class="if-card"><div class="if-kv">{rows}</div></div>', unsafe_allow_html=True)
