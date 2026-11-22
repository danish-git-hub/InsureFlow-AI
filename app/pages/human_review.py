"""Human Review Center — the only place a claim outcome is decided, by a person."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from app.components import backend
from app.components.ui import df_height, inr, label, page_header, recommendation_banner, section, status_badge
from database.connection import SessionLocal
from database.models import ReviewDecision
from database.repository import ClaimRepository, ReviewRepository
from services.claim_service import ROLE_PERMISSIONS, SENIOR_APPROVAL_THRESHOLD, ServiceError
from services.schemas import ReviewSubmission

page_header("Human Review Center", "AI recommendations are advisory. Authorised reviewers record the decisions here.")

with SessionLocal() as s:
    queue = [c for c in ClaimRepository(s).review_queue() if c.decision_support]
    priority = {"POLICY_NOT_FOUND": 0, "REVIEW_REQUIRED": 1, "ADDITIONAL_INFORMATION_REQUIRED": 2,
                "ELIGIBLE_FOR_HUMAN_REVIEW": 3}
    queue.sort(key=lambda c: ({"HIGH": 0, "MEDIUM": 1}.get(c.risk_level or "", 2), priority.get(c.claim_status, 9),
                              c.submitted_at))
    rows = [{"Claim ID": c.claim_number, "Customer": c.customer_name_submitted, "Amount (₹)": c.claim_amount,
             "AI recommendation": label(c.claim_status), "Risk": c.risk_level or "—",
             "Submitted": c.submitted_at.date()} for c in queue]
    recent_reviews = [{"When": r.reviewed_at.strftime("%Y-%m-%d %H:%M"), "Claim": r.claim.claim_number,
                       "Reviewer": r.reviewer_name, "Role": r.reviewer_role,
                       "AI recommended": label(r.ai_recommendation_snapshot), "Human decision": label(r.decision)}
                      for r in ReviewRepository(s).recent(10)]

section("Review queue", f"{len(rows)} claim(s) waiting · high risk first")
if not rows:
    st.success("The review queue is empty.")
else:
    df = pd.DataFrame(rows)
    event = st.dataframe(df, hide_index=True, on_select="rerun", selection_mode="single-row",
                         height=df_height(len(df), 8), column_config={"Amount (₹)": st.column_config.NumberColumn(format="₹%d")})
    pick = event.selection.rows
    default = st.session_state.get("selected_claim")
    if pick:
        number = df.iloc[pick[0]]["Claim ID"]
    elif default in df["Claim ID"].values:
        number = default
    else:
        number = df.iloc[0]["Claim ID"]
    st.session_state["selected_claim"] = number

    with SessionLocal() as s:
        claim = ClaimRepository(s).get_by_number(number)
        d, pv, rag, risk = claim.decision_support, claim.policy_verification, claim.retrieved_clauses, claim.risk_assessment
        amount = claim.claim_amount
        info = (claim.customer_name_submitted, claim.policy_number_submitted, claim.diagnosis, claim.hospital_name,
                claim.admission_date, claim.discharge_date)

    section(f"Claim {number}", f"{info[0]} · {info[1]} · {inr(amount)}")
    ai_col, human_col = st.columns([3, 2], gap="large")

    with ai_col:
        st.markdown("##### AI assessment (advisory)")
        recommendation_banner(d)
        st.write(d["summary"])
        with st.expander("Claim information", expanded=False):
            st.write(f"**Diagnosis:** {info[2]}  \n**Hospital:** {info[3]}  \n**Stay:** {info[4]} → {info[5]}")
        with st.expander("Policy verification", expanded=True):
            st.markdown(status_badge(pv["status"]) + " " + pv["summary"], unsafe_allow_html=True)
            for c in pv["checks"]:
                if c["status"] in ("FAIL", "WARN"):
                    st.markdown(f"{status_badge(c['status'])} **{c['name']}** — {c['message']}", unsafe_allow_html=True)
        with st.expander("Retrieved policy clauses"):
            if not rag:
                st.caption("No policy wording available (policy unknown).")
            else:
                for c in rag["clauses"]:
                    st.markdown(f"**{c['plan_code']} {c['clause_code']} {c['title']}** "
                                f"({c['source_document']} p.{c['page_number']})  \n> {c['text']}")
        with st.expander("Risk indicators"):
            if not risk:
                st.caption("Risk screening deferred (claim incomplete).")
            else:
                st.markdown(f"Score **{risk['risk_score']}/100** {status_badge(risk['risk_level'])}", unsafe_allow_html=True)
                for i in risk["indicators"]:
                    st.markdown(f"- **{label(i['code'])}** ({i['severity']}): {i['description']}")
                st.caption(risk["disclaimer"])
        with st.expander("Required actions", expanded=True):
            for a in d["required_actions"] or ["None"]:
                st.markdown(f"- {a}")

    with human_col:
        st.markdown('<div class="if-human"><b>Human decision</b> · recorded separately from the AI output, '
                    'with your name and role</div>', unsafe_allow_html=True)
        st.write("")
        with st.form(f"review_{number}", clear_on_submit=True):
            name = st.text_input("Reviewer name *")
            role = st.selectbox("Role *", list(ROLE_PERMISSIONS), format_func=label)
            decision = st.radio("Decision *", [x.value for x in ReviewDecision], format_func=label, horizontal=True)
            comments = st.text_area("Comments", placeholder="Required for reject, info request and escalation")
            st.caption(f"Investigators may only request information or escalate. Final outcomes on claims above "
                       f"{inr(SENIOR_APPROVAL_THRESHOLD)} need a senior reviewer.")
            submit = st.form_submit_button("Record decision", type="primary", icon=":material/gavel:")
        if submit:
            try:
                review = ReviewSubmission(reviewer_name=name, reviewer_role=role, decision=decision,
                                          comments=comments or None)
                result, via = backend.record_review(number, review)
                st.success(f"Recorded **{label(decision)}** for {number} → status "
                           f"{label(result['claim_status'])} ({via}).")
                st.session_state.pop("selected_claim", None)
                st.rerun()
            except ValidationError as exc:
                st.error(" ".join(e["msg"].removeprefix("Value error, ") for e in exc.errors()))
            except ServiceError as exc:
                st.error(exc.message)

section("Recent human decisions", "AI recommendation snapshot kept next to the human decision")
if recent_reviews:
    st.dataframe(pd.DataFrame(recent_reviews), hide_index=True)
else:
    st.caption("No decisions recorded yet.")
