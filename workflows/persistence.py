"""Save a finished workflow run to the database (claim results, agent executions, risk flags, log)."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from database.models import Claim
from database.repository import (
    AgentExecutionRepository,
    ClaimRepository,
    LogRepository,
    RiskFlagRepository,
)
from workflows.state import ClaimState


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def persist_run(session: Session, state: ClaimState) -> Claim | None:
    claim = session.get(Claim, state.get("claim_id")) if state.get("claim_id") else None
    if claim is None:
        return None
    decision = state.get("decision_support") or {}
    risk = state.get("risk_assessment") or {}
    started, finished = _dt(state.get("started_at")), _dt(state.get("finished_at"))
    seconds = round((finished - started).total_seconds(), 3) if started and finished else None

    ClaimRepository(session).save_agent_results(
        claim,
        extracted_data=state.get("extracted_document_information"),
        policy_verification=state.get("policy_verification"),
        retrieved_clauses=state.get("retrieved_policy_clauses"),
        validation_result=state.get("claim_validation"),
        risk_assessment=state.get("risk_assessment"),
        decision_support=decision or None,
        risk_level=risk.get("risk_level"),
        risk_score=risk.get("risk_score"),
        claim_status=decision.get("recommendation", "REVIEW_REQUIRED"),
        processing_status=state.get("processing_status", "COMPLETED"),
        processed_at=finished,
        processing_time_seconds=seconds,
    )

    # New claims only carry what the claimant typed; keep the invoice number and bill
    # total read from the documents so later duplicate checks can find this claim.
    fields = (state.get("extracted_document_information") or {}).get("fields") or {}
    if not claim.invoice_number and fields.get("invoice_number"):
        claim.invoice_number = fields["invoice_number"]
    if claim.billed_amount is None and fields.get("billed_amount") is not None:
        claim.billed_amount = fields["billed_amount"]

    executions = AgentExecutionRepository(session)
    for rec in state.get("execution_history") or []:
        executions.record(
            claim_id=claim.id, run_id=state["run_id"], agent_name=rec["node"], sequence=rec["sequence"],
            status=rec.get("status", "SUCCESS"), started_at=_dt(rec.get("started_at")),
            finished_at=_dt(rec.get("finished_at")), duration_ms=rec.get("duration_ms"),
            used_llm=bool(rec.get("used_llm")), model_name=rec.get("model_name"),
            input_summary=rec.get("input_summary"),
            output={**(rec.get("output_summary") or {}), "next_step": rec.get("next_step"),
                    "route_reason": rec.get("route_reason")},
            error_message=rec.get("error"),
        )

    RiskFlagRepository(session).replace_for_claim(claim.id, [
        {"code": i["code"], "severity": i["severity"], "description": i["description"], "evidence": i["evidence"]}
        for i in risk.get("indicators", [])
    ])

    failed = state.get("processing_status") == "FAILED"
    LogRepository(session).log(
        "workflow", "PROCESSING_FAILED" if failed else "PROCESSING_COMPLETED",
        f"{claim.claim_number}: {decision.get('recommendation')} "
        f"({len(state.get('execution_history') or [])} steps, {seconds}s)",
        claim_id=claim.id, level="ERROR" if failed else "INFO",
        details={"run_id": state["run_id"], "skipped_agents": state.get("skipped_agents") or [],
                 "errors": state.get("error_information") or []},
    )
    return claim
