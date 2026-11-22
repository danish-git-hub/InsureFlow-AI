"""
LangGraph nodes (one per agent) and the routing decisions between them.

Every node is wrapped by `_wrap`, which adds the cross-cutting concerns once:

    * timing and an `ExecutionRecord` appended to `execution_history`,
    * honest LLM reporting (`used_llm`, `model_name`) taken from the agent result,
    * error handling: an exception never crashes the graph — it is written to
      `error_information`, the status becomes FAILED, and the router sends the
      claim to the technical-failure handler (and from there to a human),
    * the routing decision taken *after* the node, with a plain-English reason,
      so the dashboard can show why the workflow went where it went.

Routers (`route_after_*`) are pure functions of the state, so they are easy to
unit-test and LangGraph can draw them as conditional edges.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from agents.decision_agent import DecisionSupportAgent
from agents.document_agent import DocumentExtractionAgent, DocumentInput
from agents.fraud_agent import FraudRiskAgent
from agents.policy_agent import PolicyVerificationAgent, build_input
from agents.rag_agent import PolicyRAGAgent, build_rag_input
from agents.schemas import (
    DecisionReason,
    DecisionSupport,
    DocumentExtractionResult,
    PolicyRAGResult,
    PolicyVerificationResult,
    RiskAssessment,
    ValidationResult,
)
from agents.validation_agent import ValidationAgent, build_validation_input
from app.config import PROJECT_ROOT, get_logger
from database.models import Claim
from workflows.state import AGENT_TITLES, ClaimState, ExecutionRecord

log = get_logger("workflow")

Route = tuple[str, str, list[str]]  # (next node, reason, agents skipped by this route)


@dataclass
class WorkflowContext:
    """Dependencies shared by all nodes (injected, so tests can use temp databases and fakes)."""

    session: Session
    rag_index: Any = None              # PolicyIndex; None -> process-wide default index
    use_llm: bool | None = None        # None -> Gemini only if a key is configured
    llm_model: Any = None              # test hook: fake chat model
    today: date | None = None
    persist: bool = True               # write results to the database in the final node
    fail_on: set[str] = field(default_factory=set)  # test hook: nodes that raise a simulated failure
    _agents: dict[str, Any] = field(default_factory=dict)

    def claim(self, state: ClaimState) -> Claim | None:
        cid = state.get("claim_id")
        return self.session.get(Claim, cid) if cid else None

    def rag_agent(self) -> PolicyRAGAgent:
        if "rag" not in self._agents:
            self._agents["rag"] = PolicyRAGAgent(index=self.rag_index, use_llm=self.use_llm, llm_model=self.llm_model)
        return self._agents["rag"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _dump(model) -> dict | None:
    return model.model_dump(mode="json") if model is not None else None


def _failed(state: ClaimState) -> bool:
    return state.get("processing_status") == "FAILED"


# =========================================================================== #
# Routing decisions
# =========================================================================== #
def decide_after_intake(state: ClaimState) -> Route:
    if _failed(state):
        return "technical_failure", "The claim could not be loaded.", []
    n = len(state.get("uploaded_documents") or [])
    return "document_agent", f"Claim loaded with {n} document(s).", []


def decide_after_documents(state: ClaimState) -> Route:
    if _failed(state):
        return "technical_failure", "Technical failure during document extraction.", []
    ex = state.get("extracted_document_information")
    if not ex:
        return "policy_agent", "No readable documents: continue with claim-form values (missing documents).", []
    missing = ex.get("missing_fields") or []
    if missing:
        return "policy_agent", f"Documents read; {len(missing)} field(s) missing: {', '.join(missing)}.", []
    return "policy_agent", "All required fields extracted from the documents.", []


def decide_after_policy(state: ClaimState) -> Route:
    if _failed(state):
        return "technical_failure", "Technical failure during policy verification.", []
    pv = state.get("policy_verification") or {}
    flags = pv.get("flags", {})
    if pv.get("status") == "POLICY_NOT_FOUND":
        return ("validation_agent", "Unknown policy: there is no policy wording to search, so the RAG agent is "
                "skipped.", ["rag_agent"])
    if flags.get("waiting_period_incomplete"):
        return "rag_agent", "Waiting period incomplete: retrieve the waiting-period wording for the reviewer.", []
    if pv.get("status") == "FAILED":
        return "rag_agent", "Policy rule(s) failed: retrieve the relevant wording as evidence.", []
    return "rag_agent", "Policy verified: retrieve the applicable policy wording.", []


def decide_after_rag(state: ClaimState) -> Route:
    if _failed(state):
        return "technical_failure", "Technical failure during policy retrieval.", []
    rag = state.get("retrieved_policy_clauses") or {}
    if rag.get("status") != "EVIDENCE_FOUND":
        return "validation_agent", "Uncertain policy coverage: insufficient wording evidence, flagged for a human.", []
    if rag.get("possible_exclusions"):
        return "validation_agent", "Exclusion wording mentions the diagnosis: flagged for a human.", []
    return "validation_agent", "Policy wording retrieved with citations.", []


def decide_after_validation(state: ClaimState) -> Route:
    if _failed(state):
        return "technical_failure", "Technical failure during claim validation.", []
    flags = (state.get("claim_validation") or {}).get("flags", {})
    if flags.get("missing_documents") or flags.get("missing_fields"):
        return ("decision_agent", "Missing documents or fields: risk screening is deferred until the claim is "
                "complete.", ["fraud_agent"])
    if flags.get("invalid_dates"):
        return ("decision_agent", "Invalid claim information (dates): risk screening is deferred until corrected.",
                ["fraud_agent"])
    return "fraud_agent", "Claim is complete: run risk screening.", []


def decide_after_fraud(state: ClaimState) -> Route:
    if _failed(state):
        return "technical_failure", "Technical failure during risk screening.", []
    risk = state.get("risk_assessment") or {}
    if risk.get("risk_level") == "HIGH":
        return "escalate_high_risk", f"High-risk indicators (score {risk.get('risk_score')}): escalate.", []
    return "decision_agent", f"Risk level {risk.get('risk_level', 'LOW')}: no escalation needed.", []


def decide_after_decision(state: ClaimState) -> Route:
    if _failed(state):
        return "technical_failure", "Technical failure during decision support.", []
    rec = (state.get("decision_support") or {}).get("recommendation")
    return "human_review_queue", f"Recommendation {rec}: queue for a human claims officer.", []


ROUTERS: dict[str, Callable[[ClaimState], Route]] = {
    "intake": decide_after_intake,
    "document_agent": decide_after_documents,
    "policy_agent": decide_after_policy,
    "rag_agent": decide_after_rag,
    "validation_agent": decide_after_validation,
    "fraud_agent": decide_after_fraud,
    "decision_agent": decide_after_decision,
}


def router(node: str) -> Callable[[ClaimState], str]:
    """LangGraph conditional-edge function for `node`."""
    decide = ROUTERS[node]
    return lambda state: decide(state)[0]


# =========================================================================== #
# Node wrapper
# =========================================================================== #
NodeFn = Callable[[ClaimState, WorkflowContext], tuple[dict, dict]]


def _wrap(name: str, fn: NodeFn, ctx: WorkflowContext) -> Callable[[ClaimState], dict]:
    def node(state: ClaimState) -> dict:
        started, t0 = _now(), time.perf_counter()
        sequence = len(state.get("execution_history") or []) + 1
        record: ExecutionRecord = {"node": name, "title": AGENT_TITLES.get(name, name), "sequence": sequence,
                                   "started_at": started, "used_llm": False, "model_name": None, "error": None}
        try:
            if name in ctx.fail_on:
                raise RuntimeError(f"Simulated technical failure in {name} (test hook)")
            update, meta = fn(state, ctx)
            record.update(status=meta.get("status", "SUCCESS"), used_llm=bool(meta.get("used_llm")),
                          model_name=meta.get("model_name"), input_summary=meta.get("input", {}),
                          output_summary=meta.get("output", {}))
        except Exception as exc:  # technical failure: record it, never crash the workflow
            log.exception("Node %s failed", name)
            error = {"node": name, "type": exc.__class__.__name__, "message": str(exc)[:500], "at": _now()}
            update = {"processing_status": "FAILED", "error_information": [error]}
            record.update(status="FAILED", error=f"{error['type']}: {error['message']}", input_summary={},
                          output_summary={})

        record["finished_at"] = _now()
        record["duration_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        update["current_stage"] = name

        skipped: list[str] = []
        if name in ROUTERS:
            nxt, reason, skipped = ROUTERS[name]({**state, **update})
            record.update(next_step=nxt, route_reason=reason)
        update["execution_history"] = [record]
        if skipped:
            update["skipped_agents"] = skipped
        return update

    node.__name__ = name
    return node


# =========================================================================== #
# Nodes
# =========================================================================== #
def intake(state: ClaimState, ctx: WorkflowContext) -> tuple[dict, dict]:
    claim = ctx.claim(state)
    if claim is None:
        raise LookupError(f"Claim {state.get('claim_id')} not found")
    customer = claim.customer
    documents = [{"path": d.file_path, "document_type": d.document_type, "filename": d.original_filename,
                  "size_bytes": d.file_size_bytes} for d in claim.documents]
    update = {
        "claim_number": claim.claim_number,
        "processing_status": "RUNNING",
        "started_at": _now(),
        "customer_information": {
            "customer_code": customer.customer_code if customer else None,
            "full_name": customer.full_name if customer else claim.customer_name_submitted,
            "email": customer.email if customer else None,
            "phone": customer.phone if customer else None,
            "city": customer.city if customer else None,
        },
        "submitted_claim": {
            "customer_name": claim.customer_name_submitted, "policy_number": claim.policy_number_submitted,
            "insurance_type": claim.insurance_type, "claim_amount": claim.claim_amount,
            "hospital_name": claim.hospital_name, "admission_date": str(claim.admission_date or ""),
            "discharge_date": str(claim.discharge_date or ""), "diagnosis": claim.diagnosis,
            "invoice_number": claim.invoice_number, "description": claim.description,
        },
        "uploaded_documents": documents,
    }
    return update, {"input": {"claim_number": claim.claim_number},
                    "output": {"documents": len(documents), "customer": update["customer_information"]["full_name"]}}


def document_node(state: ClaimState, ctx: WorkflowContext) -> tuple[dict, dict]:
    docs = state.get("uploaded_documents") or []
    if not docs:
        return ({"extracted_document_information": None},
                {"status": "WARNING", "input": {"documents": 0}, "output": {"note": "No documents uploaded."}})
    inputs = []
    for d in docs:
        path = Path(d["path"])
        inputs.append(DocumentInput(path if path.is_absolute() else PROJECT_ROOT / path, d.get("document_type"),
                                    d.get("filename")))
    result = DocumentExtractionAgent(use_llm=ctx.use_llm, llm_model=ctx.llm_model).run(inputs)
    status = "WARNING" if result.missing_fields or result.conflicts or result.warnings else "SUCCESS"
    return ({"extracted_document_information": _dump(result)},
            {"status": status, "used_llm": result.used_llm, "model_name": result.llm_model,
             "input": {"documents": [d["filename"] for d in docs]},
             "output": {"document_types": result.document_types_found, "missing_fields": result.missing_fields,
                        "conflicts": len(result.conflicts), "ocr_used": result.ocr_used,
                        "mode": result.extraction_mode}})


def _extraction(state: ClaimState) -> DocumentExtractionResult | None:
    ex = state.get("extracted_document_information")
    return DocumentExtractionResult.model_validate(ex) if ex else None


def policy_node(state: ClaimState, ctx: WorkflowContext) -> tuple[dict, dict]:
    facts = build_input(ctx.claim(state), _extraction(state))
    result = PolicyVerificationAgent(ctx.session).run(facts)
    status = {"VERIFIED": "SUCCESS"}.get(result.status, "WARNING")
    return ({"policy_verification": _dump(result), "policy_information": _dump(result.policy)},
            {"status": status, "input": {"policy_number": facts.policy_number, "sources": facts.sources},
             "output": {"status": result.status, "flags": [k for k, v in result.flags.items() if v],
                        "issues": result.issues}})


def rag_node(state: ClaimState, ctx: WorkflowContext) -> tuple[dict, dict]:
    facts = build_input(ctx.claim(state), _extraction(state))
    verification = PolicyVerificationResult.model_validate(state["policy_verification"])
    data = build_rag_input(verification, facts)
    result = ctx.rag_agent().run(data)
    status = "SUCCESS" if result.status == "EVIDENCE_FOUND" and not result.possible_exclusions else "WARNING"
    return ({"retrieved_policy_clauses": _dump(result)},
            {"status": status, "used_llm": result.used_llm, "model_name": result.llm_model,
             "input": {"plan_code": data.plan_code, "diagnosis": data.diagnosis},
             "output": {"status": result.status, "clauses": [f"{c.plan_code} {c.clause_code}" for c in result.clauses],
                        "possible_exclusions": result.possible_exclusions, "embeddings": result.embedding_model}})


def _validation_input(state: ClaimState, ctx: WorkflowContext):
    claim = ctx.claim(state)
    extraction = _extraction(state)
    types = None if extraction else [d["document_type"] for d in state.get("uploaded_documents") or []]
    return build_validation_input(claim, extraction, document_types=types)


def validation_node(state: ClaimState, ctx: WorkflowContext) -> tuple[dict, dict]:
    vin = _validation_input(state, ctx)
    result = ValidationAgent(ctx.session, today=ctx.today).run(vin)
    status = "SUCCESS" if result.status == "VALID" else "WARNING"
    return ({"claim_validation": _dump(result)},
            {"status": status, "input": {"documents": vin.document_types, "documents_read": vin.documents_read},
             "output": {"status": result.status, "flags": [k for k, v in result.flags.items() if v],
                        "issues": result.issues}})


def fraud_node(state: ClaimState, ctx: WorkflowContext) -> tuple[dict, dict]:
    vin = _validation_input(state, ctx)
    result = FraudRiskAgent(ctx.session).run(
        vin, ValidationResult.model_validate(state["claim_validation"]),
        PolicyVerificationResult.model_validate(state["policy_verification"]), _extraction(state))
    return ({"risk_assessment": _dump(result)},
            {"status": "SUCCESS" if result.risk_level == "LOW" else "WARNING",
             "input": {"claim_number": vin.claim_number},
             "output": {"risk_score": result.risk_score, "risk_level": result.risk_level,
                        "indicators": [i.code for i in result.indicators]}})


def escalate_node(state: ClaimState, ctx: WorkflowContext) -> tuple[dict, dict]:
    risk = state.get("risk_assessment") or {}
    escalation = {"priority": "HIGH", "referral_suggested": True,
                  "reason": f"Risk score {risk.get('risk_score')}/100 with indicators: "
                            + ", ".join(i["code"] for i in risk.get("indicators", [])),
                  "note": "Escalation prioritises the review queue; it is not a finding of wrongdoing."}
    return {"escalation": escalation}, {"status": "WARNING", "output": escalation}


def decision_node(state: ClaimState, ctx: WorkflowContext) -> tuple[dict, dict]:
    skipped = set(state.get("skipped_agents") or [])
    rag = (PolicyRAGResult.model_validate(state["retrieved_policy_clauses"]) if state.get("retrieved_policy_clauses")
           else PolicyRAGResult(status="POLICY_CONTEXT_UNAVAILABLE", requires_human_review=True,
                                uncertainty_reasons=["Policy RAG was skipped: the policy is unknown."]))
    risk = (RiskAssessment.model_validate(state["risk_assessment"]) if state.get("risk_assessment")
            else RiskAssessment(risk_score=0, risk_level="LOW",
                                summary="Risk screening deferred until the claim is complete."))
    result = DecisionSupportAgent(use_llm=ctx.use_llm, llm_model=ctx.llm_model).run(
        PolicyVerificationResult.model_validate(state["policy_verification"]), rag,
        ValidationResult.model_validate(state["claim_validation"]), risk, _extraction(state))
    if "fraud_agent" in skipped:
        result.confidence_notes.append("Risk screening was deferred by the workflow (claim incomplete or invalid).")
    if state.get("escalation"):
        result.required_actions.insert(0, "PRIORITY: high-risk indicators — " + state["escalation"]["reason"])
    return ({"decision_support": _dump(result)},
        {"used_llm": result.used_llm, "model_name": result.llm_model,
         "status": "WARNING" if result.warnings else "SUCCESS",
         "input": {"skipped_agents": sorted(skipped)},
         "output": {"recommendation": result.recommendation,
                    "confidence": result.confidence,
                    "actions": len(result.required_actions)}})


def technical_failure_node(state: ClaimState, ctx: WorkflowContext) -> tuple[dict, dict]:
    errors = state.get("error_information") or []
    last = errors[-1] if errors else {"node": "unknown", "message": "unknown error"}
    decision = DecisionSupport(
        recommendation="REVIEW_REQUIRED", confidence=0.3,
        confidence_notes=["Automated processing did not complete."],
        reasons=[DecisionReason(source="workflow", code="TECHNICAL_FAILURE", blocking=True,
                                message=f"{AGENT_TITLES.get(last['node'], last['node'])} failed: {last['message']}")],
        required_actions=["Assess the claim manually or re-run processing after the issue is fixed."],
        reviewer_checklist=["Check which agent results are missing before relying on any output."],
        summary="Automated processing stopped because of a technical failure. The claim is routed to a claims "
                "officer for manual review. Final decision rests with an authorised claims officer.",
        disclaimer="Decision support only. A technical failure occurred; no automated assessment is complete.",
    )
    return ({"decision_support": _dump(decision)},
            {"status": "WARNING", "output": {"failed_node": last["node"], "recommendation": "REVIEW_REQUIRED"}})


def human_review_queue_node(state: ClaimState, ctx: WorkflowContext) -> tuple[dict, dict]:
    failed = _failed(state)
    final_status = "FAILED" if failed else "COMPLETED"
    rec = (state.get("decision_support") or {}).get("recommendation", "REVIEW_REQUIRED")
    finished = _now()
    if ctx.persist and state.get("claim_id"):
        from workflows.persistence import persist_run

        persist_run(ctx.session, {**state, "processing_status": final_status, "finished_at": finished})
    return ({"processing_status": final_status, "finished_at": finished},
            {"status": "FAILED" if failed else "SUCCESS",
             "output": {"queued_as": rec, "processing_status": final_status, "persisted": bool(ctx.persist)}})


NODES: dict[str, NodeFn] = {
    "intake": intake,
    "document_agent": document_node,
    "policy_agent": policy_node,
    "rag_agent": rag_node,
    "validation_agent": validation_node,
    "fraud_agent": fraud_node,
    "escalate_high_risk": escalate_node,
    "decision_agent": decision_node,
    "technical_failure": technical_failure_node,
    "human_review_queue": human_review_queue_node,
}


def build_nodes(ctx: WorkflowContext) -> dict[str, Callable[[ClaimState], dict]]:
    return {name: _wrap(name, fn, ctx) for name, fn in NODES.items()}
