"""
The InsureFlow AI claim workflow as a LangGraph `StateGraph`.

    START -> intake -> document_agent -> policy_agent -> rag_agent -> validation_agent
          -> fraud_agent -> [escalate_high_risk] -> decision_agent -> human_review_queue -> END

Conditional edges (see workflows/nodes.py for the exact rules and reasons):

    * unknown policy            policy_agent      -> validation_agent   (RAG skipped)
    * missing docs / bad dates  validation_agent  -> decision_agent     (risk screening deferred)
    * high-risk indicators      fraud_agent       -> escalate_high_risk -> decision_agent
    * any technical failure     <agent>           -> technical_failure  -> human_review_queue
    * waiting period / uncertain coverage are carried as flags into the decision.

Every path ends in the human review queue: no claim is ever finalised by the AI.

    python -m workflows.graph CLM-...      # process one claim and save the results
    python -m workflows.graph --all        # process every claim
    python -m workflows.graph --mermaid    # print the graph diagram
"""

from __future__ import annotations

import argparse
import uuid
from datetime import date
from typing import Any

from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session

from app.config import get_logger
from database.models import Claim
from database.repository import ClaimRepository, LogRepository
from workflows.nodes import WorkflowContext, build_nodes, router
from workflows.state import ClaimState, initial_state

log = get_logger("graph")


def build_graph(ctx: WorkflowContext):
    """Create and compile the StateGraph with the given dependencies."""
    graph = StateGraph(ClaimState)
    for name, fn in build_nodes(ctx).items():
        graph.add_node(name, fn)

    graph.add_edge(START, "intake")
    graph.add_conditional_edges("intake", router("intake"), ["document_agent", "technical_failure"])
    graph.add_conditional_edges("document_agent", router("document_agent"), ["policy_agent", "technical_failure"])
    graph.add_conditional_edges("policy_agent", router("policy_agent"),
                                ["rag_agent", "validation_agent", "technical_failure"])
    graph.add_conditional_edges("rag_agent", router("rag_agent"), ["validation_agent", "technical_failure"])
    graph.add_conditional_edges("validation_agent", router("validation_agent"),
                                ["fraud_agent", "decision_agent", "technical_failure"])
    graph.add_conditional_edges("fraud_agent", router("fraud_agent"),
                                ["escalate_high_risk", "decision_agent", "technical_failure"])
    graph.add_edge("escalate_high_risk", "decision_agent")
    graph.add_conditional_edges("decision_agent", router("decision_agent"),
                                ["human_review_queue", "technical_failure"])
    graph.add_edge("technical_failure", "human_review_queue")
    graph.add_edge("human_review_queue", END)
    return graph.compile()


def graph_mermaid() -> str:
    """Mermaid diagram of the compiled graph (structure only; no database needed)."""
    return build_graph(WorkflowContext(session=None, persist=False)).get_graph().draw_mermaid()  # type: ignore[arg-type]


def process_claim(session: Session, claim: Claim | int | str, *, rag_index: Any = None, use_llm: bool | None = None,
                  llm_model: Any = None, today: date | None = None, persist: bool = True,
                  fail_on: set[str] | None = None, commit: bool = True) -> ClaimState:
    """Run the full workflow for one claim and (by default) save everything to the database."""
    repo = ClaimRepository(session)
    if isinstance(claim, int):
        claim = session.get(Claim, claim)
    elif isinstance(claim, str):
        claim = repo.get_by_number(claim)
    if claim is None:
        raise LookupError("Claim not found")

    run_id = str(uuid.uuid4())
    if persist:
        repo.set_status(claim, processing_status="RUNNING")
        LogRepository(session).log("workflow", "PROCESSING_STARTED", f"{claim.claim_number}: workflow started",
                                   claim_id=claim.id, details={"run_id": run_id})
        session.flush()
    ctx = WorkflowContext(session=session, rag_index=rag_index, use_llm=use_llm, llm_model=llm_model, today=today,
                          persist=persist, fail_on=set(fail_on or ()))
    try:
        final: ClaimState = build_graph(ctx).invoke(initial_state(run_id, claim.id, claim.claim_number))
        if commit:
            session.commit()
    except Exception:
        session.rollback()
        raise
    log.info("%s -> %s (%s)", claim.claim_number, (final.get("decision_support") or {}).get("recommendation"),
             final.get("processing_status"))
    return final


def process_all(session: Session, **kwargs) -> list[ClaimState]:
    return [process_claim(session, c, **kwargs) for c in ClaimRepository(session).list()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LangGraph claim workflow")
    parser.add_argument("claim_number", nargs="?")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--mermaid", action="store_true")
    args = parser.parse_args()
    if args.mermaid:
        print(graph_mermaid())
        return

    from database.connection import SessionLocal

    with SessionLocal() as session:
        claims = (ClaimRepository(session).list() if args.all
                  else [ClaimRepository(session).get_by_number(args.claim_number)] if args.claim_number else [])
        if not claims or claims[0] is None:
            parser.error("give a claim number or --all")
        for claim in claims:
            state = process_claim(session, claim)
            path = " -> ".join(r["node"] for r in state["execution_history"])
            rec = (state.get("decision_support") or {}).get("recommendation")
            print(f"{claim.claim_number}  {claim.scenario or '':20} {rec:32} {state['processing_status']}")
            if not args.all:
                print(f"  path: {path}")
                for r in state["execution_history"]:
                    print(f"  {r['sequence']:>2}. {r['title']:28} {r['status']:8} {r['duration_ms']:>8.1f} ms"
                          f"  -> {r.get('next_step') or ''}  {r.get('route_reason') or ''}")


if __name__ == "__main__":
    main()
