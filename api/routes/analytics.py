"""Analytics (all values computed from database records) and system information."""

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from api.schemas import AgentInfo
from database.connection import get_db
from database.repository import AnalyticsRepository

router = APIRouter(tags=["Analytics"])


@router.get("/analytics/kpis")
def kpis(db: Session = Depends(get_db)):
    return AnalyticsRepository(db).kpis()


@router.get("/analytics/summary")
def summary(db: Session = Depends(get_db)):
    a = AnalyticsRepository(db)
    return {"kpis": a.kpis(), "status_distribution": a.status_distribution(), "claims_by_type": a.claims_by_type(),
            "risk_distribution": a.risk_distribution(), "monthly_claims": a.monthly_claims(),
            "agent_performance": a.agent_performance(), "review_workload": a.review_workload()}


@router.get("/agents", response_model=list[AgentInfo], tags=["Workflow"])
def agents():
    from agents.decision_agent import DecisionSupportAgent
    from agents.document_agent import DocumentExtractionAgent
    from agents.fraud_agent import FraudRiskAgent
    from agents.policy_agent import PolicyVerificationAgent
    from agents.rag_agent import PolicyRAGAgent
    from agents.validation_agent import ValidationAgent

    return [AgentInfo(key=a.key, name=a.name, purpose=a.purpose) for a in (
        DocumentExtractionAgent, PolicyVerificationAgent, PolicyRAGAgent, ValidationAgent, FraudRiskAgent,
        DecisionSupportAgent)]


@router.get("/workflow/graph", response_class=PlainTextResponse, tags=["Workflow"])
def workflow_graph():
    """The compiled LangGraph as a Mermaid diagram."""
    from workflows.graph import graph_mermaid

    return graph_mermaid()
