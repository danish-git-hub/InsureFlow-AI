"""Policy management and the RAG policy-document library."""

from datetime import timedelta

from fastapi import APIRouter, Depends, File, Query, UploadFile
from sqlalchemy.orm import Session

from api.schemas import ClauseHit, ClauseOut, PolicyCreate, PolicyDocumentResult, PolicyOut, PolicyStatusUpdate
from api.uploads import read_upload
from database.catalog import PLANS
from database.connection import get_db
from database.models import Policy, PolicyStatus
from database.repository import CustomerRepository, LogRepository, PolicyClauseRepository, PolicyRepository
from services import claim_service
from services.claim_service import ServiceError

router = APIRouter(prefix="/policies", tags=["Policies"])


def policy_out(p: Policy) -> PolicyOut:
    return PolicyOut(
        policy_number=p.policy_number, customer_code=p.customer.customer_code, customer_name=p.customer.full_name,
        insurance_type=p.insurance_type, plan_name=p.plan_name, plan_code=p.plan_code, sum_insured=p.sum_insured,
        annual_premium=p.annual_premium, start_date=p.start_date, end_date=p.end_date,
        waiting_period_days=p.waiting_period_days, status=p.status, exclusions=p.exclusions,
    )


@router.get("", response_model=list[PolicyOut])
def list_policies(status: PolicyStatus | None = None, insurance_type: str | None = None,
                  search: str | None = Query(None, max_length=100), db: Session = Depends(get_db)):
    return [policy_out(p) for p in PolicyRepository(db).list(status=status.value if status else None,
                                                             insurance_type=insurance_type, search=search)]


@router.get("/clauses", response_model=list[ClauseOut])
def list_clauses(plan_code: str | None = None, category: str | None = None, db: Session = Depends(get_db)):
    """Policy clauses currently indexed for retrieval."""
    return PolicyClauseRepository(db).list(plan_code=plan_code, category=category)


@router.get("/search", response_model=list[ClauseHit])
def search_wording(q: str = Query(min_length=2, max_length=300), plan_code: str | None = None,
                   k: int = Query(4, ge=1, le=10)):
    """Semantic search over policy wording (the same FAISS index the Policy RAG Agent uses)."""
    from rag.retriever import get_policy_index

    hits = get_policy_index().search(q, k=k, plan_code=plan_code)
    return [ClauseHit(score=round(h.score, 4), clause=ClauseOut(**{**h.chunk.db_fields(),
                                                                   "source_document": h.chunk.source_document}))
            for h in hits]


@router.post("/documents", response_model=PolicyDocumentResult, status_code=201)
async def upload_policy_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload a policy wording PDF; the FAISS index is rebuilt and clauses are synced to the database."""
    content = await read_upload(file)
    return claim_service.ingest_policy_document(db, file.filename or "policy.pdf", content)


@router.get("/{policy_number}", response_model=PolicyOut)
def get_policy(policy_number: str, db: Session = Depends(get_db)):
    policy = PolicyRepository(db).get_by_number(policy_number)
    if policy is None:
        raise ServiceError(f"Policy {policy_number} not found.", 404)
    return policy_out(policy)


@router.post("", response_model=PolicyOut, status_code=201)
def create_policy(body: PolicyCreate, db: Session = Depends(get_db)):
    """Create a synthetic policy from the product catalogue."""
    plan = PLANS.get(body.plan_code.upper())
    if plan is None:
        raise ServiceError(f"Unknown plan code {body.plan_code}. Valid: {', '.join(PLANS)}.", 422)
    customer = CustomerRepository(db).get_by_code(body.customer_id.upper())
    if customer is None:
        raise ServiceError(f"Customer {body.customer_id} not found.", 404)
    repo = PolicyRepository(db)
    seq = repo.count() + 1
    while repo.get_by_number(f"{plan.code}-{body.start_date:%Y}-{seq:06d}"):
        seq += 1
    policy = repo.create(
        policy_number=f"{plan.code}-{body.start_date:%Y}-{seq:06d}", customer_id=customer.id,
        insurance_type=plan.insurance_type.value, plan_name=plan.name, plan_code=plan.code,
        sum_insured=body.sum_insured, annual_premium=round(body.sum_insured * plan.premium_rate, 2),
        start_date=body.start_date, end_date=body.start_date + timedelta(days=364),
        waiting_period_days=plan.default_waiting_days if body.waiting_period_days is None else body.waiting_period_days,
        status=PolicyStatus.ACTIVE.value, hospitalization_covered=True,
        exclusions=[e.title for e in plan.exclusions], policy_document=plan.document_filename,
    )
    LogRepository(db).log("api", "POLICY_CREATED", f"{policy.policy_number} created for {customer.customer_code}")
    return policy_out(policy)


@router.patch("/{policy_number}/status", response_model=PolicyOut)
def update_policy_status(policy_number: str, body: PolicyStatusUpdate, db: Session = Depends(get_db)):
    if body.status not in {s.value for s in PolicyStatus}:
        raise ServiceError(f"Invalid status {body.status}.", 422)
    policy = PolicyRepository(db).update_status(policy_number, body.status)
    if policy is None:
        raise ServiceError(f"Policy {policy_number} not found.", 404)
    LogRepository(db).log("api", "POLICY_STATUS_CHANGED", f"{policy.policy_number} -> {body.status}")
    return policy_out(policy)
