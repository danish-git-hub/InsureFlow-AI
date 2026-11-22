"""Human review — the only way a claim reaches a final outcome."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.schemas import ClaimSummary, ReviewOut, ReviewRecorded
from database.connection import get_db
from database.repository import ClaimRepository, ReviewRepository
from services import claim_service
from services.claim_service import ServiceError
from services.schemas import ReviewSubmission

router = APIRouter(tags=["Human Review"])


@router.get("/reviews/queue", response_model=list[ClaimSummary])
def review_queue(db: Session = Depends(get_db)):
    """Processed claims waiting for a human decision."""
    return [c for c in ClaimRepository(db).review_queue() if c.decision_support]


@router.post("/claims/{claim_number}/reviews", response_model=ReviewRecorded, status_code=201)
def record_review(claim_number: str, body: ReviewSubmission, db: Session = Depends(get_db)):
    """Record a human decision. Role rules: investigators may only request information or escalate;
    final outcomes on high-value claims need a senior reviewer."""
    review = claim_service.record_review(db, claim_number, body)
    return ReviewRecorded(claim_number=claim_number, decision=review.decision,
                          claim_status=review.claim.claim_status,
                          ai_recommendation_snapshot=review.ai_recommendation_snapshot)


@router.get("/claims/{claim_number}/reviews", response_model=list[ReviewOut])
def list_reviews(claim_number: str, db: Session = Depends(get_db)):
    claim = ClaimRepository(db).get_by_number(claim_number)
    if claim is None:
        raise ServiceError(f"Claim {claim_number} not found.", 404)
    return ReviewRepository(db).for_claim(claim.id)
