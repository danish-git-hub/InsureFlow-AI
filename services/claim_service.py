"""
Application services: the operations behind both the FastAPI endpoints and the dashboard.

Keeping business operations here (not in API routes or Streamlit pages) means
there is exactly one implementation of "submit a claim", "process a claim" and
"record a human review", with the same validation and audit logging everywhere.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.config import get_logger, settings
from database.models import Claim, ClaimStatus, ProcessingStatus, ReviewDecision
from database.repository import (
    ClaimRepository,
    CustomerRepository,
    DocumentRepository,
    LogRepository,
    PolicyRepository,
    ReviewRepository,
)
from documents.storage import DocumentValidationError, sha256_bytes, store_claim_document, validate_pdf_bytes
from services.schemas import ClaimSubmission, ReviewSubmission, UploadedFile

log = get_logger("claim_service")

MAX_FILES_PER_CLAIM = 10

# Basic role separation: which reviewer roles may record which outcomes.
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "CLAIMS_REVIEWER": {d.value for d in ReviewDecision},
    "SENIOR_REVIEWER": {d.value for d in ReviewDecision},
    "INVESTIGATOR": {ReviewDecision.INFO_REQUESTED.value, ReviewDecision.ESCALATED.value},
}
# High-value claims need a senior reviewer to record a final outcome.
SENIOR_APPROVAL_THRESHOLD = 500_000


class ServiceError(Exception):
    """A user-facing error with an HTTP-like status code."""

    def __init__(self, message: str, status_code: int = 400, details: Any = None):
        super().__init__(message)
        self.message, self.status_code, self.details = message, status_code, details


# --------------------------------------------------------------------------- #
# Claim submission
# --------------------------------------------------------------------------- #
def submit_claim(session: Session, form: ClaimSubmission, files: list[UploadedFile]) -> tuple[Claim, list[str]]:
    """Validate and store a new claim with its PDFs. Returns (claim, warnings). Does not commit."""
    if len(files) > MAX_FILES_PER_CLAIM:
        raise ServiceError(f"At most {MAX_FILES_PER_CLAIM} documents per claim.", 422)
    customer = CustomerRepository(session).get_by_code(form.customer_id)
    if customer is None:
        raise ServiceError(f"Customer ID {form.customer_id} was not found.", 404)

    # Validate every file before writing anything (all-or-nothing upload).
    errors = []
    for f in files:
        try:
            validate_pdf_bytes(f.content, f.filename)
        except DocumentValidationError as exc:
            errors.append(str(exc))
    if errors:
        raise ServiceError("One or more documents were rejected.", 422, errors)

    warnings: list[str] = []
    policy = PolicyRepository(session).get_by_number(form.policy_number)
    if policy is None:
        warnings.append(f"Policy {form.policy_number} is not on record; the agents will flag this for review.")

    claims = ClaimRepository(session)
    claim = claims.create(
        customer_id=customer.id,
        policy_id=policy.id if policy else None,
        customer_name_submitted=form.customer_name,
        policy_number_submitted=form.policy_number,
        insurance_type=form.insurance_type.value,
        claim_amount=form.claim_amount,
        hospital_name=form.hospital_name,
        admission_date=form.admission_date,
        discharge_date=form.discharge_date,
        diagnosis=form.diagnosis,
        description=form.description,
        submitted_at=datetime.now(timezone.utc),
        claim_status=ClaimStatus.SUBMITTED.value,
        processing_status=ProcessingStatus.PENDING.value,
    )
    docs = DocumentRepository(session)
    for f in files:
        digest = sha256_bytes(f.content)
        for existing in docs.find_by_hash(digest):
            if existing.claim_id != claim.id:
                warnings.append(f"'{f.filename}' is identical to a document already submitted with claim "
                                f"{existing.claim.claim_number}.")
        stored = store_claim_document(claim.claim_number, f.filename, f.content)
        docs.add(claim_id=claim.id, document_type=f.declared_type or "OTHER",
                 original_filename=stored.original_filename, stored_filename=stored.stored_filename,
                 file_path=_relative(stored.file_path), file_size_bytes=stored.file_size_bytes,
                 sha256=stored.sha256, page_count=stored.page_count)
    LogRepository(session).log("claim_service", "CLAIM_SUBMITTED",
                               f"{claim.claim_number} submitted with {len(files)} document(s)",
                               claim_id=claim.id, details={"warnings": warnings})
    session.flush()
    return claim, warnings


def _relative(path) -> str:
    from database.init_db import _project_relative

    return _project_relative(path)


# --------------------------------------------------------------------------- #
# Processing
# --------------------------------------------------------------------------- #
def process(session: Session, claim_number: str, **kwargs) -> dict:
    """Run the LangGraph workflow for one claim (commits)."""
    from workflows.graph import process_claim

    claim = ClaimRepository(session).get_by_number(claim_number)
    if claim is None:
        raise ServiceError(f"Claim {claim_number} not found.", 404)
    if claim.claim_status in (ClaimStatus.HUMAN_APPROVED.value, ClaimStatus.HUMAN_REJECTED.value):
        raise ServiceError(f"Claim {claim_number} already has a final human decision; it cannot be reprocessed.", 409)
    return process_claim(session, claim, **kwargs)


def process_pending(session: Session, **kwargs) -> list[dict]:
    from workflows.graph import process_claim

    return [process_claim(session, c, **kwargs) for c in ClaimRepository(session).pending_processing()]


# --------------------------------------------------------------------------- #
# Human review
# --------------------------------------------------------------------------- #
def record_review(session: Session, claim_number: str, review: ReviewSubmission):
    """Record a human decision (the only way a claim reaches an approved/rejected state)."""
    claim = ClaimRepository(session).get_by_number(claim_number)
    if claim is None:
        raise ServiceError(f"Claim {claim_number} not found.", 404)
    if not claim.decision_support:
        raise ServiceError("Run the AI workflow before recording a review, so the reviewer sees the evidence.", 409)
    if claim.claim_status in (ClaimStatus.HUMAN_APPROVED.value, ClaimStatus.HUMAN_REJECTED.value):
        raise ServiceError(f"Claim {claim_number} already has a final decision ({claim.claim_status}).", 409)
    decision = review.decision.value
    if decision not in ROLE_PERMISSIONS[review.reviewer_role]:
        raise ServiceError(f"Role {review.reviewer_role} may not record the outcome {decision}.", 403)
    if (decision in ("APPROVED", "REJECTED") and (claim.claim_amount or 0) > SENIOR_APPROVAL_THRESHOLD
            and review.reviewer_role != "SENIOR_REVIEWER"):
        raise ServiceError(f"Claims above INR {SENIOR_APPROVAL_THRESHOLD:,} need a SENIOR_REVIEWER for a final "
                           "outcome.", 403)
    record = ReviewRepository(session).record(claim, review.reviewer_name, decision, review.comments,
                                              reviewer_role=review.reviewer_role)
    LogRepository(session).log("human_review", "HUMAN_REVIEW_RECORDED",
                               f"{claim.claim_number}: {decision} by {review.reviewer_role}",
                               claim_id=claim.id,
                               details={"ai_recommendation": record.ai_recommendation_snapshot})
    session.flush()
    return record


# --------------------------------------------------------------------------- #
# Policy documents for RAG
# --------------------------------------------------------------------------- #
def ingest_policy_document(session: Session, filename: str, content: bytes) -> dict:
    """Save an uploaded policy PDF, rebuild the FAISS index and sync clauses to the database."""
    from documents.storage import safe_filename
    from rag.retriever import PolicyIndex, get_policy_index, sync_clauses_to_db

    try:
        validate_pdf_bytes(content, filename)
    except DocumentValidationError as exc:
        raise ServiceError(str(exc), 422) from exc
    name = safe_filename(filename)
    path = settings.policies_dir / name
    replaced = path.exists()
    path.write_bytes(content)
    index = PolicyIndex.load_or_build(rebuild=True)
    get_policy_index.cache_clear()
    clauses = sync_clauses_to_db(session, index)
    added = [c for c in index.chunks if c.source_document == name]
    LogRepository(session).log("rag", "POLICY_DOCUMENT_INDEXED", f"{name}: {len(added)} chunk(s) indexed",
                               details={"replaced": replaced, "total_clauses": clauses})
    session.flush()
    return {"filename": name, "replaced": replaced, "chunks": len(added), "total_clauses": clauses,
            "structured": any(c.clause_code for c in added), "embedding_model": index.embedding_id}
