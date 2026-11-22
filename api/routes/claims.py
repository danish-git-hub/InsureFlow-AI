"""Claim submission, documents, AI processing, status, details and agent history."""
from app.config import PROJECT_ROOT, settings
from pathlib import Path

from fastapi.responses import FileResponse
from datetime import date

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from pydantic import ValidationError
from sqlalchemy.orm import Session

from api.schemas import (
    ClaimCreated,
    ClaimDetail,
    ClaimStatusOut,
    ClaimSummary,
    DocumentOut,
    ExecutionOut,
    ProcessingResult,
    RiskFlagOut,
)
from api.uploads import read_upload
from database.connection import get_db
from database.models import Claim, ClaimStatus, DocumentType, InsuranceType, RiskLevel
from database.repository import (
    AgentExecutionRepository,
    ClaimRepository,
    DocumentRepository,
    LogRepository,
    RiskFlagRepository,
)
from documents.storage import DocumentValidationError, store_claim_document
from services import claim_service
from services.claim_service import ServiceError
from services.schemas import ClaimSubmission, UploadedFile

router = APIRouter(prefix="/claims", tags=["Claims"])
VALID_DOC_TYPES = {t.value for t in DocumentType}


def _claim(db: Session, claim_number: str) -> Claim:
    claim = ClaimRepository(db).get_by_number(claim_number)
    if claim is None:
        raise ServiceError(f"Claim {claim_number} not found.", 404)
    return claim


def _summary(state: dict) -> ProcessingResult:
    d = state.get("decision_support") or {}
    return ProcessingResult(
        claim_number=state.get("claim_number"), run_id=state.get("run_id"),
        processing_status=state.get("processing_status"), recommendation=d.get("recommendation"),
        confidence=d.get("confidence"), path=[r["node"] for r in state.get("execution_history") or []],
        skipped_agents=state.get("skipped_agents") or [], errors=state.get("error_information") or [],
    )


def _doc_types(raw: str | None, n: int) -> list[str | None]:
    types = [t.strip().upper() or None for t in (raw or "").split(",")] if raw else []
    types = [None if t in (None, "OTHER", "AUTO-DETECT") else t for t in types]
    bad = [t for t in types if t and t not in VALID_DOC_TYPES]
    if bad:
        raise ServiceError(f"Unknown document type(s): {', '.join(bad)}.", 422)
    return (types + [None] * n)[:n]


@router.post("", response_model=ClaimCreated, status_code=201)
async def submit_claim(
    customer_name: str = Form(...), customer_id: str = Form(...), policy_number: str = Form(...),
    insurance_type: InsuranceType = Form(...), claim_amount: float = Form(...), hospital_name: str = Form(...),
    admission_date: date = Form(...), discharge_date: date = Form(...), diagnosis: str = Form(...),
    description: str | None = Form(None),
    files: list[UploadFile] = File(..., description="Claim documents (PDF)"),
    document_types: str | None = Form(None, description="Comma-separated document type per file (optional)"),
    process_now: bool = Form(False, description="Run the AI workflow immediately"),
    db: Session = Depends(get_db),
):
    """Submit a new claim with its PDF documents (multipart/form-data)."""
    try:
        form = ClaimSubmission(customer_name=customer_name, customer_id=customer_id, policy_number=policy_number,
                               insurance_type=insurance_type, claim_amount=claim_amount, hospital_name=hospital_name,
                               admission_date=admission_date, discharge_date=discharge_date, diagnosis=diagnosis,
                               description=description)
    except ValidationError as exc:
        raise ServiceError("Invalid claim form.", 422,
                           [{"field": ".".join(map(str, e["loc"])), "message": e["msg"]} for e in exc.errors()]) from exc
    types = _doc_types(document_types, len(files))
    uploads = [UploadedFile(filename=f.filename or "document.pdf", content=await read_upload(f), declared_type=t)
               for f, t in zip(files, types)]
    claim, warnings = claim_service.submit_claim(db, form, uploads)
    number = claim.claim_number
    db.commit()
    processing = _summary(claim_service.process(db, number)) if process_now else None
    return ClaimCreated(claim_number=number, documents=len(uploads), warnings=warnings, processing=processing)


@router.get("", response_model=list[ClaimSummary])
def list_claims(status: ClaimStatus | None = None, insurance_type: InsuranceType | None = None,
                risk_level: RiskLevel | None = None, search: str | None = Query(None, max_length=100),
                date_from: date | None = None, date_to: date | None = None,
                limit: int = Query(200, ge=1, le=1000), db: Session = Depends(get_db)):
    return ClaimRepository(db).list(status=status.value if status else None,
                                    insurance_type=insurance_type.value if insurance_type else None,
                                    risk_level=risk_level.value if risk_level else None, search=search,
                                    date_from=date_from, date_to=date_to, limit=limit)


@router.get("/{claim_number}", response_model=ClaimDetail)
def get_claim(claim_number: str, db: Session = Depends(get_db)):
    """Full claim record including every agent's stored output."""
    return _claim(db, claim_number)


@router.get("/{claim_number}/status", response_model=ClaimStatusOut)
def get_status(claim_number: str, db: Session = Depends(get_db)):
    c = _claim(db, claim_number)
    return ClaimStatusOut(claim_number=c.claim_number, claim_status=c.claim_status,
                          processing_status=c.processing_status,
                          ai_recommendation=(c.decision_support or {}).get("recommendation"),
                          human_decision_required=c.claim_status not in ("HUMAN_APPROVED", "HUMAN_REJECTED"),
                          processed_at=c.processed_at)


@router.get("/{claim_number}/documents", response_model=list[DocumentOut])
def list_documents(claim_number: str, db: Session = Depends(get_db)):
    return _claim(db, claim_number).documents

@router.get("/{claim_number}/documents/{document_id}/file")
def download_claim_document(
    claim_number: str,
    document_id: int,
    db: Session = Depends(get_db),
):
    """Securely download a PDF belonging to a specific claim."""
    claim = _claim(db, claim_number)

    document = next(
        (doc for doc in claim.documents if doc.id == document_id),
        None,
    )

    if document is None:
        raise ServiceError("Document not found for this claim.", 404)

    stored_path = Path(document.file_path)

    if not stored_path.is_absolute():
        stored_path = PROJECT_ROOT / stored_path

    try:
        resolved_path = stored_path.resolve()
        resolved_path.relative_to(settings.data_dir.resolve())
    except (ValueError, OSError):
        raise ServiceError("Invalid document storage path.", 403)

    if not resolved_path.is_file():
        raise ServiceError("The PDF file is no longer available on the server.", 404)

    return FileResponse(
        path=resolved_path,
        media_type="application/pdf",
        filename=document.original_filename,
    )


@router.post("/{claim_number}/documents", response_model=list[DocumentOut], status_code=201)
async def upload_documents(claim_number: str, files: list[UploadFile] = File(...),
                           document_types: str | None = Form(None), db: Session = Depends(get_db)):
    """Add documents to an existing claim (e.g. after ADDITIONAL_INFORMATION_REQUIRED). Re-process afterwards."""
    claim = _claim(db, claim_number)
    if claim.claim_status in ("HUMAN_APPROVED", "HUMAN_REJECTED"):
        raise ServiceError("The claim already has a final human decision.", 409)
    repo, added = DocumentRepository(db), []
    for f, t in zip(files, _doc_types(document_types, len(files))):
        content = await read_upload(f)
        try:
            stored = store_claim_document(claim.claim_number, f.filename or "document.pdf", content)
        except DocumentValidationError as exc:
            raise ServiceError(str(exc), 422) from exc
        from database.init_db import _project_relative

        added.append(repo.add(claim_id=claim.id, document_type=t or "OTHER", original_filename=stored.original_filename,
                              stored_filename=stored.stored_filename, file_path=_project_relative(stored.file_path),
                              file_size_bytes=stored.file_size_bytes, sha256=stored.sha256,
                              page_count=stored.page_count))
    LogRepository(db).log("api", "DOCUMENTS_ADDED", f"{claim.claim_number}: {len(added)} document(s) added",
                          claim_id=claim.id)
    return added


@router.post("/{claim_number}/process", response_model=ProcessingResult)
def process_claim(claim_number: str, db: Session = Depends(get_db)):
    """Run the six-agent LangGraph workflow and store the results. The result is a recommendation only."""
    return _summary(claim_service.process(db, claim_number))


@router.get("/{claim_number}/executions", response_model=list[ExecutionOut])
def agent_history(claim_number: str, all_runs: bool = False, db: Session = Depends(get_db)):
    """Agent execution history (latest run by default)."""
    claim = _claim(db, claim_number)
    repo = AgentExecutionRepository(db)
    return repo.for_claim(claim.id) if all_runs else repo.latest_run(claim.id)


@router.get("/{claim_number}/risk-flags", response_model=list[RiskFlagOut])
def risk_flags(claim_number: str, db: Session = Depends(get_db)):
    return RiskFlagRepository(db).for_claim(_claim(db, claim_number).id)
