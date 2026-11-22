"""Response models for the REST API (request models live in services/schemas.py)."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Any = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class CustomerOut(ORM):
    customer_code: str
    full_name: str
    date_of_birth: date | None
    email: str | None
    phone: str | None
    city: str | None
    state: str | None


class CustomerCreate(BaseModel):
    full_name: str = Field(min_length=2, max_length=120)
    date_of_birth: date
    email: str | None = Field(default=None, max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    phone: str | None = Field(default=None, max_length=30)
    city: str | None = Field(default=None, max_length=60)
    state: str | None = Field(default=None, max_length=60)


class PolicyOut(ORM):
    policy_number: str
    customer_code: str
    customer_name: str
    insurance_type: str
    plan_name: str
    plan_code: str
    sum_insured: float
    annual_premium: float | None
    start_date: date
    end_date: date
    waiting_period_days: int
    status: str
    exclusions: list[str] | None


class PolicyCreate(BaseModel):
    customer_id: str
    plan_code: str
    sum_insured: float = Field(gt=0)
    start_date: date
    waiting_period_days: int | None = Field(default=None, ge=0, le=1460)


class PolicyStatusUpdate(BaseModel):
    status: str


class ClauseOut(ORM):
    source_document: str
    plan_code: str | None
    clause_code: str | None
    title: str | None
    category: str
    page_number: int | None
    text: str


class ClauseHit(BaseModel):
    score: float
    clause: ClauseOut


class PolicyDocumentResult(BaseModel):
    filename: str
    replaced: bool
    chunks: int
    total_clauses: int
    structured: bool
    embedding_model: str


class DocumentOut(ORM):
    document_type: str
    original_filename: str
    file_size_bytes: int
    page_count: int | None
    sha256: str
    uploaded_at: datetime


class ClaimSummary(ORM):
    claim_number: str
    customer_name_submitted: str
    policy_number_submitted: str | None
    insurance_type: str
    claim_amount: float
    submitted_at: datetime
    claim_status: str
    processing_status: str
    risk_level: str | None
    risk_score: float | None


class ClaimStatusOut(BaseModel):
    claim_number: str
    claim_status: str
    processing_status: str
    ai_recommendation: str | None
    human_decision_required: bool
    processed_at: datetime | None


class ClaimDetail(ClaimSummary):
    hospital_name: str | None
    admission_date: date | None
    discharge_date: date | None
    diagnosis: str | None
    description: str | None
    invoice_number: str | None
    billed_amount: float | None
    processed_at: datetime | None
    processing_time_seconds: float | None
    documents: list[DocumentOut] = []
    extracted_data: dict | None = None
    policy_verification: dict | None = None
    retrieved_clauses: dict | None = None
    validation_result: dict | None = None
    risk_assessment: dict | None = None
    decision_support: dict | None = None


class ClaimCreated(BaseModel):
    claim_number: str
    documents: int
    warnings: list[str]
    processing: ProcessingResult | None = None


class ProcessingResult(BaseModel):
    claim_number: str | None
    run_id: str | None
    processing_status: str | None
    recommendation: str | None
    confidence: float | None
    path: list[str]
    skipped_agents: list[str]
    errors: list[dict]


class ExecutionOut(ORM):
    run_id: str
    agent_name: str
    sequence: int
    status: str
    started_at: datetime | None
    finished_at: datetime | None
    duration_ms: float | None
    used_llm: bool
    model_name: str | None
    input_summary: dict | None
    output: dict | None
    error_message: str | None


class RiskFlagOut(ORM):
    code: str
    severity: str
    description: str
    evidence: dict | None


class ReviewOut(ORM):
    reviewer_name: str
    reviewer_role: str
    ai_recommendation_snapshot: str | None
    decision: str
    comments: str | None
    reviewed_at: datetime


class ReviewRecorded(BaseModel):
    claim_number: str
    decision: str
    claim_status: str
    ai_recommendation_snapshot: str | None


class AgentInfo(BaseModel):
    key: str
    name: str
    purpose: str


ClaimCreated.model_rebuild()
