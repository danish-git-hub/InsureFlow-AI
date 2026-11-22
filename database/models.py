"""
SQLAlchemy ORM models (database tables) for InsureFlow AI.

Nine entities:
    Customer ─┬─< Policy
              └─< Claim >── Policy (nullable: the policy number may be unknown)
    Claim ─┬─< ClaimDocument
           ├─< AgentExecution
           ├─< RiskFlag
           ├─< HumanReview
           └─< ProcessingLog
    PolicyClause  (clauses indexed from policy PDFs for RAG)

Design choices (PostgreSQL-compatible):
    * Status fields are plain strings validated by CHECK constraints, instead of
      database-specific ENUM types, so the schema is portable.
    * Flexible agent outputs are stored in JSON columns (JSON in SQLite, JSON in
      PostgreSQL) while key reporting fields stay as indexed columns.
    * Money uses Numeric(14, 2) — exact decimals, not floats, at storage level.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _in(column: str, enum_cls: type[StrEnum]) -> str:
    """Build a CHECK constraint expression: column IN ('A','B',...)."""
    values = ", ".join(f"'{v.value}'" for v in enum_cls)
    return f"{column} IN ({values})"


# --------------------------------------------------------------------------- #
# Enumerations (stored as strings)
# --------------------------------------------------------------------------- #
class InsuranceType(StrEnum):
    HEALTH = "Individual Health"
    FAMILY = "Family Floater"
    SENIOR = "Senior Citizen Health"
    ACCIDENT = "Personal Accident"
    CRITICAL = "Critical Illness"


class PolicyStatus(StrEnum):
    ACTIVE = "ACTIVE"
    EXPIRED = "EXPIRED"
    LAPSED = "LAPSED"
    CANCELLED = "CANCELLED"


class ClaimStatus(StrEnum):
    """Business status. Note: no AI-driven APPROVED/REJECTED values exist —
    only a human reviewer can move a claim to a HUMAN_* final state."""

    SUBMITTED = "SUBMITTED"
    ELIGIBLE_FOR_HUMAN_REVIEW = "ELIGIBLE_FOR_HUMAN_REVIEW"
    ADDITIONAL_INFORMATION_REQUIRED = "ADDITIONAL_INFORMATION_REQUIRED"
    POLICY_NOT_FOUND = "POLICY_NOT_FOUND"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    HUMAN_APPROVED = "HUMAN_APPROVED"
    HUMAN_REJECTED = "HUMAN_REJECTED"
    HUMAN_INFO_REQUESTED = "HUMAN_INFO_REQUESTED"


class ProcessingStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class RiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class DocumentType(StrEnum):
    CLAIM_FORM = "CLAIM_FORM"
    HOSPITAL_BILL = "HOSPITAL_BILL"
    DISCHARGE_SUMMARY = "DISCHARGE_SUMMARY"
    MEDICAL_REPORT = "MEDICAL_REPORT"
    DIAGNOSTIC_REPORT = "DIAGNOSTIC_REPORT"
    INVOICE = "INVOICE"
    POLICY_DOCUMENT = "POLICY_DOCUMENT"
    OTHER = "OTHER"


class AgentStatus(StrEnum):
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class ReviewDecision(StrEnum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    INFO_REQUESTED = "INFO_REQUESTED"
    ESCALATED = "ESCALATED"


class ClauseCategory(StrEnum):
    COVERAGE = "COVERAGE"
    EXCLUSION = "EXCLUSION"
    WAITING_PERIOD = "WAITING_PERIOD"
    CONDITION = "CONDITION"
    DEFINITION = "DEFINITION"
    GENERAL = "GENERAL"


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
class Customer(TimestampMixin, Base):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(primary_key=True)
    customer_code: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(120), index=True)
    date_of_birth: Mapped[date | None] = mapped_column(Date)
    email: Mapped[str | None] = mapped_column(String(160))
    phone: Mapped[str | None] = mapped_column(String(30))
    city: Mapped[str | None] = mapped_column(String(80))
    state: Mapped[str | None] = mapped_column(String(80))

    policies: Mapped[list[Policy]] = relationship(back_populates="customer", cascade="all, delete-orphan")
    claims: Mapped[list[Claim]] = relationship(back_populates="customer")

    def __repr__(self) -> str:
        return f"<Customer {self.customer_code} {self.full_name}>"


class Policy(TimestampMixin, Base):
    __tablename__ = "policies"
    __table_args__ = (
        CheckConstraint("sum_insured > 0", name="ck_policy_sum_insured_positive"),
        CheckConstraint("end_date >= start_date", name="ck_policy_dates"),
        CheckConstraint("waiting_period_days >= 0", name="ck_policy_waiting"),
        CheckConstraint(_in("status", PolicyStatus), name="ck_policy_status"),
        CheckConstraint(_in("insurance_type", InsuranceType), name="ck_policy_type"),
        Index("ix_policies_status_type", "status", "insurance_type"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    policy_number: Mapped[str] = mapped_column(String(30), unique=True, index=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id", ondelete="CASCADE"), index=True)
    insurance_type: Mapped[str] = mapped_column(String(40))
    plan_name: Mapped[str] = mapped_column(String(80))
    plan_code: Mapped[str] = mapped_column(String(20))
    sum_insured: Mapped[float] = mapped_column(Numeric(14, 2, asdecimal=False))
    annual_premium: Mapped[float] = mapped_column(Numeric(12, 2, asdecimal=False))
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date] = mapped_column(Date)
    waiting_period_days: Mapped[int] = mapped_column(Integer, default=30)
    status: Mapped[str] = mapped_column(String(20), default=PolicyStatus.ACTIVE.value)
    hospitalization_covered: Mapped[bool] = mapped_column(Boolean, default=True)
    exclusions: Mapped[list[Any]] = mapped_column(JSON, default=list)
    policy_document: Mapped[str | None] = mapped_column(String(200))

    customer: Mapped[Customer] = relationship(back_populates="policies")
    claims: Mapped[list[Claim]] = relationship(back_populates="policy")

    def __repr__(self) -> str:
        return f"<Policy {self.policy_number} {self.status}>"


class Claim(TimestampMixin, Base):
    __tablename__ = "claims"
    __table_args__ = (
        CheckConstraint("claim_amount >= 0", name="ck_claim_amount_non_negative"),
        CheckConstraint(_in("claim_status", ClaimStatus), name="ck_claim_status"),
        CheckConstraint(_in("processing_status", ProcessingStatus), name="ck_claim_processing_status"),
        CheckConstraint("risk_level IS NULL OR " + _in("risk_level", RiskLevel), name="ck_claim_risk_level"),
        Index("ix_claims_status_submitted", "claim_status", "submitted_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    claim_number: Mapped[str] = mapped_column(String(30), unique=True, index=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), index=True)
    # Nullable on purpose: a claim may quote a policy number that does not exist.
    policy_id: Mapped[int | None] = mapped_column(ForeignKey("policies.id"), index=True)

    # What the claimant *submitted* (kept even if it disagrees with our records).
    customer_name_submitted: Mapped[str] = mapped_column(String(120))
    policy_number_submitted: Mapped[str] = mapped_column(String(30), index=True)
    insurance_type: Mapped[str] = mapped_column(String(40), index=True)
    claim_amount: Mapped[float] = mapped_column(Numeric(14, 2, asdecimal=False))
    hospital_name: Mapped[str | None] = mapped_column(String(160))
    admission_date: Mapped[date | None] = mapped_column(Date)
    discharge_date: Mapped[date | None] = mapped_column(Date)
    diagnosis: Mapped[str | None] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text)
    invoice_number: Mapped[str | None] = mapped_column(String(40), index=True)
    billed_amount: Mapped[float | None] = mapped_column(Numeric(14, 2, asdecimal=False))

    claim_status: Mapped[str] = mapped_column(String(40), default=ClaimStatus.SUBMITTED.value, index=True)
    processing_status: Mapped[str] = mapped_column(String(20), default=ProcessingStatus.PENDING.value)
    risk_level: Mapped[str | None] = mapped_column(String(10), index=True)
    risk_score: Mapped[float | None] = mapped_column(Float)

    # Agent outputs (JSON snapshots of the latest workflow run)
    extracted_data: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    policy_verification: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    retrieved_clauses: Mapped[list[Any] | None] = mapped_column(JSON)
    validation_result: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    risk_assessment: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    decision_support: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    processing_time_seconds: Mapped[float | None] = mapped_column(Float)
    scenario: Mapped[str | None] = mapped_column(String(40))  # synthetic test-scenario tag

    customer: Mapped[Customer] = relationship(back_populates="claims")
    policy: Mapped[Policy | None] = relationship(back_populates="claims")
    documents: Mapped[list[ClaimDocument]] = relationship(back_populates="claim", cascade="all, delete-orphan")
    agent_executions: Mapped[list[AgentExecution]] = relationship(
        back_populates="claim", cascade="all, delete-orphan", order_by="AgentExecution.id"
    )
    risk_flags: Mapped[list[RiskFlag]] = relationship(back_populates="claim", cascade="all, delete-orphan")
    reviews: Mapped[list[HumanReview]] = relationship(
        back_populates="claim", cascade="all, delete-orphan", order_by="HumanReview.reviewed_at"
    )
    logs: Mapped[list[ProcessingLog]] = relationship(back_populates="claim", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Claim {self.claim_number} {self.claim_status}>"


class ClaimDocument(Base):
    __tablename__ = "claim_documents"
    __table_args__ = (CheckConstraint("file_size_bytes >= 0", name="ck_doc_size"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    claim_id: Mapped[int] = mapped_column(ForeignKey("claims.id", ondelete="CASCADE"), index=True)
    document_type: Mapped[str] = mapped_column(String(30), default=DocumentType.OTHER.value)
    original_filename: Mapped[str] = mapped_column(String(255))
    stored_filename: Mapped[str] = mapped_column(String(255))
    file_path: Mapped[str] = mapped_column(String(500))
    file_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    sha256: Mapped[str | None] = mapped_column(String(64), index=True)  # duplicate-upload detection
    page_count: Mapped[int | None] = mapped_column(Integer)
    extraction_method: Mapped[str | None] = mapped_column(String(20))  # TEXT | OCR | NONE
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    claim: Mapped[Claim] = relationship(back_populates="documents")


class AgentExecution(Base):
    """One row per agent per workflow run — the audit trail of the AI pipeline."""

    __tablename__ = "agent_executions"
    __table_args__ = (
        CheckConstraint(_in("status", AgentStatus), name="ck_agent_status"),
        Index("ix_agent_exec_claim_run", "claim_id", "run_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    claim_id: Mapped[int] = mapped_column(ForeignKey("claims.id", ondelete="CASCADE"), index=True)
    run_id: Mapped[str] = mapped_column(String(36))
    agent_name: Mapped[str] = mapped_column(String(60), index=True)
    sequence: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[float | None] = mapped_column(Float)
    used_llm: Mapped[bool] = mapped_column(Boolean, default=False)  # honest LLM-vs-rules flag
    model_name: Mapped[str | None] = mapped_column(String(60))
    input_summary: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    output: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    error_message: Mapped[str | None] = mapped_column(Text)

    claim: Mapped[Claim] = relationship(back_populates="agent_executions")


class PolicyClause(Base):
    """Clause extracted from a policy PDF; indexed into FAISS for RAG."""

    __tablename__ = "policy_clauses"
    __table_args__ = (CheckConstraint(_in("category", ClauseCategory), name="ck_clause_category"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    source_document: Mapped[str] = mapped_column(String(255), index=True)
    plan_code: Mapped[str | None] = mapped_column(String(20), index=True)
    clause_code: Mapped[str | None] = mapped_column(String(20))
    title: Mapped[str | None] = mapped_column(String(200))
    category: Mapped[str] = mapped_column(String(20), default=ClauseCategory.GENERAL.value)
    text: Mapped[str] = mapped_column(Text)
    page_number: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class RiskFlag(Base):
    """Explainable risk indicator — a reason to investigate, never an accusation."""

    __tablename__ = "risk_flags"
    __table_args__ = (CheckConstraint(_in("severity", RiskLevel), name="ck_risk_severity"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    claim_id: Mapped[int] = mapped_column(ForeignKey("claims.id", ondelete="CASCADE"), index=True)
    code: Mapped[str] = mapped_column(String(40), index=True)
    severity: Mapped[str] = mapped_column(String(10))
    description: Mapped[str] = mapped_column(Text)
    evidence: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    claim: Mapped[Claim] = relationship(back_populates="risk_flags")


class HumanReview(Base):
    """A human reviewer's decision. Stored separately from AI recommendations."""

    __tablename__ = "human_reviews"
    __table_args__ = (CheckConstraint(_in("decision", ReviewDecision), name="ck_review_decision"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    claim_id: Mapped[int] = mapped_column(ForeignKey("claims.id", ondelete="CASCADE"), index=True)
    reviewer_name: Mapped[str] = mapped_column(String(120))
    reviewer_role: Mapped[str] = mapped_column(String(40), default="CLAIMS_REVIEWER")
    ai_recommendation_snapshot: Mapped[str | None] = mapped_column(String(40))
    decision: Mapped[str] = mapped_column(String(20))
    comments: Mapped[str | None] = mapped_column(Text)
    reviewed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    claim: Mapped[Claim] = relationship(back_populates="reviews")


class ProcessingLog(Base):
    __tablename__ = "processing_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    claim_id: Mapped[int | None] = mapped_column(ForeignKey("claims.id", ondelete="CASCADE"), index=True)
    level: Mapped[str] = mapped_column(String(10), default="INFO")
    source: Mapped[str] = mapped_column(String(60))
    event: Mapped[str] = mapped_column(String(60))
    message: Mapped[str] = mapped_column(Text)
    details: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    claim: Mapped[Claim | None] = relationship(back_populates="logs")
