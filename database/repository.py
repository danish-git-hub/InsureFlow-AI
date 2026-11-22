"""
Repository layer — the ONLY place that writes SQL queries.

Beginner note (repository pattern):
    Pages, API routes and agents never build queries themselves. They call
    methods like `ClaimRepository(session).get_by_number("CLM-...")`. This
    keeps business code readable, makes queries reusable and testable, and
    means swapping SQLite for PostgreSQL touches nothing outside this layer.

Each repository receives a `Session`; the caller controls commit/rollback
(see `database.connection.session_scope`).
"""

from __future__ import annotations

import secrets
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy import case, func, or_, select
from sqlalchemy.orm import Session, selectinload

from database.models import (
    AgentExecution,
    Claim,
    ClaimDocument,
    ClaimStatus,
    Customer,
    HumanReview,
    Policy,
    PolicyClause,
    ProcessingLog,
    ProcessingStatus,
    RiskFlag,
)

# Claim statuses that mean "waiting for a human reviewer".
REVIEW_QUEUE_STATUSES = (
    ClaimStatus.ELIGIBLE_FOR_HUMAN_REVIEW.value,
    ClaimStatus.ADDITIONAL_INFORMATION_REQUIRED.value,
    ClaimStatus.POLICY_NOT_FOUND.value,
    ClaimStatus.REVIEW_REQUIRED.value,
)
HUMAN_FINAL_STATUSES = (
    ClaimStatus.HUMAN_APPROVED.value,
    ClaimStatus.HUMAN_REJECTED.value,
    ClaimStatus.HUMAN_INFO_REQUESTED.value,
)


class _Repo:
    def __init__(self, session: Session):
        self.session = session


# --------------------------------------------------------------------------- #
class CustomerRepository(_Repo):
    def create(self, **fields: Any) -> Customer:
        customer = Customer(**fields)
        self.session.add(customer)
        self.session.flush()  # assigns customer.id without committing
        return customer

    def get(self, customer_id: int) -> Customer | None:
        return self.session.get(Customer, customer_id)

    def get_by_code(self, code: str) -> Customer | None:
        return self.session.scalar(select(Customer).where(Customer.customer_code == code))

    def list(self, search: str | None = None, limit: int = 500) -> Sequence[Customer]:
        stmt = select(Customer).order_by(Customer.customer_code).limit(limit)
        if search:
            like = f"%{search.strip()}%"
            stmt = stmt.where(or_(Customer.full_name.ilike(like), Customer.customer_code.ilike(like)))
        return self.session.scalars(stmt).all()

    def count(self) -> int:
        return self.session.scalar(select(func.count(Customer.id))) or 0

    def next_code(self) -> str:
        last = self.session.scalar(select(func.max(Customer.id))) or 0
        return f"CUST-{last + 1:04d}"


# --------------------------------------------------------------------------- #
class PolicyRepository(_Repo):
    def create(self, **fields: Any) -> Policy:
        policy = Policy(**fields)
        self.session.add(policy)
        self.session.flush()
        return policy

    def get(self, policy_id: int) -> Policy | None:
        return self.session.get(Policy, policy_id)

    def get_by_number(self, policy_number: str) -> Policy | None:
        if not policy_number:
            return None
        stmt = (
            select(Policy)
            .options(selectinload(Policy.customer))
            .where(Policy.policy_number == policy_number.strip().upper())
        )
        return self.session.scalar(stmt)

    def list(
        self,
        status: str | None = None,
        insurance_type: str | None = None,
        search: str | None = None,
        limit: int = 1000,
    ) -> Sequence[Policy]:
        stmt = select(Policy).options(selectinload(Policy.customer)).join(Customer)
        if status:
            stmt = stmt.where(Policy.status == status)
        if insurance_type:
            stmt = stmt.where(Policy.insurance_type == insurance_type)
        if search:
            like = f"%{search.strip()}%"
            stmt = stmt.where(or_(Policy.policy_number.ilike(like), Customer.full_name.ilike(like)))
        return self.session.scalars(stmt.order_by(Policy.policy_number).limit(limit)).all()

    def for_customer(self, customer_id: int) -> Sequence[Policy]:
        return self.session.scalars(select(Policy).where(Policy.customer_id == customer_id)).all()

    def update_status(self, policy_number: str, status: str) -> Policy | None:
        policy = self.get_by_number(policy_number)
        if policy:
            policy.status = status
        return policy

    def count(self) -> int:
        return self.session.scalar(select(func.count(Policy.id))) or 0


# --------------------------------------------------------------------------- #
class ClaimRepository(_Repo):
    @staticmethod
    def generate_claim_number(now: datetime | None = None) -> str:
        """Unique, non-guessable claim ID such as CLM-20261120-4F9A2C."""
        now = now or datetime.now(timezone.utc)
        return f"CLM-{now:%Y%m%d}-{secrets.token_hex(3).upper()}"

    def create(self, **fields: Any) -> Claim:
        fields.setdefault("claim_number", self._unique_claim_number())
        claim = Claim(**fields)
        self.session.add(claim)
        self.session.flush()
        return claim

    def _unique_claim_number(self) -> str:
        while True:
            number = self.generate_claim_number()
            if not self.get_by_number(number):
                return number

    def get(self, claim_id: int) -> Claim | None:
        return self.session.get(Claim, claim_id)

    def get_by_number(self, claim_number: str) -> Claim | None:
        stmt = (
            select(Claim)
            .options(
                selectinload(Claim.customer),
                selectinload(Claim.policy),
                selectinload(Claim.documents),
                selectinload(Claim.risk_flags),
                selectinload(Claim.reviews),
            )
            .where(Claim.claim_number == claim_number)
        )
        return self.session.scalar(stmt)

    def list(
        self,
        status: str | Iterable[str] | None = None,
        insurance_type: str | None = None,
        risk_level: str | None = None,
        search: str | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
        limit: int = 1000,
    ) -> Sequence[Claim]:
        stmt = select(Claim).options(selectinload(Claim.customer))
        if status:
            stmt = stmt.where(Claim.claim_status == status) if isinstance(status, str) else stmt.where(
                Claim.claim_status.in_(list(status))
            )
        if insurance_type:
            stmt = stmt.where(Claim.insurance_type == insurance_type)
        if risk_level:
            stmt = stmt.where(Claim.risk_level == risk_level)
        if search:
            like = f"%{search.strip()}%"
            stmt = stmt.where(
                or_(
                    Claim.claim_number.ilike(like),
                    Claim.customer_name_submitted.ilike(like),
                    Claim.policy_number_submitted.ilike(like),
                    Claim.hospital_name.ilike(like),
                )
            )
        if date_from:
            stmt = stmt.where(Claim.submitted_at >= datetime.combine(date_from, datetime.min.time()))
        if date_to:
            stmt = stmt.where(Claim.submitted_at <= datetime.combine(date_to, datetime.max.time()))
        return self.session.scalars(stmt.order_by(Claim.submitted_at.desc()).limit(limit)).all()

    def pending_processing(self) -> Sequence[Claim]:
        return self.list(status=ClaimStatus.SUBMITTED.value)

    def review_queue(self) -> Sequence[Claim]:
        """Claims the AI has finished with that still need a human decision."""
        return self.list(status=REVIEW_QUEUE_STATUSES)

    def find_by_invoice(self, invoice_number: str, exclude_claim_id: int | None = None) -> Sequence[Claim]:
        """Other claims quoting the same invoice number (duplicate detection)."""
        if not invoice_number:
            return []
        stmt = select(Claim).where(func.upper(Claim.invoice_number) == invoice_number.strip().upper())
        if exclude_claim_id is not None:
            stmt = stmt.where(Claim.id != exclude_claim_id)
        return self.session.scalars(stmt).all()

    def for_policy(self, policy_id: int, exclude_claim_id: int | None = None) -> Sequence[Claim]:
        stmt = select(Claim).where(Claim.policy_id == policy_id)
        if exclude_claim_id is not None:
            stmt = stmt.where(Claim.id != exclude_claim_id)
        return self.session.scalars(stmt).all()

    def set_status(self, claim: Claim, claim_status: str | None = None, processing_status: str | None = None) -> Claim:
        if claim_status:
            claim.claim_status = claim_status
        if processing_status:
            claim.processing_status = processing_status
        return claim

    def save_agent_results(self, claim: Claim, **results: Any) -> Claim:
        """Store JSON outputs from the workflow (extracted_data, decision_support, ...)."""
        allowed = {
            "extracted_data", "policy_verification", "retrieved_clauses", "validation_result",
            "risk_assessment", "decision_support", "risk_level", "risk_score",
            "processing_time_seconds", "processed_at", "claim_status", "processing_status",
        }
        for key, value in results.items():
            if key not in allowed:
                raise ValueError(f"Unknown claim result field: {key}")
            setattr(claim, key, value)
        return claim

    def count(self) -> int:
        return self.session.scalar(select(func.count(Claim.id))) or 0


# --------------------------------------------------------------------------- #
class DocumentRepository(_Repo):
    def add(self, **fields: Any) -> ClaimDocument:
        doc = ClaimDocument(**fields)
        self.session.add(doc)
        self.session.flush()
        return doc

    def for_claim(self, claim_id: int) -> Sequence[ClaimDocument]:
        return self.session.scalars(select(ClaimDocument).where(ClaimDocument.claim_id == claim_id)).all()

    def find_by_hash(self, sha256: str, exclude_claim_id: int | None = None) -> Sequence[ClaimDocument]:
        stmt = select(ClaimDocument).where(ClaimDocument.sha256 == sha256)
        if exclude_claim_id is not None:
            stmt = stmt.where(ClaimDocument.claim_id != exclude_claim_id)
        return self.session.scalars(stmt).all()


# --------------------------------------------------------------------------- #
class AgentExecutionRepository(_Repo):
    def record(self, **fields: Any) -> AgentExecution:
        execution = AgentExecution(**fields)
        self.session.add(execution)
        self.session.flush()
        return execution

    def for_claim(self, claim_id: int, run_id: str | None = None) -> Sequence[AgentExecution]:
        stmt = select(AgentExecution).where(AgentExecution.claim_id == claim_id)
        if run_id:
            stmt = stmt.where(AgentExecution.run_id == run_id)
        return self.session.scalars(stmt.order_by(AgentExecution.id)).all()

    def latest_run(self, claim_id: int) -> Sequence[AgentExecution]:
        last = self.session.scalar(
            select(AgentExecution.run_id)
            .where(AgentExecution.claim_id == claim_id)
            .order_by(AgentExecution.id.desc())
            .limit(1)
        )
        return self.for_claim(claim_id, last) if last else []

    def recent(self, limit: int = 50) -> Sequence[AgentExecution]:
        return self.session.scalars(select(AgentExecution).order_by(AgentExecution.id.desc()).limit(limit)).all()


# --------------------------------------------------------------------------- #
class PolicyClauseRepository(_Repo):
    def replace_for_document(self, source_document: str, clauses: Iterable[dict[str, Any]]) -> int:
        """Re-index a policy PDF: delete its old clauses, insert new ones."""
        for old in self.session.scalars(select(PolicyClause).where(PolicyClause.source_document == source_document)):
            self.session.delete(old)
        count = 0
        for clause in clauses:
            self.session.add(PolicyClause(source_document=source_document, **clause))
            count += 1
        self.session.flush()
        return count

    def list(self, plan_code: str | None = None, category: str | None = None) -> Sequence[PolicyClause]:
        stmt = select(PolicyClause)
        if plan_code:
            stmt = stmt.where(PolicyClause.plan_code == plan_code)
        if category:
            stmt = stmt.where(PolicyClause.category == category)
        return self.session.scalars(stmt.order_by(PolicyClause.source_document, PolicyClause.id)).all()

    def count(self) -> int:
        return self.session.scalar(select(func.count(PolicyClause.id))) or 0


# --------------------------------------------------------------------------- #
class RiskFlagRepository(_Repo):
    def replace_for_claim(self, claim_id: int, flags: Iterable[dict[str, Any]]) -> list[RiskFlag]:
        """Each workflow run replaces the previous run's flags for that claim."""
        for old in self.for_claim(claim_id):
            self.session.delete(old)
        new = [RiskFlag(claim_id=claim_id, **f) for f in flags]
        self.session.add_all(new)
        self.session.flush()
        return new

    def for_claim(self, claim_id: int) -> Sequence[RiskFlag]:
        return self.session.scalars(select(RiskFlag).where(RiskFlag.claim_id == claim_id)).all()


# --------------------------------------------------------------------------- #
class ReviewRepository(_Repo):
    DECISION_TO_STATUS = {
        "APPROVED": ClaimStatus.HUMAN_APPROVED.value,
        "REJECTED": ClaimStatus.HUMAN_REJECTED.value,
        "INFO_REQUESTED": ClaimStatus.HUMAN_INFO_REQUESTED.value,
        "ESCALATED": ClaimStatus.REVIEW_REQUIRED.value,
    }

    def record(self, claim: Claim, reviewer_name: str, decision: str, comments: str | None = None,
               reviewer_role: str = "CLAIMS_REVIEWER") -> HumanReview:
        """Record a human decision and move the claim to the matching human status.

        The AI recommendation is snapshotted so the audit trail always shows
        what the AI suggested next to what the human decided.
        """
        if decision not in self.DECISION_TO_STATUS:
            raise ValueError(f"Invalid review decision: {decision}")
        ai_rec = (claim.decision_support or {}).get("recommendation") or claim.claim_status
        review = HumanReview(
            claim_id=claim.id,
            reviewer_name=reviewer_name,
            reviewer_role=reviewer_role,
            ai_recommendation_snapshot=ai_rec,
            decision=decision,
            comments=comments,
        )
        self.session.add(review)
        claim.claim_status = self.DECISION_TO_STATUS[decision]
        self.session.flush()
        return review

    def for_claim(self, claim_id: int) -> Sequence[HumanReview]:
        return self.session.scalars(
            select(HumanReview).where(HumanReview.claim_id == claim_id).order_by(HumanReview.reviewed_at)
        ).all()

    def recent(self, limit: int = 100) -> Sequence[HumanReview]:
        return self.session.scalars(select(HumanReview).order_by(HumanReview.reviewed_at.desc()).limit(limit)).all()


# --------------------------------------------------------------------------- #
class LogRepository(_Repo):
    def log(self, source: str, event: str, message: str, claim_id: int | None = None,
            level: str = "INFO", details: dict[str, Any] | None = None) -> ProcessingLog:
        entry = ProcessingLog(claim_id=claim_id, source=source, event=event, message=message,
                              level=level, details=details)
        self.session.add(entry)
        self.session.flush()
        return entry

    def recent(self, limit: int = 50, claim_id: int | None = None) -> Sequence[ProcessingLog]:
        stmt = select(ProcessingLog)
        if claim_id is not None:
            stmt = stmt.where(ProcessingLog.claim_id == claim_id)
        return self.session.scalars(stmt.order_by(ProcessingLog.id.desc()).limit(limit)).all()


# --------------------------------------------------------------------------- #
class AnalyticsRepository(_Repo):
    """Read-only aggregates for the dashboard and reports.

    Aggregations are done with portable SQL (COUNT/SUM/GROUP BY) or in Python
    for date bucketing, so they work on both SQLite and PostgreSQL.
    """

    def kpis(self) -> dict[str, Any]:
        s = self.session
        total = s.scalar(select(func.count(Claim.id))) or 0
        total_amount = s.scalar(select(func.coalesce(func.sum(Claim.claim_amount), 0))) or 0
        under_review = s.scalar(select(func.count(Claim.id)).where(Claim.claim_status.in_(REVIEW_QUEUE_STATUSES))) or 0
        info_required = s.scalar(
            select(func.count(Claim.id)).where(
                Claim.claim_status == ClaimStatus.ADDITIONAL_INFORMATION_REQUIRED.value
            )
        ) or 0
        processed = s.scalar(
            select(func.count(Claim.id)).where(Claim.processing_status == ProcessingStatus.COMPLETED.value)
        ) or 0
        avg_time = s.scalar(
            select(func.avg(Claim.processing_time_seconds)).where(Claim.processing_time_seconds.is_not(None))
        )
        return {
            "total_claims": total,
            "claims_under_review": under_review,
            "claims_processed": processed,
            "additional_info_required": info_required,
            "total_claim_amount": float(total_amount),
            "avg_processing_seconds": float(avg_time) if avg_time is not None else None,
            "pending_processing": s.scalar(
                select(func.count(Claim.id)).where(Claim.processing_status == ProcessingStatus.PENDING.value)
            ) or 0,
        }

    def _group_count(self, column) -> dict[str, int]:
        rows = self.session.execute(select(column, func.count()).group_by(column)).all()
        return {str(k) if k is not None else "UNASSESSED": int(v) for k, v in rows}

    def status_distribution(self) -> dict[str, int]:
        return self._group_count(Claim.claim_status)

    def claims_by_type(self) -> dict[str, int]:
        return self._group_count(Claim.insurance_type)

    def risk_distribution(self) -> dict[str, int]:
        return self._group_count(Claim.risk_level)

    def monthly_claims(self) -> list[dict[str, Any]]:
        """[{month: '2026-05', claims: 4, amount: 312000.0}, ...] sorted by month."""
        buckets: dict[str, dict[str, float]] = defaultdict(lambda: {"claims": 0, "amount": 0.0})
        for submitted_at, amount in self.session.execute(select(Claim.submitted_at, Claim.claim_amount)):
            key = submitted_at.strftime("%Y-%m")
            buckets[key]["claims"] += 1
            buckets[key]["amount"] += float(amount or 0)
        return [{"month": k, "claims": int(v["claims"]), "amount": v["amount"]} for k, v in sorted(buckets.items())]

    def agent_performance(self) -> list[dict[str, Any]]:
        rows = self.session.execute(
            select(
                AgentExecution.agent_name,
                func.count(AgentExecution.id),
                func.avg(AgentExecution.duration_ms),
                func.sum(case((AgentExecution.status == "FAILED", 1), else_=0)),
            ).group_by(AgentExecution.agent_name)
        ).all()
        return [
            {"agent": name, "runs": int(runs), "avg_ms": float(avg or 0), "failures": int(fails or 0)}
            for name, runs, avg, fails in rows
        ]

    def review_workload(self) -> dict[str, int]:
        return dict(Counter(r.decision for r in self.session.scalars(select(HumanReview))))

    def table_counts(self) -> dict[str, int]:
        from database.models import Base

        counts = {}
        for table in Base.metadata.sorted_tables:
            counts[table.name] = self.session.scalar(select(func.count()).select_from(table)) or 0
        return counts
