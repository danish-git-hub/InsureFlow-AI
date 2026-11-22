"""
Pydantic schemas shared by the agents.

Beginner note:
    Pydantic models are typed "contracts". When an agent returns a
    `DocumentExtractionResult`, Pydantic guarantees the types are right (dates
    are dates, amounts are numbers). LLM output is validated against the same
    models, so malformed AI output is rejected instead of silently stored.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator

ExtractionMethod = Literal["TEXT", "OCR", "TEXT+OCR", "NONE"]
FieldSource = Literal["rules", "gemini"]

# Fields the Document Extraction Agent tries to find.
CLAIM_FIELDS: tuple[str, ...] = (
    "customer_name", "customer_id", "policy_number", "claim_amount", "billed_amount", "hospital_name",
    "invoice_number", "admission_date", "discharge_date", "diagnosis",
)
REQUIRED_CLAIM_FIELDS: tuple[str, ...] = (
    "customer_name", "policy_number", "claim_amount", "billed_amount", "hospital_name",
    "invoice_number", "admission_date", "discharge_date", "diagnosis",
)


class BillItem(BaseModel):
    description: str
    amount: float = Field(ge=0)


class ClaimFields(BaseModel):
    """Structured claim information (every field optional: documents may lack it)."""

    customer_name: str | None = None
    customer_id: str | None = None
    policy_number: str | None = None
    claim_amount: float | None = Field(default=None, ge=0)
    billed_amount: float | None = Field(default=None, ge=0)
    hospital_name: str | None = None
    invoice_number: str | None = None
    admission_date: date | None = None
    discharge_date: date | None = None
    diagnosis: str | None = None

    @field_validator("policy_number", "invoice_number", "customer_id")
    @classmethod
    def _upper(cls, v: str | None) -> str | None:
        return v.strip().upper() if v else v


class FieldConflict(BaseModel):
    """Two documents disagree on the same field (evidence for validation/risk agents)."""

    field: str
    values: dict[str, str]  # document file name -> value as printed


class ExtractedDocument(BaseModel):
    filename: str
    document_type: str
    declared_type: str | None = None
    page_count: int
    extraction_method: ExtractionMethod
    ocr_confidence: float | None = None
    characters: int
    fields: dict[str, str | float | None] = Field(default_factory=dict)
    bill_items: list[BillItem] = Field(default_factory=list)
    text_preview: str = ""
    warnings: list[str] = Field(default_factory=list)


class DocumentExtractionResult(BaseModel):
    """Output of Agent 1 — the Document Extraction Agent."""

    fields: ClaimFields
    field_sources: dict[str, FieldSource] = Field(default_factory=dict)
    field_documents: dict[str, str] = Field(default_factory=dict)  # field -> source file
    bill_items: list[BillItem] = Field(default_factory=list)
    documents: list[ExtractedDocument] = Field(default_factory=list)
    document_types_found: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    conflicts: list[FieldConflict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    extraction_mode: Literal["RULE_BASED", "RULE_BASED+GEMINI"] = "RULE_BASED"
    used_llm: bool = False
    llm_model: str | None = None
    ocr_used: bool = False

    @property
    def complete(self) -> bool:
        return not self.missing_fields


# =========================================================================== #
# Agent 2 — Policy Verification
# =========================================================================== #
CheckStatus = Literal["PASS", "WARN", "FAIL", "NOT_EVALUATED", "NOT_APPLICABLE"]
VerificationStatus = Literal[
    "VERIFIED", "VERIFIED_WITH_WARNINGS", "FAILED", "POLICY_NOT_FOUND", "INSUFFICIENT_DATA"
]


class PolicyVerificationInput(BaseModel):
    """Claim facts the policy rules need. `sources` records where each came from."""

    policy_number: str | None = None
    customer_name: str | None = None
    customer_id: str | None = None
    insurance_type: str | None = None
    claim_amount: float | None = Field(default=None, ge=0)
    admission_date: date | None = None
    discharge_date: date | None = None
    diagnosis: str | None = None
    claim_number: str | None = None  # excluded when summing other claims on the policy
    sources: dict[str, str] = Field(default_factory=dict)


class PolicySnapshot(BaseModel):
    policy_number: str
    holder_name: str
    customer_code: str
    insurance_type: str
    plan_name: str
    plan_code: str
    sum_insured: float
    start_date: date
    end_date: date
    waiting_period_days: int
    status: str
    hospitalization_covered: bool
    exclusions: list[str]
    copay_percent: float = 0.0


class RuleCheck(BaseModel):
    """One deterministic business rule and its outcome — the unit of explainability."""

    code: str
    name: str
    status: CheckStatus
    message: str
    evidence: dict[str, str | float | int | bool | None] = Field(default_factory=dict)


class CoverageSummary(BaseModel):
    """Indicative figures only — the final payable amount is decided by a human assessor."""

    sum_insured: float
    other_claims_on_policy: float
    remaining_sum_insured: float
    claim_amount: float | None
    copay_percent: float
    indicative_admissible_amount: float | None


class PolicyVerificationResult(BaseModel):
    """Output of Agent 2 — the Policy Verification Agent."""

    status: VerificationStatus
    policy: PolicySnapshot | None = None
    checks: list[RuleCheck] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    flags: dict[str, bool] = Field(default_factory=dict)  # consumed by LangGraph routing
    coverage: CoverageSummary | None = None
    suggestions: list[str] = Field(default_factory=list)
    summary: str = ""
    mode: Literal["DETERMINISTIC_RULES"] = "DETERMINISTIC_RULES"
    used_llm: bool = False

    def check(self, code: str) -> RuleCheck | None:
        return next((c for c in self.checks if c.code == code), None)


# =========================================================================== #
# Agent 3 — Policy RAG
# =========================================================================== #
RAGStatus = Literal["EVIDENCE_FOUND", "INSUFFICIENT_EVIDENCE", "POLICY_CONTEXT_UNAVAILABLE"]


class PolicyRAGInput(BaseModel):
    plan_code: str | None = None
    policy_number: str | None = None
    insurance_type: str | None = None
    diagnosis: str | None = None
    policy_flags: dict[str, bool] = Field(default_factory=dict)  # from Agent 2


class RetrievedClause(BaseModel):
    """A piece of real policy wording with its citation. Text is copied, never generated."""

    topic: str
    clause_code: str | None
    title: str | None
    category: str
    plan_code: str | None
    source_document: str
    page_number: int
    score: float
    text: str
    matched_terms: list[str] = Field(default_factory=list)

    @property
    def citation(self) -> str:
        return f"{self.plan_code} {self.clause_code} '{self.title}', {self.source_document} p.{self.page_number}"


class RAGQuery(BaseModel):
    topic: str
    query: str
    categories: list[str]
    hits: int


class PolicyRAGResult(BaseModel):
    """Output of Agent 3 — the Policy RAG Agent."""

    status: RAGStatus
    plan_code: str | None = None
    queries: list[RAGQuery] = Field(default_factory=list)
    clauses: list[RetrievedClause] = Field(default_factory=list)
    possible_exclusions: list[str] = Field(default_factory=list)  # clause codes whose wording matches the diagnosis
    findings: list[str] = Field(default_factory=list)
    uncertainty_reasons: list[str] = Field(default_factory=list)
    requires_human_review: bool = True
    flags: dict[str, bool] = Field(default_factory=dict)
    explanation: str = ""
    explanation_source: Literal["template", "gemini"] = "template"
    warnings: list[str] = Field(default_factory=list)
    embedding_model: str = ""
    retrieval_mode: Literal["FAISS"] = "FAISS"
    min_score: float = 0.0
    used_llm: bool = False
    llm_model: str | None = None


# =========================================================================== #
# Agent 4 — Validation
# =========================================================================== #
ValidationStatus = Literal["VALID", "VALID_WITH_WARNINGS", "INCOMPLETE", "INVALID"]


class ValidationInput(BaseModel):
    """What the claimant submitted, what the documents say, and which documents arrived."""

    claim_number: str | None = None
    claim_id: int | None = None
    submitted: ClaimFields
    extracted: ClaimFields | None = None  # values read from the documents
    documents_read: bool = True  # False when only stored values stand in for the PDFs
    document_types: list[str] = Field(default_factory=list)
    conflicts: list[FieldConflict] = Field(default_factory=list)
    extraction_missing_fields: list[str] = Field(default_factory=list)
    bill_items_total: float | None = None
    submitted_on: date | None = None

    def value(self, field: str):
        """Document value if available, otherwise the submitted value."""
        doc = getattr(self.extracted, field, None) if self.extracted else None
        return doc if doc not in (None, "") else getattr(self.submitted, field, None)


class ValidationResult(BaseModel):
    """Output of Agent 4 — the Validation Agent."""

    status: ValidationStatus
    checks: list[RuleCheck] = Field(default_factory=list)
    missing_documents: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    duplicate_claims: list[str] = Field(default_factory=list)
    flags: dict[str, bool] = Field(default_factory=dict)  # consumed by LangGraph routing
    summary: str = ""
    mode: Literal["DETERMINISTIC_RULES"] = "DETERMINISTIC_RULES"
    used_llm: bool = False

    def check(self, code: str) -> RuleCheck | None:
        return next((c for c in self.checks if c.code == code), None)


# =========================================================================== #
# Agent 5 — Fraud Risk (indicators for investigation, never accusations)
# =========================================================================== #
RiskLevelName = Literal["LOW", "MEDIUM", "HIGH"]


class RiskIndicator(BaseModel):
    code: str
    severity: RiskLevelName
    points: int
    description: str  # neutral wording: what was observed, not what anyone intended
    evidence: dict[str, str | float | int | bool | None] = Field(default_factory=dict)


class RiskAssessment(BaseModel):
    """Output of Agent 5 — the Fraud Risk Agent."""

    risk_score: int = Field(ge=0, le=100)
    risk_level: RiskLevelName
    indicators: list[RiskIndicator] = Field(default_factory=list)
    refer_for_investigation: bool = False
    summary: str = ""
    features_used: list[str] = Field(default_factory=list)
    features_not_used: list[str] = Field(default_factory=list)
    disclaimer: str = ""
    mode: Literal["DETERMINISTIC_INDICATORS"] = "DETERMINISTIC_INDICATORS"
    used_llm: bool = False


# =========================================================================== #
# Agent 6 — Decision Support (recommendation for a human, never a decision)
# =========================================================================== #
Recommendation = Literal[
    "ELIGIBLE_FOR_HUMAN_REVIEW", "ADDITIONAL_INFORMATION_REQUIRED", "POLICY_NOT_FOUND", "REVIEW_REQUIRED"
]


class DecisionReason(BaseModel):
    source: str  # which agent produced the evidence
    code: str
    message: str
    blocking: bool = False  # True if it determined the recommendation


class DecisionSupport(BaseModel):
    """Output of Agent 6 — the Decision Support Agent."""

    recommendation: Recommendation
    confidence: float = Field(ge=0, le=1)
    confidence_notes: list[str] = Field(default_factory=list)
    reasons: list[DecisionReason] = Field(default_factory=list)
    required_actions: list[str] = Field(default_factory=list)
    reviewer_checklist: list[str] = Field(default_factory=list)
    indicative_admissible_amount: float | None = None
    summary: str = ""
    summary_source: Literal["template", "gemini"] = "template"
    warnings: list[str] = Field(default_factory=list)
    human_decision_required: Literal[True] = True
    disclaimer: str = ""
    used_llm: bool = False
    llm_model: str | None = None
