"""Request/response contracts shared by the FastAPI backend and the Streamlit dashboard."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from database.models import InsuranceType, ReviewDecision

ReviewerRole = Literal["CLAIMS_REVIEWER", "SENIOR_REVIEWER", "INVESTIGATOR"]


class ClaimSubmission(BaseModel):
    """New claim form. Validation errors are returned to the user field by field."""

    customer_name: str = Field(min_length=2, max_length=120)
    customer_id: str = Field(min_length=3, max_length=20, description="Customer code, e.g. CUST-0001")
    policy_number: str = Field(min_length=3, max_length=40)
    insurance_type: InsuranceType
    claim_amount: float = Field(gt=0, le=50_000_000)
    hospital_name: str = Field(min_length=3, max_length=200)
    admission_date: date
    discharge_date: date
    diagnosis: str = Field(min_length=3, max_length=255)
    description: str | None = Field(default=None, max_length=2000)

    @field_validator("customer_id", "policy_number")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("customer_name", "hospital_name", "diagnosis")
    @classmethod
    def _strip(cls, v: str) -> str:
        return " ".join(v.split())

    @model_validator(mode="after")
    def _dates(self):
        # Hard form errors only; subtler date problems are left for the Validation Agent to explain.
        if self.discharge_date < self.admission_date:
            raise ValueError("Discharge date cannot be before the admission date.")
        return self


class UploadedFile(BaseModel):
    filename: str
    content: bytes
    declared_type: str | None = None


class ReviewSubmission(BaseModel):
    reviewer_name: str = Field(min_length=2, max_length=120)
    reviewer_role: ReviewerRole = "CLAIMS_REVIEWER"
    decision: ReviewDecision
    comments: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def _comment_required(self):
        if self.decision in (ReviewDecision.REJECTED, ReviewDecision.INFO_REQUESTED, ReviewDecision.ESCALATED) \
                and not (self.comments and self.comments.strip()):
            raise ValueError(f"A comment is required when the decision is {self.decision.value}.")
        return self
