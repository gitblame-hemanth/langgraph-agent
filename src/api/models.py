"""Pydantic request/response models for the document processing API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

# --- Requests ---


class AnalyzeRequest(BaseModel):
    """Start a new document analysis job."""

    document_text: str = Field(..., min_length=1, description="Raw document text to analyse")
    document_type: Literal["contract", "invoice", "report", "unknown"] | None = Field(
        default=None,
        description="Override automatic classification with an explicit document type",
    )


class ApprovalRequest(BaseModel):
    """Approve a job waiting at the human-review checkpoint."""

    feedback: str | None = Field(default=None, description="Optional reviewer notes")


class RejectionRequest(BaseModel):
    """Reject a job waiting at the human-review checkpoint."""

    feedback: str = Field(..., min_length=1, description="Reason for rejection (required)")


# --- Responses ---


class TraceStepResponse(BaseModel):
    """Single node execution record."""

    node_name: str
    duration_ms: float
    cost: float
    error: str = ""


class TraceResponse(BaseModel):
    """Full execution trace for a job."""

    job_id: str
    steps: list[TraceStepResponse]
    total_cost: float
    total_duration_ms: float


class AnalyzeResponse(BaseModel):
    """Returned immediately when a job is accepted."""

    job_id: str
    status: str
    document_type: str | None = None
    summary: str | None = None
    extracted_data: dict[str, Any] | None = None


class JobStatusResponse(BaseModel):
    """Current state of a job."""

    job_id: str
    status: str
    document_type: str | None = None
    requires_review: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None


class HealthResponse(BaseModel):
    """Service health check."""

    status: str
    redis_connected: bool
    pending_jobs: int


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    detail: str
    job_id: str | None = None
