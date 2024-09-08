"""Typed state schema for the LangGraph document processing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, TypedDict


@dataclass
class TraceStep:
    """Records timing, cost, and I/O metadata for a single graph node execution."""

    node_name: str
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    completed_at: str = ""
    duration_ms: float = 0.0
    input_keys: list[str] = field(default_factory=list)
    output_keys: list[str] = field(default_factory=list)
    cost: float = 0.0
    error: str = ""

    def complete(self) -> None:
        """Mark the step as completed and compute duration."""
        now = datetime.now(UTC)
        self.completed_at = now.isoformat()
        started = datetime.fromisoformat(self.started_at)
        self.duration_ms = (now - started).total_seconds() * 1000


class DocumentState(TypedDict, total=False):
    """Full state schema passed through the LangGraph document processing pipeline.

    Uses total=False so nodes only need to return the keys they update.
    """

    # Identity
    job_id: str

    # Document content
    document_text: str
    document_type: Literal["contract", "invoice", "report", "unknown"]

    # Classification
    classification_confidence: float

    # Extraction
    extracted_data: dict

    # Validation
    validation_results: list

    # Analysis
    analysis: dict
    risks: list
    anomalies: list
    insights: list

    # Output
    summary: str

    # Pipeline control
    status: Literal[
        "pending",
        "classifying",
        "extracting",
        "validating",
        "analyzing",
        "review",
        "complete",
        "failed",
    ]

    # Human-in-the-loop
    requires_review: bool
    review_feedback: str
    approved: bool

    # Observability
    trace: list  # list[TraceStep] serialized as dicts
    total_cost: float
    errors: list

    # Timestamps
    created_at: str
    updated_at: str

    # Internal — dependency-injected LLM provider (not serialized)
    _llm_provider: Any
