"""FastAPI routes for the document processing API."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import redis.asyncio as aioredis
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from src.api.models import (
    AnalyzeRequest,
    AnalyzeResponse,
    ApprovalRequest,
    ErrorResponse,
    HealthResponse,
    JobStatusResponse,
    RejectionRequest,
    TraceResponse,
    TraceStepResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_persistence(request: Request) -> dict[str, Any]:
    """Return the in-memory job store from app state."""
    return request.app.state.jobs  # type: ignore[no-any-return]


def _get_redis(request: Request) -> aioredis.Redis:
    """Return the Redis client from app state."""
    return request.app.state.redis  # type: ignore[no-any-return]


async def _run_graph(request: Request, job_id: str) -> None:
    """Execute the LangGraph pipeline for *job_id* in the background.

    The function mutates the in-memory job store so that subsequent status
    polls reflect progress.
    """
    jobs = _get_persistence(request)
    job = jobs.get(job_id)
    if job is None:
        logger.error("Background task: job %s not found", job_id)
        return

    try:
        graph = request.app.state.graph  # compiled LangGraph graph

        # Build initial state from stored job data
        initial_state: dict[str, Any] = {
            "job_id": job_id,
            "document_text": job["document_text"],
            "status": "classifying",
            "created_at": job["created_at"],
            "updated_at": datetime.now(UTC).isoformat(),
            "trace": [],
            "total_cost": 0.0,
            "errors": [],
        }
        if job.get("document_type"):
            initial_state["document_type"] = job["document_type"]

        # Update status before running
        job["status"] = "classifying"
        job["updated_at"] = datetime.now(UTC).isoformat()

        # Run the graph
        result = await graph.ainvoke(initial_state)

        # Persist final state
        job.update(
            {
                "status": result.get("status", "complete"),
                "document_type": result.get("document_type"),
                "summary": result.get("summary"),
                "extracted_data": result.get("extracted_data"),
                "requires_review": result.get("requires_review", False),
                "trace": result.get("trace", []),
                "total_cost": result.get("total_cost", 0.0),
                "updated_at": datetime.now(UTC).isoformat(),
            }
        )
    except Exception:
        logger.exception("Graph execution failed for job %s", job_id)
        job["status"] = "failed"
        job["updated_at"] = datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=202,
    responses={500: {"model": ErrorResponse}},
)
async def analyze(
    body: AnalyzeRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> AnalyzeResponse:
    """Accept a document for asynchronous analysis."""
    job_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()

    job: dict[str, Any] = {
        "job_id": job_id,
        "status": "pending",
        "document_text": body.document_text,
        "document_type": body.document_type,
        "summary": None,
        "extracted_data": None,
        "requires_review": False,
        "trace": [],
        "total_cost": 0.0,
        "created_at": now,
        "updated_at": now,
    }

    jobs = _get_persistence(request)
    jobs[job_id] = job

    # Kick off graph execution without blocking the response
    background_tasks.add_task(_run_graph, request, job_id)

    return AnalyzeResponse(
        job_id=job_id,
        status="pending",
        document_type=body.document_type,
    )


@router.get(
    "/jobs/{job_id}/status",
    response_model=JobStatusResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_job_status(job_id: str, request: Request) -> JobStatusResponse:
    """Return the current state of a job."""
    jobs = _get_persistence(request)
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        document_type=job.get("document_type"),
        requires_review=job.get("requires_review", False),
        created_at=job.get("created_at"),
        updated_at=job.get("updated_at"),
    )


@router.get(
    "/jobs/{job_id}/trace",
    response_model=TraceResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_job_trace(job_id: str, request: Request) -> TraceResponse:
    """Return the execution trace for a job."""
    jobs = _get_persistence(request)
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    raw_steps: list[dict[str, Any]] = job.get("trace", [])
    steps = [
        TraceStepResponse(
            node_name=s.get("node_name", "unknown"),
            duration_ms=s.get("duration_ms", 0.0),
            cost=s.get("cost", 0.0),
            error=s.get("error", ""),
        )
        for s in raw_steps
    ]
    total_duration = sum(s.duration_ms for s in steps)

    return TraceResponse(
        job_id=job_id,
        steps=steps,
        total_cost=job.get("total_cost", 0.0),
        total_duration_ms=total_duration,
    )


@router.post(
    "/jobs/{job_id}/approve",
    response_model=JobStatusResponse,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
async def approve_job(
    job_id: str,
    body: ApprovalRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> JobStatusResponse:
    """Approve a job waiting at the human-review checkpoint."""
    jobs = _get_persistence(request)
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job["status"] != "review":
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} is not awaiting review (status={job['status']})",
        )

    job["approved"] = True
    job["review_feedback"] = body.feedback or ""
    job["status"] = "complete"
    job["updated_at"] = datetime.now(UTC).isoformat()

    # If the graph supports resuming after approval, kick that off
    if hasattr(request.app.state, "graph") and request.app.state.graph is not None:
        background_tasks.add_task(_run_graph, request, job_id)

    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        document_type=job.get("document_type"),
        requires_review=False,
        created_at=job.get("created_at"),
        updated_at=job.get("updated_at"),
    )


@router.post(
    "/jobs/{job_id}/reject",
    response_model=JobStatusResponse,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
async def reject_job(
    job_id: str,
    body: RejectionRequest,
    request: Request,
) -> JobStatusResponse:
    """Reject a job waiting at the human-review checkpoint."""
    jobs = _get_persistence(request)
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job["status"] != "review":
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} is not awaiting review (status={job['status']})",
        )

    job["approved"] = False
    job["review_feedback"] = body.feedback
    job["status"] = "failed"
    job["updated_at"] = datetime.now(UTC).isoformat()

    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        document_type=job.get("document_type"),
        requires_review=False,
        created_at=job.get("created_at"),
        updated_at=job.get("updated_at"),
    )


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Liveness / readiness probe."""
    redis_ok = False
    try:
        rc: aioredis.Redis = _get_redis(request)
        await rc.ping()
        redis_ok = True
    except Exception:
        logger.warning("Redis health-check failed", exc_info=True)

    jobs = _get_persistence(request)
    pending = sum(
        1
        for j in jobs.values()
        if j.get("status") in ("pending", "classifying", "extracting", "validating", "analyzing")
    )

    return HealthResponse(
        status="ok" if redis_ok else "degraded",
        redis_connected=redis_ok,
        pending_jobs=pending,
    )
