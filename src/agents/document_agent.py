"""High-level document processing agent that orchestrates graph execution."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import structlog
from langgraph.checkpoint.memory import MemorySaver

from src.agents.graph import build_document_graph
from src.config import AppConfig, get_config
from src.llm.base import BaseLLMProvider
from src.state.schema import DocumentState, TraceStep

logger = structlog.get_logger(__name__)


class DocumentAgent:
    """Orchestrates document analysis through the LangGraph pipeline.

    Manages graph execution, state persistence, cost tracking, and
    human-in-the-loop approval workflows.
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        persistence: Any | None = None,
        llm_provider: BaseLLMProvider | None = None,
    ) -> None:
        self._config = config or get_config()
        self._persistence = persistence
        self._llm_provider = llm_provider
        self._checkpointer = MemorySaver()
        self._graph = build_document_graph(
            checkpointer=self._checkpointer,
            llm_provider=self._llm_provider,
        )
        # In-memory job state cache (supplemented by persistence layer when available)
        self._jobs: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze(
        self,
        document_text: str,
        job_id: str | None = None,
    ) -> DocumentState:
        """Run the full document analysis pipeline.

        Args:
            document_text: Raw document text to process.
            job_id: Optional job identifier. Generated if not provided.

        Returns:
            The final DocumentState after pipeline execution.
        """
        job_id = job_id or str(uuid.uuid4())
        log = logger.bind(job_id=job_id)
        log.info("agent.analyze.start")

        now = datetime.now(UTC).isoformat()
        initial_state: DocumentState = {
            "job_id": job_id,
            "document_text": document_text,
            "document_type": "unknown",
            "classification_confidence": 0.0,
            "extracted_data": {},
            "validation_results": [],
            "analysis": {},
            "risks": [],
            "anomalies": [],
            "insights": [],
            "summary": "",
            "status": "pending",
            "requires_review": False,
            "review_feedback": "",
            "approved": False,
            "trace": [],
            "total_cost": 0.0,
            "errors": [],
            "created_at": now,
            "updated_at": now,
            "_llm_provider": self._llm_provider,
        }

        thread_config = {"configurable": {"thread_id": job_id}}

        try:
            result = await self._graph.ainvoke(initial_state, config=thread_config)
            self._jobs[job_id] = {
                "state": result,
                "thread_config": thread_config,
                "created_at": now,
            }

            if self._persistence is not None:
                await self._persist_state(job_id, result)

            log.info("agent.analyze.done", status=result.get("status"))
            return result  # type: ignore[return-value]
        except Exception as exc:
            log.error("agent.analyze.error", error=str(exc))
            initial_state["status"] = "failed"
            initial_state["errors"] = [str(exc)]
            self._jobs[job_id] = {
                "state": initial_state,
                "thread_config": thread_config,
                "created_at": now,
            }
            raise

    async def get_status(self, job_id: str) -> dict[str, Any]:
        """Return the current status of a job.

        Args:
            job_id: The job identifier.

        Returns:
            Dict with job_id, status, requires_review, created_at, updated_at.

        Raises:
            KeyError: If the job_id is not found.
        """
        job = self._get_job(job_id)
        state = job["state"]
        return {
            "job_id": job_id,
            "status": state.get("status", "unknown"),
            "requires_review": state.get("requires_review", False),
            "document_type": state.get("document_type", "unknown"),
            "created_at": state.get("created_at", ""),
            "updated_at": state.get("updated_at", ""),
            "error_count": len(state.get("errors", [])),
        }

    async def get_trace(self, job_id: str) -> list[TraceStep]:
        """Return the execution trace for a job.

        Args:
            job_id: The job identifier.

        Returns:
            List of TraceStep objects reconstructed from stored dicts.

        Raises:
            KeyError: If the job_id is not found.
        """
        job = self._get_job(job_id)
        raw_trace = job["state"].get("trace", [])
        steps: list[TraceStep] = []
        for entry in raw_trace:
            if isinstance(entry, dict):
                step = TraceStep(node_name=entry.get("node_name", ""))
                step.started_at = entry.get("started_at", "")
                step.completed_at = entry.get("completed_at", "")
                step.duration_ms = entry.get("duration_ms", 0.0)
                step.input_keys = entry.get("input_keys", [])
                step.output_keys = entry.get("output_keys", [])
                step.cost = entry.get("cost", 0.0)
                step.error = entry.get("error", "")
                steps.append(step)
            elif isinstance(entry, TraceStep):
                steps.append(entry)
        return steps

    async def approve(
        self,
        job_id: str,
        feedback: str = "",
    ) -> DocumentState:
        """Approve a document that is paused for human review.

        Resumes graph execution through the mark_review node and on to output.

        Args:
            job_id: The job identifier.
            feedback: Optional reviewer feedback.

        Returns:
            The updated DocumentState after resuming.

        Raises:
            KeyError: If the job_id is not found.
            ValueError: If the job is not in review status.
        """
        log = logger.bind(job_id=job_id)
        job = self._get_job(job_id)
        state = job["state"]

        if state.get("status") != "review" and not state.get("requires_review"):
            raise ValueError(f"Job {job_id} is not awaiting review (status={state.get('status')})")

        log.info("agent.approve", feedback=feedback)
        thread_config = job["thread_config"]

        # Update state with approval before resuming
        update: DocumentState = {
            "approved": True,
            "review_feedback": feedback,
            "updated_at": datetime.now(UTC).isoformat(),
        }  # type: ignore[typeddict-item]

        result = await self._graph.ainvoke(update, config=thread_config)
        job["state"] = result

        if self._persistence is not None:
            await self._persist_state(job_id, result)

        log.info("agent.approve.done", status=result.get("status"))
        return result  # type: ignore[return-value]

    async def reject(
        self,
        job_id: str,
        feedback: str,
    ) -> DocumentState:
        """Reject a document that is paused for human review.

        Args:
            job_id: The job identifier.
            feedback: Reviewer feedback explaining the rejection.

        Returns:
            The updated DocumentState with rejection recorded.

        Raises:
            KeyError: If the job_id is not found.
            ValueError: If the job is not in review status.
        """
        log = logger.bind(job_id=job_id)
        job = self._get_job(job_id)
        state = job["state"]

        if state.get("status") != "review" and not state.get("requires_review"):
            raise ValueError(f"Job {job_id} is not awaiting review (status={state.get('status')})")

        log.info("agent.reject", feedback=feedback)

        now = datetime.now(UTC).isoformat()
        state["approved"] = False
        state["review_feedback"] = feedback
        state["status"] = "failed"
        state["updated_at"] = now
        job["state"] = state

        if self._persistence is not None:
            await self._persist_state(job_id, state)

        log.info("agent.reject.done")
        return state  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_job(self, job_id: str) -> dict[str, Any]:
        """Retrieve a job by ID or raise KeyError."""
        if job_id not in self._jobs:
            raise KeyError(f"Job not found: {job_id}")
        return self._jobs[job_id]

    async def _persist_state(self, job_id: str, state: Any) -> None:
        """Persist state via the configured persistence layer."""
        try:
            if hasattr(self._persistence, "save"):
                await self._persistence.save(job_id, state)
            elif hasattr(self._persistence, "set"):
                await self._persistence.set(job_id, state)
        except Exception as exc:
            logger.error("agent.persist.error", job_id=job_id, error=str(exc))
