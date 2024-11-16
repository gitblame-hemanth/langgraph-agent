"""Human-in-the-loop review node — checkpoint for graph pause/resume."""

from __future__ import annotations

from datetime import UTC, datetime

import structlog

from src.state.schema import DocumentState, TraceStep

logger = structlog.get_logger(__name__)


def check_confidence(state: DocumentState) -> str:
    """Conditional edge: decide whether human review is needed.

    Returns "review_needed" if classification_confidence < 0.8 or any
    validation rule failed. Otherwise returns "output".
    """
    confidence = state.get("classification_confidence", 0.0)
    validation_results = state.get("validation_results", [])
    has_failures = any(not r.get("passed", True) for r in validation_results)

    if confidence < 0.8:
        logger.info(
            "review_triggered_low_confidence",
            confidence=confidence,
            job_id=state.get("job_id"),
        )
        return "review_needed"

    if has_failures:
        failed_rules = [r["rule"] for r in validation_results if not r.get("passed", True)]
        logger.info(
            "review_triggered_validation_failures",
            failed_rules=failed_rules,
            job_id=state.get("job_id"),
        )
        return "review_needed"

    logger.info("review_not_needed", confidence=confidence, job_id=state.get("job_id"))
    return "output"


def mark_for_review(state: DocumentState) -> dict:
    """Set state to require human review — graph pauses at this checkpoint.

    The graph infrastructure (LangGraph interrupt / checkpoint) should be
    configured to pause execution after this node until a human calls
    apply_approval with feedback.
    """
    trace_step = TraceStep(
        node_name="mark_for_review",
        input_keys=["classification_confidence", "validation_results"],
        output_keys=["requires_review", "status"],
    )
    errors: list[str] = list(state.get("errors", []))

    try:
        logger.info("marking_for_review", job_id=state.get("job_id"))

        trace_step.complete()
        trace = list(state.get("trace", []))
        trace.append(trace_step.__dict__)

        return {
            "requires_review": True,
            "status": "review",
            "trace": trace,
            "updated_at": datetime.now(UTC).isoformat(),
        }

    except Exception as exc:
        logger.error("mark_for_review_failed", error=str(exc), job_id=state.get("job_id"))
        trace_step.error = str(exc)
        trace_step.complete()
        trace = list(state.get("trace", []))
        trace.append(trace_step.__dict__)
        errors.append(f"mark_for_review: {exc}")

        return {
            "requires_review": True,
            "status": "failed",
            "trace": trace,
            "errors": errors,
            "updated_at": datetime.now(UTC).isoformat(),
        }


def apply_approval(state: DocumentState, feedback: str, approved: bool = True) -> dict:
    """Apply human review decision — called when the graph resumes from checkpoint.

    Args:
        state: Current document state.
        feedback: Human reviewer's feedback text.
        approved: Whether the document is approved to proceed.

    Returns:
        Partial state update with approval decision.
    """
    trace_step = TraceStep(
        node_name="apply_approval",
        input_keys=["requires_review"],
        output_keys=["approved", "review_feedback", "status"],
    )
    errors: list[str] = list(state.get("errors", []))

    try:
        logger.info(
            "applying_approval",
            approved=approved,
            job_id=state.get("job_id"),
        )

        new_status = "complete" if approved else "failed"

        trace_step.complete()
        trace = list(state.get("trace", []))
        trace.append(trace_step.__dict__)

        return {
            "approved": approved,
            "review_feedback": feedback,
            "requires_review": False,
            "status": new_status,
            "trace": trace,
            "updated_at": datetime.now(UTC).isoformat(),
        }

    except Exception as exc:
        logger.error("apply_approval_failed", error=str(exc), job_id=state.get("job_id"))
        trace_step.error = str(exc)
        trace_step.complete()
        trace = list(state.get("trace", []))
        trace.append(trace_step.__dict__)
        errors.append(f"apply_approval: {exc}")

        return {
            "approved": False,
            "review_feedback": feedback,
            "status": "failed",
            "trace": trace,
            "errors": errors,
            "updated_at": datetime.now(UTC).isoformat(),
        }
