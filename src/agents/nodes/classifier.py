"""Document classification node — uses LLM to determine document type."""

from __future__ import annotations

from datetime import UTC, datetime

import structlog

from src.state.schema import DocumentState, TraceStep

logger = structlog.get_logger(__name__)


def _get_provider(state: DocumentState):
    """Get LLM provider from state, falling back to factory default."""
    provider = state.get("_llm_provider")
    if provider is not None:
        return provider
    from src.llm.factory import get_llm_provider

    return get_llm_provider()


def classify_document(state: DocumentState) -> dict:
    """Classify document_text as contract, invoice, or report using an LLM.

    Returns partial state with document_type, classification_confidence,
    status, and an appended TraceStep.
    """
    trace_step = TraceStep(
        node_name="classify_document",
        input_keys=["document_text"],
        output_keys=["document_type", "classification_confidence", "status"],
    )
    errors: list[str] = list(state.get("errors", []))

    try:
        llm = _get_provider(state)
        document_text = state.get("document_text", "")

        system_prompt = (
            "You are a document classification expert. "
            "Classify the following document into exactly one of these types: "
            "contract, invoice, report.\n\n"
            "Respond ONLY with valid JSON in this exact format:\n"
            '{"type": "<contract|invoice|report>", "confidence": <float between 0 and 1>}\n\n'
            "Base your confidence on how clearly the document matches the type. "
            "If the document is ambiguous, use the best-fit type with a lower confidence."
        )

        user_prompt = f"Document to classify:\n\n{document_text[:8000]}"

        logger.info("classifying_document", job_id=state.get("job_id"))
        parsed = llm.generate_json(user_prompt, system_message=system_prompt)

        doc_type = parsed.get("type", "unknown")
        confidence = float(parsed.get("confidence", 0.0))

        # Clamp and validate
        if doc_type not in ("contract", "invoice", "report"):
            doc_type = "unknown"
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        logger.info(
            "classification_complete",
            job_id=state.get("job_id"),
            document_type=doc_type,
            confidence=confidence,
        )

        trace_step.complete()
        trace = list(state.get("trace", []))
        trace.append(trace_step.__dict__)

        return {
            "document_type": doc_type,
            "classification_confidence": confidence,
            "status": "classifying",
            "trace": trace,
            "updated_at": datetime.now(UTC).isoformat(),
        }

    except Exception as exc:
        logger.error("classification_failed", error=str(exc), job_id=state.get("job_id"))
        trace_step.error = str(exc)
        trace_step.complete()
        trace = list(state.get("trace", []))
        trace.append(trace_step.__dict__)
        errors.append(f"classifier: {exc}")

        return {
            "document_type": "unknown",
            "classification_confidence": 0.0,
            "status": "failed",
            "trace": trace,
            "errors": errors,
            "updated_at": datetime.now(UTC).isoformat(),
        }
