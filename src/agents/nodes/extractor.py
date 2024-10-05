"""Document extraction nodes — type-specific data extraction via LLM."""

from __future__ import annotations

from datetime import UTC, datetime

import structlog

from src.state.schema import DocumentState, TraceStep

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM_PREFIX = (
    "You are a document data extraction expert. "
    "Extract the requested fields from the document text below. "
    "Respond ONLY with valid JSON matching the schema described.\n\n"
)


def _get_provider(state: DocumentState):
    """Get LLM provider from state, falling back to factory default."""
    provider = state.get("_llm_provider")
    if provider is not None:
        return provider
    from src.llm.factory import get_llm_provider

    return get_llm_provider()


def _invoke_extraction(state: DocumentState, schema_prompt: str, node_name: str) -> dict:
    """Run LLM extraction with the given schema prompt, return extracted_data."""
    trace_step = TraceStep(
        node_name=node_name,
        input_keys=["document_text", "document_type"],
        output_keys=["extracted_data", "status"],
    )
    errors: list[str] = list(state.get("errors", []))

    try:
        llm = _get_provider(state)
        document_text = state.get("document_text", "")

        system_prompt = _EXTRACT_SYSTEM_PREFIX + schema_prompt
        user_prompt = f"Document:\n\n{document_text[:8000]}"

        logger.info("extracting_data", node=node_name, job_id=state.get("job_id"))
        extracted_data = llm.generate_json(user_prompt, system_message=system_prompt)

        logger.info("extraction_complete", node=node_name, job_id=state.get("job_id"))
        trace_step.complete()
        trace = list(state.get("trace", []))
        trace.append(trace_step.__dict__)

        return {
            "extracted_data": extracted_data,
            "status": "extracting",
            "trace": trace,
            "updated_at": datetime.now(UTC).isoformat(),
        }

    except Exception as exc:
        logger.error("extraction_failed", node=node_name, error=str(exc), job_id=state.get("job_id"))
        trace_step.error = str(exc)
        trace_step.complete()
        trace = list(state.get("trace", []))
        trace.append(trace_step.__dict__)
        errors.append(f"{node_name}: {exc}")

        return {
            "extracted_data": {},
            "status": "failed",
            "trace": trace,
            "errors": errors,
            "updated_at": datetime.now(UTC).isoformat(),
        }


# ---------------------------------------------------------------------------
# Type-specific extractors
# ---------------------------------------------------------------------------


def extract_contract(state: DocumentState) -> dict:
    """Extract contract-specific fields: parties, dates, clauses, obligations, terms."""
    schema_prompt = (
        "Extract the following from this contract document as JSON:\n"
        "{\n"
        '  "parties": ["<list of party names>"],\n'
        '  "effective_date": "<YYYY-MM-DD or null>",\n'
        '  "expiration_date": "<YYYY-MM-DD or null>",\n'
        '  "clauses": [{"title": "<clause title>", "text": "<clause summary>"}],\n'
        '  "obligations": [{"party": "<name>", "obligation": "<description>"}],\n'
        '  "terms": {"duration": "<string>", "renewal": "<string>", "termination": "<string>"}\n'
        "}"
    )
    return _invoke_extraction(state, schema_prompt, "extract_contract")


def extract_invoice(state: DocumentState) -> dict:
    """Extract invoice-specific fields: vendor, line items, tax, grand total."""
    schema_prompt = (
        "Extract the following from this invoice document as JSON:\n"
        "{\n"
        '  "vendor": "<vendor name>",\n'
        '  "invoice_number": "<string>",\n'
        '  "invoice_date": "<YYYY-MM-DD or null>",\n'
        '  "due_date": "<YYYY-MM-DD or null>",\n'
        '  "line_items": [\n'
        '    {"description": "<string>", "qty": <number>, "unit_price": <number>, "total": <number>}\n'
        "  ],\n"
        '  "tax": <number>,\n'
        '  "grand_total": <number>\n'
        "}"
    )
    return _invoke_extraction(state, schema_prompt, "extract_invoice")


def extract_report(state: DocumentState) -> dict:
    """Extract report-specific fields: title, period, KPIs, sections."""
    schema_prompt = (
        "Extract the following from this report document as JSON:\n"
        "{\n"
        '  "title": "<report title>",\n'
        '  "period": "<reporting period>",\n'
        '  "kpis": [\n'
        '    {"name": "<KPI name>", "value": <number or string>, "unit": "<string>"}\n'
        "  ],\n"
        '  "sections": [\n'
        '    {"heading": "<section heading>", "summary": "<brief summary>"}\n'
        "  ]\n"
        "}"
    )
    return _invoke_extraction(state, schema_prompt, "extract_report")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def route_extraction(state: DocumentState) -> str:
    """Return the extraction function name to call based on document_type.

    Used as a LangGraph conditional edge router.
    """
    doc_type = state.get("document_type", "unknown")
    mapping = {
        "contract": "extract_contract",
        "invoice": "extract_invoice",
        "report": "extract_report",
    }
    route = mapping.get(doc_type, "extract_contract")  # default fallback
    logger.info("routing_extraction", document_type=doc_type, route=route, job_id=state.get("job_id"))
    return route
