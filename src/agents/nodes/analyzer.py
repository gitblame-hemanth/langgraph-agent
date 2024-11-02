"""Document analysis node — deep LLM-driven analysis per document type."""

from __future__ import annotations

import json
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


def analyze_document(state: DocumentState) -> dict:
    """Perform deep analysis based on document type using LLM.

    Contract: identify risks, generate summary.
    Invoice: flag anomalies, generate summary.
    Report: generate insights with recommendations, compare to benchmarks.
    """
    trace_step = TraceStep(
        node_name="analyze_document",
        input_keys=["document_text", "document_type", "extracted_data", "validation_results"],
        output_keys=["analysis", "risks", "anomalies", "insights", "summary", "status"],
    )
    errors: list[str] = list(state.get("errors", []))

    try:
        doc_type = state.get("document_type", "unknown")
        document_text = state.get("document_text", "")
        extracted_data = state.get("extracted_data", {})
        validation_results = state.get("validation_results", [])

        logger.info("analyzing_document", document_type=doc_type, job_id=state.get("job_id"))

        llm = _get_provider(state)

        analyzers = {
            "contract": _analyze_contract,
            "invoice": _analyze_invoice,
            "report": _analyze_report,
        }

        analyze_fn = analyzers.get(doc_type, _analyze_contract)
        result = analyze_fn(llm, document_text, extracted_data, validation_results)

        logger.info("analysis_complete", document_type=doc_type, job_id=state.get("job_id"))

        trace_step.complete()
        trace = list(state.get("trace", []))
        trace.append(trace_step.__dict__)

        return {
            "analysis": result.get("analysis", {}),
            "risks": result.get("risks", []),
            "anomalies": result.get("anomalies", []),
            "insights": result.get("insights", []),
            "summary": result.get("summary", ""),
            "status": "analyzing",
            "trace": trace,
            "updated_at": datetime.now(UTC).isoformat(),
        }

    except Exception as exc:
        logger.error("analysis_failed", error=str(exc), job_id=state.get("job_id"))
        trace_step.error = str(exc)
        trace_step.complete()
        trace = list(state.get("trace", []))
        trace.append(trace_step.__dict__)
        errors.append(f"analyzer: {exc}")

        return {
            "analysis": {},
            "risks": [],
            "anomalies": [],
            "insights": [],
            "summary": "",
            "status": "failed",
            "trace": trace,
            "errors": errors,
            "updated_at": datetime.now(UTC).isoformat(),
        }


# ---------------------------------------------------------------------------
# Contract analysis
# ---------------------------------------------------------------------------


def _analyze_contract(
    llm,
    document_text: str,
    extracted_data: dict,
    validation_results: list,
) -> dict:
    system_prompt = (
        "You are a legal contract analyst. Analyze the contract document and extracted data below.\n\n"
        "Respond ONLY with valid JSON in this format:\n"
        "{\n"
        '  "risks": [\n'
        '    {"risk": "<description>", "severity": "<high|medium|low>", "clause": "<clause reference>"}\n'
        "  ],\n"
        '  "analysis": {\n'
        '    "overall_risk_level": "<high|medium|low>",\n'
        '    "key_obligations": ["<obligation summary>"],\n'
        '    "notable_terms": ["<term summary>"]\n'
        "  },\n"
        '  "summary": "<2-3 sentence executive summary>"\n'
        "}"
    )
    user_content = (
        f"Document:\n{document_text[:6000]}\n\n"
        f"Extracted data:\n{json.dumps(extracted_data, indent=2, default=str)[:2000]}\n\n"
        f"Validation issues:\n{json.dumps(validation_results, indent=2, default=str)[:1000]}"
    )
    return llm.generate_json(user_content, system_message=system_prompt)


# ---------------------------------------------------------------------------
# Invoice analysis
# ---------------------------------------------------------------------------


def _analyze_invoice(
    llm,
    document_text: str,
    extracted_data: dict,
    validation_results: list,
) -> dict:
    system_prompt = (
        "You are a financial auditor. Analyze the invoice document and extracted data below.\n\n"
        "Respond ONLY with valid JSON in this format:\n"
        "{\n"
        '  "anomalies": [\n'
        '    {"type": "<pricing|duplicate|missing|calculation>", '
        '"detail": "<desc>", "severity": "<high|medium|low>"}\n'
        "  ],\n"
        '  "analysis": {\n'
        '    "total_amount": <number>,\n'
        '    "item_count": <number>,\n'
        '    "flags": ["<flag summary>"]\n'
        "  },\n"
        '  "summary": "<2-3 sentence executive summary>"\n'
        "}"
    )
    user_content = (
        f"Document:\n{document_text[:6000]}\n\n"
        f"Extracted data:\n{json.dumps(extracted_data, indent=2, default=str)[:2000]}\n\n"
        f"Validation issues:\n{json.dumps(validation_results, indent=2, default=str)[:1000]}"
    )
    return llm.generate_json(user_content, system_message=system_prompt)


# ---------------------------------------------------------------------------
# Report analysis
# ---------------------------------------------------------------------------


def _analyze_report(
    llm,
    document_text: str,
    extracted_data: dict,
    validation_results: list,
) -> dict:
    system_prompt = (
        "You are a business intelligence analyst. Analyze the report and extracted data below.\n\n"
        "Respond ONLY with valid JSON in this format:\n"
        "{\n"
        '  "insights": [\n'
        '    {"insight": "<finding>", "metric": "<related KPI/metric>", "recommendation": "<action>"}\n'
        "  ],\n"
        '  "analysis": {\n'
        '    "trend_direction": "<improving|stable|declining>",\n'
        '    "benchmark_comparison": "<above|at|below> industry average",\n'
        '    "key_drivers": ["<driver summary>"]\n'
        "  },\n"
        '  "summary": "<2-3 sentence executive summary>"\n'
        "}"
    )
    user_content = (
        f"Document:\n{document_text[:6000]}\n\n"
        f"Extracted data:\n{json.dumps(extracted_data, indent=2, default=str)[:2000]}\n\n"
        f"Validation issues:\n{json.dumps(validation_results, indent=2, default=str)[:1000]}"
    )
    return llm.generate_json(user_content, system_message=system_prompt)
