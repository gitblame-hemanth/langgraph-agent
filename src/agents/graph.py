"""LangGraph workflow for document processing pipeline."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import structlog
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.llm.base import BaseLLMProvider
from src.llm.factory import get_llm_provider
from src.state.schema import DocumentState, TraceStep
from src.tools.calculator import CalculatorTool
from src.tools.extractor import ExtractorTool

logger = structlog.get_logger(__name__)


def _get_provider(state: DocumentState) -> BaseLLMProvider:
    """Get LLM provider from state, falling back to factory default."""
    provider = state.get("_llm_provider")
    if provider is not None:
        return provider
    return get_llm_provider()


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------


async def classify(state: DocumentState) -> DocumentState:
    """Classify the document type using the LLM."""
    log = logger.bind(job_id=state.get("job_id"))
    step = TraceStep(node_name="classify", input_keys=["document_text"])
    try:
        log.info("classify.start")
        llm = _get_provider(state)

        text = state.get("document_text", "")
        system_msg = (
            "Classify the following document into exactly one of these categories: "
            "contract, invoice, report. Respond with ONLY a JSON object like "
            '{"type": "<category>", "confidence": <0.0-1.0>}.'
        )
        user_prompt = f"Document:\n{text[:4000]}"

        parsed = llm.generate_json(user_prompt, system_message=system_msg)
        doc_type = parsed.get("type", "unknown")
        confidence = float(parsed.get("confidence", 0.0))

        if doc_type not in ("contract", "invoice", "report"):
            doc_type = "unknown"

        step.output_keys = ["document_type", "classification_confidence"]
        step.complete()
        log.info("classify.done", document_type=doc_type, confidence=confidence)

        return {
            "document_type": doc_type,
            "classification_confidence": confidence,
            "status": "classifying",
            "trace": state.get("trace", []) + [step.__dict__],
            "updated_at": datetime.now(UTC).isoformat(),
        }  # type: ignore[return-value]
    except Exception as exc:
        step.error = str(exc)
        step.complete()
        log.error("classify.error", error=str(exc))
        return {
            "document_type": "unknown",
            "classification_confidence": 0.0,
            "status": "classifying",
            "errors": state.get("errors", []) + [f"classify: {exc}"],
            "trace": state.get("trace", []) + [step.__dict__],
            "updated_at": datetime.now(UTC).isoformat(),
        }  # type: ignore[return-value]


def route_extraction(state: DocumentState) -> str:
    """Route to the appropriate extraction path based on document type.

    All routes converge on the same 'extract' node but the node reads
    document_type to decide which fields to pull.
    """
    doc_type = state.get("document_type", "unknown")
    logger.info("route_extraction", document_type=doc_type)
    # All routes go to extract — the node handles type-specific logic internally
    return "extract"


async def extract(state: DocumentState) -> DocumentState:
    """Extract structured fields from the document based on its type."""
    log = logger.bind(job_id=state.get("job_id"))
    step = TraceStep(node_name="extract", input_keys=["document_text", "document_type"])
    try:
        log.info("extract.start")
        llm = _get_provider(state)

        doc_type = state.get("document_type", "unknown")
        text = state.get("document_text", "")

        # Type-specific extraction schemas
        field_schemas: dict[str, list[str]] = {
            "contract": [
                "parties",
                "effective_date",
                "termination_date",
                "governing_law",
                "payment_terms",
                "obligations",
            ],
            "invoice": [
                "invoice_number",
                "date",
                "due_date",
                "vendor",
                "customer",
                "line_items",
                "subtotal",
                "tax",
                "total",
            ],
            "report": [
                "title",
                "author",
                "date",
                "executive_summary",
                "key_findings",
                "recommendations",
            ],
        }
        fields = field_schemas.get(doc_type, ["summary", "key_points", "dates", "amounts"])

        # Use the injected provider's generate_json for field extraction
        fields_desc = ", ".join(fields)
        extraction_system = (
            "Extract the following fields from this document. "
            f"Fields: {fields_desc}\n\n"
            "Return ONLY a JSON object with these exact field names as keys. "
            "Use null for any field you cannot find."
        )
        extraction_prompt = f"Document:\n{text[:6000]}"
        extracted = llm.generate_json(extraction_prompt, system_message=extraction_system)
        # Ensure all requested fields are present
        extracted = {f: extracted.get(f) for f in fields}

        extractor = ExtractorTool()

        # Also pull out dates and amounts via regex
        dates = extractor.extract_dates(text)
        amounts = extractor.extract_amounts(text)
        extracted["_extracted_dates"] = dates
        extracted["_extracted_amounts"] = amounts

        step.output_keys = ["extracted_data"]
        step.complete()
        log.info("extract.done", field_count=len(extracted))

        return {
            "extracted_data": extracted,
            "status": "extracting",
            "trace": state.get("trace", []) + [step.__dict__],
            "updated_at": datetime.now(UTC).isoformat(),
        }  # type: ignore[return-value]
    except Exception as exc:
        step.error = str(exc)
        step.complete()
        log.error("extract.error", error=str(exc))
        return {
            "extracted_data": {},
            "status": "extracting",
            "errors": state.get("errors", []) + [f"extract: {exc}"],
            "trace": state.get("trace", []) + [step.__dict__],
            "updated_at": datetime.now(UTC).isoformat(),
        }  # type: ignore[return-value]


async def validate(state: DocumentState) -> DocumentState:
    """Validate extracted data for completeness and correctness."""
    log = logger.bind(job_id=state.get("job_id"))
    step = TraceStep(node_name="validate", input_keys=["extracted_data", "document_type"])
    try:
        log.info("validate.start")
        extracted = state.get("extracted_data", {})
        doc_type = state.get("document_type", "unknown")
        results: list[dict[str, Any]] = []

        # Check required fields are non-empty
        required_fields: dict[str, list[str]] = {
            "contract": ["parties", "effective_date"],
            "invoice": ["invoice_number", "total"],
            "report": ["title", "key_findings"],
        }
        for fld in required_fields.get(doc_type, []):
            value = extracted.get(fld)
            present = bool(value) if not isinstance(value, (int, float)) else True
            results.append(
                {
                    "field": fld,
                    "check": "required_present",
                    "passed": present,
                    "message": f"{'Present' if present else 'Missing'}: {fld}",
                }
            )

        # Invoice-specific: validate totals
        if doc_type == "invoice":
            calc = CalculatorTool()
            amounts = state.get("extracted_data", {}).get("_extracted_amounts", [])
            expected_total = extracted.get("total")
            if amounts and expected_total is not None:
                try:
                    total_ok = calc.validate_total(amounts, float(expected_total))
                    results.append(
                        {
                            "field": "total",
                            "check": "total_validation",
                            "passed": total_ok,
                            "message": f"Total validation {'passed' if total_ok else 'failed'}",
                        }
                    )
                except (ValueError, TypeError):
                    pass

        step.output_keys = ["validation_results"]
        step.complete()
        log.info("validate.done", check_count=len(results))

        return {
            "validation_results": results,
            "status": "validating",
            "trace": state.get("trace", []) + [step.__dict__],
            "updated_at": datetime.now(UTC).isoformat(),
        }  # type: ignore[return-value]
    except Exception as exc:
        step.error = str(exc)
        step.complete()
        log.error("validate.error", error=str(exc))
        return {
            "validation_results": [],
            "status": "validating",
            "errors": state.get("errors", []) + [f"validate: {exc}"],
            "trace": state.get("trace", []) + [step.__dict__],
            "updated_at": datetime.now(UTC).isoformat(),
        }  # type: ignore[return-value]


async def analyze(state: DocumentState) -> DocumentState:
    """Perform deeper analysis: risks, anomalies, insights."""
    log = logger.bind(job_id=state.get("job_id"))
    step = TraceStep(node_name="analyze", input_keys=["document_text", "extracted_data", "validation_results"])
    try:
        log.info("analyze.start")
        llm = _get_provider(state)

        text = state.get("document_text", "")
        extracted = state.get("extracted_data", {})
        doc_type = state.get("document_type", "unknown")

        system_msg = (
            f"You are a {doc_type} analysis expert. Given this document and its extracted data, "
            "identify risks, anomalies, and key insights. Respond with ONLY a JSON object:\n"
            '{"risks": [<strings>], "anomalies": [<strings>], "insights": [<strings>], '
            '"summary": "<2-3 sentence analysis>"}'
        )
        user_prompt = f"Extracted data: {extracted}\n\nDocument:\n{text[:4000]}"

        parsed = llm.generate_json(user_prompt, system_message=system_msg)

        risks = parsed.get("risks", [])
        anomalies = parsed.get("anomalies", [])
        insights = parsed.get("insights", [])
        summary = parsed.get("summary", "")

        # Determine confidence: base on classification + validation pass rate
        validation = state.get("validation_results", [])
        pass_rate = sum(1 for v in validation if v.get("passed")) / len(validation) if validation else 0.5
        classification_conf = state.get("classification_confidence", 0.5)
        overall_confidence = (classification_conf + pass_rate) / 2

        step.output_keys = ["analysis", "risks", "anomalies", "insights", "summary"]
        step.complete()
        log.info("analyze.done", risk_count=len(risks), anomaly_count=len(anomalies))

        return {
            "analysis": {
                "confidence": overall_confidence,
                "pass_rate": pass_rate,
                "risk_count": len(risks),
                "anomaly_count": len(anomalies),
            },
            "risks": risks,
            "anomalies": anomalies,
            "insights": insights,
            "summary": summary,
            "status": "analyzing",
            "trace": state.get("trace", []) + [step.__dict__],
            "updated_at": datetime.now(UTC).isoformat(),
        }  # type: ignore[return-value]
    except Exception as exc:
        step.error = str(exc)
        step.complete()
        log.error("analyze.error", error=str(exc))
        return {
            "analysis": {},
            "risks": [],
            "anomalies": [],
            "insights": [],
            "summary": "",
            "status": "analyzing",
            "errors": state.get("errors", []) + [f"analyze: {exc}"],
            "trace": state.get("trace", []) + [step.__dict__],
            "updated_at": datetime.now(UTC).isoformat(),
        }  # type: ignore[return-value]


def check_confidence(state: DocumentState) -> str:
    """Conditional edge: route to output (high confidence) or mark_review (low)."""
    analysis = state.get("analysis", {})
    confidence = analysis.get("confidence", 0.0)
    threshold = 0.7
    anomaly_count = analysis.get("anomaly_count", 0)

    if confidence >= threshold and anomaly_count == 0:
        logger.info("check_confidence.high", confidence=confidence)
        return "output"
    logger.info("check_confidence.low", confidence=confidence, anomaly_count=anomaly_count)
    return "mark_review"


async def mark_review(state: DocumentState) -> DocumentState:
    """Flag the document for human review (graph will interrupt here)."""
    log = logger.bind(job_id=state.get("job_id"))
    step = TraceStep(node_name="mark_review", input_keys=["analysis"])
    log.info("mark_review.flagged")

    step.output_keys = ["requires_review", "status"]
    step.complete()

    return {
        "requires_review": True,
        "status": "review",
        "trace": state.get("trace", []) + [step.__dict__],
        "updated_at": datetime.now(UTC).isoformat(),
    }  # type: ignore[return-value]


async def output(state: DocumentState) -> DocumentState:
    """Produce the final output state."""
    log = logger.bind(job_id=state.get("job_id"))
    step = TraceStep(node_name="output", input_keys=["analysis", "extracted_data", "summary"])
    log.info("output.finalize")

    # Compute total cost from trace
    trace = state.get("trace", [])
    total_cost = sum(t.get("cost", 0.0) if isinstance(t, dict) else 0.0 for t in trace)

    step.output_keys = ["status", "total_cost"]
    step.complete()

    return {
        "status": "complete",
        "requires_review": False,
        "approved": True,
        "total_cost": total_cost,
        "trace": state.get("trace", []) + [step.__dict__],
        "updated_at": datetime.now(UTC).isoformat(),
    }  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_document_graph(
    checkpointer: BaseCheckpointSaver | None = None,
    llm_provider: BaseLLMProvider | None = None,
) -> CompiledStateGraph:
    """Build and compile the document processing LangGraph workflow.

    Args:
        checkpointer: Optional checkpoint saver for persistence and
            human-in-the-loop resume support.
        llm_provider: Optional LLM provider instance. If not provided,
            nodes will create a default provider from config.

    Returns:
        A compiled LangGraph state graph ready for invocation.
    """
    graph = StateGraph(DocumentState)

    # Register nodes
    graph.add_node("classify", classify)
    graph.add_node("extract", extract)
    graph.add_node("validate", validate)
    graph.add_node("analyze", analyze)
    graph.add_node("mark_review", mark_review)
    graph.add_node("output", output)

    # Edges
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route_extraction, {"extract": "extract"})
    graph.add_edge("extract", "validate")
    graph.add_edge("validate", "analyze")
    graph.add_conditional_edges(
        "analyze",
        check_confidence,
        {
            "output": "output",
            "mark_review": "mark_review",
        },
    )
    graph.add_edge("mark_review", END)
    graph.add_edge("output", END)

    compile_kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    compile_kwargs["interrupt_before"] = ["mark_review"]

    return graph.compile(**compile_kwargs)
