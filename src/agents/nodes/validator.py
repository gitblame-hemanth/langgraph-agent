"""Document validation node — business rule checks per document type."""

from __future__ import annotations

from datetime import UTC, datetime

import structlog

from src.state.schema import DocumentState, TraceStep

logger = structlog.get_logger(__name__)


def validate_document(state: DocumentState) -> dict:
    """Run business rule validations based on document_type and extracted_data.

    Returns partial state with validation_results list of
    {rule: str, passed: bool, detail: str}.
    """
    trace_step = TraceStep(
        node_name="validate_document",
        input_keys=["document_type", "extracted_data"],
        output_keys=["validation_results", "status"],
    )
    errors: list[str] = list(state.get("errors", []))

    try:
        doc_type = state.get("document_type", "unknown")
        data = state.get("extracted_data", {})

        logger.info("validating_document", document_type=doc_type, job_id=state.get("job_id"))

        validators = {
            "contract": _validate_contract,
            "invoice": _validate_invoice,
            "report": _validate_report,
        }

        validate_fn = validators.get(doc_type, _validate_generic)
        results = validate_fn(data)

        logger.info(
            "validation_complete",
            job_id=state.get("job_id"),
            total_rules=len(results),
            passed=sum(1 for r in results if r["passed"]),
            failed=sum(1 for r in results if not r["passed"]),
        )

        trace_step.complete()
        trace = list(state.get("trace", []))
        trace.append(trace_step.__dict__)

        return {
            "validation_results": results,
            "status": "validating",
            "trace": trace,
            "updated_at": datetime.now(UTC).isoformat(),
        }

    except Exception as exc:
        logger.error("validation_failed", error=str(exc), job_id=state.get("job_id"))
        trace_step.error = str(exc)
        trace_step.complete()
        trace = list(state.get("trace", []))
        trace.append(trace_step.__dict__)
        errors.append(f"validator: {exc}")

        return {
            "validation_results": [],
            "status": "failed",
            "trace": trace,
            "errors": errors,
            "updated_at": datetime.now(UTC).isoformat(),
        }


# ---------------------------------------------------------------------------
# Contract validations
# ---------------------------------------------------------------------------


def _validate_contract(data: dict) -> list[dict]:
    results: list[dict] = []

    # Check for missing dates
    effective = data.get("effective_date")
    expiration = data.get("expiration_date")
    results.append(
        {
            "rule": "effective_date_present",
            "passed": bool(effective),
            "detail": "Effective date is present" if effective else "Missing effective date",
        }
    )
    results.append(
        {
            "rule": "expiration_date_present",
            "passed": bool(expiration),
            "detail": "Expiration date is present" if expiration else "Missing expiration date",
        }
    )

    # Check parties
    parties = data.get("parties", [])
    results.append(
        {
            "rule": "parties_identified",
            "passed": len(parties) >= 2,
            "detail": f"Found {len(parties)} parties" if len(parties) >= 2 else "Fewer than 2 parties identified",
        }
    )

    # Check for clauses
    clauses = data.get("clauses", [])
    results.append(
        {
            "rule": "clauses_present",
            "passed": len(clauses) > 0,
            "detail": f"Found {len(clauses)} clauses" if clauses else "No clauses extracted",
        }
    )

    # Check for unsigned / missing signature clauses
    clause_titles = [c.get("title", "").lower() for c in clauses]
    has_signature_clause = any("sign" in t for t in clause_titles)
    results.append(
        {
            "rule": "signature_clause_present",
            "passed": has_signature_clause,
            "detail": "Signature clause found" if has_signature_clause else "No signature/signing clause detected",
        }
    )

    # Check for unusual terms
    terms = data.get("terms", {})
    termination = terms.get("termination", "")
    unusual_keywords = ["perpetual", "irrevocable", "unlimited liability", "no termination"]
    has_unusual = any(kw in termination.lower() for kw in unusual_keywords)
    results.append(
        {
            "rule": "no_unusual_terms",
            "passed": not has_unusual,
            "detail": "Unusual termination terms detected" if has_unusual else "Termination terms appear standard",
        }
    )

    return results


# ---------------------------------------------------------------------------
# Invoice validations
# ---------------------------------------------------------------------------


def _validate_invoice(data: dict) -> list[dict]:
    results: list[dict] = []

    line_items = data.get("line_items", [])

    # Verify line item math
    math_errors: list[str] = []
    computed_subtotal = 0.0
    for idx, item in enumerate(line_items):
        qty = float(item.get("qty", 0))
        unit_price = float(item.get("unit_price", 0))
        line_total = float(item.get("total", 0))
        expected = round(qty * unit_price, 2)
        computed_subtotal += expected
        if abs(expected - line_total) > 0.01:
            math_errors.append(f"Line {idx + 1}: {qty} x {unit_price} = {expected}, but listed as {line_total}")

    results.append(
        {
            "rule": "line_item_math",
            "passed": len(math_errors) == 0,
            "detail": "All line item calculations correct" if not math_errors else "; ".join(math_errors),
        }
    )

    # Check for duplicate line items
    descriptions = [item.get("description", "").strip().lower() for item in line_items]
    seen: set[str] = set()
    duplicates: list[str] = []
    for desc in descriptions:
        if desc in seen and desc:
            duplicates.append(desc)
        seen.add(desc)

    results.append(
        {
            "rule": "no_duplicate_items",
            "passed": len(duplicates) == 0,
            "detail": "No duplicate items" if not duplicates else f"Duplicate items: {', '.join(duplicates)}",
        }
    )

    # Validate grand total
    tax = float(data.get("tax", 0))
    grand_total = float(data.get("grand_total", 0))
    expected_total = round(computed_subtotal + tax, 2)
    total_match = abs(expected_total - grand_total) < 0.01

    results.append(
        {
            "rule": "grand_total_correct",
            "passed": total_match,
            "detail": (
                "Grand total matches computed total"
                if total_match
                else f"Expected {expected_total} (subtotal {computed_subtotal} + tax {tax}), got {grand_total}"
            ),
        }
    )

    # Check invoice has required fields
    results.append(
        {
            "rule": "vendor_present",
            "passed": bool(data.get("vendor")),
            "detail": "Vendor identified" if data.get("vendor") else "Missing vendor name",
        }
    )
    results.append(
        {
            "rule": "invoice_number_present",
            "passed": bool(data.get("invoice_number")),
            "detail": "Invoice number present" if data.get("invoice_number") else "Missing invoice number",
        }
    )

    return results


# ---------------------------------------------------------------------------
# Report validations
# ---------------------------------------------------------------------------


def _validate_report(data: dict) -> list[dict]:
    results: list[dict] = []

    # Check for KPIs
    kpis = data.get("kpis", [])
    results.append(
        {
            "rule": "kpis_present",
            "passed": len(kpis) > 0,
            "detail": f"Found {len(kpis)} KPIs" if kpis else "No KPIs extracted",
        }
    )

    # Check for out-of-range KPI values
    range_issues: list[str] = []
    for kpi in kpis:
        value = kpi.get("value")
        name = kpi.get("name", "unknown")
        if isinstance(value, (int, float)):
            # Flag negative values and extremely large values as potential issues
            if value < 0:
                range_issues.append(f"{name}: negative value ({value})")
            elif abs(value) > 1e12:
                range_issues.append(f"{name}: extremely large value ({value})")

    results.append(
        {
            "rule": "kpi_values_in_range",
            "passed": len(range_issues) == 0,
            "detail": "All KPI values in expected range" if not range_issues else "; ".join(range_issues),
        }
    )

    # Check for report title and period
    results.append(
        {
            "rule": "title_present",
            "passed": bool(data.get("title")),
            "detail": "Report title present" if data.get("title") else "Missing report title",
        }
    )
    results.append(
        {
            "rule": "period_present",
            "passed": bool(data.get("period")),
            "detail": "Reporting period present" if data.get("period") else "Missing reporting period",
        }
    )

    # Check sections
    sections = data.get("sections", [])
    results.append(
        {
            "rule": "sections_present",
            "passed": len(sections) > 0,
            "detail": f"Found {len(sections)} sections" if sections else "No sections extracted",
        }
    )

    return results


# ---------------------------------------------------------------------------
# Generic fallback
# ---------------------------------------------------------------------------


def _validate_generic(data: dict) -> list[dict]:
    return [
        {
            "rule": "document_type_supported",
            "passed": False,
            "detail": "Unknown document type — no specific validations available",
        }
    ]
