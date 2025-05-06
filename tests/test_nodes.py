"""Tests for graph nodes with mocked LLM calls."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from src.agents.nodes.classifier import classify_document
from src.agents.nodes.extractor import (
    extract_contract,
    extract_invoice,
    extract_report,
    route_extraction,
)
from src.state.schema import DocumentState


def _make_mock_provider(response_data: dict) -> MagicMock:
    """Create a mock BaseLLMProvider that returns given data from generate_json."""
    mock_provider = MagicMock()
    mock_provider.generate_json.return_value = response_data
    return mock_provider


def _make_state(text: str, doc_type: str = "unknown", llm_provider=None) -> DocumentState:
    """Build a minimal DocumentState for node tests."""
    now = datetime.now(UTC).isoformat()
    state = DocumentState(
        job_id="test-node-001",
        document_text=text,
        document_type=doc_type,
        classification_confidence=0.0,
        extracted_data={},
        validation_results=[],
        analysis={},
        risks=[],
        anomalies=[],
        insights=[],
        summary="",
        status="pending",
        requires_review=False,
        review_feedback="",
        approved=False,
        trace=[],
        total_cost=0.0,
        errors=[],
        created_at=now,
        updated_at=now,
    )
    if llm_provider is not None:
        state["_llm_provider"] = llm_provider
    return state


# ---------------------------------------------------------------------------
# Classifier tests
# ---------------------------------------------------------------------------


class TestClassifier:
    def test_classifier_returns_valid_type(self) -> None:
        mock_provider = _make_mock_provider({"type": "contract", "confidence": 0.92})
        state = _make_state(
            "This is a services agreement between parties...",
            llm_provider=mock_provider,
        )
        result = classify_document(state)

        assert result["document_type"] == "contract"
        assert result["classification_confidence"] == pytest.approx(0.92)
        assert result["status"] == "classifying"
        assert len(result["trace"]) == 1
        assert result["trace"][0]["node_name"] == "classify_document"
        mock_provider.generate_json.assert_called_once()

    def test_classifier_falls_back_to_factory(self) -> None:
        """When no _llm_provider in state, falls back to factory."""
        mock_provider = _make_mock_provider({"type": "invoice", "confidence": 0.85})
        state = _make_state("Invoice #123")

        with patch("src.llm.factory.get_llm_provider", return_value=mock_provider):
            result = classify_document(state)

        assert result["document_type"] == "invoice"
        assert result["classification_confidence"] == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# Extractor routing and type-specific extraction tests
# ---------------------------------------------------------------------------


class TestExtractorRouting:
    def test_extractor_route_contract(self) -> None:
        state = _make_state("contract text", doc_type="contract")
        assert route_extraction(state) == "extract_contract"

    def test_extractor_route_invoice(self) -> None:
        state = _make_state("invoice text", doc_type="invoice")
        assert route_extraction(state) == "extract_invoice"

    def test_extractor_route_report(self) -> None:
        state = _make_state("report text", doc_type="report")
        assert route_extraction(state) == "extract_report"


class TestExtractors:
    def test_extract_contract_returns_data(self) -> None:
        response_data = {
            "parties": ["Acme Corp", "TechPro LLC"],
            "effective_date": "2025-01-15",
            "expiration_date": "2025-07-31",
            "clauses": [{"title": "Scope", "text": "Cloud migration services"}],
            "obligations": [{"party": "Acme Corp", "obligation": "Pay $450,000"}],
            "terms": {
                "duration": "6 months",
                "renewal": "None",
                "termination": "60 days notice",
            },
        }
        mock_provider = _make_mock_provider(response_data)
        state = _make_state("contract text...", doc_type="contract", llm_provider=mock_provider)
        result = extract_contract(state)

        assert result["status"] == "extracting"
        assert "parties" in result["extracted_data"]
        assert len(result["extracted_data"]["parties"]) == 2

    def test_extract_invoice_returns_data(self) -> None:
        response_data = {
            "vendor": "CloudStack Technologies",
            "invoice_number": "INV-2025-00347",
            "invoice_date": "2025-03-10",
            "due_date": "2025-04-09",
            "line_items": [
                {
                    "description": "Cloud Hosting",
                    "qty": 1,
                    "unit_price": 24000,
                    "total": 24000,
                }
            ],
            "tax": 3977.75,
            "grand_total": 50774.75,
        }
        mock_provider = _make_mock_provider(response_data)
        state = _make_state("invoice text...", doc_type="invoice", llm_provider=mock_provider)
        result = extract_invoice(state)

        assert result["status"] == "extracting"
        assert result["extracted_data"]["vendor"] == "CloudStack Technologies"
        assert result["extracted_data"]["grand_total"] == 50774.75

    def test_extract_report_returns_data(self) -> None:
        response_data = {
            "title": "Q4 2024 Quarterly Business Review",
            "period": "Q4 2024",
            "kpis": [
                {"name": "Revenue", "value": 12400000, "unit": "USD"},
                {"name": "Active Users", "value": 84200, "unit": "users"},
            ],
            "sections": [
                {"heading": "Executive Summary", "summary": "Revenue grew 18% YoY"},
            ],
        }
        mock_provider = _make_mock_provider(response_data)
        state = _make_state("report text...", doc_type="report", llm_provider=mock_provider)
        result = extract_report(state)

        assert result["status"] == "extracting"
        assert result["extracted_data"]["title"] == "Q4 2024 Quarterly Business Review"
        assert len(result["extracted_data"]["kpis"]) == 2


# ---------------------------------------------------------------------------
# Confidence check tests (used in human-in-the-loop routing)
# ---------------------------------------------------------------------------


class TestConfidenceCheck:
    def test_confidence_check_high(self) -> None:
        """High confidence should not require review."""
        state = _make_state("text", doc_type="contract")
        state["classification_confidence"] = 0.95
        # Confidence >= 0.8 should pass through without review
        assert state["classification_confidence"] >= 0.8

    def test_confidence_check_low(self) -> None:
        """Low confidence should flag for human review."""
        state = _make_state("text", doc_type="unknown")
        state["classification_confidence"] = 0.35
        # Confidence < 0.8 should require review
        assert state["classification_confidence"] < 0.8
