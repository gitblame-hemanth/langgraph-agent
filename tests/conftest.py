"""Shared fixtures for the LangGraph document processing test suite."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.state.persistence import InMemoryStatePersistence
from src.state.schema import DocumentState

# ---------------------------------------------------------------------------
# Sample documents
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_contract_text() -> str:
    return (
        "SERVICES AGREEMENT\n\n"
        'This Services Agreement (the "Agreement") is entered into as of January 15, 2025, '
        'by and between Acme Corporation, a Delaware corporation ("Client"), and TechPro '
        'Solutions LLC, a California limited liability company ("Provider").\n\n'
        "1. SCOPE OF SERVICES\n"
        "Provider shall deliver cloud infrastructure migration services as described in "
        "Exhibit A. Services shall commence on February 1, 2025, and conclude no later than "
        "July 31, 2025.\n\n"
        "2. COMPENSATION\n"
        "Client agrees to pay Provider a total fee of $450,000.00 USD, payable in monthly "
        "installments of $75,000.00. Payment is due within 30 days of receipt of each invoice.\n\n"
        "3. TERMINATION\n"
        "Either party may terminate this Agreement with 60 days written notice. In the event "
        "of a material breach, the non-breaching party may terminate immediately upon written "
        "notice. Upon termination, Client shall pay for all services rendered through the "
        "termination date."
    )


@pytest.fixture()
def sample_invoice_text() -> str:
    return (
        "INVOICE\n\n"
        "Vendor: CloudStack Technologies Inc.\n"
        "Invoice Number: INV-2025-00347\n"
        "Invoice Date: 2025-03-10\n"
        "Due Date: 2025-04-09\n\n"
        "Bill To:\n"
        "Meridian Financial Group\n"
        "500 Market Street, Suite 1200\n"
        "San Francisco, CA 94105\n\n"
        "Line Items:\n"
        "1. Enterprise Cloud Hosting (Annual)    Qty: 1    Unit Price: $24,000.00    Total: $24,000.00\n"
        "2. SSL Certificate Renewal              Qty: 3    Unit Price: $199.00       Total: $597.00\n"
        "3. 24/7 Premium Support (Monthly)       Qty: 12   Unit Price: $1,500.00     Total: $18,000.00\n"
        "4. Data Backup Service (Monthly)        Qty: 12   Unit Price: $350.00       Total: $4,200.00\n\n"
        "Subtotal: $46,797.00\n"
        "Tax (8.5%): $3,977.75\n"
        "Grand Total: $50,774.75\n\n"
        "Payment Terms: Net 30\n"
        "Please remit to: CloudStack Technologies Inc., Account #8827-4419"
    )


@pytest.fixture()
def sample_report_text() -> str:
    return (
        "Q4 2024 QUARTERLY BUSINESS REVIEW\n"
        "Prepared by: Strategic Analytics Division\n"
        "Period: October 1, 2024 - December 31, 2024\n\n"
        "EXECUTIVE SUMMARY\n"
        "Revenue grew 18% year-over-year to $12.4M, driven by strong enterprise "
        "adoption. Customer acquisition cost decreased 12% to $2,340 per account.\n\n"
        "KEY PERFORMANCE INDICATORS\n"
        "- Revenue: $12,400,000 (target: $11,500,000) +7.8% above target\n"
        "- Active Users: 84,200 (+22% YoY)\n"
        "- Churn Rate: 3.1% (target: <5%)\n"
        "- Net Promoter Score: 72 (+8 from Q3)\n"
        "- Average Deal Size: $38,500 (+15% YoY)\n\n"
        "OPERATIONAL HIGHLIGHTS\n"
        "Infrastructure costs reduced by 23% through migration to spot instances. "
        "Engineering velocity improved 31% as measured by deployment frequency. "
        "Support ticket resolution time decreased from 4.2 hours to 2.8 hours.\n\n"
        "RISKS AND OUTLOOK\n"
        "Competitor pricing pressure in the mid-market segment may impact Q1 growth. "
        "Two key enterprise renewals ($1.2M combined) are scheduled for January."
    )


# ---------------------------------------------------------------------------
# State fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def initial_state() -> DocumentState:
    now = datetime.now(UTC).isoformat()
    return DocumentState(
        job_id="test-job-001",
        document_text="",
        document_type="unknown",
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


# ---------------------------------------------------------------------------
# Persistence fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_persistence() -> InMemoryStatePersistence:
    return InMemoryStatePersistence()
