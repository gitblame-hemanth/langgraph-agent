#!/usr/bin/env python3
"""Demo script — shows end-to-end document analysis with trace output."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.nodes.classifier import classify_document
from src.agents.nodes.extractor import extract_contract, route_extraction
from src.state.persistence import InMemoryStatePersistence
from src.state.schema import DocumentState


def _load_sample(filename: str) -> str:
    """Load a sample document from the examples directory."""
    path = Path(__file__).resolve().parent / filename
    return path.read_text(encoding="utf-8")


def _print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


async def main() -> None:
    # ---- Load sample contract ------------------------------------------------
    contract_text = _load_sample("sample_contract.txt")
    _print_section("Document Analysis Demo")
    print(f"Loaded document ({len(contract_text)} chars)\n")

    # ---- Build initial state -------------------------------------------------
    now = datetime.now(UTC).isoformat()
    state: DocumentState = DocumentState(
        job_id="demo-001",
        document_text=contract_text,
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

    # ---- Step 1: Classification ----------------------------------------------
    _print_section("Step 1 — Classification")
    result = classify_document(state)
    state.update(result)
    print(f"Type:       {state['document_type']}")
    print(f"Confidence: {state['classification_confidence']:.2f}")

    # ---- Step 2: Routing & Extraction ----------------------------------------
    _print_section("Step 2 — Extraction")
    route = route_extraction(state)
    print(f"Route:      {route}")

    extractors = {
        "extract_contract": extract_contract,
    }
    extractor_fn = extractors.get(route)
    if extractor_fn:
        extraction_result = extractor_fn(state)
        state.update(extraction_result)
        print(f"Extracted:  {json.dumps(state.get('extracted_data', {}), indent=2)}")

    # ---- Step 3: Human review check ------------------------------------------
    _print_section("Step 3 — Human Review Check")
    confidence = state.get("classification_confidence", 0.0)
    if confidence < 0.8:
        state["requires_review"] = True
        state["status"] = "review"
        print("Low confidence — flagged for human review.")
        print("Simulating reviewer approval...")
        state["approved"] = True
        state["review_feedback"] = "Confirmed as contract after manual inspection."
        state["status"] = "complete"
        print(f"Feedback:   {state['review_feedback']}")
    else:
        state["status"] = "complete"
        print("High confidence — no review needed.")

    # ---- Trace ---------------------------------------------------------------
    _print_section("Execution Trace")
    print(json.dumps(state.get("trace", []), indent=2, default=str))

    # ---- Persist snapshot demo -----------------------------------------------
    _print_section("Persistence Demo")
    persistence = InMemoryStatePersistence()
    await persistence.save(state["job_id"], dict(state))
    snap_id = await persistence.snapshot(state["job_id"])
    print(f"Snapshot created: {snap_id}")

    loaded = await persistence.load(state["job_id"])
    print(f"Loaded job:       {loaded['job_id'] if loaded else 'N/A'}")
    print(f"Final status:     {loaded['status'] if loaded else 'N/A'}")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
