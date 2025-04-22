"""Tests for state schema and persistence layer."""

from __future__ import annotations

import pytest

from src.state.persistence import InMemoryStatePersistence
from src.state.schema import DocumentState, TraceStep


class TestDocumentState:
    """Verify DocumentState typed dict has all required pipeline fields."""

    def test_initial_state_has_required_fields(self, initial_state: DocumentState) -> None:
        # Identity
        assert "job_id" in initial_state
        assert isinstance(initial_state["job_id"], str)

        # Document content
        assert "document_text" in initial_state
        assert "document_type" in initial_state
        assert initial_state["document_type"] in ("contract", "invoice", "report", "unknown")

        # Classification
        assert "classification_confidence" in initial_state
        assert isinstance(initial_state["classification_confidence"], float)

        # Extraction / Analysis
        assert "extracted_data" in initial_state
        assert "validation_results" in initial_state
        assert "analysis" in initial_state
        assert "risks" in initial_state
        assert "anomalies" in initial_state
        assert "insights" in initial_state
        assert "summary" in initial_state

        # Pipeline control
        assert "status" in initial_state
        assert initial_state["status"] == "pending"

        # Human-in-the-loop
        assert "requires_review" in initial_state
        assert "review_feedback" in initial_state
        assert "approved" in initial_state

        # Observability
        assert "trace" in initial_state
        assert "total_cost" in initial_state
        assert "errors" in initial_state

        # Timestamps
        assert "created_at" in initial_state
        assert "updated_at" in initial_state


class TestPersistence:
    """Verify in-memory persistence save/load and snapshot/rollback."""

    @pytest.mark.asyncio
    async def test_persistence_save_load(self, mock_persistence: InMemoryStatePersistence) -> None:
        state = {"job_id": "j1", "status": "pending", "document_text": "hello"}
        await mock_persistence.save("j1", state)

        loaded = await mock_persistence.load("j1")
        assert loaded is not None
        assert loaded["job_id"] == "j1"
        assert loaded["status"] == "pending"
        assert loaded["document_text"] == "hello"

        # Verify deep copy semantics — mutation should not affect stored state
        loaded["status"] = "mutated"
        reloaded = await mock_persistence.load("j1")
        assert reloaded is not None
        assert reloaded["status"] == "pending"

    @pytest.mark.asyncio
    async def test_persistence_snapshot_rollback(self, mock_persistence: InMemoryStatePersistence) -> None:
        # Save initial state
        state_v1 = {"job_id": "j2", "status": "classifying", "score": 0.5}
        await mock_persistence.save("j2", state_v1)

        # Create snapshot
        snap_id = await mock_persistence.snapshot("j2")
        assert isinstance(snap_id, str)
        assert len(snap_id) > 0

        # Mutate state
        state_v2 = {"job_id": "j2", "status": "extracting", "score": 0.9}
        await mock_persistence.save("j2", state_v2)
        current = await mock_persistence.load("j2")
        assert current is not None
        assert current["status"] == "extracting"

        # Rollback
        restored = await mock_persistence.rollback("j2", snap_id)
        assert restored is not None
        assert restored["status"] == "classifying"
        assert restored["score"] == 0.5

        # Verify current state matches rollback
        current_after = await mock_persistence.load("j2")
        assert current_after is not None
        assert current_after["status"] == "classifying"

    @pytest.mark.asyncio
    async def test_persistence_load_nonexistent(self, mock_persistence: InMemoryStatePersistence) -> None:
        result = await mock_persistence.load("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_persistence_rollback_bad_snapshot(self, mock_persistence: InMemoryStatePersistence) -> None:
        await mock_persistence.save("j3", {"job_id": "j3"})
        result = await mock_persistence.rollback("j3", "bad-snap-id")
        assert result is None


class TestTraceStep:
    """Verify TraceStep creation and completion."""

    def test_trace_step_creation(self) -> None:
        step = TraceStep(
            node_name="classify_document",
            input_keys=["document_text"],
            output_keys=["document_type", "classification_confidence"],
        )

        assert step.node_name == "classify_document"
        assert step.input_keys == ["document_text"]
        assert step.output_keys == ["document_type", "classification_confidence"]
        assert step.started_at != ""
        assert step.completed_at == ""
        assert step.duration_ms == 0.0
        assert step.cost == 0.0
        assert step.error == ""

        # Complete the step
        step.complete()
        assert step.completed_at != ""
        assert step.duration_ms >= 0.0
