"""Tests for the FastAPI endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api.routes import router


def _create_test_app() -> FastAPI:
    """Build a minimal FastAPI app with the router and mock app state."""
    app = FastAPI()
    app.include_router(router)

    # Mock app state
    app.state.jobs = {}
    app.state.redis = MagicMock()
    app.state.graph = None

    # Make redis.ping() an async mock that raises so health reports degraded
    app.state.redis.ping = AsyncMock(side_effect=ConnectionError("no redis"))

    return app


@pytest.fixture()
def test_app() -> FastAPI:
    return _create_test_app()


@pytest.fixture()
def async_client(test_app: FastAPI):
    transport = ASGITransport(app=test_app)
    return AsyncClient(transport=transport, base_url="http://test")


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_endpoint(self, async_client: AsyncClient) -> None:
        async with async_client as client:
            resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "degraded")
        assert "redis_connected" in data
        assert "pending_jobs" in data


class TestAnalyzeEndpoint:
    @pytest.mark.asyncio
    async def test_analyze_returns_job_id(self, async_client: AsyncClient) -> None:
        async with async_client as client:
            resp = await client.post(
                "/analyze",
                json={"document_text": "This is a sample contract between two parties."},
            )
        assert resp.status_code == 202
        data = resp.json()
        assert "job_id" in data
        assert isinstance(data["job_id"], str)
        assert len(data["job_id"]) > 0
        assert "status" in data


class TestJobStatusEndpoint:
    @pytest.mark.asyncio
    async def test_job_status_not_found(self, async_client: AsyncClient) -> None:
        async with async_client as client:
            resp = await client.get("/jobs/nonexistent-id/status")
        assert resp.status_code == 404
        data = resp.json()
        assert "detail" in data
