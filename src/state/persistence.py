"""State persistence backends — Redis (primary) with in-memory fallback."""

from __future__ import annotations

import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.config import AppConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serialize(state: dict[str, Any]) -> str:
    """JSON-serialize state, converting non-serializable objects to strings."""

    def _default(obj: Any) -> Any:
        return str(obj)

    return json.dumps(state, default=_default)


def _deserialize(raw: str | bytes) -> dict[str, Any]:
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class StatePersistence(ABC):
    """Interface for persisting and snapshotting pipeline state."""

    @abstractmethod
    async def save(self, job_id: str, state: dict[str, Any]) -> None: ...

    @abstractmethod
    async def load(self, job_id: str) -> dict[str, Any] | None: ...

    @abstractmethod
    async def snapshot(self, job_id: str) -> str:
        """Create a point-in-time snapshot. Returns the snapshot id."""
        ...

    @abstractmethod
    async def rollback(self, job_id: str, snapshot_id: str) -> dict[str, Any] | None:
        """Restore state from a snapshot. Returns the restored state or None."""
        ...

    @abstractmethod
    async def list_snapshots(self, job_id: str) -> list[dict[str, Any]]: ...


# ---------------------------------------------------------------------------
# Redis implementation
# ---------------------------------------------------------------------------


class RedisStatePersistence(StatePersistence):
    """Async Redis-backed persistence with versioned snapshots."""

    def __init__(self, redis_url: str) -> None:
        import redis.asyncio as aioredis

        self._redis = aioredis.from_url(redis_url, decode_responses=True)
        self._prefix = "langgraph:state"
        self._snap_prefix = "langgraph:snap"

    def _key(self, job_id: str) -> str:
        return f"{self._prefix}:{job_id}"

    def _snap_key(self, job_id: str, snap_id: str) -> str:
        return f"{self._snap_prefix}:{job_id}:{snap_id}"

    def _snap_index_key(self, job_id: str) -> str:
        return f"{self._snap_prefix}:{job_id}:index"

    async def save(self, job_id: str, state: dict[str, Any]) -> None:
        await self._redis.set(self._key(job_id), _serialize(state))

    async def load(self, job_id: str) -> dict[str, Any] | None:
        raw = await self._redis.get(self._key(job_id))
        if raw is None:
            return None
        return _deserialize(raw)

    async def snapshot(self, job_id: str) -> str:
        raw = await self._redis.get(self._key(job_id))
        if raw is None:
            msg = f"No state found for job {job_id}"
            raise ValueError(msg)

        snap_id = uuid.uuid4().hex[:12]
        ts = time.time()

        await self._redis.set(self._snap_key(job_id, snap_id), raw)

        # Append to ordered index (score = timestamp)
        await self._redis.zadd(self._snap_index_key(job_id), {snap_id: ts})

        return snap_id

    async def rollback(self, job_id: str, snapshot_id: str) -> dict[str, Any] | None:
        raw = await self._redis.get(self._snap_key(job_id, snapshot_id))
        if raw is None:
            return None

        await self._redis.set(self._key(job_id), raw)
        return _deserialize(raw)

    async def list_snapshots(self, job_id: str) -> list[dict[str, Any]]:
        entries = await self._redis.zrangebyscore(self._snap_index_key(job_id), "-inf", "+inf", withscores=True)
        return [{"snapshot_id": snap_id, "created_at": ts} for snap_id, ts in entries]

    async def close(self) -> None:
        await self._redis.aclose()


# ---------------------------------------------------------------------------
# In-memory fallback
# ---------------------------------------------------------------------------


class InMemoryStatePersistence(StatePersistence):
    """Dict-based persistence for local development and testing."""

    def __init__(self) -> None:
        self._states: dict[str, dict[str, Any]] = {}
        self._snapshots: dict[str, dict[str, dict[str, Any]]] = {}  # job_id -> {snap_id: state}
        self._snap_order: dict[str, list[dict[str, Any]]] = {}  # job_id -> [{id, ts}]

    async def save(self, job_id: str, state: dict[str, Any]) -> None:
        self._states[job_id] = deepcopy(state)

    async def load(self, job_id: str) -> dict[str, Any] | None:
        state = self._states.get(job_id)
        return deepcopy(state) if state is not None else None

    async def snapshot(self, job_id: str) -> str:
        state = self._states.get(job_id)
        if state is None:
            msg = f"No state found for job {job_id}"
            raise ValueError(msg)

        snap_id = uuid.uuid4().hex[:12]
        ts = time.time()

        self._snapshots.setdefault(job_id, {})[snap_id] = deepcopy(state)
        self._snap_order.setdefault(job_id, []).append({"snapshot_id": snap_id, "created_at": ts})

        return snap_id

    async def rollback(self, job_id: str, snapshot_id: str) -> dict[str, Any] | None:
        snap_store = self._snapshots.get(job_id, {})
        state = snap_store.get(snapshot_id)
        if state is None:
            return None

        self._states[job_id] = deepcopy(state)
        return deepcopy(state)

    async def list_snapshots(self, job_id: str) -> list[dict[str, Any]]:
        return list(self._snap_order.get(job_id, []))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


async def get_persistence(config: AppConfig) -> StatePersistence:
    """Create the best available persistence backend.

    Tries Redis first; falls back to in-memory with a warning.
    """
    try:
        store = RedisStatePersistence(config.redis_url)
        # Verify connectivity
        await store._redis.ping()
        logger.info("Using Redis persistence at %s", config.redis_url)
        return store
    except Exception:
        logger.warning(
            "Redis unavailable at %s — falling back to in-memory persistence",
            config.redis_url,
        )
        return InMemoryStatePersistence()
