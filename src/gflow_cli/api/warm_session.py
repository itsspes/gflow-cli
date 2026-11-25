"""Warm persistent browser session pool for consecutive Flow operations.

Cold-launching headed Chrome per CLI/MCP/daemon call incurs 15-30 seconds of
startup overhead and repeated macOS window activations. ``WarmSessionManager``
keeps at most one live :class:`~gflow_cli.api.client.FlowApiClient` per profile
directory across consecutive calls, enforcing:

- **Per-profile serialization** via an ``asyncio.Lock`` (preventing
  ``ProfileLease`` contention and Playwright page races).
- **Idle TTL eviction** using a monotonic clock (injectable for tests).
- **Health verification & crash recovery**: dead or mismatched clients are
  evicted and transparently replaced on the next borrow.
- **Safe page parking**: between borrows, the pooled page is parked on
  ``about:blank`` so consecutive jobs never inherit a dirty composer state.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import structlog

from gflow_cli.api.client import FlowApiClient
from gflow_cli.config import Settings, get_settings

if TYPE_CHECKING:
    from playwright.async_api import Page

log = structlog.get_logger(__name__)

_DEFAULT_TTL_SECONDS = 1200.0
_PARK_TIMEOUT_MS = 5_000


@dataclass(frozen=True)
class WarmSessionStats:
    """Point-in-time snapshot of the warm session pool."""

    active_profiles: int
    hits: int
    misses: int
    evictions: int
    profile_idle_seconds: dict[str, float]


@dataclass
class _WarmEntry:
    client: Any
    headless: bool
    transport: str | None
    last_used_at: float


def _profile_key(profile_dir: Path) -> str:
    return str(profile_dir.expanduser().resolve())


async def _park_client(client: Any) -> None:
    """Park a warm client's page on ``about:blank`` between borrows (best-effort)."""
    transport: Any = getattr(client, "transport", None)
    if transport is not None:
        park_deferred = getattr(transport, "park_deferred_page", None)
        if callable(park_deferred):
            try:
                await cast("Callable[[], Awaitable[None]]", park_deferred)()
            except Exception as exc:  # noqa: BLE001 - parking is best-effort
                log.debug("warm_session.deferred_park_error", error=type(exc).__name__)
    page: Page | None = getattr(client, "_page", None)
    if page is None:
        return
    try:
        if getattr(page, "url", "") != "about:blank":
            await page.goto("about:blank", wait_until="commit", timeout=_PARK_TIMEOUT_MS)
    except Exception as exc:  # noqa: BLE001 - parking is best-effort
        log.debug("warm_session.park_page_error", error=type(exc).__name__)


async def _is_client_alive(client: Any) -> bool:
    """Check whether a cached client is still usable without raising."""
    health_check = getattr(client, "health_check", None)
    if callable(health_check):
        try:
            return bool(await cast("Callable[[], Awaitable[bool]]", health_check)())
        except Exception:  # noqa: BLE001 - treat any probe failure as dead
            return False
    return getattr(client, "_context", None) is not None


async def _close_client_safely(client: Any) -> None:
    """Tear down a pooled client via ``__aexit__``, never raising."""
    try:
        await client.__aexit__(None, None, None)
    except Exception as exc:  # noqa: BLE001 - eviction cleanup must not raise
        log.warning("warm_session.close_error", error=type(exc).__name__)


class WarmSessionManager:
    """Per-profile warm ``FlowApiClient`` pool with idle TTL and crash recovery."""

    def __init__(
        self,
        *,
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
        evict_on_error: bool = True,
        park_on_release: bool = True,
        client_factory: Callable[..., Any] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if ttl_seconds <= 0:
            msg = f"ttl_seconds must be positive, got {ttl_seconds}"
            raise ValueError(msg)
        self._ttl_seconds = float(ttl_seconds)
        self._evict_on_error = evict_on_error
        self._park_on_release = park_on_release
        self._client_factory: Callable[..., Any] = client_factory or FlowApiClient
        self._clock: Callable[[], float] = clock or time.monotonic
        self._entries: dict[str, _WarmEntry] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @property
    def ttl_seconds(self) -> float:
        return self._ttl_seconds

    def _lock_for(self, key: str) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    def _is_expired(self, entry: _WarmEntry, now: float) -> bool:
        return (now - entry.last_used_at) >= self._ttl_seconds

    def has_session(self, profile_dir: Path) -> bool:
        """True if ``profile_dir`` currently has a pooled entry."""
        return _profile_key(profile_dir) in self._entries

    def stats(self) -> WarmSessionStats:
        """Return pool counters and current per-profile idle ages."""
        now = self._clock()
        idle = {key: max(0.0, now - entry.last_used_at) for key, entry in self._entries.items()}
        return WarmSessionStats(
            active_profiles=len(self._entries),
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            profile_idle_seconds=idle,
        )

    async def _evict_key_unlocked(self, key: str, *, reason: str) -> bool:
        entry = self._entries.pop(key, None)
        if entry is None:
            return False
        self._evictions += 1
        log.info("warm_session.evict", profile_key=key, reason=reason)
        await _close_client_safely(entry.client)
        return True

    async def evict(self, profile_dir: Path, *, reason: str = "manual") -> bool:
        """Evict and close the pooled client for ``profile_dir``, if any."""
        key = _profile_key(profile_dir)
        async with self._lock_for(key):
            return await self._evict_key_unlocked(key, reason=reason)

    async def evict_expired(self) -> int:
        """Evict every idle session whose age meets or exceeds ``ttl_seconds``."""
        now = self._clock()
        expired_keys = [
            key for key, entry in list(self._entries.items()) if self._is_expired(entry, now)
        ]
        evicted = 0
        for key in expired_keys:
            lock = self._lock_for(key)
            if lock.locked():
                continue
            async with lock:
                entry = self._entries.get(key)
                if entry is not None and self._is_expired(entry, self._clock()):
                    await self._evict_key_unlocked(key, reason="ttl_expired")
                    evicted += 1
        return evicted

    async def close_all(self) -> int:
        """Close every pooled session and release all profile leases."""
        closed = 0
        for key in list(self._entries):
            async with self._lock_for(key):
                if await self._evict_key_unlocked(key, reason="shutdown"):
                    closed += 1
        return closed

    async def _get_or_create_unlocked(
        self,
        key: str,
        profile_dir: Path,
        *,
        headless: bool,
        transport: str | None,
        out_dir: Path | None,
        settings: Settings | None,
    ) -> Any:
        entry = self._entries.get(key)
        now = self._clock()
        if entry is not None:
            if self._is_expired(entry, now):
                await self._evict_key_unlocked(key, reason="ttl_expired")
            elif entry.headless != headless or entry.transport != transport:
                await self._evict_key_unlocked(key, reason="config_changed")
            elif not await _is_client_alive(entry.client):
                await self._evict_key_unlocked(key, reason="unhealthy")
            else:
                self._hits += 1
                entry.last_used_at = now
                log.debug("warm_session.hit", profile_key=key)
                return entry.client

        self._misses += 1
        client_kwargs: dict[str, Any] = {
            "profile_dir": profile_dir,
            "headless": headless,
        }
        if transport is not None:
            client_kwargs["transport"] = transport
        if out_dir is not None:
            client_kwargs["out_dir"] = out_dir
        if settings is not None:
            client_kwargs["settings"] = settings
        client = self._client_factory(**client_kwargs)
        entered = await client.__aenter__()
        self._entries[key] = _WarmEntry(
            client=entered,
            headless=headless,
            transport=transport,
            last_used_at=self._clock(),
        )
        log.info("warm_session.miss_created", profile_key=key, headless=headless)
        return entered

    @asynccontextmanager
    async def borrow(
        self,
        profile_dir: Path,
        *,
        headless: bool = False,
        transport: str | None = None,
        out_dir: Path | None = None,
        settings: Settings | None = None,
    ) -> AsyncGenerator[Any]:
        """Borrow a warm ``FlowApiClient`` for ``profile_dir``, serialized per profile."""
        key = _profile_key(profile_dir)
        async with self._lock_for(key):
            client = await self._get_or_create_unlocked(
                key,
                profile_dir,
                headless=headless,
                transport=transport,
                out_dir=out_dir,
                settings=settings,
            )
            failed = False
            try:
                yield client
            except BaseException:
                failed = True
                raise
            finally:
                if failed and self._evict_on_error:
                    await self._evict_key_unlocked(key, reason="borrow_error")
                elif key in self._entries:
                    if self._park_on_release:
                        await _park_client(client)
                    self._entries[key].last_used_at = self._clock()


_default_manager: WarmSessionManager | None = None


def get_warm_session_manager(*, settings: Settings | None = None) -> WarmSessionManager:
    """Return the process-wide ``WarmSessionManager``, initializing on first call."""
    global _default_manager
    if _default_manager is None:
        cfg = settings or get_settings()
        _default_manager = WarmSessionManager(ttl_seconds=cfg.warm_session_ttl_seconds)
    return _default_manager


async def close_default_warm_session_manager() -> int:
    """Close all sessions in the process-wide manager, if initialized."""
    global _default_manager
    if _default_manager is None:
        return 0
    manager = _default_manager
    _default_manager = None
    return await manager.close_all()


def reset_warm_session_manager() -> None:
    """Clear the process-wide manager reference (intended for unit tests)."""
    global _default_manager
    _default_manager = None


@asynccontextmanager
async def open_flow_client(
    profile_dir: Path,
    *,
    headless: bool = False,
    transport: str | None = None,
    out_dir: Path | None = None,
    settings: Settings | None = None,
) -> AsyncGenerator[Any]:
    """Yield a warm pooled ``FlowApiClient`` when enabled, else a cold per-call client."""
    cfg = settings or get_settings()
    if cfg.warm_session_enabled:
        manager = get_warm_session_manager(settings=cfg)
        async with manager.borrow(
            profile_dir,
            headless=headless,
            transport=transport,
            out_dir=out_dir,
            settings=cfg,
        ) as client:
            yield client
        return
    async with FlowApiClient(
        profile_dir=profile_dir,
        headless=headless,
        transport=transport,
        out_dir=out_dir,
        settings=cfg,
    ) as client:
        yield client
