"""Local loopback warm-browser daemon backed by ``WarmSessionManager``.

Keeps a single hidden headed-Chrome session alive across consecutive CLI/MCP
runs so repeated commands reuse one warm browser with zero pop-up latency.
State (PID, host, port, bearer token, TTL) is stored in
``<GFLOW_CLI_HOME>/warm_daemon.json`` with ``0o600`` permissions on POSIX.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import secrets
import signal
import sys
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import structlog
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from gflow_cli.api.warm_session import WarmSessionManager, WarmSessionStats
from gflow_cli.config import Settings, get_settings

log = structlog.get_logger(__name__)

_STATE_FILENAME = "warm_daemon.json"
DEFAULT_DAEMON_HOST = "127.0.0.1"
DEFAULT_DAEMON_PORT = 8765


@dataclass(frozen=True)
class DaemonState:
    """Persisted metadata for a running local warm daemon."""

    pid: int
    host: str
    port: int
    token: str
    ttl_seconds: float
    started_at: str

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


def daemon_state_path(settings: Settings | None = None) -> Path:
    """Return the path to the warm daemon PID/port state file."""
    cfg = settings or get_settings()
    return cfg.home / _STATE_FILENAME


def write_daemon_state(state: DaemonState, settings: Settings | None = None) -> Path:
    """Persist daemon state atomically with owner-only permissions on POSIX."""
    path = daemon_state_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")
    if sys.platform != "win32":
        try:
            tmp.chmod(0o600)
        except OSError:
            pass
    tmp.replace(path)
    return path


def read_daemon_state(settings: Settings | None = None) -> DaemonState | None:
    """Read persisted daemon state, returning ``None`` if absent or malformed."""
    path = daemon_state_path(settings)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return DaemonState(
            pid=int(raw["pid"]),
            host=str(raw["host"]),
            port=int(raw["port"]),
            token=str(raw["token"]),
            ttl_seconds=float(raw["ttl_seconds"]),
            started_at=str(raw["started_at"]),
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None


def clear_daemon_state(settings: Settings | None = None) -> None:
    """Remove the daemon state file if present."""
    daemon_state_path(settings).unlink(missing_ok=True)


def is_pid_alive(pid: int) -> bool:
    """Return True if process ``pid`` is alive."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _stats_dict(stats: WarmSessionStats) -> dict[str, Any]:
    return {
        "active_profiles": stats.active_profiles,
        "hits": stats.hits,
        "misses": stats.misses,
        "evictions": stats.evictions,
        "profile_idle_seconds": stats.profile_idle_seconds,
    }


def _authorized(request: Request, expected_token: str) -> bool:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer":
        return False
    return hmac.compare_digest(token.encode("utf-8"), expected_token.encode("utf-8"))


def build_warm_daemon_app(
    manager: WarmSessionManager,
    *,
    token: str,
    on_shutdown: Callable[[], None] | None = None,
) -> Starlette:
    """Build the loopback ASGI app exposing warm daemon health, status, evict, and shutdown."""

    async def health(_request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "pid": os.getpid(),
                "warm_session_enabled": True,
                "ttl_seconds": manager.ttl_seconds,
                "stats": _stats_dict(manager.stats()),
            }
        )

    async def status(request: Request) -> JSONResponse:
        if not _authorized(request, token):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return JSONResponse(
            {
                "status": "running",
                "pid": os.getpid(),
                "warm_session_enabled": True,
                "ttl_seconds": manager.ttl_seconds,
                "stats": _stats_dict(manager.stats()),
            }
        )

    async def evict(request: Request) -> JSONResponse:
        if not _authorized(request, token):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        evicted = await manager.evict_expired()
        return JSONResponse({"status": "ok", "evicted": evicted})

    async def shutdown(request: Request) -> JSONResponse:
        if not _authorized(request, token):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        closed = await manager.close_all()
        if on_shutdown is not None:
            on_shutdown()
        return JSONResponse({"status": "shutting_down", "closed_sessions": closed})

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncGenerator[None]:
        try:
            yield
        finally:
            await manager.close_all()

    return Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/status", status, methods=["GET"]),
            Route("/evict", evict, methods=["POST"]),
            Route("/shutdown", shutdown, methods=["POST"]),
        ],
        lifespan=lifespan,
    )


async def run_warm_daemon(
    *,
    host: str = DEFAULT_DAEMON_HOST,
    port: int = DEFAULT_DAEMON_PORT,
    ttl_seconds: float | None = None,
    settings: Settings | None = None,
) -> None:
    """Run the warm browser daemon over loopback HTTP until stopped."""
    import uvicorn

    cfg = settings or get_settings()
    ttl = ttl_seconds if ttl_seconds is not None else cfg.warm_session_ttl_seconds
    os.environ["GFLOW_CLI_WARM_SESSION_ENABLED"] = "1"
    token = secrets.token_urlsafe(24)
    manager = WarmSessionManager(ttl_seconds=ttl)

    server_holder: list[uvicorn.Server] = []

    def _trigger_exit() -> None:
        if server_holder:
            server_holder[0].should_exit = True

    app = build_warm_daemon_app(manager, token=token, on_shutdown=_trigger_exit)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    server_holder.append(server)

    state = DaemonState(
        pid=os.getpid(),
        host=host,
        port=port,
        token=token,
        ttl_seconds=ttl,
        started_at=datetime.now(UTC).isoformat(),
    )
    write_daemon_state(state, cfg)
    log.info("warm_daemon.started", host=host, port=port, pid=state.pid, ttl_seconds=ttl)
    try:
        await server.serve()
    finally:
        await manager.close_all()
        clear_daemon_state(cfg)
        log.info("warm_daemon.stopped", pid=state.pid)


def query_daemon_status(settings: Settings | None = None) -> dict[str, Any]:
    """Check whether the local warm daemon is running and reachable."""
    cfg = settings or get_settings()
    state = read_daemon_state(cfg)
    if state is None:
        return {"status": "stopped", "running": False, "state_file": str(daemon_state_path(cfg))}
    if not is_pid_alive(state.pid):
        clear_daemon_state(cfg)
        return {
            "status": "stopped",
            "running": False,
            "stale_cleaned": True,
            "state_file": str(daemon_state_path(cfg)),
        }
    try:
        resp = httpx.get(
            f"{state.base_url}/status",
            headers={"Authorization": f"Bearer {state.token}"},
            timeout=2.0,
        )
        if resp.status_code == 200:
            payload: dict[str, Any] = resp.json()
            payload.update(
                {
                    "running": True,
                    "host": state.host,
                    "port": state.port,
                    "started_at": state.started_at,
                }
            )
            return payload
    except httpx.HTTPError:
        pass
    return {
        "status": "unreachable",
        "running": True,
        "pid": state.pid,
        "host": state.host,
        "port": state.port,
        "started_at": state.started_at,
    }


def stop_warm_daemon(settings: Settings | None = None) -> dict[str, Any]:
    """Request graceful shutdown of the running warm daemon, falling back to SIGTERM."""
    cfg = settings or get_settings()
    state = read_daemon_state(cfg)
    if state is None:
        return {"status": "not_running", "stopped": False}
    closed_sessions = 0
    if is_pid_alive(state.pid):
        try:
            resp = httpx.post(
                f"{state.base_url}/shutdown",
                headers={"Authorization": f"Bearer {state.token}"},
                timeout=3.0,
            )
            if resp.status_code == 200:
                closed_sessions = int(resp.json().get("closed_sessions", 0))
        except httpx.HTTPError:
            try:
                os.kill(state.pid, signal.SIGTERM)
            except OSError:
                pass
    clear_daemon_state(cfg)
    return {
        "status": "stopped",
        "stopped": True,
        "pid": state.pid,
        "closed_sessions": closed_sessions,
    }


def start_warm_daemon_sync(
    *,
    host: str = DEFAULT_DAEMON_HOST,
    port: int = DEFAULT_DAEMON_PORT,
    ttl_seconds: float | None = None,
) -> None:
    """Synchronous entry point for ``gflow daemon start``."""
    asyncio.run(run_warm_daemon(host=host, port=port, ttl_seconds=ttl_seconds))
