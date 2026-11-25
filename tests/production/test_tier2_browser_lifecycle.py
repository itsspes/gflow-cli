"""Tier 2: macOS Browser Policy & Warm-Session Lifecycle Tests.

Validates macOS anti-focus-steal safety, non-darwin no-op behavior, login
foreground exemption, and WarmSessionManager reuse, TTL eviction, crash
recovery, error eviction, and per-profile concurrency serialization.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from gflow_cli import macos_window
from gflow_cli.api import warm_session
from gflow_cli.api.warm_session import (
    WarmSessionManager,
    close_default_warm_session_manager,
    get_warm_session_manager,
    open_flow_client,
    reset_warm_session_manager,
)
from gflow_cli.config import Settings

# ---------------------------------------------------------------------------
# 1. macOS Anti-Focus-Steal Policy
# ---------------------------------------------------------------------------


def test_background_launch_disabled_under_pytest_and_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(macos_window.sys, "platform", "darwin")
    # PYTEST_CURRENT_TEST is set by pytest itself -> must stay inactive by default
    assert macos_window.is_background_launch_active(headless=False, enabled=True) is False


def test_background_launch_active_matrix_on_darwin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(macos_window.sys, "platform", "darwin")
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    assert macos_window.is_background_launch_active(headless=False, enabled=True) is True
    assert macos_window.is_background_launch_active(headless=True, enabled=True) is False
    assert macos_window.is_background_launch_active(headless=False, enabled=False) is False


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_non_darwin_platforms_are_strict_noops(
    monkeypatch: pytest.MonkeyPatch, platform: str
) -> None:
    monkeypatch.setattr(macos_window.sys, "platform", platform)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    calls: list[list[str]] = []
    monkeypatch.setattr(
        macos_window.subprocess,
        "run",
        lambda cmd, **_: calls.append(list(cmd)),
    )

    assert macos_window.is_background_launch_active(headless=False, enabled=True) is False
    assert macos_window.capture_frontmost_app() is None
    assert macos_window.background_chrome_and_restore_focus("Terminal") is False
    assert calls == []


def test_capture_frontmost_app_filters_chrome_and_unsafe_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(macos_window.sys, "platform", "darwin")

    monkeypatch.setattr(macos_window, "_run_osascript", lambda _script: "Antigravity")
    assert macos_window.capture_frontmost_app() == "Antigravity"

    monkeypatch.setattr(macos_window, "_run_osascript", lambda _script: "Google Chrome")
    assert macos_window.capture_frontmost_app() is None

    # Injection attempt must be rejected by allowlist
    monkeypatch.setattr(
        macos_window,
        "_run_osascript",
        lambda _script: 'Terminal"; do shell script "id',
    )
    assert macos_window.capture_frontmost_app() is None


def test_background_chrome_and_restore_focus_builds_safe_script_and_swallows_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(macos_window.sys, "platform", "darwin")
    recorded_scripts: list[str] = []

    def fake_run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        recorded_scripts.append(cmd[2])
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(macos_window.subprocess, "run", fake_run)
    assert macos_window.background_chrome_and_restore_focus("iTerm2") is True
    assert len(recorded_scripts) == 1
    assert 'set visible of process "Google Chrome" to false' in recorded_scripts[0]
    assert 'set frontmost of process "iTerm2" to true' in recorded_scripts[0]

    # Timeout / OSError must be best-effort (never raise)
    def failing_run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=1.5)

    monkeypatch.setattr(macos_window.subprocess, "run", failing_run)
    assert macos_window.background_chrome_and_restore_focus("iTerm2") is False


async def test_macos_background_launch_guard_captures_before_and_restores_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(macos_window, "is_background_launch_active", lambda **_: True)
    events: list[str] = []

    def fake_capture() -> str:
        events.append("capture")
        return "Cursor"

    def fake_restore(app: str | None) -> bool:
        events.append(f"restore:{app}")
        return True

    monkeypatch.setattr(macos_window, "capture_frontmost_app", fake_capture)
    monkeypatch.setattr(macos_window, "background_chrome_and_restore_focus", fake_restore)

    async with macos_window.macos_background_launch_guard(headless=False, enabled=True):
        events.append("launch_chrome")

    assert events == ["capture", "launch_chrome", "restore:Cursor"]


def test_auth_login_module_does_not_invoke_background_launch_guard() -> None:
    """Interactive login must always remain foregrounded for manual Google sign-in."""
    auth_dir = Path(macos_window.__file__).resolve().parent / "auth"
    for py_file in auth_dir.glob("*.py"):
        source = py_file.read_text(encoding="utf-8")
        assert "macos_background_launch_guard" not in source, (
            f"{py_file.name} must not hide the interactive login window"
        )


# ---------------------------------------------------------------------------
# 2. Warm Persistent Browser Session Pool (WarmSessionManager)
# ---------------------------------------------------------------------------


class _FakePage:
    def __init__(self, url: str = "https://labs.google/fx/tools/flow") -> None:
        self.url = url
        self.goto_calls: list[str] = []

    async def goto(self, url: str, **_: Any) -> None:
        self.goto_calls.append(url)
        self.url = url


class _FakeFlowClient:
    def __init__(
        self,
        *,
        profile_dir: Path,
        headless: bool = False,
        transport: str | None = None,
        out_dir: Path | None = None,
        settings: Settings | None = None,
        alive: bool = True,
    ) -> None:
        self.profile_dir = profile_dir
        self.headless = headless
        self.transport_name = transport
        self.out_dir = out_dir
        self.settings = settings
        self.alive = alive
        self.entered = 0
        self.exited = 0
        self._page = _FakePage()
        self._context: object | None = object()
        self.transport = MagicMock()
        self.transport.park_deferred_page = None

    async def __aenter__(self) -> _FakeFlowClient:
        self.entered += 1
        return self

    async def __aexit__(self, *_: object) -> None:
        self.exited += 1
        self._context = None
        self.alive = False

    async def health_check(self) -> bool:
        return self.alive


class _FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


async def test_warm_session_reuses_client_and_parks_page_on_release(tmp_path: Path) -> None:
    clock = _FakeClock()
    created: list[_FakeFlowClient] = []

    def factory(**kwargs: Any) -> _FakeFlowClient:
        client = _FakeFlowClient(**kwargs)
        created.append(client)
        return client

    manager = WarmSessionManager(ttl_seconds=60.0, client_factory=factory, clock=clock)
    profile = tmp_path / "prof-a"

    async with manager.borrow(profile) as first:
        assert first.entered == 1
        first._page.url = "https://labs.google/fx/tools/flow/project/123"

    # Page should be parked on about:blank after borrow release
    assert first._page.url == "about:blank"
    assert first._page.goto_calls == ["about:blank"]
    assert manager.has_session(profile) is True

    clock.advance(10.0)
    async with manager.borrow(profile) as second:
        assert second is first

    stats = manager.stats()
    assert len(created) == 1
    assert stats.hits == 1
    assert stats.misses == 1
    assert stats.active_profiles == 1

    closed = await manager.close_all()
    assert closed == 1
    assert first.exited == 1
    assert manager.has_session(profile) is False


async def test_warm_session_idle_ttl_eviction(tmp_path: Path) -> None:
    clock = _FakeClock()
    created: list[_FakeFlowClient] = []

    def factory(**kwargs: Any) -> _FakeFlowClient:
        client = _FakeFlowClient(**kwargs)
        created.append(client)
        return client

    manager = WarmSessionManager(ttl_seconds=30.0, client_factory=factory, clock=clock)
    profile = tmp_path / "prof-ttl"

    async with manager.borrow(profile) as first:
        pass

    clock.advance(31.0)
    evicted = await manager.evict_expired()
    assert evicted == 1
    assert first.exited == 1
    assert manager.has_session(profile) is False

    # Next borrow transparently spawns a fresh client
    async with manager.borrow(profile) as second:
        assert second is not first
        assert len(created) == 2

    await manager.close_all()


async def test_warm_session_recovers_from_unhealthy_or_crashed_browser(tmp_path: Path) -> None:
    clock = _FakeClock()
    created: list[_FakeFlowClient] = []

    def factory(**kwargs: Any) -> _FakeFlowClient:
        client = _FakeFlowClient(**kwargs)
        created.append(client)
        return client

    manager = WarmSessionManager(ttl_seconds=300.0, client_factory=factory, clock=clock)
    profile = tmp_path / "prof-crash"

    async with manager.borrow(profile) as first:
        pass

    # Simulate browser crash between calls
    first.alive = False

    async with manager.borrow(profile) as recovered:
        assert recovered is not first
        assert first.exited == 1
        assert recovered.entered == 1
        assert manager.stats().evictions == 1

    await manager.close_all()


async def test_warm_session_evicts_on_borrow_error_and_config_change(tmp_path: Path) -> None:
    clock = _FakeClock()
    created: list[_FakeFlowClient] = []

    def factory(**kwargs: Any) -> _FakeFlowClient:
        client = _FakeFlowClient(**kwargs)
        created.append(client)
        return client

    manager = WarmSessionManager(ttl_seconds=300.0, client_factory=factory, clock=clock)
    profile = tmp_path / "prof-err"

    # Exception inside borrow evicts the potentially poisoned client
    broken: _FakeFlowClient | None = None
    with pytest.raises(RuntimeError, match="boom"):
        async with manager.borrow(profile) as client:
            broken = client
            raise RuntimeError("boom")
    assert broken is not None
    assert broken.exited == 1
    assert manager.has_session(profile) is False

    # Config change (headless False -> True) evicts and recreates
    async with manager.borrow(profile, headless=False) as headed:
        pass
    async with manager.borrow(profile, headless=True) as headless_client:
        assert headless_client is not headed
        assert headed.exited == 1

    await manager.close_all()


async def test_warm_session_serializes_same_profile_and_isolates_distinct_profiles(
    tmp_path: Path,
) -> None:
    clock = _FakeClock()
    manager = WarmSessionManager(
        ttl_seconds=300.0,
        client_factory=lambda **kw: _FakeFlowClient(**kw),
        clock=clock,
    )
    prof_a = tmp_path / "prof-a"
    prof_b = tmp_path / "prof-b"

    overlap_detected = False
    in_flight = False

    async def worker_a() -> None:
        nonlocal overlap_detected, in_flight
        async with manager.borrow(prof_a):
            if in_flight:
                overlap_detected = True
            in_flight = True
            await asyncio.sleep(0.02)
            in_flight = False

    await asyncio.gather(worker_a(), worker_a())
    assert overlap_detected is False

    async with manager.borrow(prof_a) as client_a, manager.borrow(prof_b) as client_b:
        assert client_a is not client_b
        assert manager.stats().active_profiles == 2

    assert await manager.close_all() == 2


async def test_open_flow_client_respects_warm_session_toggle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reset_warm_session_manager()
    created: list[_FakeFlowClient] = []

    def factory(**kwargs: Any) -> _FakeFlowClient:
        client = _FakeFlowClient(**kwargs)
        created.append(client)
        return client

    monkeypatch.setattr(warm_session, "FlowApiClient", factory)
    profile = tmp_path / "prof-toggle"

    # Disabled (default): cold launch + immediate exit per call
    cold_settings = Settings(warm_session_enabled=False)
    async with open_flow_client(profile, settings=cold_settings) as c1:
        assert c1.entered == 1
    assert c1.exited == 1

    # Enabled: warm singleton keeps client alive across calls until shutdown
    warm_settings = Settings(warm_session_enabled=True, warm_session_ttl_seconds=60.0)
    mgr = get_warm_session_manager(settings=warm_settings)
    assert mgr.ttl_seconds == 60.0
    async with open_flow_client(profile, settings=warm_settings) as w1:
        pass
    assert w1.exited == 0
    async with open_flow_client(profile, settings=warm_settings) as w2:
        assert w2 is w1
    assert await close_default_warm_session_manager() == 1
    assert w1.exited == 1
    reset_warm_session_manager()
