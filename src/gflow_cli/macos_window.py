"""macOS anti-focus-steal helper for headed generation browser launches.

Google Flow requires headed Chrome (``headless=False``) for reCAPTCHA Enterprise
and Chrome cookie-store symmetry. On macOS, setting ``--window-position=-30000,-30000``
does not prevent WindowServer from activating Chrome and stealing keyboard focus
from the user's editor or terminal.

This module captures the frontmost macOS application before Playwright launches
headed Chrome and immediately hides Chrome / restores the previous application
after launch. Interactive login (``gflow auth login``) deliberately does not use
this helper so the sign-in window stays foreground.

All operations are strictly guarded on ``sys.platform == "darwin"``, skipped in
headless/CI environments, bounded by a short subprocess timeout, and best-effort
(never raising or aborting a generation run).
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog

log = structlog.get_logger(__name__)

_OSASCRIPT_BIN = "/usr/bin/osascript"
_OSASCRIPT_TIMEOUT_S = 1.5
# Strict allowlist for macOS process names interpolated into AppleScript.
_SAFE_APP_NAME_RE = re.compile(r"^[A-Za-z0-9 ._+-]{1,128}$")
_CHROME_PROCESS_NAMES: tuple[str, ...] = (
    "Google Chrome",
    "Google Chrome for Testing",
    "Chromium",
)


def _is_ci_env() -> bool:
    """True when running in a CI or pytest harness where GUI focus calls are unwanted."""
    return bool(
        os.environ.get("CI")
        or os.environ.get("GITHUB_ACTIONS")
        or os.environ.get("PYTEST_CURRENT_TEST")
    )


def is_background_launch_active(*, headless: bool, enabled: bool) -> bool:
    """Return True when macOS background-launch focus protection should run."""
    return sys.platform == "darwin" and enabled and not headless and not _is_ci_env()


def _run_osascript(script: str, *, timeout: float = _OSASCRIPT_TIMEOUT_S) -> str | None:
    """Run a short AppleScript via ``/usr/bin/osascript``; return stdout or None."""
    if sys.platform != "darwin":
        return None
    try:
        completed = subprocess.run(  # noqa: S603 - fixed absolute binary, no shell
            [_OSASCRIPT_BIN, "-e", script],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("macos_window.osascript_error", error=type(exc).__name__)
        return None
    if completed.returncode != 0:
        log.debug("macos_window.osascript_nonzero", returncode=completed.returncode)
        return None
    return completed.stdout.strip() or None


def capture_frontmost_app() -> str | None:
    """Return the name of the currently active non-Chrome macOS app, or None."""
    if sys.platform != "darwin":
        return None
    script = (
        'tell application "System Events" to get name of first application process '
        "whose frontmost is true"
    )
    app_name = _run_osascript(script)
    if not app_name or app_name in _CHROME_PROCESS_NAMES:
        return None
    if not _SAFE_APP_NAME_RE.fullmatch(app_name):
        return None
    return app_name


def background_chrome_and_restore_focus(previous_app: str | None) -> bool:
    """Hide headed Chrome processes and restore focus to ``previous_app`` on macOS.

    Best-effort: never raises. Returns True when AppleScript completed cleanly.
    """
    if sys.platform != "darwin":
        return False
    lines = ['tell application "System Events"']
    for proc in _CHROME_PROCESS_NAMES:
        lines.append(f'  if exists process "{proc}" then set visible of process "{proc}" to false')
    if previous_app and _SAFE_APP_NAME_RE.fullmatch(previous_app):
        lines.append(
            f'  if exists process "{previous_app}" then '
            f'set frontmost of process "{previous_app}" to true'
        )
    lines.append("end tell")
    lines.append('return "ok"')
    ok = _run_osascript("\n".join(lines)) == "ok"
    if ok:
        log.debug("macos_window.focus_restored", previous_app=previous_app)
    return ok


@asynccontextmanager
async def macos_background_launch_guard(
    *,
    headless: bool,
    enabled: bool,
) -> AsyncGenerator[None]:
    """Capture frontmost app before launch and restore focus immediately after.

    Intended to wrap ``launch_persistent_context(...)`` for non-login headed
    generation launches. No-op off macOS, when headless, in CI/pytest, or when
    disabled via ``GFLOW_CLI_MACOS_BACKGROUND_LAUNCH=0``.
    """
    if not is_background_launch_active(headless=headless, enabled=enabled):
        yield
        return
    previous_app = await asyncio.to_thread(capture_frontmost_app)
    yield
    await asyncio.to_thread(background_chrome_and_restore_focus, previous_app)
