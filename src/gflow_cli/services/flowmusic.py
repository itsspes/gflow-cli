"""Native Google Flow Music (https://www.flowmusic.app/) automation service.

Drives Google Flow Music over a Playwright persistent Chrome context sharing
the user's existing ``gflow`` profile directory, reusing ``ProfileLease``,
``channel_for_profile``, ``window_position_args``, ``disable_features_args``,
and ``macos_background_launch_guard`` (foreground for interactive ``login``,
silent background for ``status``, ``probe``, and ``generate``).
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import structlog
from playwright.async_api import BrowserContext, Page, Response, async_playwright

from gflow_cli.api._engine import (
    CONTEXT_TEARDOWN_TIMEOUT_S,
    DRIVER_STOP_TIMEOUT_S,
    close_context_bounded,
    run_teardown_step,
)
from gflow_cli.browser_manager import (
    browser_proxy_kwargs,
    channel_for_profile,
    disable_features_args,
    ensure_profile_engine_compatible,
    window_position_args,
)
from gflow_cli.config import Settings, get_settings
from gflow_cli.errors import AuthMissingError, GFlowError
from gflow_cli.macos_window import macos_background_launch_guard
from gflow_cli.profile_lease import ProfileLease

log = structlog.get_logger(__name__)

FLOWMUSIC_URL = "https://www.flowmusic.app/"
FLOWMUSIC_SESSION_URL = "https://www.flowmusic.app/session?t=true"
FLOWMUSIC_ME_URL = "https://www.flowmusic.app/__api/users/me"
_FLOWMUSIC_HOSTS = frozenset({"www.flowmusic.app", "flowmusic.app"})

PROMPT_SELECTOR = 'textarea[aria-label="Chat message"]'
SEND_BUTTON_SELECTOR = 'button[aria-label="Send message"]'
SETTINGS_BUTTON_SELECTOR = 'button[aria-label="Chat settings"]'
UPLOAD_AUDIO_SELECTOR = 'input[type="file"][aria-label="Upload audio file"]'
UPLOAD_IMAGE_SELECTOR = 'input[type="file"][aria-label="Upload image file"]'

_AUTH_BUTTON_TEXTS: tuple[str, ...] = ("Log in", "Sign up", "Continue with Google")
_MIN_AUDIO_BYTES = 32 * 1024
_AUDIO_URL_EXT_RE = re.compile(r"\.(wav|mp3|m4a|ogg|aac|flac)(?:[?#]|$)", re.IGNORECASE)
_CONTENT_TYPE_EXT: dict[str, str] = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/mp4": "m4a",
    "audio/x-m4a": "m4a",
    "audio/m4a": "m4a",
    "audio/ogg": "ogg",
    "audio/flac": "flac",
    "audio/aac": "aac",
}

_PROBE_CONTROLS_JS = """() => {
  const nodes = Array.from(document.querySelectorAll('button, a, textarea, input, audio'));
  return nodes.slice(0, 200).map((el) => {
    const rect = el.getBoundingClientRect();
    const visible = rect.width > 0 && rect.height > 0;
    return {
      tag: el.tagName.toLowerCase(),
      visible,
      aria_label: el.getAttribute('aria-label') || '',
      placeholder: el.getAttribute('placeholder') || '',
      type: el.getAttribute('type') || '',
      text: (el.innerText || el.textContent || '').trim().slice(0, 120),
      href: el.getAttribute('href') || '',
      src: el.getAttribute('src') || '',
    };
  });
}"""

_AUDIO_SRCS_JS = """() => {
  const urls = [];
  for (const el of document.querySelectorAll('audio, audio source')) {
    const src = el.getAttribute('src') || el.currentSrc || '';
    if (src && !src.startsWith('data:')) urls.push(src);
  }
  return urls;
}"""


class FlowMusicError(GFlowError):
    """Typed error for Google Flow Music automation failures."""

    title = "Flow Music Error"
    status = 502
    exit_code = 1


class FlowMusicAuthError(AuthMissingError):
    """Raised when the Flow Music profile is not signed in."""

    title = "Flow Music Authentication Required"
    _default_remediation = "Run: gflow music login"


@dataclass(frozen=True)
class FlowMusicAuthResult:
    """Result of a Flow Music login or auth status check."""

    signed_in: bool
    url: str
    profile_dir: str
    auth_prompt_visible: bool
    composer_visible: bool
    detail: str = ""
    account: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "signed_in" if self.signed_in else "signed_out",
            "signed_in": self.signed_in,
            "url": self.url,
            "profile_dir": self.profile_dir,
            "auth_prompt_visible": self.auth_prompt_visible,
            "composer_visible": self.composer_visible,
            "account": self.account,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class FlowMusicGenerateResult:
    """Result of generating and saving a Flow Music track."""

    output_file: Path
    format: str
    bytes: int
    prompt: str
    source_url: str | None
    capture_method: str
    analysis_file: Path | None = None
    master_file: Path | None = None
    qa_verdict: str | None = None
    qa_review_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "succeeded",
            "output_file": str(self.output_file),
            "format": self.format,
            "bytes": self.bytes,
            "prompt": self.prompt,
            "source_url": self.source_url,
            "capture_method": self.capture_method,
            "analysis_file": str(self.analysis_file) if self.analysis_file else None,
            "master_file": str(self.master_file) if self.master_file else None,
            "qa_verdict": self.qa_verdict,
            "qa_review_path": str(self.qa_review_path) if self.qa_review_path else None,
        }


@dataclass
class _CapturedAudio:
    data: bytes
    ext: str
    url: str | None
    method: str


def infer_audio_extension(
    *,
    content_type: str | None = None,
    url: str | None = None,
    fallback: str = "wav",
) -> str:
    """Infer audio file extension from HTTP Content-Type or URL."""
    if content_type:
        mime = content_type.split(";", 1)[0].strip().lower()
        if mime in _CONTENT_TYPE_EXT:
            return _CONTENT_TYPE_EXT[mime]
    if url:
        match = _AUDIO_URL_EXT_RE.search(url)
        if match:
            return match.group(1).lower()
    return fallback.lstrip(".").lower() or "wav"


def resolve_music_output_path(
    *,
    ext: str,
    out_file: Path | None = None,
    project_dir: Path | None = None,
    shot: str = "s01",
    settings: Settings | None = None,
) -> Path:
    """Resolve destination path for a generated Flow Music track."""
    clean_ext = ext.lstrip(".").lower() or "wav"
    if out_file is not None:
        dest = out_file.expanduser().resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        return dest
    if project_dir is not None:
        from gflow_cli.services.project_workspace import load_manifest, next_asset_path

        root = project_dir.expanduser().resolve()
        manifest = load_manifest(root)
        music_dir = root / "audio" / "music"
        music_dir.mkdir(parents=True, exist_ok=True)
        return next_asset_path(
            music_dir,
            slug=str(manifest["slug"]),
            shot=shot,
            kind="music",
            ext=clean_ext,
        )
    cfg = settings or get_settings()
    out_root = cfg.output_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return out_root / f"flowmusic_{stamp}.{clean_ext}"


@asynccontextmanager
async def _open_flowmusic_page(
    profile_dir: Path,
    *,
    background: bool,
    settings: Settings | None = None,
) -> AsyncGenerator[Page]:
    """Open a leased persistent Chrome context for Flow Music and guarantee teardown."""
    cfg = settings or get_settings()
    resolved_dir = profile_dir.expanduser().resolve()
    resolved_dir.mkdir(parents=True, exist_ok=True)
    lease = await ProfileLease(resolved_dir).aacquire()
    pw = None
    ctx: BrowserContext | None = None
    try:
        channel = channel_for_profile(resolved_dir)
        ensure_profile_engine_compatible(resolved_dir, channel)
        pw = await async_playwright().start()
        args = [
            "--password-store=basic",
            "--disable-blink-features=AutomationControlled",
            "--disable-dev-shm-usage",
            *disable_features_args(),
        ]
        if background:
            args.extend(window_position_args(cfg.browser_window_position))
        else:
            args.append("--window-position=0,0")
        launch_kwargs: dict[str, Any] = {
            "user_data_dir": str(resolved_dir),
            "headless": False,
            "viewport": {"width": 1280, "height": 800},
            "locale": "en-US",
            "channel": channel,
            "ignore_default_args": ["--enable-automation", "--no-sandbox"],
            "args": args,
            **browser_proxy_kwargs(cfg.browser_proxy),
        }
        async with macos_background_launch_guard(
            headless=False,
            enabled=background and cfg.macos_background_launch,
        ):
            ctx = await pw.chromium.launch_persistent_context(**launch_kwargs)
        await ctx.add_init_script(
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"
        )
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        yield page
    finally:
        if ctx is not None:
            await run_teardown_step(
                close_context_bounded(ctx, owner="flowmusic"),
                timeout=CONTEXT_TEARDOWN_TIMEOUT_S,
                owner="flowmusic",
                step="context_close",
            )
        if pw is not None:
            await run_teardown_step(
                pw.stop(),
                timeout=DRIVER_STOP_TIMEOUT_S,
                owner="flowmusic",
                step="driver_stop",
            )
        lease.release()


async def _fetch_flowmusic_me(page: Page) -> dict[str, Any] | None:
    """Query Flow Music's canonical session endpoint; return user dict when signed in."""
    try:
        resp = await page.request.get(FLOWMUSIC_ME_URL, timeout=10_000)
        if resp.status != 200:
            return None
        data: Any = await resp.json()
        if isinstance(data, dict) and data:
            return cast("dict[str, Any]", data)
    except Exception:
        return None
    return None


def _extract_account_label(me: dict[str, Any] | None) -> str | None:
    """Extract a safe display identifier from the ``/__api/users/me`` payload."""
    if not me:
        return None
    for key in ("email", "handle", "username", "displayName", "name", "id"):
        val = me.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


async def _inspect_auth_state(page: Page) -> tuple[bool, bool, bool, str | None]:
    """Return ``(signed_in, auth_prompt_visible, composer_visible, account)``."""
    host = (urlparse(page.url).hostname or "").lower()
    if host not in _FLOWMUSIC_HOSTS:
        return False, False, False, None

    auth_prompt_visible = False
    for label in _AUTH_BUTTON_TEXTS:
        try:
            loc = page.get_by_role("button", name=label, exact=False).first
            if await loc.is_visible(timeout=400):
                auth_prompt_visible = True
                break
        except Exception:
            continue

    composer_visible = False
    try:
        composer_visible = await page.locator(PROMPT_SELECTOR).first.is_visible(timeout=400)
    except Exception:
        composer_visible = False

    me = await _fetch_flowmusic_me(page)
    account = _extract_account_label(me)
    signed_in = me is not None and not auth_prompt_visible
    return signed_in, auth_prompt_visible, composer_visible, account


async def login_flowmusic(
    profile_dir: Path,
    *,
    timeout_seconds: float = 300.0,
    settings: Settings | None = None,
) -> FlowMusicAuthResult:
    """Open Chrome foreground at Flow Music and wait until Google sign-in completes."""
    async with _open_flowmusic_page(profile_dir, background=False, settings=settings) as page:
        await page.goto(FLOWMUSIC_URL, wait_until="domcontentloaded", timeout=45_000)
        deadline = time.monotonic() + timeout_seconds
        signed_in = False
        auth_prompt = True
        composer = False
        account: str | None = None
        while time.monotonic() < deadline:
            signed_in, auth_prompt, composer, account = await _inspect_auth_state(page)
            if signed_in:
                await page.wait_for_timeout(1_500)
                break
            await asyncio.sleep(1.0)
        return FlowMusicAuthResult(
            signed_in=signed_in,
            url=page.url,
            profile_dir=str(profile_dir),
            auth_prompt_visible=auth_prompt,
            composer_visible=composer,
            account=account,
            detail=(
                "Signed in to Google Flow Music."
                if signed_in
                else f"Timed out after {timeout_seconds:.0f}s waiting for Flow Music sign-in."
            ),
        )


async def check_flowmusic_status(
    profile_dir: Path,
    *,
    settings: Settings | None = None,
) -> FlowMusicAuthResult:
    """Silently check Flow Music authentication state in the background (0 credits)."""
    async with _open_flowmusic_page(profile_dir, background=True, settings=settings) as page:
        await page.goto(FLOWMUSIC_URL, wait_until="domcontentloaded", timeout=45_000)
        await page.wait_for_timeout(1_500)
        signed_in, auth_prompt, composer, account = await _inspect_auth_state(page)
        return FlowMusicAuthResult(
            signed_in=signed_in,
            url=page.url,
            profile_dir=str(profile_dir),
            auth_prompt_visible=auth_prompt,
            composer_visible=composer,
            account=account,
            detail="Signed in." if signed_in else "Not signed in. Run: gflow music login",
        )


async def probe_flowmusic_ui(
    profile_dir: Path,
    *,
    url: str = FLOWMUSIC_SESSION_URL,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Silently inspect Flow Music interactive controls and observed audio/API URLs."""
    observed_urls: list[dict[str, str]] = []

    def _on_response(resp: Response) -> None:
        ctype = resp.headers.get("content-type", "")
        if (
            ctype.startswith("audio/")
            or _AUDIO_URL_EXT_RE.search(resp.url)
            or "/api/" in resp.url
            or "flowmusic" in resp.url
        ) and len(observed_urls) < 100:
            observed_urls.append(
                {"url": resp.url, "status": str(resp.status), "content_type": ctype}
            )

    async with _open_flowmusic_page(profile_dir, background=True, settings=settings) as page:
        page.on("response", _on_response)
        await page.goto(url, wait_until="domcontentloaded", timeout=45_000)
        await page.wait_for_timeout(2_500)
        signed_in, auth_prompt, composer, account = await _inspect_auth_state(page)
        controls: list[dict[str, Any]] = await page.evaluate(_PROBE_CONTROLS_JS)
        return {
            "url": page.url,
            "signed_in": signed_in,
            "auth_prompt_visible": auth_prompt,
            "composer_visible": composer,
            "account": account,
            "controls_count": len(controls),
            "controls": controls,
            "observed_urls": observed_urls,
        }


def _attach_audio_response_listener(
    page: Page,
    captured: list[_CapturedAudio],
    *,
    preferred_format: str,
) -> None:
    """Attach passive Playwright response capture for generated audio payloads."""

    async def _handle(resp: Response) -> None:
        if resp.status >= 400:
            return
        ctype = resp.headers.get("content-type", "")
        is_audio = ctype.startswith("audio/") or bool(_AUDIO_URL_EXT_RE.search(resp.url))
        if not is_audio:
            return
        try:
            body = await resp.body()
        except Exception:
            return
        if len(body) < _MIN_AUDIO_BYTES:
            return
        ext = infer_audio_extension(
            content_type=ctype, url=resp.url, fallback=preferred_format
        )
        captured.append(
            _CapturedAudio(data=body, ext=ext, url=resp.url, method="network_response")
        )
        log.info("flowmusic.audio_captured", bytes=len(body), ext=ext, method="network_response")

    def _on_response(resp: Response) -> None:
        asyncio.create_task(_handle(resp))

    page.on("response", _on_response)


async def _try_fetch_audio_element(
    page: Page,
    *,
    seen_srcs: set[str],
    preferred_format: str,
) -> _CapturedAudio | None:
    """Check rendered ``<audio>`` elements and fetch any new audio URL via page.request."""
    try:
        srcs: list[str] = await page.evaluate(_AUDIO_SRCS_JS)
    except Exception:
        return None
    for src in srcs:
        if src in seen_srcs or src.startswith("blob:"):
            continue
        seen_srcs.add(src)
        try:
            resp = await page.request.get(src, timeout=30_000)
            if not resp.ok:
                continue
            body = await resp.body()
            if len(body) < _MIN_AUDIO_BYTES:
                continue
            ext = infer_audio_extension(
                content_type=resp.headers.get("content-type"),
                url=src,
                fallback=preferred_format,
            )
            return _CapturedAudio(data=body, ext=ext, url=src, method="audio_element_fetch")
        except Exception:
            continue
    return None


async def _try_ui_download(page: Page, *, preferred_format: str) -> _CapturedAudio | None:
    """Attempt to trigger a UI Download control (preferring ``preferred_format``)."""
    selectors = (
        f'button[aria-label*="{preferred_format}" i]',
        'button[aria-label*="Download" i]',
        "a[download]",
    )
    for sel in selectors:
        loc = page.locator(sel).last
        try:
            if not await loc.is_visible(timeout=300):
                continue
            async with page.expect_download(timeout=15_000) as dl_info:
                await loc.click()
            download = await dl_info.value
            tmp_path = await download.path()
            data = tmp_path.read_bytes()
            if len(data) < _MIN_AUDIO_BYTES:
                continue
            ext = infer_audio_extension(
                url=download.suggested_filename, fallback=preferred_format
            )
            return _CapturedAudio(
                data=data, ext=ext, url=download.url, method="ui_download"
            )
        except Exception:
            continue
    return None


def _run_postprocessing(
    output_path: Path,
    *,
    project_dir: Path | None,
    analyze: bool,
    master: bool,
    target_lufs: float,
) -> tuple[Path | None, Path | None, str | None, Path | None]:
    """Run optional beat analysis, 2-pass mastering, and Stage-1 QA check."""
    analysis_file: Path | None = None
    master_file: Path | None = None
    qa_verdict: str | None = None
    qa_review_path: Path | None = None

    if analyze:
        from gflow_cli.services.audio_analysis import analyze_audio

        out_map = (
            (project_dir.expanduser().resolve() / "audio" / "analysis" / "audio-map.json")
            if project_dir is not None
            else output_path.with_name(f"{output_path.stem}_audio_map.json")
        )
        analyze_audio(output_path, out_file=out_map, project_dir=project_dir)
        analysis_file = out_map

    if master:
        from gflow_cli.services.audio_mastering import master_audio

        master_out = output_path.with_name(f"{output_path.stem}_master.wav")
        master_audio(
            output_path,
            master_out,
            target_lufs=target_lufs,
            true_peak=-1.0,
            overwrite=True,
        )
        master_file = master_out

    if project_dir is not None:
        from gflow_cli.services.project_workspace import record_asset_status
        from gflow_cli.services.quality_gate import run_qa_check

        qa_target = master_file or output_path
        qa_res = run_qa_check(qa_target, kind="audio", project_dir=project_dir)
        qa_verdict = str(qa_res.get("verdict", ""))
        rev = qa_res.get("review_path")
        qa_review_path = Path(rev) if isinstance(rev, str) else None
        record_asset_status(
            project_dir,
            asset_path=output_path,
            status="candidate",
            review_path=qa_review_path,
        )

    return analysis_file, master_file, qa_verdict, qa_review_path


async def generate_flowmusic(
    profile_dir: Path,
    prompt: str,
    *,
    out_file: Path | None = None,
    project_dir: Path | None = None,
    shot: str = "s01",
    ref_audio: Path | None = None,
    ref_image: Path | None = None,
    preferred_format: str = "wav",
    timeout_seconds: float = 240.0,
    analyze: bool = False,
    master: bool = False,
    target_lufs: float = -14.0,
    settings: Settings | None = None,
) -> FlowMusicGenerateResult:
    """Generate a music track on Flow Music silently in the background and save it."""
    if not prompt.strip():
        msg = "Prompt must not be empty."
        raise ValueError(msg)

    captured: list[_CapturedAudio] = []
    seen_srcs: set[str] = set()
    chosen: _CapturedAudio | None = None

    async with _open_flowmusic_page(profile_dir, background=True, settings=settings) as page:
        _attach_audio_response_listener(page, captured, preferred_format=preferred_format)
        await page.goto(FLOWMUSIC_SESSION_URL, wait_until="domcontentloaded", timeout=45_000)
        await page.wait_for_timeout(1_500)

        signed_in, auth_prompt, _composer, _account = await _inspect_auth_state(page)
        if not signed_in or auth_prompt:
            raise FlowMusicAuthError(
                detail="Not signed in to Google Flow Music.",
                remediation_hint="Run: gflow music login",
            )

        if ref_audio is not None:
            await page.locator(UPLOAD_AUDIO_SELECTOR).set_input_files(
                str(ref_audio.expanduser().resolve())
            )
        if ref_image is not None:
            await page.locator(UPLOAD_IMAGE_SELECTOR).set_input_files(
                str(ref_image.expanduser().resolve())
            )

        prompt_box = page.locator(PROMPT_SELECTOR).first
        await prompt_box.wait_for(state="visible", timeout=20_000)
        await prompt_box.fill(prompt)
        await page.locator(SEND_BUTTON_SELECTOR).first.click()

        deadline = time.monotonic() + timeout_seconds
        pref = preferred_format.lstrip(".").lower()
        while time.monotonic() < deadline:
            if any(item.ext == pref for item in captured):
                chosen = next(item for item in captured if item.ext == pref)
                break
            dl_hit = await _try_ui_download(page, preferred_format=pref)
            if dl_hit is not None:
                chosen = dl_hit
                break
            elem_hit = await _try_fetch_audio_element(
                page, seen_srcs=seen_srcs, preferred_format=pref
            )
            if elem_hit is not None:
                chosen = elem_hit
                break
            if captured:
                chosen = captured[-1]
                break
            await asyncio.sleep(1.5)

    if chosen is None:
        raise FlowMusicError(
            detail=f"Timed out after {timeout_seconds:.0f}s waiting for Flow Music audio output.",
            remediation_hint=(
                "Run `gflow music probe` to inspect live page controls and network URLs."
            ),
        )

    dest = resolve_music_output_path(
        ext=chosen.ext,
        out_file=out_file,
        project_dir=project_dir,
        shot=shot,
        settings=settings,
    )
    dest.write_bytes(chosen.data)
    analysis_file, master_file, qa_verdict, qa_review_path = _run_postprocessing(
        dest,
        project_dir=project_dir,
        analyze=analyze,
        master=master,
        target_lufs=target_lufs,
    )
    return FlowMusicGenerateResult(
        output_file=dest,
        format=chosen.ext,
        bytes=len(chosen.data),
        prompt=prompt,
        source_url=chosen.url,
        capture_method=chosen.method,
        analysis_file=analysis_file,
        master_file=master_file,
        qa_verdict=qa_verdict,
        qa_review_path=qa_review_path,
    )
