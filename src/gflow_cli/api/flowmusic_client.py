"""FlowMusicClient — async context-managed Playwright client for flowmusic.app.

Mirrors ``gflow_cli.api.client.FlowApiClient``: acquires a ``ProfileLease``,
launches system Chrome via ``channel_for_profile`` with ``macos_background_launch_guard``
(foreground ``--window-position=0,0`` for login, silent background for all other
operations), passively records ``/__api/*`` JSON payloads (with secret redaction),
and drives both Compose mode and Producer chat mode.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, cast
from urllib.parse import urlparse

import structlog
from playwright.async_api import BrowserContext, Page, Request, Response, async_playwright

from gflow_cli.api._engine import (
    CONTEXT_TEARDOWN_TIMEOUT_S,
    DRIVER_STOP_TIMEOUT_S,
    close_context_bounded,
    run_teardown_step,
)
from gflow_cli.api.flowmusic_dto import (
    FlowMusicAuthResult,
    FlowMusicClip,
    FlowMusicCredits,
    FlowMusicDownloadResult,
    FlowMusicModel,
    FlowMusicResourceItem,
    FlowMusicSession,
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
FLOWMUSIC_LIBRARY_URL = "https://www.flowmusic.app/library"
FLOWMUSIC_CLIP_STORAGE_BASE = "https://storage.googleapis.com/producer-app-public/clips"
_FLOWMUSIC_HOSTS = frozenset({"www.flowmusic.app", "flowmusic.app"})

# Verified live selectors on https://www.flowmusic.app/session?t=true
SETTINGS_MENU_SELECTOR = 'button[aria-label="Settings menu"]'
COMPOSE_TOGGLE_SELECTOR = 'button[aria-label="Toggle compose panel"]'
LYRICS_SELECTOR = 'textarea[aria-label="Lyrics"]'
INSTRUMENTAL_SWITCH_SELECTOR = 'button[role="switch"][aria-label="Toggle instrumental mode"]'
SOUND_DESC_SELECTOR = 'textarea[aria-label="Sound description"]'
ADVANCED_SOUND_SWITCH_SELECTOR = 'button[role="switch"][aria-label="Toggle advanced sound mode"]'
EXPAND_DETAILS_SELECTOR = 'button[aria-label="Expand Details section"]'
GENERATE_BUTTON_SELECTOR = 'button:has-text("Generate")'

PROMPT_SELECTOR = 'textarea[aria-label="Chat message"]'
SEND_BUTTON_SELECTOR = 'button[aria-label="Send message"]'
SETTINGS_BUTTON_SELECTOR = 'button[aria-label="Chat settings"]'
UPLOAD_AUDIO_SELECTOR = 'input[type="file"][aria-label="Upload audio file"]'
UPLOAD_IMAGE_SELECTOR = 'input[type="file"][aria-label="Upload image file"]'

_AUTH_BUTTON_TEXTS: tuple[str, ...] = ("Log in", "Sign up", "Continue with Google")
_MIN_AUDIO_BYTES = 32 * 1024
_AUDIO_URL_EXT_RE = re.compile(r"\.(wav|mp3|m4a|ogg|aac|flac)(?:[?#]|$)", re.IGNORECASE)
_SESSION_ID_RE = re.compile(r"/session/([A-Za-z0-9_-]{8,})")
_SECRET_KEY_RE = re.compile(r"(token|secret|password|cookie|authorization|api_key)", re.IGNORECASE)
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


@dataclass
class CapturedAudioPayload:
    """In-memory audio bytes captured via API clip download or network response."""

    data: bytes
    ext: str
    url: str | None
    method: str
    clip: FlowMusicClip | None = None


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


def extract_session_id(url: str) -> str | None:
    """Extract session UUID/slug from ``https://www.flowmusic.app/session/<id>``."""
    match = _SESSION_ID_RE.search(urlparse(url).path)
    return match.group(1) if match else None


def _redact_payload(value: Any) -> Any:
    """Scrub secret-bearing keys from cached ``/__api/*`` JSON payloads."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in cast("dict[str, Any]", value).items():
            if _SECRET_KEY_RE.search(k) and k != "tokens_remaining":
                out[k] = "[REDACTED]"
            else:
                out[k] = _redact_payload(v)
        return out
    if isinstance(value, list):
        return [_redact_payload(item) for item in cast("list[Any]", value)]
    return value


def _extract_handle_from_settings_text(text: str) -> str | None:
    """Extract handle line (e.g. ``PlanetDomain78``) from ``Settings menu`` button text."""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line and line.upper() not in {"MEMBER", "PRO", "ULTRA", "FREE", "SETTINGS"}:
            return line
    return None


def _extract_items_list(payload: Any, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in cast("list[Any]", payload) if isinstance(x, dict)]
    if isinstance(payload, dict):
        d = cast("dict[str, Any]", payload)
        for key in keys:
            val = d.get(key)
            if isinstance(val, list):
                return [x for x in cast("list[Any]", val) if isinstance(x, dict)]
        data = d.get("data")
        if isinstance(data, list):
            return [x for x in cast("list[Any]", data) if isinstance(x, dict)]
        if isinstance(data, dict):
            return _extract_items_list(data, keys)
    return []


class FlowMusicClient:
    """Async context-managed client for Google Flow Music (flowmusic.app)."""

    def __init__(
        self,
        profile_dir: Path,
        *,
        background: bool = True,
        settings: Settings | None = None,
    ) -> None:
        self.profile_dir = profile_dir.expanduser().resolve()
        self.background = background
        self.settings = settings or get_settings()
        self._lease: ProfileLease | None = None
        self._pw: Any = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._api_cache: dict[str, Any] = {}
        self._observed_urls: list[dict[str, str]] = []
        self._captured_audio: list[CapturedAudioPayload] = []
        self._pending_tasks: set[asyncio.Task[None]] = set()
        self._listeners_attached: bool = False
        # In-memory only: never logged, never persisted, cleared in _teardown().
        self._session_auth_header: str | None = None

    @property
    def page(self) -> Page:
        assert self._page is not None, "FlowMusicClient must be entered with `async with`"
        return self._page

    async def __aenter__(self) -> Self:
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self._lease = await ProfileLease(self.profile_dir).aacquire()
        try:
            channel = channel_for_profile(self.profile_dir)
            ensure_profile_engine_compatible(self.profile_dir, channel)
            self._pw = await async_playwright().start()
            args = [
                "--password-store=basic",
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                *disable_features_args(),
            ]
            if self.background:
                args.extend(window_position_args(self.settings.browser_window_position))
            else:
                args.append("--window-position=0,0")
            launch_kwargs: dict[str, Any] = {
                "user_data_dir": str(self.profile_dir),
                "headless": False,
                "viewport": {"width": 1280, "height": 800},
                "locale": "en-US",
                "channel": channel,
                "ignore_default_args": ["--enable-automation", "--no-sandbox"],
                "args": args,
                **browser_proxy_kwargs(self.settings.browser_proxy),
            }
            async with macos_background_launch_guard(
                headless=False,
                enabled=self.background and self.settings.macos_background_launch,
            ):
                ctx: BrowserContext = await self._pw.chromium.launch_persistent_context(
                    **launch_kwargs
                )
            self._context = ctx
            await ctx.add_init_script(
                "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"
            )
            self._page = ctx.pages[0] if ctx.pages else await ctx.new_page()
            self._attach_listeners(self._page)
            return self
        except BaseException:
            await self._teardown()
            raise

    async def __aexit__(self, *exc: object) -> None:
        await self._teardown()

    async def _teardown(self) -> None:
        for task in list(self._pending_tasks):
            task.cancel()
        self._pending_tasks.clear()
        if self._context is not None:
            await run_teardown_step(
                close_context_bounded(self._context, owner="flowmusic"),
                timeout=CONTEXT_TEARDOWN_TIMEOUT_S,
                owner="flowmusic",
                step="context_close",
            )
            self._context = None
        if self._pw is not None:
            await run_teardown_step(
                self._pw.stop(),
                timeout=DRIVER_STOP_TIMEOUT_S,
                owner="flowmusic",
                step="driver_stop",
            )
            self._pw = None
        self._page = None
        self._listeners_attached = False
        self._session_auth_header = None
        if self._lease is not None:
            self._lease.release()
            self._lease = None

    def _attach_listeners(self, page: Page) -> None:
        async def _capture(resp: Response) -> None:
            url = resp.url
            parsed = urlparse(url)
            ctype = resp.headers.get("content-type", "")
            if (
                "/__api/" in parsed.path
                or ctype.startswith("audio/")
                or _AUDIO_URL_EXT_RE.search(url)
            ) and len(self._observed_urls) < 150:
                self._observed_urls.append(
                    {"url": url, "status": str(resp.status), "content_type": ctype}
                )
            if resp.status >= 400:
                return
            if "/__api/" in parsed.path and "json" in ctype.lower():
                try:
                    payload = await resp.json()
                    if payload is not None:
                        self._api_cache[parsed.path] = _redact_payload(payload)
                except Exception:
                    return
            elif ctype.startswith("audio/") or bool(_AUDIO_URL_EXT_RE.search(url)):
                try:
                    body = await resp.body()
                except Exception:
                    return
                if len(body) >= _MIN_AUDIO_BYTES:
                    ext = infer_audio_extension(content_type=ctype, url=url)
                    self._captured_audio.append(
                        CapturedAudioPayload(data=body, ext=ext, url=url, method="network_response")
                    )

        def _on_request(req: Request) -> None:
            if "/__api/" in urlparse(req.url).path:
                auth_hdr = req.headers.get("authorization")
                if auth_hdr:
                    self._session_auth_header = auth_hdr

        def _on_response(resp: Response) -> None:
            task = asyncio.create_task(_capture(resp))
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)

        page.on("request", _on_request)
        page.on("response", _on_response)
        self._listeners_attached = True

    async def _goto_settled(self, url: str, *, settle_ms: int = 2_000) -> None:
        await self.page.goto(url, wait_until="domcontentloaded", timeout=45_000)
        await self.page.wait_for_timeout(settle_ms)

    async def _wait_for_api_cache(self, path: str, *, timeout_s: float = 3.0) -> Any:
        if not self._listeners_attached:
            return self._api_cache.get(path)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if path in self._api_cache or self._session_auth_header is not None:
                return self._api_cache.get(path)
            await asyncio.sleep(0.15)
        return self._api_cache.get(path)

    async def _in_page_get_json(self, api_path: str) -> Any:
        """Fetch an ``/__api/*`` endpoint from the browser context, falling back to cache."""
        parsed_path = urlparse(api_path).path
        cached = self._api_cache.get(parsed_path)
        if cached is not None:
            return cached
        if self._session_auth_header is None and self._listeners_attached:
            cached = await self._wait_for_api_cache(parsed_path, timeout_s=1.5)
            if cached is not None:
                return cached
        try:
            if self._session_auth_header is not None:
                result: Any = await self.page.evaluate(
                    """async ([u, auth]) => {
                      const r = await fetch(u, {
                        credentials: 'include',
                        headers: { Authorization: auth },
                      });
                      if (!r.ok) return null;
                      return await r.json();
                    }""",
                    [api_path, self._session_auth_header],
                )
            else:
                result = await self.page.evaluate(
                    """async (u) => {
                      const r = await fetch(u, { credentials: 'include' });
                      if (!r.ok) return null;
                      return await r.json();
                    }""",
                    api_path,
                )
            if result is not None:
                redacted = _redact_payload(result)
                self._api_cache[parsed_path] = redacted
                return redacted
        except Exception:
            pass
        return self._api_cache.get(parsed_path)

    async def inspect_auth(self) -> FlowMusicAuthResult:
        """Inspect signed-in status using host guard, Settings menu, and ``/__api/billing/*``."""
        url = self.page.url
        host = (urlparse(url).hostname or "").lower()
        if host not in _FLOWMUSIC_HOSTS:
            return FlowMusicAuthResult(
                signed_in=False,
                url=url,
                profile_dir=str(self.profile_dir),
                auth_prompt_visible=False,
                composer_visible=False,
                account=None,
                detail="Not on flowmusic.app (OAuth or blank page).",
            )

        auth_prompt_visible = False
        for label in _AUTH_BUTTON_TEXTS:
            try:
                if await self.page.get_by_role("button", name=label, exact=False).first.is_visible(
                    timeout=400
                ):
                    auth_prompt_visible = True
                    break
            except Exception:
                continue

        settings_visible = False
        account: str | None = None
        try:
            settings_loc = self.page.locator(SETTINGS_MENU_SELECTOR).first
            if await settings_loc.is_visible(timeout=500):
                settings_visible = True
                raw_text = await settings_loc.inner_text(timeout=500)
                account = _extract_handle_from_settings_text(raw_text)
        except Exception:
            pass

        composer_visible = False
        try:
            composer_visible = await self.page.locator(PROMPT_SELECTOR).first.is_visible(
                timeout=400
            ) or await self.page.locator(SOUND_DESC_SELECTOR).first.is_visible(timeout=300)
        except Exception:
            pass

        api_auth_ok = (
            "/__api/billing/credits" in self._api_cache
            or "/__api/billing/subscription" in self._api_cache
        )
        signed_in = (settings_visible or api_auth_ok) and not auth_prompt_visible
        return FlowMusicAuthResult(
            signed_in=signed_in,
            url=url,
            profile_dir=str(self.profile_dir),
            auth_prompt_visible=auth_prompt_visible,
            composer_visible=composer_visible,
            account=account,
            detail="Signed in." if signed_in else "Not signed in. Run: gflow music login",
        )

    async def ensure_authenticated(
        self, *, url: str = FLOWMUSIC_SESSION_URL
    ) -> FlowMusicAuthResult:
        """Navigate to ``url`` and raise ``FlowMusicAuthError`` if signed out."""
        await self._goto_settled(url)
        state = await self.inspect_auth()
        if not state.signed_in:
            raise FlowMusicAuthError(
                detail="Not signed in to Google Flow Music.",
                remediation_hint="Run: gflow music login",
            )
        return state

    async def login(self, *, timeout_seconds: float = 300.0) -> FlowMusicAuthResult:
        """Open Flow Music foreground and poll until canonical signed-in signal appears."""
        await self.page.goto(FLOWMUSIC_URL, wait_until="domcontentloaded", timeout=45_000)
        deadline = time.monotonic() + timeout_seconds
        state = await self.inspect_auth()
        while time.monotonic() < deadline:
            state = await self.inspect_auth()
            if state.signed_in:
                await self.page.wait_for_timeout(1_500)
                return state
            await asyncio.sleep(1.0)
        return FlowMusicAuthResult(
            signed_in=False,
            url=self.page.url,
            profile_dir=str(self.profile_dir),
            auth_prompt_visible=state.auth_prompt_visible,
            composer_visible=state.composer_visible,
            account=state.account,
            detail=f"Timed out after {timeout_seconds:.0f}s waiting for Flow Music sign-in.",
        )

    async def get_credits(self) -> FlowMusicCredits:
        """Return Flow Music credits and subscription status."""
        await self.ensure_authenticated(url=FLOWMUSIC_URL)
        credits_raw = await self._in_page_get_json("/__api/billing/credits")
        sub_raw = await self._in_page_get_json("/__api/billing/subscription")
        return FlowMusicCredits.from_api(
            cast("dict[str, Any]", credits_raw) if isinstance(credits_raw, dict) else None,
            cast("dict[str, Any]", sub_raw) if isinstance(sub_raw, dict) else None,
        )

    async def list_models(self) -> list[FlowMusicModel]:
        """Return available Flow Music models from ``/__api/models``."""
        await self.ensure_authenticated(url=FLOWMUSIC_SESSION_URL)
        await self._wait_for_api_cache("/__api/models", timeout_s=3.0)
        raw = await self._in_page_get_json("/__api/models")
        return [FlowMusicModel.from_api(item) for item in _extract_items_list(raw, ("models",))]

    async def list_clips(self, *, limit: int = 20, offset: int = 0) -> list[FlowMusicClip]:
        """Return generated clips from ``/__api/clips/auth-user``."""
        await self.ensure_authenticated(url=FLOWMUSIC_LIBRARY_URL)
        await self._wait_for_api_cache("/__api/clips/auth-user", timeout_s=3.0)
        api_url = (
            f"/__api/clips/auth-user?limit={limit}&offset={offset}"
            "&filter=generations&include_disliked=false"
        )
        raw = await self._in_page_get_json(api_url)
        return [FlowMusicClip.from_api(item) for item in _extract_items_list(raw, ("clips",))]

    async def list_sessions(self, *, limit: int = 30, offset: int = 0) -> list[FlowMusicSession]:
        """Return Producer chat sessions from ``/__api/conversations``."""
        await self.ensure_authenticated(url=FLOWMUSIC_URL)
        raw = await self._in_page_get_json(f"/__api/conversations?limit={limit}&offset={offset}")
        items = _extract_items_list(raw, ("conversations", "sessions", "items"))
        return [FlowMusicSession.from_api(item) for item in items]

    async def list_projects(
        self, *, limit: int = 20, offset: int = 0
    ) -> list[FlowMusicResourceItem]:
        """Return Flow Music cloud projects from ``/__api/projects``."""
        await self.ensure_authenticated(url=FLOWMUSIC_URL)
        raw = await self._in_page_get_json(f"/__api/projects?offset={offset}&limit={limit}")
        items = _extract_items_list(raw, ("projects", "items"))
        return [FlowMusicResourceItem.from_api(i, kind="project") for i in items]

    async def list_playlists(
        self, *, limit: int = 20, offset: int = 0
    ) -> list[FlowMusicResourceItem]:
        """Return user playlists captured from ``/__api/users/<uid>/playlists``."""
        await self.ensure_authenticated(url=FLOWMUSIC_URL)
        raw = next(
            (v for k, v in self._api_cache.items() if k.endswith("/playlists")),
            None,
        )
        if raw is None:
            raw = await self._in_page_get_json(
                f"/__api/playlists?favorites=false&public=false&offset={offset}&limit={limit}"
            )
        items = _extract_items_list(raw, ("playlists", "items"))
        return [FlowMusicResourceItem.from_api(i, kind="playlist") for i in items]

    async def list_spaces(self, *, limit: int = 20, offset: int = 0) -> list[FlowMusicResourceItem]:
        """Return Flow Music spaces from ``/__api/spaces/get``."""
        await self.ensure_authenticated(url=FLOWMUSIC_URL)
        raw = await self._in_page_get_json(f"/__api/spaces/get?offset={offset}&limit={limit}")
        items = _extract_items_list(raw, ("spaces", "items"))
        return [FlowMusicResourceItem.from_api(i, kind="space") for i in items]

    async def list_music_videos(
        self, *, limit: int = 20, offset: int = 0
    ) -> list[FlowMusicResourceItem]:
        """Return Flow Music videos from ``/__api/music-videos/get``."""
        await self.ensure_authenticated(url=FLOWMUSIC_URL)
        raw = await self._in_page_get_json(f"/__api/music-videos/get?offset={offset}&limit={limit}")
        items = _extract_items_list(raw, ("music_videos", "videos", "items"))
        return [FlowMusicResourceItem.from_api(i, kind="music_video") for i in items]

    async def probe_ui(self, *, url: str = FLOWMUSIC_SESSION_URL) -> dict[str, Any]:
        """Silently inspect Flow Music UI controls and observed API/audio URLs."""
        await self._goto_settled(url, settle_ms=2_500)
        state = await self.inspect_auth()
        controls: list[dict[str, Any]] = await self.page.evaluate(_PROBE_CONTROLS_JS)
        return {
            "url": self.page.url,
            "signed_in": state.signed_in,
            "auth_prompt_visible": state.auth_prompt_visible,
            "composer_visible": state.composer_visible,
            "account": state.account,
            "cached_api_paths": sorted(self._api_cache.keys()),
            "controls_count": len(controls),
            "controls": controls,
            "observed_urls": list(self._observed_urls),
        }

    async def _fetch_bytes(self, url: str) -> tuple[bytes, str | None]:
        resp = await self.page.request.get(url, timeout=60_000)
        if not resp.ok:
            raise FlowMusicError(detail=f"Failed to download {url} (HTTP {resp.status}).")
        return await resp.body(), resp.headers.get("content-type")

    async def _wait_for_new_clip(
        self,
        *,
        existing_ids: set[str],
        preferred_format: str,
        timeout_seconds: float,
    ) -> CapturedAudioPayload:
        """Poll ``/__api/clips/auth-user`` + passive audio capture for a new generated clip."""
        deadline = time.monotonic() + timeout_seconds
        seen_srcs: set[str] = set()
        first_seen_at: dict[str, float] = {}
        pref = preferred_format.lstrip(".").lower()
        while time.monotonic() < deadline:
            # 1. Check cached/fresh /__api/clips/auth-user for a new clip with audio URL.
            self._api_cache.pop("/__api/clips/auth-user", None)
            raw = await self._in_page_get_json(
                "/__api/clips/auth-user?limit=10&offset=0&filter=generations&include_disliked=false"
            )
            now = time.monotonic()
            for item in _extract_items_list(raw, ("clips",)):
                clip = FlowMusicClip.from_api(item)
                if clip.id and clip.id not in existing_ids:
                    first_seen_at.setdefault(clip.id, now)
                    # When WAV is requested, wait up to 15s for wav_url to finish rendering
                    # before falling back to audio_url (.m4a).
                    if (
                        pref == "wav"
                        and not clip.wav_url
                        and clip.audio_url
                        and (now - first_seen_at[clip.id]) < 15.0
                    ):
                        continue
                    dl_url, ext = clip.preferred_download_url(pref)
                    if dl_url:
                        body, ctype = await self._fetch_bytes(dl_url)
                        if len(body) >= _MIN_AUDIO_BYTES:
                            final_ext = infer_audio_extension(
                                content_type=ctype, url=dl_url, fallback=ext
                            )
                            return CapturedAudioPayload(
                                data=body,
                                ext=final_ext,
                                url=dl_url,
                                method="api_clip_download",
                                clip=clip,
                            )

            # 2. Check passive network audio capture.
            if any(c.ext == pref for c in self._captured_audio):
                return next(c for c in self._captured_audio if c.ext == pref)
            if self._captured_audio:
                return self._captured_audio[-1]

            # 3. Fallback: rendered <audio src> element.
            try:
                srcs: list[str] = await self.page.evaluate(_AUDIO_SRCS_JS)
            except Exception:
                srcs = []
            for src in srcs:
                if src in seen_srcs or src.startswith("blob:"):
                    continue
                seen_srcs.add(src)
                try:
                    body, ctype = await self._fetch_bytes(src)
                    if len(body) >= _MIN_AUDIO_BYTES:
                        ext = infer_audio_extension(content_type=ctype, url=src, fallback=pref)
                        return CapturedAudioPayload(
                            data=body, ext=ext, url=src, method="audio_element_fetch"
                        )
                except Exception:
                    continue
            await asyncio.sleep(2.0)

        raise FlowMusicError(
            detail=f"Timed out after {timeout_seconds:.0f}s waiting for Flow Music clip output.",
            remediation_hint="Run `gflow music list` to check if the clip finished on the server.",
        )

    async def _set_switch(self, selector: str, desired: bool) -> None:
        loc = self.page.locator(selector).first
        try:
            if not await loc.is_visible(timeout=1_500):
                return
            checked = (await loc.get_attribute("aria-checked") or "").lower() == "true"
            if checked != desired:
                await loc.click()
                await self.page.wait_for_timeout(200)
        except Exception:
            return

    async def _fetch_current_clip_ids(self, *, limit: int = 20) -> set[str]:
        """Fetch existing clip IDs via in-page ``/__api/clips/auth-user`` without navigating."""
        api_url = (
            f"/__api/clips/auth-user?limit={limit}&offset=0"
            "&filter=generations&include_disliked=false"
        )
        raw = await self._in_page_get_json(api_url)
        return {
            c.id
            for c in (FlowMusicClip.from_api(i) for i in _extract_items_list(raw, ("clips",)))
            if c.id
        }

    async def compose(
        self,
        sound_description: str,
        *,
        lyrics: str | None = None,
        instrumental: bool = False,
        advanced_sound: bool = False,
        model: str | None = None,
        preferred_format: str = "wav",
        timeout_seconds: float = 240.0,
    ) -> tuple[CapturedAudioPayload, str | None]:
        """Drive Compose mode on ``/session?t=true`` and return the downloaded clip payload."""
        await self.ensure_authenticated(url=FLOWMUSIC_SESSION_URL)
        existing_ids = await self._fetch_current_clip_ids(limit=20)
        self._captured_audio.clear()

        sound_box = self.page.locator(SOUND_DESC_SELECTOR).first
        if not await sound_box.is_visible(timeout=1_500):
            toggle = self.page.locator(COMPOSE_TOGGLE_SELECTOR).first
            if await toggle.is_visible(timeout=2_000):
                await toggle.click()
                await self.page.wait_for_timeout(400)

        await self._set_switch(INSTRUMENTAL_SWITCH_SELECTOR, instrumental)
        if advanced_sound:
            await self._set_switch(ADVANCED_SOUND_SWITCH_SELECTOR, True)
        if not instrumental and lyrics:
            lyrics_box = self.page.locator(LYRICS_SELECTOR).first
            if await lyrics_box.is_visible(timeout=1_500):
                await lyrics_box.fill(lyrics)

        await sound_box.wait_for(state="visible", timeout=15_000)
        await sound_box.fill(sound_description)

        if model:
            try:
                model_btn = self.page.get_by_role("button", name=model, exact=False).first
                if await model_btn.is_visible(timeout=1_000):
                    await model_btn.click()
            except Exception:
                pass

        await self.page.locator(GENERATE_BUTTON_SELECTOR).first.click()
        payload = await self._wait_for_new_clip(
            existing_ids=existing_ids,
            preferred_format=preferred_format,
            timeout_seconds=timeout_seconds,
        )
        return payload, extract_session_id(self.page.url)

    async def chat(
        self,
        message: str,
        *,
        session_id: str | None = None,
        ref_audio: Path | None = None,
        ref_image: Path | None = None,
        wait_clip: bool = True,
        preferred_format: str = "wav",
        timeout_seconds: float = 240.0,
    ) -> tuple[str | None, str, CapturedAudioPayload | None]:
        """Send a Producer chat message (new or resumed session) and optionally wait for a clip."""
        target_url = (
            f"{FLOWMUSIC_URL.rstrip('/')}/session/{session_id}"
            if session_id
            else FLOWMUSIC_SESSION_URL
        )
        await self.ensure_authenticated(url=target_url)
        existing_ids: set[str] = (
            await self._fetch_current_clip_ids(limit=20) if wait_clip else set()
        )
        self._captured_audio.clear()

        if ref_audio is not None:
            await self.page.locator(UPLOAD_AUDIO_SELECTOR).set_input_files(
                str(ref_audio.expanduser().resolve())
            )
        if ref_image is not None:
            await self.page.locator(UPLOAD_IMAGE_SELECTOR).set_input_files(
                str(ref_image.expanduser().resolve())
            )

        prompt_box = self.page.locator(PROMPT_SELECTOR).first
        await prompt_box.wait_for(state="visible", timeout=15_000)
        await prompt_box.fill(message)
        await self.page.locator(SEND_BUTTON_SELECTOR).first.click()

        payload: CapturedAudioPayload | None = None
        if wait_clip:
            payload = await self._wait_for_new_clip(
                existing_ids=existing_ids,
                preferred_format=preferred_format,
                timeout_seconds=timeout_seconds,
            )
        else:
            await self.page.wait_for_timeout(3_000)
        return extract_session_id(self.page.url) or session_id, self.page.url, payload

    async def download_clip(
        self,
        clip_id: str,
        *,
        out_file: Path,
        preferred_format: str = "wav",
        include_cover: bool = False,
        include_lyrics_timing: bool = False,
    ) -> FlowMusicDownloadResult:
        """Download an existing clip by ID (``wav``/``m4a``) plus optional cover & lyrics timing."""
        clips = await self.list_clips(limit=50, offset=0)
        clip = next((c for c in clips if c.id == clip_id), None)
        pref = preferred_format.lstrip(".").lower()
        if clip is not None:
            dl_url, ext = clip.preferred_download_url(pref)
        else:
            ext = pref or "wav"
            dl_url = f"{FLOWMUSIC_CLIP_STORAGE_BASE}/{clip_id}.{ext}"

        if not dl_url:
            raise FlowMusicError(detail=f"Clip {clip_id!r} has no downloadable audio URL.")

        body, ctype = await self._fetch_bytes(dl_url)
        final_ext = infer_audio_extension(content_type=ctype, url=dl_url, fallback=ext)
        dest = out_file.expanduser().resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body)

        cover_file: Path | None = None
        if include_cover and clip is not None and clip.image_url:
            cover_bytes, _ = await self._fetch_bytes(clip.image_url)
            cover_file = dest.with_suffix(".jpg")
            cover_file.write_bytes(cover_bytes)

        lyrics_file: Path | None = None
        if include_lyrics_timing and clip is not None and clip.lyrics_timing is not None:
            lyrics_file = dest.with_name(f"{dest.stem}_lyrics_timing.json")
            lyrics_file.write_text(
                json.dumps(clip.lyrics_timing, indent=2) + "\n", encoding="utf-8"
            )

        return FlowMusicDownloadResult(
            clip_id=clip_id,
            output_file=dest,
            format=final_ext,
            bytes=len(body),
            source_url=dl_url,
            title=clip.title if clip is not None else "",
            cover_file=cover_file,
            lyrics_timing_file=lyrics_file,
        )
