"""Wave 2.5 Production Gate: Google Flow Music Full-Parity CLI, Client, DTO & MCP Tests.

Covers:
- Typed DTO parsing & ``.to_dict()`` (Model, Credits, Clip, Session, ResourceItem, AuthResult)
- Auth detection contract (Settings menu / billing API cache / Log in / OAuth host guard)
- Foreground login ``--window-position=0,0`` & background guard exemption
- ``FlowMusicClient`` mocked Playwright tests (compose, chat, download_clip, library readers)
- CLI ``gflow music`` surface, ``generate`` bridge routing, ``-o json|table|id``, and workspace
  post-processing (``--analyze``, ``--master``, Stage-1 QA)
- MCP twin tools & ``--no-spend`` registration gating
"""

from __future__ import annotations

import io
import json
import wave
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from click.testing import CliRunner

from gflow_cli.api import flowmusic_client as fm_client_mod
from gflow_cli.api.flowmusic_client import (
    FLOWMUSIC_SESSION_URL,
    FLOWMUSIC_URL,
    INSTRUMENTAL_SWITCH_SELECTOR,
    LYRICS_SELECTOR,
    PROMPT_SELECTOR,
    SETTINGS_MENU_SELECTOR,
    SOUND_DESC_SELECTOR,
    UPLOAD_AUDIO_SELECTOR,
    UPLOAD_IMAGE_SELECTOR,
    CapturedAudioPayload,
    FlowMusicAuthError,
    FlowMusicClient,
)
from gflow_cli.api.flowmusic_dto import (
    FlowMusicAuthResult,
    FlowMusicChatResult,
    FlowMusicClip,
    FlowMusicCredits,
    FlowMusicGenerateResult,
    FlowMusicModel,
    FlowMusicResourceItem,
    FlowMusicSession,
)
from gflow_cli.cli import main
from gflow_cli.services import flowmusic as fm_service

# ---------------------------------------------------------------------------
# Helpers & Fake Playwright Page
# ---------------------------------------------------------------------------


def _synthetic_wav_bytes(*, duration_sec: float = 4.0, sr: int = 44100) -> bytes:
    """Generate a valid >32 KB stereo WAV payload in memory."""
    n = int(sr * duration_sec)
    t = np.arange(n, dtype=np.float32) / sr
    sig = 0.25 * np.sin(2.0 * np.pi * 220.0 * t)
    pcm = (sig * 32767.0).astype(np.int16)
    stereo = np.column_stack([pcm, pcm]).ravel()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(stereo.tobytes())
    return buf.getvalue()


class _FakeLocator:
    def __init__(
        self,
        *,
        visible: bool = False,
        text: str = "",
        checked: bool = False,
        recorded: dict[str, Any] | None = None,
        name: str = "",
    ) -> None:
        self._visible = visible
        self._text = text
        self._checked = checked
        self._recorded = recorded if recorded is not None else {}
        self._name = name

    @property
    def first(self) -> _FakeLocator:
        return self

    async def is_visible(self, *, timeout: int = 1000) -> bool:
        return self._visible

    async def inner_text(self, *, timeout: int = 1000) -> str:
        return self._text

    async def get_attribute(self, attr: str) -> str | None:
        if attr == "aria-checked":
            return "true" if self._checked else "false"
        return None

    async def wait_for(self, *, state: str = "visible", timeout: int = 5000) -> None:
        return None

    async def click(self) -> None:
        clicks: list[str] = self._recorded.setdefault("clicks", [])
        clicks.append(self._name)
        self._checked = not self._checked

    async def fill(self, value: str) -> None:
        fills: dict[str, str] = self._recorded.setdefault("fills", {})
        fills[self._name] = value

    async def set_input_files(self, path: str) -> None:
        uploads: dict[str, str] = self._recorded.setdefault("uploads", {})
        uploads[self._name] = path


class _FakeRequestResponse:
    def __init__(self, data: bytes, *, content_type: str = "audio/wav", status: int = 200) -> None:
        self._data = data
        self.status = status
        self.ok = 200 <= status < 300
        self.headers = {"content-type": content_type}

    async def body(self) -> bytes:
        return self._data


class _FakePage:
    def __init__(
        self,
        *,
        url: str = FLOWMUSIC_SESSION_URL,
        locators: dict[str, _FakeLocator] | None = None,
        role_buttons: dict[str, _FakeLocator] | None = None,
        api_responses: dict[str, Any] | None = None,
        download_bytes: dict[str, tuple[bytes, str]] | None = None,
    ) -> None:
        self.url = url
        self._locators = locators or {}
        self._role_buttons = role_buttons or {}
        self._api_responses = api_responses or {}
        self._download_bytes = download_bytes or {}
        self.goto_urls: list[str] = []
        self.recorded: dict[str, Any] = {}
        self.request = MagicMock()
        self.request.get = AsyncMock(side_effect=self._handle_request_get)

    async def _handle_request_get(self, url: str, **_: Any) -> _FakeRequestResponse:
        if url in self._download_bytes:
            data, ctype = self._download_bytes[url]
            return _FakeRequestResponse(data, content_type=ctype)
        return _FakeRequestResponse(b"", status=404)

    def on(self, _event: str, _cb: Any) -> None:
        return None

    async def goto(self, url: str, **_: Any) -> None:
        self.goto_urls.append(url)
        self.url = url

    async def wait_for_timeout(self, _ms: int) -> None:
        return None

    def locator(self, selector: str) -> _FakeLocator:
        if selector in self._locators:
            return self._locators[selector]
        loc = _FakeLocator(visible=True, recorded=self.recorded, name=selector)
        self._locators[selector] = loc
        return loc

    def get_by_role(self, _role: str, *, name: str, exact: bool = False) -> _FakeLocator:
        return self._role_buttons.get(name, _FakeLocator(visible=False, name=name))

    async def evaluate(self, _js: str, arg: Any = None) -> Any:
        if isinstance(arg, str):
            for prefix, payload in self._api_responses.items():
                if arg.startswith(prefix):
                    if callable(payload):
                        return payload()
                    return payload
        return []


def _attached_client(tmp_path: Path, page: _FakePage) -> FlowMusicClient:
    client = FlowMusicClient(tmp_path / "prof", background=True)
    client._page = page  # type: ignore[assignment]
    return client


# ---------------------------------------------------------------------------
# 1. Typed DTO Parsing & .to_dict()
# ---------------------------------------------------------------------------


def test_flowmusic_dtos_parse_and_serialize() -> None:
    model = FlowMusicModel.from_api(
        {
            "public_name": "Lyria 3.5 Pro",
            "version": "3.5",
            "description": "Studio vocals & stems",
            "is_default": True,
            "max_audio_upload_duration_s": 120,
            "max_audio_upload_bytes": 10_485_760,
        }
    )
    assert model.to_dict()["public_name"] == "Lyria 3.5 Pro"
    assert model.to_dict()["is_default"] is True

    credits_obj = FlowMusicCredits.from_api(
        {"data": {"credits_remaining": 420.5, "tokens_remaining": 1200}},
        {"data": {"plan": "ultra", "status": "active"}},
    )
    assert credits_obj.to_dict() == {
        "credits_remaining": 420.5,
        "tokens_remaining": 1200.0,
        "plan": "ultra",
        "subscription_status": "active",
    }

    clip = FlowMusicClip.from_api(
        {
            "id": "clip-1",
            "title": "Espresso Groove",
            "audio_url": "https://cdn.example/clip-1.m4a",
            "wav_url": "https://cdn.example/clip-1.wav",
            "duration": 32.5,
            "has_vocals": False,
            "lyrics_timing": [{"word": "hi", "t": 0.5}],
        }
    )
    assert clip.preferred_download_url("wav") == ("https://cdn.example/clip-1.wav", "wav")
    assert clip.preferred_download_url("m4a") == ("https://cdn.example/clip-1.m4a", "m4a")
    assert clip.to_dict()["has_lyrics_timing"] is True

    session = FlowMusicSession.from_api({"id": "sess-99", "title": "Promo Track"})
    assert session.to_dict()["id"] == "sess-99"

    resource = FlowMusicResourceItem.from_api({"id": "pl-1", "name": "Favorites"}, kind="playlist")
    assert resource.to_dict()["kind"] == "playlist"
    assert resource.to_dict()["title"] == "Favorites"

    auth = FlowMusicAuthResult(
        signed_in=True,
        url=FLOWMUSIC_URL,
        profile_dir="/tmp/p",
        auth_prompt_visible=False,
        composer_visible=True,
        account="PlanetDomain78",
    )
    assert auth.to_dict()["status"] == "signed_in"
    assert auth.to_dict()["account"] == "PlanetDomain78"


# ---------------------------------------------------------------------------
# 2. Auth Detection Contract & Foreground Login Policy
# ---------------------------------------------------------------------------


async def test_inspect_auth_signed_in_via_settings_menu_and_api_cache(tmp_path: Path) -> None:
    page = _FakePage(
        url=FLOWMUSIC_URL,
        locators={
            SETTINGS_MENU_SELECTOR: _FakeLocator(visible=True, text="MEMBER\nPlanetDomain78"),
            PROMPT_SELECTOR: _FakeLocator(visible=True),
        },
    )
    client = _attached_client(tmp_path, page)
    state = await client.inspect_auth()
    assert state.signed_in is True
    assert state.account == "PlanetDomain78"
    assert state.auth_prompt_visible is False

    # Also signed in via captured /__api/billing/credits when Settings menu is not visible
    page2 = _FakePage(
        url=FLOWMUSIC_URL,
        locators={SETTINGS_MENU_SELECTOR: _FakeLocator(visible=False)},
    )
    client2 = _attached_client(tmp_path, page2)
    client2._api_cache["/__api/billing/credits"] = {"credits_remaining": 100}
    state2 = await client2.inspect_auth()
    assert state2.signed_in is True


async def test_inspect_auth_signed_out_on_login_button_or_oauth_redirect(tmp_path: Path) -> None:
    # 1. "Log in" visible on flowmusic.app -> signed out even if API cache has stale data
    page_login = _FakePage(
        url=FLOWMUSIC_URL,
        role_buttons={"Log in": _FakeLocator(visible=True, name="Log in")},
    )
    client_login = _attached_client(tmp_path, page_login)
    client_login._api_cache["/__api/billing/credits"] = {"credits_remaining": 10}
    st_login = await client_login.inspect_auth()
    assert st_login.signed_in is False
    assert st_login.auth_prompt_visible is True

    # 2. Redirected to accounts.google.com -> strictly signed out (zero false positives)
    page_oauth = _FakePage(
        url="https://accounts.google.com/v3/signin/identifier",
        locators={SETTINGS_MENU_SELECTOR: _FakeLocator(visible=True, text="User")},
    )
    client_oauth = _attached_client(tmp_path, page_oauth)
    st_oauth = await client_oauth.inspect_auth()
    assert st_oauth.signed_in is False
    with pytest.raises(FlowMusicAuthError):
        await client_oauth.ensure_authenticated(url=page_oauth.url)


async def test_foreground_login_uses_window_position_0_0_and_disables_bg_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured_launch: dict[str, Any] = {}
    guard_enabled_flags: list[bool] = []

    class _FakeCtx:
        pages: list[Any] = []

        async def add_init_script(self, _script: str) -> None:
            return None

        async def new_page(self) -> _FakePage:
            return _FakePage(url=FLOWMUSIC_URL)

        async def close(self) -> None:
            return None

    class _FakeChromium:
        async def launch_persistent_context(self, **kwargs: Any) -> _FakeCtx:
            captured_launch.update(kwargs)
            return _FakeCtx()

    class _FakePw:
        chromium = _FakeChromium()

        async def start(self) -> _FakePw:
            return self

        async def stop(self) -> None:
            return None

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _fake_guard(*, headless: bool, enabled: bool):
        guard_enabled_flags.append(enabled)
        yield

    monkeypatch.setattr(fm_client_mod, "async_playwright", lambda: _FakePw())
    monkeypatch.setattr(fm_client_mod, "channel_for_profile", lambda _p: "chrome")
    monkeypatch.setattr(fm_client_mod, "ensure_profile_engine_compatible", lambda _p, _c: None)
    monkeypatch.setattr(fm_client_mod, "macos_background_launch_guard", _fake_guard)

    async with FlowMusicClient(tmp_path / "login_prof", background=False):
        pass

    assert "--window-position=0,0" in captured_launch["args"]
    assert captured_launch["headless"] is False
    assert guard_enabled_flags == [False]


# ---------------------------------------------------------------------------
# 3. FlowMusicClient Mocked Playwright Flows (compose, chat, download, library)
# ---------------------------------------------------------------------------


async def test_client_compose_fills_controls_and_downloads_new_wav_clip(tmp_path: Path) -> None:
    wav_bytes = _synthetic_wav_bytes(duration_sec=2.0)
    clip_calls = 0

    def _clips_poll() -> dict[str, Any]:
        nonlocal clip_calls
        clip_calls += 1
        if clip_calls == 1:
            return {"clips": [{"id": "old-clip"}]}
        return {
            "clips": [
                {
                    "id": "new-clip-1",
                    "title": "Warm Cafe Beat",
                    "wav_url": "https://cdn.example/new-clip-1.wav",
                    "duration": 2.0,
                },
                {"id": "old-clip"},
            ]
        }

    recorded: dict[str, Any] = {}
    page = _FakePage(
        url=FLOWMUSIC_SESSION_URL,
        locators={
            SETTINGS_MENU_SELECTOR: _FakeLocator(visible=True, text="Creator"),
            SOUND_DESC_SELECTOR: _FakeLocator(visible=True, recorded=recorded, name="sound"),
            LYRICS_SELECTOR: _FakeLocator(visible=True, recorded=recorded, name="lyrics"),
            INSTRUMENTAL_SWITCH_SELECTOR: _FakeLocator(
                visible=True, checked=False, recorded=recorded, name="instrumental"
            ),
        },
        api_responses={"/__api/clips/auth-user": _clips_poll},
        download_bytes={"https://cdn.example/new-clip-1.wav": (wav_bytes, "audio/wav")},
    )

    class _GenerateButton(_FakeLocator):
        async def click(self) -> None:
            await super().click()
            page.url = "https://www.flowmusic.app/session/sess_abc12345"

    page._locators[fm_client_mod.GENERATE_BUTTON_SELECTOR] = _GenerateButton(
        visible=True, recorded=recorded, name="generate"
    )
    page.recorded = recorded
    client = _attached_client(tmp_path, page)

    payload, session_id = await client.compose(
        "120 BPM warm lo-fi acoustic guitar groove",
        lyrics="morning sunlight",
        instrumental=True,
        preferred_format="wav",
        timeout_seconds=10.0,
    )
    assert payload.method == "api_clip_download"
    assert payload.ext == "wav"
    assert payload.clip is not None and payload.clip.id == "new-clip-1"
    assert len(payload.data) == len(wav_bytes)
    assert session_id == "sess_abc12345"
    assert "instrumental" in recorded["clicks"]
    assert recorded["fills"]["sound"] == "120 BPM warm lo-fi acoustic guitar groove"
    # Lyrics must NOT be filled when instrumental=True
    assert "lyrics" not in recorded["fills"]


async def test_client_chat_uploads_refs_resumes_session_and_captures_clip(tmp_path: Path) -> None:
    wav_bytes = _synthetic_wav_bytes(duration_sec=2.0)
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(wav_bytes)
    ref_img = tmp_path / "mood.png"
    ref_img.write_bytes(b"\x89PNG\r\n\x1a\n")

    poll_n = 0

    def _clips_poll() -> dict[str, Any]:
        nonlocal poll_n
        poll_n += 1
        if poll_n == 1:
            return {"clips": []}
        return {
            "clips": [
                {
                    "id": "chat-clip-7",
                    "title": "Refined Drop",
                    "wav_url": "https://cdn.example/chat-clip-7.wav",
                }
            ]
        }

    recorded: dict[str, Any] = {}
    page = _FakePage(
        url="https://www.flowmusic.app/session/existing_sess_99",
        locators={
            SETTINGS_MENU_SELECTOR: _FakeLocator(visible=True, text="Creator"),
            PROMPT_SELECTOR: _FakeLocator(visible=True, recorded=recorded, name="prompt"),
            UPLOAD_AUDIO_SELECTOR: _FakeLocator(visible=True, recorded=recorded, name="up_audio"),
            UPLOAD_IMAGE_SELECTOR: _FakeLocator(visible=True, recorded=recorded, name="up_image"),
        },
        api_responses={"/__api/clips/auth-user": _clips_poll},
        download_bytes={"https://cdn.example/chat-clip-7.wav": (wav_bytes, "audio/wav")},
    )
    page.recorded = recorded
    client = _attached_client(tmp_path, page)

    sess_id, _url, payload = await client.chat(
        "Make the drop punchier at bar 5",
        session_id="existing_sess_99",
        ref_audio=ref_audio,
        ref_image=ref_img,
        wait_clip=True,
        timeout_seconds=10.0,
    )
    assert sess_id == "existing_sess_99"
    assert page.goto_urls[0].endswith("/session/existing_sess_99")
    assert recorded["uploads"]["up_audio"] == str(ref_audio.resolve())
    assert recorded["uploads"]["up_image"] == str(ref_img.resolve())
    assert recorded["fills"]["prompt"] == "Make the drop punchier at bar 5"
    assert payload is not None and payload.clip is not None and payload.clip.id == "chat-clip-7"


async def test_client_download_clip_and_library_readers(tmp_path: Path) -> None:
    wav_bytes = _synthetic_wav_bytes(duration_sec=1.0)
    page = _FakePage(
        url=FLOWMUSIC_URL,
        locators={SETTINGS_MENU_SELECTOR: _FakeLocator(visible=True, text="Creator")},
        api_responses={
            "/__api/billing/credits": {"credits_remaining": 99},
            "/__api/billing/subscription": {"plan": "pro", "status": "active"},
            "/__api/models": {"models": [{"public_name": "Lyria 3.5", "is_default": True}]},
            "/__api/clips/auth-user": {
                "clips": [
                    {
                        "id": "dl-1",
                        "title": "Hero Theme",
                        "wav_url": "https://cdn.example/dl-1.wav",
                        "audio_url": "https://cdn.example/dl-1.m4a",
                        "image_url": "https://cdn.example/dl-1.jpg",
                        "lyrics_timing": [{"word": "go", "start": 0.1}],
                    }
                ]
            },
            "/__api/conversations": {"conversations": [{"id": "s-1", "title": "Session 1"}]},
            "/__api/projects": {"projects": [{"id": "p-1", "title": "Proj 1"}]},
            "/__api/playlists": {"playlists": [{"id": "pl-1", "title": "Playlist 1"}]},
            "/__api/spaces/get": {"spaces": [{"id": "sp-1", "title": "Space 1"}]},
            "/__api/music-videos/get": {"music_videos": [{"id": "mv-1", "title": "Video 1"}]},
        },
        download_bytes={
            "https://cdn.example/dl-1.wav": (wav_bytes, "audio/wav"),
            "https://cdn.example/dl-1.jpg": (b"\xff\xd8\xff\xe0jpeg", "image/jpeg"),
        },
    )
    client = _attached_client(tmp_path, page)

    assert (await client.get_credits()).credits_remaining == 99.0
    assert (await client.list_models())[0].public_name == "Lyria 3.5"
    assert (await client.list_clips())[0].id == "dl-1"
    assert (await client.list_sessions())[0].id == "s-1"
    assert (await client.list_projects())[0].kind == "project"
    assert (await client.list_playlists())[0].kind == "playlist"
    assert (await client.list_spaces())[0].kind == "space"
    assert (await client.list_music_videos())[0].kind == "music_video"

    out_wav = tmp_path / "dl-1.wav"
    dl = await client.download_clip(
        "dl-1",
        out_file=out_wav,
        preferred_format="wav",
        include_cover=True,
        include_lyrics_timing=True,
    )
    assert dl.output_file.is_file()
    assert dl.cover_file is not None and dl.cover_file.is_file()
    assert dl.lyrics_timing_file is not None and dl.lyrics_timing_file.is_file()


# ---------------------------------------------------------------------------
# 4. CLI Surface, generate Bridge Routing, Output Formats & Workspace Pipeline
# ---------------------------------------------------------------------------


def test_cli_music_help_exposes_all_15_subcommands() -> None:
    res = CliRunner().invoke(main, ["music", "--help"])
    assert res.exit_code == 0, res.output
    for subcmd in (
        "login",
        "status",
        "credits",
        "models",
        "compose",
        "chat",
        "generate",
        "list",
        "download",
        "sessions",
        "playlists",
        "projects",
        "spaces",
        "videos",
        "probe",
    ):
        assert subcmd in res.output, f"Missing `gflow music {subcmd}`"


def test_cli_music_generate_bridge_routes_compose_vs_chat_and_formats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gflow_cli import cli_music

    monkeypatch.setattr(cli_music, "_profile_dir", lambda _p: tmp_path / "prof")
    called: list[str] = []

    async def _fake_compose(*_args: Any, **_kwargs: Any) -> FlowMusicGenerateResult:
        called.append("compose")
        return FlowMusicGenerateResult(
            output_file=tmp_path / "out.wav",
            format="wav",
            bytes=1024,
            prompt="ambient",
            source_url="https://cdn.example/out.wav",
            capture_method="api_clip_download",
            clip_id="c-compose",
        )

    async def _fake_chat(*_args: Any, **_kwargs: Any) -> FlowMusicChatResult:
        called.append("chat")
        dl = FlowMusicGenerateResult(
            output_file=tmp_path / "chat.wav",
            format="wav",
            bytes=1024,
            prompt="refine",
            source_url="https://cdn.example/chat.wav",
            capture_method="api_clip_download",
            clip_id="c-chat",
            session_id="sess-77",
        )
        return FlowMusicChatResult(
            session_id="sess-77",
            session_url="https://www.flowmusic.app/session/sess-77",
            prompt="refine",
            downloaded=(dl,),
        )

    async def _fake_list_clips(*_args: Any, **_kwargs: Any) -> list[FlowMusicClip]:
        return [FlowMusicClip(id="clip-id-1", title="Track One", duration=30.0)]

    monkeypatch.setattr(fm_service, "compose_flowmusic", _fake_compose)
    monkeypatch.setattr(fm_service, "chat_flowmusic", _fake_chat)
    monkeypatch.setattr(cli_music, "list_flowmusic_clips", _fake_list_clips)

    runner = CliRunner()
    # Default -> routes to compose
    r1 = runner.invoke(main, ["music", "generate", "ambient groove", "--json"])
    assert r1.exit_code == 0, r1.output
    assert json.loads(r1.output)["clip_id"] == "c-compose"

    # With --session -> routes to chat
    r2 = runner.invoke(main, ["music", "generate", "more bass", "--session", "sess-77", "--json"])
    assert r2.exit_code == 0, r2.output
    assert json.loads(r2.output)["clip_id"] == "c-chat"
    assert called == ["compose", "chat"]

    # -o id and -o json on `gflow music list`
    r_id = runner.invoke(main, ["music", "list", "-o", "id"])
    assert r_id.exit_code == 0, r_id.output
    assert r_id.output.strip() == "clip-id-1"

    r_json = runner.invoke(main, ["music", "list", "-o", "json"])
    assert r_json.exit_code == 0, r_json.output
    assert json.loads(r_json.output)["items"][0]["id"] == "clip-id-1"


def test_flowmusic_service_save_payload_wires_workspace_analyze_master_and_qa(
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    init_res = runner.invoke(
        main, ["workspace", "init", "music-e2e", "--root", str(tmp_path), "--json"]
    )
    assert init_res.exit_code == 0, init_res.output
    project_dir = Path(json.loads(init_res.output)["project_dir"])

    wav_bytes = _synthetic_wav_bytes(duration_sec=4.0)
    payload = CapturedAudioPayload(
        data=wav_bytes,
        ext="wav",
        url="https://cdn.example/hero.wav",
        method="api_clip_download",
        clip=FlowMusicClip(id="hero-1", title="Hero", duration=4.0),
    )

    result = fm_service._save_payload(  # pyright: ignore[reportPrivateUsage]
        payload,
        prompt="upbeat coffee promo",
        session_id="sess-1",
        out_file=None,
        project_dir=project_dir,
        shot="s01",
        analyze=True,
        master=True,
        target_lufs=-14.0,
        settings=None,
    )
    assert result.output_file == project_dir / "audio" / "music" / "music-e2e_s01_music_v01.wav"
    assert result.output_file.is_file()
    assert result.analysis_file is not None and result.analysis_file.is_file()
    assert result.master_file is not None and result.master_file.is_file()
    assert result.qa_verdict in {"PASS_TECH", "NEEDS_MINOR_EDIT"}
    assert result.qa_review_path is not None and result.qa_review_path.is_file()


# ---------------------------------------------------------------------------
# 5. MCP Twin & --no-spend Safety Gate
# ---------------------------------------------------------------------------


def test_mcp_music_tools_respect_no_spend_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    from gflow_cli.mcp import server as server_mod

    monkeypatch.delenv("GFLOW_MCP_NO_SPEND", raising=False)
    server_mod._register_surfaces()  # pyright: ignore[reportPrivateUsage]
    mgr = server_mod.server._tool_manager  # pyright: ignore[reportPrivateUsage]
    saved = {n: mgr.get_tool(n) for n in server_mod._SPEND_TOOLS}  # pyright: ignore[reportPrivateUsage]

    read_tools = {
        "gflow_music_status",
        "gflow_music_credits",
        "gflow_music_models",
        "gflow_music_list",
        "gflow_music_sessions",
        "gflow_music_download",
    }
    spend_tools = {
        "gflow_music_compose",
        "gflow_music_chat",
        "gflow_music_generate",
    }
    try:
        # Default (spend enabled): all 9 music tools registered
        server_mod._apply_no_spend_policy()  # pyright: ignore[reportPrivateUsage]
        all_names = set(mgr._tools.keys())  # pyright: ignore[reportPrivateUsage]
        assert read_tools.issubset(all_names)
        assert spend_tools.issubset(all_names)

        # --no-spend enabled: spend music tools removed, read/download tools remain
        monkeypatch.setenv("GFLOW_MCP_NO_SPEND", "1")
        server_mod._apply_no_spend_policy()  # pyright: ignore[reportPrivateUsage]
        restricted_names = set(mgr._tools.keys())  # pyright: ignore[reportPrivateUsage]
        assert read_tools.issubset(restricted_names)
        assert spend_tools.isdisjoint(restricted_names)
    finally:
        for name, tool in saved.items():
            if tool is not None and mgr.get_tool(name) is None:
                mgr._tools[name] = tool  # pyright: ignore[reportPrivateUsage]
