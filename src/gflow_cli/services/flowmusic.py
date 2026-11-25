"""High-level Google Flow Music service delegating to ``FlowMusicClient``.

Wires Flow Music generation and downloads into the MOTIONVIDEO workspace
naming grammar (``<project>/audio/music/<slug>_<shot>_music_vNN.<ext>``),
optional beat analysis (``audio_analysis.analyze_audio``), 2-pass mastering
(``audio_mastering.master_audio``), and Stage-1 deterministic QA checks
(``quality_gate.run_qa_check``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gflow_cli.api.flowmusic_client import (
    FLOWMUSIC_SESSION_URL as FLOWMUSIC_SESSION_URL,
)
from gflow_cli.api.flowmusic_client import (
    FLOWMUSIC_URL as FLOWMUSIC_URL,
)
from gflow_cli.api.flowmusic_client import (
    PROMPT_SELECTOR as PROMPT_SELECTOR,
)
from gflow_cli.api.flowmusic_client import (
    SEND_BUTTON_SELECTOR as SEND_BUTTON_SELECTOR,
)
from gflow_cli.api.flowmusic_client import (
    SETTINGS_BUTTON_SELECTOR as SETTINGS_BUTTON_SELECTOR,
)
from gflow_cli.api.flowmusic_client import (
    UPLOAD_AUDIO_SELECTOR as UPLOAD_AUDIO_SELECTOR,
)
from gflow_cli.api.flowmusic_client import (
    UPLOAD_IMAGE_SELECTOR as UPLOAD_IMAGE_SELECTOR,
)
from gflow_cli.api.flowmusic_client import (
    CapturedAudioPayload,
    FlowMusicAuthError,
    FlowMusicClient,
    FlowMusicError,
    infer_audio_extension,
)
from gflow_cli.api.flowmusic_dto import (
    FlowMusicAuthResult,
    FlowMusicChatResult,
    FlowMusicClip,
    FlowMusicCredits,
    FlowMusicDownloadResult,
    FlowMusicGenerateResult,
    FlowMusicModel,
    FlowMusicResourceItem,
    FlowMusicSession,
)
from gflow_cli.config import Settings, get_settings

__all__ = [
    "FLOWMUSIC_SESSION_URL",
    "FLOWMUSIC_URL",
    "FlowMusicAuthError",
    "FlowMusicAuthResult",
    "FlowMusicChatResult",
    "FlowMusicClip",
    "FlowMusicCredits",
    "FlowMusicDownloadResult",
    "FlowMusicError",
    "FlowMusicGenerateResult",
    "FlowMusicModel",
    "FlowMusicResourceItem",
    "FlowMusicSession",
    "check_flowmusic_status",
    "chat_flowmusic",
    "compose_flowmusic",
    "download_flowmusic_clip",
    "generate_flowmusic",
    "get_flowmusic_credits",
    "infer_audio_extension",
    "list_flowmusic_clips",
    "list_flowmusic_models",
    "list_flowmusic_playlists",
    "list_flowmusic_projects",
    "list_flowmusic_sessions",
    "list_flowmusic_spaces",
    "list_flowmusic_videos",
    "login_flowmusic",
    "probe_flowmusic_ui",
    "resolve_music_output_path",
]


def resolve_music_output_path(
    *,
    ext: str,
    out_file: Path | None = None,
    project_dir: Path | None = None,
    shot: str = "s01",
    settings: Settings | None = None,
) -> Path:
    """Resolve destination path for a generated or downloaded Flow Music track."""
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


def _save_payload(
    payload: CapturedAudioPayload,
    *,
    prompt: str,
    session_id: str | None,
    out_file: Path | None,
    project_dir: Path | None,
    shot: str,
    analyze: bool,
    master: bool,
    target_lufs: float,
    settings: Settings | None,
) -> FlowMusicGenerateResult:
    dest = resolve_music_output_path(
        ext=payload.ext,
        out_file=out_file,
        project_dir=project_dir,
        shot=shot,
        settings=settings,
    )
    dest.write_bytes(payload.data)
    analysis_file, master_file, qa_verdict, qa_review_path = _run_postprocessing(
        dest,
        project_dir=project_dir,
        analyze=analyze,
        master=master,
        target_lufs=target_lufs,
    )
    clip = payload.clip
    return FlowMusicGenerateResult(
        output_file=dest,
        format=payload.ext,
        bytes=len(payload.data),
        prompt=prompt,
        source_url=payload.url,
        capture_method=payload.method,
        clip_id=clip.id if clip else None,
        title=clip.title if clip else None,
        session_id=session_id,
        duration=clip.duration if clip else None,
        analysis_file=analysis_file,
        master_file=master_file,
        qa_verdict=qa_verdict,
        qa_review_path=qa_review_path,
    )


async def login_flowmusic(
    profile_dir: Path,
    *,
    timeout_seconds: float = 300.0,
    settings: Settings | None = None,
) -> FlowMusicAuthResult:
    async with FlowMusicClient(profile_dir, background=False, settings=settings) as client:
        return await client.login(timeout_seconds=timeout_seconds)


async def check_flowmusic_status(
    profile_dir: Path,
    *,
    settings: Settings | None = None,
) -> FlowMusicAuthResult:
    async with FlowMusicClient(profile_dir, background=True, settings=settings) as client:
        await client.page.goto(FLOWMUSIC_URL, wait_until="domcontentloaded", timeout=45_000)
        await client.page.wait_for_timeout(2_000)
        return await client.inspect_auth()


async def get_flowmusic_credits(
    profile_dir: Path,
    *,
    settings: Settings | None = None,
) -> FlowMusicCredits:
    async with FlowMusicClient(profile_dir, background=True, settings=settings) as client:
        return await client.get_credits()


async def list_flowmusic_models(
    profile_dir: Path,
    *,
    settings: Settings | None = None,
) -> list[FlowMusicModel]:
    async with FlowMusicClient(profile_dir, background=True, settings=settings) as client:
        return await client.list_models()


async def list_flowmusic_clips(
    profile_dir: Path,
    *,
    limit: int = 20,
    offset: int = 0,
    settings: Settings | None = None,
) -> list[FlowMusicClip]:
    async with FlowMusicClient(profile_dir, background=True, settings=settings) as client:
        return await client.list_clips(limit=limit, offset=offset)


async def list_flowmusic_sessions(
    profile_dir: Path,
    *,
    limit: int = 30,
    offset: int = 0,
    settings: Settings | None = None,
) -> list[FlowMusicSession]:
    async with FlowMusicClient(profile_dir, background=True, settings=settings) as client:
        return await client.list_sessions(limit=limit, offset=offset)


async def list_flowmusic_projects(
    profile_dir: Path,
    *,
    limit: int = 20,
    offset: int = 0,
    settings: Settings | None = None,
) -> list[FlowMusicResourceItem]:
    async with FlowMusicClient(profile_dir, background=True, settings=settings) as client:
        return await client.list_projects(limit=limit, offset=offset)


async def list_flowmusic_playlists(
    profile_dir: Path,
    *,
    limit: int = 20,
    offset: int = 0,
    settings: Settings | None = None,
) -> list[FlowMusicResourceItem]:
    async with FlowMusicClient(profile_dir, background=True, settings=settings) as client:
        return await client.list_playlists(limit=limit, offset=offset)


async def list_flowmusic_spaces(
    profile_dir: Path,
    *,
    limit: int = 20,
    offset: int = 0,
    settings: Settings | None = None,
) -> list[FlowMusicResourceItem]:
    async with FlowMusicClient(profile_dir, background=True, settings=settings) as client:
        return await client.list_spaces(limit=limit, offset=offset)


async def list_flowmusic_videos(
    profile_dir: Path,
    *,
    limit: int = 20,
    offset: int = 0,
    settings: Settings | None = None,
) -> list[FlowMusicResourceItem]:
    async with FlowMusicClient(profile_dir, background=True, settings=settings) as client:
        return await client.list_music_videos(limit=limit, offset=offset)


async def probe_flowmusic_ui(
    profile_dir: Path,
    *,
    url: str = FLOWMUSIC_SESSION_URL,
    settings: Settings | None = None,
) -> dict[str, Any]:
    async with FlowMusicClient(profile_dir, background=True, settings=settings) as client:
        return await client.probe_ui(url=url)


async def compose_flowmusic(
    profile_dir: Path,
    sound_description: str,
    *,
    lyrics: str | None = None,
    instrumental: bool = False,
    advanced_sound: bool = False,
    model: str | None = None,
    out_file: Path | None = None,
    project_dir: Path | None = None,
    shot: str = "s01",
    preferred_format: str = "wav",
    timeout_seconds: float = 240.0,
    analyze: bool = False,
    master: bool = False,
    target_lufs: float = -14.0,
    settings: Settings | None = None,
) -> FlowMusicGenerateResult:
    if not sound_description.strip():
        msg = "Sound description must not be empty."
        raise ValueError(msg)
    async with FlowMusicClient(profile_dir, background=True, settings=settings) as client:
        payload, session_id = await client.compose(
            sound_description,
            lyrics=lyrics,
            instrumental=instrumental,
            advanced_sound=advanced_sound,
            model=model,
            preferred_format=preferred_format,
            timeout_seconds=timeout_seconds,
        )
    return _save_payload(
        payload,
        prompt=sound_description,
        session_id=session_id,
        out_file=out_file,
        project_dir=project_dir,
        shot=shot,
        analyze=analyze,
        master=master,
        target_lufs=target_lufs,
        settings=settings,
    )


async def chat_flowmusic(
    profile_dir: Path,
    message: str,
    *,
    session_id: str | None = None,
    ref_audio: Path | None = None,
    ref_image: Path | None = None,
    wait_clip: bool = True,
    out_file: Path | None = None,
    project_dir: Path | None = None,
    shot: str = "s01",
    preferred_format: str = "wav",
    timeout_seconds: float = 240.0,
    analyze: bool = False,
    master: bool = False,
    target_lufs: float = -14.0,
    settings: Settings | None = None,
) -> FlowMusicChatResult:
    if not message.strip():
        msg = "Chat message must not be empty."
        raise ValueError(msg)
    async with FlowMusicClient(profile_dir, background=True, settings=settings) as client:
        resolved_session, session_url, payload = await client.chat(
            message,
            session_id=session_id,
            ref_audio=ref_audio,
            ref_image=ref_image,
            wait_clip=wait_clip,
            preferred_format=preferred_format,
            timeout_seconds=timeout_seconds,
        )
    downloaded: list[FlowMusicGenerateResult] = []
    clips: list[FlowMusicClip] = []
    if payload is not None:
        saved = _save_payload(
            payload,
            prompt=message,
            session_id=resolved_session,
            out_file=out_file,
            project_dir=project_dir,
            shot=shot,
            analyze=analyze,
            master=master,
            target_lufs=target_lufs,
            settings=settings,
        )
        downloaded.append(saved)
        if payload.clip is not None:
            clips.append(payload.clip)
    return FlowMusicChatResult(
        session_id=resolved_session,
        session_url=session_url,
        prompt=message,
        clips=tuple(clips),
        downloaded=tuple(downloaded),
    )


async def download_flowmusic_clip(
    profile_dir: Path,
    clip_id: str,
    *,
    out_file: Path | None = None,
    project_dir: Path | None = None,
    shot: str = "s01",
    preferred_format: str = "wav",
    include_cover: bool = False,
    include_lyrics_timing: bool = False,
    settings: Settings | None = None,
) -> FlowMusicDownloadResult:
    dest = resolve_music_output_path(
        ext=preferred_format,
        out_file=out_file,
        project_dir=project_dir,
        shot=shot,
        settings=settings,
    )
    async with FlowMusicClient(profile_dir, background=True, settings=settings) as client:
        return await client.download_clip(
            clip_id,
            out_file=dest,
            preferred_format=preferred_format,
            include_cover=include_cover,
            include_lyrics_timing=include_lyrics_timing,
        )


async def generate_flowmusic(
    profile_dir: Path,
    prompt: str,
    *,
    session_id: str | None = None,
    lyrics: str | None = None,
    instrumental: bool = False,
    advanced_sound: bool = False,
    model: str | None = None,
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
    """Bridge entry point: uses Producer chat if session/refs given, else Compose mode."""
    if session_id or ref_audio is not None or ref_image is not None:
        chat_res = await chat_flowmusic(
            profile_dir,
            prompt,
            session_id=session_id,
            ref_audio=ref_audio,
            ref_image=ref_image,
            wait_clip=True,
            out_file=out_file,
            project_dir=project_dir,
            shot=shot,
            preferred_format=preferred_format,
            timeout_seconds=timeout_seconds,
            analyze=analyze,
            master=master,
            target_lufs=target_lufs,
            settings=settings,
        )
        if not chat_res.downloaded:
            raise FlowMusicError(
                detail="Flow Music chat completed without producing an audio clip."
            )
        return chat_res.downloaded[0]
    return await compose_flowmusic(
        profile_dir,
        prompt,
        lyrics=lyrics,
        instrumental=instrumental,
        advanced_sound=advanced_sound,
        model=model,
        out_file=out_file,
        project_dir=project_dir,
        shot=shot,
        preferred_format=preferred_format,
        timeout_seconds=timeout_seconds,
        analyze=analyze,
        master=master,
        target_lufs=target_lufs,
        settings=settings,
    )
