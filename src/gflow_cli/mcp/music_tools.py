"""MCP twin tools for Google Flow Music (flowmusic.app).

Mirrors ``mcp/video_assembly.py`` and ``mcp/tools.py`` conventions. Generation
tools (``gflow_music_compose``, ``gflow_music_chat``, ``gflow_music_generate``)
are listed in ``mcp.server._SPEND_TOOLS`` so ``--no-spend`` omits them while
keeping read/download tools registered.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gflow_cli._cli_helpers import (
    _resolve_profile,  # pyright: ignore[reportPrivateUsage]
)
from gflow_cli.config import get_settings
from gflow_cli.mcp.server import server
from gflow_cli.services.flowmusic import (
    chat_flowmusic,
    check_flowmusic_status,
    compose_flowmusic,
    download_flowmusic_clip,
    generate_flowmusic,
    get_flowmusic_credits,
    list_flowmusic_clips,
    list_flowmusic_models,
    list_flowmusic_sessions,
)


def _guarded(fn: Any) -> Any:
    """Import the shared error funnel only after MCP tool registration completes."""
    from functools import wraps

    @wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        from gflow_cli.mcp.tools import _guarded as guarded  # pyright: ignore[reportPrivateUsage]

        return await guarded(fn)(*args, **kwargs)

    wrapper.__gflow_guarded__ = True  # type: ignore[attr-defined]
    return wrapper


def _profile_dir(profile: str | None) -> Path:
    return get_settings().profile_subdir(_resolve_profile(profile))


@server.tool(
    name="gflow_music_status",
    description="Check Google Flow Music (flowmusic.app) sign-in status silently in background.",
)
@_guarded
async def gflow_music_status(profile: str | None = None) -> dict[str, Any]:
    return (await check_flowmusic_status(_profile_dir(profile))).to_dict()


@server.tool(
    name="gflow_music_credits",
    description="Return Google Flow Music remaining credits, tokens, and subscription tier.",
)
@_guarded
async def gflow_music_credits(profile: str | None = None) -> dict[str, Any]:
    return (await get_flowmusic_credits(_profile_dir(profile))).to_dict()


@server.tool(
    name="gflow_music_models",
    description="List available Google Flow Music models (e.g. Lyria 3.5, Lyria 3 Pro).",
)
@_guarded
async def gflow_music_models(profile: str | None = None) -> dict[str, Any]:
    models = [m.to_dict() for m in await list_flowmusic_models(_profile_dir(profile))]
    return {"status": "ok", "models": models, "count": len(models)}


@server.tool(
    name="gflow_music_list",
    description="List generated Google Flow Music clips in the signed-in user's library.",
)
@_guarded
async def gflow_music_list(
    limit: int = 20, offset: int = 0, profile: str | None = None
) -> dict[str, Any]:
    clips = [
        c.to_dict()
        for c in await list_flowmusic_clips(_profile_dir(profile), limit=limit, offset=offset)
    ]
    return {"status": "ok", "clips": clips, "count": len(clips)}


@server.tool(
    name="gflow_music_sessions",
    description="List Google Flow Music Producer conversation sessions.",
)
@_guarded
async def gflow_music_sessions(
    limit: int = 30, offset: int = 0, profile: str | None = None
) -> dict[str, Any]:
    sessions = [
        s.to_dict()
        for s in await list_flowmusic_sessions(_profile_dir(profile), limit=limit, offset=offset)
    ]
    return {"status": "ok", "sessions": sessions, "count": len(sessions)}


@server.tool(
    name="gflow_music_download",
    description="Download an existing Flow Music clip (lossless WAV or M4A) into out_file.",
)
@_guarded
async def gflow_music_download(
    clip_id: str,
    out_file: str | None = None,
    project_dir: str | None = None,
    shot: str = "s01",
    preferred_format: str = "wav",
    include_cover: bool = False,
    include_lyrics_timing: bool = False,
    profile: str | None = None,
) -> dict[str, Any]:
    res = await download_flowmusic_clip(
        _profile_dir(profile),
        clip_id,
        out_file=Path(out_file).expanduser() if out_file else None,
        project_dir=Path(project_dir).expanduser() if project_dir else None,
        shot=shot,
        preferred_format=preferred_format,
        include_cover=include_cover,
        include_lyrics_timing=include_lyrics_timing,
    )
    return res.to_dict()


@server.tool(
    name="gflow_music_compose",
    description="Compose a new track in Google Flow Music Compose mode (spends Flow Music quota).",
)
@_guarded
async def gflow_music_compose(
    sound_description: str,
    lyrics: str | None = None,
    instrumental: bool = False,
    advanced_sound: bool = False,
    model: str | None = None,
    out_file: str | None = None,
    project_dir: str | None = None,
    shot: str = "s01",
    preferred_format: str = "wav",
    analyze: bool = False,
    master: bool = False,
    target_lufs: float = -14.0,
    profile: str | None = None,
) -> dict[str, Any]:
    res = await compose_flowmusic(
        _profile_dir(profile),
        sound_description,
        lyrics=lyrics,
        instrumental=instrumental,
        advanced_sound=advanced_sound,
        model=model,
        out_file=Path(out_file).expanduser() if out_file else None,
        project_dir=Path(project_dir).expanduser() if project_dir else None,
        shot=shot,
        preferred_format=preferred_format,
        analyze=analyze,
        master=master,
        target_lufs=target_lufs,
    )
    return res.to_dict()


@server.tool(
    name="gflow_music_chat",
    description="Send a Producer chat turn on Google Flow Music (spends Flow Music quota).",
)
@_guarded
async def gflow_music_chat(
    message: str,
    session_id: str | None = None,
    ref_audio: str | None = None,
    ref_image: str | None = None,
    wait_clip: bool = True,
    out_file: str | None = None,
    project_dir: str | None = None,
    shot: str = "s01",
    preferred_format: str = "wav",
    analyze: bool = False,
    master: bool = False,
    target_lufs: float = -14.0,
    profile: str | None = None,
) -> dict[str, Any]:
    res = await chat_flowmusic(
        _profile_dir(profile),
        message,
        session_id=session_id,
        ref_audio=Path(ref_audio).expanduser() if ref_audio else None,
        ref_image=Path(ref_image).expanduser() if ref_image else None,
        wait_clip=wait_clip,
        out_file=Path(out_file).expanduser() if out_file else None,
        project_dir=Path(project_dir).expanduser() if project_dir else None,
        shot=shot,
        preferred_format=preferred_format,
        analyze=analyze,
        master=master,
        target_lufs=target_lufs,
    )
    return res.to_dict()


@server.tool(
    name="gflow_music_generate",
    description=(
        "Generate a track on Google Flow Music (Compose by default, Chat if session/refs given)."
    ),
)
@_guarded
async def gflow_music_generate(
    prompt: str,
    session_id: str | None = None,
    lyrics: str | None = None,
    instrumental: bool = False,
    model: str | None = None,
    ref_audio: str | None = None,
    ref_image: str | None = None,
    out_file: str | None = None,
    project_dir: str | None = None,
    shot: str = "s01",
    preferred_format: str = "wav",
    analyze: bool = False,
    master: bool = False,
    target_lufs: float = -14.0,
    profile: str | None = None,
) -> dict[str, Any]:
    res = await generate_flowmusic(
        _profile_dir(profile),
        prompt,
        session_id=session_id,
        lyrics=lyrics,
        instrumental=instrumental,
        model=model,
        ref_audio=Path(ref_audio).expanduser() if ref_audio else None,
        ref_image=Path(ref_image).expanduser() if ref_image else None,
        out_file=Path(out_file).expanduser() if out_file else None,
        project_dir=Path(project_dir).expanduser() if project_dir else None,
        shot=shot,
        preferred_format=preferred_format,
        analyze=analyze,
        master=master,
        target_lufs=target_lufs,
    )
    return res.to_dict()
