"""Local zero-credit MCP twin tools for QA, audio analysis/mastering, and local Whisper.

All five tools run 100% locally with zero Flow/Veo credit spend and remain
registered even when ``--no-spend`` / ``GFLOW_MCP_NO_SPEND=1`` is active.
"""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
from typing import Any

from gflow_cli.mcp.server import server
from gflow_cli.services.audio_analysis import analyze_audio
from gflow_cli.services.audio_mastering import master_audio
from gflow_cli.services.quality_gate import record_qa_decision, run_qa_check


def _guarded(fn: Any) -> Any:
    """Import the shared error funnel only after MCP tool registration completes."""
    from functools import wraps

    @wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        from gflow_cli.mcp.tools import _guarded as guarded  # pyright: ignore[reportPrivateUsage]

        return await guarded(fn)(*args, **kwargs)

    wrapper.__gflow_guarded__ = True  # type: ignore[attr-defined]
    return wrapper


@server.tool(
    name="gflow_qa_check",
    description=(
        "Run deterministic Stage-1 QA checks locally on an image, video, or audio file "
        "(sharpness, ffprobe + 1 fps contact strip, LUFS/true-peak). Zero credits."
    ),
)
@_guarded
async def gflow_qa_check(
    file_path: str,
    kind: str = "auto",
    project_dir: str | None = None,
    target_duration: float | None = None,
    target_bpm: float | None = None,
) -> dict[str, Any]:
    return await asyncio.to_thread(
        run_qa_check,
        Path(file_path).expanduser(),
        kind=kind,
        project_dir=Path(project_dir).expanduser() if project_dir else None,
        target_duration=target_duration,
        target_bpm=target_bpm,
    )


@server.tool(
    name="gflow_qa_decide",
    description=(
        "Record a critic review decision (approve|minor_edit|reject) and optionally route "
        "approved/rejected assets in a project workspace. Zero credits."
    ),
)
@_guarded
async def gflow_qa_decide(
    file_path: str,
    verdict: str,
    notes: str = "",
    project_dir: str | None = None,
    move_asset: bool = True,
) -> dict[str, Any]:
    return await asyncio.to_thread(
        record_qa_decision,
        Path(file_path).expanduser(),
        verdict=verdict,
        notes=notes,
        project_dir=Path(project_dir).expanduser() if project_dir else None,
        move_asset=move_asset,
    )


@server.tool(
    name="gflow_audio_analyze",
    description=(
        "Analyze local audio/music for BPM, beats, downbeats, energy peaks, and shot cut plans "
        "(writes audio-map.json). Zero credits."
    ),
)
@_guarded
async def gflow_audio_analyze(
    audio_file: str,
    out_file: str | None = None,
    project_dir: str | None = None,
    shot_durations: list[float] | None = None,
) -> dict[str, Any]:
    durations = tuple(shot_durations) if shot_durations else (4.0, 6.0, 8.0, 10.0, 15.0)
    res = await asyncio.to_thread(
        analyze_audio,
        Path(audio_file).expanduser(),
        shot_durations=durations,
        out_file=Path(out_file).expanduser() if out_file else None,
        project_dir=Path(project_dir).expanduser() if project_dir else None,
    )
    return res.to_dict()


@server.tool(
    name="gflow_audio_master",
    description=(
        "Master a local audio/music track with 2-pass EBU R128 loudness normalization "
        "(-14 LUFS / -1.0 dBTP) and optional voiceover 4:1 vocal-pocket ducking. Zero credits."
    ),
)
@_guarded
async def gflow_audio_master(
    input_file: str,
    out_file: str,
    target_lufs: float = -14.0,
    true_peak: float = -1.0,
    lra: float = 11.0,
    start_sec: float = 0.0,
    duration_sec: float | None = None,
    fade_in_sec: float = 0.05,
    fade_out_sec: float = 0.0,
    voiceover: str | None = None,
    music_volume: float = 0.32,
    overwrite: bool = False,
) -> dict[str, Any]:
    res = await asyncio.to_thread(
        master_audio,
        Path(input_file).expanduser(),
        Path(out_file).expanduser(),
        target_lufs=target_lufs,
        true_peak=true_peak,
        lra=lra,
        start_sec=start_sec,
        duration_sec=duration_sec,
        fade_in_sec=fade_in_sec,
        fade_out_sec=fade_out_sec,
        voiceover=Path(voiceover).expanduser() if voiceover else None,
        music_volume=music_volume,
        overwrite=overwrite,
    )
    return res.to_dict()


@server.tool(
    name="gflow_local_transcribe",
    description=(
        "Transcribe any local audio or video file on Apple Silicon M4 (mlx-whisper) or CPU "
        "(faster-whisper) with word-level timestamps and optional .srt/.json output. Zero credits."
    ),
)
@_guarded
async def gflow_local_transcribe(
    media_file: str,
    backend: str = "auto",
    model: str | None = None,
    language: str | None = None,
    word_timestamps: bool = True,
    out_srt: str | None = None,
    out_json: str | None = None,
) -> dict[str, Any]:
    lw = importlib.import_module("gflow_cli.services.local_whisper")
    tr_fn: Any = lw.transcribe_local
    res: Any = await asyncio.to_thread(
        tr_fn,
        Path(media_file).expanduser(),
        backend=backend,
        model=model,
        language=language,
        word_timestamps=word_timestamps,
        out_srt=Path(out_srt).expanduser() if out_srt else None,
        out_json=Path(out_json).expanduser() if out_json else None,
    )
    return res.to_dict() if hasattr(res, "to_dict") else dict(res)
