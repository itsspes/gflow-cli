"""Local speech synthesis, FlowMusic beat analysis, and audio mastering commands."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click

from gflow_cli import json_output
from gflow_cli._cli_helpers import (
    OutputFormat,
    output_format_option,
    run_with_handlers,
    spends_credits,
)
from gflow_cli.services.audio_analysis import DEFAULT_SHOT_DURATIONS, analyze_audio
from gflow_cli.services.audio_mastering import (
    DEFAULT_LRA_LU,
    DEFAULT_TARGET_LUFS,
    DEFAULT_TRUE_PEAK_DB,
    master_audio,
)
from gflow_cli.services.audio_narration import Provider, narrate


def _parse_shot_durations(raw: str | None) -> tuple[float, ...]:
    if raw is None or not raw.strip():
        return DEFAULT_SHOT_DURATIONS
    values: list[float] = []
    for token in raw.split(","):
        cleaned = token.strip()
        if not cleaned:
            continue
        try:
            dur = float(cleaned)
        except ValueError as exc:
            raise click.BadParameter(
                f"Invalid shot duration {cleaned!r}; expected comma-separated positive numbers."
            ) from exc
        if dur <= 0.0:
            raise click.BadParameter(f"Shot durations must be positive, got {dur}.")
        values.append(dur)
    if not values:
        raise click.BadParameter("Provide at least one positive shot duration (e.g. 4,6,8).")
    return tuple(values)


@click.group()
def audio() -> None:
    """Create narration, analyze FlowMusic beats/loudness, and master audio locally."""


@spends_credits(False)
@audio.command("narrate")
@click.argument("text", required=False)
@click.option(
    "--text-file",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Read narration text from a UTF-8 file instead of TEXT.",
)
@click.option(
    "--out-file",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help=(
        "Write the synthesized audio here. Existing outputs are refused, except an empty "
        "leftover older than 60 s from an interrupted run, which is cleared."
    ),
)
@click.option(
    "--provider",
    type=click.Choice(("edge", "gemini"), case_sensitive=False),
    default="edge",
    show_default=True,
    help=(
        "Speech provider. Narration text is sent to the selected third-party provider; "
        "no Flow credits are used."
    ),
)
@click.option("--voice", help="Provider voice ID.")
@click.option(
    "--rate", default="+0%", show_default=True, help="Speech rate adjustment (edge only)."
)
@click.option(
    "--pitch", default="+0Hz", show_default=True, help="Speech pitch adjustment (edge only)."
)
@output_format_option(has_out_file=True)
def narrate_command(
    text: str | None,
    text_file: Path | None,
    out_file: Path,
    provider: Provider,
    voice: str | None,
    rate: str,
    pitch: str,
    as_json: bool,
    output_format: str = "table",
) -> None:
    """Synthesize TEXT (or --text-file) into a local audio file.

    Narration text is sent to the selected third-party provider. This does not use Flow credits.
    """
    if text is not None and text_file is not None:
        raise click.UsageError("TEXT and --text-file are mutually exclusive.")
    if text is None and text_file is None:
        raise click.UsageError("Provide TEXT or --text-file.")
    is_json = as_json or output_format == OutputFormat.JSON.value

    async def execute() -> None:
        result = await asyncio.to_thread(
            narrate,
            text=text,
            text_file=text_file.expanduser() if text_file else None,
            out_file=out_file.expanduser(),
            provider=provider,
            voice=voice,
            rate=rate,
            pitch=pitch,
        )
        if output_format == OutputFormat.ID.value:
            click.echo(result.out_file)
        elif is_json:
            json_output.emit(result.payload())
        else:
            click.echo(f"Narration saved to {result.out_file} ({result.provider}).")

    run_with_handlers(
        execute, cli_command="audio narrate", as_json=is_json, output_format=output_format
    )


@spends_credits(False)
@audio.command(
    "analyze",
    short_help="Analyze FlowMusic BPM, beats, loudness, silence, and shot cut points.",
)
@click.argument("audio_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--out-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write structured audio-map.json to this path.",
)
@click.option(
    "--project-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Project workspace directory (defaults audio-map.json under audio/analysis/).",
)
@click.option(
    "--shot-durations",
    default="4,6,8,10,15",
    show_default=True,
    help="Comma-separated Omni/Veo shot durations in seconds for bar-aligned cut plans.",
)
@output_format_option(has_out_file=True)
def analyze_command(
    audio_file: Path,
    out_file: Path | None,
    project_dir: Path | None,
    shot_durations: str,
    as_json: bool,
    output_format: str = "table",
) -> None:
    """Analyze AUDIO_FILE for BPM, beats, downbeats, LUFS, true peak, and shot cut plans.

    Local, credit-free analysis using librosa (when installed) or deterministic FFmpeg PCM DSP.
    """
    durations = _parse_shot_durations(shot_durations)
    is_json = as_json or output_format == OutputFormat.JSON.value

    async def execute() -> None:
        result = await asyncio.to_thread(
            analyze_audio,
            audio_file.expanduser(),
            shot_durations=durations,
            out_file=out_file.expanduser() if out_file is not None else None,
            project_dir=project_dir.expanduser() if project_dir is not None else None,
        )
        if output_format == OutputFormat.ID.value:
            click.echo(result.out_file or result.source)
        elif is_json:
            json_output.emit(result.to_dict())
        else:
            dest_note = f" -> {result.out_file}" if result.out_file is not None else ""
            click.echo(
                f"Analyzed {result.source.name}: {result.bpm:.1f} BPM, "
                f"{result.duration_sec:.2f}s, {result.lufs_integrated:.1f} LUFS, "
                f"{result.true_peak_db:.1f} dBTP, {len(result.downbeats)} downbeats{dest_note}."
            )

    run_with_handlers(
        execute, cli_command="audio analyze", as_json=is_json, output_format=output_format
    )


@spends_credits(False)
@audio.command(
    "master",
    short_help="Master audio with 2-pass EBU R128 LUFS, fades, and optional VO ducking.",
)
@click.argument("audio_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--out-file",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Destination mastered audio path (.wav, .mp3, .m4a, .aac, .flac, .ogg, .opus).",
)
@click.option(
    "--target-lufs",
    type=click.FloatRange(-70.0, -5.0),
    default=DEFAULT_TARGET_LUFS,
    show_default=True,
    help="Target integrated loudness in LUFS (-14.0 web/social, -16.0 mobile/VO).",
)
@click.option(
    "--true-peak",
    type=click.FloatRange(-9.0, 0.0),
    default=DEFAULT_TRUE_PEAK_DB,
    show_default=True,
    help="Maximum true peak ceiling in dBTP.",
)
@click.option(
    "--lra",
    type=click.FloatRange(1.0, 50.0),
    default=DEFAULT_LRA_LU,
    show_default=True,
    help="Target loudness range in LU.",
)
@click.option(
    "--start",
    "start_sec",
    type=click.FloatRange(min=0.0),
    default=0.0,
    show_default=True,
    help="Start trim offset in seconds.",
)
@click.option(
    "--duration",
    "duration_sec",
    type=click.FloatRange(min=0.1),
    default=None,
    help="Trim duration in seconds (e.g. bar-aligned runtime).",
)
@click.option(
    "--fade-in",
    "fade_in_sec",
    type=click.FloatRange(min=0.0),
    default=0.05,
    show_default=True,
    help="Anti-click fade-in duration in seconds.",
)
@click.option(
    "--fade-out",
    "fade_out_sec",
    type=click.FloatRange(min=0.0),
    default=0.0,
    show_default=True,
    help="Smooth exponential fade-out duration in seconds.",
)
@click.option(
    "--voiceover",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional voiceover file; enables vocal-pocket EQ + gentle 4:1 sidechain ducking.",
)
@click.option(
    "--music-volume",
    type=click.FloatRange(0.0, 1.0),
    default=0.32,
    show_default=True,
    help="Baseline music bed gain when --voiceover is supplied.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Replace --out-file if it already exists.",
)
@output_format_option(has_out_file=True)
def master_command(
    audio_file: Path,
    out_file: Path,
    target_lufs: float,
    true_peak: float,
    lra: float,
    start_sec: float,
    duration_sec: float | None,
    fade_in_sec: float,
    fade_out_sec: float,
    voiceover: Path | None,
    music_volume: float,
    overwrite: bool,
    as_json: bool,
    output_format: str = "table",
) -> None:
    """Master AUDIO_FILE locally via FFmpeg 2-pass EBU R128 normalization and optional VO mix.

    Local, credit-free processing. Verifies final LUFS and true peak via FFmpeg ebur128.
    """
    is_json = as_json or output_format == OutputFormat.JSON.value

    async def execute() -> None:
        result = await asyncio.to_thread(
            master_audio,
            audio_file.expanduser(),
            out_file.expanduser(),
            target_lufs=target_lufs,
            true_peak=true_peak,
            lra=lra,
            start_sec=start_sec,
            duration_sec=duration_sec,
            fade_in_sec=fade_in_sec,
            fade_out_sec=fade_out_sec,
            voiceover=voiceover.expanduser() if voiceover is not None else None,
            music_volume=music_volume,
            overwrite=overwrite,
        )
        if output_format == OutputFormat.ID.value:
            click.echo(result.output_file)
        elif is_json:
            json_output.emit(result.to_dict())
        else:
            click.echo(
                f"Mastered {result.output_file} ({result.verified_output_lufs:.1f} LUFS, "
                f"{result.verified_output_true_peak_db:.1f} dBTP, {result.duration_sec:.2f}s)."
            )

    run_with_handlers(
        execute, cli_command="audio master", as_json=is_json, output_format=output_format
    )
