"""CLI commands for native Google Flow Music (`gflow music login|status|probe|generate`)."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from gflow_cli import json_output
from gflow_cli._cli_helpers import (
    OutputFormat,
    _resolve_profile,  # pyright: ignore[reportPrivateUsage]
    output_format_option,
    run_with_handlers,
    spends_credits,
)
from gflow_cli.config import get_settings
from gflow_cli.services.flowmusic import (
    FLOWMUSIC_SESSION_URL,
    check_flowmusic_status,
    generate_flowmusic,
    login_flowmusic,
    probe_flowmusic_ui,
)

console = Console()


def _profile_dir(profile: str | None) -> Path:
    name = _resolve_profile(profile)
    return get_settings().profile_subdir(name)


@spends_credits(False)
@click.group("music")
def music() -> None:
    """Generate and inspect music via Google Flow Music (flowmusic.app)."""


@spends_credits(False)
@music.command("login")
@click.option("--profile", default=None, help="Profile name (overrides default).")
@click.option(
    "--timeout",
    "timeout_seconds",
    type=float,
    default=300.0,
    show_default=True,
    help="Seconds to wait for interactive Google sign-in.",
)
@output_format_option(allow_id=False)
def music_login(
    profile: str | None,
    timeout_seconds: float,
    as_json: bool,
    output_format: str = "table",
) -> None:
    """Open Chrome foreground at flowmusic.app to sign in with Google."""
    is_json = as_json or output_format == OutputFormat.JSON.value

    async def _run() -> None:
        result = await login_flowmusic(_profile_dir(profile), timeout_seconds=timeout_seconds)
        if is_json:
            json_output.emit(result.to_dict())
        elif result.signed_in:
            console.print(f"[green]Signed in[/green] to Google Flow Music ({result.url}).")
        else:
            console.print(f"[yellow]{result.detail}[/yellow]")

    run_with_handlers(
        _run,
        cli_command="music login",
        as_json=is_json,
        output_format=output_format,
    )


@spends_credits(False)
@music.command("status")
@click.option("--profile", default=None, help="Profile name (overrides default).")
@output_format_option(allow_id=False)
def music_status(
    profile: str | None,
    as_json: bool,
    output_format: str = "table",
) -> None:
    """Silently check Google Flow Music authentication state (0 Veo credits)."""
    is_json = as_json or output_format == OutputFormat.JSON.value

    async def _run() -> None:
        result = await check_flowmusic_status(_profile_dir(profile))
        if is_json:
            json_output.emit(result.to_dict())
        elif result.signed_in:
            console.print(f"Flow Music: [green]signed in[/green] ({result.url})")
        else:
            console.print("Flow Music: [yellow]signed out[/yellow] — run `gflow music login`.")

    run_with_handlers(
        _run,
        cli_command="music status",
        as_json=is_json,
        output_format=output_format,
    )


@spends_credits(False)
@music.command("probe")
@click.option("--profile", default=None, help="Profile name (overrides default).")
@click.option(
    "--url",
    default=FLOWMUSIC_SESSION_URL,
    show_default=True,
    help="Flow Music URL to inspect.",
)
@output_format_option(allow_id=False)
def music_probe(
    profile: str | None,
    url: str,
    as_json: bool,
    output_format: str = "table",
) -> None:
    """Silently inspect Flow Music UI controls and observed audio/API URLs."""
    is_json = as_json or output_format == OutputFormat.JSON.value

    async def _run() -> None:
        payload = await probe_flowmusic_ui(_profile_dir(profile), url=url)
        if is_json:
            json_output.emit(payload)
        else:
            console.print(
                f"Probed {payload['url']} — signed_in={payload['signed_in']}, "
                f"composer_visible={payload['composer_visible']}, "
                f"controls={payload['controls_count']}, "
                f"observed_urls={len(payload['observed_urls'])}."
            )

    run_with_handlers(
        _run,
        cli_command="music probe",
        as_json=is_json,
        output_format=output_format,
    )


@spends_credits(False)
@music.command("generate")
@click.argument("prompt")
@click.option(
    "--out-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Explicit output audio path.",
)
@click.option(
    "--project-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Project workspace directory (saves into audio/music/ and runs QA check).",
)
@click.option("--shot", default="s01", show_default=True, help="Shot ID for project naming.")
@click.option(
    "--ref-audio",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional reference audio file to upload.",
)
@click.option(
    "--ref-image",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional reference image file to upload.",
)
@click.option(
    "--format",
    "preferred_format",
    type=click.Choice(["wav", "mp3", "m4a"], case_sensitive=False),
    default="wav",
    show_default=True,
    help="Preferred audio export format.",
)
@click.option(
    "--timeout",
    "timeout_seconds",
    type=float,
    default=240.0,
    show_default=True,
    help="Maximum seconds to wait for music generation.",
)
@click.option(
    "--analyze",
    is_flag=True,
    default=False,
    help="Run beat/BPM analysis after generation and write audio-map.json.",
)
@click.option(
    "--master",
    is_flag=True,
    default=False,
    help="Run 2-pass EBU R128 mastering (-14 LUFS / -1.0 dBTP) after generation.",
)
@click.option(
    "--target-lufs",
    type=float,
    default=-14.0,
    show_default=True,
    help="Target integrated loudness (LUFS) when --master is enabled.",
)
@click.option("--profile", default=None, help="Profile name (overrides default).")
@output_format_option(allow_id=False)
def music_generate(
    prompt: str,
    out_file: Path | None,
    project_dir: Path | None,
    shot: str,
    ref_audio: Path | None,
    ref_image: Path | None,
    preferred_format: str,
    timeout_seconds: float,
    analyze: bool,
    master: bool,
    target_lufs: float,
    profile: str | None,
    as_json: bool,
    output_format: str = "table",
) -> None:
    """Generate a music track from PROMPT on Google Flow Music."""
    is_json = as_json or output_format == OutputFormat.JSON.value

    async def _run() -> None:
        result = await generate_flowmusic(
            _profile_dir(profile),
            prompt,
            out_file=out_file,
            project_dir=project_dir,
            shot=shot,
            ref_audio=ref_audio,
            ref_image=ref_image,
            preferred_format=preferred_format.lower(),
            timeout_seconds=timeout_seconds,
            analyze=analyze,
            master=master,
            target_lufs=target_lufs,
        )
        if is_json:
            json_output.emit(result.to_dict())
        else:
            console.print(
                f"Saved Flow Music track: [bold]{result.output_file}[/bold] "
                f"({result.format}, {result.bytes:,} bytes, via {result.capture_method})."
            )
            if result.master_file:
                console.print(f"Mastered track: {result.master_file}")
            if result.analysis_file:
                console.print(f"Beat map: {result.analysis_file}")
            if result.qa_verdict:
                console.print(f"QA verdict: {result.qa_verdict}")

    run_with_handlers(
        _run,
        cli_command="music generate",
        as_json=is_json,
        output_format=output_format,
    )
