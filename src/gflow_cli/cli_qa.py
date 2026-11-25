"""CLI commands for the Stage-1 Deterministic Quality Gate (`gflow qa check|decide`)."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from gflow_cli import json_output
from gflow_cli._cli_helpers import OutputFormat, output_format_option, spends_credits
from gflow_cli.services.quality_gate import (
    DECISION_VERDICTS,
    record_qa_decision,
    run_qa_check,
)

console = Console()


@spends_credits(False)
@click.group("qa")
def qa() -> None:
    """Deterministic Stage-1 quality checks and critic review decisions."""


@spends_credits(False)
@qa.command("check")
@click.argument("file_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--kind",
    type=click.Choice(["auto", "image", "video", "audio"]),
    default="auto",
    show_default=True,
    help="Media type to inspect.",
)
@click.option(
    "--project-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Project workspace directory (writes review.json and contact strip into qa/reviews/).",
)
@click.option("--target-duration", type=float, default=None, help="Expected duration in seconds.")
@click.option("--target-bpm", type=float, default=None, help="Expected music tempo in BPM.")
@click.option(
    "--expect-audio",
    is_flag=True,
    default=False,
    help="Expect an intentional audio track on video (suppresses Veo unstripped-audio warning).",
)
@output_format_option(allow_id=False)
def qa_check(
    file_path: Path,
    kind: str,
    project_dir: Path | None,
    target_duration: float | None,
    target_bpm: float | None,
    expect_audio: bool,
    as_json: bool,
    output_format: str = "table",
) -> None:
    """Run deterministic technical QA checks on an image, video, or audio asset."""
    try:
        payload = run_qa_check(
            file_path,
            kind=kind,
            project_dir=project_dir,
            target_duration=target_duration,
            target_bpm=target_bpm,
            expect_audio=expect_audio,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    if as_json or output_format == OutputFormat.JSON.value:
        json_output.emit(payload)
        return
    verdict = payload["verdict"]
    color = {"PASS_TECH": "green", "NEEDS_MINOR_EDIT": "yellow", "FAIL_TECH": "red"}.get(
        verdict, "white"
    )
    console.print(f"[{color}]{verdict}[/{color}] ({payload['kind']}): {payload['file']}")
    for issue in payload.get("issues", []):
        console.print(f"  - [{issue['severity']}] {issue['code']}: {issue['detail']}")
    for action in payload.get("next_actions", []):
        console.print(f"  -> {action}")


@spends_credits(False)
@qa.command("decide")
@click.argument("file_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--verdict",
    type=click.Choice(list(DECISION_VERDICTS)),
    required=True,
    help="Critic review verdict.",
)
@click.option("--notes", default="", help="Short rationale or edit instructions.")
@click.option(
    "--project-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Project workspace directory (updates review.json, project.json, and routes folders).",
)
@click.option(
    "--move/--no-move",
    default=True,
    show_default=True,
    help="Move approved/rejected files into the workspace's approved/ or rejected/ folder.",
)
@output_format_option(allow_id=False)
def qa_decide(
    file_path: Path,
    verdict: str,
    notes: str,
    project_dir: Path | None,
    move: bool,
    as_json: bool,
    output_format: str = "table",
) -> None:
    """Record a critic review decision and route approved/rejected assets."""
    try:
        payload = record_qa_decision(
            file_path,
            verdict=verdict,
            notes=notes,
            project_dir=project_dir,
            move_asset=move,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    if as_json or output_format == OutputFormat.JSON.value:
        json_output.emit(payload)
        return
    console.print(
        f"Recorded [bold]{payload['verdict']}[/bold] -> {payload['destination']}"
        + (f" ({notes})" if notes else "")
    )
