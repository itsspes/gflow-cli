"""CLI commands for native Google Flow Music (`gflow music ...`)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

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
    chat_flowmusic,
    check_flowmusic_status,
    compose_flowmusic,
    download_flowmusic_clip,
    generate_flowmusic,
    get_flowmusic_credits,
    list_flowmusic_clips,
    list_flowmusic_models,
    list_flowmusic_playlists,
    list_flowmusic_projects,
    list_flowmusic_sessions,
    list_flowmusic_spaces,
    list_flowmusic_videos,
    login_flowmusic,
    probe_flowmusic_ui,
    rename_flowmusic_session,
)

console = Console()


def _profile_dir(profile: str | None) -> Path:
    return get_settings().profile_subdir(_resolve_profile(profile))


def _fmt(as_json: bool, output_format: str) -> str:
    return OutputFormat.JSON.value if as_json else output_format


def _emit_resource_list(
    items: list[dict[str, Any]],
    *,
    title: str,
    fmt: str,
    columns: tuple[tuple[str, str], ...],
) -> None:
    if fmt == OutputFormat.JSON.value:
        json_output.emit({"items": items, "count": len(items)})
        return
    if fmt == OutputFormat.ID.value:
        for item in items:
            click.echo(str(item.get("id") or item.get("public_name") or ""))
        return
    table = Table(title=title)
    for label, _key in columns:
        table.add_column(label)
    for item in items:
        table.add_row(
            *[str(item.get(key) if item.get(key) is not None else "-") for _, key in columns]
        )
    console.print(table)


@spends_credits(False)
@click.group("music")
def music() -> None:
    """Generate, inspect, and download music via Google Flow Music (flowmusic.app)."""


@spends_credits(False)
@music.command("login")
@click.option("--profile", default=None, help="Profile name (overrides default).")
@click.option("--timeout", "timeout_seconds", type=float, default=300.0, show_default=True)
@output_format_option(allow_id=False)
def music_login(
    profile: str | None,
    timeout_seconds: float,
    as_json: bool,
    output_format: str = "table",
) -> None:
    """Open Chrome foreground at flowmusic.app to sign in with Google."""
    fmt = _fmt(as_json, output_format)

    async def _run() -> None:
        result = await login_flowmusic(_profile_dir(profile), timeout_seconds=timeout_seconds)
        if fmt == OutputFormat.JSON.value:
            json_output.emit(result.to_dict())
        elif result.signed_in:
            acct = f" as {result.account}" if result.account else ""
            console.print(f"[green]Signed in[/green] to Google Flow Music{acct} ({result.url}).")
        else:
            console.print(f"[yellow]{result.detail}[/yellow]")

    run_with_handlers(_run, cli_command="music login", as_json=fmt == "json", output_format=fmt)


@spends_credits(False)
@music.command("status")
@click.option("--profile", default=None, help="Profile name (overrides default).")
@output_format_option(allow_id=False)
def music_status(profile: str | None, as_json: bool, output_format: str = "table") -> None:
    """Silently check Google Flow Music authentication state."""
    fmt = _fmt(as_json, output_format)

    async def _run() -> None:
        result = await check_flowmusic_status(_profile_dir(profile))
        if fmt == OutputFormat.JSON.value:
            json_output.emit(result.to_dict())
        elif result.signed_in:
            acct = f" as {result.account}" if result.account else ""
            console.print(f"Flow Music: [green]signed in[/green]{acct} ({result.url})")
        else:
            console.print("Flow Music: [yellow]signed out[/yellow] — run `gflow music login`.")

    run_with_handlers(_run, cli_command="music status", as_json=fmt == "json", output_format=fmt)


@spends_credits(False)
@music.command("credits")
@click.option("--profile", default=None, help="Profile name (overrides default).")
@output_format_option(allow_id=False)
def music_credits(profile: str | None, as_json: bool, output_format: str = "table") -> None:
    """Show Flow Music remaining credits, tokens, and subscription tier."""
    fmt = _fmt(as_json, output_format)

    async def _run() -> None:
        creds = await get_flowmusic_credits(_profile_dir(profile))
        if fmt == OutputFormat.JSON.value:
            json_output.emit(creds.to_dict())
            return
        console.print(
            f"Flow Music credits={creds.credits_remaining}, "
            f"tokens={creds.tokens_remaining}, plan={creds.plan or '-'}"
        )

    run_with_handlers(_run, cli_command="music credits", as_json=fmt == "json", output_format=fmt)


@spends_credits(False)
@music.command("models")
@click.option("--profile", default=None, help="Profile name (overrides default).")
@output_format_option()
def music_models(profile: str | None, as_json: bool, output_format: str = "table") -> None:
    """List available Flow Music models (e.g. Lyria 3.5, Lyria 3 Pro)."""
    fmt = _fmt(as_json, output_format)

    async def _run() -> None:
        models = [m.to_dict() for m in await list_flowmusic_models(_profile_dir(profile))]
        _emit_resource_list(
            models,
            title="Flow Music Models",
            fmt=fmt,
            columns=(
                ("Name", "public_name"),
                ("Version", "version"),
                ("Default", "is_default"),
                ("Description", "description"),
            ),
        )

    run_with_handlers(_run, cli_command="music models", as_json=fmt == "json", output_format=fmt)


@spends_credits(True)
@music.command("compose")
@click.argument("sound_description")
@click.option("--lyrics", default=None, help="Optional song lyrics (ignored if --instrumental).")
@click.option("--instrumental", is_flag=True, default=False, help="Enable instrumental-only mode.")
@click.option("--advanced-sound", is_flag=True, default=False, help="Enable advanced sound mode.")
@click.option("--model", default=None, help="Model name (e.g. 'Lyria 3.5').")
@click.option("--bpm", type=click.IntRange(40, 240), default=None, help="Target tempo in BPM.")
@click.option(
    "--length",
    default=None,
    help="Target track length in seconds (e.g. 90) or M:SS (clamped to FlowMusic min 1:00).",
)
@click.option("--seed", default=None, help="Optional numeric/string generation seed.")
@click.option("--title", default=None, help="Native song title (defaults to '<slug> <shot>').")
@click.option(
    "--session-title",
    default=None,
    help="Flow Music session title (defaults to workspace slug when --project-dir is set).",
)
@click.option("--out-file", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--project-dir", type=click.Path(file_okay=False, path_type=Path), default=None)
@click.option("--shot", default="s01", show_default=True)
@click.option(
    "--format",
    "preferred_format",
    type=click.Choice(["wav", "m4a"], case_sensitive=False),
    default="wav",
    show_default=True,
)
@click.option("--timeout", "timeout_seconds", type=float, default=240.0, show_default=True)
@click.option("--analyze", is_flag=True, default=False, help="Run beat/BPM analysis after export.")
@click.option("--master", is_flag=True, default=False, help="Master to -14 LUFS / -1.0 dBTP.")
@click.option("--target-lufs", type=float, default=-14.0, show_default=True)
@click.option("--profile", default=None, help="Profile name (overrides default).")
@output_format_option(has_out_file=True)
def music_compose(
    sound_description: str,
    lyrics: str | None,
    instrumental: bool,
    advanced_sound: bool,
    model: str | None,
    bpm: int | None,
    length: str | None,
    seed: str | None,
    title: str | None,
    session_title: str | None,
    out_file: Path | None,
    project_dir: Path | None,
    shot: str,
    preferred_format: str,
    timeout_seconds: float,
    analyze: bool,
    master: bool,
    target_lufs: float,
    profile: str | None,
    as_json: bool,
    output_format: str = "table",
) -> None:
    """Compose a song directly via Flow Music's Compose panel."""
    fmt = _fmt(as_json, output_format)

    async def _run() -> None:
        res = await compose_flowmusic(
            _profile_dir(profile),
            sound_description,
            lyrics=lyrics,
            instrumental=instrumental,
            advanced_sound=advanced_sound,
            model=model,
            bpm=bpm,
            length=length,
            seed=seed,
            title=title,
            session_title=session_title,
            out_file=out_file,
            project_dir=project_dir,
            shot=shot,
            preferred_format=preferred_format.lower(),
            timeout_seconds=timeout_seconds,
            analyze=analyze,
            master=master,
            target_lufs=target_lufs,
        )
        if fmt == OutputFormat.JSON.value:
            json_output.emit(res.to_dict())
        elif fmt == OutputFormat.ID.value:
            click.echo(res.clip_id or str(res.output_file))
        else:
            console.print(
                f"Composed [bold]{res.output_file}[/bold] ({res.format}, {res.bytes:,} bytes)"
            )

    run_with_handlers(_run, cli_command="music compose", as_json=fmt == "json", output_format=fmt)


@spends_credits(True)
@music.command("chat")
@click.argument("message")
@click.option("--session", "session_id", default=None, help="Existing session ID to resume.")
@click.option("--session-title", default=None, help="Session title for Flow Music sidebar.")
@click.option("--ref-audio", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--ref-image", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--wait-clip/--no-wait-clip", default=True, show_default=True)
@click.option("--out-file", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--project-dir", type=click.Path(file_okay=False, path_type=Path), default=None)
@click.option("--shot", default="s01", show_default=True)
@click.option(
    "--format",
    "preferred_format",
    type=click.Choice(["wav", "m4a"], case_sensitive=False),
    default="wav",
    show_default=True,
)
@click.option("--timeout", "timeout_seconds", type=float, default=240.0, show_default=True)
@click.option("--analyze", is_flag=True, default=False)
@click.option("--master", is_flag=True, default=False)
@click.option("--target-lufs", type=float, default=-14.0, show_default=True)
@click.option("--profile", default=None, help="Profile name (overrides default).")
@output_format_option(has_out_file=True)
def music_chat(
    message: str,
    session_id: str | None,
    session_title: str | None,
    ref_audio: Path | None,
    ref_image: Path | None,
    wait_clip: bool,
    out_file: Path | None,
    project_dir: Path | None,
    shot: str,
    preferred_format: str,
    timeout_seconds: float,
    analyze: bool,
    master: bool,
    target_lufs: float,
    profile: str | None,
    as_json: bool,
    output_format: str = "table",
) -> None:
    """Chat with Flow Music Producer (new or resumed session) to refine or generate music."""
    fmt = _fmt(as_json, output_format)

    async def _run() -> None:
        res = await chat_flowmusic(
            _profile_dir(profile),
            message,
            session_id=session_id,
            session_title=session_title,
            ref_audio=ref_audio,
            ref_image=ref_image,
            wait_clip=wait_clip,
            out_file=out_file,
            project_dir=project_dir,
            shot=shot,
            preferred_format=preferred_format.lower(),
            timeout_seconds=timeout_seconds,
            analyze=analyze,
            master=master,
            target_lufs=target_lufs,
        )
        if fmt == OutputFormat.JSON.value:
            json_output.emit(res.to_dict())
        elif fmt == OutputFormat.ID.value:
            click.echo(res.session_id or "")
        else:
            console.print(f"Session [bold]{res.session_id or '-'}[/bold] ({res.session_url})")
            for dl in res.downloaded:
                console.print(f"  -> {dl.output_file} ({dl.format}, {dl.bytes:,} bytes)")

    run_with_handlers(_run, cli_command="music chat", as_json=fmt == "json", output_format=fmt)


@spends_credits(True)
@music.command("generate")
@click.argument("prompt")
@click.option("--session", "session_id", default=None, help="Resume Producer session (uses chat).")
@click.option("--lyrics", default=None, help="Optional song lyrics (Compose mode).")
@click.option("--instrumental", is_flag=True, default=False, help="Instrumental-only Compose mode.")
@click.option("--advanced-sound", is_flag=True, default=False)
@click.option("--model", default=None, help="Model name (e.g. 'Lyria 3.5').")
@click.option("--bpm", type=click.IntRange(40, 240), default=None, help="Target tempo in BPM.")
@click.option(
    "--length",
    default=None,
    help="Target track length in seconds (e.g. 90) or M:SS (clamped to FlowMusic min 1:00).",
)
@click.option("--seed", default=None, help="Optional generation seed.")
@click.option("--title", default=None, help="Native song title (defaults to '<slug> <shot>').")
@click.option("--session-title", default=None, help="Session title for Flow Music sidebar.")
@click.option("--ref-audio", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--ref-image", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out-file", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--project-dir", type=click.Path(file_okay=False, path_type=Path), default=None)
@click.option("--shot", default="s01", show_default=True)
@click.option(
    "--format",
    "preferred_format",
    type=click.Choice(["wav", "m4a"], case_sensitive=False),
    default="wav",
    show_default=True,
)
@click.option("--timeout", "timeout_seconds", type=float, default=240.0, show_default=True)
@click.option("--analyze", is_flag=True, default=False)
@click.option("--master", is_flag=True, default=False)
@click.option("--target-lufs", type=float, default=-14.0, show_default=True)
@click.option("--profile", default=None, help="Profile name (overrides default).")
@output_format_option(has_out_file=True)
def music_generate(
    prompt: str,
    session_id: str | None,
    lyrics: str | None,
    instrumental: bool,
    advanced_sound: bool,
    model: str | None,
    bpm: int | None,
    length: str | None,
    seed: str | None,
    title: str | None,
    session_title: str | None,
    ref_audio: Path | None,
    ref_image: Path | None,
    out_file: Path | None,
    project_dir: Path | None,
    shot: str,
    preferred_format: str,
    timeout_seconds: float,
    analyze: bool,
    master: bool,
    target_lufs: float,
    profile: str | None,
    as_json: bool,
    output_format: str = "table",
) -> None:
    """Bridge verb: defaults to Compose mode; routes to Producer chat when session/refs given."""
    fmt = _fmt(as_json, output_format)

    async def _run() -> None:
        res = await generate_flowmusic(
            _profile_dir(profile),
            prompt,
            session_id=session_id,
            lyrics=lyrics,
            instrumental=instrumental,
            advanced_sound=advanced_sound,
            model=model,
            bpm=bpm,
            length=length,
            seed=seed,
            title=title,
            session_title=session_title,
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
        if fmt == OutputFormat.JSON.value:
            json_output.emit(res.to_dict())
        elif fmt == OutputFormat.ID.value:
            click.echo(res.clip_id or str(res.output_file))
        else:
            console.print(
                f"Saved Flow Music track: [bold]{res.output_file}[/bold] "
                f"({res.format}, {res.bytes:,} bytes, via {res.capture_method})."
            )

    run_with_handlers(_run, cli_command="music generate", as_json=fmt == "json", output_format=fmt)


@spends_credits(False)
@music.command("list")
@click.option("--limit", type=int, default=20, show_default=True)
@click.option("--offset", type=int, default=0, show_default=True)
@click.option("--profile", default=None, help="Profile name (overrides default).")
@output_format_option()
def music_list(
    limit: int, offset: int, profile: str | None, as_json: bool, output_format: str = "table"
) -> None:
    """List generated Flow Music clips in your library."""
    fmt = _fmt(as_json, output_format)

    async def _run() -> None:
        clips = [
            c.to_dict()
            for c in await list_flowmusic_clips(_profile_dir(profile), limit=limit, offset=offset)
        ]
        _emit_resource_list(
            clips,
            title="Flow Music Clips",
            fmt=fmt,
            columns=(
                ("ID", "id"),
                ("Title", "title"),
                ("Dur(s)", "duration"),
                ("Created", "created_at"),
            ),
        )

    run_with_handlers(_run, cli_command="music list", as_json=fmt == "json", output_format=fmt)


@spends_credits(False)
@music.command("download")
@click.argument("clip_id")
@click.option("--out-file", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--project-dir", type=click.Path(file_okay=False, path_type=Path), default=None)
@click.option("--shot", default="s01", show_default=True)
@click.option(
    "--format",
    "preferred_format",
    type=click.Choice(["wav", "m4a"], case_sensitive=False),
    default="wav",
    show_default=True,
)
@click.option("--cover", is_flag=True, default=False, help="Also download cover artwork (.jpg).")
@click.option("--lyrics-timing", is_flag=True, default=False, help="Save lyrics_timing.json.")
@click.option("--profile", default=None, help="Profile name (overrides default).")
@output_format_option(has_out_file=True)
def music_download(
    clip_id: str,
    out_file: Path | None,
    project_dir: Path | None,
    shot: str,
    preferred_format: str,
    cover: bool,
    lyrics_timing: bool,
    profile: str | None,
    as_json: bool,
    output_format: str = "table",
) -> None:
    """Download an existing Flow Music clip (lossless WAV or M4A) and optional sidecars."""
    fmt = _fmt(as_json, output_format)

    async def _run() -> None:
        res = await download_flowmusic_clip(
            _profile_dir(profile),
            clip_id,
            out_file=out_file,
            project_dir=project_dir,
            shot=shot,
            preferred_format=preferred_format.lower(),
            include_cover=cover,
            include_lyrics_timing=lyrics_timing,
        )
        if fmt == OutputFormat.JSON.value:
            json_output.emit(res.to_dict())
        elif fmt == OutputFormat.ID.value:
            click.echo(str(res.output_file))
        else:
            console.print(f"Downloaded [bold]{res.output_file}[/bold] ({res.bytes:,} bytes)")

    run_with_handlers(_run, cli_command="music download", as_json=fmt == "json", output_format=fmt)


@spends_credits(False)
@music.command("sessions")
@click.option("--limit", type=int, default=30, show_default=True)
@click.option("--offset", type=int, default=0, show_default=True)
@click.option("--profile", default=None, help="Profile name (overrides default).")
@output_format_option()
def music_sessions(
    limit: int, offset: int, profile: str | None, as_json: bool, output_format: str = "table"
) -> None:
    """List Flow Music Producer conversation sessions."""
    fmt = _fmt(as_json, output_format)

    async def _run() -> None:
        items = [
            s.to_dict()
            for s in await list_flowmusic_sessions(
                _profile_dir(profile), limit=limit, offset=offset
            )
        ]
        _emit_resource_list(
            items,
            title="Flow Music Sessions",
            fmt=fmt,
            columns=(
                ("ID", "id"),
                ("Title", "title"),
                ("Kind", "kind"),
                ("Last msg", "last_message_at"),
            ),
        )

    run_with_handlers(_run, cli_command="music sessions", as_json=fmt == "json", output_format=fmt)


def _make_library_command(name: str, title: str, loader: Any) -> click.Command:
    @spends_credits(False)
    @click.command(name)
    @click.option("--limit", type=int, default=20, show_default=True)
    @click.option("--offset", type=int, default=0, show_default=True)
    @click.option("--profile", default=None, help="Profile name (overrides default).")
    @output_format_option()
    def _cmd(
        limit: int, offset: int, profile: str | None, as_json: bool, output_format: str = "table"
    ) -> None:
        fmt = _fmt(as_json, output_format)

        async def _run() -> None:
            items = [
                i.to_dict() for i in await loader(_profile_dir(profile), limit=limit, offset=offset)
            ]
            _emit_resource_list(
                items,
                title=title,
                fmt=fmt,
                columns=(("ID", "id"), ("Title", "title"), ("Created", "created_at")),
            )

        run_with_handlers(
            _run, cli_command=f"music {name}", as_json=fmt == "json", output_format=fmt
        )

    _cmd.help = f"List Flow Music {name}."
    return _cmd


music.add_command(
    _make_library_command("playlists", "Flow Music Playlists", list_flowmusic_playlists)
)
music.add_command(_make_library_command("projects", "Flow Music Projects", list_flowmusic_projects))
music.add_command(_make_library_command("spaces", "Flow Music Spaces", list_flowmusic_spaces))
music.add_command(_make_library_command("videos", "Flow Music Videos", list_flowmusic_videos))


@spends_credits(False)
@music.command("probe")
@click.option("--profile", default=None, help="Profile name (overrides default).")
@click.option("--url", default=FLOWMUSIC_SESSION_URL, show_default=True)
@output_format_option(allow_id=False)
def music_probe(profile: str | None, url: str, as_json: bool, output_format: str = "table") -> None:
    """Silently inspect Flow Music UI controls and observed API/audio URLs."""
    fmt = _fmt(as_json, output_format)

    async def _run() -> None:
        payload = await probe_flowmusic_ui(_profile_dir(profile), url=url)
        if fmt == OutputFormat.JSON.value:
            json_output.emit(payload)
        else:
            console.print(
                f"Probed {payload['url']} — signed_in={payload['signed_in']}, "
                f"account={payload.get('account')}, controls={payload['controls_count']}, "
                f"cached_apis={len(payload.get('cached_api_paths', []))}"
            )

    run_with_handlers(_run, cli_command="music probe", as_json=fmt == "json", output_format=fmt)


@spends_credits(False)
@music.command("rename-session")
@click.argument("session_id")
@click.argument("title")
@click.option("--profile", default=None, help="Profile name (overrides default).")
@output_format_option(allow_id=False)
def music_rename_session(
    session_id: str,
    title: str,
    profile: str | None,
    as_json: bool,
    output_format: str = "table",
) -> None:
    """Rename an existing Flow Music conversation session."""
    fmt = _fmt(as_json, output_format)

    async def _run() -> None:
        payload = await rename_flowmusic_session(_profile_dir(profile), session_id, title)
        if fmt == OutputFormat.JSON.value:
            json_output.emit(payload)
        elif payload.get("renamed"):
            console.print(f"Renamed session [bold]{session_id}[/bold] -> {payload['title']}")
        else:
            console.print(f"[yellow]Could not rename session {session_id}[/yellow]")

    run_with_handlers(
        _run, cli_command="music rename-session", as_json=fmt == "json", output_format=fmt
    )
