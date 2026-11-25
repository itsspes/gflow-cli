"""CLI commands for the local warm-browser daemon (`gflow daemon start|status|stop`)."""

from __future__ import annotations

import subprocess
import sys

import click
from rich.console import Console

from gflow_cli import json_output
from gflow_cli._cli_helpers import OutputFormat, output_format_option, spends_credits
from gflow_cli.api.warm_daemon import (
    DEFAULT_DAEMON_HOST,
    DEFAULT_DAEMON_PORT,
    query_daemon_status,
    start_warm_daemon_sync,
    stop_warm_daemon,
)

console = Console()


@spends_credits(False)
@click.group("daemon")
def daemon() -> None:
    """Manage the local warm persistent browser daemon."""


@spends_credits(False)
@daemon.command("start")
@click.option("--host", default=DEFAULT_DAEMON_HOST, show_default=True, help="Loopback bind host.")
@click.option(
    "--port",
    type=click.IntRange(1, 65535),
    default=DEFAULT_DAEMON_PORT,
    show_default=True,
    help="Loopback bind port.",
)
@click.option(
    "--ttl",
    "ttl_seconds",
    type=click.FloatRange(10.0, 86400.0),
    default=None,
    help="Idle session TTL in seconds (defaults to GFLOW_CLI_WARM_SESSION_TTL_SECONDS).",
)
@click.option(
    "--background",
    is_flag=True,
    default=False,
    help="Launch the warm daemon in a detached background process.",
)
@output_format_option(allow_id=False)
def daemon_start(
    host: str,
    port: int,
    ttl_seconds: float | None,
    background: bool,
    as_json: bool,
    output_format: str = "table",
) -> None:
    """Start the local warm browser session daemon."""
    if background:
        cmd = [
            sys.executable,
            "-m",
            "gflow_cli",
            "daemon",
            "start",
            "--host",
            host,
            "--port",
            str(port),
        ]
        if ttl_seconds is not None:
            cmd.extend(["--ttl", str(ttl_seconds)])
        proc = subprocess.Popen(  # noqa: S603 - launching current Python executable
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        payload = {"status": "starting", "pid": proc.pid, "host": host, "port": port}
        if as_json or output_format == OutputFormat.JSON.value:
            json_output.emit(payload)
        else:
            console.print(f"Warm daemon starting in background (pid={proc.pid}, {host}:{port}).")
        return

    start_warm_daemon_sync(host=host, port=port, ttl_seconds=ttl_seconds)


@spends_credits(False)
@daemon.command("status")
@output_format_option(allow_id=False)
def daemon_status(as_json: bool, output_format: str = "table") -> None:
    """Show the status of the local warm browser daemon."""
    payload = query_daemon_status()
    if as_json or output_format == OutputFormat.JSON.value:
        json_output.emit(payload)
        return
    if payload.get("running"):
        console.print(
            f"Warm daemon is [green]{payload['status']}[/green] "
            f"(pid={payload.get('pid')}, {payload.get('host')}:{payload.get('port')})."
        )
    else:
        console.print("Warm daemon is [yellow]stopped[/yellow].")


@spends_credits(False)
@daemon.command("stop")
@output_format_option(allow_id=False)
def daemon_stop(as_json: bool, output_format: str = "table") -> None:
    """Stop the running local warm browser daemon."""
    payload = stop_warm_daemon()
    if as_json or output_format == OutputFormat.JSON.value:
        json_output.emit(payload)
        return
    if payload.get("stopped"):
        console.print(f"Stopped warm daemon (pid={payload.get('pid')}).")
    else:
        console.print("Warm daemon was not running.")
