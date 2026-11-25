"""CLI commands for project-scoped workspaces and SHA-256 Ingredients cache."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from gflow_cli import json_output
from gflow_cli._cli_helpers import OutputFormat, output_format_option, spends_credits
from gflow_cli.services.project_workspace import (
    INGREDIENT_CATEGORIES,
    add_ingredient,
    cache_ingredient_uuid,
    init_workspace,
    list_ingredients,
    sha256_file,
    workspace_status,
)

console = Console()


@spends_credits(False)
@click.group("workspace")
def workspace() -> None:
    """Manage project-scoped asset workspaces and SHA-256 Ingredients caches."""


@spends_credits(False)
@workspace.command("init")
@click.argument("slug")
@click.option(
    "--root",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Parent directory for projects (default: ./projects).",
)
@click.option("--flow-project", "flow_project_id", default=None, help="Associated Flow project ID.")
@click.option(
    "--modules",
    default="all",
    show_default=True,
    help="Comma-separated modalities to scaffold: image, video, music, or all.",
)
@output_format_option(allow_id=False)
def workspace_init(
    slug: str,
    root: Path | None,
    flow_project_id: str | None,
    modules: str,
    as_json: bool,
    output_format: str = "table",
) -> None:
    """Initialize a modular project workspace tree and project.json."""
    try:
        payload = init_workspace(slug, root=root, flow_project_id=flow_project_id, modules=modules)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    if as_json or output_format == OutputFormat.JSON.value:
        json_output.emit(payload)
        return
    mods = ",".join(payload.get("modules", []))
    console.print(
        f"Workspace [bold]{payload['slug']}[/bold] ({payload['status']}, modules={mods}): "
        f"{payload['project_dir']}"
    )


@spends_credits(False)
@workspace.command("status")
@click.option(
    "--project-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("."),
    show_default=True,
    help="Project workspace directory containing project.json.",
)
@output_format_option(allow_id=False)
def workspace_status_cmd(
    project_dir: Path,
    as_json: bool,
    output_format: str = "table",
) -> None:
    """Show asset counts and review statuses for a project workspace."""
    try:
        payload = workspace_status(project_dir)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc
    if as_json or output_format == OutputFormat.JSON.value:
        json_output.emit(payload)
        return
    table = Table(title=f"Workspace: {payload['slug']} ({', '.join(payload.get('modules', []))})")
    table.add_column("Bucket", style="bold")
    table.add_column("Count", justify="right")
    for bucket, count in payload["counts"].items():
        table.add_row(bucket, str(count))
    table.add_row("ingredients", str(payload["ingredients_count"]))
    table.add_row("cached_uuids", str(payload["upload_cache_count"]))
    console.print(table)


@spends_credits(False)
@workspace.group("ingredients")
def ingredients() -> None:
    """Manage project Ingredients (subjects, scenes, styles) and SHA-256 UUID cache."""


@spends_credits(False)
@ingredients.command("add")
@click.argument("file_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--category",
    type=click.Choice(list(INGREDIENT_CATEGORIES)),
    required=True,
    help="Ingredient category.",
)
@click.option("--name", required=True, help="Short kebab-case name for this ingredient.")
@click.option(
    "--project-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("."),
    show_default=True,
    help="Project workspace directory containing project.json.",
)
@output_format_option(allow_id=False)
def ingredients_add(
    file_path: Path,
    category: str,
    name: str,
    project_dir: Path,
    as_json: bool,
    output_format: str = "table",
) -> None:
    """Add a local reference file to ingredients/ with SHA-256 deduplication."""
    try:
        payload = add_ingredient(project_dir, file_path, category=category, name=name)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    if as_json or output_format == OutputFormat.JSON.value:
        json_output.emit(payload)
        return
    item = payload["ingredient"]
    console.print(
        f"Ingredient [bold]{item['name']}[/bold] ({payload['status']}): "
        f"{item['path']} [dim]sha256={item['sha256'][:12]}...[/dim]"
    )


@spends_credits(False)
@ingredients.command("list")
@click.option(
    "--category",
    type=click.Choice(list(INGREDIENT_CATEGORIES)),
    default=None,
    help="Filter by ingredient category.",
)
@click.option(
    "--project-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("."),
    show_default=True,
    help="Project workspace directory containing project.json.",
)
@output_format_option(allow_id=False)
def ingredients_list(
    category: str | None,
    project_dir: Path,
    as_json: bool,
    output_format: str = "table",
) -> None:
    """List registered Ingredients in the project workspace."""
    try:
        items = list_ingredients(project_dir, category=category)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    if as_json or output_format == OutputFormat.JSON.value:
        json_output.emit({"ingredients": items, "count": len(items)})
        return
    table = Table(title="Project Ingredients")
    table.add_column("Name", style="bold")
    table.add_column("Category")
    table.add_column("Path")
    table.add_column("SHA-256")
    for item in items:
        table.add_row(
            str(item["name"]),
            str(item["category"]),
            str(item["path"]),
            str(item["sha256"])[:12],
        )
    console.print(table)


@spends_credits(False)
@ingredients.command("cache-uuid")
@click.option("--file", "file_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--sha256", default=None, help="SHA-256 hex digest (alternative to --file).")
@click.option("--flow-project", "flow_project_id", required=True, help="Flow project ID.")
@click.option("--media-uuid", required=True, help="Uploaded Flow media UUID.")
@click.option(
    "--project-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("."),
    show_default=True,
    help="Project workspace directory containing project.json.",
)
@output_format_option(allow_id=False)
def ingredients_cache_uuid(
    file_path: Path | None,
    sha256: str | None,
    flow_project_id: str,
    media_uuid: str,
    project_dir: Path,
    as_json: bool,
    output_format: str = "table",
) -> None:
    """Cache a local file's SHA-256 -> Flow media UUID mapping in project.json."""
    if file_path is None and sha256 is None:
        raise click.UsageError("Provide either --file or --sha256.")
    digest = sha256 or sha256_file(file_path.expanduser().resolve())  # type: ignore[union-attr]
    try:
        payload = cache_ingredient_uuid(
            project_dir,
            sha256=digest,
            flow_project_id=flow_project_id,
            media_uuid=media_uuid,
        )
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc
    if as_json or output_format == OutputFormat.JSON.value:
        json_output.emit(payload)
        return
    console.print(
        f"Cached [bold]{digest[:12]}...[/bold] -> {media_uuid} (project {flow_project_id})."
    )
