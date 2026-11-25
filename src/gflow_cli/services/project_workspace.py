"""Project-scoped workspace management, asset naming, and SHA-256 Ingredients cache."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

WORKSPACE_SCHEMA_VERSION = 1
VALID_WORKSPACE_MODULES: tuple[str, ...] = ("image", "video", "music")
INGREDIENT_CATEGORIES: tuple[str, ...] = ("subject", "scene", "style")
ASSET_STATUSES: tuple[str, ...] = ("candidate", "approved", "minor_edit", "rejected")

_BASE_SUBDIRS: tuple[str, ...] = ("qa/reviews", "exports")
_MODULE_SUBDIRS: dict[str, tuple[str, ...]] = {
    "image": (
        "ingredients/subjects",
        "ingredients/scenes",
        "ingredients/styles",
        "images/candidates",
        "images/approved",
        "images/rejected",
    ),
    "video": (
        "ingredients/subjects",
        "ingredients/scenes",
        "ingredients/styles",
        "videos/drafts",
        "videos/masters",
        "videos/approved",
        "videos/rejected",
    ),
    "music": (
        "audio/music",
        "audio/voiceover",
        "audio/analysis",
        "audio/approved",
        "audio/rejected",
    ),
}

WORKSPACE_SUBDIRS: tuple[str, ...] = (
    "ingredients/subjects",
    "ingredients/scenes",
    "ingredients/styles",
    "images/candidates",
    "images/approved",
    "images/rejected",
    "videos/drafts",
    "videos/masters",
    "videos/approved",
    "videos/rejected",
    "audio/music",
    "audio/voiceover",
    "audio/analysis",
    "audio/approved",
    "audio/rejected",
    "qa/reviews",
    "exports",
)


def parse_workspace_modules(modules: Sequence[str] | str | None) -> tuple[str, ...]:
    """Normalize ``modules`` spec (e.g. ``'image,music'``, ``['video']``, ``'all'``, ``None``)."""
    if modules is None:
        return VALID_WORKSPACE_MODULES
    raw_tokens: list[str] = []
    if isinstance(modules, str):
        raw_tokens.extend(modules.split(","))
    else:
        for item in modules:
            raw_tokens.extend(item.split(","))
    cleaned = [t.strip().lower() for t in raw_tokens if t.strip()]
    if not cleaned or "all" in cleaned:
        return VALID_WORKSPACE_MODULES
    unknown = [t for t in cleaned if t not in VALID_WORKSPACE_MODULES]
    if unknown:
        msg = (
            f"Unknown workspace module(s): {', '.join(unknown)}. "
            f"Choose from {', '.join(VALID_WORKSPACE_MODULES)} or all."
        )
        raise ValueError(msg)
    return tuple(m for m in VALID_WORKSPACE_MODULES if m in cleaned)


def subdirs_for_modules(modules: Sequence[str]) -> tuple[str, ...]:
    """Return deduplicated directory list for the requested workspace modules."""
    ordered: list[str] = []
    for mod in modules:
        for rel in _MODULE_SUBDIRS.get(mod, ()):
            if rel not in ordered:
                ordered.append(rel)
    for rel in _BASE_SUBDIRS:
        if rel not in ordered:
            ordered.append(rel)
    return tuple(ordered)

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_TOKEN_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")


def normalize_token(value: str, *, field: str = "token") -> str:
    """Normalize and validate a lowercase kebab-case identifier."""
    cleaned = re.sub(r"[^a-z0-9-]+", "-", value.strip().lower()).strip("-")
    if not cleaned or not _TOKEN_RE.fullmatch(cleaned):
        msg = f"Invalid {field} {value!r}: must contain alphanumeric characters or hyphens."
        raise ValueError(msg)
    return cleaned


def validate_slug(slug: str) -> str:
    """Validate or normalize a project slug (lowercase kebab-case)."""
    cleaned = re.sub(r"[^a-z0-9-]+", "-", slug.strip().lower()).strip("-")
    if not cleaned or not _SLUG_RE.fullmatch(cleaned):
        msg = f"Invalid project slug {slug!r}: use lowercase letters, digits, and hyphens."
        raise ValueError(msg)
    return cleaned


def build_asset_filename(slug: str, shot: str, kind: str, version: int, ext: str) -> str:
    """Build deterministic asset name ``<slug>_<shot>_<kind>_v<NN>.<ext>``."""
    if version < 1:
        msg = f"Asset version must be >= 1, got {version}"
        raise ValueError(msg)
    s_slug = validate_slug(slug)
    s_shot = normalize_token(shot, field="shot")
    s_kind = normalize_token(kind, field="kind")
    s_ext = ext.lstrip(".").strip().lower()
    if not s_ext or not re.fullmatch(r"[a-z0-9]{1,10}", s_ext):
        msg = f"Invalid file extension {ext!r}"
        raise ValueError(msg)
    return f"{s_slug}_{s_shot}_{s_kind}_v{version:02d}.{s_ext}"


def next_asset_path(
    directory: Path,
    *,
    slug: str,
    shot: str,
    kind: str,
    ext: str,
) -> Path:
    """Return the next unused versioned asset path in ``directory``."""
    version = 1
    while True:
        candidate = directory / build_asset_filename(slug, shot, kind, version, ext)
        if not candidate.exists():
            return candidate
        version += 1


def sha256_file(path: Path) -> str:
    """Compute the SHA-256 hex digest of a local file in 64 KiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(65536):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path(project_dir: Path) -> Path:
    """Return ``<project_dir>/project.json``."""
    return project_dir.expanduser().resolve() / "project.json"


def load_manifest(project_dir: Path) -> dict[str, Any]:
    """Load and return ``project.json`` from ``project_dir``."""
    path = manifest_path(project_dir)
    if not path.is_file():
        msg = f"Not a gflow project workspace (missing {path}). Run `gflow workspace init <slug>`."
        raise FileNotFoundError(msg)
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def save_manifest(project_dir: Path, data: dict[str, Any]) -> Path:
    """Atomically write ``project.json`` into ``project_dir``."""
    path = manifest_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = datetime.now(UTC).isoformat()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def init_workspace(
    slug: str,
    *,
    root: Path | None = None,
    flow_project_id: str | None = None,
    modules: Sequence[str] | str | None = None,
) -> dict[str, Any]:
    """Create the project workspace tree (all or selected modules) and ``project.json``."""
    clean_slug = validate_slug(slug)
    resolved_modules = parse_workspace_modules(modules)
    base_root = (root or Path.cwd() / "projects").expanduser().resolve()
    project_dir = base_root / clean_slug
    project_dir.mkdir(parents=True, exist_ok=True)
    for rel in subdirs_for_modules(resolved_modules):
        (project_dir / rel).mkdir(parents=True, exist_ok=True)

    mpath = manifest_path(project_dir)
    if mpath.is_file():
        manifest = load_manifest(project_dir)
        changed = False
        if flow_project_id is not None:
            manifest["flow_project_id"] = flow_project_id
            changed = True
        if modules is not None:
            existing_mods = list(manifest.get("modules") or [])
            merged = [
                m for m in VALID_WORKSPACE_MODULES if m in existing_mods or m in resolved_modules
            ]
            manifest["modules"] = merged
            changed = True
        if changed:
            save_manifest(project_dir, manifest)
        return {
            "status": "existing",
            "slug": manifest["slug"],
            "project_dir": str(project_dir),
            "manifest_path": str(mpath),
            "flow_project_id": manifest.get("flow_project_id"),
            "modules": list(manifest.get("modules") or VALID_WORKSPACE_MODULES),
        }

    now = datetime.now(UTC).isoformat()
    manifest: dict[str, Any] = {
        "schema_version": WORKSPACE_SCHEMA_VERSION,
        "slug": clean_slug,
        "modules": list(resolved_modules),
        "flow_project_id": flow_project_id,
        "created_at": now,
        "updated_at": now,
        "ingredients": [],
        "upload_cache": {},
        "assets": {},
    }
    save_manifest(project_dir, manifest)
    log.info(
        "workspace.initialized",
        slug=clean_slug,
        modules=list(resolved_modules),
        project_dir=str(project_dir),
    )
    return {
        "status": "created",
        "slug": clean_slug,
        "project_dir": str(project_dir),
        "manifest_path": str(mpath),
        "flow_project_id": flow_project_id,
        "modules": list(resolved_modules),
    }


def add_ingredient(
    project_dir: Path,
    source: Path,
    *,
    category: str,
    name: str,
    copy_file: bool = True,
) -> dict[str, Any]:
    """Register a local reference asset in ``ingredients/`` with SHA-256 deduplication."""
    if category not in INGREDIENT_CATEGORIES:
        msg = f"Invalid category {category!r}; choose from {INGREDIENT_CATEGORIES}"
        raise ValueError(msg)
    src = source.expanduser().resolve()
    if not src.is_file():
        msg = f"Ingredient file does not exist: {src}"
        raise FileNotFoundError(msg)

    root = project_dir.expanduser().resolve()
    manifest = load_manifest(root)
    digest = sha256_file(src)
    ingredients: list[dict[str, Any]] = manifest.setdefault("ingredients", [])

    for existing in ingredients:
        if existing.get("sha256") == digest and existing.get("category") == category:
            return {"status": "duplicate", "deduplicated": True, "ingredient": existing}

    clean_name = normalize_token(name, field="name")
    ext = src.suffix.lstrip(".") or "png"
    target_dir = root / "ingredients" / f"{category}s"
    target_dir.mkdir(parents=True, exist_ok=True)
    if copy_file:
        dest = next_asset_path(
            target_dir, slug=manifest["slug"], shot=clean_name, kind=category, ext=ext
        )
        shutil.copy2(src, dest)
        stored_rel = str(dest.relative_to(root))
    else:
        stored_rel = str(src)

    entry: dict[str, Any] = {
        "name": clean_name,
        "category": category,
        "sha256": digest,
        "path": stored_rel,
        "added_at": datetime.now(UTC).isoformat(),
        "flow_uuids": {},
    }
    ingredients.append(entry)
    save_manifest(root, manifest)
    return {"status": "added", "deduplicated": False, "ingredient": entry}


def list_ingredients(
    project_dir: Path,
    *,
    category: str | None = None,
) -> list[dict[str, Any]]:
    """List registered Ingredients, optionally filtered by category."""
    if category is not None and category not in INGREDIENT_CATEGORIES:
        msg = f"Invalid category {category!r}; choose from {INGREDIENT_CATEGORIES}"
        raise ValueError(msg)
    manifest = load_manifest(project_dir)
    items: list[dict[str, Any]] = list(manifest.get("ingredients", []))
    if category is not None:
        items = [item for item in items if item.get("category") == category]
    return items


def cache_ingredient_uuid(
    project_dir: Path,
    *,
    sha256: str,
    flow_project_id: str,
    media_uuid: str,
) -> dict[str, Any]:
    """Store ``(flow_project_id, sha256) -> media_uuid`` in ``project.json``."""
    root = project_dir.expanduser().resolve()
    manifest = load_manifest(root)
    upload_cache: dict[str, dict[str, str]] = manifest.setdefault("upload_cache", {})
    proj_cache = upload_cache.setdefault(flow_project_id, {})
    proj_cache[sha256] = media_uuid

    matched = False
    for item in manifest.get("ingredients", []):
        if item.get("sha256") == sha256:
            item.setdefault("flow_uuids", {})[flow_project_id] = media_uuid
            matched = True

    save_manifest(root, manifest)
    return {
        "status": "cached",
        "flow_project_id": flow_project_id,
        "sha256": sha256,
        "media_uuid": media_uuid,
        "ingredient_matched": matched,
    }


def lookup_cached_uuid(
    project_dir: Path,
    *,
    flow_project_id: str,
    file_path: Path | None = None,
    sha256: str | None = None,
) -> str | None:
    """Look up a previously cached Flow media UUID by file or SHA-256 digest."""
    digest = sha256
    if digest is None:
        if file_path is None:
            msg = "Provide either file_path or sha256."
            raise ValueError(msg)
        digest = sha256_file(file_path.expanduser().resolve())
    manifest = load_manifest(project_dir)
    upload_cache: dict[str, dict[str, str]] = manifest.get("upload_cache", {})
    return upload_cache.get(flow_project_id, {}).get(digest)


def record_asset_status(
    project_dir: Path,
    *,
    asset_path: Path,
    status: str,
    notes: str = "",
    review_path: Path | None = None,
) -> dict[str, Any]:
    """Record or update an asset's review status in ``project.json``."""
    if status not in ASSET_STATUSES:
        msg = f"Invalid asset status {status!r}; choose from {ASSET_STATUSES}"
        raise ValueError(msg)
    root = project_dir.expanduser().resolve()
    manifest = load_manifest(root)
    resolved_asset = asset_path.expanduser().resolve()
    try:
        key = str(resolved_asset.relative_to(root))
    except ValueError:
        key = str(resolved_asset)
    record: dict[str, Any] = {
        "path": key,
        "status": status,
        "notes": notes,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    if review_path is not None:
        rev_resolved = review_path.expanduser().resolve()
        try:
            record["review_path"] = str(rev_resolved.relative_to(root))
        except ValueError:
            record["review_path"] = str(rev_resolved)
    assets: dict[str, dict[str, Any]] = manifest.setdefault("assets", {})
    assets[key] = record
    save_manifest(root, manifest)
    return record


def _scan_dir_files(root: Path, rel_dir: str) -> list[str]:
    folder = root / rel_dir
    if not folder.is_dir():
        return []
    return sorted(
        str(p.relative_to(root))
        for p in folder.iterdir()
        if p.is_file() and not p.name.startswith(".")
    )


def workspace_status(project_dir: Path) -> dict[str, Any]:
    """Summarize project workspace inventory grouped by review status."""
    root = project_dir.expanduser().resolve()
    manifest = load_manifest(root)
    by_status: dict[str, list[str]] = {status: [] for status in ASSET_STATUSES}
    tracked: set[str] = set()

    for rel_path, meta in sorted(manifest.get("assets", {}).items()):
        st = str(meta.get("status", "candidate"))
        if st in by_status:
            by_status[st].append(rel_path)
            tracked.add(rel_path)

    dir_mapping = (
        ("images/candidates", "candidate"),
        ("videos/drafts", "candidate"),
        ("videos/masters", "candidate"),
        ("audio/music", "candidate"),
        ("images/approved", "approved"),
        ("videos/approved", "approved"),
        ("audio/approved", "approved"),
        ("images/rejected", "rejected"),
        ("videos/rejected", "rejected"),
        ("audio/rejected", "rejected"),
    )
    for rel_dir, bucket in dir_mapping:
        for rel_file in _scan_dir_files(root, rel_dir):
            if rel_file not in tracked:
                by_status[bucket].append(rel_file)
                tracked.add(rel_file)

    upload_cache: dict[str, dict[str, str]] = manifest.get("upload_cache", {})
    cached_count = sum(len(entries) for entries in upload_cache.values())
    return {
        "slug": manifest["slug"],
        "project_dir": str(root),
        "modules": list(manifest.get("modules") or VALID_WORKSPACE_MODULES),
        "flow_project_id": manifest.get("flow_project_id"),
        "ingredients_count": len(manifest.get("ingredients", [])),
        "upload_cache_count": cached_count,
        "counts": {k: len(v) for k, v in by_status.items()},
        "assets_by_status": by_status,
    }
