"""Tier 4: Skill & Prompt Contract Linting.

Validates YAML frontmatter, kebab-case naming, secret hygiene, and retired-model
compliance across umbrella ``skills/**/SKILL.md`` and core ``gflow-cli`` skills.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE_ROOT = _REPO_ROOT.parent.parent
_UMBRELLA_SKILLS_DIR = _WORKSPACE_ROOT / "skills"
_FORK_PROD_SKILLS = (
    _REPO_ROOT / "skills" / "gflow-cli" / "SKILL.md",
    _REPO_ROOT / "skills" / "video-production" / "SKILL.md",
)

_KEBAB_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
_RETIRED_MODEL_CLI_RE = re.compile(
    r"--model\s+(?:veo-fast|veo-quality|omni-1\.1|omni-flash)\b",
    re.IGNORECASE,
)
_RETIREMENT_CONTEXT_RE = re.compile(
    r"(?:retir|deprecat|avoid|never|do not|don't|legacy|leaving|replace|migrat|banned|forbid)",
    re.IGNORECASE,
)
_SECRET_LEAK_RE = re.compile(
    r"(?:AIza[0-9A-Za-z\-_]{35}|sk-[A-Za-z0-9]{20,}|SAPISIDHASH\s+[0-9]+_[a-f0-9]{40})"
)


def _umbrella_skill_files() -> list[Path]:
    if not _UMBRELLA_SKILLS_DIR.is_dir():
        return []
    return sorted(_UMBRELLA_SKILLS_DIR.glob("*/SKILL.md"))


def _all_production_skill_files() -> list[Path]:
    files = list(_umbrella_skill_files())
    files.extend(p for p in _FORK_PROD_SKILLS if p.is_file())
    return files


def _parse_frontmatter(text: str) -> dict[str, str]:
    match = _FRONTMATTER_RE.match(text)
    assert match is not None, "SKILL.md must begin with YAML frontmatter delimited by ---"
    parsed: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line and not line.startswith(" "):
            key, _, val = line.partition(":")
            parsed[key.strip()] = val.strip().strip('"').strip("'")
    return parsed


def test_umbrella_skills_exist() -> None:
    files = _umbrella_skill_files()
    names = {p.parent.name for p in files}
    expected = {
        "mv-flow-prompting",
        "mv-flowmusic-sync",
        "mv-quality-gate",
        "mv-project-workspace",
    }
    missing = expected - names
    assert not missing, f"Missing expected umbrella skills under {_UMBRELLA_SKILLS_DIR}: {missing}"


@pytest.mark.parametrize(
    "skill_path",
    _all_production_skill_files(),
    ids=lambda p: f"{p.parent.parent.name}/{p.parent.name}",
)
def test_skill_frontmatter_and_no_secret_leaks(skill_path: Path) -> None:
    text = skill_path.read_text(encoding="utf-8")
    fm = _parse_frontmatter(text)
    assert fm.get("name"), f"{skill_path}: frontmatter missing non-empty 'name'"
    assert fm.get("description"), f"{skill_path}: frontmatter missing non-empty 'description'"
    assert _KEBAB_RE.fullmatch(fm["name"]), (
        f"{skill_path}: skill name {fm['name']!r} must be kebab-case"
    )
    if skill_path.is_relative_to(_UMBRELLA_SKILLS_DIR):
        assert fm["name"] == skill_path.parent.name, (
            f"{skill_path}: frontmatter name {fm['name']!r} must match directory "
            f"{skill_path.parent.name!r}"
        )
    assert not _SECRET_LEAK_RE.search(text), f"{skill_path}: potential secret/token leak detected"


@pytest.mark.parametrize(
    "skill_path",
    _umbrella_skill_files(),
    ids=lambda p: p.parent.name,
)
def test_umbrella_skills_never_recommend_retired_2026_12_05_models(skill_path: Path) -> None:
    """Umbrella skills must route new runs to omni-1.2-flash / veo-lite, never retired models."""
    lines = skill_path.read_text(encoding="utf-8").splitlines()
    offenders: list[tuple[int, str]] = []
    for idx, line in enumerate(lines, start=1):
        if _RETIRED_MODEL_CLI_RE.search(line) and not _RETIREMENT_CONTEXT_RE.search(line):
            offenders.append((idx, line.strip()))
    assert not offenders, (
        f"{skill_path} recommends retired 2026-12-05 models outside a warning/migration context: "
        f"{offenders}"
    )
