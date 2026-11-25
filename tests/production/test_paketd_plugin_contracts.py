"""Paket D Production Gate: Cross-Harness Plugin Schema, Zero-Drift Sync & Hygiene Contracts.

Validates ``plugin/motionvideo/`` (Antigravity + Claude Code thin plugin bundle),
``plugin/sync.sh``, ``plugin/install.sh``, and byte-identical skill synchronization
with umbrella ``skills/*/SKILL.md``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE_ROOT = _REPO_ROOT.parent.parent
_UMBRELLA_SKILLS_DIR = _WORKSPACE_ROOT / "skills"
_PLUGIN_ROOT = _WORKSPACE_ROOT / "plugin"
_BUNDLE_DIR = _PLUGIN_ROOT / "motionvideo"

_ANTIGRAVITY_ALLOWED_KEYS = frozenset(
    {"name", "displayName", "version", "description", "author", "logo", "suggestedPrompts"}
)
_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
_SECRET_LEAK_RE = re.compile(
    r"(?:AIza[0-9A-Za-z\-_]{35}|sk-[A-Za-z0-9]{20,}|SAPISIDHASH\s+[0-9]+_[a-f0-9]{40})"
)


def test_antigravity_plugin_json_strict_schema_and_logo() -> None:
    manifest_path = _BUNDLE_DIR / "plugin.json"
    assert manifest_path.is_file()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Must contain ONLY valid Antigravity top-level keys (no unknown keys like license/keywords)
    unknown = set(data.keys()) - _ANTIGRAVITY_ALLOWED_KEYS
    assert not unknown, f"Unknown top-level keys in plugin.json: {unknown}"
    assert set(data.keys()) == _ANTIGRAVITY_ALLOWED_KEYS

    assert data["name"] == "motionvideo"
    assert data["displayName"]
    assert data["version"]
    assert data["description"]
    assert isinstance(data["author"], dict) and data["author"].get("name")
    assert isinstance(data["suggestedPrompts"], list) and len(data["suggestedPrompts"]) == 3
    assert all(isinstance(p, str) and p.strip() for p in data["suggestedPrompts"])

    logo_path = _BUNDLE_DIR / data["logo"]
    assert logo_path.is_file(), f"Missing plugin logo at {logo_path}"
    svg = logo_path.read_text(encoding="utf-8")
    assert "<svg" in svg and "viewBox" in svg


def test_claude_code_plugin_json_manifest() -> None:
    cc_path = _BUNDLE_DIR / ".claude-plugin" / "plugin.json"
    assert cc_path.is_file()
    data = json.loads(cc_path.read_text(encoding="utf-8"))
    assert data["name"] == "motionvideo"
    assert data["version"]
    assert data["description"]
    assert data["license"] == "MIT"
    assert isinstance(data["author"], dict) and data["author"].get("name")


def test_dual_mcp_configs_prefixed_portable_and_identical() -> None:
    ag_mcp = json.loads((_BUNDLE_DIR / "mcp_config.json").read_text(encoding="utf-8"))
    cc_mcp = json.loads((_BUNDLE_DIR / ".mcp.json").read_text(encoding="utf-8"))
    assert ag_mcp == cc_mcp

    servers = ag_mcp["mcpServers"]
    assert list(servers.keys()) == ["motionvideo-gflow"]
    srv = servers["motionvideo-gflow"]
    assert srv["command"] == "/bin/sh"
    assert "exec gflow mcp run" in " ".join(srv["args"])
    assert srv["env"]["GFLOW_CLI_WARM_SESSION_ENABLED"] == "1"
    assert srv["env"]["GFLOW_CLI_MACOS_BACKGROUND_LAUNCH"] == "1"


def test_rules_agents_md_is_plain_markdown_without_frontmatter() -> None:
    rules_path = _BUNDLE_DIR / "rules" / "AGENTS.md"
    assert rules_path.is_file()
    text = rules_path.read_text(encoding="utf-8")
    assert not text.startswith("---\n"), "rules/AGENTS.md must be plain markdown (no frontmatter)"
    assert "motionvideo" in text.lower() or "modular" in text.lower()


@pytest.mark.parametrize("agent_name", ["mv-director", "mv-critic"])
def test_plugin_agents_have_valid_frontmatter(agent_name: str) -> None:
    agent_path = _BUNDLE_DIR / "agents" / f"{agent_name}.md"
    assert agent_path.is_file()
    text = agent_path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(text)
    assert match is not None, f"{agent_path.name} must begin with YAML frontmatter"
    fm = match.group(1)
    assert f"name: {agent_name}" in fm
    assert "description:" in fm
    assert "tools:" in fm
    assert "mainAgent: true" in fm
    assert "subagent: true" in fm


def test_bundled_skills_are_byte_identical_to_umbrella_skills() -> None:
    umbrella = {p.parent.name: p for p in sorted(_UMBRELLA_SKILLS_DIR.glob("*/SKILL.md"))}
    bundled = {p.parent.name: p for p in sorted((_BUNDLE_DIR / "skills").glob("*/SKILL.md"))}
    assert set(bundled.keys()) == set(umbrella.keys()), (
        f"Skill set drift: umbrella={sorted(umbrella)}, bundled={sorted(bundled)}"
    )
    for name, src_path in umbrella.items():
        assert bundled[name].read_bytes() == src_path.read_bytes(), (
            f"Skill {name!r} drifted between skills/ and plugin/motionvideo/skills/; "
            "run ./plugin/sync.sh"
        )


def test_plugin_scripts_executable_valid_syntax_and_zero_drift() -> None:
    sync_sh = _PLUGIN_ROOT / "sync.sh"
    install_sh = _PLUGIN_ROOT / "install.sh"
    for script in (sync_sh, install_sh):
        assert script.is_file()
        assert os.access(script, os.X_OK), f"{script} must be executable (+x)"

    syntax = subprocess.run(
        ["bash", "-n", str(sync_sh), str(install_sh)],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert syntax.returncode == 0, syntax.stderr

    drift_check = subprocess.run(
        [str(sync_sh), "--check"],
        cwd=str(_WORKSPACE_ROOT),
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert drift_check.returncode == 0, drift_check.stdout + drift_check.stderr


def test_plugin_bundle_has_no_hardcoded_user_paths_or_secrets() -> None:
    for path in sorted(_PLUGIN_ROOT.rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert "/Users/" not in text, f"Hardcoded /Users/ path found in {path}"
        assert not _SECRET_LEAK_RE.search(text), f"Potential secret leak in {path}"
