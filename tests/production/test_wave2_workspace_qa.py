"""Wave 2 Production Gate: Project Workspace, SHA-256 Ingredients, Warm Daemon, --draft & QA Gate.

Covers:
- Project workspace initialization, naming grammar, and SHA-256 Ingredients UUID dedup
- ``--draft`` credit-saving preset and automatic ``videos/drafts/`` routing
- Warm daemon loopback ASGI endpoints, bearer auth, state file permissions, and macOS watcher
- Deterministic ``gflow qa check`` (image/video/audio + 1 fps strip) and ``gflow qa decide``
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
import wave
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image, ImageDraw

from gflow_cli import macos_window
from gflow_cli.api.warm_daemon import (
    DaemonState,
    build_warm_daemon_app,
    clear_daemon_state,
    daemon_state_path,
    read_daemon_state,
    write_daemon_state,
)
from gflow_cli.api.warm_session import WarmSessionManager
from gflow_cli.cli import main
from gflow_cli.cli_video import resolve_draft_preset
from gflow_cli.services.project_workspace import (
    WORKSPACE_SUBDIRS,
    build_asset_filename,
    lookup_cached_uuid,
    next_asset_path,
    sha256_file,
)
from gflow_cli.services.video_assembly import assemble_videos


def _write_sharp_png(path: Path) -> Path:
    """Create a high-contrast checkerboard PNG that scores high on edge sharpness."""
    img = Image.new("RGB", (320, 180), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    for x in range(0, 320, 20):
        for y in range(0, 180, 20):
            if (x // 20 + y // 20) % 2 == 0:
                draw.rectangle([x, y, x + 19, y + 19], fill=(20, 40, 90))
    img.save(path, format="PNG")
    return path


def _write_black_png(path: Path) -> Path:
    Image.new("RGB", (320, 180), (0, 0, 0)).save(path, format="PNG")
    return path


def _write_tone_wav(path: Path, *, duration_sec: float = 2.0, amplitude: float = 0.25) -> Path:
    sr = 44100
    t = np.arange(int(sr * duration_sec), dtype=np.float32) / sr
    samples = (amplitude * np.sin(2.0 * np.pi * 440.0 * t) * 32767.0).astype(np.int16)
    stereo = np.column_stack([samples, samples]).ravel()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(stereo.tobytes())
    return path


# ---------------------------------------------------------------------------
# 1. Project Workspace, Naming Grammar & SHA-256 Ingredients Dedup
# ---------------------------------------------------------------------------


def test_workspace_init_naming_and_status(tmp_path: Path) -> None:
    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "workspace",
            "init",
            "espresso-promo",
            "--root",
            str(tmp_path),
            "--flow-project",
            "proj-123",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data["status"] == "created"
    assert data["slug"] == "espresso-promo"
    assert data["flow_project_id"] == "proj-123"

    project_dir = Path(data["project_dir"])
    assert (project_dir / "project.json").is_file()
    for rel in WORKSPACE_SUBDIRS:
        assert (project_dir / rel).is_dir(), f"Missing workspace subdir {rel}"

    # Deterministic naming & auto-versioning
    assert build_asset_filename("espresso-promo", "s01", "draft", 1, "mp4") == (
        "espresso-promo_s01_draft_v01.mp4"
    )
    drafts = project_dir / "videos" / "drafts"
    p1 = next_asset_path(drafts, slug="espresso-promo", shot="s01", kind="draft", ext="mp4")
    assert p1.name == "espresso-promo_s01_draft_v01.mp4"
    p1.write_bytes(b"v1")
    p2 = next_asset_path(drafts, slug="espresso-promo", shot="s01", kind="draft", ext="mp4")
    assert p2.name == "espresso-promo_s01_draft_v02.mp4"

    # Workspace status JSON
    st = runner.invoke(main, ["workspace", "status", "--project-dir", str(project_dir), "--json"])
    assert st.exit_code == 0, st.output
    st_data = json.loads(st.output)
    assert st_data["slug"] == "espresso-promo"
    assert st_data["counts"]["candidate"] == 1


def test_ingredients_sha256_dedup_and_uuid_cache(tmp_path: Path) -> None:
    runner = CliRunner()
    init_res = runner.invoke(
        main, ["workspace", "init", "brand-launch", "--root", str(tmp_path), "--json"]
    )
    assert init_res.exit_code == 0, init_res.output
    project_dir = Path(json.loads(init_res.output)["project_dir"])

    hero_a = _write_sharp_png(tmp_path / "hero_a.png")
    hero_copy = tmp_path / "hero_copy.png"
    hero_copy.write_bytes(hero_a.read_bytes())

    # First add succeeds
    r1 = runner.invoke(
        main,
        [
            "workspace",
            "ingredients",
            "add",
            str(hero_a),
            "--category",
            "subject",
            "--name",
            "hero-cup",
            "--project-dir",
            str(project_dir),
            "--json",
        ],
    )
    assert r1.exit_code == 0, r1.output
    p1 = json.loads(r1.output)
    assert p1["status"] == "added"
    assert p1["deduplicated"] is False
    digest = p1["ingredient"]["sha256"]
    assert digest == sha256_file(hero_a)

    # Identical bytes from a different path deduplicate
    r2 = runner.invoke(
        main,
        [
            "workspace",
            "ingredients",
            "add",
            str(hero_copy),
            "--category",
            "subject",
            "--name",
            "hero-cup-dup",
            "--project-dir",
            str(project_dir),
            "--json",
        ],
    )
    assert r2.exit_code == 0, r2.output
    p2 = json.loads(r2.output)
    assert p2["status"] == "duplicate"
    assert p2["deduplicated"] is True

    # Cache Flow UUID and verify lookup by file bytes
    c_res = runner.invoke(
        main,
        [
            "workspace",
            "ingredients",
            "cache-uuid",
            "--file",
            str(hero_a),
            "--flow-project",
            "flow-proj-9",
            "--media-uuid",
            "uuid-abc-123",
            "--project-dir",
            str(project_dir),
            "--json",
        ],
    )
    assert c_res.exit_code == 0, c_res.output
    assert (
        lookup_cached_uuid(project_dir, flow_project_id="flow-proj-9", file_path=hero_copy)
        == "uuid-abc-123"
    )

    # Modified file misses cache
    modified = _write_black_png(tmp_path / "modified.png")
    assert (
        lookup_cached_uuid(project_dir, flow_project_id="flow-proj-9", file_path=modified) is None
    )


# ---------------------------------------------------------------------------
# 2. --draft Credit-Saving Preset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("subcmd", ["t2v", "i2v", "r2v"])
def test_video_commands_expose_draft_and_project_dir_flags(subcmd: str) -> None:
    res = CliRunner().invoke(main, ["video", subcmd, "--help"])
    assert res.exit_code == 0, res.output
    assert "--draft" in res.output
    assert "--project-dir" in res.output
    assert "--shot" in res.output


def test_resolve_draft_preset_defaults_and_workspace_routing(tmp_path: Path) -> None:
    runner = CliRunner()
    init_res = runner.invoke(
        main, ["workspace", "init", "draft-demo", "--root", str(tmp_path), "--json"]
    )
    assert init_res.exit_code == 0, init_res.output
    project_dir = Path(json.loads(init_res.output)["project_dir"])

    # Draft disabled -> passthrough
    assert resolve_draft_preset(
        draft=False,
        model=None,
        resolution=None,
        project_dir=project_dir,
        shot="s01",
        output_file=None,
        quiet=True,
    ) == (None, None, None)

    # Draft enabled -> defaults omni-1.2 + 360p + auto-routed draft path
    model, res, out_path = resolve_draft_preset(
        draft=True,
        model=None,
        resolution=None,
        project_dir=project_dir,
        shot="s02",
        output_file=None,
        quiet=True,
    )
    assert model == "omni-1.2"
    assert res == "360p"
    assert out_path == project_dir / "videos" / "drafts" / "draft-demo_s02_draft_v01.mp4"

    # Explicit overrides preserved
    m2, r2, o2 = resolve_draft_preset(
        draft=True,
        model="veo-lite",
        resolution="720p",
        project_dir=project_dir,
        shot="s02",
        output_file=tmp_path / "custom.mp4",
        quiet=True,
    )
    assert (m2, r2, o2) == ("veo-lite", "720p", tmp_path / "custom.mp4")


# ---------------------------------------------------------------------------
# 3. Warm Daemon Lifecycle & macOS Pre-Launch Hide Watcher
# ---------------------------------------------------------------------------


def test_daemon_state_roundtrip_and_posix_permissions() -> None:
    state = DaemonState(
        pid=os.getpid(),
        host="127.0.0.1",
        port=18765,
        token="secret-tok-123",
        ttl_seconds=600.0,
        started_at="2026-11-26T00:00:00+00:00",
    )
    written = write_daemon_state(state)
    assert written == daemon_state_path()
    if sys.platform != "win32":
        mode = stat.S_IMODE(written.stat().st_mode)
        assert mode == 0o600
    assert read_daemon_state() == state
    clear_daemon_state()
    assert read_daemon_state() is None


async def test_warm_daemon_asgi_auth_status_evict_and_shutdown() -> None:
    shutdown_called = False

    def _on_shutdown() -> None:
        nonlocal shutdown_called
        shutdown_called = True

    manager = WarmSessionManager(ttl_seconds=60.0)
    app = build_warm_daemon_app(manager, token="bearer-xyz", on_shutdown=_on_shutdown)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        # /health is unauthenticated
        h = await client.get("/health")
        assert h.status_code == 200
        assert h.json()["status"] == "ok"

        # /status requires bearer token
        unauth = await client.get("/status")
        assert unauth.status_code == 401

        auth_headers = {"Authorization": "Bearer bearer-xyz"}
        st = await client.get("/status", headers=auth_headers)
        assert st.status_code == 200
        assert st.json()["status"] == "running"

        ev = await client.post("/evict", headers=auth_headers)
        assert ev.status_code == 200
        assert ev.json()["evicted"] == 0

        sd = await client.post("/shutdown", headers=auth_headers)
        assert sd.status_code == 200
        assert sd.json()["status"] == "shutting_down"
        assert shutdown_called is True


async def test_macos_prelaunch_watcher_polls_during_launch_and_cancels_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(macos_window, "is_background_launch_active", lambda **_: True)
    monkeypatch.setattr(macos_window, "capture_frontmost_app", lambda: "Antigravity")
    hide_calls: list[str] = []
    restore_calls: list[str | None] = []

    monkeypatch.setattr(
        macos_window,
        "hide_chrome_processes",
        lambda: hide_calls.append("hide") or True,
    )
    monkeypatch.setattr(
        macos_window,
        "background_chrome_and_restore_focus",
        lambda app: restore_calls.append(app) or True,
    )

    async with macos_window.macos_background_launch_guard(headless=False, enabled=True):
        await asyncio.sleep(0.15)

    assert len(hide_calls) >= 1
    assert restore_calls == ["Antigravity"]


# ---------------------------------------------------------------------------
# 4. Deterministic Stage-1 Quality Gate (gflow qa check & decide)
# ---------------------------------------------------------------------------


def test_qa_check_image_sharp_vs_solid_black_and_decide_routing(tmp_path: Path) -> None:
    runner = CliRunner()
    init_res = runner.invoke(
        main, ["workspace", "init", "qa-demo", "--root", str(tmp_path), "--json"]
    )
    project_dir = Path(json.loads(init_res.output)["project_dir"])

    sharp = _write_sharp_png(project_dir / "images" / "candidates" / "qa-demo_s01_still_v01.png")
    black = _write_black_png(project_dir / "images" / "candidates" / "qa-demo_s02_still_v01.png")

    r_sharp = runner.invoke(
        main, ["qa", "check", str(sharp), "--project-dir", str(project_dir), "--json"]
    )
    assert r_sharp.exit_code == 0, r_sharp.output
    d_sharp = json.loads(r_sharp.output)
    assert d_sharp["verdict"] == "PASS_TECH"
    assert d_sharp["metrics"]["sharpness_score"] > 8.0
    assert Path(d_sharp["review_path"]).is_file()

    r_black = runner.invoke(
        main, ["qa", "check", str(black), "--project-dir", str(project_dir), "--json"]
    )
    assert r_black.exit_code == 0, r_black.output
    d_black = json.loads(r_black.output)
    assert d_black["verdict"] == "FAIL_TECH"
    assert any(i["code"] == "solid_black_frame" for i in d_black["issues"])

    # Decide approve -> moves to images/approved/
    dec_ok = runner.invoke(
        main,
        [
            "qa",
            "decide",
            str(sharp),
            "--verdict",
            "approve",
            "--notes",
            "crisp hero still",
            "--project-dir",
            str(project_dir),
            "--json",
        ],
    )
    assert dec_ok.exit_code == 0, dec_ok.output
    approved_dest = Path(json.loads(dec_ok.output)["destination"])
    assert approved_dest.parent == project_dir / "images" / "approved"
    assert approved_dest.is_file()
    assert not sharp.exists()

    # Decide reject -> moves to images/rejected/
    dec_rej = runner.invoke(
        main,
        [
            "qa",
            "decide",
            str(black),
            "--verdict",
            "reject",
            "--notes",
            "blank frame",
            "--project-dir",
            str(project_dir),
            "--json",
        ],
    )
    assert dec_rej.exit_code == 0, dec_rej.output
    rejected_dest = Path(json.loads(dec_rej.output)["destination"])
    assert rejected_dest.parent == project_dir / "images" / "rejected"
    assert rejected_dest.is_file()


def test_qa_check_video_with_and_without_audio_and_1fps_contact_strip(
    assembly_media: dict[str, Path], tmp_path: Path
) -> None:
    runner = CliRunner()
    init_res = runner.invoke(
        main, ["workspace", "init", "vid-qa", "--root", str(tmp_path), "--json"]
    )
    project_dir = Path(json.loads(init_res.output)["project_dir"])

    # 1. Raw clip WITH audio -> flags has_unstripped_audio + produces 1 fps contact strip
    r_raw = runner.invoke(
        main,
        [
            "qa",
            "check",
            str(assembly_media["clip"]),
            "--project-dir",
            str(project_dir),
            "--json",
        ],
    )
    assert r_raw.exit_code == 0, r_raw.output
    d_raw: dict[str, Any] = json.loads(r_raw.output)
    assert d_raw["kind"] == "video"
    assert d_raw["metrics"]["has_unstripped_audio"] is True
    assert any(i["code"] == "has_unstripped_audio" for i in d_raw["issues"])
    strip_path = Path(d_raw["metrics"]["contact_sheet"])
    assert strip_path.is_file()
    assert strip_path.name.endswith("_qa_strip.jpg")

    # 2. Stripped clip WITHOUT audio -> no has_unstripped_audio issue
    stripped = tmp_path / "stripped_broll.mp4"
    assemble_videos(
        [assembly_media["clip"]],
        stripped,
        strip_clip_audio=True,
        prefer_hardware_encoder=False,
    )
    r_clean = runner.invoke(
        main, ["qa", "check", str(stripped), "--project-dir", str(project_dir), "--json"]
    )
    assert r_clean.exit_code == 0, r_clean.output
    d_clean = json.loads(r_clean.output)
    assert d_clean["metrics"]["has_unstripped_audio"] is False
    assert not any(i["code"] == "has_unstripped_audio" for i in d_clean["issues"])


def test_qa_check_audio_integrates_with_audio_analysis(tmp_path: Path) -> None:
    wav = _write_tone_wav(tmp_path / "track.wav", duration_sec=2.0, amplitude=0.25)
    runner = CliRunner()
    res = runner.invoke(main, ["qa", "check", str(wav), "--kind", "audio", "--json"])
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data["kind"] == "audio"
    assert data["verdict"] in {"PASS_TECH", "NEEDS_MINOR_EDIT"}
    assert "integrated_lufs" in data["metrics"] or "lufs_integrated" in data["metrics"]


def test_qa_check_video_expect_audio_suppresses_room_tone_and_warns_when_silent(
    assembly_media: dict[str, Path], tmp_path: Path
) -> None:
    runner = CliRunner()

    # 1. Mastered video WITH audio + --expect-audio -> NO has_unstripped_audio warning
    r_ok = runner.invoke(
        main, ["video", "qa", str(assembly_media["clip"]), "--expect-audio", "--json"]
    )
    assert r_ok.exit_code == 0, r_ok.output
    d_ok = json.loads(r_ok.output)
    assert d_ok["metrics"]["expect_audio"] is True
    assert d_ok["metrics"]["has_unstripped_audio"] is False
    assert not any(i["code"] == "has_unstripped_audio" for i in d_ok["issues"])
    assert not any(i["code"] == "missing_expected_audio" for i in d_ok["issues"])

    # 2. Silent/stripped video WITHOUT audio + --expect-audio -> missing_expected_audio warn
    silent_mp4 = tmp_path / "silent_master.mp4"
    assemble_videos(
        [assembly_media["clip"]],
        silent_mp4,
        strip_clip_audio=True,
        prefer_hardware_encoder=False,
    )
    r_warn = runner.invoke(main, ["qa", "check", str(silent_mp4), "--expect-audio", "--json"])
    assert r_warn.exit_code == 0, r_warn.output
    d_warn = json.loads(r_warn.output)
    assert d_warn["metrics"]["expect_audio"] is True
    assert d_warn["verdict"] == "NEEDS_MINOR_EDIT"
    assert any(i["code"] == "missing_expected_audio" for i in d_warn["issues"])
