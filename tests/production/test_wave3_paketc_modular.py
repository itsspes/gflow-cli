"""Wave 3 Production Gate: Paket C (M4 Local Whisper, Modularity & 5 Local MCP Twins).

Covers:
- Modular ``gflow workspace init --modules image|video|music|all`` directory isolation
- Standalone single-modality QA (``gflow image qa``, ``gflow video qa``, ``gflow audio qa``)
  working both WITHOUT ``--project-dir`` and WITH ``--project-dir``
- ``local_whisper.py`` backend detection (``darwin/arm64`` MLX priority), mocked ``mlx_whisper``
  word timestamps -> ``.srt`` & ``.json``, ``LocalWhisperUnavailableError`` remediation, and
  real FFmpeg 16 kHz mono WAV extraction from synthetic MP4 + WAV
- ``gflow audio transcribe`` & ``gflow video transcribe --backend mlx`` CLI contracts
- 5 local zero-credit MCP twins registered even under ``GFLOW_MCP_NO_SPEND=1``
"""

from __future__ import annotations

import json
import sys
import types
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image, ImageDraw

from gflow_cli.cli import main
from gflow_cli.services import local_whisper
from gflow_cli.services.local_whisper import (
    LocalWhisperUnavailableError,
    detect_local_whisper_backend,
    extract_audio_to_wav,
    transcribe_local,
)
from gflow_cli.services.project_workspace import load_manifest


def _write_sharp_png(path: Path) -> Path:
    img = Image.new("RGB", (320, 180), (235, 235, 235))
    draw = ImageDraw.Draw(img)
    for x in range(0, 320, 20):
        for y in range(0, 180, 20):
            if (x // 20 + y // 20) % 2 == 0:
                draw.rectangle([x, y, x + 19, y + 19], fill=(15, 35, 85))
    img.save(path, format="PNG")
    return path


def _write_wav(path: Path, *, duration_sec: float = 2.0, sr: int = 44100) -> Path:
    t = np.arange(int(sr * duration_sec), dtype=np.float32) / sr
    pcm = (0.25 * np.sin(2.0 * np.pi * 440.0 * t) * 32767.0).astype(np.int16)
    stereo = np.column_stack([pcm, pcm]).ravel()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(stereo.tobytes())
    return path


def _make_fake_mlx_whisper_module() -> types.ModuleType:
    mod = types.ModuleType("mlx_whisper")

    def _transcribe(_path: str, **_kwargs: Any) -> dict[str, Any]:
        return {
            "text": "Fresh roasted espresso.",
            "language": "en",
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 1.8,
                    "text": "Fresh roasted espresso.",
                    "words": [
                        {"word": "Fresh", "start": 0.0, "end": 0.45},
                        {"word": "roasted", "start": 0.46, "end": 1.0},
                        {"word": "espresso.", "start": 1.01, "end": 1.8},
                    ],
                }
            ],
        }

    mod.transcribe = _transcribe  # type: ignore[attr-defined]
    return mod


# ---------------------------------------------------------------------------
# 1. Modular Project Workspaces (--modules image|video|music|all)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("modules_arg", "expected_modules", "must_exist", "must_not_exist"),
    [
        (
            "image",
            ["image"],
            ["images/candidates", "ingredients/subjects", "qa/reviews", "exports"],
            ["videos", "audio"],
        ),
        (
            "video",
            ["video"],
            ["videos/drafts", "videos/approved", "ingredients/subjects", "qa/reviews"],
            ["images", "audio"],
        ),
        (
            "music",
            ["music"],
            ["audio/music", "audio/analysis", "audio/approved", "qa/reviews"],
            ["images", "videos", "ingredients"],
        ),
        (
            "image,music",
            ["image", "music"],
            ["images/candidates", "audio/music", "ingredients/subjects", "qa/reviews"],
            ["videos"],
        ),
        (
            "all",
            ["image", "video", "music"],
            ["images/candidates", "videos/drafts", "audio/music", "qa/reviews", "exports"],
            [],
        ),
    ],
)
def test_modular_workspace_init_directory_isolation_and_manifest(
    tmp_path: Path,
    modules_arg: str,
    expected_modules: list[str],
    must_exist: list[str],
    must_not_exist: list[str],
) -> None:
    slug = f"mod-{modules_arg.replace(',', '-')}"
    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "workspace",
            "init",
            slug,
            "--root",
            str(tmp_path),
            "--modules",
            modules_arg,
            "--json",
        ],
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["modules"] == expected_modules

    project_dir = Path(payload["project_dir"])
    manifest = load_manifest(project_dir)
    assert manifest["modules"] == expected_modules

    for rel in must_exist:
        assert (project_dir / rel).is_dir(), f"Expected {rel} for --modules {modules_arg}"
    for rel in must_not_exist:
        assert not (project_dir / rel).exists(), f"Unexpected {rel} for --modules {modules_arg}"

    st = runner.invoke(main, ["workspace", "status", "--project-dir", str(project_dir), "--json"])
    assert st.exit_code == 0, st.output
    assert json.loads(st.output)["modules"] == expected_modules


# ---------------------------------------------------------------------------
# 2. Standalone Single-Modality QA (WITH and WITHOUT --project-dir)
# ---------------------------------------------------------------------------


def test_standalone_and_project_single_modality_qa_commands(
    assembly_media: dict[str, Path], tmp_path: Path
) -> None:
    runner = CliRunner()
    png = _write_sharp_png(tmp_path / "hero.png")
    wav = _write_wav(tmp_path / "bed.wav")
    mp4 = assembly_media["clip"]

    # Standalone mode (NO --project-dir)
    r_img = runner.invoke(main, ["image", "qa", str(png), "--json"])
    assert r_img.exit_code == 0, r_img.output
    d_img = json.loads(r_img.output)
    assert d_img["kind"] == "image"
    assert d_img["verdict"] == "PASS_TECH"
    assert d_img.get("review_path") is None

    r_vid = runner.invoke(main, ["video", "qa", str(mp4), "--json"])
    assert r_vid.exit_code == 0, r_vid.output
    d_vid = json.loads(r_vid.output)
    assert d_vid["kind"] == "video"
    assert d_vid["metrics"]["has_unstripped_audio"] is True
    assert Path(d_vid["metrics"]["contact_sheet"]).is_file()

    r_aud = runner.invoke(main, ["audio", "qa", str(wav), "--json"])
    assert r_aud.exit_code == 0, r_aud.output
    d_aud = json.loads(r_aud.output)
    assert d_aud["kind"] == "audio"
    assert d_aud["verdict"] in {"PASS_TECH", "NEEDS_MINOR_EDIT"}

    # With --project-dir -> writes review.json into <project>/qa/reviews/
    init_res = runner.invoke(
        main, ["workspace", "init", "qa-mod", "--root", str(tmp_path), "--json"]
    )
    project_dir = Path(json.loads(init_res.output)["project_dir"])

    for subcmd, asset in (("image", png), ("video", mp4), ("audio", wav)):
        res = runner.invoke(
            main, [subcmd, "qa", str(asset), "--project-dir", str(project_dir), "--json"]
        )
        assert res.exit_code == 0, res.output
        data = json.loads(res.output)
        assert Path(data["review_path"]).is_file()


# ---------------------------------------------------------------------------
# 3. Local Whisper Engine (M4 MLX Priority, Word Timestamps, FFmpeg Extract)
# ---------------------------------------------------------------------------


def test_local_whisper_backend_detection_and_unavailable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Apple Silicon darwin/arm64 prefers mlx when available
    monkeypatch.setattr(local_whisper.sys, "platform", "darwin")
    monkeypatch.setattr(local_whisper.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        local_whisper,
        "_module_available",
        lambda name: name in {"mlx_whisper", "faster_whisper"},
    )
    assert detect_local_whisper_backend() == "mlx"

    # Non-Apple-Silicon falls back to faster_whisper
    monkeypatch.setattr(local_whisper.sys, "platform", "linux")
    monkeypatch.setattr(local_whisper.platform, "machine", lambda: "x86_64")
    assert detect_local_whisper_backend() == "faster_whisper"

    # None installed -> platform-aware LocalWhisperUnavailableError remediation
    monkeypatch.setattr(local_whisper, "_module_available", lambda _name: False)
    assert detect_local_whisper_backend() is None
    wav = _write_wav(tmp_path / "missing.wav")
    with pytest.raises(LocalWhisperUnavailableError) as linux_exc:
        transcribe_local(wav, backend="auto")
    assert "faster-whisper" in linux_exc.value.remediation_hint

    monkeypatch.setattr(local_whisper.sys, "platform", "darwin")
    monkeypatch.setattr(local_whisper.platform, "machine", lambda: "arm64")
    with pytest.raises(LocalWhisperUnavailableError) as mac_exc:
        transcribe_local(wav, backend="auto")
    assert "mlx-whisper" in mac_exc.value.remediation_hint


def test_local_whisper_extract_audio_from_mp4_and_transcribe_with_mocked_mlx(
    assembly_media: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Real FFmpeg extraction from synthetic MP4 -> valid mono 16 kHz PCM WAV
    extracted_wav = extract_audio_to_wav(assembly_media["clip"], tmp_path / "extracted.wav")
    assert extracted_wav.is_file()
    with wave.open(str(extracted_wav), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 16000
        assert wf.getnframes() > 0

    # Mocked mlx_whisper transcription -> word timestamps + .srt + .json
    fake_mlx = _make_fake_mlx_whisper_module()
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mlx)
    monkeypatch.setattr(local_whisper.sys, "platform", "darwin")
    monkeypatch.setattr(local_whisper.platform, "machine", lambda: "arm64")

    out_srt = tmp_path / "out.srt"
    out_json = tmp_path / "out.json"
    result = transcribe_local(
        assembly_media["clip"],
        backend="mlx",
        word_timestamps=True,
        out_srt=out_srt,
        out_json=out_json,
    )
    assert result.backend == "mlx"
    assert result.text == "Fresh roasted espresso."
    assert len(result.segments) == 1
    assert [w.word for w in result.segments[0].words] == ["Fresh", "roasted", "espresso."]
    assert out_srt.is_file()
    assert "00:00:00,000 --> 00:00:01,800" in out_srt.read_text(encoding="utf-8")
    assert out_json.is_file()
    assert json.loads(out_json.read_text(encoding="utf-8"))["backend"] == "mlx"


# ---------------------------------------------------------------------------
# 4. CLI Transcribe Contracts (gflow audio transcribe & gflow video transcribe)
# ---------------------------------------------------------------------------


def test_cli_audio_and_video_transcribe_local_mlx(
    assembly_media: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_mlx = _make_fake_mlx_whisper_module()
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mlx)
    monkeypatch.setattr(local_whisper.sys, "platform", "darwin")
    monkeypatch.setattr(local_whisper.platform, "machine", lambda: "arm64")

    runner = CliRunner()
    wav = _write_wav(tmp_path / "voice.wav")
    srt_a = tmp_path / "voice.srt"
    r_aud = runner.invoke(
        main,
        [
            "audio",
            "transcribe",
            str(wav),
            "--backend",
            "mlx",
            "--out-srt",
            str(srt_a),
            "--json",
        ],
    )
    assert r_aud.exit_code == 0, r_aud.output
    d_aud = json.loads(r_aud.output)
    assert d_aud["backend"] == "mlx"
    assert d_aud["text"] == "Fresh roasted espresso."
    assert srt_a.is_file()

    # gflow video transcribe --backend mlx
    srt_v = tmp_path / "clip.srt"
    r_vid = runner.invoke(
        main,
        [
            "video",
            "transcribe",
            str(assembly_media["clip"]),
            "--backend",
            "mlx",
            "--out-file",
            str(srt_v),
            "--json",
        ],
    )
    assert r_vid.exit_code == 0, r_vid.output
    d_vid = json.loads(r_vid.output)
    assert d_vid["backend"] == "mlx"
    assert srt_v.is_file()

    # gflow video qa --transcribe --whisper-backend mlx
    r_vqa = runner.invoke(
        main,
        [
            "video",
            "qa",
            str(assembly_media["clip"]),
            "--transcribe",
            "--whisper-backend",
            "mlx",
            "--json",
        ],
    )
    assert r_vqa.exit_code == 0, r_vqa.output
    d_vqa = json.loads(r_vqa.output)
    assert d_vqa["transcript"]["text"] == "Fresh roasted espresso."


# ---------------------------------------------------------------------------
# 5. 5 Local Zero-Credit MCP Twins Survive GFLOW_MCP_NO_SPEND=1
# ---------------------------------------------------------------------------


def test_five_local_qa_audio_mcp_twins_registered_under_no_spend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gflow_cli.mcp import server as server_mod

    monkeypatch.delenv("GFLOW_MCP_NO_SPEND", raising=False)
    server_mod._register_surfaces()  # pyright: ignore[reportPrivateUsage]
    mgr = server_mod.server._tool_manager  # pyright: ignore[reportPrivateUsage]
    saved = {n: mgr.get_tool(n) for n in server_mod._SPEND_TOOLS}  # pyright: ignore[reportPrivateUsage]

    local_twins = {
        "gflow_qa_check",
        "gflow_qa_decide",
        "gflow_audio_analyze",
        "gflow_audio_master",
        "gflow_local_transcribe",
    }
    try:
        monkeypatch.setenv("GFLOW_MCP_NO_SPEND", "1")
        server_mod._apply_no_spend_policy()  # pyright: ignore[reportPrivateUsage]
        names = set(mgr._tools.keys())  # pyright: ignore[reportPrivateUsage]
        assert local_twins.issubset(names), (
            f"Missing local twins under --no-spend: {local_twins - names}"
        )
    finally:
        for name, tool in saved.items():
            if tool is not None and mgr.get_tool(name) is None:
                mgr._tools[name] = tool  # pyright: ignore[reportPrivateUsage]
