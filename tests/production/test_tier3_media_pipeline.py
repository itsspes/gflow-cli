"""Tier 3: Local Media Pipeline & FFmpeg/VideoToolbox Smoke Tests.

Exercises real synthetic clip assembly, Veo room-tone audio stripping
(``--strip-veo-audio`` / ``strip_clip_audio``), macOS Apple Silicon
``h264_videotoolbox`` selection & fallback, and ``ffprobe`` format verification.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from gflow_cli.cli import main
from gflow_cli.services.video_assembly import (
    assemble_videos,
    assembly_result_payload,
    select_video_encoder,
)


def _ffprobe(path: Path) -> dict[str, Any]:
    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin is None:
        pytest.skip("ffprobe not installed on PATH")
    proc = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return json.loads(proc.stdout)


def _streams_by_type(probe: dict[str, Any], codec_type: str) -> list[dict[str, Any]]:
    return [s for s in probe.get("streams", []) if s.get("codec_type") == codec_type]


# ---------------------------------------------------------------------------
# 1. Encoder Selection & Hardware Fallback
# ---------------------------------------------------------------------------


def test_select_video_encoder_darwin_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    import av

    # Explicit software preference always selects libx264
    assert select_video_encoder(av, prefer_hardware=False) == "libx264"

    # Non-darwin always selects libx264
    monkeypatch.setattr("gflow_cli.services.video_assembly.sys.platform", "linux")
    assert select_video_encoder(av, prefer_hardware=True) == "libx264"

    # On darwin, if Codec("h264_videotoolbox", "w") fails, gracefully falls back to libx264
    monkeypatch.setattr("gflow_cli.services.video_assembly.sys.platform", "darwin")

    class _FakeCodecModule:
        @staticmethod
        def Codec(_name: str, _mode: str) -> object:  # noqa: N802 - mirrors PyAV av.codec.Codec
            raise RuntimeError("no videotoolbox")

    class _FakeAv:
        codec = _FakeCodecModule()

    assert select_video_encoder(_FakeAv(), prefer_hardware=True) == "libx264"


@pytest.mark.skipif(sys.platform != "darwin", reason="Apple VideoToolbox is macOS-only")
def test_macos_videotoolbox_assembly_smoke_verified_with_ffprobe(
    assembly_media: dict[str, Path], tmp_path: Path
) -> None:
    out_file = tmp_path / "vt_smoke.mp4"
    result = assemble_videos(
        [assembly_media["clip"]],
        out_file,
        prefer_hardware_encoder=True,
    )
    assert out_file.exists()
    assert out_file.stat().st_size > 0
    assert result.video_encoder in {"h264_videotoolbox", "libx264"}
    assert result.has_audio_stream is True

    probe = _ffprobe(out_file)
    v_streams = _streams_by_type(probe, "video")
    a_streams = _streams_by_type(probe, "audio")
    assert len(v_streams) == 1
    assert v_streams[0]["codec_name"] == "h264"
    assert int(v_streams[0]["width"]) == 1280
    assert int(v_streams[0]["height"]) == 720
    assert float(probe["format"]["duration"]) >= 3.5
    assert len(a_streams) == 1
    assert a_streams[0]["codec_name"] == "aac"


# ---------------------------------------------------------------------------
# 2. Veo Room-Tone Audio Stripping (--strip-veo-audio)
# ---------------------------------------------------------------------------


def test_strip_clip_audio_without_music_omits_audio_stream_entirely(
    assembly_media: dict[str, Path], tmp_path: Path
) -> None:
    out_file = tmp_path / "silent_broll.mp4"
    result = assemble_videos(
        [assembly_media["clip"], assembly_media["clip"]],
        out_file,
        strip_clip_audio=True,
        prefer_hardware_encoder=False,
    )
    assert result.clip_audio_stripped is True
    assert result.has_audio_stream is False
    assert result.video_encoder == "libx264"

    payload = assembly_result_payload(result)
    assert payload["clip_audio_stripped"] is True
    assert payload["audio_stream"] is None
    assert payload["video_stream"]["encoder"] == "libx264"

    probe = _ffprobe(out_file)
    v_streams = _streams_by_type(probe, "video")
    a_streams = _streams_by_type(probe, "audio")
    assert len(v_streams) == 1
    assert len(a_streams) == 0, "Expected 0 audio streams when stripping clip audio without music"
    assert float(probe["format"]["duration"]) >= 7.5


def test_strip_clip_audio_with_music_preserves_music_audio_stream(
    assembly_media: dict[str, Path], tmp_path: Path
) -> None:
    out_file = tmp_path / "broll_with_music.mp4"
    result = assemble_videos(
        [assembly_media["clip"]],
        out_file,
        music=assembly_media["music"],
        strip_clip_audio=True,
        prefer_hardware_encoder=False,
    )
    assert result.clip_audio_stripped is True
    assert result.has_audio_stream is True

    probe = _ffprobe(out_file)
    a_streams = _streams_by_type(probe, "audio")
    assert len(a_streams) == 1
    assert a_streams[0]["codec_name"] == "aac"
    assert int(a_streams[0]["sample_rate"]) in {44100, 48000}


def test_cli_video_assemble_strip_veo_audio_json_end_to_end(
    assembly_media: dict[str, Path], tmp_path: Path
) -> None:
    out_file = tmp_path / "cli_stripped.mp4"
    result = CliRunner().invoke(
        main,
        [
            "video",
            "assemble",
            str(assembly_media["clip"]),
            "--strip-veo-audio",
            "--out-file",
            str(out_file),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["clip_audio_stripped"] is True
    assert payload["audio_stream"] is None
    assert Path(payload["output_file"]) == out_file

    probe = _ffprobe(out_file)
    assert _streams_by_type(probe, "audio") == []
    assert len(_streams_by_type(probe, "video")) == 1
