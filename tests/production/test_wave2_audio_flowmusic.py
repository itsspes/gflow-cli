"""Wave 2 Production Gate: FlowMusic Beat Analysis, 2-Pass -14 LUFS Mastering & 4:1 Ducking.

Exercises real FFmpeg/PyAV/DSP pipelines on synthetic rhythmic audio and clips:
- ``gflow audio analyze`` -> ``audio-map.json`` (BPM accuracy, downbeats, shotCutPlans)
- ``gflow audio master`` -> 2-pass EBU R128 ``-14 LUFS`` / ``<= -1.0 dBTP`` verified via ``ebur128``
- ``assemble_videos`` with upgraded 4:1 vocal-pocket ducking + ``master_lufs=-14.0`` + VideoToolbox
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from click.testing import CliRunner

from gflow_cli.cli import main
from gflow_cli.services.audio_mastering import verify_loudness
from gflow_cli.services.video_assembly import assemble_videos, assembly_result_payload


def _write_rhythmic_wav(
    path: Path,
    *,
    bpm: float = 120.0,
    duration_sec: float = 8.0,
    sample_rate: int = 44100,
) -> Path:
    """Synthesize a deterministic 4/4 rhythmic track (kick pulses + warm pad) at ``bpm``."""
    n = int(sample_rate * duration_sec)
    t = np.arange(n, dtype=np.float32) / sample_rate
    # Soft harmonic pad so integrated LUFS is well above silence
    pad = 0.08 * np.sin(2.0 * np.pi * 220.0 * t) + 0.04 * np.sin(2.0 * np.pi * 330.0 * t)
    signal = pad.copy()

    beat_interval = 60.0 / bpm
    beat_idx = 0
    while beat_idx * beat_interval < duration_sec:
        start = int(beat_idx * beat_interval * sample_rate)
        pulse_len = min(int(0.08 * sample_rate), n - start)
        if pulse_len > 0:
            pt = np.arange(pulse_len, dtype=np.float32) / sample_rate
            env = np.exp(-pt * 35.0)
            freq = 110.0 if (beat_idx % 4 == 0) else 165.0
            gain = 0.55 if (beat_idx % 4 == 0) else 0.35
            signal[start : start + pulse_len] += gain * env * np.sin(2.0 * np.pi * freq * pt)
        beat_idx += 1

    signal = np.clip(signal, -0.95, 0.95)
    pcm = (signal * 32767.0).astype(np.int16)
    stereo = np.column_stack([pcm, pcm]).ravel()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(stereo.tobytes())
    return path


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


# ---------------------------------------------------------------------------
# 1. FlowMusic Beat & Shot Cut-Point Analyzer (gflow audio analyze)
# ---------------------------------------------------------------------------


def test_audio_analyze_cli_produces_valid_audio_map_json(tmp_path: Path) -> None:
    runner = CliRunner()
    init_res = runner.invoke(
        main, ["workspace", "init", "music-sync", "--root", str(tmp_path), "--json"]
    )
    assert init_res.exit_code == 0, init_res.output
    project_dir = Path(json.loads(init_res.output)["project_dir"])

    wav = _write_rhythmic_wav(tmp_path / "flowmusic_120bpm.wav", bpm=120.0, duration_sec=8.0)
    res = runner.invoke(
        main,
        [
            "audio",
            "analyze",
            str(wav),
            "--project-dir",
            str(project_dir),
            "--shot-durations",
            "4,6,8",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.output
    payload: dict[str, Any] = json.loads(res.output)

    # Default output path under <project_dir>/audio/analysis/audio-map.json
    map_path = project_dir / "audio" / "analysis" / "audio-map.json"
    assert map_path.is_file()
    persisted = json.loads(map_path.read_text(encoding="utf-8"))
    assert persisted["bpm"] == payload["bpm"]

    # Tempo accuracy (allow half/double-time tolerance or exact 120 BPM within ±4 BPM)
    bpm = float(payload["bpm"])
    assert any(abs(bpm - target) <= 4.0 for target in (60.0, 120.0, 240.0)), (
        f"Unexpected BPM {bpm} for 120 BPM synthetic track"
    )
    assert float(payload["durationSec"]) == pytest.approx(8.0, abs=0.2)
    assert len(payload["beats"]) >= 4
    assert len(payload["downbeats"]) >= 1
    assert "4s" in payload["shotCutPlans"]
    assert "6s" in payload["shotCutPlans"]
    assert "8s" in payload["shotCutPlans"]
    assert isinstance(payload["recommendedCutPoints"], list)


# ---------------------------------------------------------------------------
# 2. 2-Pass EBU R128 Mastering (gflow audio master -> -14 LUFS / -1.0 dBTP)
# ---------------------------------------------------------------------------


def test_audio_master_cli_hits_minus14_lufs_and_true_peak_ceiling(tmp_path: Path) -> None:
    wav = _write_rhythmic_wav(tmp_path / "raw_music.wav", bpm=120.0, duration_sec=6.0)
    out_wav = tmp_path / "mastered_music.wav"

    res = CliRunner().invoke(
        main,
        [
            "audio",
            "master",
            str(wav),
            "--out-file",
            str(out_wav),
            "--target-lufs",
            "-14.0",
            "--true-peak",
            "-1.0",
            "--fade-out",
            "0.4",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.output
    assert out_wav.is_file()

    # Independent verification via FFmpeg ebur128 -> (integrated_lufs, true_peak_dbtp, lra_lu)
    measured_lufs, measured_tp, _lra = verify_loudness(out_wav)
    assert -15.5 <= measured_lufs <= -12.5, f"Expected ~-14 LUFS, got {measured_lufs:.2f}"
    assert measured_tp <= -0.7, f"Expected true peak <= -0.7 dBTP, got {measured_tp:.2f}"


def test_audio_master_with_voiceover_mix_produces_valid_output(
    assembly_media: dict[str, Path], tmp_path: Path
) -> None:
    out_m4a = tmp_path / "mixed_vo_music.m4a"
    res = CliRunner().invoke(
        main,
        [
            "audio",
            "master",
            str(assembly_media["music"]),
            "--voiceover",
            str(assembly_media["voice"]),
            "--out-file",
            str(out_m4a),
            "--target-lufs",
            "-14.0",
            "--true-peak",
            "-1.0",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.output
    assert out_m4a.is_file()
    probe = _ffprobe(out_m4a)
    a_streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "audio"]
    assert len(a_streams) == 1
    assert a_streams[0]["codec_name"] == "aac"


# ---------------------------------------------------------------------------
# 3. Upgraded Video Assembly: 4:1 Vocal-Pocket Ducking + -14 LUFS + VideoToolbox
# ---------------------------------------------------------------------------


def test_video_assemble_4to1_vocal_pocket_ducking_and_master_lufs(
    assembly_media: dict[str, Path], tmp_path: Path
) -> None:
    out_mp4 = tmp_path / "final_promo.mp4"
    result = assemble_videos(
        [assembly_media["clip"]],
        out_mp4,
        music=assembly_media["music"],
        voiceover=assembly_media["voice"],
        duck=True,
        strip_clip_audio=True,
        master_lufs=-14.0,
        prefer_hardware_encoder=(sys.platform == "darwin"),
    )
    assert out_mp4.is_file()
    assert result.ducking_applied is True
    assert result.vocal_pocket_eq_applied is True
    assert result.ducking_ratio == pytest.approx(4.0)
    assert result.clip_audio_stripped is True
    assert result.master_lufs_target == pytest.approx(-14.0)
    assert result.measured_lufs is not None
    assert -15.5 <= result.measured_lufs <= -12.5
    assert result.measured_true_peak_db is not None
    assert result.measured_true_peak_db <= -0.7

    payload = assembly_result_payload(result)
    assert payload["ducking_applied"] is True
    assert payload["vocal_pocket_eq_applied"] is True
    assert payload["ducking_ratio"] == pytest.approx(4.0)
    if sys.platform == "darwin":
        assert result.video_encoder in {"h264_videotoolbox", "libx264"}

    probe = _ffprobe(out_mp4)
    v_streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"]
    a_streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "audio"]
    assert len(v_streams) == 1
    assert v_streams[0]["codec_name"] == "h264"
    assert len(a_streams) == 1
    assert a_streams[0]["codec_name"] == "aac"
