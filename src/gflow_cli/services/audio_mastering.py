"""FFmpeg 9.0.2 audio mastering, 2-pass EBU R128 loudnorm, vocal-pocket EQ, and ducking helpers.

Deliberately stdlib-only (no NumPy import) so video assembly stays NumPy-free.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import structlog

from gflow_cli.errors import VideoAssemblyError

_log = structlog.get_logger(__name__)

DEFAULT_TARGET_LUFS: float = -14.0
DEFAULT_TRUE_PEAK_DB: float = -1.0
DEFAULT_LRA_LU: float = 11.0

DEFAULT_DUCKING_THRESHOLD: float = 0.01
DEFAULT_DUCKING_RATIO: float = 4.0
DEFAULT_DUCKING_ATTACK_MS: float = 20.0
DEFAULT_DUCKING_RELEASE_MS: float = 350.0
DEFAULT_DUCKING_KNEE: float = 4.0

_SILENCE_LUFS_FLOOR = -70.0
_SILENCE_PEAK_FLOOR = -99.0

_EBUR128_I_RE = re.compile(r"^\s*I:\s*([-+]?\d+(?:\.\d+)?|-?inf)\s*LUFS", re.MULTILINE)
_EBUR128_LRA_RE = re.compile(r"^\s*LRA:\s*([-+]?\d+(?:\.\d+)?|-?inf)\s*LU", re.MULTILINE)
_EBUR128_PEAK_RE = re.compile(r"^\s*Peak:\s*([-+]?\d+(?:\.\d+)?|-?inf)\s*dBFS", re.MULTILINE)
_JSON_BLOCK_RE = re.compile(r"\{[^{}]*\"input_i\"[^{}]*\}", re.DOTALL)

_SUPPORTED_AUDIO_CODECS: dict[str, tuple[str, list[str]]] = {
    ".wav": ("pcm_s24le", ["-c:a", "pcm_s24le"]),
    ".mp3": ("libmp3lame", ["-c:a", "libmp3lame"]),
    ".m4a": ("aac", ["-c:a", "aac"]),
    ".aac": ("aac", ["-c:a", "aac"]),
    ".flac": ("flac", ["-c:a", "flac"]),
    ".ogg": ("libopus", ["-c:a", "libopus"]),
    ".opus": ("libopus", ["-c:a", "libopus"]),
}
_LOSSY_SUFFIXES: frozenset[str] = frozenset({".mp3", ".m4a", ".aac", ".ogg", ".opus"})


@dataclass(frozen=True)
class LoudnormMeasurement:
    """Pass-1 FFmpeg ``loudnorm`` EBU R128 measurements."""

    input_i: float
    input_tp: float
    input_lra: float
    input_thresh: float
    target_offset: float
    is_finite: bool


@dataclass(frozen=True)
class MasteringResult:
    """Structured outcome from 2-pass EBU R128 audio mastering and optional VO ducking."""

    input_file: Path
    output_file: Path
    target_lufs: float
    target_true_peak_db: float
    target_lra_lu: float
    measured_input_lufs: float
    measured_input_true_peak_db: float
    verified_output_lufs: float
    verified_output_true_peak_db: float
    verified_output_lra_lu: float
    duration_sec: float
    sample_rate: int
    codec: str
    linear_mode: bool
    fade_in_sec: float
    fade_out_sec: float
    has_voiceover: bool
    vocal_pocket_eq_applied: bool
    ducking_applied: bool
    bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "succeeded",
            "input_file": str(self.input_file),
            "output_file": str(self.output_file),
            "target_lufs": round(self.target_lufs, 2),
            "target_true_peak_db": round(self.target_true_peak_db, 2),
            "target_lra_lu": round(self.target_lra_lu, 2),
            "measured_input_lufs": round(self.measured_input_lufs, 2),
            "measured_input_true_peak_db": round(self.measured_input_true_peak_db, 2),
            "verified_output_lufs": round(self.verified_output_lufs, 2),
            "verified_output_true_peak_db": round(self.verified_output_true_peak_db, 2),
            "verified_output_lra_lu": round(self.verified_output_lra_lu, 2),
            "duration_sec": round(self.duration_sec, 3),
            "sample_rate": self.sample_rate,
            "codec": self.codec,
            "linear_mode": self.linear_mode,
            "fade_in_sec": round(self.fade_in_sec, 3),
            "fade_out_sec": round(self.fade_out_sec, 3),
            "has_voiceover": self.has_voiceover,
            "vocal_pocket_eq_applied": self.vocal_pocket_eq_applied,
            "ducking_applied": self.ducking_applied,
            "bytes": self.bytes,
        }


def build_ducking_filter(
    *,
    threshold: float = DEFAULT_DUCKING_THRESHOLD,
    ratio: float = DEFAULT_DUCKING_RATIO,
    attack_ms: float = DEFAULT_DUCKING_ATTACK_MS,
    release_ms: float = DEFAULT_DUCKING_RELEASE_MS,
    knee: float = DEFAULT_DUCKING_KNEE,
) -> str:
    """Build a smooth FFmpeg ``sidechaincompress`` parameter string (default gentle 4:1)."""
    return (
        f"threshold={threshold:g}:ratio={ratio:g}:attack={attack_ms:g}"
        f":release={release_ms:g}:knee={knee:g}"
    )


def vocal_pocket_eq_specs(
    *,
    low_mid_db: float = -2.0,
    mid_db: float = -3.5,
    presence_db: float = -2.5,
) -> tuple[tuple[str, str], ...]:
    """Return ``(filter_name, args)`` pairs carving a gentle 300 Hz–3 kHz vocal pocket."""
    return (
        ("equalizer", f"f=600:t=q:w=1.0:g={low_mid_db:g}"),
        ("equalizer", f"f=1400:t=q:w=1.0:g={mid_db:g}"),
        ("equalizer", f"f=2800:t=q:w=1.2:g={presence_db:g}"),
    )


def build_vocal_pocket_eq_chain(
    *,
    low_mid_db: float = -2.0,
    mid_db: float = -3.5,
    presence_db: float = -2.5,
) -> str:
    """Build an FFmpeg audio filter chain string for gentle vocal-pocket EQ carving."""
    return ",".join(
        f"{name}={args}"
        for name, args in vocal_pocket_eq_specs(
            low_mid_db=low_mid_db, mid_db=mid_db, presence_db=presence_db
        )
    )


def build_trim_fade_filter(
    *,
    duration_sec: float | None = None,
    fade_in_sec: float = 0.05,
    fade_out_sec: float = 0.0,
    highpass_hz: float | None = 30.0,
) -> str:
    """Build high-pass + smooth anti-click fade-in + exponential fade-out FFmpeg filter chain."""
    parts: list[str] = []
    if highpass_hz is not None and highpass_hz > 0.0:
        parts.append(f"highpass=f={highpass_hz:g}")
    if fade_in_sec > 0.0:
        parts.append(f"afade=t=in:st=0:d={fade_in_sec:.3f}:curve=tri")
    if fade_out_sec > 0.0 and duration_sec is not None and duration_sec > 0.0:
        effective_fade = min(fade_out_sec, duration_sec)
        fade_start = max(0.0, duration_sec - effective_fade)
        parts.append(f"afade=t=out:st={fade_start:.3f}:d={effective_fade:.3f}:curve=exp")
    return ",".join(parts)


def build_vo_mix_filter_complex(
    *,
    music_volume: float = 0.32,
    duck_threshold: float = DEFAULT_DUCKING_THRESHOLD,
    duck_ratio: float = DEFAULT_DUCKING_RATIO,
    duck_attack_ms: float = DEFAULT_DUCKING_ATTACK_MS,
    duck_release_ms: float = DEFAULT_DUCKING_RELEASE_MS,
    duck_knee: float = DEFAULT_DUCKING_KNEE,
    music_trim_fade: str = "",
) -> str:
    """Build an FFmpeg ``-filter_complex`` mixing music ``[0:a]`` under voiceover ``[1:a]``.

    Applies vocal-pocket EQ (600 Hz / 1.4 kHz / 2.8 kHz) on the music bed and smooth 4:1
    sidechain ducking keyed by the leveled voiceover track.
    """
    eq_chain = build_vocal_pocket_eq_chain()
    music_chain_parts = [p for p in (music_trim_fade, eq_chain, f"volume={music_volume:g}") if p]
    music_chain = ",".join(music_chain_parts)
    duck_params = build_ducking_filter(
        threshold=duck_threshold,
        ratio=duck_ratio,
        attack_ms=duck_attack_ms,
        release_ms=duck_release_ms,
        knee=duck_knee,
    )
    return (
        f"[1:a]highpass=f=80,"
        f"acompressor=threshold=-18dB:ratio=3:attack=10:release=120:makeup=2dB,"
        f"asplit=2[vo_mix][vo_sc];"
        f"[0:a]{music_chain}[mus_eq];"
        f"[mus_eq][vo_sc]sidechaincompress={duck_params}[mus_ducked];"
        f"[vo_mix][mus_ducked]amix=inputs=2:duration=longest:dropout_transition=2:normalize=0,"
        f"alimiter=limit=-1.0dB[out]"
    )


def _find_binary(name: str, *, route: str = "audio.master") -> str:
    binary = shutil.which(name)
    if binary is None:
        raise VideoAssemblyError(
            detail=f"Required binary {name!r} was not found on PATH.",
            route=route,
            charged="no",
            remediation_hint=f"Install FFmpeg (`brew install ffmpeg`) so `{name}` is available.",
        )
    return binary


def _parse_float_or(raw: str | None, fallback: float) -> float:
    if raw is None:
        return fallback
    try:
        val = float(raw)
    except ValueError:
        return fallback
    return val if math.isfinite(val) else fallback


def _probe_duration(path: Path) -> float:
    ffprobe = _find_binary("ffprobe")
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(  # noqa: S603 - resolved binary path, fixed argv list, no shell
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if proc.returncode != 0:
        raise VideoAssemblyError(
            detail=f"ffprobe failed for {path}: {proc.stderr.strip() or 'unreadable file'}",
            route="audio.master",
            charged="no",
        )
    data = cast("dict[str, Any]", json.loads(proc.stdout or "{}"))
    fmt = cast("dict[str, Any]", data.get("format") or {})
    return max(0.0, _parse_float_or(str(fmt.get("duration") or "0"), 0.0))


def verify_loudness(path: Path | str) -> tuple[float, float, float]:
    """Measure ``(integrated_lufs, true_peak_db, lra_lu)`` using FFmpeg ``ebur128``."""
    resolved = Path(path).expanduser().resolve()
    ffmpeg = _find_binary("ffmpeg")
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-i",
        str(resolved),
        "-vn",
        "-af",
        "ebur128=peak=true",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(  # noqa: S603 - resolved binary path, fixed argv list, no shell
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if proc.returncode != 0:
        err_tail = proc.stderr.strip()[-300:]
        raise VideoAssemblyError(
            detail=f"FFmpeg ebur128 verification failed for {resolved}: {err_tail}",
            route="audio.master",
            charged="no",
        )
    summary = proc.stderr.rsplit("Summary:", 1)[-1]
    i_match = _EBUR128_I_RE.search(summary)
    lra_match = _EBUR128_LRA_RE.search(summary)
    peak_match = _EBUR128_PEAK_RE.search(summary)
    return (
        _parse_float_or(i_match.group(1) if i_match else None, _SILENCE_LUFS_FLOOR),
        _parse_float_or(peak_match.group(1) if peak_match else None, _SILENCE_PEAK_FLOOR),
        max(0.0, _parse_float_or(lra_match.group(1) if lra_match else None, 0.0)),
    )


def measure_loudnorm_pass1(
    input_file: Path | str,
    *,
    target_lufs: float = DEFAULT_TARGET_LUFS,
    true_peak: float = DEFAULT_TRUE_PEAK_DB,
    lra: float = DEFAULT_LRA_LU,
    pre_filter: str = "",
) -> LoudnormMeasurement:
    """Run Pass 1 of FFmpeg 2-pass EBU R128 ``loudnorm`` and parse JSON measurements."""
    resolved = Path(input_file).expanduser().resolve()
    ffmpeg = _find_binary("ffmpeg")
    ln_filter = f"loudnorm=I={target_lufs:g}:TP={true_peak:g}:LRA={lra:g}:print_format=json"
    af = f"{pre_filter},{ln_filter}" if pre_filter else ln_filter
    cmd = [ffmpeg, "-hide_banner", "-i", str(resolved), "-vn", "-af", af, "-f", "null", "-"]
    proc = subprocess.run(  # noqa: S603 - resolved binary path, fixed argv list, no shell
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if proc.returncode != 0:
        raise VideoAssemblyError(
            detail=f"FFmpeg loudnorm pass 1 failed for {resolved}: {proc.stderr.strip()[-300:]}",
            route="audio.master",
            charged="no",
        )
    matches = _JSON_BLOCK_RE.findall(proc.stderr)
    if not matches:
        raise VideoAssemblyError(
            detail=f"FFmpeg loudnorm pass 1 returned no measurement JSON for {resolved}.",
            route="audio.master",
            charged="no",
        )
    stats = cast("dict[str, Any]", json.loads(matches[-1]))
    raw_i = str(stats.get("input_i", "-inf"))
    raw_tp = str(stats.get("input_tp", "-inf"))
    raw_lra = str(stats.get("input_lra", "0"))
    raw_thresh = str(stats.get("input_thresh", "-70"))
    raw_offset = str(stats.get("target_offset", "0"))
    try:
        vals = (float(raw_i), float(raw_tp), float(raw_lra), float(raw_thresh), float(raw_offset))
        is_finite = all(math.isfinite(v) for v in vals) and vals[0] > -65.0
    except ValueError:
        is_finite = False
    return LoudnormMeasurement(
        input_i=_parse_float_or(raw_i, _SILENCE_LUFS_FLOOR),
        input_tp=_parse_float_or(raw_tp, _SILENCE_PEAK_FLOOR),
        input_lra=max(0.0, _parse_float_or(raw_lra, 0.0)),
        input_thresh=_parse_float_or(raw_thresh, _SILENCE_LUFS_FLOOR),
        target_offset=_parse_float_or(raw_offset, 0.0),
        is_finite=is_finite,
    )


def _codec_args_for_output(out_file: Path, *, audio_bitrate: str) -> tuple[str, list[str]]:
    suffix = out_file.suffix.lower()
    spec = _SUPPORTED_AUDIO_CODECS.get(suffix)
    if spec is None:
        supported = ", ".join(sorted(_SUPPORTED_AUDIO_CODECS))
        raise VideoAssemblyError(
            detail=f"Unsupported audio output extension {suffix!r}. Supported: {supported}.",
            route="audio.master",
            charged="no",
        )
    codec_name, base_args = spec
    args = list(base_args)
    if suffix in _LOSSY_SUFFIXES:
        args.extend(["-b:a", audio_bitrate])
    return codec_name, args


def _validate_master_inputs(
    src: Path,
    dst: Path,
    vo: Path | None,
    *,
    target_lufs: float,
    true_peak: float,
    lra: float,
    start_sec: float,
    duration_sec: float | None,
    fade_in_sec: float,
    fade_out_sec: float,
    overwrite: bool,
) -> None:
    if not src.is_file():
        raise VideoAssemblyError(
            detail=f"Input audio file does not exist: {src}", route="audio.master", charged="no"
        )
    if vo is not None and not vo.is_file():
        raise VideoAssemblyError(
            detail=f"Voiceover audio file does not exist: {vo}", route="audio.master", charged="no"
        )
    if dst == src or (vo is not None and dst == vo):
        raise VideoAssemblyError(
            detail="Output path must differ from every input.", route="audio.master", charged="no"
        )
    if not dst.parent.is_dir():
        raise VideoAssemblyError(
            detail=f"Output directory does not exist: {dst.parent}",
            route="audio.master",
            charged="no",
        )
    if dst.exists() and not overwrite:
        raise VideoAssemblyError(
            detail=f"Output already exists: {dst} (pass --overwrite to replace).",
            route="audio.master",
            charged="no",
        )
    if not (-70.0 <= target_lufs <= -5.0):
        raise VideoAssemblyError(
            detail="target_lufs must be between -70.0 and -5.0 LUFS.",
            route="audio.master",
            charged="no",
        )
    if not (-9.0 <= true_peak <= 0.0):
        raise VideoAssemblyError(
            detail="true_peak must be between -9.0 and 0.0 dBTP.",
            route="audio.master",
            charged="no",
        )
    if not (1.0 <= lra <= 50.0):
        raise VideoAssemblyError(
            detail="lra must be between 1.0 and 50.0 LU.", route="audio.master", charged="no"
        )
    if start_sec < 0.0 or fade_in_sec < 0.0 or fade_out_sec < 0.0:
        raise VideoAssemblyError(
            detail="start_sec, fade_in_sec, and fade_out_sec must be non-negative.",
            route="audio.master",
            charged="no",
        )
    if duration_sec is not None and duration_sec <= 0.0:
        raise VideoAssemblyError(
            detail="duration_sec must be greater than 0.", route="audio.master", charged="no"
        )


def _stage_preprocessed_audio(
    ffmpeg: str,
    *,
    src: Path,
    vo: Path | None,
    staging_wav: Path,
    start_sec: float,
    duration_sec: float | None,
    fade_in_sec: float,
    fade_out_sec: float,
    music_volume: float,
    sample_rate: int,
) -> None:
    source_duration = _probe_duration(src)
    available = max(0.0, source_duration - start_sec)
    effective_duration = min(duration_sec, available) if duration_sec is not None else available
    trim_fade = build_trim_fade_filter(
        duration_sec=effective_duration,
        fade_in_sec=fade_in_sec,
        fade_out_sec=fade_out_sec,
        highpass_hz=30.0,
    )
    cmd = [ffmpeg, "-y", "-hide_banner", "-v", "error"]
    if start_sec > 0.0:
        cmd.extend(["-ss", f"{start_sec:.3f}"])
    if duration_sec is not None:
        cmd.extend(["-t", f"{duration_sec:.3f}"])
    cmd.extend(["-i", str(src)])
    if vo is not None:
        cmd.extend(["-i", str(vo)])
        fc = build_vo_mix_filter_complex(
            music_volume=music_volume,
            music_trim_fade=trim_fade,
        )
        cmd.extend(["-filter_complex", fc, "-map", "[out]"])
    elif trim_fade:
        cmd.extend(["-vn", "-af", trim_fade])
    else:
        cmd.extend(["-vn"])
    cmd.extend(["-ar", str(sample_rate), "-ac", "2", "-c:a", "pcm_s24le", str(staging_wav)])
    proc = subprocess.run(  # noqa: S603 - resolved binary path, fixed argv list, no shell
        cmd,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if proc.returncode != 0:
        raise VideoAssemblyError(
            detail=f"Audio preprocessing failed: {proc.stderr.strip()[-300:]}",
            route="audio.master",
            charged="no",
        )


def _tame_high_crest_if_needed(
    ffmpeg: str,
    *,
    source_path: Path,
    measurement: LoudnormMeasurement,
    staging_wav: Path,
    target_lufs: float,
    true_peak: float,
    lra: float,
    sample_rate: int,
) -> tuple[Path, LoudnormMeasurement]:
    """Gently tame excessive transient peaks only when linear gain would exceed true peak."""
    if not measurement.is_finite:
        return source_path, measurement
    needed_gain = target_lufs - measurement.input_i
    avail_headroom = true_peak - measurement.input_tp
    if needed_gain <= avail_headroom + 1.5:
        return source_path, measurement
    pre_af = (
        f"volume={needed_gain:.2f}dB,"
        f"alimiter=limit={true_peak:.2f}dB:attack=5:release=50:level=disabled"
    )
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-v",
        "error",
        "-i",
        str(source_path),
        "-vn",
        "-af",
        pre_af,
        "-ar",
        str(sample_rate),
        "-ac",
        "2",
        "-c:a",
        "pcm_s24le",
        str(staging_wav),
    ]
    proc = subprocess.run(  # noqa: S603 - resolved binary path, fixed argv list, no shell
        cmd,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if proc.returncode != 0:
        return source_path, measurement
    remeasured = measure_loudnorm_pass1(
        staging_wav,
        target_lufs=target_lufs,
        true_peak=true_peak,
        lra=lra,
    )
    return (staging_wav, remeasured) if remeasured.is_finite else (source_path, measurement)


def _build_loudnorm_pass2_filter(
    measurement: LoudnormMeasurement,
    *,
    target_lufs: float,
    true_peak: float,
    lra: float,
    lossy_output: bool,
) -> tuple[str, bool]:
    # Lossy codecs (MP3/AAC) can add ~0.2 dB of inter-sample peak overshoot on decode.
    effective_tp = max(-9.0, true_peak - 0.25) if lossy_output else true_peak
    if not measurement.is_finite:
        return "", False
    ln = (
        f"loudnorm=I={target_lufs:g}:TP={effective_tp:g}:LRA={lra:g}"
        f":measured_I={measurement.input_i:.2f}"
        f":measured_TP={measurement.input_tp:.2f}"
        f":measured_LRA={measurement.input_lra:.2f}"
        f":measured_thresh={measurement.input_thresh:.2f}"
        f":offset={measurement.target_offset:.2f}"
        f":linear=true:dual_mono=false"
    )
    return ln, True


def master_audio(
    input_file: Path | str,
    out_file: Path | str,
    *,
    target_lufs: float = DEFAULT_TARGET_LUFS,
    true_peak: float = DEFAULT_TRUE_PEAK_DB,
    lra: float = DEFAULT_LRA_LU,
    start_sec: float = 0.0,
    duration_sec: float | None = None,
    fade_in_sec: float = 0.05,
    fade_out_sec: float = 0.0,
    voiceover: Path | str | None = None,
    music_volume: float = 0.32,
    sample_rate: int = 48000,
    audio_bitrate: str = "256k",
    overwrite: bool = False,
) -> MasteringResult:
    """Master an audio track with optional trim/fade, VO vocal-pocket ducking, and 2-pass LUFS."""
    src = Path(input_file).expanduser().resolve()
    dst = Path(out_file).expanduser().resolve()
    vo = Path(voiceover).expanduser().resolve() if voiceover is not None else None
    _validate_master_inputs(
        src,
        dst,
        vo,
        target_lufs=target_lufs,
        true_peak=true_peak,
        lra=lra,
        start_sec=start_sec,
        duration_sec=duration_sec,
        fade_in_sec=fade_in_sec,
        fade_out_sec=fade_out_sec,
        overwrite=overwrite,
    )
    codec_name, codec_args = _codec_args_for_output(dst, audio_bitrate=audio_bitrate)
    ffmpeg = _find_binary("ffmpeg")
    lossy_output = dst.suffix.lower() in _LOSSY_SUFFIXES

    with tempfile.TemporaryDirectory(prefix=".gflow-master-", dir=dst.parent) as tmp_dir:
        tmp_root = Path(tmp_dir)
        needs_pre = (
            vo is not None
            or start_sec > 0.0
            or duration_sec is not None
            or fade_in_sec > 0.0
            or fade_out_sec > 0.0
        )
        if needs_pre:
            pass1_input = tmp_root / "pre_norm.wav"
            _stage_preprocessed_audio(
                ffmpeg,
                src=src,
                vo=vo,
                staging_wav=pass1_input,
                start_sec=start_sec,
                duration_sec=duration_sec,
                fade_in_sec=fade_in_sec,
                fade_out_sec=fade_out_sec,
                music_volume=music_volume,
                sample_rate=sample_rate,
            )
        else:
            pass1_input = src

        measurement = measure_loudnorm_pass1(
            pass1_input,
            target_lufs=target_lufs,
            true_peak=true_peak,
            lra=lra,
        )
        pass2_input, pass2_measurement = _tame_high_crest_if_needed(
            ffmpeg,
            source_path=pass1_input,
            measurement=measurement,
            staging_wav=tmp_root / "tamed.wav",
            target_lufs=target_lufs,
            true_peak=true_peak,
            lra=lra,
            sample_rate=sample_rate,
        )
        pass2_filter, linear_mode = _build_loudnorm_pass2_filter(
            pass2_measurement,
            target_lufs=target_lufs,
            true_peak=true_peak,
            lra=lra,
            lossy_output=lossy_output,
        )

        staged_out = tmp_root / f"mastered{dst.suffix.lower()}"
        cmd = [ffmpeg, "-y", "-hide_banner", "-v", "error", "-i", str(pass2_input), "-vn"]
        if pass2_filter:
            cmd.extend(["-af", pass2_filter])
        cmd.extend(["-ar", str(sample_rate), "-ac", "2", *codec_args, str(staged_out)])
        proc = subprocess.run(  # noqa: S603 - resolved binary path, fixed argv list, no shell
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if proc.returncode != 0:
            raise VideoAssemblyError(
                detail=f"FFmpeg loudnorm pass 2 failed: {proc.stderr.strip()[-300:]}",
                route="audio.master",
                charged="no",
            )

        out_lufs, out_tp, out_lra = verify_loudness(staged_out)
        out_dur = _probe_duration(staged_out)
        os.replace(staged_out, dst)

    _log.info(
        "audio_mastering.completed",
        output=str(dst),
        target_lufs=target_lufs,
        verified_lufs=out_lufs,
        verified_tp=out_tp,
    )
    return MasteringResult(
        input_file=src,
        output_file=dst,
        target_lufs=target_lufs,
        target_true_peak_db=true_peak,
        target_lra_lu=lra,
        measured_input_lufs=measurement.input_i,
        measured_input_true_peak_db=measurement.input_tp,
        verified_output_lufs=out_lufs,
        verified_output_true_peak_db=out_tp,
        verified_output_lra_lu=out_lra,
        duration_sec=out_dur,
        sample_rate=sample_rate,
        codec=codec_name,
        linear_mode=linear_mode,
        fade_in_sec=fade_in_sec,
        fade_out_sec=fade_out_sec,
        has_voiceover=vo is not None,
        vocal_pocket_eq_applied=vo is not None,
        ducking_applied=vo is not None,
        bytes=dst.stat().st_size,
    )


def normalize_video_audio_track(
    input_video: Path,
    output_video: Path,
    *,
    target_lufs: float = DEFAULT_TARGET_LUFS,
    true_peak: float = DEFAULT_TRUE_PEAK_DB,
    lra: float = DEFAULT_LRA_LU,
    audio_bitrate: str = "192k",
    sample_rate: int = 44100,
) -> tuple[float, float, float]:
    """Copy video stream losslessly while 2-pass EBU R128 mastering the MP4 audio stream."""
    ffmpeg = _find_binary("ffmpeg", route="video.assemble")
    measurement = measure_loudnorm_pass1(
        input_video,
        target_lufs=target_lufs,
        true_peak=true_peak,
        lra=lra,
    )
    pass2_filter, _ = _build_loudnorm_pass2_filter(
        measurement,
        target_lufs=target_lufs,
        true_peak=true_peak,
        lra=lra,
        lossy_output=True,
    )
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-v",
        "error",
        "-i",
        str(input_video),
        "-c:v",
        "copy",
    ]
    if pass2_filter:
        cmd.extend(["-af", pass2_filter])
    cmd.extend(["-c:a", "aac", "-b:a", audio_bitrate, "-ar", str(sample_rate), str(output_video)])
    proc = subprocess.run(  # noqa: S603 - resolved binary path, fixed argv list, no shell
        cmd,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if proc.returncode != 0:
        raise VideoAssemblyError(
            detail=f"FFmpeg audio mastering failed during assembly: {proc.stderr.strip()[-300:]}",
            route="video.assemble",
            charged="no",
        )
    return verify_loudness(output_video)
