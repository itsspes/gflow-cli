"""FlowMusic beat, BPM, onset, loudness, silence, and energy-envelope analyzer."""

from __future__ import annotations

import array
import importlib
import json
import math
import re
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import structlog

from gflow_cli.errors import VideoAssemblyError

_log = structlog.get_logger(__name__)

AUDIO_MAP_VERSION = "1.0"
DEFAULT_SHOT_DURATIONS: tuple[float, ...] = (4.0, 6.0, 8.0, 10.0, 15.0)
_ANALYSIS_SAMPLE_RATE = 22050
_HOP_LENGTH = 256
_WIN_LENGTH = 1024
_MIN_BPM = 60.0
_MAX_BPM = 180.0
_SILENCE_LUFS_FLOOR = -70.0
_SILENCE_PEAK_FLOOR = -99.0

_EBUR128_I_RE = re.compile(r"^\s*I:\s*([-+]?\d+(?:\.\d+)?|-?inf)\s*LUFS", re.MULTILINE)
_EBUR128_LRA_RE = re.compile(r"^\s*LRA:\s*([-+]?\d+(?:\.\d+)?|-?inf)\s*LU", re.MULTILINE)
_EBUR128_PEAK_RE = re.compile(r"^\s*Peak:\s*([-+]?\d+(?:\.\d+)?|-?inf)\s*dBFS", re.MULTILINE)
_SILENCE_START_RE = re.compile(r"silence_start:\s*([-+]?\d+(?:\.\d+)?)")
_SILENCE_END_RE = re.compile(
    r"silence_end:\s*([-+]?\d+(?:\.\d+)?)\s*\|\s*silence_duration:\s*([-+]?\d+(?:\.\d+)?)"
)


@dataclass(frozen=True)
class SilenceSegment:
    """Detected contiguous silence interval in seconds."""

    start: float
    end: float
    duration: float

    def to_dict(self) -> dict[str, float]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
        }


@dataclass(frozen=True)
class EnergyPoint:
    """Windowed RMS loudness sample along the track timeline."""

    time: float
    rms_db: float

    def to_dict(self) -> dict[str, float]:
        return {"time": round(self.time, 3), "rmsDb": round(self.rms_db, 2)}


@dataclass(frozen=True)
class EnergyPeak:
    """Prominent structural energy peak (e.g. drop or climax)."""

    time: float
    label: str
    rms_db: float

    def to_dict(self) -> dict[str, Any]:
        return {"time": round(self.time, 3), "label": self.label, "rmsDb": round(self.rms_db, 2)}


@dataclass(frozen=True)
class ShotCutPlan:
    """Recommended bar-aligned shot timing and cut points for a target clip length."""

    target_duration_sec: float
    bars: int
    aligned_duration_sec: float
    drift_sec: float
    cut_points: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "targetDurationSec": round(self.target_duration_sec, 3),
            "bars": self.bars,
            "alignedDurationSec": round(self.aligned_duration_sec, 3),
            "driftSec": round(self.drift_sec, 3),
            "cutPoints": [round(t, 3) for t in self.cut_points],
        }


@dataclass(frozen=True)
class AudioQualityReport:
    """Typed technical quality report reusable by Quality Gate and audio critics.

    All fields are JSON-native primitives/collections so ``dict(report.__dict__)``
    can be serialized directly by ``quality_gate.py`` without custom encoders.
    """

    source: str
    codec: str
    sample_rate: int
    channels: int
    bit_rate: int | None
    duration_seconds: float
    integrated_lufs: float
    lra_lu: float
    true_peak_dbtp: float
    clipping_detected: bool
    exceeds_true_peak_ceiling: bool
    is_silent: bool
    silence_ratio: float
    silence_segments: tuple[dict[str, float], ...]
    bpm: float | None = None
    tempo_confidence: float | None = None
    warnings: tuple[str, ...] = ()

    @property
    def duration_sec(self) -> float:
        return self.duration_seconds

    @property
    def lufs_integrated(self) -> float:
        return self.integrated_lufs

    @property
    def true_peak_db(self) -> float:
        return self.true_peak_dbtp

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "codec": self.codec,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "bit_rate": self.bit_rate,
            "duration_seconds": round(self.duration_seconds, 3),
            "integrated_lufs": round(self.integrated_lufs, 2),
            "lra_lu": round(self.lra_lu, 2),
            "true_peak_dbtp": round(self.true_peak_dbtp, 2),
            "clipping_detected": self.clipping_detected,
            "exceeds_true_peak_ceiling": self.exceeds_true_peak_ceiling,
            "is_silent": self.is_silent,
            "silence_ratio": round(self.silence_ratio, 4),
            "silence_segments": list(self.silence_segments),
            "bpm": round(self.bpm, 2) if self.bpm is not None else None,
            "tempo_confidence": (
                round(self.tempo_confidence, 3) if self.tempo_confidence is not None else None
            ),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class AudioMap:
    """Full FlowMusic beat, structure, loudness, and bar-aligned cut map (`audio-map.json`)."""

    version: str
    source: Path
    analyzer_backend: str
    codec: str
    sample_rate: int
    channels: int
    bit_rate: int | None
    duration_sec: float
    lufs_integrated: float
    lra_lu: float
    true_peak_db: float
    clipping_detected: bool
    exceeds_true_peak_ceiling: bool
    is_silent: bool
    silence_ratio: float
    silence_segments: tuple[SilenceSegment, ...]
    bpm: float
    tempo_confidence: float
    beat_interval_sec: float
    bar_duration_sec: float
    beats: tuple[float, ...]
    downbeats: tuple[float, ...]
    onsets: tuple[float, ...]
    energy_envelope: tuple[EnergyPoint, ...]
    energy_peaks: tuple[EnergyPeak, ...]
    shot_cut_plans: dict[str, ShotCutPlan]
    recommended_cut_points: tuple[float, ...]
    out_file: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": self.version,
            "source": str(self.source),
            "analyzerBackend": self.analyzer_backend,
            "codec": self.codec,
            "sampleRate": self.sample_rate,
            "channels": self.channels,
            "bitRate": self.bit_rate,
            "durationSec": round(self.duration_sec, 3),
            "lufsIntegrated": round(self.lufs_integrated, 2),
            "lraLu": round(self.lra_lu, 2),
            "truePeakDb": round(self.true_peak_db, 2),
            "clippingDetected": self.clipping_detected,
            "exceedsTruePeakCeiling": self.exceeds_true_peak_ceiling,
            "isSilent": self.is_silent,
            "silenceRatio": round(self.silence_ratio, 4),
            "silenceSegments": [seg.to_dict() for seg in self.silence_segments],
            "bpm": round(self.bpm, 2),
            "tempoConfidence": round(self.tempo_confidence, 3),
            "beatIntervalSec": round(self.beat_interval_sec, 4),
            "barDurationSec": round(self.bar_duration_sec, 4),
            "beats": [round(t, 3) for t in self.beats],
            "downbeats": [round(t, 3) for t in self.downbeats],
            "onsets": [round(t, 3) for t in self.onsets],
            "energyEnvelope": [pt.to_dict() for pt in self.energy_envelope],
            "energyPeaks": [pk.to_dict() for pk in self.energy_peaks],
            "shotCutPlans": {key: plan.to_dict() for key, plan in self.shot_cut_plans.items()},
            "recommendedCutPoints": [round(t, 3) for t in self.recommended_cut_points],
        }
        if self.out_file is not None:
            payload["outFile"] = str(self.out_file)
        return payload

    def write_json(self, out_file: Path) -> Path:
        """Atomically write ``audio-map.json`` to ``out_file``."""
        target = out_file.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        tmp.replace(target)
        return target


@dataclass(frozen=True)
class _StreamInfo:
    codec: str
    sample_rate: int
    channels: int
    bit_rate: int | None
    duration_sec: float


@dataclass(frozen=True)
class _LoudnessStats:
    lufs_integrated: float
    lra_lu: float
    true_peak_db: float
    silence_segments: tuple[SilenceSegment, ...]
    silence_ratio: float


@dataclass(frozen=True)
class _RhythmAnalysis:
    backend: str
    bpm: float
    tempo_confidence: float
    beats: tuple[float, ...]
    downbeats: tuple[float, ...]
    onsets: tuple[float, ...]
    energy_envelope: tuple[EnergyPoint, ...]
    energy_peaks: tuple[EnergyPeak, ...]


def _find_binary(name: str, *, route: str = "audio.analyze") -> str:
    binary = shutil.which(name)
    if binary is None:
        raise VideoAssemblyError(
            detail=f"Required binary {name!r} was not found on PATH.",
            route=route,
            charged="no",
            remediation_hint=f"Install FFmpeg (`brew install ffmpeg`) so `{name}` is available.",
        )
    return binary


def _validate_input_file(path: Path | str, *, route: str = "audio.analyze") -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise VideoAssemblyError(
            detail=f"Audio file does not exist: {resolved}",
            route=route,
            charged="no",
            remediation_hint="Provide an existing local audio or media file path.",
        )
    return resolved


def _parse_float_or(raw: str | None, fallback: float) -> float:
    if raw is None:
        return fallback
    try:
        val = float(raw)
    except ValueError:
        return fallback
    return val if math.isfinite(val) else fallback


def _probe_audio_stream(path: Path) -> _StreamInfo:
    ffprobe = _find_binary("ffprobe")
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate,channels,bit_rate,duration",
        "-show_entries",
        "format=duration,bit_rate",
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
            route="audio.analyze",
            charged="no",
        )
    data = cast("dict[str, Any]", json.loads(proc.stdout or "{}"))
    streams = cast("list[dict[str, Any]]", data.get("streams") or [])
    if not streams:
        raise VideoAssemblyError(
            detail=f"Input file has no audio stream: {path}",
            route="audio.analyze",
            charged="no",
        )
    stream = streams[0]
    fmt = cast("dict[str, Any]", data.get("format") or {})
    duration_sec = _parse_float_or(
        str(stream.get("duration") or fmt.get("duration") or "0"),
        0.0,
    )
    raw_br = stream.get("bit_rate") or fmt.get("bit_rate")
    bit_rate: int | None = int(raw_br) if raw_br is not None and str(raw_br).isdigit() else None
    return _StreamInfo(
        codec=str(stream.get("codec_name") or "unknown"),
        sample_rate=int(stream.get("sample_rate") or 0),
        channels=int(stream.get("channels") or 0),
        bit_rate=bit_rate,
        duration_sec=max(0.0, duration_sec),
    )


def _parse_silence_segments(stderr: str, duration_sec: float) -> tuple[SilenceSegment, ...]:
    segments: list[SilenceSegment] = []
    pending_start: float | None = None
    for line in stderr.splitlines():
        start_match = _SILENCE_START_RE.search(line)
        if start_match:
            pending_start = _parse_float_or(start_match.group(1), 0.0)
            continue
        end_match = _SILENCE_END_RE.search(line)
        if end_match:
            end_val = _parse_float_or(end_match.group(1), 0.0)
            dur_val = _parse_float_or(end_match.group(2), 0.0)
            start_val = pending_start if pending_start is not None else max(0.0, end_val - dur_val)
            segments.append(
                SilenceSegment(start=start_val, end=end_val, duration=max(0.0, dur_val))
            )
            pending_start = None
    if pending_start is not None and duration_sec > pending_start:
        segments.append(
            SilenceSegment(
                start=pending_start,
                end=duration_sec,
                duration=max(0.0, duration_sec - pending_start),
            )
        )
    return tuple(segments)


def _measure_ebur128_and_silence(
    path: Path,
    *,
    duration_sec: float,
    silence_threshold_db: float = -50.0,
    min_silence_sec: float = 0.5,
) -> _LoudnessStats:
    ffmpeg = _find_binary("ffmpeg")
    af = (
        f"ebur128=peak=true,silencedetect=noise={silence_threshold_db:.1f}dB"
        f":d={min_silence_sec:.3f}"
    )
    cmd = [ffmpeg, "-hide_banner", "-i", str(path), "-vn", "-af", af, "-f", "null", "-"]
    proc = subprocess.run(  # noqa: S603 - resolved binary path, fixed argv list, no shell
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if proc.returncode != 0:
        raise VideoAssemblyError(
            detail=f"FFmpeg ebur128 analysis failed for {path}: {proc.stderr.strip()[-300:]}",
            route="audio.analyze",
            charged="no",
        )
    summary = proc.stderr.rsplit("Summary:", 1)[-1]
    i_match = _EBUR128_I_RE.search(summary)
    lra_match = _EBUR128_LRA_RE.search(summary)
    peak_match = _EBUR128_PEAK_RE.search(summary)
    lufs = _parse_float_or(i_match.group(1) if i_match else None, _SILENCE_LUFS_FLOOR)
    lra = _parse_float_or(lra_match.group(1) if lra_match else None, 0.0)
    peak = _parse_float_or(peak_match.group(1) if peak_match else None, _SILENCE_PEAK_FLOOR)
    segments = _parse_silence_segments(proc.stderr, duration_sec)
    total_silence = sum(seg.duration for seg in segments)
    silence_ratio = min(1.0, total_silence / duration_sec) if duration_sec > 0.0 else 0.0
    if lufs <= -65.0 and peak <= -65.0:
        silence_ratio = 1.0
    return _LoudnessStats(
        lufs_integrated=lufs,
        lra_lu=max(0.0, lra),
        true_peak_db=peak,
        silence_segments=segments,
        silence_ratio=silence_ratio,
    )


def _decode_mono_pcm(path: Path, *, sample_rate: int = _ANALYSIS_SAMPLE_RATE) -> bytes:
    ffmpeg = _find_binary("ffmpeg")
    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    proc = subprocess.run(  # noqa: S603 - resolved binary path, fixed argv list, no shell
        cmd,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if proc.returncode != 0:
        stderr_text = proc.stderr.decode("utf-8", errors="replace").strip()
        raise VideoAssemblyError(
            detail=f"FFmpeg PCM decode failed for {path}: {stderr_text[-300:]}",
            route="audio.analyze",
            charged="no",
        )
    return proc.stdout


def _try_import_librosa() -> Any | None:
    try:
        return importlib.import_module("librosa")
    except ImportError:
        return None


def _try_import_numpy() -> Any | None:
    try:
        return importlib.import_module("numpy")
    except ImportError:
        return None


def _label_energy_peaks(
    envelope: Sequence[EnergyPoint],
    *,
    duration_sec: float,
    min_spacing_sec: float = 2.0,
) -> tuple[EnergyPeak, ...]:
    if not envelope:
        return ()
    rms_values = [pt.rms_db for pt in envelope]
    peak_max = max(rms_values)
    if peak_max <= -60.0:
        return ()
    candidates: list[EnergyPoint] = []
    for idx, pt in enumerate(envelope):
        prev_v = envelope[idx - 1].rms_db if idx > 0 else -120.0
        next_v = envelope[idx + 1].rms_db if idx + 1 < len(envelope) else -120.0
        if pt.rms_db >= prev_v and pt.rms_db >= next_v and pt.rms_db >= peak_max - 9.0:
            candidates.append(pt)
    if not candidates:
        candidates = [max(envelope, key=lambda p: p.rms_db)]
    selected: list[EnergyPoint] = []
    for pt in sorted(candidates, key=lambda p: p.rms_db, reverse=True):
        if all(abs(pt.time - kept.time) >= min_spacing_sec for kept in selected):
            selected.append(pt)
        if len(selected) >= 5:
            break
    selected.sort(key=lambda p: p.time)
    climax_time = max(selected, key=lambda p: p.rms_db).time
    labeled: list[EnergyPeak] = []
    for idx, pt in enumerate(selected):
        if pt.time == climax_time and len(selected) > 1 and pt.time > duration_sec * 0.45:
            label = "climax"
        elif idx == 0 and pt.time >= 1.0:
            label = "drop"
        elif pt.time == climax_time:
            label = "climax"
        else:
            label = "peak"
        labeled.append(EnergyPeak(time=pt.time, label=label, rms_db=pt.rms_db))
    return tuple(labeled)


def _moving_mean_np(np: Any, values: Any, radius: int) -> Any:
    if len(values) == 0:
        return values
    window = 2 * radius + 1
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _compute_onset_and_rms_np(
    np: Any,
    samples: Any,
    *,
    sample_rate: int = _ANALYSIS_SAMPLE_RATE,
) -> tuple[Any, Any, float]:
    half_win = _WIN_LENGTH // 2
    padded = np.pad(samples.astype(np.float32), (half_win, half_win), mode="constant")
    n = int(len(padded))
    n_frames = max(1, 1 + (n - _WIN_LENGTH) // _HOP_LENGTH)
    indices = (
        np.arange(_WIN_LENGTH, dtype=np.int64)[None, :]
        + (np.arange(n_frames, dtype=np.int64) * _HOP_LENGTH)[:, None]
    )
    frames = padded[indices].astype(np.float64)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    hann = np.hanning(_WIN_LENGTH)[None, :]
    mag = np.abs(np.fft.rfft(frames * hann, axis=1))
    log_mag = np.log1p(10.0 * mag)
    flux = np.zeros(n_frames, dtype=np.float64)
    if n_frames > 0:
        flux[0] = float(np.sum(log_mag[0]))
    if n_frames > 1:
        flux[1:] = np.sum(np.maximum(0.0, np.diff(log_mag, axis=0)), axis=1)
    rms_diff = np.zeros(n_frames, dtype=np.float64)
    if n_frames > 0:
        rms_diff[0] = float(rms[0])
    if n_frames > 1:
        rms_diff[1:] = np.maximum(0.0, np.diff(rms))

    flux_scale = float(np.percentile(flux, 95)) if n_frames > 0 else 0.0
    rms_scale = float(np.percentile(rms_diff, 95)) if n_frames > 0 else 0.0
    flux_norm = flux / flux_scale if flux_scale > 1e-9 else np.zeros_like(flux)
    rms_norm = rms_diff / rms_scale if rms_scale > 1e-9 else np.zeros_like(rms_diff)
    odf = 0.65 * flux_norm + 0.35 * rms_norm
    odf_clean = np.maximum(0.0, odf - _moving_mean_np(np, odf, radius=15))
    fps = float(sample_rate) / float(_HOP_LENGTH)
    return odf_clean, rms, fps


def _pick_onsets_np(np: Any, odf_clean: Any, rms: Any, fps: float) -> tuple[float, ...]:
    if len(odf_clean) < 3 or float(np.max(rms)) < 1e-4:
        return ()
    threshold = float(np.mean(odf_clean) + 0.8 * np.std(odf_clean))
    min_val = max(threshold, 0.12)
    min_gap = max(1, int(round(0.12 * fps)))
    onsets: list[float] = []
    last_idx = -min_gap
    for idx in range(1, len(odf_clean) - 1):
        val = float(odf_clean[idx])
        if (
            val >= min_val
            and val >= float(odf_clean[idx - 1])
            and val >= float(odf_clean[idx + 1])
            and (idx - last_idx) >= min_gap
        ):
            onsets.append(round(idx / fps, 3))
            last_idx = idx
    return tuple(onsets)


def _estimate_bpm_autocorr_np(np: Any, odf_clean: Any, fps: float) -> tuple[float, float]:
    n = int(len(odf_clean))
    if n < 16 or float(np.sum(odf_clean)) <= 1e-6:
        return 0.0, 0.0
    n_fft = 1 << (2 * n - 1).bit_length()
    spec = np.fft.rfft(odf_clean, n=n_fft)
    ac = np.fft.irfft(spec.real * spec.real + spec.imag * spec.imag, n=n_fft)[:n]
    zero_val = float(ac[0])
    if zero_val <= 1e-9:
        return 0.0, 0.0
    ac = ac / zero_val

    lag_min = max(2, int(round(fps * 60.0 / _MAX_BPM)))
    lag_max = min(n - 2, int(round(fps * 60.0 / _MIN_BPM)))
    if lag_max <= lag_min:
        return 0.0, 0.0

    best_lag = lag_min
    best_score = -1.0
    for lag in range(lag_min, lag_max + 1):
        bpm_cand = 60.0 * fps / float(lag)
        prior = math.exp(-0.5 * (math.log2(bpm_cand / 118.0) / 0.65) ** 2)
        h2 = float(ac[2 * lag]) if 2 * lag < n else 0.0
        score = (float(ac[lag]) + 0.35 * h2) * (0.7 + 0.3 * prior)
        if score > best_score:
            best_score = score
            best_lag = lag

    confidence = max(0.0, min(1.0, float(ac[best_lag])))
    if confidence < 0.08:
        return 0.0, 0.0

    # Parabolic interpolation around best_lag for sub-frame BPM accuracy.
    refined_lag = float(best_lag)
    if lag_min < best_lag < lag_max:
        y0 = float(ac[best_lag - 1])
        y1 = float(ac[best_lag])
        y2 = float(ac[best_lag + 1])
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-12:
            delta = max(-0.5, min(0.5, 0.5 * (y0 - y2) / denom))
            refined_lag = best_lag + delta

    bpm = 60.0 * fps / refined_lag
    return round(bpm, 2), round(confidence, 3)


def _align_beat_grid_np(
    np: Any,
    odf_clean: Any,
    rms: Any,
    *,
    bpm: float,
    fps: float,
    duration_sec: float,
    onsets: Sequence[float],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if bpm <= 0.0 or duration_sec <= 0.0:
        return (), ()
    beat_period_frames = 60.0 * fps / bpm
    n = int(len(odf_clean))
    best_phase = 0.0
    best_energy = -1.0
    steps = max(8, int(round(beat_period_frames * 4)))
    for step in range(steps):
        phase = (step / float(steps)) * beat_period_frames
        positions = np.arange(phase, n, beat_period_frames)
        idx = np.clip(np.round(positions).astype(np.int64), 0, n - 1)
        score = float(np.sum(odf_clean[idx]))
        if score > best_energy:
            best_energy = score
            best_phase = phase

    beat_interval = 60.0 / bpm
    phase_sec = best_phase / fps
    raw_beats: list[float] = []
    t = phase_sec
    while t < duration_sec - 1e-4:
        snapped = t
        if onsets:
            nearest = min(onsets, key=lambda o: abs(o - t))
            if abs(nearest - t) <= 0.05:
                snapped = nearest
        snapped = round(max(0.0, snapped), 3)
        if not raw_beats or snapped > raw_beats[-1] + 0.05:
            raw_beats.append(snapped)
        t += beat_interval

    if not raw_beats:
        return (), ()

    # Pick 4/4 downbeat offset (0..3) with strongest average accent; prefer 0 on ties.
    best_offset = 0
    best_accent = -1.0
    for offset in range(min(4, len(raw_beats))):
        subset = raw_beats[offset::4]
        accent = 0.0
        for b_time in subset:
            frame_i = min(n - 1, max(0, int(round(b_time * fps))))
            accent += float(rms[frame_i]) + 0.5 * float(odf_clean[frame_i])
        avg_accent = accent / float(len(subset))
        if offset == 0 or avg_accent > best_accent * 1.03:
            best_accent = avg_accent
            best_offset = offset

    downbeats = tuple(raw_beats[best_offset::4])
    return tuple(raw_beats), downbeats


def _build_energy_envelope_np(
    np: Any,
    rms: Any,
    fps: float,
    *,
    step_sec: float = 0.5,
) -> tuple[EnergyPoint, ...]:
    if len(rms) == 0:
        return ()
    frames_per_bin = max(1, int(round(step_sec * fps)))
    points: list[EnergyPoint] = []
    for start in range(0, len(rms), frames_per_bin):
        chunk = rms[start : start + frames_per_bin]
        mean_rms = float(np.sqrt(np.mean(chunk * chunk)))
        rms_db = 20.0 * math.log10(max(mean_rms, 1e-5))
        points.append(EnergyPoint(time=round(start / fps, 3), rms_db=round(rms_db, 2)))
    return tuple(points)


def _analyze_rhythm_librosa(
    librosa: Any,
    np: Any,
    path: Path,
    duration_sec: float,
) -> _RhythmAnalysis:
    y, sr = librosa.load(str(path), sr=_ANALYSIS_SAMPLE_RATE, mono=True)
    if len(y) == 0 or float(np.max(np.abs(y))) < 1e-4:
        return _RhythmAnalysis("librosa", 0.0, 0.0, (), (), (), (), ())
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=_HOP_LENGTH)
    tempo_raw, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=_HOP_LENGTH
    )
    bpm = round(float(np.atleast_1d(tempo_raw)[0]), 2)
    beats = tuple(
        round(float(t), 3)
        for t in librosa.frames_to_time(beat_frames, sr=sr, hop_length=_HOP_LENGTH)
    )
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=_HOP_LENGTH
    )
    onsets = tuple(
        round(float(t), 3)
        for t in librosa.frames_to_time(onset_frames, sr=sr, hop_length=_HOP_LENGTH)
    )
    rms = librosa.feature.rms(y=y, frame_length=_WIN_LENGTH, hop_length=_HOP_LENGTH)[0]
    fps = float(sr) / float(_HOP_LENGTH)
    envelope = _build_energy_envelope_np(np, rms, fps)
    peaks = _label_energy_peaks(envelope, duration_sec=duration_sec)
    downbeats = beats[::4]
    confidence = 0.85 if bpm > 0.0 and len(beats) >= 2 else 0.0
    return _RhythmAnalysis(
        backend="librosa",
        bpm=bpm,
        tempo_confidence=confidence,
        beats=beats,
        downbeats=downbeats,
        onsets=onsets,
        energy_envelope=envelope,
        energy_peaks=peaks,
    )


def _analyze_rhythm_stdlib(pcm_bytes: bytes, duration_sec: float) -> _RhythmAnalysis:
    """Pure-stdlib fallback when neither librosa nor numpy is installed."""
    samples = array.array("f")
    samples.frombytes(pcm_bytes[: (len(pcm_bytes) // 4) * 4])
    if not samples:
        return _RhythmAnalysis("ffmpeg_stdlib", 0.0, 0.0, (), (), (), (), ())
    fps = float(_ANALYSIS_SAMPLE_RATE) / float(_HOP_LENGTH)
    rms_list: list[float] = []
    for start in range(0, max(1, len(samples) - _HOP_LENGTH), _HOP_LENGTH):
        end = min(len(samples), start + _WIN_LENGTH)
        chunk = samples[start:end]
        mean_sq = sum(x * x for x in chunk) / float(max(1, len(chunk)))
        rms_list.append(math.sqrt(mean_sq))
    if not rms_list or max(rms_list) < 1e-4:
        return _RhythmAnalysis("ffmpeg_stdlib", 0.0, 0.0, (), (), (), (), ())
    odf = [0.0] + [max(0.0, rms_list[i] - rms_list[i - 1]) for i in range(1, len(rms_list))]
    mean_o = sum(odf) / float(len(odf))
    odf_clean = [max(0.0, v - mean_o) for v in odf]
    zero = sum(v * v for v in odf_clean)
    if zero <= 1e-9:
        return _RhythmAnalysis("ffmpeg_stdlib", 0.0, 0.0, (), (), (), (), ())
    lag_min = max(2, int(round(fps * 60.0 / _MAX_BPM)))
    lag_max = min(len(odf_clean) - 2, int(round(fps * 60.0 / _MIN_BPM)))
    best_lag, best_ac = lag_min, 0.0
    for lag in range(lag_min, lag_max + 1):
        ac_val = sum(odf_clean[i] * odf_clean[i - lag] for i in range(lag, len(odf_clean))) / zero
        if ac_val > best_ac:
            best_ac = ac_val
            best_lag = lag
    bpm = round(60.0 * fps / float(best_lag), 2) if best_ac >= 0.08 else 0.0
    beats = (
        tuple(round(i * (60.0 / bpm), 3) for i in range(int(duration_sec * bpm / 60.0)))
        if bpm > 0.0
        else ()
    )
    step = max(1, int(round(0.5 * fps)))
    envelope = tuple(
        EnergyPoint(
            time=round(i / fps, 3),
            rms_db=round(20.0 * math.log10(max(rms_list[i], 1e-5)), 2),
        )
        for i in range(0, len(rms_list), step)
    )
    peaks = _label_energy_peaks(envelope, duration_sec=duration_sec)
    return _RhythmAnalysis(
        backend="ffmpeg_stdlib",
        bpm=bpm,
        tempo_confidence=round(min(1.0, best_ac), 3) if bpm > 0.0 else 0.0,
        beats=beats,
        downbeats=beats[::4],
        onsets=beats,
        energy_envelope=envelope,
        energy_peaks=peaks,
    )


def _analyze_rhythm(path: Path, duration_sec: float) -> _RhythmAnalysis:
    np = _try_import_numpy()
    librosa = _try_import_librosa()
    if librosa is not None and np is not None:
        try:
            return _analyze_rhythm_librosa(librosa, np, path, duration_sec)
        except Exception as exc:  # noqa: BLE001 - fall back cleanly to deterministic FFmpeg path
            _log.debug("audio_analysis.librosa_fallback", error=str(exc))

    pcm_bytes = _decode_mono_pcm(path)
    if np is None:
        return _analyze_rhythm_stdlib(pcm_bytes, duration_sec)

    aligned_len = (len(pcm_bytes) // 4) * 4
    samples = np.frombuffer(pcm_bytes[:aligned_len], dtype=np.float32)
    if len(samples) == 0:
        return _RhythmAnalysis("ffmpeg_numpy", 0.0, 0.0, (), (), (), (), ())

    odf_clean, rms, fps = _compute_onset_and_rms_np(np, samples)
    onsets = _pick_onsets_np(np, odf_clean, rms, fps)
    bpm, confidence = _estimate_bpm_autocorr_np(np, odf_clean, fps)
    beats, downbeats = _align_beat_grid_np(
        np,
        odf_clean,
        rms,
        bpm=bpm,
        fps=fps,
        duration_sec=duration_sec,
        onsets=onsets,
    )
    envelope = _build_energy_envelope_np(np, rms, fps)
    peaks = _label_energy_peaks(envelope, duration_sec=duration_sec)
    return _RhythmAnalysis(
        backend="ffmpeg_numpy",
        bpm=bpm,
        tempo_confidence=confidence,
        beats=beats,
        downbeats=downbeats,
        onsets=onsets,
        energy_envelope=envelope,
        energy_peaks=peaks,
    )


def _format_duration_key(duration_sec: float) -> str:
    if abs(duration_sec - round(duration_sec)) < 1e-6:
        return f"{int(round(duration_sec))}s"
    return f"{duration_sec:g}s"


def build_shot_cut_plans(
    duration_sec: float,
    bar_duration_sec: float,
    downbeats: Sequence[float],
    shot_durations: Sequence[float] = DEFAULT_SHOT_DURATIONS,
) -> dict[str, ShotCutPlan]:
    """Compute bar-aligned cut plans for Omni/Veo shot lengths (e.g. 4s/6s/8s/10s/15s)."""
    plans: dict[str, ShotCutPlan] = {}
    for raw_target in shot_durations:
        target = float(raw_target)
        if target <= 0.0:
            continue
        key = _format_duration_key(target)
        if bar_duration_sec > 0.0:
            bars = max(1, round(target / bar_duration_sec))
            aligned = round(bars * bar_duration_sec, 3)
            drift = round(aligned - target, 3)
            if downbeats:
                cuts = [round(t, 3) for t in downbeats[::bars] if t < duration_sec - 0.1]
            else:
                cuts = [
                    round(k * aligned, 3)
                    for k in range(max(1, int(duration_sec // max(aligned, 0.1)) + 1))
                    if k * aligned < duration_sec - 0.1
                ]
            if not cuts and duration_sec > 0.0:
                cuts = [0.0]
            plans[key] = ShotCutPlan(
                target_duration_sec=target,
                bars=bars,
                aligned_duration_sec=aligned,
                drift_sec=drift,
                cut_points=tuple(cuts),
            )
        else:
            step = target
            cuts = [
                round(k * step, 3)
                for k in range(max(1, int(duration_sec // step) + 1))
                if k * step < duration_sec - 0.1
            ]
            plans[key] = ShotCutPlan(
                target_duration_sec=target,
                bars=0,
                aligned_duration_sec=round(target, 3),
                drift_sec=0.0,
                cut_points=tuple(cuts or [0.0]),
            )
    return plans


def resolve_analysis_out_file(out_file: Path | None, project_dir: Path | None) -> Path | None:
    """Resolve the destination path for ``audio-map.json``."""
    if out_file is not None:
        return out_file.expanduser().resolve()
    if project_dir is None:
        return None
    root = project_dir.expanduser().resolve()
    if (root / "audio" / "analysis").is_dir() or (root / "project.json").is_file():
        return root / "audio" / "analysis" / "audio-map.json"
    if (root / "assets" / "music").is_dir():
        return root / "assets" / "music" / "audio-map.json"
    return root / "audio" / "analysis" / "audio-map.json"


def _build_quality_warnings(
    *,
    loudness: _LoudnessStats,
    duration_sec: float,
    target_lufs: float,
    true_peak_ceiling_db: float,
    clipping_detected: bool,
    exceeds_ceiling: bool,
    is_silent: bool,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if is_silent:
        warnings.append("Audio track is silent or below -65 LUFS.")
    else:
        if clipping_detected:
            warnings.append(
                f"Digital clipping detected (true peak {loudness.true_peak_db:.2f} dBTP > -0.1)."
            )
        elif exceeds_ceiling:
            warnings.append(
                f"True peak {loudness.true_peak_db:.2f} dBTP exceeds {true_peak_ceiling_db:.1f} "
                "dBTP mastering ceiling."
            )
        lufs_diff = abs(loudness.lufs_integrated - target_lufs)
        if lufs_diff > 2.0:
            warnings.append(
                f"Integrated loudness {loudness.lufs_integrated:.1f} LUFS differs from target "
                f"{target_lufs:.1f} LUFS by {lufs_diff:.1f} LU."
            )
        if loudness.silence_ratio >= 0.3:
            warnings.append(
                f"High silence ratio ({loudness.silence_ratio * 100:.1f}% of {duration_sec:.1f}s)."
            )
    return tuple(warnings)


def measure_audio_quality(
    path: Path | str,
    *,
    include_bpm: bool = True,
    silence_threshold_db: float = -50.0,
    min_silence_sec: float = 0.5,
    target_lufs: float = -14.0,
    true_peak_ceiling_db: float = -1.0,
) -> AudioQualityReport:
    """Measure technical audio quality (LUFS, true peak, clipping, silence, optional BPM).

    Designed for direct reuse by ``quality_gate.py`` and audio critic checks.
    """
    resolved = _validate_input_file(path, route="audio.quality")
    info = _probe_audio_stream(resolved)
    loudness = _measure_ebur128_and_silence(
        resolved,
        duration_sec=info.duration_sec,
        silence_threshold_db=silence_threshold_db,
        min_silence_sec=min_silence_sec,
    )
    is_silent = loudness.silence_ratio >= 0.95 or loudness.lufs_integrated <= -65.0
    clipping_detected = not is_silent and loudness.true_peak_db > -0.1
    exceeds_ceiling = not is_silent and loudness.true_peak_db > true_peak_ceiling_db
    bpm: float | None = None
    confidence: float | None = None
    if include_bpm and not is_silent and info.duration_sec >= 0.5:
        rhythm = _analyze_rhythm(resolved, info.duration_sec)
        bpm = rhythm.bpm
        confidence = rhythm.tempo_confidence
    warnings = _build_quality_warnings(
        loudness=loudness,
        duration_sec=info.duration_sec,
        target_lufs=target_lufs,
        true_peak_ceiling_db=true_peak_ceiling_db,
        clipping_detected=clipping_detected,
        exceeds_ceiling=exceeds_ceiling,
        is_silent=is_silent,
    )
    return AudioQualityReport(
        source=str(resolved),
        codec=info.codec,
        sample_rate=info.sample_rate,
        channels=info.channels,
        bit_rate=info.bit_rate,
        duration_seconds=round(info.duration_sec, 3),
        integrated_lufs=round(loudness.lufs_integrated, 2),
        lra_lu=round(loudness.lra_lu, 2),
        true_peak_dbtp=round(loudness.true_peak_db, 2),
        clipping_detected=clipping_detected,
        exceeds_true_peak_ceiling=exceeds_ceiling,
        is_silent=is_silent,
        silence_ratio=round(loudness.silence_ratio, 4),
        silence_segments=tuple(seg.to_dict() for seg in loudness.silence_segments),
        bpm=round(bpm, 2) if bpm is not None else None,
        tempo_confidence=round(confidence, 3) if confidence is not None else None,
        warnings=warnings,
    )


def analyze_audio(
    path: Path | str,
    *,
    shot_durations: Sequence[float] = DEFAULT_SHOT_DURATIONS,
    out_file: Path | None = None,
    project_dir: Path | None = None,
    silence_threshold_db: float = -50.0,
    min_silence_sec: float = 0.5,
) -> AudioMap:
    """Analyze a FlowMusic or local audio track and optionally write ``audio-map.json``."""
    resolved = _validate_input_file(path, route="audio.analyze")
    info = _probe_audio_stream(resolved)
    loudness = _measure_ebur128_and_silence(
        resolved,
        duration_sec=info.duration_sec,
        silence_threshold_db=silence_threshold_db,
        min_silence_sec=min_silence_sec,
    )
    is_silent = loudness.silence_ratio >= 0.95 or loudness.lufs_integrated <= -65.0
    clipping_detected = not is_silent and loudness.true_peak_db > -0.1
    exceeds_ceiling = not is_silent and loudness.true_peak_db > -1.0
    rhythm = (
        _RhythmAnalysis("ffmpeg_numpy", 0.0, 0.0, (), (), (), (), ())
        if is_silent
        else _analyze_rhythm(resolved, info.duration_sec)
    )
    beat_interval = 60.0 / rhythm.bpm if rhythm.bpm > 0.0 else 0.0
    bar_duration = 4.0 * beat_interval if beat_interval > 0.0 else 0.0
    cut_plans = build_shot_cut_plans(
        info.duration_sec,
        bar_duration,
        rhythm.downbeats,
        shot_durations=shot_durations,
    )
    first_plan = next(iter(cut_plans.values()), None)
    recommended = first_plan.cut_points if first_plan is not None else rhythm.downbeats
    destination = resolve_analysis_out_file(out_file, project_dir)
    audio_map = AudioMap(
        version=AUDIO_MAP_VERSION,
        source=resolved,
        analyzer_backend=rhythm.backend,
        codec=info.codec,
        sample_rate=info.sample_rate,
        channels=info.channels,
        bit_rate=info.bit_rate,
        duration_sec=info.duration_sec,
        lufs_integrated=loudness.lufs_integrated,
        lra_lu=loudness.lra_lu,
        true_peak_db=loudness.true_peak_db,
        clipping_detected=clipping_detected,
        exceeds_true_peak_ceiling=exceeds_ceiling,
        is_silent=is_silent,
        silence_ratio=loudness.silence_ratio,
        silence_segments=loudness.silence_segments,
        bpm=rhythm.bpm,
        tempo_confidence=rhythm.tempo_confidence,
        beat_interval_sec=beat_interval,
        bar_duration_sec=bar_duration,
        beats=rhythm.beats,
        downbeats=rhythm.downbeats,
        onsets=rhythm.onsets,
        energy_envelope=rhythm.energy_envelope,
        energy_peaks=rhythm.energy_peaks,
        shot_cut_plans=cut_plans,
        recommended_cut_points=recommended,
        out_file=destination,
    )
    if destination is not None:
        audio_map.write_json(destination)
        _log.info("audio_analysis.wrote_map", path=str(destination), bpm=audio_map.bpm)
    return audio_map
