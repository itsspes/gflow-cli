"""Deterministic Stage-1 Quality Gate engine for images, videos, and audio."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from gflow_cli.services.project_workspace import record_asset_status

log = structlog.get_logger(__name__)

QA_VERDICTS: tuple[str, ...] = ("PASS_TECH", "NEEDS_MINOR_EDIT", "FAIL_TECH")
DECISION_VERDICTS: tuple[str, ...] = ("approve", "minor_edit", "reject")
_VERDICT_TO_STATUS: dict[str, str] = {
    "approve": "approved",
    "minor_edit": "minor_edit",
    "reject": "rejected",
}

_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"})
_VIDEO_EXTS = frozenset({".mp4", ".mov", ".mkv", ".webm", ".m4v"})
_AUDIO_EXTS = frozenset({".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"})

_BLACK_LUMA_THRESHOLD = 12.0
_SOLID_STDDEV_THRESHOLD = 2.0
_LOW_SHARPNESS_THRESHOLD = 8.0
_TRUE_PEAK_CEILING_DBTP = -1.0
_SILENT_LUFS_THRESHOLD = -60.0

_LUFS_RE = re.compile(r"I:\s*(-?\d+(?:\.\d+)?)\s*LUFS")
_PEAK_RE = re.compile(r"Peak:\s*(-?\d+(?:\.\d+)?)\s*dBFS")


def detect_media_kind(path: Path) -> str:
    """Infer ``image``, ``video``, or ``audio`` from file extension."""
    ext = path.suffix.lower()
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in _VIDEO_EXTS:
        return "video"
    if ext in _AUDIO_EXTS:
        return "audio"
    msg = f"Cannot infer media kind for {path.name!r}; pass --kind image|video|audio."
    raise ValueError(msg)


def _worst_verdict(issues: list[dict[str, str]]) -> str:
    severities = {item["severity"] for item in issues}
    if "fail" in severities:
        return "FAIL_TECH"
    if "warn" in severities:
        return "NEEDS_MINOR_EDIT"
    return "PASS_TECH"


def check_image(path: Path) -> dict[str, Any]:
    """Run deterministic technical checks on an image using Pillow."""
    from PIL import Image, ImageFilter, ImageStat

    issues: list[dict[str, str]] = []
    next_actions: list[str] = []
    try:
        with Image.open(path) as img:
            img.load()
            width, height = img.size
            mode = img.mode
            gray = img.convert("L")
            stat = ImageStat.Stat(gray)
            mean_luma = float(stat.mean[0])
            stddev_luma = float(stat.stddev[0])
            edges = gray.filter(ImageFilter.FIND_EDGES)
            sharpness = float(ImageStat.Stat(edges).stddev[0])
    except Exception as exc:  # noqa: BLE001 - report unreadable/corrupt image cleanly
        return {
            "kind": "image",
            "verdict": "FAIL_TECH",
            "metrics": {"readable": False},
            "issues": [{"code": "unreadable_image", "severity": "fail", "detail": str(exc)}],
            "next_actions": ["Regenerate or re-download the image file."],
        }

    if stddev_luma < _SOLID_STDDEV_THRESHOLD:
        code = "solid_black_frame" if mean_luma < _BLACK_LUMA_THRESHOLD else "solid_color_frame"
        issues.append(
            {
                "code": code,
                "severity": "fail",
                "detail": f"Image appears uniform/blank (stddev={stddev_luma:.2f}).",
            }
        )
        next_actions.append("Reject and regenerate; frame is blank or solid color.")
    elif sharpness < _LOW_SHARPNESS_THRESHOLD:
        issues.append(
            {
                "code": "low_sharpness",
                "severity": "warn",
                "detail": (
                    f"Edge sharpness proxy is low ({sharpness:.2f} < {_LOW_SHARPNESS_THRESHOLD})."
                ),
            }
        )
        next_actions.append("Inspect for blur; consider i2i refinement or upscale.")

    return {
        "kind": "image",
        "verdict": _worst_verdict(issues),
        "metrics": {
            "readable": True,
            "width": width,
            "height": height,
            "aspect_ratio": round(width / height, 4) if height else 0.0,
            "mode": mode,
            "mean_luma": round(mean_luma, 2),
            "stddev_luma": round(stddev_luma, 2),
            "sharpness_score": round(sharpness, 2),
        },
        "issues": issues,
        "next_actions": next_actions,
    }


def _ffprobe_json(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        msg = "ffprobe is not installed or not on PATH."
        raise RuntimeError(msg)
    completed = subprocess.run(  # noqa: S603 - resolved ffprobe binary, no shell
        [
            ffprobe,
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
        timeout=30.0,
        check=False,
    )
    if completed.returncode != 0:
        msg = completed.stderr.strip() or f"ffprobe exited with code {completed.returncode}"
        raise RuntimeError(msg)
    data: dict[str, Any] = json.loads(completed.stdout)
    return data


def _parse_fps(rate_str: str | None) -> float:
    if not rate_str or rate_str == "0/0":
        return 0.0
    if "/" in rate_str:
        num_s, den_s = rate_str.split("/", 1)
        den = float(den_s)
        return round(float(num_s) / den, 3) if den else 0.0
    return round(float(rate_str), 3)


def _sample_video_frames_and_strip(
    path: Path,
    *,
    fps: float,
    strip_path: Path | None,
) -> tuple[float, str | None]:
    """Sample ~1 fps via PyAV to measure black-frame ratio and build a contact sheet."""
    import av
    from PIL import Image, ImageStat

    step = max(1, round(fps)) if fps > 0 else 24
    thumbs: list[Image.Image] = []
    black_count = 0
    sampled = 0
    _av: Any = av
    with _av.open(str(path)) as container:
        if not container.streams.video:
            return 1.0, None
        for idx, frame in enumerate(container.decode(container.streams.video[0])):
            if idx % step != 0:
                continue
            pil_img = frame.to_image()
            gray = pil_img.convert("L")
            stat = ImageStat.Stat(gray)
            if float(stat.mean[0]) < _BLACK_LUMA_THRESHOLD and float(stat.stddev[0]) < 4.0:
                black_count += 1
            sampled += 1
            if strip_path is not None and len(thumbs) < 16:
                thumb = pil_img.copy()
                thumb.thumbnail((320, 180))
                thumbs.append(thumb)

    black_ratio = round(black_count / sampled, 4) if sampled else 1.0
    saved_strip: str | None = None
    if strip_path is not None and thumbs:
        strip_path.parent.mkdir(parents=True, exist_ok=True)
        total_w = sum(im.width for im in thumbs)
        max_h = max(im.height for im in thumbs)
        sheet = Image.new("RGB", (total_w, max_h), (0, 0, 0))
        x_offset = 0
        for im in thumbs:
            sheet.paste(im, (x_offset, 0))
            x_offset += im.width
        sheet.save(strip_path, format="JPEG", quality=85)
        saved_strip = str(strip_path)
    return black_ratio, saved_strip


def check_video(
    path: Path,
    *,
    target_duration: float | None = None,
    strip_dir: Path | None = None,
    expect_audio: bool = False,
) -> dict[str, Any]:
    """Run deterministic ffprobe, black-frame, audio-hygiene, and 1 fps strip checks."""
    issues: list[dict[str, str]] = []
    next_actions: list[str] = []
    try:
        probe = _ffprobe_json(path)
    except Exception as exc:  # noqa: BLE001 - report unreadable video cleanly
        return {
            "kind": "video",
            "verdict": "FAIL_TECH",
            "metrics": {"readable": False},
            "issues": [{"code": "ffprobe_failed", "severity": "fail", "detail": str(exc)}],
            "next_actions": ["Inspect or regenerate corrupt video file."],
        }

    streams: list[dict[str, Any]] = probe.get("streams", [])
    v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    a_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if v_stream is None:
        return {
            "kind": "video",
            "verdict": "FAIL_TECH",
            "metrics": {"readable": False, "has_video_stream": False},
            "issues": [
                {"code": "missing_video_stream", "severity": "fail", "detail": "No video stream."}
            ],
            "next_actions": ["Regenerate video; container has no video stream."],
        }

    width = int(v_stream.get("width", 0))
    height = int(v_stream.get("height", 0))
    fps = _parse_fps(v_stream.get("avg_frame_rate") or v_stream.get("r_frame_rate"))
    duration = float(probe.get("format", {}).get("duration") or v_stream.get("duration") or 0.0)
    has_audio = a_stream is not None

    strip_out = (strip_dir or path.parent) / f"{path.stem}_qa_strip.jpg"
    try:
        black_ratio, contact_sheet = _sample_video_frames_and_strip(
            path, fps=fps, strip_path=strip_out
        )
    except Exception as exc:  # noqa: BLE001 - best-effort frame sampling
        log.debug("quality_gate.video_sample_error", error=str(exc))
        black_ratio, contact_sheet = 0.0, None

    if expect_audio:
        has_unstripped_audio = False
        if not has_audio:
            issues.append(
                {
                    "code": "missing_expected_audio",
                    "severity": "warn",
                    "detail": (
                        "Expected an audio track on this mastered video, "
                        "but no audio stream was found."
                    ),
                }
            )
            next_actions.append(
                "Assemble with music/voiceover (`gflow video assemble --music ...`) "
                "or omit --expect-audio for silent B-roll."
            )
    else:
        has_unstripped_audio = has_audio
        if has_audio:
            issues.append(
                {
                    "code": "has_unstripped_audio",
                    "severity": "warn",
                    "detail": (
                        "Video carries an audio stream (Veo clips often include room-tone noise)."
                    ),
                }
            )
            next_actions.append(
                "Strip clip audio with `gflow video assemble --strip-veo-audio` for B-roll."
            )
    if black_ratio >= 0.8:
        issues.append(
            {
                "code": "excessive_black_frames",
                "severity": "fail",
                "detail": f"Black frame ratio is {black_ratio:.0%}.",
            }
        )
        next_actions.append("Reject and regenerate; clip is mostly black frames.")
    if target_duration is not None and abs(duration - target_duration) > max(
        0.5, target_duration * 0.15
    ):
        issues.append(
            {
                "code": "duration_mismatch",
                "severity": "warn",
                "detail": f"Duration {duration:.2f}s differs from target {target_duration:.2f}s.",
            }
        )
        next_actions.append("Trim or re-time clip during assembly to match target duration.")

    return {
        "kind": "video",
        "verdict": _worst_verdict(issues),
        "metrics": {
            "readable": True,
            "codec": str(v_stream.get("codec_name", "")),
            "width": width,
            "height": height,
            "fps": fps,
            "duration_seconds": round(duration, 3),
            "has_audio_stream": has_audio,
            "has_unstripped_audio": has_unstripped_audio,
            "expect_audio": expect_audio,
            "black_frame_ratio": black_ratio,
            "contact_sheet": contact_sheet,
        },
        "issues": issues,
        "next_actions": next_actions,
    }


def _probe_audio_ffmpeg(path: Path) -> dict[str, Any]:
    """Measure integrated LUFS, true peak, and duration using ffprobe + ffmpeg ebur128."""
    probe = _ffprobe_json(path)
    duration = float(probe.get("format", {}).get("duration") or 0.0)
    a_stream = next((s for s in probe.get("streams", []) if s.get("codec_type") == "audio"), None)
    if a_stream is None:
        msg = "No audio stream found."
        raise RuntimeError(msg)

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        msg = "ffmpeg is not installed or not on PATH."
        raise RuntimeError(msg)
    completed = subprocess.run(  # noqa: S603 - resolved ffmpeg binary, no shell
        [
            ffmpeg,
            "-nostats",
            "-i",
            str(path),
            "-filter_complex",
            "ebur128=peak=true",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    stderr = completed.stderr
    lufs_matches = _LUFS_RE.findall(stderr)
    peak_matches = _PEAK_RE.findall(stderr)
    lufs = float(lufs_matches[-1]) if lufs_matches else -70.0
    true_peak = float(peak_matches[-1]) if peak_matches else -70.0
    return {
        "duration_seconds": round(duration, 3),
        "sample_rate": int(a_stream.get("sample_rate") or 0),
        "channels": int(a_stream.get("channels") or 0),
        "integrated_lufs": round(lufs, 2),
        "true_peak_dbtp": round(true_peak, 2),
        "bpm": None,
    }


def _measure_audio_metrics(path: Path) -> dict[str, Any]:
    """Use ``audio_analysis.measure_audio_quality`` when available, else FFmpeg fallback."""
    import importlib

    try:
        mod = importlib.import_module("gflow_cli.services.audio_analysis")
        measure_fn: Any = getattr(mod, "measure_audio_quality", None)
        if callable(measure_fn):
            raw: Any = measure_fn(path)
            if isinstance(raw, dict):
                return raw  # type: ignore[return-value]
            to_dict = getattr(raw, "to_dict", None)
            if callable(to_dict):
                return dict(to_dict())  # type: ignore[arg-type]
            if hasattr(raw, "__dict__"):
                return dict(raw.__dict__)
    except Exception:  # noqa: BLE001 - fall back to built-in FFmpeg ebur128 probe
        pass
    return _probe_audio_ffmpeg(path)


def check_audio(
    path: Path,
    *,
    target_duration: float | None = None,
    target_bpm: float | None = None,
) -> dict[str, Any]:
    """Run deterministic loudness, clipping, silence, and duration/BPM checks on audio."""
    issues: list[dict[str, str]] = []
    next_actions: list[str] = []
    try:
        metrics = _measure_audio_metrics(path)
    except Exception as exc:  # noqa: BLE001 - report unreadable audio cleanly
        return {
            "kind": "audio",
            "verdict": "FAIL_TECH",
            "metrics": {"readable": False},
            "issues": [{"code": "audio_probe_failed", "severity": "fail", "detail": str(exc)}],
            "next_actions": ["Verify audio file integrity or re-export as WAV."],
        }

    lufs = float(metrics.get("integrated_lufs", -70.0))
    peak = float(metrics.get("true_peak_dbtp", -70.0))
    duration = float(metrics.get("duration_seconds", 0.0))
    bpm = metrics.get("bpm")

    if lufs <= _SILENT_LUFS_THRESHOLD:
        issues.append(
            {
                "code": "silent_audio",
                "severity": "fail",
                "detail": f"Audio is effectively silent ({lufs:.1f} LUFS).",
            }
        )
        next_actions.append("Reject silent export and re-render or regenerate track.")
    if peak > _TRUE_PEAK_CEILING_DBTP:
        issues.append(
            {
                "code": "true_peak_clipping",
                "severity": "warn",
                "detail": (
                    f"True peak {peak:.2f} dBTP exceeds {_TRUE_PEAK_CEILING_DBTP:.1f} dBTP ceiling."
                ),
            }
        )
        next_actions.append("Master with `gflow audio master` (-14 LUFS / -1.0 dBTP limiter).")
    if target_duration is not None and abs(duration - target_duration) > max(
        1.0, target_duration * 0.15
    ):
        issues.append(
            {
                "code": "duration_mismatch",
                "severity": "warn",
                "detail": (
                    f"Audio duration {duration:.2f}s differs from target {target_duration:.2f}s."
                ),
            }
        )
        next_actions.append("Trim on bar boundary with exponential fade-out.")
    if (
        target_bpm is not None
        and isinstance(bpm, (int, float))
        and abs(float(bpm) - target_bpm) > 6.0
    ):
        issues.append(
            {
                "code": "bpm_mismatch",
                "severity": "warn",
                "detail": f"Measured BPM {float(bpm):.1f} differs from target {target_bpm:.1f}.",
            }
        )
        next_actions.append("Adjust BPM in FlowMusic Producer chat or regenerate.")

    metrics["readable"] = True
    return {
        "kind": "audio",
        "verdict": _worst_verdict(issues),
        "metrics": metrics,
        "issues": issues,
        "next_actions": next_actions,
    }


def _write_review_json(project_dir: Path, stem: str, payload: dict[str, Any]) -> Path:
    reviews_dir = project_dir.expanduser().resolve() / "qa" / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    review_path = reviews_dir / f"{stem}_review.json"
    review_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return review_path


def run_qa_check(
    file_path: Path,
    *,
    kind: str = "auto",
    project_dir: Path | None = None,
    target_duration: float | None = None,
    target_bpm: float | None = None,
    expect_audio: bool = False,
) -> dict[str, Any]:
    """Run Stage-1 deterministic QA check and optionally persist ``review.json``."""
    target = file_path.expanduser().resolve()
    if not target.is_file():
        msg = f"File does not exist: {target}"
        raise FileNotFoundError(msg)

    resolved_kind = detect_media_kind(target) if kind == "auto" else kind
    strip_dir = (project_dir.expanduser().resolve() / "qa" / "reviews") if project_dir else None

    if resolved_kind == "image":
        result = check_image(target)
    elif resolved_kind == "video":
        result = check_video(
            target,
            target_duration=target_duration,
            strip_dir=strip_dir,
            expect_audio=expect_audio,
        )
    elif resolved_kind == "audio":
        result = check_audio(target, target_duration=target_duration, target_bpm=target_bpm)
    else:
        msg = f"Unsupported media kind {resolved_kind!r}; choose image, video, or audio."
        raise ValueError(msg)

    payload: dict[str, Any] = {
        "file": str(target),
        "checked_at": datetime.now(UTC).isoformat(),
        **result,
    }
    if project_dir is not None:
        review_path = _write_review_json(project_dir, target.stem, payload)
        payload["review_path"] = str(review_path)
    return payload


def _destination_for_decision(
    project_dir: Path,
    source: Path,
    *,
    kind: str,
    verdict: str,
) -> Path | None:
    if verdict not in ("approve", "reject"):
        return None
    bucket = "approved" if verdict == "approve" else "rejected"
    root = project_dir.expanduser().resolve()
    if kind == "image":
        dest_dir = root / "images" / bucket
    elif kind == "video":
        dest_dir = root / "videos" / bucket
    else:
        dest_dir = root / "audio" / bucket
    dest_dir.mkdir(parents=True, exist_ok=True)
    return dest_dir / source.name


def record_qa_decision(
    file_path: Path,
    *,
    verdict: str,
    notes: str = "",
    project_dir: Path | None = None,
    move_asset: bool = True,
) -> dict[str, Any]:
    """Record a critic decision (approve|minor_edit|reject) and route asset folders."""
    if verdict not in DECISION_VERDICTS:
        msg = f"Invalid verdict {verdict!r}; choose from {DECISION_VERDICTS}"
        raise ValueError(msg)
    source = file_path.expanduser().resolve()
    if not source.is_file():
        msg = f"File does not exist: {source}"
        raise FileNotFoundError(msg)

    try:
        kind = detect_media_kind(source)
    except ValueError:
        kind = "video"

    final_path = source
    review_path: Path | None = None
    if project_dir is not None:
        if move_asset:
            dest = _destination_for_decision(project_dir, source, kind=kind, verdict=verdict)
            if dest is not None and dest != source:
                shutil.move(str(source), str(dest))
                final_path = dest

        reviews_dir = project_dir.expanduser().resolve() / "qa" / "reviews"
        reviews_dir.mkdir(parents=True, exist_ok=True)
        review_path = reviews_dir / f"{source.stem}_review.json"
        existing: dict[str, Any] = {}
        if review_path.is_file():
            try:
                existing = json.loads(review_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                existing = {}
        existing["decision"] = {
            "verdict": verdict,
            "notes": notes,
            "decided_at": datetime.now(UTC).isoformat(),
            "destination": str(final_path),
        }
        review_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
        record_asset_status(
            project_dir,
            asset_path=final_path,
            status=_VERDICT_TO_STATUS[verdict],
            notes=notes,
            review_path=review_path,
        )

    return {
        "status": "recorded",
        "file": str(source),
        "destination": str(final_path),
        "verdict": verdict,
        "workspace_status": _VERDICT_TO_STATUS[verdict],
        "notes": notes,
        "review_path": str(review_path) if review_path else None,
    }
