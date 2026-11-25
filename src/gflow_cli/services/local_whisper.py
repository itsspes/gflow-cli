"""Modular local transcription engine (Apple Silicon M4 MLX Whisper + faster-whisper).

Accepts any audio or video file (extracts mono 16 kHz PCM WAV via FFmpeg into a
temporary directory when needed). Optional whisper wheels are strictly lazy-imported
so CI, Linux/Windows environments, and strict type checking never require them.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import structlog

from gflow_cli.errors import ConfigurationError, VideoAssemblyError

_log = structlog.get_logger(__name__)

VALID_LOCAL_BACKENDS: tuple[str, ...] = ("auto", "mlx", "faster_whisper")
DEFAULT_MLX_MODEL: str = "mlx-community/whisper-small-mlx"
DEFAULT_FASTER_WHISPER_MODEL: str = "base"
SUPPORTED_MEDIA_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".wav",
        ".mp3",
        ".m4a",
        ".aac",
        ".flac",
        ".ogg",
        ".opus",
        ".mp4",
        ".mov",
        ".mkv",
        ".webm",
        ".m4v",
    }
)


class LocalWhisperUnavailableError(ConfigurationError):
    """Raised when no local Whisper backend (mlx-whisper / faster-whisper) is installed."""

    problem_type = "https://gflow-cli.dev/errors/local-whisper-unavailable"
    charged = "no"
    next_action = "fix_input"
    title = "Local Whisper backend unavailable"
    _default_remediation = (
        "Install a local Whisper backend: `uv pip install mlx-whisper` on Apple Silicon macOS "
        "(or `pip install 'gflow-cli[mlx]'`), `uv pip install faster-whisper` on CPU/CUDA, "
        "or pass `--backend remote` to use a configured remote Whisper endpoint."
    )


@dataclass(frozen=True)
class TranscriptWord:
    """Single word-level timestamp within a transcript segment."""

    word: str
    start: float
    end: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
        }


@dataclass(frozen=True)
class TranscriptSegment:
    """Timed speech segment with optional word-level timestamps."""

    id: int
    start: float
    end: float
    text: str
    words: tuple[TranscriptWord, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "words": [w.to_dict() for w in self.words],
        }


@dataclass(frozen=True)
class LocalTranscriptResult:
    """Structured local transcription output shared by CLI, video QA, and MCP tools."""

    source: str
    backend: str
    model: str
    language: str | None
    text: str
    duration_seconds: float | None
    segments: tuple[TranscriptSegment, ...]
    srt_path: Path | None = None
    json_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "source": self.source,
            "backend": self.backend,
            "model": self.model,
            "language": self.language,
            "text": self.text,
            "duration_seconds": (
                round(self.duration_seconds, 3) if self.duration_seconds is not None else None
            ),
            "segments_count": len(self.segments),
            "segments": [seg.to_dict() for seg in self.segments],
            "srt_path": str(self.srt_path) if self.srt_path is not None else None,
            "json_path": str(self.json_path) if self.json_path is not None else None,
        }


def _format_srt_timestamp(seconds: float) -> str:
    millis = max(0, round(seconds * 1000.0))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def render_srt(segments: Sequence[TranscriptSegment]) -> str:
    """Render transcript segments as standard SubRip (``.srt``) text."""
    blocks: list[str] = []
    cue_index = 1
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        start = max(0.0, seg.start)
        end = seg.end if seg.end > start else start + 0.001
        blocks.append(
            f"{cue_index}\n"
            f"{_format_srt_timestamp(start)} --> {_format_srt_timestamp(end)}\n"
            f"{text}"
        )
        cue_index += 1
    return ("\n\n".join(blocks) + "\n") if blocks else ""


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine().lower() in {"arm64", "aarch64"}


def _module_available(module_name: str) -> bool:
    if module_name in sys.modules:
        loaded: object = sys.modules.get(module_name)
        return loaded is not None
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def detect_local_whisper_backend() -> str | None:
    """Detect the fastest installed local Whisper backend (``mlx`` on M-series macOS first)."""
    if _is_apple_silicon() and _module_available("mlx_whisper"):
        return "mlx"
    if _module_available("faster_whisper"):
        return "faster_whisper"
    return None


def _remediation_for_backend(requested: str) -> str:
    if requested == "mlx" or _is_apple_silicon():
        return (
            "Install Apple Silicon MLX Whisper with `uv pip install mlx-whisper` "
            "(or `pip install 'gflow-cli[mlx]'`), alternatively `uv pip install faster-whisper`, "
            "or pass `--backend remote`."
        )
    return (
        "Install local Whisper with `uv pip install faster-whisper`, "
        "or pass `--backend remote` to use a remote Whisper endpoint."
    )


def _resolve_backend(requested: str) -> str:
    normalized = requested.strip().lower()
    if normalized not in VALID_LOCAL_BACKENDS:
        choices = ", ".join(VALID_LOCAL_BACKENDS)
        raise ConfigurationError(
            detail=f"Unsupported local Whisper backend {requested!r}; expected one of: {choices}.",
            route="audio.transcribe",
            charged="no",
            remediation_hint=f"Pass --backend with one of: {choices}.",
        )
    if normalized == "auto":
        detected = detect_local_whisper_backend()
        if detected is None:
            raise LocalWhisperUnavailableError(
                detail="No local Whisper backend (mlx-whisper or faster-whisper) is installed.",
                route="audio.transcribe",
                remediation_hint=_remediation_for_backend("auto"),
            )
        return detected
    module_name = "mlx_whisper" if normalized == "mlx" else "faster_whisper"
    if not _module_available(module_name):
        raise LocalWhisperUnavailableError(
            detail=(
                f"Requested local Whisper backend {normalized!r} ({module_name}) is not installed."
            ),
            route="audio.transcribe",
            remediation_hint=_remediation_for_backend(normalized),
        )
    return normalized


def _find_binary(name: str) -> str:
    binary = shutil.which(name)
    if binary is None:
        raise VideoAssemblyError(
            detail=f"Required binary {name!r} was not found on PATH.",
            route="audio.transcribe",
            charged="no",
            remediation_hint=f"Install FFmpeg (`brew install ffmpeg`) so `{name}` is available.",
        )
    return binary


def extract_audio_to_wav(
    media_path: Path | str,
    wav_out: Path,
    *,
    sample_rate: int = 16000,
) -> Path:
    """Extract mono 16 kHz PCM WAV from any audio or video file using FFmpeg."""
    source = Path(media_path).expanduser().resolve()
    ffmpeg = _find_binary("ffmpeg")
    wav_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-v",
        "error",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(wav_out),
    ]
    proc = subprocess.run(  # noqa: S603 - resolved ffmpeg binary, fixed argv list, no shell
        cmd,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if proc.returncode != 0 or not wav_out.is_file() or wav_out.stat().st_size == 0:
        err_tail = proc.stderr.strip()[-300:] or "no audio stream extracted"
        raise VideoAssemblyError(
            detail=f"Failed to extract audio from {source.name}: {err_tail}",
            route="audio.transcribe",
            charged="no",
            remediation_hint="Ensure the input file contains a valid audio track.",
        )
    return wav_out


def _probe_duration_seconds(path: Path) -> float | None:
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as wf:
                rate = wf.getframerate()
                if rate > 0:
                    return wf.getnframes() / float(rate)
        except (wave.Error, OSError):
            pass
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    proc = subprocess.run(  # noqa: S603 - resolved ffprobe binary, fixed argv list, no shell
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if proc.returncode != 0:
        return None
    try:
        data = cast("dict[str, Any]", json.loads(proc.stdout or "{}"))
        raw = cast("dict[str, Any]", data.get("format") or {}).get("duration")
        return float(raw) if raw is not None else None
    except (ValueError, TypeError):
        return None


def _coerce_words(raw_words: Any) -> tuple[TranscriptWord, ...]:
    if not raw_words:
        return ()
    words: list[TranscriptWord] = []
    for item in raw_words:
        if isinstance(item, dict):
            d = cast("dict[str, Any]", item)
            word_text = str(d.get("word") or d.get("text") or "").strip()
            start = float(d.get("start") or 0.0)
            end = float(d.get("end") or start)
        else:
            word_text = str(getattr(item, "word", "") or getattr(item, "text", "")).strip()
            start = float(getattr(item, "start", 0.0) or 0.0)
            end = float(getattr(item, "end", start) or start)
        if word_text:
            words.append(TranscriptWord(word=word_text, start=start, end=max(start, end)))
    return tuple(words)


def _coerce_segment(item: Any, fallback_id: int) -> TranscriptSegment:
    if isinstance(item, dict):
        d = cast("dict[str, Any]", item)
        seg_id = int(d.get("id", fallback_id))
        start = float(d.get("start") or 0.0)
        end = float(d.get("end") or start)
        text = str(d.get("text") or "").strip()
        words = _coerce_words(d.get("words"))
    else:
        seg_id = int(getattr(item, "id", fallback_id))
        start = float(getattr(item, "start", 0.0) or 0.0)
        end = float(getattr(item, "end", start) or start)
        text = str(getattr(item, "text", "") or "").strip()
        words = _coerce_words(getattr(item, "words", None))
    return TranscriptSegment(
        id=seg_id,
        start=start,
        end=max(start, end),
        text=text,
        words=words,
    )


def _run_mlx_whisper(
    audio_path: Path,
    *,
    model: str,
    language: str | None,
    word_timestamps: bool,
) -> tuple[str, str | None, tuple[TranscriptSegment, ...]]:
    try:
        mlx_whisper: Any = importlib.import_module("mlx_whisper")
    except ImportError as exc:
        raise LocalWhisperUnavailableError(
            detail="mlx_whisper could not be imported.",
            route="audio.transcribe",
            remediation_hint=_remediation_for_backend("mlx"),
        ) from exc

    kwargs: dict[str, Any] = {
        "path_or_hf_repo": model,
        "word_timestamps": word_timestamps,
    }
    if language:
        kwargs["language"] = language
    raw: Any = mlx_whisper.transcribe(str(audio_path), **kwargs)
    payload = cast("dict[str, Any]", raw if isinstance(raw, dict) else {})
    raw_segments: list[Any] = list(payload.get("segments") or [])
    segments = tuple(
        _coerce_segment(seg, idx) for idx, seg in enumerate(raw_segments, start=1)
    )
    text = str(payload.get("text") or "").strip()
    if not text and segments:
        text = " ".join(seg.text for seg in segments if seg.text).strip()
    detected_lang = payload.get("language")
    resolved_lang = (
        str(detected_lang) if isinstance(detected_lang, str) and detected_lang else language
    )
    return text, resolved_lang, segments


def _run_faster_whisper(
    audio_path: Path,
    *,
    model: str,
    language: str | None,
    word_timestamps: bool,
) -> tuple[str, str | None, tuple[TranscriptSegment, ...]]:
    try:
        fw: Any = importlib.import_module("faster_whisper")
    except ImportError as exc:
        raise LocalWhisperUnavailableError(
            detail="faster_whisper could not be imported.",
            route="audio.transcribe",
            remediation_hint=_remediation_for_backend("faster_whisper"),
        ) from exc

    whisper_cls: Any = fw.WhisperModel
    try:
        engine = whisper_cls(model, device="auto", compute_type="int8")
    except TypeError:
        engine = whisper_cls(model)
    raw_segments, info = engine.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=word_timestamps,
    )
    segments = tuple(
        _coerce_segment(seg, idx) for idx, seg in enumerate(raw_segments, start=1)
    )
    text = " ".join(seg.text for seg in segments if seg.text).strip()
    detected_lang = getattr(info, "language", None)
    resolved_lang = (
        str(detected_lang) if isinstance(detected_lang, str) and detected_lang else language
    )
    return text, resolved_lang, segments


def _atomic_write_text(path: Path, content: str) -> Path:
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(target)
    return target


def transcribe_local(
    media_path: Path | str,
    *,
    backend: str = "auto",
    model: str | None = None,
    language: str | None = None,
    word_timestamps: bool = True,
    out_srt: Path | None = None,
    out_json: Path | None = None,
) -> LocalTranscriptResult:
    """Transcribe any local audio or video file using MLX Whisper or faster-whisper."""
    source = Path(media_path).expanduser().resolve()
    if not source.is_file():
        raise ConfigurationError(
            detail=f"Media file does not exist: {source}",
            route="audio.transcribe",
            charged="no",
            remediation_hint="Provide an existing local audio or video file path.",
        )

    resolved_backend = _resolve_backend(backend)
    resolved_model = model or (
        DEFAULT_MLX_MODEL if resolved_backend == "mlx" else DEFAULT_FASTER_WHISPER_MODEL
    )
    duration_seconds = _probe_duration_seconds(source)

    with tempfile.TemporaryDirectory(prefix="gflow-whisper-") as tmp_dir:
        if source.suffix.lower() == ".wav":
            whisper_input = source
        else:
            whisper_input = extract_audio_to_wav(source, Path(tmp_dir) / "extracted_16k.wav")
            if duration_seconds is None:
                duration_seconds = _probe_duration_seconds(whisper_input)

        if resolved_backend == "mlx":
            text, resolved_lang, segments = _run_mlx_whisper(
                whisper_input,
                model=resolved_model,
                language=language,
                word_timestamps=word_timestamps,
            )
        else:
            text, resolved_lang, segments = _run_faster_whisper(
                whisper_input,
                model=resolved_model,
                language=language,
                word_timestamps=word_timestamps,
            )

    if duration_seconds is None and segments:
        duration_seconds = segments[-1].end

    written_srt = _atomic_write_text(out_srt, render_srt(segments)) if out_srt is not None else None
    result = LocalTranscriptResult(
        source=str(source),
        backend=resolved_backend,
        model=resolved_model,
        language=resolved_lang,
        text=text,
        duration_seconds=duration_seconds,
        segments=segments,
        srt_path=written_srt,
        json_path=out_json.expanduser().resolve() if out_json is not None else None,
    )
    if out_json is not None:
        written_json = _atomic_write_text(out_json, json.dumps(result.to_dict(), indent=2) + "\n")
        result = LocalTranscriptResult(
            source=result.source,
            backend=result.backend,
            model=result.model,
            language=result.language,
            text=result.text,
            duration_seconds=result.duration_seconds,
            segments=result.segments,
            srt_path=result.srt_path,
            json_path=written_json,
        )
    _log.info(
        "local_whisper.completed",
        backend=resolved_backend,
        model=resolved_model,
        segments=len(segments),
        source=str(source),
    )
    return result
