"""Typed value objects (DTOs) for Google Flow Music (flowmusic.app).

Pure module — no I/O, no Playwright imports. Mirrors ``gflow_cli.api.dto``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast


def _str_or_none(val: object) -> str | None:
    if isinstance(val, str) and val.strip():
        return val.strip()
    return None


def _float_or_none(val: object) -> float | None:
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    return None


@dataclass(frozen=True)
class FlowMusicModel:
    """One model entry from ``GET /__api/models``."""

    public_name: str
    version: str = ""
    description: str = ""
    is_default: bool = False
    max_audio_upload_duration_s: float | None = None
    max_audio_upload_bytes: int | None = None

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> FlowMusicModel:
        max_bytes = raw.get("max_audio_upload_bytes")
        return cls(
            public_name=str(raw.get("public_name") or raw.get("name") or ""),
            version=str(raw.get("version") or ""),
            description=str(raw.get("description") or ""),
            is_default=bool(raw.get("is_default", False)),
            max_audio_upload_duration_s=_float_or_none(raw.get("max_audio_upload_duration_s")),
            max_audio_upload_bytes=int(max_bytes) if isinstance(max_bytes, int) else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "public_name": self.public_name,
            "version": self.version,
            "description": self.description,
            "is_default": self.is_default,
            "max_audio_upload_duration_s": self.max_audio_upload_duration_s,
            "max_audio_upload_bytes": self.max_audio_upload_bytes,
        }


@dataclass(frozen=True)
class FlowMusicCredits:
    """Billing/quota snapshot from ``/__api/billing/credits`` & ``subscription``."""

    credits_remaining: float | None = None
    tokens_remaining: float | None = None
    plan: str | None = None
    subscription_status: str | None = None

    @classmethod
    def from_api(
        cls,
        credits_payload: dict[str, Any] | None,
        subscription_payload: dict[str, Any] | None = None,
    ) -> FlowMusicCredits:
        data: dict[str, Any] = {}
        if isinstance(credits_payload, dict):
            nested = credits_payload.get("data")
            data = cast("dict[str, Any]", nested) if isinstance(nested, dict) else credits_payload
        sub: dict[str, Any] = {}
        if isinstance(subscription_payload, dict):
            nested_sub = subscription_payload.get("data")
            sub = (
                cast("dict[str, Any]", nested_sub)
                if isinstance(nested_sub, dict)
                else subscription_payload
            )
        plan = _str_or_none(sub.get("plan") or sub.get("tier") or sub.get("plan_name"))
        sub_status = _str_or_none(sub.get("status") or sub.get("state"))
        return cls(
            credits_remaining=_float_or_none(data.get("credits_remaining")),
            tokens_remaining=_float_or_none(data.get("tokens_remaining")),
            plan=plan,
            subscription_status=sub_status,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "credits_remaining": self.credits_remaining,
            "tokens_remaining": self.tokens_remaining,
            "plan": self.plan,
            "subscription_status": self.subscription_status,
        }


@dataclass(frozen=True)
class FlowMusicClip:
    """One generated music clip from ``/__api/clips/auth-user``."""

    id: str
    title: str = ""
    audio_url: str | None = None
    wav_url: str | None = None
    image_url: str | None = None
    video_url: str | None = None
    duration: float | None = None
    lyrics: str | None = None
    lyrics_timing: Any = None
    has_vocals: bool | None = None
    is_favorite: bool = False
    is_remix_eligible: bool = False
    created_at: str | None = None
    op_id: str | None = None
    op_type: str | None = None

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> FlowMusicClip:
        vocals = raw.get("has_vocals")
        return cls(
            id=str(raw.get("id") or ""),
            title=str(raw.get("title") or ""),
            audio_url=_str_or_none(raw.get("audio_url")),
            wav_url=_str_or_none(raw.get("wav_url")),
            image_url=_str_or_none(raw.get("image_url")),
            video_url=_str_or_none(raw.get("video_url")),
            duration=_float_or_none(raw.get("duration")),
            lyrics=_str_or_none(raw.get("lyrics")),
            lyrics_timing=raw.get("lyrics_timing"),
            has_vocals=bool(vocals) if isinstance(vocals, bool) else None,
            is_favorite=bool(raw.get("is_favorite", False)),
            is_remix_eligible=bool(raw.get("is_remix_eligible", False)),
            created_at=_str_or_none(raw.get("created_at")),
            op_id=_str_or_none(raw.get("op_id")),
            op_type=_str_or_none(raw.get("op_type")),
        )

    def preferred_download_url(self, preferred_format: str = "wav") -> tuple[str | None, str]:
        """Return ``(url, ext)`` honoring ``preferred_format`` with safe fallback."""
        pref = preferred_format.lstrip(".").lower()
        if pref == "wav" and self.wav_url:
            return self.wav_url, "wav"
        if pref == "m4a" and self.audio_url:
            return self.audio_url, "m4a"
        if self.wav_url:
            return self.wav_url, "wav"
        if self.audio_url:
            return self.audio_url, "m4a"
        return None, pref or "wav"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "audio_url": self.audio_url,
            "wav_url": self.wav_url,
            "image_url": self.image_url,
            "video_url": self.video_url,
            "duration": self.duration,
            "lyrics": self.lyrics,
            "has_lyrics_timing": self.lyrics_timing is not None,
            "has_vocals": self.has_vocals,
            "is_favorite": self.is_favorite,
            "is_remix_eligible": self.is_remix_eligible,
            "created_at": self.created_at,
            "op_id": self.op_id,
            "op_type": self.op_type,
        }


@dataclass(frozen=True)
class FlowMusicSession:
    """One Producer conversation session from ``/__api/conversations``."""

    id: str
    title: str = ""
    created_at: str | None = None
    last_message_at: str | None = None
    project_id: str | None = None
    kind: str | None = None

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> FlowMusicSession:
        return cls(
            id=str(raw.get("id") or ""),
            title=str(raw.get("title") or ""),
            created_at=_str_or_none(raw.get("created_at")),
            last_message_at=_str_or_none(raw.get("last_message_at")),
            project_id=_str_or_none(raw.get("project_id")),
            kind=_str_or_none(raw.get("kind")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "last_message_at": self.last_message_at,
            "project_id": self.project_id,
            "kind": self.kind,
        }


def _empty_metadata() -> dict[str, Any]:
    return {}


@dataclass(frozen=True)
class FlowMusicResourceItem:
    """Generic cloud library item (playlist, project, space, or music video)."""

    id: str
    title: str = ""
    kind: str = ""
    created_at: str | None = None
    url: str | None = None
    metadata: dict[str, Any] = field(default_factory=_empty_metadata)

    @classmethod
    def from_api(cls, raw: dict[str, Any], *, kind: str) -> FlowMusicResourceItem:
        title = str(raw.get("title") or raw.get("name") or "")
        url = _str_or_none(raw.get("video_url") or raw.get("url") or raw.get("image_url"))
        extra = {
            k: v
            for k, v in raw.items()
            if k not in {"id", "title", "name", "created_at", "video_url", "url", "image_url"}
            and isinstance(v, (str, int, float, bool))
        }
        return cls(
            id=str(raw.get("id") or ""),
            title=title,
            kind=kind,
            created_at=_str_or_none(raw.get("created_at")),
            url=url,
            metadata=extra,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "kind": self.kind,
            "created_at": self.created_at,
            "url": self.url,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class FlowMusicAuthResult:
    """Result of a Flow Music login or auth status check."""

    signed_in: bool
    url: str
    profile_dir: str
    auth_prompt_visible: bool
    composer_visible: bool
    detail: str = ""
    account: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "signed_in" if self.signed_in else "signed_out",
            "signed_in": self.signed_in,
            "url": self.url,
            "profile_dir": self.profile_dir,
            "auth_prompt_visible": self.auth_prompt_visible,
            "composer_visible": self.composer_visible,
            "account": self.account,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class FlowMusicGenerateResult:
    """Result of generating and saving a Flow Music track (compose or chat)."""

    output_file: Path
    format: str
    bytes: int
    prompt: str
    source_url: str | None
    capture_method: str
    clip_id: str | None = None
    title: str | None = None
    session_id: str | None = None
    duration: float | None = None
    analysis_file: Path | None = None
    master_file: Path | None = None
    qa_verdict: str | None = None
    qa_review_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "succeeded",
            "output_file": str(self.output_file),
            "format": self.format,
            "bytes": self.bytes,
            "prompt": self.prompt,
            "source_url": self.source_url,
            "capture_method": self.capture_method,
            "clip_id": self.clip_id,
            "title": self.title,
            "session_id": self.session_id,
            "duration": self.duration,
            "analysis_file": str(self.analysis_file) if self.analysis_file else None,
            "master_file": str(self.master_file) if self.master_file else None,
            "qa_verdict": self.qa_verdict,
            "qa_review_path": str(self.qa_review_path) if self.qa_review_path else None,
        }


@dataclass(frozen=True)
class FlowMusicDownloadResult:
    """Result of downloading an existing Flow Music clip and optional sidecars."""

    clip_id: str
    output_file: Path
    format: str
    bytes: int
    source_url: str
    title: str = ""
    cover_file: Path | None = None
    lyrics_timing_file: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "succeeded",
            "clip_id": self.clip_id,
            "title": self.title,
            "output_file": str(self.output_file),
            "format": self.format,
            "bytes": self.bytes,
            "source_url": self.source_url,
            "cover_file": str(self.cover_file) if self.cover_file else None,
            "lyrics_timing_file": str(self.lyrics_timing_file) if self.lyrics_timing_file else None,
        }


@dataclass(frozen=True)
class FlowMusicChatResult:
    """Result of sending a Producer chat turn on Flow Music."""

    session_id: str | None
    session_url: str
    prompt: str
    response_text: str = ""
    clips: tuple[FlowMusicClip, ...] = ()
    downloaded: tuple[FlowMusicGenerateResult, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "succeeded",
            "session_id": self.session_id,
            "session_url": self.session_url,
            "prompt": self.prompt,
            "response_text": self.response_text,
            "clips": [c.to_dict() for c in self.clips],
            "downloaded": [d.to_dict() for d in self.downloaded],
        }
