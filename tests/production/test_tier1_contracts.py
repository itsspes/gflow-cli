"""Tier 1: Deterministic Unit & Contract Gates.

Covers CLI surface & flag contracts, JSON output schemas, stable exit codes,
2026-12-05 model retirement & migration hints, and credit budget guards.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from structlog.testing import capture_logs

from gflow_cli import json_output
from gflow_cli.api.credit_cap import resolve_max_credits
from gflow_cli.api.model_catalog import (
    RECOMMENDED_REPLACEMENT,
    RETIRES_ON,
    recommended_replacement,
    retirement_hint,
    retires_on,
    warn_if_retiring,
)
from gflow_cli.api.video import (
    Aspect,
    GenerateVideoRequest,
    Mode,
    VideoModel,
    VideoResult,
    VideoStatus,
)
from gflow_cli.cli import main
from gflow_cli.cli_models import build_catalog
from gflow_cli.config import Settings
from gflow_cli.errors import (
    AuthExpiredError,
    ConfigurationError,
    CreditCapExceededError,
    ProfileLockedError,
    exit_code_for,
    legacy_exit_code_for,
)

# ---------------------------------------------------------------------------
# 1. CLI Surface & Flag Contracts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "expected_flags"),
    [
        (["--help"], ["--version", "video", "image", "models"]),
        (["video", "t2v", "--help"], ["--model", "--max-credits", "--no-max-credits", "--json"]),
        (["video", "i2v", "--help"], ["--model", "--max-credits", "--no-max-credits", "--json"]),
        (["video", "r2v", "--help"], ["--model", "--max-credits", "--no-max-credits", "--json"]),
        (["video", "chain", "--help"], ["--dry-run", "--max-credits", "--no-max-credits"]),
        (["video", "assemble", "--help"], ["--strip-veo-audio", "--out-file", "--json"]),
        (["models", "--help"], ["--json"]),
    ],
)
def test_cli_help_exposes_production_flags(argv: list[str], expected_flags: list[str]) -> None:
    result = CliRunner().invoke(main, argv)
    assert result.exit_code == 0, result.output
    for flag in expected_flags:
        assert flag in result.output, f"Missing expected flag {flag!r} in `gflow {' '.join(argv)}`"


def test_models_json_command_emits_valid_catalog_with_replacements() -> None:
    result = CliRunner().invoke(main, ["models", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert "video" in payload
    video_models: list[dict[str, Any]] = payload["video"]["models"]
    by_name = {entry["name"]: entry for entry in video_models}

    for retiring_model, replacement in RECOMMENDED_REPLACEMENT.items():
        entry = by_name[retiring_model.value]
        assert entry["retires_on"] == "2026-12-05"
        assert entry["recommended_replacement"] == replacement.value

    for active_model in (VideoModel.OMNI_1_2_FLASH, VideoModel.VEO_3_1_LITE):
        entry = by_name[active_model.value]
        assert entry.get("retires_on") is None


def test_negative_max_credits_rejected_with_usage_error() -> None:
    result = CliRunner().invoke(main, ["video", "t2v", "studio shot", "--max-credits", "-5"])
    assert result.exit_code == 2


# ---------------------------------------------------------------------------
# 2. JSON Output & Exit Code Contracts
# ---------------------------------------------------------------------------


def test_video_result_json_schema_contract(tmp_path: Path) -> None:
    req = GenerateVideoRequest(
        prompt="slow push-in on ceramic espresso cup",
        mode=Mode.I2V,
        aspect=Aspect.LANDSCAPE,
        model=VideoModel.OMNI_1_2_FLASH,
        duration=8,
        start_image=Path("hero.png"),
    )
    out_file = tmp_path / "shot.mp4"
    result = VideoResult(
        status=VideoStatus(media_id="media-001", status="MEDIA_GENERATION_STATUS_SUCCESSFUL"),
        local_path=out_file,
    )
    payload = json_output.video_result(command="video i2v", request=req, result=result)
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)

    assert decoded["status"] == "ok"
    assert decoded["succeeded"] is True
    assert decoded["command"] == "video i2v"
    assert decoded["media_id"] == "media-001"
    assert decoded["local_path"] == str(out_file)
    assert decoded["request"]["model"] == "omni_1_2_flash"
    assert decoded["request"]["duration"] == 8


def test_critical_exit_codes_stable_across_legacy_and_v2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert legacy_exit_code_for(CreditCapExceededError("over budget")) == 45
    assert legacy_exit_code_for(AuthExpiredError("expired")) == 3
    assert legacy_exit_code_for(ConfigurationError("bad cfg")) == 11

    monkeypatch.setenv("GFLOW_CLI_EXIT_CODES", "2")
    assert exit_code_for(CreditCapExceededError("over budget")) == 45
    assert exit_code_for(AuthExpiredError("expired")) == 3
    assert exit_code_for(ConfigurationError("bad cfg")) == 11
    assert exit_code_for(ProfileLockedError("locked")) != 0


# ---------------------------------------------------------------------------
# 3. 2026-12-05 Model Retirement Compliance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("retiring", "expected_successor", "expected_cli_flag"),
    [
        (VideoModel.OMNI_FLASH, VideoModel.OMNI_1_2_FLASH, "--model omni-1.2-flash"),
        (VideoModel.VEO_3_1_FAST, VideoModel.VEO_3_1_LITE, "--model veo-lite"),
        (VideoModel.VEO_3_1_QUALITY, VideoModel.VEO_3_1_LITE, "--model veo-lite"),
    ],
)
def test_retiring_models_have_2026_12_05_date_and_actionable_hint(
    retiring: VideoModel,
    expected_successor: VideoModel,
    expected_cli_flag: str,
) -> None:
    assert retires_on(retiring) == date(2026, 12, 5)
    assert recommended_replacement(retiring) is expected_successor

    hint_before = retirement_hint(retiring, today=date(2026, 12, 4))
    assert hint_before is not None
    assert "retires on 2026-12-05" in hint_before
    assert expected_cli_flag in hint_before

    hint_after = retirement_hint(retiring, today=date(2026, 12, 5))
    assert hint_after is not None
    assert "retired on 2026-12-05" in hint_after
    assert expected_cli_flag in hint_after

    with capture_logs() as logs:
        notice = warn_if_retiring(retiring, today=date(2026, 12, 5))
    assert notice is not None
    assert notice.past is True
    assert notice.replacement is expected_successor
    assert expected_cli_flag in notice.hint
    assert any(
        entry["event"] == "model.retiring"
        and entry["replacement"] == expected_successor.value
        and entry["past"] is True
        for entry in logs
    )


@pytest.mark.parametrize("active_model", [VideoModel.OMNI_1_2_FLASH, VideoModel.VEO_3_1_LITE])
def test_non_retiring_models_emit_no_retirement_warning(active_model: VideoModel) -> None:
    assert active_model not in RETIRES_ON
    assert retires_on(active_model) is None
    assert recommended_replacement(active_model) is None
    assert retirement_hint(active_model, today=date(2026, 12, 6)) is None
    with capture_logs() as logs:
        assert warn_if_retiring(active_model, today=date(2026, 12, 6)) is None
    assert not logs


def test_build_catalog_lists_omni_1_2_and_veo_lite_as_active() -> None:
    catalog = build_catalog()
    video_by_name = {m["name"]: m for m in catalog["video"]["models"]}
    assert "omni_1_2_flash" in video_by_name
    assert "veo_3_1_lite" in video_by_name
    assert video_by_name["omni_1_2_flash"]["retires_on"] is None
    assert video_by_name["veo_3_1_lite"]["retires_on"] is None


# ---------------------------------------------------------------------------
# 4. Credit Budget & New Upgrade Settings Contracts
# ---------------------------------------------------------------------------


def test_credit_cap_resolution_precedence() -> None:
    assert resolve_max_credits(25, 10) == 25
    assert resolve_max_credits(None, 10) == 10
    assert resolve_max_credits(None, None) is None


def test_upgrade_settings_defaults_and_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    defaults = Settings()
    assert defaults.macos_background_launch is True
    assert defaults.warm_session_enabled is False
    assert defaults.warm_session_ttl_seconds == 1200.0
    assert defaults.video_hw_accel is True

    monkeypatch.setenv("GFLOW_CLI_MACOS_BACKGROUND_LAUNCH", "0")
    monkeypatch.setenv("GFLOW_CLI_WARM_SESSION_ENABLED", "1")
    monkeypatch.setenv("GFLOW_CLI_WARM_SESSION_TTL_SECONDS", "300")
    monkeypatch.setenv("GFLOW_CLI_VIDEO_HW_ACCEL", "0")
    overridden = Settings()
    assert overridden.macos_background_launch is False
    assert overridden.warm_session_enabled is True
    assert overridden.warm_session_ttl_seconds == 300.0
    assert overridden.video_hw_accel is False
