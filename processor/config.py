"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from processor.exceptions import ConfigError


@dataclass(frozen=True)
class Config:
    icloud_root: Path
    output_dir: Path
    background_music: Path
    music_volume_db: float
    voiceover_gain_db: float
    caption_template_id: str
    mirage_api_key: str
    poll_interval_seconds: int
    max_poll_attempts: int
    video_trim_extra_seconds: float
    max_file_size_mb: int
    output_width: int
    output_height: int
    enable_hdr: bool
    video_crf: int
    mirage_upload_crf: int
    mirage_base_url: str = "https://api.mirage.app/v1"

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


def _expand_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def load_config(config_path: Path) -> Config:
    if not config_path.exists():
        raise ConfigError(
            f"Config file not found: {config_path}\n"
            f"Copy config.yaml.example to config.yaml and fill in your values."
        )

    with config_path.open(encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    required_fields = (
        "icloud_root",
        "output_dir",
        "background_music",
        "caption_template_id",
        "mirage_api_key",
    )
    missing = [f for f in required_fields if not raw.get(f)]
    if missing:
        raise ConfigError(f"Missing required config fields: {', '.join(missing)}")

    return Config(
        icloud_root=_expand_path(str(raw["icloud_root"])),
        output_dir=_expand_path(str(raw["output_dir"])),
        background_music=_expand_path(str(raw["background_music"])),
        music_volume_db=float(raw.get("music_volume_db", -25)),
        voiceover_gain_db=float(raw.get("voiceover_gain_db", 0)),
        caption_template_id=str(raw["caption_template_id"]),
        mirage_api_key=str(raw["mirage_api_key"]),
        poll_interval_seconds=int(raw.get("poll_interval_seconds", 5)),
        max_poll_attempts=int(raw.get("max_poll_attempts", 20)),
        video_trim_extra_seconds=float(raw.get("video_trim_extra_seconds", 1)),
        max_file_size_mb=int(raw.get("max_file_size_mb", 50)),
        output_width=int(raw.get("output_width", 2160)),
        output_height=int(raw.get("output_height", 3840)),
        enable_hdr=bool(raw.get("enable_hdr", True)),
        video_crf=int(raw.get("video_crf", 22)),
        mirage_upload_crf=int(raw.get("mirage_upload_crf", 30)),
        mirage_base_url=str(raw.get("mirage_base_url", "https://api.mirage.app/v1")),
    )
