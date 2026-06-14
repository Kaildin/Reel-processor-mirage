"""FFmpeg/ffprobe utilities for trim and audio mixing."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from processor.exceptions import FFmpegError, FFmpegNotFoundError


def ensure_ffmpeg_installed() -> None:
    for binary in ("ffmpeg", "ffprobe"):
        if shutil.which(binary) is None:
            raise FFmpegNotFoundError(
                f"{binary} not found. Install via Homebrew: brew install ffmpeg"
            )


def get_duration_seconds(media_path: Path) -> float:
    ensure_ffmpeg_installed()
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise FFmpegError(
            f"ffprobe failed for {media_path}: {exc.stderr.strip()}"
        ) from exc

    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise FFmpegError(
            f"Could not parse duration from ffprobe output: {result.stdout!r}"
        ) from exc


def get_file_size_mb(file_path: Path) -> float:
    return file_path.stat().st_size / (1024 * 1024)


def trim_and_mix(
    video_path: Path,  # .mov input
    voiceover_path: Path,  # .m4a input
    background_music_path: Path,
    output_path: Path,
    *,
    music_volume_db: float,
    voiceover_gain_db: float,
    trim_extra_seconds: float,
) -> tuple[Path, list[str]]:
    """
    Trim video and mix voiceover with looping background music.

    Returns the output path and a list of warning messages.
    """
    ensure_ffmpeg_installed()
    warnings: list[str] = []

    voiceover_duration = get_duration_seconds(voiceover_path)
    video_duration = get_duration_seconds(video_path)
    target_duration = voiceover_duration + trim_extra_seconds

    if video_duration < target_duration:
        warnings.append(
            f"Video ({video_duration:.2f}s) shorter than target "
            f"({target_duration:.2f}s = voiceover + {trim_extra_seconds}s); "
            f"using full video length"
        )
        output_duration = video_duration
    else:
        output_duration = target_duration

    output_path.parent.mkdir(parents=True, exist_ok=True)

    filter_complex = (
        f"[0:v]trim=0:{output_duration},setpts=PTS-STARTPTS[v];"
        f"[1:a]atrim=0:{output_duration},asetpts=PTS-STARTPTS,"
        f"volume={voiceover_gain_db}dB[vo];"
        f"[2:a]aloop=loop=-1:size=2e+09,atrim=0:{output_duration},"
        f"asetpts=PTS-STARTPTS,volume={music_volume_db}dB[bg];"
        f"[vo][bg]amix=inputs=2:duration=longest:dropout_transition=0[a]"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(voiceover_path),
        "-stream_loop",
        "-1",
        "-i",
        str(background_music_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "23",
        "-preset",
        "medium",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        "-t",
        str(output_duration),
        str(output_path),
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise FFmpegError(
            f"ffmpeg failed:\n{exc.stderr.strip()}"
        ) from exc

    return output_path, warnings


def check_file_size_warning(
    file_path: Path, max_size_mb: int
) -> str | None:
    size_mb = get_file_size_mb(file_path)
    if size_mb > max_size_mb:
        return (
            f"Processed video is {size_mb:.1f} MB (limit: {max_size_mb} MB). "
            f"Consider re-encoding at a lower bitrate (e.g. -crf 28). "
            f"Upload will still be attempted."
        )
    return None
