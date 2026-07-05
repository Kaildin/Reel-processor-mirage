"""FFmpeg/ffprobe utilities for trim and audio mixing."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from processor.exceptions import FFmpegError, FFmpegNotFoundError

# Regex to parse FFmpeg progress lines from stderr:
# frame=  120 fps= 30 q=28.0 size=    1024kB time=00:00:04.00 bitrate= ...
_FFMPEG_PROGRESS_RE = re.compile(
    r"frame=\s*(?P<frame>\d+).*?"
    r"fps=\s*(?P<fps>[\d.]+).*?"
    r"time=(?P<time>[\d:.]+).*?"
    r"speed=\s*(?P<speed>[\d.]+)x",
    re.DOTALL,
)


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


def get_video_resolution(video_path: Path) -> tuple[int, int]:
    ensure_ffmpeg_installed()
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise FFmpegError(
            f"ffprobe failed for {video_path}: {exc.stderr.strip()}"
        ) from exc

    try:
        width_str, height_str = result.stdout.strip().split("x")
        return int(width_str), int(height_str)
    except ValueError as exc:
        raise FFmpegError(
            f"Could not parse resolution from ffprobe output: {result.stdout!r}"
        ) from exc


def get_file_size_mb(file_path: Path) -> float:
    return file_path.stat().st_size / (1024 * 1024)


def _scale_filter(width: int, height: int) -> str:
    return (
        f"scale={width}:{height}:flags=lanczos:"
        f"force_original_aspect_ratio=increase,"
        f"crop={width}:{height}"
    )


def _hlg_to_sdr_filter_chain(width: int, height: int) -> str:
    """Scale + convert HLG (BT.2020) -> SDR (BT.709) for the Mirage upload intermediate.

    The raw .mov source files are natively BT.2020 HLG (confirmed via macOS Get Info).
    Mirage expects a standard SDR input to apply captions correctly, so we convert
    down to BT.709 here. The HLG grade is restored in remux_and_upscale after Mirage
    returns the captioned file.
    """
    scale = _scale_filter(width, height)
    return (
        f"{scale},"
        "zscale=rangein=limited:range=limited:"
        "primariesin=bt2020:primaries=bt709:"
        "matrixin=bt2020nc:matrix=bt709:"
        "transferin=arib-std-b67:transfer=bt709,"
        "format=yuv420p"
    )


def _upscale_hlg_filter_chain(width: int, height: int) -> str:
    """Upscale Mirage output to target resolution and restore HLG colour space.

    The Mirage output is SDR BT.709 limited range (tv). We convert
    it back to BT.2020 HLG here in a single zscale pass, then upscale to 4K.
    Both input and output are limited range (tv) — matching what Mirage returns
    and what iOS expects for HLG delivery.
    """
    scale = _scale_filter(width, height)
    return (
        f"{scale},"
        "zscale=rangein=limited:range=limited:"
        "primariesin=bt709:primaries=bt2020:"
        "matrixin=bt709:matrix=bt2020nc:"
        "transferin=bt709:transfer=arib-std-b67,"
        "format=yuv420p10le"
    )


# Standard BT.2020 HLG display primaries (values x50000 per SMPTE ST 2086).
# These SEI NAL units are required for iOS to display the HDR badge.
_HLG_MASTER_DISPLAY = (
    "G(13250,34500)B(7500,3000)R(34000,16000)"
    "WP(15635,16450)L(10000000,1)"
)
_HLG_MAX_CLL = "1000,400"


def _hlg_encode_args(crf: int) -> list[str]:
    """libx265 encode args for HLG 4K delivery.

    Includes master-display and max-cll SEI metadata so that iOS
    recognises the file as HDR and shows the HDR badge in Files / Photos.
    Values are the standard BT.2020 HLG primaries (D65 white point,
    peak luminance 1000 nits, max frame average 400 nits).
    """
    x265_params = (
        "repeat-headers=1:"
        "colorprim=bt2020:"
        "transfer=arib-std-b67:"
        "colormatrix=bt2020nc:"
        "range=limited:"
        f"master-display={_HLG_MASTER_DISPLAY}:"
        f"max-cll={_HLG_MAX_CLL}"
    )
    return [
        "-c:v", "libx265",
        "-pix_fmt", "yuv420p10le",
        "-crf", str(crf),
        "-preset", "medium",
        "-tag:v", "hvc1",
        "-color_range", "tv",
        "-color_primaries", "bt2020",
        "-color_trc", "arib-std-b67",
        "-colorspace", "bt2020nc",
        "-movflags", "+faststart",
        "-x265-params", x265_params,
    ]


def _sdr_encode_args(crf: int) -> list[str]:
    """libx264 encode args for SDR delivery (Mirage upload intermediate)."""
    return [
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", str(crf),
        "-preset", "medium",
        "-movflags", "+faststart",
    ]


def _parse_time_to_seconds(time_str: str) -> float:
    """Convert HH:MM:SS.ss string from FFmpeg to total seconds."""
    try:
        parts = time_str.split(":")
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        return float(time_str)
    except (ValueError, AttributeError):
        return 0.0


def run_ffmpeg_with_progress(
    cmd: list[str],
    duration_seconds: float,
    progress: Progress,
    task_id: TaskID,
    label: str = "FFmpeg",
) -> None:
    """Run an FFmpeg command while streaming stderr to update a Rich progress bar."""
    augmented_cmd = [cmd[0], "-progress", "pipe:1", "-nostats"] + cmd[1:]

    try:
        proc = subprocess.Popen(
            augmented_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as exc:
        raise FFmpegError(f"ffmpeg executable not found: {exc}") from exc

    assert proc.stdout is not None
    assert proc.stderr is not None

    stderr_lines: list[str] = []
    out_time_s: float = 0.0
    speed: str = ""
    fps: str = ""

    for line in proc.stdout:
        line = line.strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()

        if key == "out_time":
            out_time_s = _parse_time_to_seconds(value)
        elif key == "speed":
            speed = value
        elif key == "fps":
            fps = value
        elif key == "progress":
            if duration_seconds > 0:
                pct = min(out_time_s / duration_seconds, 1.0)
                completed = int(pct * 100)
                desc_parts = [label]
                if fps and fps not in ("0", "0.00", "N/A"):
                    desc_parts.append(f"{fps}fps")
                if speed and speed not in ("0x", "N/A"):
                    desc_parts.append(f"{speed}")
                progress.update(
                    task_id,
                    completed=completed,
                    description=" ".join(desc_parts),
                )

    proc.wait()

    for line in proc.stderr:
        stderr_lines.append(line)

    if proc.returncode != 0:
        stderr_text = "".join(stderr_lines).strip()
        raise FFmpegError(f"ffmpeg failed (exit {proc.returncode}):\n{stderr_text}")

    progress.update(task_id, completed=100, description=f"{label} ✓")


def _run_ffmpeg(cmd: list[str]) -> None:
    """Simple blocking FFmpeg run (no progress display). Kept for internal use."""
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise FFmpegError(f"ffmpeg failed:\n{exc.stderr.strip()}") from exc


def trim_and_mix(
    video_path: Path,
    voiceover_path: Path,
    background_music_path: Path,
    output_path: Path,
    *,
    music_volume_db: float,
    voiceover_gain_db: float,
    trim_extra_seconds: float,
    output_width: int,
    output_height: int,
    video_crf: int,
    progress: Optional[Progress] = None,
    task_id: Optional[TaskID] = None,
) -> tuple[Path, list[str]]:
    """
    Prepare the Mirage upload intermediate: trim, scale, mix audio, convert HLG->SDR.
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

    warnings.append(
        "Converting HLG source -> SDR BT.709 for Mirage upload. "
        "HLG will be restored after captioning."
    )

    video_filter = _hlg_to_sdr_filter_chain(output_width, output_height)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    filter_complex = (
        f"[0:v]trim=0:{output_duration},setpts=PTS-STARTPTS,{video_filter}[v];"
        f"[1:a]atrim=0:{output_duration},asetpts=PTS-STARTPTS,"
        f"volume={voiceover_gain_db}dB[vo];"
        f"[2:a]aloop=loop=-1:size=2e+09,atrim=0:{output_duration},"
        f"asetpts=PTS-STARTPTS,volume={music_volume_db}dB[bg];"
        f"[vo][bg]amix=inputs=2:duration=longest:dropout_transition=0[a]"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-i", str(voiceover_path),
        "-stream_loop", "-1",
        "-i", str(background_music_path),
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "[a]",
        *_sdr_encode_args(video_crf),
        "-c:a", "aac",
        "-b:a", "192k",
        "-ac", "2",
        "-t", str(output_duration),
        str(output_path),
    ]

    if progress is not None and task_id is not None:
        run_ffmpeg_with_progress(cmd, output_duration, progress, task_id, label="TRIM")
    else:
        _run_ffmpeg(cmd)

    return output_path, warnings


def remux_and_upscale(
    input_path: Path,
    output_path: Path,
    *,
    output_width: int,
    output_height: int,
    video_crf: int,
    upscale: bool = True,
    progress: Optional[Progress] = None,
    task_id: Optional[TaskID] = None,
) -> list[str]:
    """
    Post-Mirage finalization step.

    upscale=True: upscale to output_width x output_height + SDR->HLG BT.2020.
    upscale=False: pure remux, stream-copy video, inject HLG container tags only.
    Audio is always re-encoded to stereo AAC 192 kbps.
    """
    ensure_ffmpeg_installed()
    warnings: list[str] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if upscale:
        src_w, src_h = get_video_resolution(input_path)
        if src_w != output_width or src_h != output_height:
            warnings.append(
                f"Upscaling Mirage output {src_w}x{src_h} -> "
                f"{output_width}x{output_height} + restoring HLG."
            )
        video_filter = _upscale_hlg_filter_chain(output_width, output_height)
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(input_path),
            "-vf", video_filter,
            "-map", "0:v:0",
            "-map", "0:a:0?",
            *_hlg_encode_args(video_crf),
            "-c:a", "aac",
            "-b:a", "192k",
            "-ac", "2",
            str(output_path),
        ]
    else:
        warnings.append(
            "Remux only (no upscale): stream-copying video, injecting HLG container tags."
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(input_path),
            "-map", "0:v:0",
            "-map", "0:a:0?",
            "-c:v", "copy",
            "-tag:v", "hvc1",
            "-color_range", "tv",
            "-color_primaries", "bt2020",
            "-color_trc", "arib-std-b67",
            "-colorspace", "bt2020nc",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ac", "2",
            "-movflags", "+faststart",
            str(output_path),
        ]

    duration = get_duration_seconds(input_path) if upscale else 0.0
    label = "FINALIZE" if upscale else "FINALIZE (remux)"

    if progress is not None and task_id is not None:
        run_ffmpeg_with_progress(cmd, duration, progress, task_id, label=label)
    else:
        _run_ffmpeg(cmd)

    return warnings


def check_file_size_warning(file_path: Path, max_size_mb: int) -> str | None:
    size_mb = get_file_size_mb(file_path)
    if size_mb > max_size_mb:
        return (
            f"Processed video is {size_mb:.1f} MB (Mirage limit: {max_size_mb} MB). "
            f"Try raising video_crf or mirage_upload_crf in config.yaml. "
            f"Upload will still be attempted."
        )
    return None
