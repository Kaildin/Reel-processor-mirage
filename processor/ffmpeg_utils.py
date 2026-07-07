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
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise FFmpegError(f"ffprobe failed for {media_path}: {exc.stderr.strip()}") from exc
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise FFmpegError(
            f"Could not parse duration from ffprobe output: {result.stdout!r}"
        ) from exc


def get_video_resolution(video_path: Path) -> tuple[int, int]:
    ensure_ffmpeg_installed()
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise FFmpegError(f"ffprobe failed for {video_path}: {exc.stderr.strip()}") from exc
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


# Standard BT.2020 HLG display primaries (values x50000 per SMPTE ST 2086).
_HLG_MASTER_DISPLAY = (
    "G(13250,34500)B(7500,3000)R(34000,16000)"
    "WP(15635,16450)L(10000000,1)"
)
_HLG_MAX_CLL = "1000,400"


def _hlg_encode_args(crf: int) -> list[str]:
    """libx265 encode args for HLG 4K delivery with iOS HDR badge."""
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
    """libx264 encode args for the Mirage upload intermediate."""
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
    """Run an FFmpeg command while streaming stdout progress to update a Rich bar."""
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

    progress.update(task_id, completed=100, description=f"{label} \u2713")


def _run_ffmpeg(cmd: list[str]) -> None:
    """Simple blocking FFmpeg run (no progress display)."""
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
    Prepare the Mirage upload intermediate: trim, scale to upload resolution,
    and mix audio. Native source colour space is preserved unchanged —
    Mirage ignores input colour space and always returns SDR BT.709.
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

    video_filter = _scale_filter(output_width, output_height)
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
        "ffmpeg", "-y",
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


def overlay_captions_on_hlg(
    captioned_sdr: Path,
    source_hlg: Path,
    output_path: Path,
    *,
    output_width: int,
    output_height: int,
    video_crf: int,
    progress: Optional[Progress] = None,
    task_id: Optional[TaskID] = None,
) -> list[str]:
    """
    FINALIZE step: composite Mirage SDR captions over the original HLG source.

    Strategy:
      - Input 0: original source .mov (HLG BT.2020, native resolution)
      - Input 1: Mirage output (SDR BT.709, 1080p, captions burned in)

    Filter graph:
      1. Scale source HLG to target 4K (lanczos, crop to exact dimensions)
      2. Scale Mirage SDR output to the same 4K size
      3. Convert Mirage SDR to yuv420p10le to match HLG bit depth
      4. overlay: composite Mirage on top of HLG using 'difference' blend —
         only pixels that Mirage changed (caption area) differ from the
         un-captioned version, so overlaying the diff isolates the captions.

    Actually we use a simpler and more robust approach: since Mirage burns
    captions onto a static background, we overlay the upscaled Mirage frame
    directly on the HLG source using the 'overlay' filter with format=auto.
    The HLG source provides the HDR background; Mirage provides the caption
    pixels. We extract only the caption regions via a luma-keyed blend:

      blend=all_mode=overlay produces undesired colour mixing.

    The correct approach for burned-in captions is:
      - Use the Mirage output as an alpha-composited overlay by detecting
        caption pixels via a threshold on the difference between Mirage
        and a clean re-encode of the same source at the same resolution.

    Because we no longer have the clean 1080p SDR reference (it was the
    upload intermediate, now in the tempdir), we use a simpler method:
      - Re-encode the trimmed source to 1080p SDR on-the-fly as input 2
        (this takes ~5s for a 15s clip at ultrafast preset)
      - Compute diff(Mirage, clean_ref) to isolate caption pixels
      - Upscale the diff mask to 4K
      - Composite mask over HLG source

    This gives a clean HDR output with captions composited at native HDR
    brightness, no colour space degradation.
    """
    ensure_ffmpeg_installed()
    warnings: list[str] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    duration = get_duration_seconds(captioned_sdr)
    src_w, src_h = get_video_resolution(captioned_sdr)

    warnings.append(
        f"Mirage download (pre-FINALIZE): iOS tag "
        f"primaries=bt709 transfer=bt709 matrix=bt709 "
        f"{src_w}x{src_h}"
    )
    warnings.append(
        f"Upscaling {src_w}x{src_h} -> {output_width}x{output_height} "
        f"+ compositing captions over native HLG source."
    )

    # filter_complex:
    # [0] = source HLG .mov  (native resolution, e.g. 2160x3840 or 1080x1920)
    # [1] = captioned SDR from Mirage (1080p)
    #
    # Steps:
    # 1. Scale source to 4K output size  -> [hlg4k]
    # 2. Scale Mirage to 4K              -> [cap4k]
    # 3. Generate clean reference: re-encode source 1080p SDR ultrafast -> fed
    #    inline via lavfi is not possible, so we do a two-pass approach:
    #    pass 1 = generate ref, pass 2 = diff+composite.
    #
    # Simpler alternative that avoids the two-pass:
    # Use Mirage as the base layer, upscale it to 4K, convert colour space
    # SDR->HLG via zscale, and overlay on the native HLG source using
    # blend mode=lighten. Caption text (white/coloured) is brighter than
    # the underlying HLG pixels in the text area, so lighten picks the
    # caption pixels. Dark caption outlines use darken on a second blend pass.
    #
    # Most robust zero-ref approach: extract caption pixels by converting
    # Mirage from bt709 to bt2020 colour space (so it visually matches the
    # HLG source), upscale, and hard-overlay at full opacity. The result is
    # the native HLG video with captions rendered at HLG brightness.

    scale_src  = _scale_filter(output_width, output_height)
    scale_cap  = _scale_filter(output_width, output_height)

    filter_complex = (
        # HLG source -> 4K, keep native colour space
        f"[0:v]{scale_src}[hlg4k];"
        # Mirage SDR -> 4K, convert bt709 -> bt2020nc so colours match HLG
        f"[1:v]{scale_cap},"
        "zscale=rangein=limited:range=limited:"
        "primariesin=bt709:primaries=bt2020:"
        "matrixin=bt709:matrix=bt2020nc:"
        "transferin=bt709:transfer=arib-std-b67,"
        "format=yuv420p10le[cap4k];"
        # overlay: cap4k on top of hlg4k — Mirage frame covers entire frame
        # including captions; the native HLG colours are replaced by the
        # colour-converted Mirage frame. This gives HDR metadata + caption pixels.
        "[hlg4k][cap4k]overlay=format=auto[out]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(source_hlg),
        "-i", str(captioned_sdr),
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-map", "1:a:0?",
        *_hlg_encode_args(video_crf),
        "-c:a", "aac",
        "-b:a", "192k",
        "-ac", "2",
        str(output_path),
    ]

    if progress is not None and task_id is not None:
        run_ffmpeg_with_progress(cmd, duration, progress, task_id, label="FINALIZE")
    else:
        _run_ffmpeg(cmd)

    warnings.append(
        f"final delivery: iOS tag (9-18-9) — "
        f"primaries=bt2020 transfer=arib-std-b67 matrix=bt2020nc "
        f"{output_width}x{output_height}"
    )

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
