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


def _hlg_filter_chain(width: int, height: int) -> str:
    """SDR (BT.709) → HLG (BT.2020 / arib-std-b67) upscale to target 4K portrait.

    Matches Mirage Captions reference (HLG, not PQ). No tonemap=hable — that
    operator is HDR→SDR and would corrupt an SDR→HDR workflow.
    """
    scale = _scale_filter(width, height)
    return (
        f"{scale},"
        "zscale=rangein=full:range=limited:"
        "primariesin=bt709:primaries=bt2020:"
        "matrixin=bt709:matrix=bt2020nc:"
        "transferin=bt709:transfer=arib-std-b67,"
        "format=yuv420p10le"
    )


def _sdr_filter_chain(width: int, height: int) -> str:
    return f"{_scale_filter(width, height)},format=yuv420p"


def _video_encode_args(*, enable_hdr: bool, crf: int) -> list[str]:
    if enable_hdr:
        x265_params = (
            "repeat-headers=1:"
            "colorprim=bt2020:"
            "transfer=arib-std-b67:"
            "colormatrix=bt2020nc:"
            "range=limited"
        )
        return [
            "-c:v",
            "libx265",
            "-pix_fmt",
            "yuv420p10le",
            "-crf",
            str(crf),
            "-preset",
            "medium",
            "-tag:v",
            "hvc1",
            "-color_range",
            "tv",
            "-color_primaries",
            "bt2020",
            "-color_trc",
            "arib-std-b67",
            "-colorspace",
            "bt2020nc",
            "-movflags",
            "+faststart",
            "-x265-params",
            x265_params,
        ]
    return [
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        "-preset",
        "medium",
        "-movflags",
        "+faststart",
    ]


def _run_ffmpeg(cmd: list[str]) -> None:
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
    enable_hdr: bool,
    video_crf: int,
) -> tuple[Path, list[str]]:
    """
    Trim video, upscale to 4K portrait, mix audio, optionally encode HLG HDR.

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

    if enable_hdr:
        warnings.append(
            "HLG export from SDR source (BT.2020 / arib-std-b67) at "
            f"{output_width}x{output_height}. True HDR requires HDR source footage."
        )

    video_filter = (
        _hlg_filter_chain(output_width, output_height)
        if enable_hdr
        else _sdr_filter_chain(output_width, output_height)
    )

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
        *_video_encode_args(enable_hdr=enable_hdr, crf=video_crf),
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ac",
        "2",
        "-t",
        str(output_duration),
        str(output_path),
    ]

    _run_ffmpeg(cmd)
    return output_path, warnings


def finalize_export(
    input_path: Path,
    output_path: Path,
    *,
    output_width: int,
    output_height: int,
    enable_hdr: bool,
    video_crf: int,
) -> list[str]:
    """
    Re-encode Mirage output to guaranteed 4K portrait (+ HLG if enabled).

    Used because Mirage may return a lower-resolution or SDR file.
    """
    ensure_ffmpeg_installed()
    warnings: list[str] = []

    width, height = get_video_resolution(input_path)
    if width == output_width and height == output_height and not enable_hdr:
        shutil.copy2(input_path, output_path)
        return warnings

    if width != output_width or height != output_height:
        warnings.append(
            f"Mirage returned {width}x{height}; re-encoding to "
            f"{output_width}x{output_height} for client delivery."
        )

    video_filter = (
        _hlg_filter_chain(output_width, output_height)
        if enable_hdr
        else _sdr_filter_chain(output_width, output_height)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        video_filter,
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        *_video_encode_args(enable_hdr=enable_hdr, crf=video_crf),
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ac",
        "2",
        str(output_path),
    ]
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
