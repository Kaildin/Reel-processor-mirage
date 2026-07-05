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


# ---------------------------------------------------------------------------
# NVENC availability probe
# ---------------------------------------------------------------------------

def check_nvenc_available() -> bool:
    """Return True if hevc_nvenc is usable on this machine.

    Runs a 1-frame null encode to verify both driver and NVENC engine
    are present. Cheap: completes in <1 second, no output file written.
    """
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=black:s=64x64:r=1",
                "-vframes", "1",
                "-c:v", "hevc_nvenc",
                "-f", "null", "-",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


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
    """Scale + convert HLG (BT.2020) -> SDR (BT.709) for the Mirage upload intermediate."""
    scale = _scale_filter(width, height)
    return (
        f"{scale},"
        "zscale=rangein=limited:range=full:"
        "primariesin=bt2020:primaries=bt709:"
        "matrixin=bt2020nc:matrix=bt709:"
        "transferin=arib-std-b67:transfer=bt709,"
        "format=yuv420p"
    )


def _upscale_hlg_filter_chain(width: int, height: int) -> str:
    """Upscale Mirage output + restore HLG colour space via single zscale pass."""
    scale = _scale_filter(width, height)
    return (
        f"{scale},"
        "zscale=rangein=full:range=limited:"
        "primariesin=bt709:primaries=bt2020:"
        "matrixin=bt709:matrix=bt2020nc:"
        "transferin=bt709:transfer=arib-std-b67,"
        "format=yuv420p10le"
    )


def _upscale_hlg_filter_chain_nvenc(width: int, height: int) -> str:
    """Like _upscale_hlg_filter_chain but outputs yuv420p (8-bit) for NVENC.

    MX130 / Pascal-generation NVENC does not support 10-bit output.
    Colour space conversion is still applied via zscale; only the final
    pixel format differs. iOS reads HDR metadata from the container
    (colr box) and SEI, not from bit-depth alone.
    """
    scale = _scale_filter(width, height)
    return (
        f"{scale},"
        "zscale=rangein=full:range=limited:"
        "primariesin=bt709:primaries=bt2020:"
        "matrixin=bt709:matrix=bt2020nc:"
        "transferin=bt709:transfer=arib-std-b67,"
        "format=yuv420p"
    )


# ---------------------------------------------------------------------------
# Encode arg builders
# ---------------------------------------------------------------------------

def _hlg_encode_args_cpu(crf: int) -> list[str]:
    """libx265 CPU encode args for HLG delivery.

    Three fixes for iOS HDR badge (same as test/hlg-passthrough branch):
    - write_colr: container colr box so iOS recognises HDR before reading SEI
    - profile:v main10: forces HEVC Main10 so iOS gates HDR correctly
    - master-display / max-cll omitted: HLG is scene-referred (not HDR10)
    """
    x265_params = (
        "repeat-headers=1:"
        "colorprim=bt2020:"
        "transfer=arib-std-b67:"
        "colormatrix=bt2020nc:"
        "range=limited"
    )
    return [
        "-c:v", "libx265",
        "-profile:v", "main10",
        "-pix_fmt", "yuv420p10le",
        "-crf", str(crf),
        "-preset", "medium",
        "-tag:v", "hvc1",
        "-color_range", "tv",
        "-color_primaries", "bt2020",
        "-color_trc", "arib-std-b67",
        "-colorspace", "bt2020nc",
        "-movflags", "+faststart+write_colr",
        "-x265-params", x265_params,
    ]


def _hlg_encode_args_nvenc(cq: int) -> list[str]:
    """hevc_nvenc encode args for HLG delivery on NVIDIA GPUs.

    Tuned for MX130 / Pascal-tier NVENC constraints:
    - yuv420p (8-bit): MX130 NVENC engine does not support 10-bit output.
    - -bf 0: no B-frames (not supported on MX-class NVENC).
    - -rc vbr -cq <cq>: VBR mode with target quality; equivalent to
      libx265 CRF. cq=23 ~ crf=22 visually.
    - -preset p4: balanced speed/quality for NVENC (p1=fastest, p7=best).
      p4 is recommended for MX130 to avoid overwhelming the limited
      NVENC engine with look-ahead.
    - HDR metadata injected via container flags and SEI (same approach
      as CPU path); colr box written via -movflags +write_colr.
    - master-display / max-cll intentionally omitted (HLG, not HDR10).
    """
    return [
        "-c:v", "hevc_nvenc",
        "-pix_fmt", "yuv420p",
        "-rc", "vbr",
        "-cq", str(cq),
        "-preset", "p4",
        "-bf", "0",
        "-tag:v", "hvc1",
        "-color_range", "tv",
        "-color_primaries", "bt2020",
        "-color_trc", "arib-std-b67",
        "-colorspace", "bt2020nc",
        "-movflags", "+faststart+write_colr",
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


def _run_ffmpeg(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise FFmpegError(f"ffmpeg failed:\n{exc.stderr.strip()}") from exc


# ---------------------------------------------------------------------------
# Public pipeline functions
# ---------------------------------------------------------------------------

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
) -> tuple[Path, list[str]]:
    """
    Prepare the Mirage upload intermediate: trim, scale, mix audio, convert HLG->SDR.

    Source .mov files are natively BT.2020 HLG. This step converts them to
    SDR BT.709 (yuv420p) so Mirage receives a clean, standard input.
    HLG is restored in remux_and_upscale after Mirage returns the captioned video.

    The intermediate is always encoded with libx264 (CPU) — NVENC is only
    used in the final delivery encode in remux_and_upscale.

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
    use_nvenc: bool | None = None,
) -> list[str]:
    """
    Post-Mirage finalization step.

    The Mirage output is SDR BT.709. This step upscales to the target
    resolution and restores HLG colour space, then encodes the delivery file.

    GPU path (use_nvenc=True or auto-detected):
        Uses hevc_nvenc for the final encode. The upscale filter (zscale)
        still runs on the CPU — only the encode itself uses the GPU.
        For a typical 60s reel this saves ~3-4 minutes vs libx265 medium.
        Output is yuv420p 8-bit (MX130 NVENC limitation). HDR badge metadata
        is injected via the container colr box and SEI.

    CPU path (use_nvenc=False or NVENC unavailable):
        Uses libx265 with Main10 profile and yuv420p10le. Includes write_colr
        and omits master-display/max-cll (HLG, not HDR10).

    use_nvenc:
        None (default) = auto-detect via check_nvenc_available()
        True           = force GPU (raises FFmpegError if unavailable)
        False          = force CPU (libx265)

    Audio is always re-encoded to stereo AAC 192 kbps.
    """
    ensure_ffmpeg_installed()
    warnings: list[str] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve GPU availability
    if use_nvenc is None:
        use_nvenc = check_nvenc_available()
        if use_nvenc:
            warnings.append("[nvenc] NVIDIA GPU detected — using hevc_nvenc for final encode.")
        else:
            warnings.append("[nvenc] hevc_nvenc not available — falling back to libx265 (CPU).")
    elif use_nvenc:
        warnings.append("[nvenc] GPU encode forced by caller (use_nvenc=True).")
    else:
        warnings.append("[nvenc] CPU encode forced by caller (use_nvenc=False).")

    if upscale:
        src_w, src_h = get_video_resolution(input_path)
        if src_w != output_width or src_h != output_height:
            warnings.append(
                f"Upscaling Mirage output {src_w}x{src_h} -> "
                f"{output_width}x{output_height} + restoring HLG."
            )

        if use_nvenc:
            video_filter = _upscale_hlg_filter_chain_nvenc(output_width, output_height)
            encode_args = _hlg_encode_args_nvenc(cq=video_crf)
        else:
            video_filter = _upscale_hlg_filter_chain(output_width, output_height)
            encode_args = _hlg_encode_args_cpu(video_crf)

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(input_path),
            "-vf", video_filter,
            "-map", "0:v:0",
            "-map", "0:a:0?",
            *encode_args,
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
            "-movflags", "+faststart+write_colr",
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
