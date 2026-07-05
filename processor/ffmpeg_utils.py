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


def probe_colorspace(video_path: Path) -> dict[str, str]:
    """Return color_transfer, color_primaries, color_space from ffprobe.

    Used to inspect what Mirage actually returns so we can decide
    whether SDR conversion before upload is necessary.
    """
    ensure_ffmpeg_installed()
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=color_transfer,color_primaries,color_space,width,height",
        "-of", "default=noprint_wrappers=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise FFmpegError(
            f"ffprobe failed for {video_path}: {exc.stderr.strip()}"
        ) from exc

    info: dict[str, str] = {}
    for line in result.stdout.strip().splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            info[k.strip()] = v.strip()
    return info


def _scale_filter(width: int, height: int) -> str:
    return (
        f"scale={width}:{height}:flags=lanczos:"
        f"force_original_aspect_ratio=increase,"
        f"crop={width}:{height}"
    )


def _hlg_passthrough_filter_chain(width: int, height: int) -> str:
    """Scale only - no colorspace conversion.

    TEST BRANCH: sends HLG source directly to Mirage without converting to SDR.
    Output format is yuv420p (H.264 compatible) but colour metadata is NOT touched,
    so if Mirage can handle HLG it will see the correct values.
    """
    scale = _scale_filter(width, height)
    return f"{scale},format=yuv420p"


def _upscale_passthrough_filter_chain(width: int, height: int) -> str:
    """Upscale + tag as HLG, but do NOT apply any zscale conversion.

    TEST BRANCH: if Mirage preserved HLG, the pixel values are already correct
    and we only need to scale + inject the BT.2020 metadata tags.
    If the output looks wrong, it means Mirage stripped/changed the colour space
    and we need the SDR->HLG zscale from main branch.
    """
    scale = _scale_filter(width, height)
    return f"{scale},format=yuv420p10le"


def _hlg_encode_args(crf: int) -> list[str]:
    """libx265 encode args for HLG 4K delivery.

    Key fixes for iOS HDR badge recognition:

    1. -movflags +write_colr: writes the ISO 'colr' box in the container.
       iOS/QuickTime reads this box first when deciding whether to show
       the HDR badge — before inspecting x265 SEI NAL units.

    2. -profile:v main10: explicitly forces HEVC Main10 profile.
       Without this, libx265 may silently encode as 'Main' even when
       yuv420p10le is requested. iOS uses the HEVC profile level to
       gate HDR recognition.

    3. master-display and max-cll are intentionally OMITTED.
       Those are SMPTE ST 2086 / CEA-861.3 metadata for HDR10
       (a display-referred format). HLG (ARIB STD-B67) is scene-referred
       and does not carry absolute luminance metadata. Including HDR10
       SEI in an HLG stream can cause iOS to misidentify the file as
       malformed HDR10 and refuse to display the badge.

    The combination of colr box + Main10 profile + arib-std-b67 transfer
    matches what Apple AVFoundation writes when exporting HLG from Photos
    or Final Cut Pro X.
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


def _sdr_encode_args(crf: int) -> list[str]:
    """libx264 encode args for SDR delivery."""
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
    Prepare the Mirage upload intermediate: trim, scale, mix audio.

    TEST BRANCH (hlg-passthrough): HLG source is NOT converted to SDR.
    Sends HLG pixel values directly to Mirage to test whether Mirage
    handles HLG natively. probe_colorspace() is called on the Mirage
    output in remux_and_upscale to inspect what came back.

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
        "[hlg-passthrough] Sending HLG source directly to Mirage - NO SDR conversion."
    )

    video_filter = _hlg_passthrough_filter_chain(output_width, output_height)

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
) -> list[str]:
    """
    Post-Mirage finalization step.

    TEST BRANCH (hlg-passthrough):
    Runs probe_colorspace() first and logs what Mirage returned.
    No zscale conversion - just scale + HLG metadata tags.
    If the output looks correct: Mirage preserved HLG and this branch
    is the right approach. If colours are wrong: use main branch.

    Audio is always re-encoded to stereo AAC 192 kbps.
    """
    ensure_ffmpeg_installed()
    warnings: list[str] = []

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Log what Mirage actually returned so we can inspect it
    try:
        cs = probe_colorspace(input_path)
        warnings.append(
            f"[hlg-passthrough] Mirage output colorspace: "
            f"transfer={cs.get('color_transfer', 'N/A')} "
            f"primaries={cs.get('color_primaries', 'N/A')} "
            f"matrix={cs.get('color_space', 'N/A')} "
            f"{cs.get('width', '?')}x{cs.get('height', '?')}"
        )
    except FFmpegError as exc:
        warnings.append(f"[hlg-passthrough] Could not probe Mirage output: {exc}")

    if upscale:
        src_w, src_h = get_video_resolution(input_path)
        if src_w != output_width or src_h != output_height:
            warnings.append(
                f"Upscaling Mirage output {src_w}x{src_h} -> "
                f"{output_width}x{output_height} (scale + HLG tags, no zscale)."
            )
        video_filter = _upscale_passthrough_filter_chain(output_width, output_height)
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
