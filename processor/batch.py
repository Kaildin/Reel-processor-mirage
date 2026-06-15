"""Orchestrate the full processing pipeline per folder."""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from rich.console import Console

from processor.config import Config
from processor.exceptions import (
    FFmpegError,
    MirageAPIError,
    MirageConnectionError,
    MirageJobFailedError,
    MiragePollTimeoutError,
    ReelProcessorError,
)
from processor.ffmpeg_utils import (
    check_file_size_warning,
    ensure_ffmpeg_installed,
    finalize_export,
    trim_and_mix,
)
from processor.mirage_api import MirageClient
from processor.scanner import FolderScanResult, scan_all


class RunStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class PipelineStep(str, Enum):
    TRIM = "TRIM"
    UPLOAD = "UPLOAD"
    PROCESSING = "PROCESSING"
    COMPLETE = "COMPLETE"


@dataclass
class RunLogEntry:
    folder_number: int
    basename: str
    status: RunStatus
    timestamp: str
    error: str | None = None
    output_path: str | None = None
    skip_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineState:
    steps: dict[PipelineStep, str] = field(default_factory=dict)

    def set_step(self, step: PipelineStep, value: str) -> None:
        self.steps[step] = value

    def format_status_line(self) -> str:
        parts = []
        for step in PipelineStep:
            if step in self.steps:
                parts.append(f"{step.value} {self.steps[step]}")
        return " | ".join(parts)


def output_filename(video_basename: str) -> str:
    return f"captions_{video_basename}.mp4"


def output_path_for(config: Config, video_basename: str) -> Path:
    return config.output_dir / output_filename(video_basename)


def load_previous_run_log(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    with log_path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _print_progress(
    console: Console,
    current: int,
    total: int,
    basename: str,
    state: PipelineState,
) -> None:
    line = f"[{current}/{total}] {basename} → {state.format_status_line()}"
    console.print(line)


def _should_process_only_failed(
    folder_number: int,
    output_path: Path,
    previous_log: list[dict[str, Any]],
) -> bool:
    if output_path.exists():
        return False

    for entry in reversed(previous_log):
        if entry.get("folder_number") == folder_number:
            return entry.get("status") == "failed"
    return True


def process_folder(
    scan_result: FolderScanResult,
    config: Config,
    *,
    dry_run: bool = False,
    force: bool = False,
    console: Console | None = None,
    current: int = 1,
    total: int = 1,
) -> RunLogEntry:
    console = console or Console()
    folder_number = scan_result.folder_number
    basename = scan_result.basename or "unknown"
    final_output = output_path_for(config, basename)
    state = PipelineState()

    def make_entry(
        status: RunStatus,
        *,
        error: str | None = None,
        output: Path | None = None,
        skip_reason: str | None = None,
    ) -> RunLogEntry:
        return RunLogEntry(
            folder_number=folder_number,
            basename=basename,
            status=status,
            timestamp=_timestamp(),
            error=error,
            output_path=str(output) if output else None,
            skip_reason=skip_reason,
        )

    if not scan_result.is_processable:
        state.set_step(PipelineStep.TRIM, "⊘")
        _print_progress(console, current, total, basename, state)
        return make_entry(
            RunStatus.SKIPPED,
            skip_reason=scan_result.status.value,
            error=scan_result.message,
        )

    if final_output.exists() and not force:
        state.set_step(PipelineStep.COMPLETE, "⊘ skip")
        _print_progress(console, current, total, basename, state)
        return make_entry(
            RunStatus.SKIPPED,
            skip_reason="output_exists",
            output=final_output,
        )

    if dry_run:
        state.set_step(PipelineStep.TRIM, "✓ (dry)")
        state.set_step(PipelineStep.UPLOAD, "✓ (dry)")
        state.set_step(PipelineStep.PROCESSING, "… (dry)")
        state.set_step(PipelineStep.COMPLETE, "✓ (dry)")
        _print_progress(console, current, total, basename, state)
        return make_entry(RunStatus.SUCCESS, output=final_output)

    try:
        ensure_ffmpeg_installed()
        client = MirageClient(config)

        with tempfile.TemporaryDirectory(prefix="reel-processor-") as tmp_dir:
            intermediate = Path(tmp_dir) / f"{basename}_trimmed.mp4"
            captioned_raw = Path(tmp_dir) / f"{basename}_captioned_raw.mp4"

            state.set_step(PipelineStep.TRIM, "…")
            _print_progress(console, current, total, basename, state)

            _, trim_warnings = trim_and_mix(
                scan_result.video_path,  # type: ignore[arg-type]
                scan_result.audio_path,  # type: ignore[arg-type]
                config.background_music,
                intermediate,
                music_volume_db=config.music_volume_db,
                voiceover_gain_db=config.voiceover_gain_db,
                trim_extra_seconds=config.video_trim_extra_seconds,
                output_width=config.output_width,
                output_height=config.output_height,
                enable_hdr=config.enable_hdr,
                video_crf=config.mirage_upload_crf,
            )

            for warning in trim_warnings:
                console.print(f"  [yellow]⚠ {warning}[/yellow]")

            size_warning = check_file_size_warning(
                intermediate, config.max_file_size_mb
            )
            if size_warning:
                console.print(f"  [yellow]⚠ {size_warning}[/yellow]")

            state.set_step(PipelineStep.TRIM, "✓")
            state.set_step(PipelineStep.UPLOAD, "…")
            _print_progress(console, current, total, basename, state)

            video_id = client.upload_for_captions(intermediate)

            state.set_step(PipelineStep.UPLOAD, "✓")
            state.set_step(PipelineStep.PROCESSING, "…")
            _print_progress(console, current, total, basename, state)

            client.poll_until_complete(video_id)

            state.set_step(PipelineStep.PROCESSING, "✓")
            state.set_step(PipelineStep.COMPLETE, "…")
            _print_progress(console, current, total, basename, state)

            client.download_video(video_id, captioned_raw)

            finalize_warnings = finalize_export(
                captioned_raw,
                final_output,
                output_width=config.output_width,
                output_height=config.output_height,
                enable_hdr=config.enable_hdr,
                video_crf=config.video_crf,
            )
            for warning in finalize_warnings:
                console.print(f"  [yellow]⚠ {warning}[/yellow]")

            state.set_step(PipelineStep.COMPLETE, "✓")
            _print_progress(console, current, total, basename, state)

            return make_entry(RunStatus.SUCCESS, output=final_output)

    except MirageJobFailedError as exc:
        state.set_step(PipelineStep.PROCESSING, "✗")
        _print_progress(console, current, total, basename, state)
        error = f"{exc.status}"
        if exc.error_code:
            error += f" [{exc.error_code}]"
        if exc.error_message:
            error += f": {exc.error_message}"
        return make_entry(RunStatus.FAILED, error=error)

    except (MiragePollTimeoutError, MirageConnectionError, MirageAPIError) as exc:
        if PipelineStep.UPLOAD in state.steps and PipelineStep.PROCESSING not in state.steps:
            state.set_step(PipelineStep.UPLOAD, "✗")
        else:
            state.set_step(PipelineStep.PROCESSING, "✗")
        _print_progress(console, current, total, basename, state)
        return make_entry(RunStatus.FAILED, error=str(exc))

    except FFmpegError as exc:
        state.set_step(PipelineStep.TRIM, "✗")
        _print_progress(console, current, total, basename, state)
        return make_entry(RunStatus.FAILED, error=str(exc))

    except ReelProcessorError as exc:
        _print_progress(console, current, total, basename, state)
        return make_entry(RunStatus.FAILED, error=str(exc))


def run_batch(
    config: Config,
    *,
    dry_run: bool = False,
    only_failed: bool = False,
    folder: int | None = None,
    force: bool = False,
    console: Console | None = None,
) -> list[RunLogEntry]:
    console = console or Console()
    log_path = config.output_dir / "run_log.json"
    previous_log = load_previous_run_log(log_path)

    all_results = scan_all(config.icloud_root)
    processable = [r for r in all_results if r.is_processable]

    if folder is not None:
        processable = [r for r in processable if r.folder_number == folder]
        if not processable:
            console.print(f"[red]No processable folder found for number {folder}[/red]")

    if only_failed:
        filtered: list[FolderScanResult] = []
        for result in processable:
            out = output_path_for(config, result.basename or "unknown")
            if _should_process_only_failed(result.folder_number, out, previous_log):
                filtered.append(result)
        processable = filtered
        console.print(
            f"[bold]Re-running {len(processable)} failed/pending folders[/bold]"
        )

    total = len(processable)
    entries: list[RunLogEntry] = []

    for idx, result in enumerate(processable, start=1):
        entry = process_folder(
            result,
            config,
            dry_run=dry_run,
            force=force,
            console=console,
            current=idx,
            total=total,
        )
        entries.append(entry)

    skipped_scan = [
        RunLogEntry(
            folder_number=r.folder_number,
            basename=r.basename or "unknown",
            status=RunStatus.SKIPPED,
            timestamp=_timestamp(),
            skip_reason=r.status.value,
            error=r.message,
        )
        for r in all_results
        if not r.is_processable and (folder is None or r.folder_number == folder)
    ]

    merged_log = previous_log + [e.to_dict() for e in entries + skipped_scan]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fh:
        json.dump(merged_log, fh, indent=2, ensure_ascii=False)

    success = sum(1 for e in entries if e.status == RunStatus.SUCCESS)
    failed = sum(1 for e in entries if e.status == RunStatus.FAILED)
    skipped = sum(1 for e in entries if e.status == RunStatus.SKIPPED)

    console.print()
    console.print(
        f"[bold green]{success} success[/bold green] | "
        f"[bold red]{failed} failed[/bold red] | "
        f"[bold yellow]{skipped} skipped[/bold yellow]"
    )
    console.print(f"Run log saved to [cyan]{log_path}[/cyan]")

    return entries
