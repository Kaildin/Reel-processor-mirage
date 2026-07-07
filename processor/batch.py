"""Orchestrate the full processing pipeline per folder."""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TransferSpeedColumn,
)
from rich.table import Table
from rich.text import Text

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
    get_duration_seconds,
    overlay_captions_on_hlg,
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
    DOWNLOAD = "DOWNLOAD"
    FINALIZE = "FINALIZE"


# ─────────────────────────────────────────────────────────────────────────────
# Step timing tracker
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StepTiming:
    """Records start/end wall-clock times for a single pipeline step."""
    name: PipelineStep
    start: float = field(default_factory=time.monotonic)
    end: float | None = None
    ok: bool | None = None

    def finish(self, *, ok: bool = True) -> None:
        self.end = time.monotonic()
        self.ok = ok

    @property
    def elapsed(self) -> float:
        if self.end is not None:
            return self.end - self.start
        return time.monotonic() - self.start

    def elapsed_str(self) -> str:
        s = self.elapsed
        if s < 60:
            return f"{s:.1f}s"
        m, sec = divmod(s, 60)
        return f"{int(m)}m{sec:.0f}s"


def _render_step_summary(timings: list[StepTiming]) -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="left")
    table.add_column(justify="right")

    for t in timings:
        if t.ok is True:
            icon = "[bold green]\u2713[/bold green]"
            name_style = "green"
        elif t.ok is False:
            icon = "[bold red]\u2717[/bold red]"
            name_style = "red"
        else:
            icon = "[bold yellow]\u27f3[/bold yellow]"
            name_style = "yellow"

        table.add_row(
            f"{icon} [{name_style}]{t.name.value}[/{name_style}]",
            f"[dim]{t.elapsed_str()}[/dim]",
        )

    return Panel(table, title="[bold]Pipeline[/bold]", border_style="dim", expand=False)


# ─────────────────────────────────────────────────────────────────────────────
# Log entry / state
# ─────────────────────────────────────────────────────────────────────────────

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


def output_filename(video_basename: str) -> str:
    return f"Captions {video_basename}.mp4"


def output_path_for(scan_result: FolderScanResult) -> Path:
    basename = scan_result.basename or "unknown"
    folder = scan_result.video_path.parent  # type: ignore[union-attr]
    return folder / output_filename(basename)


def load_previous_run_log(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    with log_path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# Progress bar factories
# ─────────────────────────────────────────────────────────────────────────────

def _make_step_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=28),
        TextColumn("{task.percentage:>5.1f}%"),
        TimeElapsedColumn(),
        transient=False,
    )


def _make_transfer_progress() -> Progress:
    return Progress(
        SpinnerColumn("bouncingBar"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=24),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeElapsedColumn(),
        transient=False,
    )


def _make_batch_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold white]{task.description}"),
        BarColumn(bar_width=36),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Upload helper
# ─────────────────────────────────────────────────────────────────────────────

def _upload_with_progress(
    client: MirageClient,
    intermediate: Path,
    console: Console,
    basename: str,
) -> str:
    file_size = intermediate.stat().st_size
    up_progress = _make_transfer_progress()
    task = up_progress.add_task(f"UPLOAD  {basename}", total=file_size)

    def _on_progress(sent: int, _total: int) -> None:
        up_progress.update(task, completed=sent)

    with Live(up_progress, console=console, refresh_per_second=15):
        video_id = client.upload_for_captions(intermediate, progress_callback=_on_progress)
        up_progress.update(task, completed=file_size, description=f"UPLOAD  {basename}  \u2713")

    return video_id


# ─────────────────────────────────────────────────────────────────────────────
# Download helper
# ─────────────────────────────────────────────────────────────────────────────

def _download_with_progress(
    client: MirageClient,
    video_id: str,
    output_path: Path,
    console: Console,
    basename: str,
) -> None:
    dl_progress = _make_transfer_progress()
    task = dl_progress.add_task(f"DOWNLOAD  {basename}", total=None)

    def _on_progress(received: int, total: int) -> None:
        if total > 0 and dl_progress.tasks[0].total != total:
            dl_progress.update(task, total=total)
        dl_progress.update(task, completed=received)

    with Live(dl_progress, console=console, refresh_per_second=15):
        client.download_video(video_id, output_path, progress_callback=_on_progress)
        dl_progress.update(task, description=f"DOWNLOAD  {basename}  \u2713")


# ─────────────────────────────────────────────────────────────────────────────
# Mirage polling helper
# ─────────────────────────────────────────────────────────────────────────────

_MIRAGE_STATUS_LABEL: dict[str, str] = {
    "QUEUED":     "queued \u2014 waiting for a worker",
    "PROCESSING": "processing captions\u2026",
    "COMPLETE":   "complete \u2713",
    "FAILED":     "failed \u2717",
    "CANCELLED":  "cancelled \u2717",
}


def _poll_with_progress(
    client: MirageClient,
    video_id: str,
    console: Console,
    basename: str,
) -> dict[str, Any]:
    interval = client._config.poll_interval_seconds
    max_attempts = client._config.max_poll_attempts

    poll_progress = Progress(
        SpinnerColumn("dots2"),
        TextColumn("[bold yellow]{task.description}"),
        TimeElapsedColumn(),
        transient=False,
    )
    task = poll_progress.add_task(
        f"PROCESSING  {basename}  \u2014  attempt 1/{max_attempts}",
        total=None,
    )

    with Live(poll_progress, console=console, refresh_per_second=10):
        for attempt in range(1, max_attempts + 1):
            data = client.get_video_status(video_id)
            raw_status = data.get("status", "UNKNOWN")
            label = _MIRAGE_STATUS_LABEL.get(raw_status, raw_status)

            poll_progress.update(
                task,
                description=(
                    f"PROCESSING  {basename}  \u2014  "
                    f"{label}  "
                    f"[dim](attempt {attempt}/{max_attempts}, "
                    f"poll every {interval}s)[/dim]"
                ),
            )

            if raw_status == "COMPLETE":
                poll_progress.update(
                    task,
                    description=f"PROCESSING  {basename}  \u2014  [bold green]complete \u2713[/bold green]",
                )
                return data

            if raw_status in {"FAILED", "CANCELLED"}:
                error = data.get("error") or {}
                raise MirageJobFailedError(
                    message=f"Mirage job {raw_status}: {error.get('message', 'No details')}",
                    status=raw_status,
                    error_code=error.get("code"),
                    error_message=error.get("message"),
                )

            if attempt < max_attempts:
                for _ in range(int(interval * 5)):
                    time.sleep(0.2)

    raise MiragePollTimeoutError(
        f"Polling timed out after {max_attempts} attempts "
        f"({max_attempts * interval}s) for video {video_id}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-folder processor
# ─────────────────────────────────────────────────────────────────────────────

def process_folder(
    scan_result: FolderScanResult,
    config: Config,
    *,
    dry_run: bool = False,
    force: bool = False,
    stop_after_download: bool = False,
    console: Console | None = None,
    current: int = 1,
    total: int = 1,
) -> RunLogEntry:
    console = console or Console()
    folder_number = scan_result.folder_number
    basename = scan_result.basename or "unknown"
    final_output = output_path_for(scan_result)
    source_hlg = scan_result.video_path  # original .mov, never deleted

    header = f"[bold]Folder {folder_number}/{total}[/bold]  {basename}"
    console.rule(header)

    timings: list[StepTiming] = []

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
        console.print(f"  [dim]\u2298 Skipped ({scan_result.status.value}): {scan_result.message}[/dim]")
        return make_entry(
            RunStatus.SKIPPED,
            skip_reason=scan_result.status.value,
            error=scan_result.message,
        )

    if final_output.exists() and not force and not stop_after_download:
        console.print("  [dim]\u2298 Output already exists \u2014 skipping (use --force to overwrite)[/dim]")
        return make_entry(
            RunStatus.SKIPPED,
            skip_reason="output_exists",
            output=final_output,
        )

    if dry_run:
        console.print("  [dim]dry-run: TRIM \u2713 | UPLOAD \u2713 | PROCESSING \u2026 | DOWNLOAD \u2713 | FINALIZE \u2713[/dim]")
        return make_entry(RunStatus.SUCCESS, output=final_output)

    try:
        ensure_ffmpeg_installed()
        client = MirageClient(config)

        with tempfile.TemporaryDirectory(prefix="reel-processor-") as tmp_dir:
            intermediate = Path(tmp_dir) / f"{basename}_upload.mp4"
            captioned_raw = Path(tmp_dir) / f"{basename}_captioned_raw.mp4"

            # ── Step 1: TRIM ──────────────────────────────────────────────────
            t_trim = StepTiming(PipelineStep.TRIM)
            timings.append(t_trim)

            trim_progress = _make_step_progress()
            trim_task = trim_progress.add_task("TRIM  preparing\u2026", total=100)

            with Live(trim_progress, console=console, refresh_per_second=15):
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
                    video_crf=config.mirage_upload_crf,
                    progress=trim_progress,
                    task_id=trim_task,
                )
            t_trim.finish(ok=True)

            for warning in trim_warnings:
                console.print(f"  [yellow]\u26a0  {warning}[/yellow]")

            size_warning = check_file_size_warning(intermediate, config.max_file_size_mb)
            if size_warning:
                console.print(f"  [yellow]\u26a0  {size_warning}[/yellow]")

            # ── Step 2: UPLOAD ────────────────────────────────────────────────
            t_upload = StepTiming(PipelineStep.UPLOAD)
            timings.append(t_upload)
            video_id = _upload_with_progress(client, intermediate, console, basename)
            t_upload.finish(ok=True)

            # ── Step 3: PROCESSING (Mirage polling) ───────────────────────────
            t_proc = StepTiming(PipelineStep.PROCESSING)
            timings.append(t_proc)
            _poll_with_progress(client, video_id, console, basename)
            t_proc.finish(ok=True)

            # ── Step 4: DOWNLOAD ──────────────────────────────────────────────
            t_dl = StepTiming(PipelineStep.DOWNLOAD)
            timings.append(t_dl)
            _download_with_progress(client, video_id, captioned_raw, console, basename)
            t_dl.finish(ok=True)

            # ── Stop here if --stop-after-download was passed ─────────────────
            if stop_after_download:
                inspect_output = final_output.parent / f"{basename}_mirage_raw_sdr.mp4"
                shutil.copy2(captioned_raw, inspect_output)
                console.print()
                console.print(_render_step_summary(timings))
                console.print(
                    f"  [bold yellow]\u23f9 Stopped after DOWNLOAD.[/bold yellow]\n"
                    f"  SDR file saved \u2192 [cyan]{inspect_output}[/cyan]\n"
                    f"  Run ffprobe on it, then implement the HDR overlay strategy."
                )
                return make_entry(RunStatus.SUCCESS, output=inspect_output)

            # ── Step 5: FINALIZE (overlay captions on native HLG) ─────────────
            t_fin = StepTiming(PipelineStep.FINALIZE)
            timings.append(t_fin)

            fin_progress = _make_step_progress()
            fin_task = fin_progress.add_task("FINALIZE  preparing\u2026", total=100)

            with Live(fin_progress, console=console, refresh_per_second=15):
                finalize_warnings = overlay_captions_on_hlg(
                    captioned_raw,
                    source_hlg,  # type: ignore[arg-type]
                    final_output,
                    output_width=config.output_width,
                    output_height=config.output_height,
                    video_crf=config.video_crf,
                    progress=fin_progress,
                    task_id=fin_task,
                )
            t_fin.finish(ok=True)

            for warning in finalize_warnings:
                console.print(f"  [yellow]\u26a0  {warning}[/yellow]")

            console.print()
            console.print(_render_step_summary(timings))
            console.print(f"  [bold green]\u2713 Done \u2192[/bold green] {final_output}")
            return make_entry(RunStatus.SUCCESS, output=final_output)

    except MirageJobFailedError as exc:
        if timings:
            timings[-1].finish(ok=False)
        console.print()
        console.print(_render_step_summary(timings))
        console.print(f"  [bold red]\u2717 PROCESSING failed:[/bold red] {exc}")
        error = f"{exc.status}"
        if exc.error_code:
            error += f" [{exc.error_code}]"
        if exc.error_message:
            error += f": {exc.error_message}"
        return make_entry(RunStatus.FAILED, error=error)

    except (MiragePollTimeoutError, MirageConnectionError, MirageAPIError) as exc:
        if timings:
            timings[-1].finish(ok=False)
        console.print()
        console.print(_render_step_summary(timings))
        console.print(f"  [bold red]\u2717 Mirage error:[/bold red] {exc}")
        return make_entry(RunStatus.FAILED, error=str(exc))

    except FFmpegError as exc:
        if timings:
            timings[-1].finish(ok=False)
        console.print()
        console.print(_render_step_summary(timings))
        console.print(f"  [bold red]\u2717 FFmpeg error:[/bold red] {exc}")
        return make_entry(RunStatus.FAILED, error=str(exc))

    except ReelProcessorError as exc:
        if timings:
            timings[-1].finish(ok=False)
        console.print()
        console.print(_render_step_summary(timings))
        console.print(f"  [bold red]\u2717 Error:[/bold red] {exc}")
        return make_entry(RunStatus.FAILED, error=str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Batch runner
# ─────────────────────────────────────────────────────────────────────────────

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


def run_batch(
    config: Config,
    *,
    dry_run: bool = False,
    only_failed: bool = False,
    folder: int | None = None,
    force: bool = False,
    stop_after_download: bool = False,
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
            out = output_path_for(result)
            if _should_process_only_failed(result.folder_number, out, previous_log):
                filtered.append(result)
        processable = filtered
        console.print(f"[bold]Re-running {len(processable)} failed/pending folders[/bold]")

    total = len(processable)

    batch_progress = _make_batch_progress()
    batch_task = batch_progress.add_task(f"[white]Batch ({total} folders)", total=total)

    console.print()
    with Live(batch_progress, console=console, refresh_per_second=4):
        pass

    entries: list[RunLogEntry] = []

    for idx, result in enumerate(processable, start=1):
        entry = process_folder(
            result,
            config,
            dry_run=dry_run,
            force=force,
            stop_after_download=stop_after_download,
            console=console,
            current=idx,
            total=total,
        )
        entries.append(entry)
        batch_progress.update(batch_task, advance=1)

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
    failed  = sum(1 for e in entries if e.status == RunStatus.FAILED)
    skipped = sum(1 for e in entries if e.status == RunStatus.SKIPPED)

    console.print()
    console.rule("[bold]Batch complete[/bold]")
    console.print(
        f"  [bold green]{success} success[/bold green]  "
        f"[bold red]{failed} failed[/bold red]  "
        f"[bold yellow]{skipped} skipped[/bold yellow]"
    )
    console.print(f"  Run log \u2192 [cyan]{log_path}[/cyan]")

    return entries
