"""Orchestrate the full processing pipeline per folder."""

from __future__ import annotations

import json
import tempfile
import threading
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
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

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
    remux_and_upscale,
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
    return f"Captions {video_basename}.mp4"


def output_path_for(scan_result: FolderScanResult) -> Path:
    """Final output lives inside the numbered folder, next to the source files."""
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


def _make_step_progress() -> Progress:
    """Progress bar used for individual FFmpeg steps (TRIM / FINALIZE)."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=28),
        TextColumn("{task.percentage:>5.1f}%"),
        TimeElapsedColumn(),
        transient=False,
    )


def _make_batch_progress() -> Progress:
    """Overall batch progress bar (1 per folder)."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold white]{task.description}"),
        BarColumn(bar_width=36),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )


def _poll_with_spinner(
    client: MirageClient,
    video_id: str,
    console: Console,
    basename: str,
) -> dict[str, Any]:
    """Poll Mirage with a live spinner showing elapsed time and attempt count."""
    interval = client._config.poll_interval_seconds
    max_attempts = client._config.max_poll_attempts

    spin_progress = Progress(
        SpinnerColumn("dots"),
        TextColumn("[yellow]{task.description}"),
        TimeElapsedColumn(),
        transient=True,
    )
    task = spin_progress.add_task(f"PROCESSING {basename}  attempt 1/{max_attempts}", total=None)

    with Live(spin_progress, console=console, refresh_per_second=10):
        for attempt in range(1, max_attempts + 1):
            spin_progress.update(
                task,
                description=(
                    f"PROCESSING {basename}  "
                    f"attempt {attempt}/{max_attempts}  "
                    f"(polling every {interval}s)"
                ),
            )
            data = client.get_video_status(video_id)
            status = data.get("status", "UNKNOWN")

            if status == "COMPLETE":
                spin_progress.update(task, description=f"PROCESSING {basename}  ✓ complete")
                return data

            if status in {"FAILED", "CANCELLED"}:
                error = data.get("error") or {}
                raise MirageJobFailedError(
                    message=f"Mirage job {status}: {error.get('message', 'No details')}",
                    status=status,
                    error_code=error.get("code"),
                    error_message=error.get("message"),
                )

            if attempt < max_attempts:
                time.sleep(interval)

    raise MiragePollTimeoutError(
        f"Polling timed out after {max_attempts} attempts "
        f"({max_attempts * interval}s) for video {video_id}"
    )


def _upload_with_progress(
    client: MirageClient,
    intermediate: Path,
    console: Console,
    basename: str,
) -> str:
    """Upload with a spinner + live byte counter."""
    file_size = intermediate.stat().st_size
    uploaded_bytes: list[int] = [0]
    video_id_holder: list[str] = [""]
    error_holder: list[Exception | None] = [None]

    up_progress = Progress(
        SpinnerColumn("bouncingBar"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=28),
        TextColumn("{task.percentage:>5.1f}%"),
        TimeElapsedColumn(),
        transient=True,
    )
    task = up_progress.add_task(f"UPLOAD {basename}", total=file_size)

    def _do_upload() -> None:
        try:
            video_id_holder[0] = client.upload_for_captions(intermediate)
        except Exception as exc:  # noqa: BLE001
            error_holder[0] = exc

    thread = threading.Thread(target=_do_upload, daemon=True)

    with Live(up_progress, console=console, refresh_per_second=10):
        thread.start()
        while thread.is_alive():
            # Approximate progress by checking how many bytes have been read
            # from the file (the upload stream hasn't completed yet).
            try:
                current = intermediate.stat().st_size  # file won't change; keep at max
            except OSError:
                current = 0
            # We can't hook into requests internals easily, so we animate
            # the spinner and advance the bar smoothly as time passes.
            elapsed = up_progress.tasks[0].elapsed or 0.0
            # Assume ~2 MB/s upload; cap at 95% until thread finishes.
            estimated = min(int(elapsed * 2 * 1024 * 1024), int(file_size * 0.95))
            up_progress.update(task, completed=estimated)
            time.sleep(0.2)

        up_progress.update(task, completed=file_size, description=f"UPLOAD {basename}  ✓")

    if error_holder[0] is not None:
        raise error_holder[0]  # type: ignore[misc]

    return video_id_holder[0]


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
    final_output = output_path_for(scan_result)
    state = PipelineState()

    header = f"[bold]Folder {folder_number}/{total}[/bold]  {basename}"
    console.rule(header)

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
        console.print(f"  [dim]⊘ Skipped ({scan_result.status.value}): {scan_result.message}[/dim]")
        return make_entry(
            RunStatus.SKIPPED,
            skip_reason=scan_result.status.value,
            error=scan_result.message,
        )

    if final_output.exists() and not force:
        console.print("  [dim]⊘ Output already exists — skipping (use --force to overwrite)[/dim]")
        return make_entry(
            RunStatus.SKIPPED,
            skip_reason="output_exists",
            output=final_output,
        )

    if dry_run:
        console.print("  [dim]dry-run: TRIM ✓ | UPLOAD ✓ | PROCESSING … | COMPLETE ✓[/dim]")
        return make_entry(RunStatus.SUCCESS, output=final_output)

    try:
        ensure_ffmpeg_installed()
        client = MirageClient(config)

        with tempfile.TemporaryDirectory(prefix="reel-processor-") as tmp_dir:
            intermediate = Path(tmp_dir) / f"{basename}_upload.mp4"
            captioned_raw = Path(tmp_dir) / f"{basename}_captioned_raw.mp4"

            # ── Step 1: TRIM (FFmpeg with live progress bar) ─────────────────
            trim_progress = _make_step_progress()
            trim_task = trim_progress.add_task("TRIM  preparing…", total=100)

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

            for warning in trim_warnings:
                console.print(f"  [yellow]⚠  {warning}[/yellow]")

            size_warning = check_file_size_warning(intermediate, config.max_file_size_mb)
            if size_warning:
                console.print(f"  [yellow]⚠  {size_warning}[/yellow]")

            # ── Step 2: UPLOAD (spinner + simulated byte progress) ───────────
            video_id = _upload_with_progress(client, intermediate, console, basename)

            # ── Step 3: PROCESSING (Mirage polling spinner) ──────────────────
            _poll_with_spinner(client, video_id, console, basename)

            # ── Step 4: DOWNLOAD ─────────────────────────────────────────────
            dl_progress = Progress(
                SpinnerColumn("bouncingBar"),
                TextColumn("[bold cyan]{task.description}"),
                TimeElapsedColumn(),
                transient=True,
            )
            dl_task = dl_progress.add_task(f"DOWNLOAD {basename}", total=None)
            with Live(dl_progress, console=console, refresh_per_second=10):
                client.download_video(video_id, captioned_raw)
                dl_progress.update(dl_task, description=f"DOWNLOAD {basename}  ✓")

            # ── Step 5: FINALIZE (FFmpeg re-encode with live progress bar) ───
            fin_progress = _make_step_progress()
            fin_task = fin_progress.add_task("FINALIZE  preparing…", total=100)

            with Live(fin_progress, console=console, refresh_per_second=15):
                finalize_warnings = remux_and_upscale(
                    captioned_raw,
                    final_output,
                    output_width=config.output_width,
                    output_height=config.output_height,
                    video_crf=config.video_crf,
                    upscale=config.upscale_output,
                    progress=fin_progress,
                    task_id=fin_task,
                )

            for warning in finalize_warnings:
                console.print(f"  [yellow]⚠  {warning}[/yellow]")

            console.print(f"  [bold green]✓ Done →[/bold green] {final_output}")
            return make_entry(RunStatus.SUCCESS, output=final_output)

    except MirageJobFailedError as exc:
        console.print(f"  [bold red]✗ PROCESSING failed:[/bold red] {exc}")
        error = f"{exc.status}"
        if exc.error_code:
            error += f" [{exc.error_code}]"
        if exc.error_message:
            error += f": {exc.error_message}"
        return make_entry(RunStatus.FAILED, error=error)

    except (MiragePollTimeoutError, MirageConnectionError, MirageAPIError) as exc:
        console.print(f"  [bold red]✗ Mirage error:[/bold red] {exc}")
        return make_entry(RunStatus.FAILED, error=str(exc))

    except FFmpegError as exc:
        console.print(f"  [bold red]✗ FFmpeg error:[/bold red] {exc}")
        return make_entry(RunStatus.FAILED, error=str(exc))

    except ReelProcessorError as exc:
        console.print(f"  [bold red]✗ Error:[/bold red] {exc}")
        return make_entry(RunStatus.FAILED, error=str(exc))


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
        console.print(
            f"[bold]Re-running {len(processable)} failed/pending folders[/bold]"
        )

    total = len(processable)

    # ── Overall batch progress bar ────────────────────────────────────────────
    batch_progress = _make_batch_progress()
    batch_task = batch_progress.add_task(
        f"[white]Batch ({total} folders)", total=total
    )

    console.print()
    with Live(batch_progress, console=console, refresh_per_second=4):
        pass  # print it once, then use console normally below

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
    failed = sum(1 for e in entries if e.status == RunStatus.FAILED)
    skipped = sum(1 for e in entries if e.status == RunStatus.SKIPPED)

    console.print()
    console.rule("[bold]Batch complete[/bold]")
    console.print(
        f"  [bold green]{success} success[/bold green]  "
        f"[bold red]{failed} failed[/bold red]  "
        f"[bold yellow]{skipped} skipped[/bold yellow]"
    )
    console.print(f"  Run log → [cyan]{log_path}[/cyan]")

    return entries
