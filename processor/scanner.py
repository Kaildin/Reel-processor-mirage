"""Scan numbered folders and match mov+m4a pairs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from rich.console import Console
from rich.table import Table

from processor.exceptions import ScanError

VIDEO_EXTENSION = ".mov"
AUDIO_EXTENSION = ".m4a"


class ScanStatus(str, Enum):
    OK = "ok"
    MISSING_PAIR = "missing_pair"
    AMBIGUOUS_FILES = "ambiguous_files"
    NOT_DOWNLOADED = "not_downloaded_from_icloud"


@dataclass(frozen=True)
class FolderScanResult:
    folder_number: int
    status: ScanStatus
    basename: str | None = None
    video_path: Path | None = None
    audio_path: Path | None = None
    message: str | None = None

    @property
    def is_processable(self) -> bool:
        return self.status == ScanStatus.OK


FOLDER_PATTERN = re.compile(r"^\d+$")
SKIP_DIRS = {"_output", ".tmp", "__pycache__"}


def is_icloud_placeholder(path: Path) -> bool:
    if path.suffix == ".icloud":
        return True
    if path.exists() and path.stat().st_size == 0:
        return True
    icloud_sibling = path.parent / f"{path.name}.icloud"
    return icloud_sibling.exists()


def _list_media_files(folder: Path, extension: str) -> list[Path]:
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() == extension
        and not p.name.startswith(".")
    )


def scan_folder(folder_number: int, folder_path: Path) -> FolderScanResult:
    video_files = _list_media_files(folder_path, VIDEO_EXTENSION)
    audio_files = _list_media_files(folder_path, AUDIO_EXTENSION)

    if not video_files or not audio_files:
        missing = []
        if not video_files:
            missing.append(VIDEO_EXTENSION)
        if not audio_files:
            missing.append(AUDIO_EXTENSION)
        return FolderScanResult(
            folder_number=folder_number,
            status=ScanStatus.MISSING_PAIR,
            message=f"Missing: {', '.join(missing)}",
        )

    if len(video_files) > 1 or len(audio_files) > 1:
        return FolderScanResult(
            folder_number=folder_number,
            status=ScanStatus.AMBIGUOUS_FILES,
            message=(
                f"Found {len(video_files)} {VIDEO_EXTENSION} and "
                f"{len(audio_files)} {AUDIO_EXTENSION} files "
                f"(expected exactly 1 each)"
            ),
        )

    video_path = video_files[0]
    audio_path = audio_files[0]

    for media_path in (video_path, audio_path):
        if is_icloud_placeholder(media_path):
            return FolderScanResult(
                folder_number=folder_number,
                status=ScanStatus.NOT_DOWNLOADED,
                basename=video_path.stem,
                video_path=video_path,
                audio_path=audio_path,
                message=f"Not downloaded from iCloud: {media_path.name}",
            )

    return FolderScanResult(
        folder_number=folder_number,
        status=ScanStatus.OK,
        basename=video_path.stem,
        video_path=video_path,
        audio_path=audio_path,
    )


def discover_folders(root: Path) -> list[tuple[int, Path]]:
    if not root.exists():
        raise ScanError(f"Project root does not exist: {root}")

    folders: list[tuple[int, Path]] = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if not entry.is_dir() or entry.name in SKIP_DIRS:
            continue
        if not FOLDER_PATTERN.match(entry.name):
            continue
        folders.append((int(entry.name), entry))
    return folders


def scan_all(root: Path) -> list[FolderScanResult]:
    return [scan_folder(num, path) for num, path in discover_folders(root)]


def print_scan_table(results: list[FolderScanResult], console: Console | None = None) -> None:
    console = console or Console()
    table = Table(title="Folder Scan Results")
    table.add_column("Folder", justify="right", style="cyan")
    table.add_column("Basename")
    table.add_column("Status")
    table.add_column("Details")

    status_styles = {
        ScanStatus.OK: "green",
        ScanStatus.MISSING_PAIR: "yellow",
        ScanStatus.AMBIGUOUS_FILES: "red",
        ScanStatus.NOT_DOWNLOADED: "yellow",
    }

    for result in results:
        style = status_styles.get(result.status, "white")
        table.add_row(
            str(result.folder_number),
            result.basename or "—",
            f"[{style}]{result.status.value}[/{style}]",
            result.message or "",
        )

    ok_count = sum(1 for r in results if r.is_processable)
    console.print(table)
    console.print(
        f"\n[bold]{ok_count}[/bold] processable / "
        f"[bold]{len(results)}[/bold] total folders"
    )
