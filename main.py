#!/usr/bin/env python3
"""CLI entrypoint for reel-processor."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from processor.batch import run_batch
from processor.config import Config, load_config
from processor.exceptions import ConfigError, MirageAPIError, ReelProcessorError
from processor.ffmpeg_utils import ensure_ffmpeg_installed
from processor.mirage_api import MirageClient
from processor.scanner import print_scan_table, scan_all

app = typer.Typer(
    name="reel-processor",
    help="Batch process physiotherapy exercise reels with FFmpeg and Mirage captions.",
    no_args_is_help=True,
)
console = Console()

DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


def _load(config: Path) -> Config:
    try:
        return load_config(config)
    except ConfigError as exc:
        console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command("scan")
def scan_cmd(
    config_path: Path = typer.Option(
        DEFAULT_CONFIG, "--config", "-c", help="Path to config.yaml"
    ),
) -> None:
    """Scan all folders and print a table of matches/errors."""
    cfg = _load(config_path)
    results = scan_all(cfg.icloud_root)
    print_scan_table(results, console)


@app.command("run")
def run_cmd(
    dry_run: bool = typer.Option(False, "--dry-run", help="Simulate without FFmpeg or API"),
    only_failed: bool = typer.Option(
        False, "--only-failed", help="Re-run only failed or pending folders"
    ),
    folder: Optional[int] = typer.Option(
        None, "--folder", help="Process only this folder number"
    ),
    force: bool = typer.Option(
        False, "--force", help="Overwrite existing output files"
    ),
    # ↓ nuovo flag
    stop_after_download: bool = typer.Option(
        False, "--stop-after-download",
        help="Download Mirage output (SDR) and save it, skip FINALIZE. For ffprobe inspection."
    ),
    config_path: Path = typer.Option(
        DEFAULT_CONFIG, "--config", "-c", help="Path to config.yaml"
    ),
) -> None:
    """Process all matched pairs end-to-end."""
    cfg = _load(config_path)

    try:
        if not dry_run:
            ensure_ffmpeg_installed()
    except ReelProcessorError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    try:
        run_batch(
            cfg,
            dry_run=dry_run,
            only_failed=only_failed,
            folder=folder,
            force=force,
            stop_after_download=stop_after_download,  # ← passa a batch
            console=console,
        )
    except ReelProcessorError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command("list-templates")
def list_templates_cmd(
    config_path: Path = typer.Option(
        DEFAULT_CONFIG, "--config", "-c", help="Path to config.yaml"
    ),
) -> None:
    """List all available Mirage caption templates."""
    cfg = _load(config_path)

    try:
        client = MirageClient(cfg)
        templates = client.list_caption_templates()
    except MirageAPIError as exc:
        console.print(f"[red]Mirage API error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if not templates:
        console.print("[yellow]No caption templates found.[/yellow]")
        return

    table = Table(title="Mirage Caption Templates")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Description")

    for template in templates:
        table.add_row(
            template.get("id", "—"),
            template.get("name", "—"),
            template.get("description", "") or "",
        )

    console.print(table)
    console.print(
        "\nCopy the desired [bold]ID[/bold] into "
        "[cyan]caption_template_id[/cyan] in your config.yaml"
    )


if __name__ == "__main__":
    app()
