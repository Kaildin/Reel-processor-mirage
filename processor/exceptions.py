"""Custom exceptions for reel-processor."""

from __future__ import annotations


class ReelProcessorError(Exception):
    """Base exception for all reel-processor errors."""


class ConfigError(ReelProcessorError):
    """Raised when configuration is missing or invalid."""


class ScanError(ReelProcessorError):
    """Raised when folder scanning encounters an unrecoverable error."""


class FFmpegError(ReelProcessorError):
    """Raised when ffmpeg/ffprobe fails."""


class FFmpegNotFoundError(FFmpegError):
    """Raised when ffmpeg or ffprobe is not installed."""


class MirageAPIError(ReelProcessorError):
    """Raised when the Mirage API returns an error response."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class MirageConnectionError(MirageAPIError):
    """Raised when the Mirage API is unreachable."""


class MirageJobFailedError(MirageAPIError):
    """Raised when a Mirage job ends in FAILED or CANCELLED status."""

    def __init__(
        self,
        message: str,
        status: str,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.error_code = error_code
        self.error_message = error_message


class MiragePollTimeoutError(MirageAPIError):
    """Raised when polling exceeds max attempts without COMPLETE status."""
