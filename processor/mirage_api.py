"""Mirage API client for caption upload, polling, and download."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Callable

import requests
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import RequestException, Timeout
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor

from processor.config import Config
from processor.exceptions import (
    MirageAPIError,
    MirageConnectionError,
    MirageJobFailedError,
    MiragePollTimeoutError,
)

TERMINAL_FAILURE_STATUSES = {"FAILED", "CANCELLED"}
ACTIVE_STATUSES = {"QUEUED", "PROCESSING", "COMPLETE"}

# HTTP status codes that are transient and worth retrying.
# 408 = Request Timeout (server closed before we finished sending)
# 429 = Rate limit
# 5xx = Server-side errors
_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}

# Timeout tuple: (connect_timeout_seconds, read_timeout_seconds)
# Upload needs a long read timeout because sending large files takes time.
_DEFAULT_TIMEOUT = (30, 120)
_UPLOAD_TIMEOUT = (30, 600)  # up to 10 min for large video uploads


def _backoff(attempt: int) -> float:
    """Exponential backoff with full jitter: sleep [0, 2^attempt] seconds, max 30s."""
    return random.uniform(0, min(2**attempt, 30))


class MirageClient:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._session = requests.Session()
        self._session.headers.update({"x-api-key": config.mirage_api_key})

    def _url(self, path: str) -> str:
        base = self._config.mirage_base_url.rstrip("/")
        return f"{base}{path}"

    def _request_with_retry(
        self,
        method: str,
        url: str,
        timeout: tuple[int, int] = _DEFAULT_TIMEOUT,
        **kwargs: Any,
    ) -> requests.Response:
        max_retries = 3
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                response = self._session.request(method, url, timeout=timeout, **kwargs)
            except (RequestsConnectionError, Timeout) as exc:
                last_error = exc
                if attempt < max_retries:
                    time.sleep(_backoff(attempt))
                    continue
                raise MirageConnectionError(
                    "Cannot reach Mirage API. Check your internet connection."
                ) from exc
            except RequestException as exc:
                raise MirageConnectionError(
                    f"Network error contacting Mirage API: {exc}"
                ) from exc

            if response.status_code in _RETRYABLE_STATUS_CODES:
                if attempt < max_retries:
                    time.sleep(_backoff(attempt))
                    continue
                raise MirageAPIError(
                    f"Mirage API error {response.status_code}: {response.text}",
                    status_code=response.status_code,
                )

            if response.status_code >= 400:
                raise MirageAPIError(
                    f"Mirage API error {response.status_code}: {response.text}",
                    status_code=response.status_code,
                )

            return response

        raise MirageConnectionError(
            f"Request failed after {max_retries} retries: {last_error}"
        )

    def upload_for_captions(
        self,
        video_path: Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> str:
        """Upload *video_path* and return the Mirage video_id.

        Uses a longer timeout (_UPLOAD_TIMEOUT) so large files don't trigger
        a 408 before the upload finishes. The 408 is also in the retry list
        so transient server-side timeouts are automatically recovered.

        If *progress_callback* is provided it is called with
        ``(bytes_sent, total_bytes)`` after each chunk.
        """
        url = self._url("/videos/captions")
        file_size = video_path.stat().st_size

        with video_path.open("rb") as video_file:
            encoder = MultipartEncoder(
                fields={
                    "caption_template_id": self._config.caption_template_id,
                    "video": (video_path.name, video_file, "video/mp4"),
                }
            )

            def _monitor_callback(monitor: MultipartEncoderMonitor) -> None:
                if progress_callback is not None:
                    progress_callback(monitor.bytes_read, file_size)

            monitor = MultipartEncoderMonitor(encoder, _monitor_callback)

            response = self._request_with_retry(
                "POST",
                url,
                timeout=_UPLOAD_TIMEOUT,
                data=monitor,
                headers={"Content-Type": monitor.content_type},
            )

        data = response.json()
        video_id = data.get("id") or data.get("video_id")
        if not video_id:
            raise MirageAPIError(f"No video_id in upload response: {data}")
        return str(video_id)

    def get_video_status(self, video_id: str) -> dict[str, Any]:
        url = self._url(f"/videos/{video_id}")
        response = self._request_with_retry("GET", url)
        return response.json()

    def poll_until_complete(self, video_id: str) -> dict[str, Any]:
        interval = self._config.poll_interval_seconds
        max_attempts = self._config.max_poll_attempts

        for attempt in range(1, max_attempts + 1):
            data = self.get_video_status(video_id)
            status = data.get("status", "UNKNOWN")

            if status == "COMPLETE":
                return data

            if status in TERMINAL_FAILURE_STATUSES:
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

    def download_video(
        self,
        video_id: str,
        output_path: Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Download the captioned video, optionally reporting progress.

        *progress_callback* is called with ``(bytes_downloaded, total_bytes)``
        after each chunk.  *total_bytes* is -1 when Content-Length is absent.
        """
        url = self._url(f"/videos/{video_id}/content")
        response = self._request_with_retry(
            "GET", url, allow_redirects=True, stream=True
        )
        total = int(response.headers.get("Content-Length", -1))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        downloaded = 0
        with output_path.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=256 * 1024):
                if not chunk:
                    continue
                fh.write(chunk)
                downloaded += len(chunk)
                if progress_callback is not None:
                    progress_callback(downloaded, total)

        return output_path

    def list_caption_templates(self) -> list[dict[str, Any]]:
        templates: list[dict[str, Any]] = []
        after: str | None = None

        while True:
            params: dict[str, Any] = {"limit": 100}
            if after:
                params["after"] = after

            response = self._request_with_retry(
                "GET",
                self._url("/videos/captions/templates"),
                params=params,
            )
            payload = response.json()
            batch = payload.get("data", [])
            templates.extend(batch)

            if not payload.get("has_more") or not batch:
                break
            after = batch[-1].get("id")
            if not after:
                break

        return templates
