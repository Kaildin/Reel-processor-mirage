"""Mirage API client for caption upload, polling, and download."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import requests
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import RequestException, Timeout

from processor.config import Config
from processor.exceptions import (
    MirageAPIError,
    MirageConnectionError,
    MirageJobFailedError,
    MiragePollTimeoutError,
)

TERMINAL_FAILURE_STATUSES = {"FAILED", "CANCELLED"}
ACTIVE_STATUSES = {"QUEUED", "PROCESSING", "COMPLETE"}


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
        **kwargs: Any,
    ) -> requests.Response:
        max_retries = 3
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                response = self._session.request(method, url, timeout=120, **kwargs)
            except (RequestsConnectionError, Timeout) as exc:
                last_error = exc
                if attempt < max_retries:
                    time.sleep(2**attempt)
                    continue
                raise MirageConnectionError(
                    "Cannot reach Mirage API. Check your internet connection."
                ) from exc
            except RequestException as exc:
                raise MirageConnectionError(
                    f"Network error contacting Mirage API: {exc}"
                ) from exc

            if response.status_code in (429, 500, 502, 503, 504):
                if attempt < max_retries:
                    time.sleep(2**attempt)
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
            f"Request failed after retries: {last_error}"
        )

    def upload_for_captions(self, video_path: Path) -> str:
        url = self._url("/videos/captions")
        with video_path.open("rb") as video_file:
            response = self._request_with_retry(
                "POST",
                url,
                data={"caption_template_id": self._config.caption_template_id},
                files={"video": (video_path.name, video_file, "video/mp4")},
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

    def download_video(self, video_id: str, output_path: Path) -> Path:
        url = self._url(f"/videos/{video_id}/content")
        response = self._request_with_retry("GET", url, allow_redirects=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.content)
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
