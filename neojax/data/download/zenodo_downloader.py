"""Downloader that fetches files from Zenodo record IDs."""

import logging
from pathlib import Path
from typing import final

import requests

from neojax.data.download.base_downloader import BaseDownloader
from neojax.data.download.http_downloader import HTTPDownloader

logger = logging.getLogger(__name__)


@final
class ZenodoDownloader(BaseDownloader):
    """Downloader that fetches files from Zenodo record IDs.

    Queries the Zenodo Record API to discover download URLs and checksums,
    then delegates the download to HTTPDownloader.

    Args:
        record_id: The Zenodo Record ID.
        filenames: Optional list of specific filenames to download.
            If None, downloads all files in the record.
        max_retries: Retries for the HTTP downloader. Default is 5.
        backoff_factor: Backoff factor for retries. Default is 1.5.

    ??? info "Internal Attributes"
        These fields store the configuration of the downloader.

        * **record_id** (`str`): The Zenodo Record ID.
        * **filenames** (`list[str] | None`): Optional list of specific filenames to download.
        * **max_retries** (`int`): Max download retries.
        * **backoff_factor** (`float`): Backoff factor.
    """

    record_id: str
    filenames: list[str] | None
    max_retries: int
    backoff_factor: float

    def __init__(
        self,
        record_id: str,
        filenames: list[str] | None = None,
        max_retries: int = 5,
        backoff_factor: float = 1.5,
    ) -> None:
        self.record_id = record_id
        self.filenames = filenames
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def download(self, target_dir: str | Path, force: bool = False) -> list[str]:
        """Query Zenodo API to list and download files.

        Args:
            target_dir: Directory where the files should be saved.
            force: If True, forces re-downloading even if files exist and pass validation.

        Returns:
            A list of absolute file paths to the downloaded files.

        Raises:
            requests.HTTPError: If the HTTP request to the Zenodo API fails.
            ValueError: If no files are found in the Zenodo record, or if none of
                the specified filenames are present in the record.
        """
        url = f"https://zenodo.org/api/records/{self.record_id}"
        headers = {"User-Agent": "neojax-downloader/0.1"}

        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        record_data = response.json()

        files_metadata = record_data.get("files", [])
        if not files_metadata:
            raise ValueError(f"No files found in Zenodo record {self.record_id}.")

        urls = {}
        checksums = {}

        for file_info in files_metadata:
            filename = file_info.get("key") or file_info.get("filename")
            if not filename:
                continue

            # If filenames list is provided, filter by it
            if self.filenames is not None and filename not in self.filenames:
                continue

            download_url = None
            if "links" in file_info:
                download_url = file_info["links"].get("content") or file_info[
                    "links"
                ].get("self")

            if not download_url:
                download_url = (
                    f"https://zenodo.org/records/{self.record_id}/files/"
                    f"{filename}/content"
                )

            urls[filename] = download_url

            # Extract checksum if present
            checksum = file_info.get("checksum")
            if checksum:
                checksums[filename] = checksum

        if not urls:
            raise ValueError(
                f"None of the requested files {self.filenames} were found in "
                f"Zenodo record {self.record_id}."
            )

        http_downloader = HTTPDownloader(
            urls=urls,
            checksums=checksums,
            max_retries=self.max_retries,
            backoff_factor=self.backoff_factor,
        )
        return http_downloader.download(target_dir, force=force)
