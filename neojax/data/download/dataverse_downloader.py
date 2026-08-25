"""Downloader that fetches files from Dataverse datasets using DOIs."""

import logging
from pathlib import Path
from typing import final

import requests

from neojax.data.download.base_downloader import BaseDownloader
from neojax.data.download.http_downloader import HTTPDownloader

logger = logging.getLogger(__name__)


@final
class DataverseDownloader(BaseDownloader):
    """Downloader that fetches files from Dataverse datasets using DOIs.

    Queries the Dataverse API to resolve file IDs and checksums (MD5),
    then delegates the download to HTTPDownloader.

    Args:
        doi: The persistent ID (DOI) of the dataset (e.g. '10.18419/darus-2986').
        filenames: Optional list of specific filenames to download.
            If None, downloads all files in the dataset.
        server_url: Base URL of the Dataverse server instance.
            Defaults to "https://darus.uni-stuttgart.de".
        max_retries: Retries for the HTTP downloader. Default is 5.
        backoff_factor: Backoff factor for retries. Default is 1.5.

    ??? info "Internal Attributes"
        These fields store the configuration of the downloader.

        * **doi** (`str`): Dataset DOI.
        * **filenames** (`list[str] | None`): Optional list of specific files to download.
        * **server_url** (`str`): Base URL of the Dataverse instance.
        * **max_retries** (`int`): Max download retries.
        * **backoff_factor** (`float`): Backoff factor.
    """

    doi: str
    filenames: list[str] | None
    server_url: str
    max_retries: int
    backoff_factor: float

    def __init__(
        self,
        doi: str,
        filenames: list[str] | None = None,
        server_url: str = "https://darus.uni-stuttgart.de",
        max_retries: int = 5,
        backoff_factor: float = 1.5,
    ) -> None:
        self.doi = doi
        self.filenames = filenames
        self.server_url = server_url.rstrip("/")
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def download(self, target_dir: str | Path, force: bool = False) -> list[str]:
        """Query Dataverse API to list and download files.

        Args:
            target_dir: Directory where the files should be saved.
            force: If True, forces re-downloading even if files exist and pass validation.

        Returns:
            A list of absolute file paths to the downloaded files.

        Raises:
            requests.HTTPError: If the HTTP request to the Dataverse API fails.
            ValueError: If no files are found in the dataset, or if none of
                the specified filenames are present in the dataset.
        """
        # Ensure doi format is clean for persistentId query parameter
        doi_param = self.doi
        if not doi_param.startswith("doi:"):
            doi_param = f"doi:{doi_param}"

        url = f"{self.server_url}/api/datasets/:persistentId/versions/:latest"
        params = {"persistentId": doi_param}
        headers = {"User-Agent": "neojax-downloader/0.1"}

        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        dataset_data = response.json()

        files_metadata = dataset_data.get("data", {}).get("files", [])
        if not files_metadata:
            raise ValueError(f"No files found in Dataverse dataset {self.doi}.")

        urls = {}
        checksums = {}

        for file_info in files_metadata:
            filename = file_info.get("label")
            if not filename:
                continue

            # If filenames list is provided, filter by it
            if self.filenames is not None and filename not in self.filenames:
                continue

            file_id = file_info.get("dataFile", {}).get("id")
            if not file_id:
                continue

            download_url = f"{self.server_url}/api/access/datafile/{file_id}"
            urls[filename] = download_url

            # Extract MD5 checksum if present (or look under checksum field)
            checksum_val = file_info.get("dataFile", {}).get("md5")
            if not checksum_val:
                checksum_info = file_info.get("dataFile", {}).get("checksum", {})
                if checksum_info.get("type", "").lower() == "md5":
                    checksum_val = checksum_info.get("value")

            if checksum_val:
                checksums[filename] = f"md5:{checksum_val}"

        if not urls:
            raise ValueError(
                f"None of the requested files {self.filenames} were found in "
                f"Dataverse dataset {self.doi}."
            )

        http_downloader = HTTPDownloader(
            urls=urls,
            checksums=checksums,
            max_retries=self.max_retries,
            backoff_factor=self.backoff_factor,
        )
        return http_downloader.download(target_dir, force=force)
