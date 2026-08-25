"""Downloader that fetches files from Hugging Face Hub repositories."""

import logging
from pathlib import Path
from typing import final

import requests

from neojax.data.download.base_downloader import BaseDownloader
from neojax.data.download.http_downloader import HTTPDownloader

logger = logging.getLogger(__name__)


@final
class HuggingFaceDownloader(BaseDownloader):
    """Downloader that fetches files from Hugging Face Hub repositories.

    Queries the Hugging Face Dataset API to list repository siblings,
    filters out repository configuration/metadata files, and downloads
    the dataset files using direct LFS resolve links.

    Args:
        repo_id: Hugging Face dataset repository ID (e.g. 'camlab-ethz/ACE').
        filenames: Optional list of specific files to download. If None,
            downloads all non-metadata files.
        revision: Git branch or commit hash to download from. Defaults to 'main'.
        max_retries: Retries for the HTTP downloader. Default is 5.
        backoff_factor: Backoff factor for retries. Default is 1.5.

    ??? info "Internal Attributes"
        These fields store the configuration of the downloader.

        * **repo_id** (`str`): Hugging Face repository ID.
        * **filenames** (`list[str] | None`): Optional list of specific files to download.
        * **revision** (`str`): Git branch/tag/commit hash.
        * **max_retries** (`int`): Max download retries.
        * **backoff_factor** (`float`): Backoff factor.
    """

    repo_id: str
    filenames: list[str] | None
    revision: str
    max_retries: int
    backoff_factor: float

    def __init__(
        self,
        repo_id: str,
        filenames: list[str] | None = None,
        revision: str = "main",
        max_retries: int = 5,
        backoff_factor: float = 1.5,
    ) -> None:
        self.repo_id = repo_id
        self.filenames = filenames
        self.revision = revision
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def download(self, target_dir: str | Path, force: bool = False) -> list[str]:
        """Query Hugging Face API to list files and download them.

        Args:
            target_dir: Directory where the files should be saved.
            force: If True, forces re-downloading even if files exist and pass validation.

        Returns:
            A list of absolute file paths to the downloaded files.

        Raises:
            requests.HTTPError: If the HTTP request to the Hugging Face API fails.
            ValueError: If no files are found in the Hugging Face repository, or if
                no matching files are found to download.
        """
        url = f"https://huggingface.co/api/datasets/{self.repo_id}"
        headers = {"User-Agent": "neojax-downloader/0.1"}

        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        repo_info = response.json()

        siblings = repo_info.get("siblings", [])
        if not siblings:
            raise ValueError(
                f"No files found in Hugging Face repository {self.repo_id}."
            )

        urls = {}

        for sibling in siblings:
            rpath = sibling.get("rpath") or sibling.get("rfilename")
            if not rpath:
                continue

            # If filenames is specified, filter by it
            if self.filenames is not None:
                if rpath not in self.filenames:
                    continue
            else:
                # Default filter: skip metadata and source code files
                lower_path = rpath.lower()
                if (
                    lower_path.startswith(".")
                    or lower_path in ("readme.md", "license", "metadata.yaml")
                    or lower_path.endswith(
                        (".py", ".md", ".txt", ".gitattributes", ".gitignore")
                    )
                ):
                    continue

            download_url = (
                f"https://huggingface.co/datasets/{self.repo_id}/resolve/"
                f"{self.revision}/{rpath}"
            )
            urls[rpath] = download_url

        if not urls:
            raise ValueError(
                f"No matching files to download in Hugging Face repository {self.repo_id}."
            )

        http_downloader = HTTPDownloader(
            urls=urls,
            max_retries=self.max_retries,
            backoff_factor=self.backoff_factor,
        )
        return http_downloader.download(target_dir, force=force)
