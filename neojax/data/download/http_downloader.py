"""Downloader that fetches files from HTTP/HTTPS URLs."""

import hashlib
import importlib
import logging
import time
from pathlib import Path
from typing import Any, final

import requests

from neojax.data.download.base_downloader import BaseDownloader

logger = logging.getLogger(__name__)


def _show_progress(downloaded: int, total: int, filename: str) -> None:
    """Fallback progress bar that prints progress to the console."""
    if total <= 0:
        print(f"Downloading {filename}: {downloaded / 1024 / 1024:.2f} MB", end="\r")
        return
    percent = (downloaded / total) * 100
    bar_length = 40
    filled_length = round(bar_length * downloaded / float(total))
    bar = "=" * filled_length + "-" * (bar_length - filled_length)
    print(
        f"Downloading {filename}: [{bar}] {percent:.1f}% "
        f"({downloaded / 1024 / 1024:.2f} / {total / 1024 / 1024:.2f} MB)",
        end="\r",
    )


@final
class HTTPDownloader(BaseDownloader):
    """Downloader that fetches files from HTTP/HTTPS URLs.

    Resilience features include connection retries with exponential
    backoff, range-based resuming of interrupted downloads, and
    checksum verification.

    Args:
        urls: A dictionary mapping target filenames (relative to target_dir)
            to their download URLs.
        checksums: Optional dictionary mapping filenames to their expected
            checksum strings (e.g. 'md5:xxx' or 'sha256:xxx').
        chunk_size: Chunk size in bytes for reading the stream. Default is 1MB.
        max_retries: Maximum number of download retries on failure. Default is 5.
        backoff_factor: Exponential backoff factor for retries. Default is 1.5.

    ??? info "Internal Attributes"
        These fields store the configuration of the downloader.

        * **urls** (`dict[str, str]`): Target filenames mapped to download URLs.
        * **checksums** (`dict[str, str]`): Target filenames mapped to expected checksums.
        * **chunk_size** (`int`): Chunk size in bytes.
        * **max_retries** (`int`): Max download retries on failure.
        * **backoff_factor** (`float`): Backoff factor for exponential backoff.
    """

    urls: dict[str, str]
    checksums: dict[str, str]
    chunk_size: int
    max_retries: int
    backoff_factor: float

    def __init__(
        self,
        urls: dict[str, str],
        checksums: dict[str, str] | None = None,
        chunk_size: int = 1024 * 1024,  # 1 MB
        max_retries: int = 5,
        backoff_factor: float = 1.5,
    ) -> None:
        self.urls = urls
        self.checksums = checksums or {}
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def _verify_checksum(self, filepath: str, expected_checksum_str: str) -> bool:
        """Verifies the checksum of a file.

        Args:
            filepath: The path to the file to verify.
            expected_checksum_str: Expected checksum string, e.g. 'md5:hash'.

        Returns:
            True if checksum matches or if no expected checksum is specified, False otherwise.
        """
        if not expected_checksum_str:
            return True

        parts = expected_checksum_str.split(":", 1)
        if len(parts) == 2:
            algo, expected = parts[0].lower(), parts[1]
        else:
            algo, expected = "md5", parts[0]

        if algo not in ("md5", "sha256"):
            logger.warning(
                f"Unsupported checksum algorithm: {algo}. Defaulting to md5."
            )
            algo = "md5"

        h = hashlib.md5() if algo == "md5" else hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest().lower() == expected.lower()
        except OSError:
            return False

    def _download_file(
        self, filename: str, url: str, target_dir: str | Path, force: bool
    ) -> str:
        """Downloads a single file from the URL with retry and resume support.

        Args:
            filename: The name of the file to save.
            url: The URL to download from.
            target_dir: The directory to save the file in.
            force: Whether to overwrite existing files.

        Returns:
            The absolute path to the downloaded file.

        Raises:
            requests.RequestException: If the HTTP request/connection fails.
            ValueError: If the downloaded file fails checksum verification.
            OSError: If a file system error occurs during reading or writing.
            RuntimeError: If the download loop terminates unexpectedly.
        """
        filepath = Path(target_dir) / filename
        filepath.parent.mkdir(exist_ok=True)

        expected_checksum = self.checksums.get(filename)

        # Check if file exists and is valid
        if not force and filepath.exists():
            if expected_checksum:
                if self._verify_checksum(filepath, expected_checksum):
                    logger.info(
                        f"File {filename} already exists and passes checksum validation."
                    )
                    return filepath
                else:
                    logger.warning(
                        f"File {filename} exists but failed checksum. Re-downloading."
                    )
            else:
                logger.info(
                    f"File {filename} already exists (no checksum provided). Skipping."
                )
                return filepath

        # Perform download with retry/resume
        retry = 0
        while retry <= self.max_retries:
            try:
                # Check for partial download
                temp_filepath = filepath.with_suffix(".part")
                resume_header = {}
                downloaded_bytes = 0

                # Force re-download ignores partial download
                if not force and temp_filepath.exists():
                    downloaded_bytes = temp_filepath.stat().st_size
                    # Use Range header for resume
                    resume_header = {"Range": f"bytes={downloaded_bytes}-"}
                    logger.info(
                        f"Resuming download of {filename} from {downloaded_bytes} bytes."
                    )

                # Start request
                headers = {"User-Agent": "neojax-downloader/0.1"}
                headers.update(resume_header)

                response = requests.get(url, stream=True, headers=headers, timeout=30)

                # Check status code
                # 206 Partial Content if resumed, 200 OK otherwise
                if response.status_code == 206:
                    write_mode = "ab"
                    total_bytes = downloaded_bytes + int(
                        response.headers.get("content-length", 0)
                    )
                elif response.status_code == 200:
                    write_mode = "wb"
                    downloaded_bytes = 0
                    total_bytes = int(response.headers.get("content-length", 0))
                elif response.status_code == 416:
                    # Requested range not satisfiable - file might be complete
                    logger.warning(
                        "HTTP Range Not Satisfiable (416). Resetting part file."
                    )
                    if temp_filepath.exists():
                        temp_filepath.unlink()
                    write_mode = "wb"
                    downloaded_bytes = 0
                    response = requests.get(
                        url,
                        stream=True,
                        headers={"User-Agent": "neojax-downloader/0.1"},
                        timeout=30,
                    )
                    total_bytes = int(response.headers.get("content-length", 0))
                else:
                    response.raise_for_status()
                    write_mode = "wb"
                    downloaded_bytes = 0
                    total_bytes = int(response.headers.get("content-length", 0))

                # Read response content in chunks
                pbar: Any = None
                try:
                    tqdm = importlib.import_module("tqdm")
                    pbar = tqdm.tqdm(
                        total=total_bytes,
                        initial=downloaded_bytes,
                        unit="B",
                        unit_scale=True,
                        desc=filename,
                        leave=True,
                    )
                except ImportError:
                    pass

                with open(temp_filepath, write_mode) as f:
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded_bytes += len(chunk)
                            if pbar:
                                pbar.update(len(chunk))
                            else:
                                _show_progress(downloaded_bytes, total_bytes, filename)

                if pbar:
                    pbar.close()
                else:
                    print()  # Add newline after print carriage return

                # Move part file to final file location
                if filepath.exists():
                    filepath.unlink()
                temp_filepath.rename(filepath)

                # Verify checksum of finished download
                if expected_checksum:
                    if not self._verify_checksum(filepath, expected_checksum):
                        raise ValueError(
                            f"Downloaded file {filename} failed checksum validation."
                        )

                logger.info(f"Successfully downloaded {filename}.")
                return filepath

            except (OSError, requests.RequestException, ValueError) as e:
                retry += 1
                if retry > self.max_retries:
                    logger.error(
                        f"Failed to download {filename} after {self.max_retries} "
                        f"attempts. Error: {e}"
                    )
                    # Cleanup partial file on fatal failure
                    temp_filepath = filepath.with_suffix(".part")
                    if temp_filepath.exists():
                        try:
                            temp_filepath.unlink()
                        except OSError:
                            pass
                    raise e

                sleep_time = self.backoff_factor**retry
                logger.warning(
                    f"Download error: {e}. Retrying {retry}/{self.max_retries} "
                    f"in {sleep_time:.2f}s..."
                )
                time.sleep(sleep_time)

        raise RuntimeError(f"Unexpected termination of download loop for {filename}.")

    def download(self, target_dir: str | Path, force: bool = False) -> list[str]:
        """Download all configured URLs.

        Args:
            target_dir: Directory where the files should be saved.
            force: If True, forces re-downloading even if files exist and pass validation.

        Returns:
            A list of absolute file paths to the downloaded files.

        Raises:
            requests.RequestException: If the HTTP request/connection fails.
            ValueError: If a downloaded file fails checksum verification.
            OSError: If a file system error occurs during reading or writing.
            RuntimeError: If the download loop terminates unexpectedly.
        """
        filepaths = []
        for filename, url in self.urls.items():
            filepath = self._download_file(filename, url, target_dir, force=force)
            filepaths.append(filepath)
        return filepaths
