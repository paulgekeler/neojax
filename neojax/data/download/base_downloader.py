"""Abstract base class for dataset downloaders."""

from abc import ABC, abstractmethod
from pathlib import Path


class BaseDownloader(ABC):
    """Abstract base class for dataset downloaders.

    All downloader implementations (e.g. HTTP, Zenodo, Hugging Face) must
    inherit from this class and implement the `download` method.
    """

    @abstractmethod
    def download(self, target_dir: str | Path, force: bool = False) -> list[str]:
        """Downloads the dataset files to target_dir.

        Args:
            target_dir: Directory where the files should be saved.
            force: If True, forces re-downloading even if files exist and pass validation.

        Returns:
            A list of absolute file paths to the downloaded files.
        """
        ...
