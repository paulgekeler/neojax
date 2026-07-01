"""Main function to resolve and download datasets."""

import logging

from neojax.data.download.base_downloader import BaseDownloader
from neojax.data.download.dataverse_downloader import DataverseDownloader
from neojax.data.download.http_downloader import HTTPDownloader
from neojax.data.download.huggingface_downloader import HuggingFaceDownloader
from neojax.data.download.registry import DATASET_REGISTRY
from neojax.data.download.zenodo_downloader import ZenodoDownloader

logger = logging.getLogger(__name__)


def download_dataset(name: str, target_dir: str, force: bool = False) -> list[str]:
    """Downloads a dataset from the registry by name.

    Args:
        name: The name of the registered dataset (case-insensitive).
        target_dir: Directory where the files should be saved.
        force: If True, forces re-downloading even if files exist and pass validation.

    Returns:
        A list of absolute file paths to the downloaded files.

    Raises:
        ValueError: If the dataset name is not registered, or if the downloader
            is invalid.
    """
    key = name.lower()
    if key not in DATASET_REGISTRY:
        raise ValueError(
            f"Dataset '{name}' is not registered. Registered datasets: "
            f"{list(DATASET_REGISTRY.keys())}"
        )

    config = DATASET_REGISTRY[key]
    downloader_type = config.get("downloader", "").lower()
    params = config.get("params", {})

    if downloader_type == "huggingface":
        downloader: BaseDownloader = HuggingFaceDownloader(**params)
    elif downloader_type == "zenodo":
        downloader = ZenodoDownloader(**params)
    elif downloader_type == "dataverse":
        downloader = DataverseDownloader(**params)
    elif downloader_type == "http":
        downloader = HTTPDownloader(**params)
    else:
        raise ValueError(
            f"Unsupported downloader type '{downloader_type}' for dataset '{name}'."
        )

    logger.info(
        f"Starting download of dataset '{name}' to {target_dir} using "
        f"{downloader_type} downloader..."
    )
    return downloader.download(target_dir, force=force)
