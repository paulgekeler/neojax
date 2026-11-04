# Copyright 2026 Paul Gekeler
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main function to resolve and download datasets."""

import logging
from pathlib import Path

from neojax.data.download.dataverse_downloader import DataverseDownloader
from neojax.data.download.http_downloader import HTTPDownloader
from neojax.data.download.huggingface_downloader import HuggingFaceDownloader
from neojax.data.download.registry import DATASET_REGISTRY
from neojax.data.download.zenodo_downloader import ZenodoDownloader

logger = logging.getLogger(__name__)


def download_dataset(
    name: str, target_dir: str | Path, force: bool = False
) -> list[str]:
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
        downloader = HuggingFaceDownloader(**params)
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
