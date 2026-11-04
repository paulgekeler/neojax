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
