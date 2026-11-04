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

"""Data utilities for neural operators."""

from neojax.data import (
    datasets,
    download,
    normalizers,
    scales,
    schemas,
)
from neojax.data.bundles import DataBundle as DataBundle
from neojax.data.download.dataset import download_dataset as download_dataset
from neojax.data.processor import BundleProcessor as BundleProcessor

__all__ = [
    "DataBundle",
    "datasets",
    "download",
    "download_dataset",
    "normalizers",
    "BundleProcessor",
    "scales",
    "schemas",
]
