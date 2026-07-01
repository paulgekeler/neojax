"""Benchmarking utilities."""

from neojax.data.download import download_dataset
from neojax.data.download.registry import DATASET_REGISTRY

__all__ = [
    "download_dataset",
    "DATASET_REGISTRY",
]
