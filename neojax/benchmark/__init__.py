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

"""Benchmarking utilities."""

from neojax.benchmark.evaluators import BaseEvaluator as BaseEvaluator
from neojax.benchmark.evaluators import SteadyStateEvaluator as SteadyStateEvaluator
from neojax.benchmark.evaluators import TimeDependentEvaluator as TimeDependentEvaluator
from neojax.benchmark.evaluators import save_results as save_results
from neojax.benchmark.runner import BenchmarkRunner as BenchmarkRunner
from neojax.data.download import download_dataset as download_dataset
from neojax.data.download.registry import DATASET_REGISTRY

__all__ = [
    "download_dataset",
    "DATASET_REGISTRY",
    "BaseEvaluator",
    "SteadyStateEvaluator",
    "TimeDependentEvaluator",
    "save_results",
    "BenchmarkRunner",
]
