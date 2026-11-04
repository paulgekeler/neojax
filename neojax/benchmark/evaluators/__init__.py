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

"""Evaluators for benchmarking neural operators."""

from neojax.benchmark.evaluators.base_evaluator import BaseEvaluator as BaseEvaluator
from neojax.benchmark.evaluators.base_evaluator import save_results as save_results
from neojax.benchmark.evaluators.steady_state_eval import (
    SteadyStateEvaluator as SteadyStateEvaluator,
)
from neojax.benchmark.evaluators.time_dependent_eval import (
    TimeDependentEvaluator as TimeDependentEvaluator,
)

__all__ = [
    "BaseEvaluator",
    "SteadyStateEvaluator",
    "TimeDependentEvaluator",
    "save_results",
]
