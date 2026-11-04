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

"""Metrics for neural operators."""

from neojax.metrics.base_metric import BaseMetric
from neojax.metrics.composed_metric import ComposedMetric
from neojax.metrics.lp_metrics import LpMetric, RelativeLpMetric
from neojax.metrics.physical_metrics import (
    BoundaryConsistencyMetric as BoundaryConsistencyMetric,
)
from neojax.metrics.physical_metrics import ConservationMetric as ConservationMetric
from neojax.metrics.physical_metrics import ResidualMetric as ResidualMetric
from neojax.metrics.regression_metrics import MSEMetric as MSEMetric
from neojax.metrics.regression_metrics import R2Metric as R2Metric
from neojax.metrics.regression_metrics import RMSEMetric as RMSEMetric
from neojax.metrics.sobolev_metrics import SobolevMetric as SobolevMetric
from neojax.metrics.utils import (
    is_learnable_metric_weight as is_learnable_metric_weight,
)

__all__ = [
    "BaseMetric",
    "ComposedMetric",
    "LpMetric",
    "RelativeLpMetric",
    "MSEMetric",
    "RMSEMetric",
    "R2Metric",
    "BoundaryConsistencyMetric",
    "ConservationMetric",
    "ResidualMetric",
    "SobolevMetric",
    "is_learnable_metric_weight",
]
