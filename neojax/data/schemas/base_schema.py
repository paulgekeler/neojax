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

"""Implementation of the base schema."""

from abc import abstractmethod
from typing import Generic, TypeVar

import equinox as eqx

from neojax.data.bundles.data_bundle import DataBundle

TIn = TypeVar("TIn")
TOut = TypeVar("TOut")


class BaseSchema(eqx.Module, Generic[TIn, TOut]):
    """Abstract base schema.

    All custom schemas should inherit from BaseSchema.
    """

    @abstractmethod
    def transform(
        self,
        bundle_or_model_output: TIn,
        /,
        reference_bundle: DataBundle | None = None,
    ) -> TOut:
        """Transforms a data bundle into the needed model format.

        Args:
            bundle_or_model_output: If schema is an input schema,
                an instance of `DataBundle`,
                if schema is an output schema, any pytree the model outputs.
            reference_bundle: If schema is an output schema, the original input
                `DataBundle` to inherit geometries and unmodified fields from.
                Ignored for input schemas.

        Returns:
            If schema is an input schema, returns a raw array or pytree in model format.
            If schema is an output schema, returns a `DataBundle` instance.
        """
        ...
