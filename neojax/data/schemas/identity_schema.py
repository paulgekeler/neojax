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

"""Schema that simply extracts fields without modification."""

from typing import final

from jaxtyping import Inexact
from typing_extensions import override

from neojax.data.bundles.data_bundle import DataBundle
from neojax.data.schemas.base_schema import BaseSchema
from neojax.data.types import BundleOrArray, JaxNpArray


@final
class IdentitySchema(BaseSchema[BundleOrArray, Inexact[JaxNpArray, "..."]]):
    """Schema that simply returns the fields array without modifying it."""

    @override
    def transform(
        self,
        bundle: BundleOrArray,
        /,
        reference_bundle: DataBundle | None = None,
    ) -> Inexact[JaxNpArray, "..."]:
        """Returns the fields array.

        Args:
            bundle: The input DataBundle.
            reference_bundle: Ignored.

        Returns:
            The fields array.
        """
        if isinstance(bundle, DataBundle):
            return bundle.fields
        return bundle
