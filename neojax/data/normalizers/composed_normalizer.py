"""Implementation of a composed normalizer."""

from typing import final

import equinox as eqx
from jaxtyping import Array, Float

from neojax.data.normalizers.base_normalizer import BaseNormalizer


@final
class ComposedNormalizer(BaseNormalizer):
    """Composes multiple normalizers into a pipeline.

    The normalizers are applied sequentially in the order they are
    provided. This class allows for complex normalization pipelines,
    such as first applying a robust scaler and then a min-max scaling.

    Args:
        *normalizers: Normalizer instances to be composed in the
            pipeline.

    !!! info "Internal Attributes"
        These fields store the internal state of the normalizer.

        * **stats** (`dict[str, tuple[BaseNormalizer, ...]]`): Dict containing 'normalizers', which stores the sequence of normalizers as a tuple.

    Examples:
        ```python
        import jax.numpy as jnp
        from neojax.data.normalizers import (
            ComposedNormalizer,
            MinMaxNormalizer,
            UnitGaussianNormalizer
        )

        norm = ComposedNormalizer(
            MinMaxNormalizer(),
            UnitGaussianNormalizer()
        )

        data = jnp.ones((3, 3, 3))
        # compute stats (sequentially)
        norm = norm.compute_stats(data)

        transformed = norm(data)
        ```

    !!! info "Notes"
        If all normalizer statistics need to be computed on raw data,
        pass `sequential=False` to `compute_stats()`.
    """

    def __init__(self, *normalizers: BaseNormalizer) -> None:
        self.stats = {"normalizers": tuple(normalizer for normalizer in normalizers)}

    def compute_stats(
        self,
        data: Float[Array, "c ..."],
        axis: int | tuple[int, ...] | None = None,
        sequential: bool = True,
    ) -> "ComposedNormalizer":
        """Computes statistics for each normalizer in the pipeline.

        Each normalizer in the pipeline computes its statistics
        independently based on the provided data, either sequentially
        of non-sequentially. See `sequential` for details.

        Args:
            data: The input array to compute statistics from.
            axis: Axis or axes along which the statistics are computed.
                The default (None) computes the global statistics.
            sequential: Whether to apply stats computations in sequence,
                i.e., compute the stats on the transformed inputs of
                the previous normalizer. Non-sequential computes each
                stat on the raw inputs (e.g. for parallel computation).
                Default is sequential.

        Returns:
            A new ComposedNormalizer instance with updated statistics
            for all sub-normalizers.
        """
        new_normalizers = []
        current_data = data
        for normalizer in self.stats["normalizers"]:
            # Compute stats on current (potentially transformed) data
            new_norm = normalizer.compute_stats(current_data, axis)
            new_normalizers.append(new_norm)

            if sequential:
                # Progress the data through the pipeline for the next step
                current_data = new_norm.transform(current_data)

        new_stats = {"normalizers": tuple(new_normalizers)}
        return eqx.tree_at(lambda n: n.stats, self, new_stats)

    def transform(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        """Applies the sequence of normalizations to the input.

        Args:
            x: The input array to transform.

        Returns:
            The transformed array.
        """
        for normalizer in self.stats["normalizers"]:
            x = normalizer(x)
        return x

    def inverse_transform(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        """Reverts the sequence of normalizations in reverse order.

        Args:
            x: The transformed array to revert.

        Returns:
            The de-normalized array.
        """
        for normalizer in self.stats["normalizers"][::-1]:
            x = normalizer.inverse_transform(x)
        return x

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        """Standardizes input by applying the transform pipeline.

        Args:
            x: Input array.

        Returns:
            Normalized input array.
        """
        return self.transform(x)
