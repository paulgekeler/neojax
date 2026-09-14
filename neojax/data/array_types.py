"""Shared array-type alias for host/device-agnostic data containers."""

import jax
import numpy as np

JaxNpArray = jax.Array | np.ndarray
"""Base type for jaxtyping shape/dtype annotations that accept either a numpy or jax array.

A sub type of jaxtyping.ArrayLike excluding python built-in types
and numpy scalar-like types (np.bool, np.number).

`DataBundle`, `BundleDataset`, and `RawDataset` sit on the boundary between host-resident
data (plain numpy, as returned by the dataset loading utilities to avoid eagerly placing
a whole dataset on an accelerator) and jax-converted data (once a batch has flowed through
normalization/schemas). Using bare `jaxtyping.Array` (which resolves to `jax.Array`)
or `jaxtyping.ArrayLike` doesn't work. This alias keeps the annotations
meaningful for both.
"""
