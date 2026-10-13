"""Implementation of the Transolver Neural Operator."""

from collections.abc import Callable
from typing import Any, Literal, final

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import Array, Float, Inexact, PRNGKeyArray, PyTree
from typing_extensions import override

from neojax.models.baseno import BaseNO
from neojax.nn.normalization import ChannelLayerNorm
from neojax.nn.pointwise_mlp import PointwiseMLP
from neojax.nn.positional_embedding import GridEmbeddingNd, ReferenceGridEmbeddingNd
from neojax.nn.transolver_block import TransolverBlock


def _is_linear(x: Any) -> bool:
    return isinstance(x, eqx.nn.Linear | PointwiseMLP)


def _linear_weights(tree: PyTree) -> tuple[Any, ...]:
    """Collects the weight matrices of all linear layers in a pytree.

    Used as `where` function of `eqx.tree_at`, which passes wrapped leaves,
    hence the loose return type.
    """
    weights: list[Any] = []
    for module in jtu.tree_leaves(tree, is_leaf=_is_linear):
        if isinstance(module, PointwiseMLP):
            weights.extend(module.weights)
        elif isinstance(module, eqx.nn.Linear):
            weights.append(module.weight)
    return tuple(weights)


def _trunc_normal_init(key: PRNGKeyArray, tree: PyTree, std: float) -> PyTree:
    """Re-initializes all linear weights of a pytree with a truncated normal.

    Mirrors `timm.trunc_normal_(std=std)` (truncation at +-2 in absolute value)
    as used by the reference implementation.
    """
    weights = _linear_weights(tree)
    keys = jr.split(key, len(weights))
    new_weights = tuple(
        std * jr.truncated_normal(k, -2.0 / std, 2.0 / std, w.shape, w.dtype)
        for k, w in zip(keys, weights, strict=True)
    )
    return eqx.tree_at(_linear_weights, tree, new_weights)


@final
class Transolver(BaseNO):
    r"""Transolver Neural Operator.

    The model consists of a lifting MLP that maps the input features
    (and positional information) to a latent space, a learnable placeholder
    that is added to every point, a stack of `TransolverBlock`s,
    and a final layer norm and linear projection.
    The implementation follows the reference.

    The model supports unstructured meshes (`use_mesh=True`), where an input
    signal `u` of shape `(in_channels, N)` is accompanied by the coordinates `x_in` of its
    `N` nodes, as well as structured grids (`use_mesh=False`) of shape
    `(in_channels, d1, ..., d_ndim)`, where the attention projections are convolutions.

    Args:
        key: PRNG key for parameter initialization.
        in_channels: Number of input channels of the signal `u`.
            On structured grids, this includes the coordinates if they are
            part of the signal (see `positional_embedding`).
        out_channels: Number of output channels.
        ndim: Number of spatial dimensions (dimension of the coordinates).
        hidden_channels: Hidden channel dimension (latent width). Default is 256.
        n_layers: Number of Transolver blocks. Default is 5.
        n_heads: Number of attention heads. `hidden_channels` must be divisible
            by `n_heads`. Default is 8.
        slice_num: Number of slices (slice tokens) of the physics attention.
            Default is 32.
        mlp_ratio: Ratio of the hidden width of the MLP in each block
            to `hidden_channels`. Default is 1.0.
        dropout: Dropout probability in the physics attention. Default is 0.0.
        activation: Activation function used within the layers.
            Defaults to `jax.nn.gelu`.
        use_mesh: Whether the model operates on unstructured meshes (`True`)
            or structured grids (`False`). On meshes, the node coordinates `x_in`
            are required in the forward pass and are appended to the input signal
            and the attention uses pointwise projections. On grids,
            the attention uses convolutional projections.
            Default is `False`.
        kernel_size: Kernel size of the convolutional attention projections
            on structured grids. Ignored if `use_mesh` is `True`. Must be odd.
            Default is 3.
        lift_channel_ratio: Ratio of the hidden width of the lifting MLP
            to `hidden_channels`. Default is 2.0.
        positional_embedding: Positional embedding to apply
            to the last channels of the raw input on structured grids.
            Pass `"grid"` for a regular `GridEmbeddingNd` on a ((0, 1), ...) grid,
            a `GridEmbeddingNd` instance, or `None` if the coordinates are part of the input
            signal already (e.g., by using `ConcatenateCoordsSchema`) or not used.
            Not supported if `use_mesh` is `True` (node coordinates are always used)
            or `unified_pos` is `True`. Default is `None`.
        unified_pos: Whether to use the "unified position" embedding of the reference:
            instead of the raw coordinates, the distances of every point to the nodes
            of a regular `ref ** ndim` reference grid (see `ReferenceGridEmbedding`)
            are appended to the input signal. Default is `False`.
        ref: Number of reference grid nodes per dimension of the unified position
            embedding. Default is 8.
        temperature_range: Interval the learnable slice temperatures are clipped to
            in the forward pass for stability. `None` disables clipping.
            Default is `(0.1, 5.0)`.
        init_std: Standard deviation of the truncated normal initialization
            of all linear weights, as in the reference. Note that this
            also overrides the orthogonal initialization of the slice projections
            (as the reference does). `None` keeps the default
            initialization of each component. Default is 0.02.

    Raises:
        ValueError: If any of `in_channels`, `out_channels`, `ndim`, `hidden_channels`,
            `n_layers`, `ref` is not a positive integer.
        ValueError: If `positional_embedding` is invalid, is passed with
            `use_mesh=True` or together with `unified_pos=True`,
            or does not match `ndim`.
        ValueError: If `init_std` is not positive.

    ??? info "Internal Attributes"
        These fields store the internal layers state (and weights).

        * **positional_embedding** (`GridEmbeddingNd | None`): Grid positional embedding
            appended to the raw input signal.
        * **reference_embedding** (`ReferenceGridEmbeddingNd | None`): Unified position embedding
            appended to the raw input signal.
        * **lifting** (`PointwiseMLP`): Lifts the input (and positional features) to `hidden_channels`.
        * **placeholder** (`Float[Array, "c"]`): Learnable vector added to every lifted point.
        * **blocks** (`tuple[TransolverBlock, ...]`): The Transolver blocks.
        * **final_norm** (`ChannelLayerNorm`): Layer norm before the projection.
        * **projection** (`PointwiseMLP`): Linear map from `hidden_channels` to `out_channels`.
        * **ndim** (`int`): Number of spatial dimensions.
        * **use_mesh** (`bool`): Whether the model operates on unstructured meshes.

    ??? cite

        [Transolver: A Fast Transformer Solver for PDEs on General Geometries](
        https://arxiv.org/abs/2402.02366)

        ```bibtex
        @inproceedings{wu2024transolver,
            title={Transolver: A Fast Transformer Solver for PDEs on General Geometries},
            author={Haixu Wu and Huakun Luo and Haowen Wang and
            Jianmin Wang and Mingsheng Long},
            booktitle={International Conference on Machine Learning},
            year={2024}
        }
        ```

    Examples:
        ```python
        import jax.numpy as jnp
        import jax.random as jr
        from neojax.models import Transolver

        key = jr.key(0)

        # unstructured mesh with 500 nodes in 2D
        model = Transolver(
            key, in_channels=1, out_channels=1, ndim=2,
            hidden_channels=64, n_layers=4, use_mesh=True,
        )
        u, x_in = jnp.ones((1, 500)), jnp.ones((500, 2))
        out = model(u, x_in=x_in)  # (1, 500)

        # structured 2D grid with appended grid coordinates
        model = Transolver(
            key, in_channels=1, out_channels=1, ndim=2,
            hidden_channels=64, n_layers=4, positional_embedding="grid",
        )
        out = model(jnp.ones((1, 32, 32)))  # (1, 32, 32)
        ```

    !!! warning "No time embedding supported yet"
        **neojax** doesn't support the time embedding of `Transolver` yet.
    """

    positional_embedding: GridEmbeddingNd | None
    reference_embedding: ReferenceGridEmbeddingNd | None
    lifting: PointwiseMLP
    placeholder: Float[Array, "c"]
    blocks: tuple[TransolverBlock, ...]
    final_norm: ChannelLayerNorm
    projection: PointwiseMLP
    ndim: int = eqx.field(static=True)
    use_mesh: bool = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        in_channels: int,
        out_channels: int,
        ndim: int,
        hidden_channels: int = 256,
        n_layers: int = 5,
        n_heads: int = 8,
        slice_num: int = 32,
        mlp_ratio: float = 1.0,
        dropout: float = 0.0,
        activation: Callable = jax.nn.gelu,
        use_mesh: bool = False,
        kernel_size: int = 3,
        lift_channel_ratio: float = 2.0,
        positional_embedding: GridEmbeddingNd | Literal["grid"] | None = None,
        unified_pos: bool = False,
        ref: int = 8,
        temperature_range: tuple[float, float] | None = (0.1, 5.0),
        init_std: float | None = 0.02,
    ) -> None:
        for name, value in (
            ("in_channels", in_channels),
            ("out_channels", out_channels),
            ("ndim", ndim),
            ("hidden_channels", hidden_channels),
            ("n_layers", n_layers),
            ("ref", ref),
        ):
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if init_std is not None and init_std <= 0:
            raise ValueError("init_std must be positive or None.")

        if positional_embedding is not None and use_mesh:
            raise ValueError(
                "`positional_embedding` is not supported for `use_mesh=True`, "
                "node coordinates are always appended to the input."
            )
        if positional_embedding is not None and unified_pos:
            raise ValueError(
                "`positional_embedding` and `unified_pos` can't be combined."
            )

        self.ndim = ndim
        self.use_mesh = use_mesh

        if positional_embedding is None:
            self.positional_embedding = None
        elif isinstance(positional_embedding, str) and positional_embedding == "grid":
            self.positional_embedding = GridEmbeddingNd(in_channels, ndim)
        elif isinstance(positional_embedding, GridEmbeddingNd):
            if positional_embedding.n_extra_channels != ndim:
                raise ValueError(
                    f"`positional_embedding` has {positional_embedding.n_extra_channels} "
                    f"dimensions but `ndim` is {ndim}."
                )
            self.positional_embedding = positional_embedding
        else:
            raise ValueError(f"Positional embedding {positional_embedding} invalid.")

        self.reference_embedding = (
            ReferenceGridEmbeddingNd(ndim, ref) if unified_pos else None
        )

        lift_in_channels = in_channels
        if self.reference_embedding is not None:
            lift_in_channels += self.reference_embedding.n_features
        elif self.positional_embedding is not None:
            lift_in_channels += self.positional_embedding.n_extra_channels
        elif use_mesh:
            lift_in_channels += ndim

        k_lift, k_blocks, k_proj, k_place, k_init = jr.split(key, 5)
        lifting = PointwiseMLP(
            key=k_lift,
            layers=(
                lift_in_channels,
                int(hidden_channels * lift_channel_ratio),
                hidden_channels,
            ),
            activations=activation,
        )
        blocks = tuple(
            TransolverBlock(
                key=k,
                hidden_channels=hidden_channels,
                n_heads=n_heads,
                slice_num=slice_num,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                activation=activation,
                ndim=None if use_mesh else ndim,
                kernel_size=kernel_size,
                temperature_range=temperature_range,
            )
            for k in jr.split(k_blocks, n_layers)
        )
        projection = PointwiseMLP(key=k_proj, layers=(hidden_channels, out_channels))

        if init_std is not None:
            lifting, blocks, projection = _trunc_normal_init(
                k_init, (lifting, blocks, projection), init_std
            )

        self.lifting = lifting
        self.blocks = blocks
        self.projection = projection
        self.final_norm = ChannelLayerNorm(hidden_channels)
        self.placeholder = (1.0 / hidden_channels) * jr.uniform(
            k_place, (hidden_channels,)
        )

    @property
    def n_extra_channels(self) -> int:
        """Number of channels (coordinate features) appended to the input signal."""
        if self.reference_embedding is not None:
            return self.reference_embedding.n_features
        if self.positional_embedding is not None:
            return self.positional_embedding.n_extra_channels
        return self.ndim if self.use_mesh else 0

    @override
    def __call__(
        self,
        u: Inexact[Array, "in_c ..."],
        *,
        x_in: Float[Array, "N d"] | None = None,
        key: PRNGKeyArray | None = None,
        inference: bool = False,
    ) -> Inexact[Array, "out_c ..."]:
        """Forward pass of the Transolver model.

        Args:
            u: Input signal. Shape is `(in_channels, N)` if `use_mesh` is `True`
                or `(in_channels, d1, ..., d_ndim)` otherwise.
            x_in: Coordinates of the `N` nodes of `u` with shape `(N, ndim)`.
                Required if `use_mesh` is `True`, must be `None` otherwise.
            key: PRNG key used for dropout masks.
            inference: If `True`, dropout is disabled.

        Returns:
            The predicted field of shape `(out_channels, N)` if `use_mesh` is `True`
            or `(out_channels, d1, ..., d_ndim)` otherwise.

        Raises:
            ValueError: If `use_mesh` is `True` and `x_in` is `None`
                or if `u` and `x_in` have mismatching shapes.
            ValueError: If `use_mesh` is `False` and `x_in` is passed
                or `u` doesn't have `ndim` spatial axes.
        """
        if self.use_mesh:
            if x_in is None:
                raise ValueError("`x_in` is required if `use_mesh` is True.")
            if u.ndim != 2 or x_in.shape != (u.shape[1], self.ndim):
                raise ValueError(
                    f"Expected `u` shaped (in_channels, N) and `x_in` shaped (N, {self.ndim}), "
                    f"but got shapes {u.shape} and {x_in.shape}."
                )
            coords = x_in.T
        else:
            if x_in is not None:
                raise ValueError("`x_in` can only be passed if `use_mesh` is True.")
            if u.ndim != self.ndim + 1:
                raise ValueError(
                    f"Expected `u` with {self.ndim} spatial axes, but got shape {u.shape}."
                )
            coords = None

        if self.reference_embedding is not None:
            if coords is None:
                coords = self.reference_embedding.regular_grid(u.shape[1:])
            u = jnp.concatenate([u, self.reference_embedding(coords)], axis=0)
        elif coords is not None:
            u = jnp.concatenate([u, coords.astype(u.dtype)], axis=0)
        elif self.positional_embedding is not None:
            u = self.positional_embedding(u)

        h = self.lifting(u)
        h = h + self.placeholder.reshape(-1, *([1] * (h.ndim - 1)))

        if key is not None and not inference:
            keys = jr.split(key, len(self.blocks))
        else:
            keys = [None] * len(self.blocks)
        for block, block_key in zip(self.blocks, keys, strict=True):
            h = block(h, key=block_key, inference=inference)

        return self.projection(self.final_norm(h))
