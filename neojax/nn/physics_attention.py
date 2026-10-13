"""Implementation of the Physics-Attention layer of the Transolver."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Inexact, PRNGKeyArray

from neojax.nn.pointwise_mlp import PointwiseMLP


class PhysicsAttention(eqx.Module):
    r"""Physics-Attention layer of the Transolver.

    Physics-Attention adaptively groups the $N$ points of the discretized domain
    into $G$ learnable *slices* of points with similar physical states,
    runs multi-head self-attention among the $G$ slice tokens only and
    scatters the result back onto the points. Its cost is thus linear in $N$.
    Per head, it computes

    1. **Slice:** soft slice weights $w_{g,n} = \operatorname{softmax}_g
       \big(\mathrm{Linear}(x_n) / \tau\big)$ and the slice tokens
       $z_g = \sum_n w_{g,n} f_n / \sum_n w_{g,n}$, where $x_n, f_n$ are
       two linear (or convolutional) projections of the input and $\tau$ is a
       learnable temperature.
    2. **Attention:** $z' = \operatorname{softmax}(QK^\top / \sqrt{d_h}) V$ among the slice
       tokens, with $Q, K, V$ projections that are shared across heads.
    3. **Deslice:** $f'_n = \sum_g w_{g,n} z'_g$.

    The heads are concatenated and linearly projected back to `dim` channels.

    Args:
        key: PRNG key for parameter initialization.
        dim: Number of input and output channels.
        n_heads: Number of attention heads. Default is 8.
        dim_head: Channels per head. Default is `None`, which uses `dim // n_heads`.
        slice_num: Number of slices (slice tokens) `G`. Default is 32.
        dropout: Dropout probability applied to the attention probabilities
            and the output. Default is 0.0.
        ndim: Number of spatial dimensions of a structured grid input.
            If an integer, the input projections are `ndim`-dimensional
            convolutions of size `kernel_size`, so the input must be shaped
            `(dim, d1, ..., d_ndim)`. If `None`, they are pointwise linear
            maps and the input can have arbitrary trailing axes
            (e.g. `(dim, N)` for unstructured meshes). Default is `None`.
        kernel_size: Kernel size of the convolutional input projections.
            Only used if `ndim` is not `None`. Must be odd. Default is 3.
        temperature_range: Interval the learnable slice temperature is clipped to
            in the forward pass for stability. `None` disables clipping.
            Default is `(0.1, 5.0)`.

    Raises:
        ValueError: If `dim`, `n_heads`, `dim_head`, `slice_num` are not positive
            or `dim` is not divisible by `n_heads` and `dim_head` is `None`.
        ValueError: If `dropout` is not in the half-open interval [0, 1).
        ValueError: If `ndim` is not positive or `kernel_size` is not a positive odd number.
        ValueError: If `temperature_range` is not a valid positive interval.

    ??? info "Internal Attributes"
        These fields store the internal layers state (and weights).

        * **in_project_x** (`PointwiseMLP | eqx.nn.Conv`): Projection of the input to slice logits features.
        * **in_project_fx** (`PointwiseMLP | eqx.nn.Conv`): Projection of the input to the features to be sliced.
        * **in_project_slice** (`PointwiseMLP`): Maps per head features to the `slice_num` slice logits.
        * **to_q** (`eqx.nn.Linear`): Query projection shared across heads.
        * **to_k** (`eqx.nn.Linear`): Key projection shared across heads.
        * **to_v** (`eqx.nn.Linear`): Value projection shared across heads.
        * **to_out** (`PointwiseMLP`): Output projection from `n_heads * dim_head` to `dim` channels.
        * **dropout** (`eqx.nn.Dropout`): Dropout layer.
        * **temperature** (`Float[Array, "1 h 1"]`): Learnable slice temperature per head.
        * **dim** (`int`): Number of input and output channels.
        * **n_heads** (`int`): Number of attention heads.
        * **dim_head** (`int`): Channels per head.
        * **slice_num** (`int`): Number of slices.
        * **ndim** (`int | None`): Number of spatial dimensions of grid inputs or `None`.
        * **temperature_range** (`tuple[float, float] | None`): Clipping interval of the temperature.

    Examples:
        ```python
        import jax.numpy as jnp
        import jax.random as jr
        from neojax.nn import PhysicsAttention

        key = jr.key(0)

        # unstructured mesh with 500 nodes
        attn = PhysicsAttention(key, dim=64, n_heads=4, slice_num=16)
        out = attn(jnp.ones((64, 500)))

        # structured 2D grid
        attn = PhysicsAttention(key, dim=64, n_heads=4, slice_num=16, ndim=2)
        out = attn(jnp.ones((64, 32, 32)))
        ```
    """

    in_project_x: PointwiseMLP | eqx.nn.Conv
    in_project_fx: PointwiseMLP | eqx.nn.Conv
    in_project_slice: PointwiseMLP
    to_q: eqx.nn.Linear
    to_k: eqx.nn.Linear
    to_v: eqx.nn.Linear
    to_out: PointwiseMLP
    dropout: eqx.nn.Dropout
    temperature: Float[Array, "1 h 1"]
    dim: int = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)
    dim_head: int = eqx.field(static=True)
    slice_num: int = eqx.field(static=True)
    ndim: int | None = eqx.field(static=True)
    temperature_range: tuple[float, float] | None = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        dim: int,
        n_heads: int = 8,
        dim_head: int | None = None,
        slice_num: int = 32,
        dropout: float = 0.0,
        ndim: int | None = None,
        kernel_size: int = 3,
        temperature_range: tuple[float, float] | None = (0.1, 5.0),
    ) -> None:
        if dim <= 0:
            raise ValueError("`dim` must be a positive integer.")
        if n_heads <= 0:
            raise ValueError("`n_heads` must be a positive integer.")
        if slice_num <= 0:
            raise ValueError("`slice_num` must be a positive integer.")
        if dim_head is None:
            if dim % n_heads != 0:
                raise ValueError(
                    f"`dim` ({dim}) must be divisible by `n_heads` ({n_heads}) "
                    "if `dim_head` is not given."
                )
            dim_head = dim // n_heads
        elif dim_head <= 0:
            raise ValueError("`dim_head` must be a positive integer.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("`dropout` probability must be in [0.0, 1.0).")
        if ndim is not None:
            if ndim <= 0:
                raise ValueError("`ndim` must be a positive integer or None.")
            if kernel_size <= 0 or kernel_size % 2 == 0:
                raise ValueError("`kernel_size` must be a positive odd integer.")
        if temperature_range is not None:
            lo, hi = temperature_range
            if not (0.0 < lo <= hi):
                raise ValueError("`temperature_range` must satisfy 0 < low <= high.")
            temperature_range = (float(lo), float(hi))

        self.dim = dim
        self.n_heads = n_heads
        self.dim_head = dim_head
        self.slice_num = slice_num
        self.ndim = ndim
        self.temperature_range = temperature_range
        inner_dim = n_heads * dim_head

        k_x, k_fx, k_slice, k_q, k_k, k_v, k_out = jr.split(key, 7)

        if ndim is None:
            self.in_project_x = PointwiseMLP(key=k_x, layers=(dim, inner_dim))
            self.in_project_fx = PointwiseMLP(key=k_fx, layers=(dim, inner_dim))
        else:
            self.in_project_x = eqx.nn.Conv(
                ndim, dim, inner_dim, kernel_size, padding=kernel_size // 2, key=k_x
            )
            self.in_project_fx = eqx.nn.Conv(
                ndim, dim, inner_dim, kernel_size, padding=kernel_size // 2, key=k_fx
            )

        in_project_slice = PointwiseMLP(key=k_slice, layers=(dim_head, slice_num))
        # Principled (orthogonal) initialization of the slice projection
        self.in_project_slice = eqx.tree_at(
            lambda m: m.weights[0],
            in_project_slice,
            jax.nn.initializers.orthogonal()(k_slice, (slice_num, dim_head)),
        )

        self.to_q = eqx.nn.Linear(dim_head, dim_head, use_bias=False, key=k_q)
        self.to_k = eqx.nn.Linear(dim_head, dim_head, use_bias=False, key=k_k)
        self.to_v = eqx.nn.Linear(dim_head, dim_head, use_bias=False, key=k_v)
        self.to_out = PointwiseMLP(key=k_out, layers=(inner_dim, dim))
        self.dropout = eqx.nn.Dropout(dropout)
        self.temperature = jnp.ones((1, n_heads, 1)) * 0.5

    def _split_heads(self, x: Float[Array, "inner ..."]) -> Float[Array, "dh h n"]:
        """Splits the channel axis into heads (head-major, as in the reference).

        Args:
            x: Projected input logit features.

        Returns:
            Input split into number of heads with head dimension each.
        """
        x = x.reshape(self.n_heads, self.dim_head, -1)
        return jnp.moveaxis(x, 1, 0)

    def _apply_linear(
        self, linear: eqx.nn.Linear, x: Inexact[Array, "c ..."]
    ) -> Inexact[Array, "c_out ..."]:
        """Applies an unbatched `eqx.nn.Linear` along the leading axis at every location.

        Args:
            linear: The linear layer.
            x: Input array shaped (c, d1, ..., dN).

        Returns:
            Output array shaped (c_out, d1, ..., dN).
        """
        flat = x.reshape(x.shape[0], -1)
        out = jax.vmap(linear, in_axes=1, out_axes=1)(flat)
        return out.reshape(out.shape[0], *x.shape[1:])

    def __call__(
        self,
        x: Inexact[Array, "c ..."],
        *,
        key: PRNGKeyArray | None = None,
        inference: bool = False,
    ) -> Inexact[Array, "c ..."]:
        """Forward pass of the Physics-Attention layer.

        Args:
            x: Input array shaped (dim, N) or (dim, d1, ..., dN). If `ndim` is
                not `None`, it must be shaped (dim, d1, ..., d_ndim).
            key: PRNG key used for dropout masks.
            inference: If `True`, dropout is disabled.

        Returns:
            Output array with the same shape as `x`.

        Raises:
            ValueError: If the number of channels of `x` doesn't match `dim`.
            ValueError: If `ndim` is not `None` and `x` has not `ndim` spatial axes.
            ValueError: If dropout is active but no `key` is passed.
        """
        if x.shape[0] != self.dim:
            raise ValueError(
                f"Expected input with {self.dim} channels along axis 0, "
                f"but got shape {x.shape}."
            )
        if self.ndim is not None and x.ndim != self.ndim + 1:
            raise ValueError(
                f"Expected input with {self.ndim} spatial axes, but got shape {x.shape}."
            )
        use_dropout = self.dropout.p > 0.0 and not inference
        if use_dropout and key is None:
            raise ValueError(
                "A PRNG key must be passed to __call__ when dropout > 0 and inference = False."
            )
        if use_dropout:
            attn_key, out_key = jr.split(key)
        else:
            attn_key, out_key = None, None

        # Project and group the points into `slice_num` slice tokens.
        x_mid = self._split_heads(self.in_project_x(x))
        fx_mid = self._split_heads(self.in_project_fx(x))
        temperature = self.temperature
        if self.temperature_range is not None:
            temperature = jnp.clip(temperature, *self.temperature_range)
        logits = self.in_project_slice(x_mid) / temperature  # (g, h, n)
        slice_weights = jax.nn.softmax(logits, axis=0)  # (g, h, n)
        slice_norm = jnp.sum(slice_weights, axis=-1)  # (g, h)
        slice_token = jnp.einsum("chn,ghn->cgh", fx_mid, slice_weights)  # (dh, g, h)
        slice_token = slice_token / (slice_norm + 1e-5)[None]

        # Attention among the slice tokens (q, k, v shared across heads).
        q = self._apply_linear(self.to_q, slice_token)
        k = self._apply_linear(self.to_k, slice_token)
        v = self._apply_linear(self.to_v, slice_token)
        dots = jnp.einsum("cgh,cph->hgp", q, k) * self.dim_head**-0.5  # (h, g, g)
        attn = jax.nn.softmax(dots, axis=-1)
        attn = self.dropout(attn, key=attn_key, inference=not use_dropout)
        out_slice_token = jnp.einsum("hgp,cph->cgh", attn, v)  # (dh, g, h)

        # Scatter the slice tokens back onto the points.
        out_x = jnp.einsum("cgh,ghn->chn", out_slice_token, slice_weights)
        out_x = jnp.moveaxis(out_x, 0, 1).reshape(self.n_heads * self.dim_head, -1)
        out = self.to_out(out_x)
        out = self.dropout(out, key=out_key, inference=not use_dropout)
        return out.reshape(self.dim, *x.shape[1:])
