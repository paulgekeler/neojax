import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import pytest

from neojax.nn.cno_blocks import CNOConvBlock, CNOLiftProjectBlock, CNOResidualBlock

RATIOS = {"invariant": 1, "upsample": 2, "downsample": 0.5}


def conv_block(**kwargs):
    """Stateless `CNOConvBlock` (without batch normalization)."""
    return CNOConvBlock(**({"batch_norm": False} | kwargs))


def residual_block(**kwargs):
    """Stateless `CNOResidualBlock` (without batch normalization)."""
    return CNOResidualBlock(**({"batch_norm": False} | kwargs))


def make_conv_block(**kwargs):
    """`CNOConvBlock` with the default (stateful) batch normalization."""
    kwargs = {"in_channels": 2, "out_channels": 4, "ndim": 2} | kwargs
    return eqx.nn.make_with_state(CNOConvBlock)(key=jr.key(0), **kwargs)


def make_residual_block(**kwargs):
    """`CNOResidualBlock` with the default (stateful) batch normalization."""
    kwargs = {"channels": 4, "ndim": 2} | kwargs
    return eqx.nn.make_with_state(CNOResidualBlock)(key=jr.key(0), **kwargs)


def batched(layer, **kwargs):
    """Vmap a stateful layer over a batch, with unmapped state."""
    return jax.vmap(
        lambda x, state: layer(x, state, **kwargs),
        axis_name="batch",
        in_axes=(0, None),
        out_axes=(0, None),
    )


def state_changed(state, other):
    return any(
        not np.array_equal(a, b)
        for a, b in zip(jtu.tree_leaves(state), jtu.tree_leaves(other), strict=True)
    )


class TestCNOConvBlock:
    @pytest.mark.parametrize("block_type", ["invariant", "upsample", "downsample"])
    def test_shapes(self, block_type):
        block, state = make_conv_block(block_type=block_type)
        assert block.is_stateful()
        assert isinstance(block.batch_norm, eqx.nn.BatchNorm)
        assert block.batch_norm.mode == "batch"
        out, new_state = batched(block)(jr.normal(jr.key(1), (4, 2, 8, 8)), state)
        assert out.shape == (4, 4, *(int(8 * RATIOS[block_type]),) * 2)
        assert isinstance(new_state, eqx.nn.State)

    def test_normalizes_over_batch_and_space(self):
        # Output of the normalization (before the activation) has zero mean and
        # unit variance per channel over batch and space
        block, state = make_conv_block()
        x = jr.normal(jr.key(1), (6, 2, 8, 8)) * 3.0 + 1.0

        def normalized(xi, state):
            out, _ = block.batch_norm(block.convolution(xi), state)
            return out

        out = jax.vmap(normalized, axis_name="batch", in_axes=(0, None), out_axes=0)(
            x, state
        )
        np.testing.assert_allclose(out.mean(axis=(0, 2, 3)), 0.0, atol=1e-4)
        np.testing.assert_allclose(out.var(axis=(0, 2, 3)), 1.0, rtol=1e-3)

    def test_custom_axis_name_and_momentum(self):
        block, state = make_conv_block(
            batch_axis_name="samples", batch_norm_momentum=0.5
        )
        assert block.batch_norm.axis_name == "samples"
        assert block.batch_norm.momentum == 0.5
        out, _ = jax.vmap(
            block, axis_name="samples", in_axes=(0, None), out_axes=(0, None)
        )(jr.normal(jr.key(1), (2, 2, 8, 8)), state)
        assert out.shape == (2, 4, 8, 8)

    def test_invalid_arguments_raise(self):
        args = {"key": jr.key(0), "in_channels": 2, "out_channels": 4, "ndim": 2}
        with pytest.raises((ValueError, TypeError, KeyError)):
            conv_block(**args, block_type="invalid")
        with pytest.raises(ValueError):  # even kernel
            conv_block(**args, conv_kernel=2)

    def test_wrap_is_periodic_equivariant(self):
        block = conv_block(
            key=jr.key(0),
            in_channels=2,
            out_channels=3,
            ndim=1,
            block_type="downsample",
            pad_mode="wrap",
        )
        x = jr.normal(jr.key(1), (2, 16))
        np.testing.assert_allclose(
            block(jnp.roll(x, 2, axis=1)),
            jnp.roll(block(x), 1, axis=1),
            rtol=1e-4,
            atol=1e-4,
        )


@pytest.mark.parametrize("kind", ["conv", "residual"])
def test_state_contract(kind):
    if kind == "conv":
        block, state = make_conv_block()
        x = jr.normal(jr.key(1), (4, 2, 8, 8))
    else:
        block, state = make_residual_block()
        x = jr.normal(jr.key(1), (4, 4, 8, 8))
    assert block.is_stateful()
    with pytest.raises(ValueError):
        block(x[0])

    # training updates the state, inference does not
    out, trained = batched(block)(x, state)
    assert out.shape[0] == 4
    assert state_changed(state, trained)
    _, inferred = batched(block, inference=True)(x, trained)
    assert not state_changed(trained, inferred)

    # `inference=None` falls back to the `BatchNorm` attribute (`eqx.nn.inference_mode`)
    _, inferred = batched(eqx.nn.inference_mode(block))(x, trained)
    assert not state_changed(trained, inferred)


class TestStatelessBlocks:
    def test_stateless_blocks(self):
        conv = conv_block(key=jr.key(0), in_channels=2, out_channels=4, ndim=2)
        res = residual_block(key=jr.key(0), channels=4, ndim=2)
        assert conv.batch_norm is None and not conv.is_stateful()
        assert not res.is_stateful()
        assert conv(jnp.ones((2, 8, 8))).shape == (4, 8, 8)
        assert res(jnp.ones((4, 8, 8))).shape == (4, 8, 8)

        _, state = make_conv_block()
        with pytest.raises(ValueError):
            conv(jnp.ones((2, 8, 8)), state)
        with pytest.raises(ValueError):
            res(jnp.ones((4, 8, 8)), state)

    def test_residual_identity_with_zeroed_convolution(self):
        block = residual_block(key=jr.key(0), channels=4, ndim=2)
        zero = lambda leaf: jnp.zeros_like(leaf) if eqx.is_array(leaf) else leaf
        block = eqx.tree_at(
            lambda b: b.convolution, block, jtu.tree_map(zero, block.convolution)
        )
        x = jr.normal(jr.key(1), (4, 8, 8))
        np.testing.assert_allclose(block(x), x, rtol=1e-6, atol=1e-6)


class TestCNOLiftProjectBlock:
    @pytest.mark.parametrize("ndim", [1, 3])
    def test_shapes_and_no_batch_norm(self, ndim):
        # as in the reference, lifting and projection are not normalized
        block = CNOLiftProjectBlock(
            key=jr.key(0), in_channels=3, out_channels=5, ndim=ndim, latent_channels=8
        )
        assert block.block.batch_norm is None
        assert not block.block.is_stateful()
        assert block(jnp.ones((3, *([8] * ndim)))).shape == (5, *([8] * ndim))


def test_in_equinox_sequential():
    # Stateful and stateless blocks compose with `eqx.nn.Sequential`
    def make():
        return eqx.nn.Sequential(
            [
                CNOConvBlock(
                    key=jr.key(0),
                    in_channels=2,
                    out_channels=4,
                    ndim=2,
                    block_type="downsample",
                ),
                CNOResidualBlock(key=jr.key(1), channels=4, ndim=2),
                conv_block(key=jr.key(2), in_channels=4, out_channels=4, ndim=2),
            ]
        )

    model, state = eqx.nn.make_with_state(make)()
    out, state = jax.vmap(
        model, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
    )(jr.normal(jr.key(3), (4, 2, 8, 8)), state)
    assert out.shape == (4, 4, 4, 4)
