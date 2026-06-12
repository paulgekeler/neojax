import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import TypeCheckError

from neojax.nn.tfno_blocks import TFNOBlock, TFNOBlocks


class TestTFNOBlock:
    def test_dimensions(self):
        key = jr.key(0)
        in_c, out_c = 3, 5

        # 2D Linear skip
        block_2d = TFNOBlock(key, in_c, out_c, (4, 4), ranks=2, local_operator="linear")
        assert block_2d(jnp.ones((in_c, 16, 16))).shape == (out_c, 16, 16)

        # 3D Soft-gating skip
        block_3d = TFNOBlock(
            key, 4, 4, (4, 4, 4), ranks=2, local_operator="soft-gating"
        )
        assert block_3d(jnp.ones((4, 8, 8, 8))).shape == (4, 8, 8, 8)

        # 4D Linear skip
        block_4d = TFNOBlock(key, 4, 4, (4, 4, 4, 4), ranks=2, local_operator="linear")
        assert block_4d(jnp.ones((4, 6, 6, 6, 6))).shape == (4, 6, 6, 6, 6)

    def test_invalid_skip(self):
        key = jr.key(0)
        with pytest.raises((ValueError, TypeCheckError)):
            TFNOBlock(key, 2, 4, (4, 4), ranks=(2, 2, 2, 2), local_operator="invalid")

    def test_normalization(self):
        key = jr.key(0)
        in_c, out_c = 4, 4
        x = jnp.ones((in_c, 8, 8))

        for norm in ["layer", "instance", "group"]:
            block = TFNOBlock(
                key, in_c, out_c, (4, 4), ranks=(2, 2, 2, 2), normalization=norm
            )
            out = block(x)
            assert out.shape == x.shape

    def test_residual(self):
        key = jr.key(0)
        in_c, out_c = 4, 4
        x = jnp.ones((in_c, 8, 8))
        block = TFNOBlock(
            key, in_c, out_c, (4, 4), ranks=(2, 2, 2, 2), use_fno_residual=True
        )
        out = block(x)
        assert out.shape == x.shape

    def test_fno_blocks_residuals(self):
        key = jr.key(0)
        in_c, out_c = 4, 4
        x = jnp.ones((in_c, 8, 8))

        # Test with identity residual around MLP
        blocks = TFNOBlocks(
            key=key,
            n_layers=2,
            in_channels=in_c,
            out_channels=out_c,
            modes=(4, 4),
            ranks=(2, 2, 2, 2),
            use_channel_mlp=True,
            channel_mlp_residual="identity",
        )
        out = blocks(x)
        assert out.shape == x.shape
        assert blocks.channel_mlp_residuals is not None
        assert isinstance(blocks.channel_mlp_residuals[0], eqx.nn.Identity)

        # Test with no residual around MLP
        blocks_no_res = TFNOBlocks(
            key=key,
            n_layers=2,
            in_channels=in_c,
            out_channels=out_c,
            modes=(4, 4),
            ranks=(2, 2, 2, 2),
            use_channel_mlp=True,
            channel_mlp_residual=None,
        )
        out = blocks_no_res(x)
        assert out.shape == x.shape
        assert blocks_no_res.channel_mlp_residuals[0] is None


class TestTFNOBlocks:
    def test_dimensions(self):
        key = jr.key(0)
        in_c, out_c = 3, 5
        blocks = TFNOBlocks(
            key=key,
            n_layers=2,
            in_channels=in_c,
            out_channels=out_c,
            modes=(4, 4),
            ranks=(2, 2, 2, 2),
            use_channel_mlp=True,
        )
        assert blocks(jnp.ones((in_c, 16, 16))).shape == (out_c, 16, 16)
