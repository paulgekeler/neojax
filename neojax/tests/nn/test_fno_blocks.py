import jax.numpy as jnp
import jax.random as jr
import pytest

from neojax.nn.fno_blocks import FNOBlock, FNOBlocks


class TestFNOBlock:
    def test_dimensions(self):
        key = jr.PRNGKey(0)
        in_c, out_c = 3, 5

        # 2D Linear skip
        block_2d = FNOBlock(key, in_c, out_c, (4, 4), fno_skip="linear")
        assert block_2d(jnp.ones((in_c, 16, 16))).shape == (out_c, 16, 16)

        # 3D Soft-gating skip
        block_3d = FNOBlock(key, 4, 4, (4, 4, 4), fno_skip="soft-gating")
        assert block_3d(jnp.ones((4, 8, 8, 8))).shape == (4, 8, 8, 8)

        # 4D Linear skip
        block_4d = FNOBlock(key, 4, 4, (4, 4, 4, 4), fno_skip="linear")
        assert block_4d(jnp.ones((4, 6, 6, 6, 6))).shape == (4, 6, 6, 6, 6)

    def test_invalid_skip(self):
        key = jr.PRNGKey(0)
        with pytest.raises(ValueError):
            FNOBlock(key, 2, 4, (4, 4), fno_skip="soft-gating")
        with pytest.raises(ValueError):
            FNOBlock(key, 2, 4, (4, 4), fno_skip="identity")


class TestFNOBlocks:
    def test_dimensions(self):
        key = jr.PRNGKey(0)
        in_c, out_c = 3, 5
        blocks = FNOBlocks(
            key=key,
            n_layers=2,
            in_channels=in_c,
            out_channels=out_c,
            modes=(4, 4),
            use_channel_mlp=True,
        )
        assert blocks(jnp.ones((in_c, 16, 16))).shape == (out_c, 16, 16)
