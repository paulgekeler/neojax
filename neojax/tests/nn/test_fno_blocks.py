import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import TypeCheckError

from neojax.nn.fno_blocks import FNOBlock, FNOBlocks


class TestFNOBlock:
    def test_dimensions(self):
        key = jr.key(0)
        in_c, out_c = 3, 5

        # 2D Linear skip
        block_2d = FNOBlock(key, in_c, out_c, (4, 4), local_operator="linear")
        assert block_2d(jnp.ones((in_c, 16, 16))).shape == (out_c, 16, 16)

        # 3D Soft-gating skip
        block_3d = FNOBlock(key, 4, 4, (4, 4, 4), local_operator="soft-gating")
        assert block_3d(jnp.ones((4, 8, 8, 8))).shape == (4, 8, 8, 8)

        # 4D Linear skip
        block_4d = FNOBlock(key, 4, 4, (4, 4, 4, 4), local_operator="linear")
        assert block_4d(jnp.ones((4, 6, 6, 6, 6))).shape == (4, 6, 6, 6, 6)

    def test_invalid_skip(self):
        key = jr.key(0)
        with pytest.raises((ValueError, TypeCheckError)):
            FNOBlock(key, 2, 4, (4, 4), local_operator="invalid")

    def test_normalization(self):
        key = jr.key(0)
        in_c, out_c = 4, 4
        x = jnp.ones((in_c, 8, 8))

        for norm in ["layer", "instance", "group"]:
            block = FNOBlock(key, in_c, out_c, (4, 4), normalization=norm)
            out = block(x)
            assert out.shape == x.shape

    def test_residual(self):
        key = jr.key(0)
        in_c, out_c = 4, 4
        x = jnp.ones((in_c, 8, 8))
        block = FNOBlock(key, in_c, out_c, (4, 4), use_fno_residual=True)
        out = block(x)
        assert out.shape == x.shape

    def test_parameters_propagation(self):
        key = jr.key(0)
        in_c, out_c = 3, 5
        block = FNOBlock(
            key=key,
            in_channels=in_c,
            out_channels=out_c,
            modes=(4, 4),
            enforce_hermitian_symmetry=False,
            fft_norm="ortho",
            is_complex_data=True,
        )
        assert block.spectral_conv.enforce_hermitian_symmetry is False
        assert block.spectral_conv.fft_norm == "ortho"
        assert block.spectral_conv.is_complex_data is True

    def test_fno_blocks_residuals(self):
        key = jr.key(0)
        in_c, out_c = 4, 4
        x = jnp.ones((in_c, 8, 8))

        # Test with identity residual around MLP
        blocks = FNOBlocks(
            key=key,
            n_layers=2,
            in_channels=in_c,
            out_channels=out_c,
            modes=(4, 4),
            use_channel_mlp=True,
            channel_mlp_residual="identity",
        )
        out = blocks(x)
        assert out.shape == x.shape
        assert blocks.channel_mlp_residuals is not None
        assert isinstance(blocks.channel_mlp_residuals[0], eqx.nn.Identity)

        # Test with no residual around MLP
        blocks_no_res = FNOBlocks(
            key=key,
            n_layers=2,
            in_channels=in_c,
            out_channels=out_c,
            modes=(4, 4),
            use_channel_mlp=True,
            channel_mlp_residual=None,
        )
        out = blocks_no_res(x)
        assert out.shape == x.shape
        assert blocks_no_res.channel_mlp_residuals[0] is None


class TestFNOBlocks:
    def test_dimensions(self):
        key = jr.key(0)
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

    def test_parameters_propagation(self):
        key = jr.key(0)
        in_c, out_c = 3, 5
        blocks = FNOBlocks(
            key=key,
            n_layers=2,
            in_channels=in_c,
            out_channels=out_c,
            modes=(4, 4),
            enforce_hermitian_symmetry=False,
            fft_norm="ortho",
            is_complex_data=True,
        )

        for fno_layer in blocks.fno_layers:
            assert fno_layer.spectral_conv.enforce_hermitian_symmetry is False
            assert fno_layer.spectral_conv.fft_norm == "ortho"
            assert fno_layer.spectral_conv.is_complex_data is True

        x = jr.normal(key, (in_c, 16, 16), dtype=jnp.complex64)
        assert blocks(x).shape == (out_c, 16, 16)

    def test_resolution_scaling(self):
        key = jr.key(0)
        in_c, out_c = 3, 5

        # Test scalar resolution scaling across layers (e.g. 1.5 twice: 16 -> 24 -> 36)
        blocks_scalar = FNOBlocks(
            key=key,
            n_layers=2,
            in_channels=in_c,
            out_channels=out_c,
            modes=(4, 4),
            resolution_scaling_factor=1.5,
        )
        assert blocks_scalar(jnp.ones((in_c, 16, 16))).shape == (out_c, 36, 36)

        # Test layerwise sequence of scaling factors (e.g. 2.0 then 0.5: 16 -> 32 -> 16)
        blocks_seq = FNOBlocks(
            key=key,
            n_layers=2,
            in_channels=in_c,
            out_channels=out_c,
            modes=(4, 4),
            resolution_scaling_factor=(2.0, 0.5),
        )
        assert blocks_seq(jnp.ones((in_c, 16, 16))).shape == (out_c, 16, 16)

