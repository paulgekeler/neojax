import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import TypeCheckError

from neojax.nn.mg_conv import MgConv
from neojax.tests.conftest import assert_filter_jittable


@pytest.fixture
def mgconv():
    k = jr.key(1)
    # Corresponds to Navier Stokes config
    return MgConv(
        k,
        n_its_per_level=[[1, 0], [1, 0], [1, 0], [2, 0], [2, 0]],
        n_channels_u=32,
        n_channels_f=1,
        padding_mode="circular",
        padding_mode_prolongation="zeros",
        use_layernorm=True,
        use_layernorm_affine=True,
        in_spatial_size=64,
        use_smoothing_bias=True,
    )


@pytest.mark.usefixtures("mgconv")
class TestMgConv:
    @pytest.mark.parametrize(
        "n_its_per_level", [[], [[1, 2, 3]], [[1, 1], [1, 1], [1, 2, 3]]]
    )
    def test_wrong_its_per_level_raises(self, n_its_per_level):
        with pytest.raises((ValueError, TypeError, TypeCheckError)):
            k = jr.key(0)
            MgConv(k, n_its_per_level, n_channels_u=2, n_channels_f=1)

    def test_use_layernorm_no_in_spatial_size_raises(self):
        with pytest.raises(ValueError):
            k = jr.key(0)
            MgConv(
                k,
                n_its_per_level=[[1, 0], [1, 0], [1, 0], [2, 0], [2, 0]],
                n_channels_u=32,
                n_channels_f=1,
                padding_mode="circular",
                padding_mode_prolongation="zeros",
                use_layernorm=True,
                use_layernorm_affine=True,
            )

    def test_helm_variant(self):
        """Tests MgConv_helm variant."""
        k = jr.key(0)
        mgconv_helm = MgConv(
            k,
            n_channels_u=20,
            n_channels_f=1,
            n_its_per_level=[1, 1, 1, 1, 2],
            prolongation_padding=0,
            restriction_padding=0,
            padding_mode_smoothing="reflect",
            prolongation_kernel_type="helm",
            use_xavier_restrict=True,
            grow_channels_w_depth=True,
            use_smoothing_bias=True,
            use_layernorm=False,
        )
        in_field = jr.normal(k, (1, 101, 101), dtype=jnp.float32)
        out = mgconv_helm(in_field)
        assert out.shape == (20,) + in_field.shape[1:]

    def test_helm2_variant(self):
        """Tests MgConv_helm2 variant."""
        k = jr.key(0)
        # Also test without xavier init
        mgconv_helm2 = MgConv(
            k,
            n_channels_u=20,
            n_channels_f=1,
            n_its_per_level=[1, 1, 1, 1, 2],
            prolongation_padding=0,
            restriction_padding=0,
            padding_mode_smoothing="reflect",
            prolongation_kernel_type="helm",
            layer_norm_type="helm",
            use_xavier_restrict=False,
            grow_channels_w_depth=True,
            use_smoothing_bias=True,
            use_layernorm=True,
            use_layernorm_affine=False,
            in_spatial_size=101,
        )
        in_field = jr.normal(k, (1, 101, 101), dtype=jnp.float32)
        out = mgconv_helm2(in_field)
        assert out.shape == (20,) + in_field.shape[1:]

    def test_helm3_variant(self):
        """Tests MgConv_helm3 variant."""
        k = jr.key(0)
        mgconv_helm3 = MgConv(
            k,
            n_channels_u=20,
            n_channels_f=1,
            n_its_per_level=[1, 1, 1, 1, 2],
            prolongation_padding=0,
            restriction_padding=0,
            padding_mode_smoothing="reflect",
            prolongation_kernel_type="helm",
            layer_norm_type="helm",
            use_xavier_restrict=False,
            grow_channels_w_depth=False,
            use_smoothing_bias=True,
            use_layernorm=True,
            use_layernorm_affine=False,
            in_spatial_size=101,
        )
        in_field = jr.normal(k, (1, 101, 101), dtype=jnp.float32)
        out = mgconv_helm3(in_field)
        assert out.shape == (20,) + in_field.shape[1:]

    def test_dc_variant(self):
        """Tests MgConv_DC variant."""
        k = jr.key(0)
        mgconv_dc = MgConv(
            k,
            n_its_per_level=[[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [2, 0]],
            n_channels_u=32,
            n_channels_f=1,
            use_layernorm=False,
            use_residual_restriction=True,
        )
        in_field = jr.normal(k, (1, 128, 128), dtype=jnp.float32)
        out = mgconv_dc(in_field)
        assert out.shape == (32,) + in_field.shape[1:]

    def test_dc_smooth_variant(self):
        """Tests MgConv_DC_smooth variant."""
        k = jr.key(0)
        mgconv_dcs = MgConv(
            k,
            n_its_per_level=[[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [2, 2]],
            n_channels_u=24,
            n_channels_f=1,
            prolongation_kernel_type="alternating",
            use_layernorm=False,
            use_residual_restriction=True,
        )
        in_field = jr.normal(k, (1, 211, 211), dtype=jnp.float32)
        out = mgconv_dcs(in_field)
        assert out.shape == (24,) + in_field.shape[1:]

    def test_ns_variant(self):
        """Tests MgConv as in MgNO_NS (Navier Stokes)."""
        k = jr.key(0)
        mgconv_ns = MgConv(
            k,
            n_its_per_level=[[1, 0], [1, 0], [1, 0], [2, 0], [2, 0]],
            n_channels_u=32,
            n_channels_f=1,
            padding_mode="circular",
            padding_mode_prolongation="zeros",
            use_layernorm=True,
            use_layernorm_affine=True,
            in_spatial_size=64,
            use_smoothing_bias=True,
        )
        in_field = jr.normal(k, (1, 64, 64), dtype=jnp.float32)
        out = mgconv_ns(in_field)
        assert out.shape == (32,) + in_field.shape[1:]

    def test_batched_grad(self, mgconv):
        import equinox as eqx

        k = jr.key(3)
        in_field = jr.normal(k, (2, 1, 64, 64), dtype=jnp.float32)

        def loss_fn(m):
            batch_out = jax.vmap(m)(in_field)
            return jnp.sum(batch_out)

        grad = eqx.filter_grad(loss_fn)(mgconv)
        grad_leaves = jax.tree_util.tree_leaves(eqx.filter(grad, eqx.is_array))
        assert grad_leaves
        assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in grad_leaves)

    def test_is_jittable(self, mgconv):
        k = jr.key(0)
        inputs = jr.normal(k, (1, 64, 64))
        assert_filter_jittable(mgconv, inputs)
