import os
import tempfile

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest

from neojax.models.fno import FNO


@pytest.fixture
def model():
    key = jr.key(0)
    return FNO(
        key=key,
        in_channels=1,
        out_channels=1,
        hidden_channels=4,
        n_layers=1,
        modes=(4,),
    )


@pytest.mark.usefixtures("model")
class TestBaseNO:
    def test_save_load_weights(self, model):
        x = jnp.ones((1, 32))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "weights.eqx")
            with open(path, "wb") as f:
                model.save_weights(f)

            skeleton = model
            with open(path, "rb") as f:
                loaded = skeleton.load_weights(f)

        assert jnp.allclose(model(x), loaded(x))

    def test_save_load_weights_file_path(self, model):
        x = jnp.ones((1, 32))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "weights.eqx")
            model.save_weights(path)

            skeleton = model
            loaded = skeleton.load_weights(path)

        assert jnp.allclose(model(x), loaded(x))

    def test_save_load(self, model):
        key = jr.key(0)
        hyperparams = dict(
            in_channels=1,
            out_channels=1,
            hidden_channels=4,
            n_layers=1,
            modes=(4,),
        )

        x = jnp.ones((1, 32))

        def make_fn(**hp):
            return FNO(key=key, **hp)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.eqx")
            with open(path, "wb") as f:
                model.save(file=f, hyperparams=hyperparams)
            with open(path, "rb") as f:
                loaded = FNO.load(file=f, make_fn=make_fn)

        assert jnp.allclose(model(x), loaded(x))

    def test_save_load_file_path(self, model):
        key = jr.key(0)
        hyperparams = dict(
            in_channels=1,
            out_channels=1,
            hidden_channels=4,
            n_layers=1,
            modes=(4,),
        )

        x = jnp.ones((1, 32))

        def make_fn(**hp):
            return FNO(key=key, **hp)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.eqx")
            model.save(file=path, hyperparams=hyperparams)
            loaded = FNO.load(file=path, make_fn=make_fn)

        assert jnp.allclose(model(x), loaded(x))

    def test_size(self, model):
        size_mb = model.size()
        assert isinstance(size_mb, float)
        assert size_mb > 0

        size_mb_2, n_params = model.size(return_n_params=True)
        assert size_mb_2 == size_mb
        assert isinstance(n_params, int)
        assert n_params > 0

    def test_astype_no_op(self, model):
        cast = model.astype(jnp.float32)
        for leaf in jtu.tree_leaves(eqx.filter(cast, eqx.is_array)):
            assert leaf.dtype in (jnp.float32, jnp.complex64)

    def test_astype_preserves_real_complex_separation(self, model):
        # bfloat16 has no native complex counterpart, so complex params
        # should fall back to complex64 while real params become bfloat16.
        cast = model.astype(jnp.bfloat16)
        for leaf in jtu.tree_leaves(eqx.filter(cast, eqx.is_array)):
            if jnp.issubdtype(leaf.dtype, jnp.complexfloating):
                assert leaf.dtype == jnp.complex64
            else:
                assert leaf.dtype == jnp.bfloat16

    def test_astype_invalid_dtype(self, model):
        with pytest.raises(ValueError):
            model.astype(jnp.int32)

    def test_profile_compile_success(self, model):
        success, summary = model.profile_compile(jnp.ones((1, 32)))
        assert success is True
        assert summary["exception"] is None
        assert summary["lowering"] == ""
        assert summary["lowering_cost_analysis"] is not None
        assert summary["compiled_cost_analysis"] is not None

    def test_profile_compile_return_lowering(self, model):
        _, summary = model.profile_compile(jnp.ones((1, 32)), return_lowering=True)
        assert isinstance(summary["lowering"], str)
        assert len(summary["lowering"]) > 0

    def test_profile_compile_failure(self, model):
        # 2D spatial input for a model configured with 1D modes.
        bad_input = jnp.ones((1, 16, 16))

        success, summary = model.profile_compile(bad_input, filter_jax_frames=False)
        assert success is False
        assert summary["exception"] is not None
        assert summary["exception"].startswith("Lowering failed:")

        success, summary = model.profile_compile(bad_input, filter_jax_frames=True)
        assert success is False
        assert "jax" not in summary["exception"]
        assert "equinox" not in summary["exception"]
