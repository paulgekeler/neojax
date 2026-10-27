import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jaxtyping import TypeCheckError

from neojax.models.baseno import BaseNO
from neojax.models.cno import CNO
from neojax.models.rno import RNO
from neojax.nn.positional_embedding import GridEmbeddingNd
from neojax.tests.conftest import assert_filter_jittable

CNO_ARGS = {
    "in_channels": 3,
    "out_channels": 1,
    "ndim": 2,
    "n_layers": 2,
    "n_res": 1,
    "n_res_neck": 1,
    "channel_multiplier": 4,
    "latent_lift_proj_channels": 8,
}


def make_cno(**kwargs):
    """CNO with the default (stateful) batch normalization and its state."""
    return eqx.nn.make_with_state(CNO)(key=jr.key(0), **(CNO_ARGS | kwargs))


def make_stateless_cno(**kwargs):
    return CNO(key=jr.key(0), **(CNO_ARGS | {"batch_norm": False} | kwargs))


def batched(model, **kwargs):
    """Vmap a stateful model over a batch, with unmapped state."""
    return jax.vmap(
        lambda x, state: model(x, state, **kwargs),
        axis_name="batch",
        in_axes=(0, None),
        out_axes=(0, None),
    )


def state_changed(state, other):
    return any(
        not np.array_equal(a, b)
        for a, b in zip(
            jax.tree_util.tree_leaves(state),
            jax.tree_util.tree_leaves(other),
            strict=True,
        )
    )


class TestCNO:
    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_shapes(self, ndim):
        model, state = make_cno(ndim=ndim, out_channels=2)
        x = jr.normal(jr.key(1), (4, 3, *([8] * ndim)))
        out, state = batched(model)(x, state)
        assert out.shape == (4, 2, *([8] * ndim))
        assert isinstance(state, eqx.nn.State)
        assert jnp.all(jnp.isfinite(out))

    def test_state_contract(self):
        model, state = make_cno()
        x = jnp.ones((3, 8, 8))
        assert isinstance(model, BaseNO)
        # batch normalization is the default; lifting and projection have none
        assert model.is_stateful()
        assert isinstance(model.encoder[0].batch_norm, eqx.nn.BatchNorm)
        assert model.lift.block.batch_norm is None
        assert model.project.block.batch_norm is None
        with pytest.raises(ValueError):
            model(x)
        with pytest.raises(NameError):
            model(x, state)

    def test_architecture(self):
        model, state = make_cno(n_layers=3, n_res=3, n_res_neck=4, channel_multiplier=8)
        assert len(model.encoder) == len(model.decoder) == len(model.res_nets) == 3
        assert [len(r) for r in model.res_nets] == [3, 3, 3]
        assert len(model.res_net_neck) == 4
        assert len(model.decoder_inv) == 3
        assert len(model.ed_expansion) == 4
        # encoder features: [4, 8, 16, 32]
        assert model.lift.convolution.in_channels == 8  # latent
        assert model.lift.convolution.out_channels == 4
        assert [b.convolution.out_channels for b in model.encoder] == [8, 16, 32]
        assert [b.convolution.out_channels for b in model.decoder] == [16, 8, 4]
        # decoder inputs: bottleneck features, then concatenated skips
        assert [b.convolution.in_channels for b in model.decoder] == [32, 32, 16]
        assert [b.convolution.in_channels for b in model.decoder_inv] == [32, 32, 16]
        assert model.project.block.convolution.in_channels == 4 + 4

        model, state = make_cno(add_inv=False)
        assert model.decoder_inv is None
        out, _ = batched(model)(jnp.ones((2, 3, 8, 8)), state)
        assert out.shape == (2, 1, 8, 8)

    def test_resolutions_and_invalid_arguments(self):
        model, state = make_cno()
        for spatial in [(8, 8), (16, 16), (8, 16)]:
            out, _ = batched(model)(jnp.ones((2, 3, *spatial)), state)
            assert out.shape == (2, 1, *spatial)
        with pytest.raises(ValueError):  # not divisible by 2 ** n_layers
            batched(model)(jnp.ones((2, 3, 10, 16)), state)
        with pytest.raises(ValueError):  # wrong number of spatial dimensions
            batched(model)(jnp.ones((2, 3, 16)), state)
        with pytest.raises(ValueError):
            make_cno(n_layers=0)

    def test_training_updates_state_inference_does_not(self):
        model, state = make_cno()
        x = jr.normal(jr.key(1), (4, 3, 8, 8))
        _, new_state = batched(model)(x, state)
        assert state_changed(state, new_state)
        # every batch norm saw one batch
        counters = [
            leaf for leaf in jax.tree_util.tree_leaves(new_state) if leaf.ndim == 0
        ]
        assert counters and all(c == 1 for c in counters)

        _, inf_state = batched(model, inference=True)(x, new_state)
        assert not state_changed(new_state, inf_state)

    def test_batch_statistics_in_training_and_inference(self):
        model, state = make_cno()
        x = jr.normal(jr.key(1), (4, 3, 8, 8))
        x_other = x.at[1:].set(jr.normal(jr.key(2), (3, 3, 8, 8)) * 3.0)

        # training: other samples in the batch change the output of a sample
        out, state = batched(model)(x, state)
        out_other, _ = batched(model)(x_other, state)
        assert not np.allclose(out[0], out_other[0], atol=1e-3)

        # inference: per-sample and independent of the rest of the batch
        out, _ = batched(model, inference=True)(x, state)
        single, _ = model(x[0], state, inference=True)
        np.testing.assert_allclose(out[0], single, rtol=1e-4, atol=1e-4)
        out_other, _ = batched(model, inference=True)(x_other, state)
        np.testing.assert_allclose(out[0], out_other[0], rtol=1e-4, atol=1e-4)

    def test_gradients_and_jit(self):
        model, state = make_cno()
        x = jr.normal(jr.key(1), (4, 3, 8, 8))

        @eqx.filter_jit
        @eqx.filter_value_and_grad(has_aux=True)
        def loss_fn(model, state):
            out, state = batched(model)(x, state)
            return jnp.mean(out**2), state

        (loss, _), grads = loss_fn(model, state)
        assert jnp.isfinite(loss)
        leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))
        assert all(jnp.all(jnp.isfinite(g)) for g in leaves)
        assert any(jnp.any(g != 0) for g in leaves)
        assert jnp.any(grads.encoder[0].batch_norm.weight != 0)

    def test_wrap_is_periodic_equivariant(self):
        # Shifts by multiples of 2 ** n_layers commute with the (periodic) model
        model, state = make_cno(ndim=1, pad_mode="wrap")
        x = jr.normal(jr.key(1), (2, 3, 16))
        shifted, _ = batched(model)(jnp.roll(x, 4, axis=2), state)
        out, _ = batched(model)(x, state)
        np.testing.assert_allclose(
            shifted, jnp.roll(out, 4, axis=2), rtol=1e-3, atol=1e-3
        )

    def test_positional_embedding(self):
        model, state = make_cno(positional_embedding="grid")
        # lifting layer should consider grid channels
        assert model.lift.block.convolution.in_channels == 3 + 2
        out, _ = batched(model)(jnp.ones((2, 3, 8, 8)), state)
        assert out.shape == (2, 1, 8, 8)

        embedding = GridEmbeddingNd(
            in_channels=3, ndim=2, grid_boundaries=((0, 2),) * 2
        )
        model, state = make_cno(positional_embedding=embedding)
        assert isinstance(model.positional_embedding, GridEmbeddingNd)

        with pytest.raises((ValueError, TypeCheckError)):
            make_cno(positional_embedding="invalid")

    def test_save_load_with_state(self, tmp_path):
        model, state = make_cno()
        x = jr.normal(jr.key(1), (4, 3, 8, 8))
        _, state = batched(model)(x, state)
        path = tmp_path / "cno_with_state.eqx"
        eqx.tree_serialise_leaves(path, (model, state))
        loaded_model, loaded_state = eqx.tree_deserialise_leaves(path, make_cno())
        out, _ = batched(model, inference=True)(x, state)
        loaded_out, _ = batched(loaded_model, inference=True)(x, loaded_state)
        np.testing.assert_allclose(loaded_out, out, rtol=1e-6, atol=1e-6)

    def test_stateless_variant(self, tmp_path):
        # without batch normalization the model is called as `out = model(x)`
        model = make_stateless_cno()
        x = jr.normal(jr.key(1), (3, 8, 8))
        assert not model.is_stateful()
        assert model(x).shape == (1, 8, 8)
        assert_filter_jittable(model, x)
        with pytest.raises(ValueError):
            model(x, make_cno()[1])

        out = jax.vmap(model)(jnp.stack([x, x * 2]))
        np.testing.assert_allclose(out[0], model(x), rtol=1e-4, atol=1e-4)
        grads = eqx.filter_grad(lambda m: jnp.sum(m(x) ** 2))(model)
        leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))
        assert all(jnp.all(jnp.isfinite(g)) for g in leaves)
        assert any(jnp.any(g != 0) for g in leaves)

        hyperparams = CNO_ARGS | {"batch_norm": False}
        path = tmp_path / "cno.eqx"
        model.save(file=path, hyperparams=hyperparams)
        loaded = CNO.load(file=path, make_fn=lambda **hp: CNO(key=jr.key(1), **hp))
        np.testing.assert_allclose(loaded(x), model(x), rtol=1e-6, atol=1e-6)

        cno = make_stateless_cno(in_channels=1, out_channels=1, ndim=1)
        rno = RNO(model=cno, time_interval=(0, 3), n_timesteps=3)
        out = rno(x=jnp.ones((1, 16)), key=jr.key(2), inference=False)
        assert out.shape == (3, 1, 16)
