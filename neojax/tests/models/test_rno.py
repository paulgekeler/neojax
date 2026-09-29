import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import pytest

from neojax.models.baseno import BaseNO
from neojax.models.fno import FNO
from neojax.models.rno import RNO, concat_channels
from neojax.models.tfno import TFNO
from neojax.tests.conftest import assert_filter_jittable

N_TIMESTEPS = 3
GRID = 32


def make_fno(in_channels=1, out_channels=1, **kwargs):
    return FNO(
        key=jr.key(0),
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=8,
        n_layers=1,
        modes=(4,),
        **kwargs,
    )


class ScaledIdentity(BaseNO):
    # tiny model with an extra kwarg in call params (like GeoFNO)

    def __call__(self, x, *, scale, key=None, inference=False):
        return scale * x


@pytest.fixture
def rno_hyperparams():
    return {"time_interval": (0, 3), "n_timesteps": N_TIMESTEPS}


@pytest.fixture
def rno_inputs():
    key = jr.key(1)
    x = jr.normal(key, (1, GRID))
    return {"key": key, "inference": False, "x": x}


@pytest.fixture
def forcing():
    # two forcing channels per step
    return jr.normal(jr.key(2), (N_TIMESTEPS, 2, GRID))


@pytest.fixture
def rno_fno(rno_hyperparams):
    return RNO(model=make_fno(), **rno_hyperparams)


@pytest.fixture
def rno_fno_forced(rno_hyperparams):
    return RNO(model=make_fno(in_channels=3), **rno_hyperparams)


@pytest.fixture
def rno_fno_w_checkpoint(rno_hyperparams):
    return RNO(model=make_fno(), **rno_hyperparams, checkpoint_timesteps=True)


@pytest.fixture
def rno_tfno(rno_hyperparams):
    tfno = TFNO(
        key=jr.key(0),
        in_channels=1,
        out_channels=1,
        hidden_channels=8,
        n_layers=1,
        modes=(4,),
        ranks=(2, 2, 2),
    )
    return RNO(model=tfno, **rno_hyperparams)


class TestConcatChannels:
    def test_no_forcing_returns_state(self):
        x = jnp.ones((2, 5))
        assert concat_channels(x, None) is x

    def test_concatenates_all_leaves(self):
        x = jnp.ones((2, 5))
        f = {"a": jnp.zeros((1, 5)), "b": jnp.zeros((3, 5))}
        assert concat_channels(x, f).shape == (6, 5)


@pytest.mark.usefixtures("rno_fno", "rno_tfno", "rno_fno_w_checkpoint", "rno_inputs")
class TestRNO:
    def test_wrapped_fno(self, rno_fno, rno_inputs):
        out = rno_fno(**rno_inputs)
        assert out.shape == (N_TIMESTEPS, 1, GRID)

    def test_wrapped_tfno(self, rno_tfno, rno_inputs):
        out = rno_tfno(**rno_inputs)
        assert out.shape == (N_TIMESTEPS, 1, GRID)

    @pytest.mark.parametrize("time_interval", [(0, 0), (5, 2), (-1, -2), (-1, 2)])
    def test_invalid_time_interval_raises(self, time_interval, rno_fno):
        with pytest.raises(ValueError):
            RNO(model=rno_fno.model, time_interval=time_interval, n_timesteps=5)

    def test_invalid_timesteps_raises(self, rno_fno):
        with pytest.raises(ValueError):
            RNO(model=rno_fno.model, time_interval=(0, 5), n_timesteps=1)

    def test_dt(self, rno_fno):
        assert rno_fno.dt == pytest.approx(1.0)

    def test_step_is_explicit_euler(self, rno_fno, rno_inputs):
        x = rno_inputs["x"]
        expected = x + rno_fno.dt * rno_fno.model(x, inference=True)
        out = rno_fno.step(x, inference=True)
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)

    def test_step_with_forcing(self, rno_fno_forced, rno_inputs, forcing):
        x = rno_inputs["x"]
        f_t = forcing[0]
        expected = x + rno_fno_forced.dt * rno_fno_forced.model(
            jnp.concatenate([x, f_t]), inference=True
        )
        out = rno_fno_forced.step(x, f_t, inference=True)
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)

    def test_call_matches_manual_rollout(self, rno_fno_forced, rno_inputs, forcing):
        x = rno_inputs["x"]
        out = rno_fno_forced(x, forcing, inference=True)
        states = []
        for t in range(N_TIMESTEPS):
            x = rno_fno_forced.step(x, forcing[t], inference=True)
            states.append(x)
        assert out.shape == (N_TIMESTEPS, 1, GRID)
        np.testing.assert_allclose(out, jnp.stack(states), rtol=1e-5, atol=1e-5)

    def test_forcing_pytree(self, rno_hyperparams, rno_inputs):
        rno = RNO(model=make_fno(in_channels=4), **rno_hyperparams)
        f = {
            "a": jr.normal(jr.key(3), (N_TIMESTEPS, 1, GRID)),
            "b": jr.normal(jr.key(4), (N_TIMESTEPS, 2, GRID)),
        }
        out = rno(rno_inputs["x"], f, inference=True)
        assert out.shape == (N_TIMESTEPS, 1, GRID)

    def test_forcing_wrong_time_axis_raises(self, rno_fno_forced, rno_inputs):
        bad = jnp.zeros((N_TIMESTEPS + 1, 2, GRID))
        with pytest.raises(ValueError):
            rno_fno_forced(rno_inputs["x"], bad, inference=True)

    def test_forcing_channel_mismatch_raises(
        self, rno_hyperparams, rno_inputs, forcing
    ):
        # state (1) + forcing (2) = 3 channels but the model takes 2
        # (in_channels=1 is avoided: the pointwise einsum broadcasts size-1 dims)
        rno = RNO(model=make_fno(in_channels=2), **rno_hyperparams)
        with pytest.raises(ValueError):
            rno(rno_inputs["x"], forcing, inference=True)

    def test_output_channel_mismatch_raises(self, rno_hyperparams, rno_inputs):
        rno = RNO(model=make_fno(out_channels=3), **rno_hyperparams)
        with pytest.raises(ValueError):
            rno(rno_inputs["x"], inference=True)

    def test_key_none(self, rno_fno, rno_inputs):
        out = rno_fno(rno_inputs["x"])
        assert out.shape == (N_TIMESTEPS, 1, GRID)

    @pytest.mark.parametrize("inference", [False, True])
    def test_inference_flag_with_dropout(self, rno_hyperparams, rno_inputs, inference):
        rno = RNO(model=make_fno(channel_mlp_dropout=0.5), **rno_hyperparams)
        x = rno_inputs["x"]
        out1 = rno(x, key=jr.key(5), inference=inference)
        out2 = rno(x, key=jr.key(6), inference=inference)
        same = np.allclose(out1, out2)
        # dropout active only in training mode
        assert same == inference

    def test_static_kwargs_reach_model(self, rno_hyperparams, rno_inputs):
        rno = RNO(model=ScaledIdentity(), **rno_hyperparams)
        x = rno_inputs["x"]
        out = rno(x, scale=jnp.asarray(2.0))
        # u_{n+1} = u_n * (1 + dt * scale)
        factor = 1 + rno.dt * 2.0
        expected = jnp.stack([x * factor ** (t + 1) for t in range(N_TIMESTEPS)])
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)

    def test_custom_combine_fn(self, rno_hyperparams, rno_inputs):
        # scalar per-step forcing added to the state instead of concatenated
        rno = RNO(
            model=ScaledIdentity(),
            combine_fn=lambda x, f_t: x + f_t,
            **rno_hyperparams,
        )
        f = jnp.ones((N_TIMESTEPS, 1, GRID))
        out = rno(rno_inputs["x"], f, scale=jnp.asarray(0.0))
        # scale 0 -> state never changes
        np.testing.assert_allclose(out[-1], rno_inputs["x"], rtol=1e-6, atol=1e-6)

    def test_checkpoint_matches_plain(self, rno_fno, rno_fno_w_checkpoint, rno_inputs):
        x = rno_inputs["x"]

        def loss(model):
            return jnp.sum(model(x, inference=True) ** 2)

        plain = jtu.tree_leaves(eqx.filter_grad(loss)(rno_fno))
        ckpt = jtu.tree_leaves(eqx.filter_grad(loss)(rno_fno_w_checkpoint))
        assert len(plain) == len(ckpt)
        for a, b in zip(plain, ckpt, strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)

    def test_is_jittable(self, rno_fno, rno_inputs):
        assert_filter_jittable(rno_fno, rno_inputs["x"], key=rno_inputs["key"])

    def test_is_jittable_with_forcing(self, rno_fno_forced, rno_inputs, forcing):
        assert_filter_jittable(
            rno_fno_forced,
            rno_inputs["x"],
            forcing,
            key=rno_inputs["key"],
            inference=True,
        )
