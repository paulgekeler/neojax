import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

from neojax.training.diagnostics import (
    compute_grad_report,
    flag_anomalous_norms,
    grad_norms,
    select_by_path,
)
from neojax.training.trainer import Trainer


@pytest.fixture
def model():
    key = jr.key(1)
    return eqx.nn.MLP(in_size=2, out_size=1, width_size=4, depth=1, key=key)


@pytest.fixture
def mlp_and_grads(model):
    key = jr.key(0)
    x = jr.normal(key, (10, 2))
    y = jr.normal(key, (10, 1))

    def loss_fn(m):
        preds = jax.vmap(m)(x)
        return jnp.mean((preds - y) ** 2)

    _, grads = eqx.filter_value_and_grad(loss_fn)(model)
    return model, grads


@pytest.mark.usefixtures("mlp_and_grads")
class TestGradNorms:
    def test_matches_manual_norms(self, mlp_and_grads):
        model, grads = mlp_and_grads
        norms = grad_norms(grads)

        manual = {
            jax.tree_util.keystr(path): jnp.linalg.norm(leaf.ravel())
            for path, leaf in jax.tree_util.tree_leaves_with_path(grads)
            if leaf is not None
        }
        assert norms.keys() == manual.keys()
        for k in norms:
            assert jnp.allclose(norms[k], manual[k])

    def test_skips_none_leaves(self, mlp_and_grads):
        # No. of filtered gradients (static marked as None) should match
        # no. of non-static model fields
        model, grads = mlp_and_grads
        n_grad_leaves = len(grad_norms(grads))
        n_model_arrays = sum(
            1 for leaf in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
        )
        assert n_grad_leaves == n_model_arrays

    def test_handles_complex_leaves(self, mlp_and_grads):
        grads = {"spectral_weight": jnp.array([1.0 + 1.0j, 2.0 - 2.0j])}
        norms = grad_norms(grads)
        expected = jnp.sqrt(jnp.abs(1 + 1j) ** 2 + jnp.abs(2 - 2j) ** 2)
        assert jnp.allclose(next(iter(norms.values())), expected)

    def test_jit_compatible(self, mlp_and_grads):
        _, grads = mlp_and_grads

        @eqx.filter_jit
        def jitted(g):
            return grad_norms(g)

        eager = grad_norms(grads)
        traced = jitted(grads)
        assert eager.keys() == traced.keys()
        for k in eager:
            assert jnp.allclose(eager[k], traced[k])


class TestSelectByPath:
    def test_selects_matching_keys(self):
        report = {
            "model.fno_blocks[0].weight": 0.1,
            "model.fno_blocks[1].weight": 0.2,
            "model.lifting.weight": 0.3,
        }
        assert select_by_path(report, "*fno_blocks*") == {
            "model.fno_blocks[0].weight": 0.1,
            "model.fno_blocks[1].weight": 0.2,
        }
        assert select_by_path(report, "*lifting*") == {"model.lifting.weight": 0.3}

    def test_no_matches_returns_empty(self):
        report = {"a.weight": 0.1}
        assert select_by_path(report, "*nonexistent*") == {}

    def test_literal_bracket_index_never_matches_naively(self):
        # fnmatch parses [...] as a character class not as literal
        # brackets
        # Writing a literal array index as pattern never matches
        report = {"a[0].weight": 1.0, "a[10].weight": 2.0}
        assert select_by_path(report, "a[0].weight") == {}
        assert select_by_path(report, "a[10].weight") == {}

    def test_plain_wildcard_spans_the_index_correctly(self):
        # Correct way to match regardless of index:
        # Asterisk matches brackets and digits
        report = {"a[0].weight": 1.0, "a[10].weight": 2.0, "b[0].weight": 3.0}
        assert select_by_path(report, "a*.weight") == {
            "a[0].weight": 1.0,
            "a[10].weight": 2.0,
        }

    def test_escaped_brackets_match_literally(self):
        # Test fnmatch's own escaping for a literal bracket
        report = {"a[0].weight": 1.0, "a[10].weight": 2.0}
        assert select_by_path(report, "a[[]0[]].weight") == {"a[0].weight": 1.0}


class TestFlagAnomalousNorms:
    def test_flags_vanishing_and_exploding(self):
        grads = {
            "dead": jnp.array(1e-12),
            "healthy": jnp.array(0.5),
            "exploding": jnp.array(1e6),
        }
        flagged = flag_anomalous_norms(grads, min_norm=1e-8, max_norm=1e4)
        keys = {k.split("[")[-1].strip("]'\"") for k in flagged}
        assert "dead" in keys
        assert "exploding" in keys
        assert "healthy" not in keys

    def test_empty_when_nothing_anomalous(self):
        grads = {"a": jnp.array(0.5), "b": jnp.array(1.0)}
        assert flag_anomalous_norms(grads, min_norm=1e-8, max_norm=1e4) == {}

    def test_disabling_a_bound(self):
        grads = {"tiny": jnp.array(1e-12)}
        assert flag_anomalous_norms(grads, min_norm=None, max_norm=1e4) == {}
        assert len(flag_anomalous_norms(grads, min_norm=1e-8, max_norm=None)) == 1


@pytest.mark.usefixtures("model")
class TestComputeGradReport:
    def test_reports_nothing_for_a_healthy_step(self, model):
        key = jr.key(0)
        optimizer = optax.adam(1e-2)

        def loss_fn(m, batch, training):
            x, y = batch
            preds = jax.vmap(m)(x)
            return jnp.mean((preds - y) ** 2)

        trainer = Trainer(optimizer=optimizer, loss_fn=loss_fn)
        state = trainer.create_train_state(model)

        x = jr.normal(key, (10, 2))
        y = jr.normal(jr.key(1), (10, 1))
        report = compute_grad_report(
            trainer, state, (x, y), min_norm=1e-12, max_norm=1e8
        )
        assert report == {}

        # Confirm Trainer didn't take a step.
        assert state.step == 0

    def test_flags_a_dead_parameter_with_zero_gradient(self):
        # Never use 'dead' linear part -> gradient zero
        # Mimics a wrongly marked static attribute or
        # a wrong filter specification passed to eqx.partition/filter_...
        key = jr.key(0)

        class WithDeadBranch(eqx.Module):
            live: eqx.nn.Linear
            dead: eqx.nn.Linear

            def __call__(self, x):
                # Only use live part
                return self.live(x)

        k1, k2 = jr.split(key)
        model = WithDeadBranch(
            live=eqx.nn.Linear(2, 1, key=k1), dead=eqx.nn.Linear(2, 1, key=k2)
        )

        def loss_fn(m, batch, training):
            x, y = batch
            preds = jax.vmap(m)(x)
            return jnp.mean((preds - y) ** 2)

        optimizer = optax.adam(1e-2)
        trainer = Trainer(optimizer=optimizer, loss_fn=loss_fn)
        state = trainer.create_train_state(model)

        x = jr.normal(key, (10, 2))
        y = jr.normal(jr.key(1), (10, 1))
        report = compute_grad_report(
            trainer, state, (x, y), min_norm=1e-8, max_norm=None
        )

        assert any("dead" in k for k in report)
        assert not any("live" in k for k in report)

    def test_has_aux_loss_fn_is_handled(self, model):
        key = jr.key(0)
        optimizer = optax.adam(1e-2)

        def loss_fn(m, batch, training):
            x, y = batch
            preds = jax.vmap(m)(x)
            loss = jnp.mean((preds - y) ** 2)
            return loss, {"training": training}

        trainer = Trainer(optimizer=optimizer, loss_fn=loss_fn, has_aux=True)
        state = trainer.create_train_state(model)

        x = jr.normal(key, (10, 2))
        y = jr.normal(jr.key(1), (10, 1))
        # Should not raise despite loss_fn returning a (loss, aux) pair.
        compute_grad_report(trainer, state, (x, y), min_norm=1e-12, max_norm=1e8)
