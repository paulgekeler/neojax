import equinox as eqx
import jax.numpy as jnp

from neojax.data.bundles.data_bundle import DataBundle
from neojax.data.normalizers import (
    ComposedNormalizer,
    MinMaxNormalizer,
    PhysicsNormalizer,
    RobustNormalizer,
    UnitGaussianNormalizer,
)
from neojax.data.processor import BundleProcessor
from neojax.data.scales import CharacteristicLengthScale
from neojax.data.schemas.bundle_reconstruct_schema import BundleReconstructSchema
from neojax.data.schemas.identity_schema import IdentitySchema


class TestBundleProcessor:
    def _make_bundle(self):
        fields = jnp.arange(32).reshape(4, 1, 1, 8).astype(jnp.float32) + 1.0
        coords = jnp.ones((1, 1, 8))
        return DataBundle(fields=fields, coords=coords)

    def test_transform_and_inverse_transform_round_trip(self):
        bundle = self._make_bundle()
        processor = BundleProcessor(
            normalizers={"fields": UnitGaussianNormalizer()}
        ).compute_stats(bundle)

        model_inputs = processor.transform(bundle, schema=IdentitySchema())
        assert jnp.allclose(
            model_inputs, processor.normalizers["fields"].transform(bundle.fields)
        )

        pred_bundle = processor.inverse_transform(
            model_inputs, schema=BundleReconstructSchema(), reference_bundle=bundle
        )
        assert jnp.allclose(pred_bundle.fields, bundle.fields, atol=1e-4)

    def test_no_normalizers_is_passthrough(self):
        bundle = self._make_bundle()
        processor = BundleProcessor()

        model_inputs = processor.transform(bundle, schema=IdentitySchema())
        assert jnp.array_equal(model_inputs, bundle.fields)

        pred_bundle = processor.inverse_transform(
            model_inputs, schema=BundleReconstructSchema(), reference_bundle=bundle
        )
        assert jnp.array_equal(pred_bundle.fields, bundle.fields)

    def test_compute_stats_squeezes_batch_dim_for_min_max_normalizer(self):
        # compute_stats computes stats with keepdims=True
        # Processor must squeeze batch axis
        # -> So Normalizer also works on vmapped data
        bundle = self._make_bundle()
        processor = BundleProcessor(
            normalizers={"fields": MinMaxNormalizer()}
        ).compute_stats(bundle)

        normed_batched = processor.normalizers["fields"].transform(bundle.fields)
        normed_unbatched = processor.normalizers["fields"].transform(bundle.fields[0])
        assert normed_batched.shape == bundle.fields.shape
        assert normed_unbatched.shape == bundle.fields.shape[1:]

    def test_compute_stats_squeezes_batch_dim_for_robust_normalizer(self):
        bundle = self._make_bundle()
        processor = BundleProcessor(
            normalizers={"fields": RobustNormalizer()}
        ).compute_stats(bundle)

        normed_unbatched = processor.normalizers["fields"].transform(bundle.fields[0])
        assert normed_unbatched.shape == bundle.fields.shape[1:]

    def test_compute_stats_squeezes_batch_dim_for_composed_normalizer(self):
        # Each composed normalizer needs its batch axis squeezed
        bundle = self._make_bundle()
        composed = ComposedNormalizer(RobustNormalizer(), UnitGaussianNormalizer())
        processor = BundleProcessor(normalizers={"fields": composed}).compute_stats(
            bundle
        )

        normed_unbatched = processor.normalizers["fields"].transform(bundle.fields[0])
        assert normed_unbatched.shape == bundle.fields.shape[1:]

    def test_compute_stats_leaves_unbatched_normalizer_stats_untouched(self):
        # PhysicsNormalizer.compute_stats doesn't reduce at all (unlike other normalizers)
        # -> stats match input shape
        # Check here if leading axes with size != 1 are kept as is
        bundle = self._make_bundle()
        bundle = eqx.tree_at(
            lambda b: b.parameters,
            bundle,
            jnp.ones((5, 3)),
            is_leaf=lambda x: x is None,
        )
        processor = BundleProcessor(
            normalizers={
                "parameters": PhysicsNormalizer(CharacteristicLengthScale(L_ref=1.0))
            }
        ).compute_stats(bundle)

        scale_product = processor.normalizers["parameters"].stats["scale_product"]
        assert scale_product.shape == bundle.parameters.shape
        normed = processor.normalizers["parameters"].transform(bundle.parameters)
        assert normed.shape == bundle.parameters.shape

    def test_compute_stats_ignores_missing_attribute(self):
        bundle = self._make_bundle()
        processor = BundleProcessor(
            normalizers={"parameters": UnitGaussianNormalizer()}
        ).compute_stats(bundle)

        # 'parameters' is None on this bundle, so the normalizer is left at its
        # default (unfitted) identity stats rather than being computed.
        assert processor.normalizers["parameters"].stats["mean"] == 0.0
        assert processor.normalizers["parameters"].stats["std"] == 1.0
