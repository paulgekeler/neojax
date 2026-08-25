import jax.random as jr
import pytest

from neojax.nn.geo_map import GeoMapNd
from neojax.tests.conftest import assert_filter_jittable


class TestGeoMap:
    def test_geo_map_shapes(self):
        key = jr.key(0)
        # Test 2D
        geomap_2d = GeoMapNd(key=key, dim=2, width=8)
        x_2d = jr.uniform(key, (10, 2))
        out_2d = geomap_2d(x_2d)
        assert out_2d.shape == (10, 2)
        assert_filter_jittable(geomap_2d, x_2d)

        # Test 1D
        geomap_1d = GeoMapNd(key=key, dim=1, width=8)
        x_1d = jr.uniform(key, (10, 1))
        out_1d = geomap_1d(x_1d)
        assert out_1d.shape == (10, 1)
        assert_filter_jittable(geomap_1d, x_1d)

        # Test 3D
        geomap_3d = GeoMapNd(key=key, dim=3, width=8)
        x_3d = jr.uniform(key, (10, 3))
        out_3d = geomap_3d(x_3d)
        assert out_3d.shape == (10, 3)
        assert_filter_jittable(geomap_3d, x_3d)

    def test_geo_map_with_code(self):
        key = jr.key(0)
        geomap = GeoMapNd(key=key, dim=2, width=8, code_dim=4)
        x = jr.uniform(key, (12, 2))
        code = jr.normal(key, (4,))
        out = geomap(x, code=code)
        assert out.shape == (12, 2)
        assert_filter_jittable(geomap, x, code=code)

        # Raises if code is provided but code_dim was not set
        geomap_err = GeoMapNd(key=key, dim=2, width=8)
        with pytest.raises(ValueError, match="code_dim was not provided"):
            geomap_err(x, code=code)
