import jax.numpy as jnp
import pytest
from jaxtyping import TypeCheckError

from neojax.nn.domain_padding import DomainPadding


class TestDomainPadding:
    def test_scalar_padding(self):
        # 2D
        padder_2d = DomainPadding(padding=0.2)
        x_2d = jnp.ones((2, 10, 10))
        x_padded_2d = padder_2d.pad(x_2d)
        assert x_padded_2d.shape == (2, 14, 14)
        assert padder_2d.unpad(x_padded_2d, x_2d.shape).shape == x_2d.shape

        # 3D
        padder_3d = DomainPadding(padding=0.5)
        x_3d = jnp.ones((2, 4, 4, 4))
        x_padded_3d = padder_3d.pad(x_3d)
        assert x_padded_3d.shape == (2, 8, 8, 8)
        assert padder_3d.unpad(x_padded_3d, x_3d.shape).shape == x_3d.shape

        # 4D
        padder_4d = DomainPadding(padding=0.25)
        x_4d = jnp.ones((1, 8, 8, 8, 8))
        x_padded_4d = padder_4d.pad(x_4d)
        assert x_padded_4d.shape == (1, 12, 12, 12, 12)
        assert padder_4d.unpad(x_padded_4d, x_4d.shape).shape == x_4d.shape

        # 5D
        padder_5d = DomainPadding(padding=0.5)
        x_5d = jnp.ones((1, 4, 4, 4, 4, 4))
        x_padded_5d = padder_5d.pad(x_5d)
        assert x_padded_5d.shape == (1, 8, 8, 8, 8, 8)
        assert padder_5d.unpad(x_padded_5d, x_5d.shape).shape == x_5d.shape

    def test_sequence_padding(self):
        padding_val = (0.2, 0.4)
        padder = DomainPadding(padding=padding_val)
        x = jnp.ones((2, 10, 10))
        x_padded = padder.pad(x)
        # 0.2*2*10=4 (size 14), 0.4*2*10=8 (size 18)
        assert x_padded.shape == (2, 14, 18)
        assert padder.unpad(x_padded, x.shape).shape == x.shape

    def test_padding_w_scalar_scaling(self):
        padding = 0.2
        res_scaling = 2.0
        padder = DomainPadding(padding, resolution_scaling_factor=res_scaling)
        x = jnp.ones((2, 10, 10))
        padded = padder(x)
        # some no layer scales padded by scaling factor
        x_scaled = jnp.ones(
            (padded.shape[0],) + tuple(round(d * res_scaling) for d in padded.shape[1:])
        )
        unpadded = padder.unpad(x_scaled, x.shape)

        assert unpadded.shape == (2, 20, 20)

    def test_padding_w_seq_scaling(self):
        padding = 0.2
        res_scaling = (2.0, 3.0)
        padder = DomainPadding(padding, resolution_scaling_factor=res_scaling)
        x = jnp.ones((2, 10, 10))
        padded = padder(x)
        # some no layer scales padded by scaling factor
        x_scaled = jnp.ones(
            (padded.shape[0],)
            + tuple(
                round(d * s)
                for (d, s) in zip(padded.shape[1:], res_scaling, strict=True)
            )
        )
        unpadded = padder.unpad(x_scaled, x.shape)

        assert unpadded.shape == (2, 20, 30)

    def test_input_verification(self):
        # unequal padding and scale length
        padding = (0.2,)
        res_scale_factor = (2.0, 3.0)
        with pytest.raises(ValueError):
            DomainPadding(padding, resolution_scaling_factor=res_scale_factor)

        # int padding or Seq[int] padding
        padding = 5
        with pytest.raises((ValueError, TypeCheckError)):
            DomainPadding(padding)
        with pytest.raises((ValueError, TypeCheckError)):
            DomainPadding([5, 4])

        # invalid pad mode
        with pytest.raises((ValueError, TypeCheckError)):
            DomainPadding((0.2, 0.2), mode="invalid")
