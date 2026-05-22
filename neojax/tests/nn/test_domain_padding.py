import jax.numpy as jnp

from neojax.nn.domain_padding import DomainPadding


class TestDomainPadding:
    def test_scalar_padding(self):
        # 2D
        padder_2d = DomainPadding(padding=0.2)
        x_2d = jnp.ones((2, 10, 10))
        x_padded_2d = padder_2d.pad(x_2d)
        assert x_padded_2d.shape == (2, 12, 12)
        assert padder_2d.unpad(x_padded_2d, x_2d.shape).shape == x_2d.shape

        # 3D
        padder_3d = DomainPadding(padding=0.5)
        x_3d = jnp.ones((2, 4, 4, 4))
        x_padded_3d = padder_3d.pad(x_3d)
        assert x_padded_3d.shape == (2, 6, 6, 6)
        assert padder_3d.unpad(x_padded_3d, x_3d.shape).shape == x_3d.shape

        # 4D
        padder_4d = DomainPadding(padding=0.25)
        x_4d = jnp.ones((1, 8, 8, 8, 8))
        x_padded_4d = padder_4d.pad(x_4d)
        assert x_padded_4d.shape == (1, 10, 10, 10, 10)
        assert padder_4d.unpad(x_padded_4d, x_4d.shape).shape == x_4d.shape

        # 5D
        padder_5d = DomainPadding(padding=0.5)
        x_5d = jnp.ones((1, 4, 4, 4, 4, 4))
        x_padded_5d = padder_5d.pad(x_5d)
        assert x_padded_5d.shape == (1, 6, 6, 6, 6, 6)
        assert padder_5d.unpad(x_padded_5d, x_5d.shape).shape == x_5d.shape

    def test_sequence_padding(self):
        padding_val = (0.2, 0.4)
        padder = DomainPadding(padding=padding_val)
        x = jnp.ones((2, 10, 10))
        x_padded = padder.pad(x)
        # 0.2*10=2 (size 12), 0.4*10=4 (size 14)
        assert x_padded.shape == (2, 12, 14)
        assert padder.unpad(x_padded, x.shape).shape == x.shape
