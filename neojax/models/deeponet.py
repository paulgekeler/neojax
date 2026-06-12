"""Implemenetation of DeepONets.

Also implements a subclass for practitioners:

- MLPDeepONet: branch mlp + trunk mlp
"""

from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from neojax.nn.pointwise_mlp import PointwiseMLP


class DeepONet(eqx.Module):
    """General DeepONet (Deep Operator Network).

    Learns a PDE operator through two separate
    branch and trunk networks.
    The branch net encodes the discrete function space
    and the trunk net encodes the domain of output functions.
    Implemented as in the original publication [[1]](#ref1).

    Args:
        branch_net: The network representing the branch (encodes u).
            Should map shape (m_sensors,) to (p,). May also take
            different input shapes, as long as it outputs (p,),
            e.g. CNN branch net.
        trunk_net: The network representing the trunk (encodes y).
            Should map shape (d_dim,) to (p,).
        out_activation: Optional output activation after dot product.
        bias: Optional learnable bias after dot product.

    !!! info "Internal Attributes"
        These fields store the internal layers state (and weights).

        * **branch_net** (`eqx.Module`): Arbitrary NN that maps (m_sensors,) -> (p,).
        * **trunk_net** (`eqx.Module`): Arbitrary NN that maps (d_dim,) -> (p,).
        * **out_activation** (`Callable | None`): Optional output activation after dot product.
        * **bias** (`Float[Array, "1"] | None`): Optional learnable bias after dot product.

    !!! info
        This is the **Unstacked (Standard)** implementation as described in
        Lu et al. (2021). It uses a single branch network and a single trunk
        network for maximum computational efficiency.

        For vector-valued operators (the "Stacked" variant),
        it is recommended to either:

        1. Increase the output dimension of the branch/trunk networks.
        2. Use `jax.vmap` to wrap this class for multiple independent operators.

    ??? cite

        [Learning nonlinear operators via DeepONet
        based on the universal approximation theorem of operators]
        (https://www.nature.com/articles/s42256-021-00302-5)

        ```bibtex
        @article{lu2021learning,
            title={Learning nonlinear operators via DeepONet
            based on the universal approximation theorem of operators},
            author={Lu, Lu and Jin, Pengzhan and Pang,
            Guofei and Zhang, Zhongqiang and Karniadakis, George Em},
            journal={Nature machine intelligence},
            volume={3},
            number={3},
            pages={218--229},
            year={2021},
            publisher={Nature Publishing Group}
        }
        ```
    """

    branch_net: eqx.Module
    trunk_net: eqx.Module
    out_activation: Callable | None = None
    bias: Float[Array, "1"] | None = None

    def __init__(
        self,
        branch_net: eqx.Module,
        trunk_net: eqx.Module,
        out_activation: Callable | None = None,
        bias: Float[Array, "1"] | None = None,
    ) -> None:
        self.branch_net = branch_net
        self.trunk_net = trunk_net
        self.out_activation = out_activation
        self.bias = bias

    def __call__(
        self, u: Float[Array, "m_sensors"], y: Float[Array, "d_dim"]
    ) -> Float[Array, ""]:
        """Forward pass of DeepONet.

        Args:
            u: Input function discretized at sensor points.
            y: Coordinates in the output domain.

        Returns:
            Scalar value of operator at y.
        """
        b = self.branch_net(u)  # (p,)
        t = self.trunk_net(y)  # (p,)
        out = jnp.dot(b.ravel(), t.ravel())

        if self.bias is not None:
            out = out + self.bias.squeeze()

        if self.out_activation is not None:
            out = self.out_activation(out)

        return out


class MLPDeepONet(DeepONet):
    """DeepONet with MLP branch and MLP trunk network.

    The branch MLP maps (m_sensors,) funtion values
    to the (p,) latent vector and the trunk MLP maps
    the (d,) evaluation vector to the (p,) latent vector.

    Args:
        key: Jax random key.
        m_sensors: Number of function values u.
        d_dim: Output space dimension.
        p_latent: Latent vector dimension.
        branch_hidden_dims: Hidden dimensions of the branch MLP.
            First layer is (m_sensors, branch_hidden_dims[0])
            and last layer is (branch_hidden_dims[-1], p_latent).
        trunk_hidden_dims: Hidden dimensions of the trunk MLP.
            First layers is (d_dim, trunk_hidden_dims[0])
            and last layer is (trunk_hidden_dims[-1], p_latent).
        branch_activations: Activation functions of branch MLP.
        trunk_activations: Activation functions of trunk MLP.
            Single callable means, all activations use this function.
        out_activation: Optional output activation after dot product.
        bias: Optional learnable bias after dot product.

    !!! info "Internal Attributes"
        These fields store the internal layers state (and weights).

        * **branch_net** (`PointwiseMLP`): MLP that maps (m_sensors,) -> (p,).
        * **trunk_net** (`PointwiseMLP`): MLP that maps (d_dim,) -> (p,).
        * **out_activation** (`Callable | None`): Optional output activation after dot product.
        * **bias** (`Float[Array, "1"] | None`): Optional learnable bias after dot product.
    """

    def __init__(
        self,
        key: PRNGKeyArray,
        m_sensors: int,
        d_dim: int,
        p_latent: int,
        branch_hidden_dims: Sequence[int],
        trunk_hidden_dims: Sequence[int],
        branch_activations: Callable | Sequence[Callable] = jax.nn.gelu,
        trunk_activations: Callable | Sequence[Callable] = jax.nn.gelu,
        out_activation: Callable | None = None,
        bias: Float[Array, "1"] | None = None,
    ) -> None:
        bkey, tkey = jr.split(key, 2)
        self.branch_net = PointwiseMLP(
            bkey,
            [m_sensors] + list(branch_hidden_dims) + [p_latent],
            branch_activations,
        )
        self.trunk_net = PointwiseMLP(
            tkey, [d_dim] + trunk_hidden_dims + [p_latent], trunk_activations
        )
        self.out_activation = out_activation
        self.bias = bias
