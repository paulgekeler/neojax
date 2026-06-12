"""Neural network layers and blocks for neural operators."""

from neojax.nn.domain_padding import DomainPadding
from neojax.nn.fno_blocks import FNOBlock, FNOBlocks
from neojax.nn.normalization import InstanceNorm
from neojax.nn.pointwise_mlp import PointwiseMLP
from neojax.nn.positional_embedding import GridEmbeddingNd
from neojax.nn.skip_connections import Flattened1dConv, SoftGating
from neojax.nn.spectral_conv import SpectralConvNd
from neojax.nn.tfno_blocks import TFNOBlock, TFNOBlocks
from neojax.nn.tucker_spectral_conv import TuckerSpectralConvNd

__all__ = [
    "DomainPadding",
    "FNOBlock",
    "FNOBlocks",
    "InstanceNorm",
    "PointwiseMLP",
    "GridEmbeddingNd",
    "Flattened1dConv",
    "SoftGating",
    "SpectralConvNd",
    "TuckerSpectralConvNd",
    "TFNOBlock",
    "TFNOBlocks",
]
