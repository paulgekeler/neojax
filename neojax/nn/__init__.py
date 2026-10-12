"""Neural network layers and blocks for neural operators."""

from neojax.nn.domain_padding import DomainPadding as DomainPadding
from neojax.nn.fno_blocks import FNOBlock as FNOBlock
from neojax.nn.fno_blocks import FNOBlocks as FNOBlocks
from neojax.nn.geo_map import GeoMapNd as GeoMapNd
from neojax.nn.geo_spectral_conv import GeoSpectralConvNd as GeoSpectralConvNd
from neojax.nn.gno_block import GNOBlock as GNOBlock
from neojax.nn.integral_transform import IntegralTransform as IntegralTransform
from neojax.nn.mg_conv import MgConv as MgConv
from neojax.nn.mg_iter import MgIter as MgIter
from neojax.nn.mg_restrict import MgRestrict as MgRestrict
from neojax.nn.mollifier_functions import bump_cutoff as bump_cutoff
from neojax.nn.mollifier_functions import half_cos_cutoff as half_cos_cutoff
from neojax.nn.mollifier_functions import make_weighting_fn as make_weighting_fn
from neojax.nn.mollifier_functions import octic_cutoff as octic_cutoff
from neojax.nn.mollifier_functions import quadr_cutoff as quadr_cutoff
from neojax.nn.mollifier_functions import quartic_cutoff as quartic_cutoff
from neojax.nn.normalization import InstanceNorm as InstanceNorm
from neojax.nn.pointwise_mlp import PointwiseMLP as PointwiseMLP
from neojax.nn.positional_embedding import GridEmbeddingNd as GridEmbeddingNd
from neojax.nn.positional_embedding import SinusoidalEmbedding as SinusoidalEmbedding
from neojax.nn.resample import Resampler as Resampler
from neojax.nn.skip_connections import Flattened1dConv as Flattened1dConv
from neojax.nn.skip_connections import SoftGating as SoftGating
from neojax.nn.skip_connections import make_skip_connection as make_skip_connection
from neojax.nn.spectral_conv import SpectralConvNd as SpectralConvNd

__all__ = [
    "DomainPadding",
    "FNOBlock",
    "FNOBlocks",
    "GeoMapNd",
    "GeoSpectralConvNd",
    "InstanceNorm",
    "PointwiseMLP",
    "GridEmbeddingNd",
    "SinusoidalEmbedding",
    "Flattened1dConv",
    "SoftGating",
    "make_skip_connection",
    "SpectralConvNd",
    "Resampler",
    "MgConv",
    "MgRestrict",
    "MgIter",
    "GNOBlock",
    "IntegralTransform",
    "make_weighting_fn",
    "bump_cutoff",
    "half_cos_cutoff",
    "quadr_cutoff",
    "quartic_cutoff",
    "octic_cutoff",
]
