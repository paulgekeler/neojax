"""Spectral tensors."""

from neojax.tensor.base_tensor import BaseTensor as BaseTensor
from neojax.tensor.cp_tensor import CPTensor as CPTensor
from neojax.tensor.dense_tensor import DenseTensor as DenseTensor
from neojax.tensor.tt_tensor import TTTensor as TTTensor
from neojax.tensor.tucker_tensor import TuckerTensor as TuckerTensor

__all__ = ["BaseTensor", "CPTensor", "TuckerTensor", "TTTensor", "DenseTensor"]
