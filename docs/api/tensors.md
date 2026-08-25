# Tensors Reference

In Neural Operators (such as the FNO and TFNO), weights are parameterized in the Fourier domain. For high-dimensional or high-channel settings, storing the full parameter tensor $(C_{out}, C_{in}, m_1, \dots, m_d)$ is computationally expensive. `neojax` provides modular tensor classes to represent these weights in dense or low-rank factorized forms.

??? cite "Useful overview on Tensor Operations"

    [Tensor Decompositions and Applications](https://epubs.siam.org/doi/epdf/10.1137/07070111X)

    ```bibtex
    @article{kolda2009tensor,
        title={Tensor decompositions and applications},
        author={Kolda, Tamara G and Bader, Brett W},
        journal={SIAM review},
        volume={51},
        number={3},
        pages={455--500},
        year={2009},
        publisher={SIAM}
    }
    ```

---

## Separable Weights (`separable=True`)

All tensor classes support depthwise-separable spectral convolutions via the `separable=True` flag.

When `separable=True`:

* Input and output channels must match (`in_channels == out_channels`).
* The weight tensor dimensions are reduced from $(C_{out}, C_{in}, m_1, \dots, m_d)$ to $(C, m_1, \dots, m_d)$.
* Contraction behaves pointwise/elementwise across the channel dimension. This is the spectral analogue of a depthwise separable convolution, which lowers parameter counts and memory usage.

---

## Base Tensor Class

All representations inherit from `BaseTensor`.

::: neojax.tensor.BaseTensor

---

## Standard Dense Tensor

A standard un-factorized representation of Fourier weights.

::: neojax.tensor.DenseTensor

---

## Tucker Tensor

Represents the weights in a Tucker decomposition format, factorized into a core tensor and mode-specific factor matrices.

::: neojax.tensor.TuckerTensor

---

## CP (Canonical-Polyadic) Tensor

Represents the weights in a Canonical Polyadic decomposition format (approximating the weight tensor as a sum of K rank-1 tensors).

::: neojax.tensor.CPTensor

---

## Tensor Train (TT) Tensor

Represents the weights in a Tensor Train format, factorizing the tensor into a chain of low-dimensional tensors connected back-to-back.

::: neojax.tensor.TTTensor
