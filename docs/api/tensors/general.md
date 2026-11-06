## General

In Neural Operators (such as the FNO and TFNO), weights are parameterized in the Fourier domain. For high-dimensional or high-channel settings, storing the full parameter tensor $(C_{out}, C_{in}, m_1, \dots, m_d)$ is computationally expensive. `neojax` provides modular tensor classes to represent these weights in dense or low-rank factorized forms.

!!! info "Weights are stored as real pairs"
    The tensors store their learnable weights as real arrays with `(real, imag)` pairs along a last axis of size 2 and assemble complex weights when they are used. For complex parameters, `jax.grad` returns the conjugate of the descent direction, so optax would move their imaginary parts in the wrong direction. With real pairs, no gradient conjugation is required, and `model.size(return_n_params=True)` counts each complex weight as two real parameters. Use the static methods `BaseTensor.to_real` and `BaseTensor.to_complex` to convert between both representations.


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

### Separable Weights (`separable=True`)

All tensor classes support depthwise-separable spectral convolutions via the `separable=True` flag.

When `separable=True`:

* Input and output channels must match (`in_channels == out_channels`).
* The weight tensor dimensions are reduced from $(C_{out}, C_{in}, m_1, \dots, m_d)$ to $(C, m_1, \dots, m_d)$.
* Contraction behaves pointwise/elementwise across the channel dimension. This is the spectral analogue of a depthwise separable convolution, which lowers parameter counts and memory usage.