"""Implementation of a Local Neural Operator."""

import equinox as eqx
from jaxtyping import Array, Float


class LocalNO(eqx.Module):
    """Local Neural Operator.

    ??? cite

        [Neural operators with localized
        integral and differential kernels](https://arxiv.org/pdf/2402.16845)

        ```bibtex
        @article{liu2024neural,
            title={Neural operators with localized
            integral and differential kernels},
            author={Liu-Schiaffini, Miguel and Berner,
            Julius and Bonev, Boris and Kurth, Thorsten and Azizzadenesheli,
            Kamyar and Anandkumar, Anima},
            journal={arXiv preprint arXiv:2402.16845},
            year={2024}
        }
        ```
    """

    def __init__(self) -> None:
        pass

    def __call__(self, x: Float[Array, "c ..."]) -> Float[Array, "c ..."]:
        """"""
        pass
