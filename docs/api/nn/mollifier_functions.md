## Mollifier Functions

Mollifier functions, also known as weighting functions, are optionally used in the output Graph Neural Operator (GNO) of the `GiNO`. They replace the hard indicator function over the ball $\mathbb{1}_{B_r(x)}$ in the kernel integral of the GNO with a smooth differentiable weighting function.

??? cite

    [Enabling Automatic Differentiation
    with Mollified Graph Neural Operators](https://arxiv.org/pdf/2504.08277)

    ```bibtex
    @article{lin2025enabling,
        title={Enabling automatic differentiation with mollified graph neural operators},
        author={Lin, Ryan Y and Berner, Julius and Duruisseaux, Valentin and Pitt, David and Leibovici, Daniel and Kossaifi, Jean and Azizzadenesheli, Kamyar and Anandkumar, Anima},
        journal={arXiv preprint arXiv:2504.08277},
        year={2025}
    }
    ```

!!! info "Squared distances"
    The weights are evaluated on squared distances and take the squared radius, i.e. $d = \Vert x - y \Vert^2 / r^2$. This follows **neuraloperator** and the pseudocode of the mGNO paper. The paper's main text writes the same functions with $d = \Vert x - y \Vert / r$. The `quadr_cutoff` weighting function is not in the paper.

!!! warning "Neighbor truncation and Mollifiers"
    Neighbors beyond `max_neighbors` are dropped in the neighbor search, even with a nonzero weight, which reintroduces a discontinuity. Choose `max_neighbors` large enough that truncation never occurs if you rely on derivatives.

::: neojax.nn.mollifier_functions.bump_cutoff

---

::: neojax.nn.mollifier_functions.half_cos_cutoff

---

::: neojax.nn.mollifier_functions.quadr_cutoff

---

::: neojax.nn.mollifier_functions.quartic_cutoff

---

::: neojax.nn.mollifier_functions.octic_cutoff

---

::: neojax.nn.mollifier_functions.make_weighting_fn
