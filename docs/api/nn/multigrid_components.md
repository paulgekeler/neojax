## Multigrid Components

Multgrid components are used in the Multigrid Neural Operator `MgNO` for the convolutional blocks
in the multigrid-style V-cycle<br>(Pre-Smoothing $\rightarrow$ Restriction $\rightarrow$ Coarse Correction $\rightarrow$ Prolongation $\rightarrow$ Post-Smoothing).


### Multigrid Convolution

::: neojax.nn.mg_conv.MgConv

---

### Multigrid Iteration

::: neojax.nn.mg_iter.MgIter

---

### Multigrid Restriction

::: neojax.nn.mg_restrict.MgRestrict