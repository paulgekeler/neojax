## Utilities Reference

This page contains the API reference for all utility functions and classes throughout the library.

---

### PyTorch Weight Conversion

These utilities help import pre-trained weights from PyTorch models (specifically the `neuraloperator` library) into their native JAX `neojax` equivalents.

!!! warning
    Weight conversion is highly experimental. Most models are not supported yet and it may not work.
    It is not a high priority right now. Contributions are very welcome!

::: neojax.utils.torch_converters.load_torch_weights_into_fno

::: neojax.utils.torch_converters.load_torch_weights_into_geo_fno

---

### Tree Utilites

::: neojax.utils.tree_utils.tree_add_broadcast_to

---

## Segment Operations

The segment operations are inspired and adapted from the [jraph](https://github.com/google-deepmind/jraph/tree/master) library.

::: neojax.utils.segment_ops.segment_mean

---

::: neojax.utils.segment_ops.segment_max

---

::: neojax.utils.segment_ops.segment_softmax