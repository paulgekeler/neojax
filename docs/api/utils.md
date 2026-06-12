# Utilities Reference

This page contains the API reference for all utility functions and classes throughout the library.

## Loss Utilities

### Filter Functions

In order to filter fixed loss weights during training updates, there is a utility function to pass to the
`equinox.filter_...` or `equinox.partition` functions as a `filter_spec`:

::: neojax.losses.utils.is_learnable_loss_weight
