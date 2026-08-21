"""Utility functions to load trained weights from PyTorch (neuraloperator library) into Neojax models."""

from typing import Any

import equinox as eqx
import jax.numpy as jnp

from neojax.models.fno import FNO
from neojax.models.geo_fno import GeoFNO


def _to_jax(tensor: Any) -> jnp.ndarray:
    """Helper to convert a PyTorch tensor, numpy array or JAX array to a JAX array."""
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu().detach().numpy()
    return jnp.array(tensor)


class WeightUpdater:
    """Helper to queue and apply parameter updates from a state dict to an Equinox model."""

    def __init__(self, model: Any, state_dict: dict[str, Any]) -> None:
        self.model = model
        self.state_dict = state_dict
        self.updates = []

    def add(
        self,
        where_fn: Any,
        key_name: str,
        transpose_axes: tuple[int, ...] | None = None,
        squeeze_axes: int | tuple[int, ...] | None = None,
    ) -> None:
        """Queues an update for a parameter if its key is present in the state dict.

        Args:
            where_fn: A function `lambda model: model.path.to.parameter` selecting
                the target parameter in the model.
            key_name: Key of the parameter in the PyTorch state dict.
            transpose_axes: Optional axis permutation to transpose the tensor.
            squeeze_axes: Optional axes to squeeze/remove from the tensor.
        """
        if key_name in self.state_dict:
            try:
                current_val = where_fn(self.model)
            except Exception:
                current_val = None
            if current_val is None:
                return  # Skip if target parameter is None or not present

            val = _to_jax(self.state_dict[key_name])
            if transpose_axes is not None:
                val = jnp.transpose(val, transpose_axes)
            if squeeze_axes is not None:
                val = jnp.squeeze(val, axis=squeeze_axes)
            self.updates.append((where_fn, val))

    def apply(self) -> Any:
        """Applies all queued updates to the model using eqx.tree_at and returns the updated model."""
        new_model = self.model
        for where_fn, val in self.updates:
            new_model = eqx.tree_at(where_fn, new_model, val)
        return new_model


def load_torch_weights_into_fno(model: FNO, state_dict: dict[str, Any]) -> FNO:
    """Loads weights from a PyTorch FNO model state dict into a Neojax FNO instance.

    Handles necessary shape transpositions (e.g. PyTorch [in, out, ...] vs
    JAX [out, in, ...] for spectral weights).

    Args:
        model: The target Neojax `FNO` model.
        state_dict: PyTorch FNO model state dict.

    Returns:
        A new FNO model instance with loaded weights.
    """
    updater = WeightUpdater(model, state_dict)

    # 1. Lifting
    updater.add(lambda m: m.lifting.weights[0], "fc0.weight")
    updater.add(lambda m: m.lifting.biases[0], "fc0.bias")

    # 2. Fourier Layers
    if model.fno_blocks is not None:
        n_layers = len(model.fno_blocks.fno_layers)
        for i in range(n_layers):
            # Conv weights: PyTorch [in, out, m1, m2] -> JAX [out, in, m1, m2]
            updater.add(
                lambda m, idx=i: m.fno_blocks.fno_layers[idx].spectral_conv.weights.weights[0],
                f"conv{i}.weights1",
                transpose_axes=(1, 0, 2, 3)
            )
            updater.add(
                lambda m, idx=i: m.fno_blocks.fno_layers[idx].spectral_conv.weights.weights[1],
                f"conv{i}.weights2",
                transpose_axes=(1, 0, 2, 3)
            )
            # Skip/w weights: PyTorch [out, in, 1, 1] -> JAX Conv1D [out, in, 1]
            updater.add(
                lambda m, idx=i: m.fno_blocks.fno_layers[idx].local_operator.conv.weight,
                f"w{i}.weight",
                squeeze_axes=-1
            )
            updater.add(
                lambda m, idx=i: m.fno_blocks.fno_layers[idx].local_operator.conv.bias,
                f"w{i}.bias"
            )

    # 3. Projection
    updater.add(lambda m: m.projection.weights[0], "fc1.weight")
    updater.add(lambda m: m.projection.biases[0], "fc1.bias")
    updater.add(lambda m: m.projection.weights[1], "fc2.weight")
    updater.add(lambda m: m.projection.biases[1], "fc2.bias")

    return updater.apply()


def load_torch_weights_into_geo_fno(model: GeoFNO, state_dict: dict[str, Any]) -> GeoFNO:
    """Loads weights from a PyTorch Geo-FNO model state dict into a Neojax GeoFNO instance.

    Handles coordinate diffeomorphism (geomap) parameters and intermediate coordinate biases.

    Args:
        model: The target Neojax `GeoFNO` model.
        state_dict: PyTorch Geo-FNO model state dict.

    Returns:
        A new GeoFNO model instance with loaded weights.
    """
    updater = WeightUpdater(model, state_dict)

    # 1. GeoMap (IPHI)
    updater.add(lambda m: m.geomap.lin0.weight, "model_iphi.fc0.weight")
    updater.add(lambda m: m.geomap.lin0.bias, "model_iphi.fc0.bias")

    if model.geomap.lin_code is not None:
        updater.add(lambda m: m.geomap.lin_code.weight, "model_iphi.fc_code.weight")
        updater.add(lambda m: m.geomap.lin_code.bias, "model_iphi.fc_code.bias")

    updater.add(lambda m: m.geomap.lin_no_code.weight, "model_iphi.fc_no_code.weight")
    updater.add(lambda m: m.geomap.lin_no_code.bias, "model_iphi.fc_no_code.bias")

    updater.add(lambda m: m.geomap.lin1.weight, "model_iphi.fc1.weight")
    updater.add(lambda m: m.geomap.lin1.bias, "model_iphi.fc1.bias")
    updater.add(lambda m: m.geomap.lin2.weight, "model_iphi.fc2.weight")
    updater.add(lambda m: m.geomap.lin2.bias, "model_iphi.fc2.bias")
    updater.add(lambda m: m.geomap.lin3.weight, "model_iphi.fc3.weight")
    updater.add(lambda m: m.geomap.lin3.bias, "model_iphi.fc3.bias")

    # 2. Lifting
    updater.add(lambda m: m.lifting.weights[0], "fc0.weight")
    updater.add(lambda m: m.lifting.biases[0], "fc0.bias")

    # 3. Input Spectral Conv (conv0)
    updater.add(
        lambda m: m.conv_in.conv.weights.weights[0],
        "conv0.weights1",
        transpose_axes=(1, 0, 2, 3)
    )
    updater.add(
        lambda m: m.conv_in.conv.weights.weights[1],
        "conv0.weights2",
        transpose_axes=(1, 0, 2, 3)
    )

    # 4. Intermediate FNO blocks and skips
    if model.fno_blocks is not None:
        n_blocks = len(model.fno_blocks.fno_layers)
        for i in range(n_blocks):
            # Conv weights: PyTorch [in, out, m1, m2] -> JAX [out, in, m1, m2]
            updater.add(
                lambda m, idx=i: m.fno_blocks.fno_layers[idx].spectral_conv.weights.weights[0],
                f"conv{i+1}.weights1",
                transpose_axes=(1, 0, 2, 3)
            )
            updater.add(
                lambda m, idx=i: m.fno_blocks.fno_layers[idx].spectral_conv.weights.weights[1],
                f"conv{i+1}.weights2",
                transpose_axes=(1, 0, 2, 3)
            )
            # Skip: w1, w2, w3 -> local_operator
            updater.add(
                lambda m, idx=i: m.fno_blocks.fno_layers[idx].local_operator.conv.weight,
                f"w{i+1}.weight",
                squeeze_axes=-1
            )
            updater.add(
                lambda m, idx=i: m.fno_blocks.fno_layers[idx].local_operator.conv.bias,
                f"w{i+1}.bias"
            )

    # 5. Output Spectral Conv
    updater.add(
        lambda m: m.conv_out.conv.weights.weights[0],
        "conv4.weights1",
        transpose_axes=(1, 0, 2, 3)
    )
    updater.add(
        lambda m: m.conv_out.conv.weights.weights[1],
        "conv4.weights2",
        transpose_axes=(1, 0, 2, 3)
    )

    # 6. Coordinate Projectors (b0, b1, b2, b3, b4)
    if model.coord_projectors is not None:
        # b0..b3 (nn.Conv2d) -> coord_projectors[0..3]
        for i in range(len(model.coord_projectors) - 1):
            updater.add(
                lambda m, idx=i: m.coord_projectors[idx].weights[0],
                f"b{i}.weight",
                squeeze_axes=(-1, -2)
            )
            updater.add(
                lambda m, idx=i: m.coord_projectors[idx].biases[0],
                f"b{i}.bias"
            )
        # b4 (nn.Conv1d) -> coord_projectors[-1]
        updater.add(
            lambda m: m.coord_projectors[-1].weights[0],
            f"b{len(model.coord_projectors)-1}.weight",
            squeeze_axes=-1
        )
        updater.add(
            lambda m: m.coord_projectors[-1].biases[0],
            f"b{len(model.coord_projectors)-1}.bias"
        )

    # 7. Projection
    updater.add(lambda m: m.projection.weights[0], "fc1.weight")
    updater.add(lambda m: m.projection.biases[0], "fc1.bias")
    updater.add(lambda m: m.projection.weights[1], "fc2.weight")
    updater.add(lambda m: m.projection.biases[1], "fc2.bias")

    return updater.apply()
