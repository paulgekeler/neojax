# Scales Reference

In SciML, physical quantities often span different magnitudes. Feeding raw physical parameters or unscaled grid coordinates directly into a Neural Operator can lead to numerical instability, slow convergence, or poor generalization. 

To address this, `neojax` provides a collection of **Scales**. 

**Scales vs. Normalizers**: Unlike statistical normalizers (which rely on dataset statistics like mean, variance, or min/max values), Scales apply transformations based on *physical invariants*, *domain characteristics*, or *grid properties*. This ensures that the transformed inputs are non-dimensionalized while retaining their underlying physical relationships.

## Using Scales with PhysicsNormalizer

Scales in `neojax` are composable. Instead of applying a single scale manually, you typically pass one or more scales into a `PhysicsNormalizer`. The `PhysicsNormalizer` evaluates all provided scales, computes their element-wise product, and uses the combined factor to non-dimensionalize your data.

Here is an example of how you might compose multiple physical scales for a fluid dynamics problem:

```python
import jax.numpy as jnp
from neojax.data.scales import CharacteristicLengthScale, ReynoldsScale
from neojax.data.normalizers import PhysicsNormalizer

# Define physical scales
length_scale = CharacteristicLengthScale(L_ref=1.0)
re_scale = ReynoldsScale(U=10.0, L=1.0, nu=1e-3)

# Compose them into a PhysicsNormalizer
physics_norm = PhysicsNormalizer(length_scale, re_scale)

# Compute the combined scale product on your input data
data = jnp.ones((3, 64, 64))
physics_norm = physics_norm.compute_stats(data)

# Non-dimensionalize the data for model input
non_dimensional_data = physics_norm(data)

# You can safely revert the transformation back to physical units during inference
physical_data = physics_norm.inverse_transform(non_dimensional_data)
```

## Available Scales

Depending on your underlying PDE or physical system, you can choose from several scaling strategies or implement your own:

- **`PhysicalScale`**: A Python `Protocol` that defines the interface for all physical scaling logic. You can implement custom non-dimensionalization by inheriting from `equinox.Module` and implementing the `get_scale(self, x: Array)` method.
- **`CharacteristicLengthScale`**: Standardizes spatial domains by mapping physical dimensions against a constant reference length scale ($L$).
- **`ReynoldsScale`**: Useful for fluid dynamics (e.g., Navier-Stokes equations). It scales parameters according to the Reynolds number ($Re = UL / \nu$), capturing the ratio of inertial to viscous forces.
- **`GridBasedScale`**: Applies spatially-varying scaling factors. Useful when the scaling depends on the location in the domain, such as varying mesh densities or coordinate-dependent physics.

---

## Physical Scale

:::neojax.data.scales.PhysicalScale

## Characteristic Length Scale

:::neojax.data.scales.CharacteristicLengthScale

## Reynolds Scale

:::neojax.data.scales.ReynoldsScale

## GridBasedScale

:::neojax.data.scales.GridBasedScale
