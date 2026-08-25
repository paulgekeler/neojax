# Components Reference

This page contains the API reference for the building blocks of Neural Operators in `neojax`.

The pre-built models (`FNO`, `TFNO`, ...) are built from these components. They can also be composed directly, using standard `equinox` composition, to construct custom neural operator architectures.

## Spectral Convolution

These layers evaluate the continuous integral operator in Fourier space. They perform the core global operations that make Fourier Neural Operators discretization-invariant.

::: neojax.nn.SpectralConvNd

::: neojax.nn.GeoSpectralConvNd

## FNO Blocks

The standard layer of the Fourier Neural Operator. An FNO Block computes the sum of the global spectral convolution and a local skip connection, followed by optional normalization and non-linear activation.

::: neojax.nn.FNOBlocks

::: neojax.nn.FNOBlock

## Pointwise MLP

Applies a Multi-Layer Perceptron independently across the spatial grid points, operating solely on the channel dimension. These are used for lifting inputs to higher-dimensional latent spaces, projecting outputs, and channel-mixing within operator blocks.

::: neojax.nn.PointwiseMLP

## Domain Padding

Fast Fourier Transforms (FFT) assume periodic boundary conditions. When learning on non-periodic domains, `DomainPadding` pads the domain before the spectral convolutions and unpads it afterward, severely mitigating boundary artifacts.

::: neojax.nn.DomainPadding

## Positional Embedding

Appends grid coordinate features (e.g., $(x, y)$ positions) to the input tensors channel dimension.

::: neojax.nn.GridEmbeddingNd

## Skip Connections

Local operators used alongside the global spectral convolutions. They process high-frequency, localized information and act as residual connections to stabilize training.

*Note: For the standard identity skip connection, `neojax` directly uses `equinox.nn.Identity` for simplicity and seamless integration with the JAX/Equinox ecosystem.*

::: neojax.nn.SoftGating

::: neojax.nn.Flattened1dConv

## Resampler

Resamples inputs using different interpolation methods. Useful for architecures like the UNO, where data needs to be explicitely up- and downsampled as it flows through the network. 

::: neojax.nn.Resampler

## Coordinate Diffeomorphism Maps

Used in geometry-aware operators (like `GeoFNO`) to learn a soft-diffeomorphism coordinate transformation from arbitrary physical grids/meshes to a regular latent grid.

::: neojax.nn.GeoMapNd