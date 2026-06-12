# Components Reference

This page contains the API reference for the building blocks of Neural Operators in `neojax`. 

While you can easily use pre-built models like `FNO` or `TFNO`, these underlying components are designed to be highly modular, allowing you to construct entirely custom neural operator architectures using standard `equinox` composition.

## Spectral Convolution

These layers evaluate the continuous integral operator in Fourier space. They perform the core global operations that make Fourier Neural Operators discretization-invariant.

::: neojax.nn.spectral_conv.SpectralConvNd
::: neojax.nn.tucker_spectral_conv.TuckerSpectralConvNd

## FNO Blocks

The standard layer of the Fourier Neural Operator. An FNO Block computes the sum of the global spectral convolution and a local skip connection, followed by optional normalization and non-linear activation.
::: neojax.nn.fno_blocks.FNOBlocks
::: neojax.nn.fno_blocks.FNOBlock

## TFNO Blocks

Tucker-factorized FNO blocks. By factorizing the spectral weights into a core tensor and factor matrices, TFNO blocks significantly reduce the parameter count and memory footprint, especially for 3D or 4D problems.
::: neojax.nn.tfno_blocks.TFNOBlocks
::: neojax.nn.tfno_blocks.TFNOBlock

## Pointwise MLP

Applies a Multi-Layer Perceptron independently across the spatial grid points, operating solely on the channel dimension. These are used for lifting inputs to higher-dimensional latent spaces, projecting outputs, and channel-mixing within operator blocks.
::: neojax.nn.pointwise_mlp.PointwiseMLP

## Domain Padding

Fast Fourier Transforms (FFT) assume periodic boundary conditions. When learning on non-periodic domains, `DomainPadding` pads the domain before the spectral convolutions and unpads it afterward, severely mitigating boundary artifacts.
::: neojax.nn.domain_padding.DomainPadding

## Positional Embedding

Appends grid coordinate features (e.g., $(x, y)$ positions) to the input tensors channel dimension.
::: neojax.nn.positional_embedding.GridEmbeddingNd

## Skip Connections

Local operators used alongside the global spectral convolutions. They process high-frequency, localized information and act as residual connections to stabilize training.

*Note: For the standard identity skip connection, `neojax` directly uses `equinox.nn.Identity` for simplicity and seamless integration with the JAX/Equinox ecosystem.*

::: neojax.nn.skip_connections.SoftGating
::: neojax.nn.skip_connections.Flattened1dConv
