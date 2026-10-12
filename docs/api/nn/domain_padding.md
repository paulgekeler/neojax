## Domain Padding

Fast Fourier Transforms (FFT) assume periodic boundary conditions. When learning on non-periodic domains, `DomainPadding` pads the domain before the spectral convolutions and unpads it afterward, severely mitigating boundary artifacts.

::: neojax.nn.domain_padding.DomainPadding