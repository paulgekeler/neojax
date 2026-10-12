## Skip Connections

Local operators used alongside the global spectral convolutions. They process high-frequency, localized information and act as residual connections to stabilize training.

*Note: For the standard identity skip connection, `neojax` directly uses `equinox.nn.Identity` for simplicity.*

::: neojax.nn.skip_connections.SoftGating

---

::: neojax.nn.skip_connections.Flattened1dConv