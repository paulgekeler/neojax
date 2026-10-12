## General

Training Neural Operators effectively requires evaluating the error in function space rather than just on discrete grid points. Standard ML metrics (like basic MSE) are discretization-dependent. To ensure our models are resolution-invariant and respect the underlying continuous functions, `neojax` provides discretization-invariant, norm-based metrics.