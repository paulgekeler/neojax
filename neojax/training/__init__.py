"""Training modules for neural operators."""

from neojax.training.diagnostics import (
    compute_grad_report,
    flag_anomalous_norms,
    grad_norms,
    select_by_path,
)
from neojax.training.trainer import Trainer, TrainState

__all__ = [
    "Trainer",
    "TrainState",
    "compute_grad_report",
    "flag_anomalous_norms",
    "grad_norms",
    "select_by_path",
]
