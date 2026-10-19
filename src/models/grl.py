"""grl.py — Gradient Reversal Layer (Ganin et al. 2016).

Forward: identitas. Backward: gradien dinegasikan & diskalakan -lambda_grl,
sehingga modul hilir (domain head) menyumbang objektif MAKSIMISASI ke backbone.
"""

import torch
from torch.autograd import Function


class _GRLFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_grl: float) -> torch.Tensor:  # type: ignore[override]
        ctx.lambda_grl = float(lambda_grl)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return -ctx.lambda_grl * grad_output, None


def gradient_reversal(x: torch.Tensor, lambda_grl: float = 1.0) -> torch.Tensor:
    """Apply Gradient Reversal Layer with strength lambda_grl."""
    return _GRLFunction.apply(x, lambda_grl)  # type: ignore[return-value]
