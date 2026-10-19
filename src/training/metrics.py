"""metrics.py — Metrik window-level: F1-macro (primary) & ROC-AUC ovr."""

import torch
from sklearn.metrics import f1_score, roc_auc_score


def win_f1_macro_metric(preds: torch.Tensor, targs: torch.Tensor) -> float:
    """
    Window-level F1-macro — PRIMARY metric for both datasets (DESIGN-3).

    Recording-level accuracy is too coarse per fold (few recordings); window-level
    F1-macro uses all N windows for much finer statistical resolution, and punishes
    class-blind models that ignore minority fault classes.
    """
    preds_cls = preds.argmax(dim=-1).cpu().numpy().astype(int)
    labels = targs.cpu().numpy().astype(int)
    n_classes = preds.shape[-1]
    return float(f1_score(labels, preds_cls, average="macro",
                          labels=list(range(n_classes)), zero_division=0))


def roc_auc_metric(preds: torch.Tensor, targs: torch.Tensor) -> float:
    """Multi-class ROC-AUC (one-vs-rest) for AccumMetric."""
    probs = torch.softmax(preds, dim=-1).cpu().numpy()
    labels = targs.cpu().numpy().astype(int)
    n_classes = probs.shape[1]
    try:
        return float(
            roc_auc_score(labels, probs, multi_class="ovr", labels=list(range(n_classes)))
        )
    except ValueError:
        return float("nan")
