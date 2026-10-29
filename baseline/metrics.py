"""Evaluation: Top-N accuracy, composition scoring, harmonic mean, calibration, confusion matrices."""
from __future__ import annotations

import numpy as np


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


def topk_accuracy(scores: np.ndarray, targets: np.ndarray, ks=(1, 5)) -> dict[str, float]:
    if len(targets) == 0:
        return {f"top{k}": float("nan") for k in ks}
    order = np.argsort(-scores, axis=1)
    out = {}
    for k in ks:
        kk = min(k, scores.shape[1])
        out[f"top{k}"] = float((order[:, :kk] == targets[:, None]).any(axis=1).mean())
    return out


def composition_scores(crop_logits: np.ndarray, disease_logits: np.ndarray,
                       pairs: list[tuple[int, int]]) -> np.ndarray:
    """score(c, d | x) = log p(c|x) + log p(d|x), only over valid crop-disease pairs.

    This is how two independent heads can rank a composition they never saw together.
    """
    p = np.asarray(pairs)
    return log_softmax(crop_logits)[:, p[:, 0]] + log_softmax(disease_logits)[:, p[:, 1]]


def add_unseen_bias(scores: np.ndarray, num_seen: int, bias: float) -> np.ndarray:
    s = scores.copy()
    s[:, num_seen:] += bias
    return s


def harmonic_mean(a: float, b: float) -> float:
    return 0.0 if a + b == 0 else 2 * a * b / (a + b)


def _trapezoid(y, x):
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(y, x))


def calibration_sweep(seen_scores, seen_gt, unseen_scores, unseen_gt, num_seen, n_points=201):
    """Add a scalar bias to unseen-pair scores and sweep it (standard in CZSL papers).

    Models are biased toward seen classes; this traces the seen/unseen accuracy trade-off.
    Bias = 0 is the uncalibrated model.
    """
    allsc = np.concatenate([seen_scores, unseen_scores])
    gaps = allsc[:, :num_seen].max(1) - allsc[:, num_seen:].max(1)
    biases = np.unique(np.concatenate([
        np.quantile(gaps, np.linspace(0, 1, n_points)),
        [0.0, gaps.min() - 1e-3, gaps.max() + 1e-3],
    ]))
    seen_acc, unseen_acc = [], []
    for b in biases:
        seen_acc.append(float((add_unseen_bias(seen_scores, num_seen, b).argmax(1) == seen_gt).mean()))
        unseen_acc.append(float((add_unseen_bias(unseen_scores, num_seen, b).argmax(1) == unseen_gt).mean()))
    seen_acc, unseen_acc = np.array(seen_acc), np.array(unseen_acc)
    hms = np.array([harmonic_mean(s, u) for s, u in zip(seen_acc, unseen_acc)])
    best = int(hms.argmax())
    order = np.argsort(seen_acc)
    return {
        "biases": biases, "seen_acc": seen_acc, "unseen_acc": unseen_acc, "hm": hms,
        "best_hm": float(hms[best]), "best_bias": float(biases[best]),
        "seen_at_best": float(seen_acc[best]), "unseen_at_best": float(unseen_acc[best]),
        "auc": _trapezoid(unseen_acc[order], seen_acc[order]),
    }
