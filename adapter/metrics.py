"""Generalized crop-disease composition metrics over valid pairs.

Candidate pairs are the training (seen) compositions followed by the test_unseen compositions,
the same candidate set the MultiTask baselines score against. Seen pairs come first, so a pair
index < num_seen is a seen composition.
"""
from __future__ import annotations
import numpy as np


def log_probs(logits):
    x = np.array(logits, dtype=np.float64)  # copy: don't mutate the caller's array
    x -= x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


def _frame_pairs(frame):
    return set(zip(frame.crop_label.astype(int).tolist(), frame.disease_label.astype(int).tolist()))


def build_pairs(train_frame, unseen_frame):
    """Return (pairs, num_seen): sorted train pairs, then sorted unseen pairs not seen in train."""
    seen = sorted(_frame_pairs(train_frame))
    unseen = sorted(_frame_pairs(unseen_frame) - set(seen))
    return seen + unseen, len(seen)


def pair_score_matrix(crop_logits, disease_logits, pairs):
    """Return B x P scores: log p(crop) + log p(disease) for each candidate pair."""
    p = np.asarray(pairs, dtype=np.int64)
    return log_probs(crop_logits)[:, p[:, 0]] + log_probs(disease_logits)[:, p[:, 1]]


def split_metrics(crop_logits, disease_logits, frame, pairs, unseen_pairs=()):
    """Composition metrics for one test split. Images whose pair is not a candidate are skipped."""
    crop_gt = frame.crop_label.to_numpy(dtype=np.int64)
    dis_gt = frame.disease_label.to_numpy(dtype=np.int64)
    index = {pair: i for i, pair in enumerate(pairs)}
    targets = np.array([index.get(pair, -1) for pair in zip(crop_gt.tolist(), dis_gt.tolist())],
                       dtype=np.int64)
    keep = targets >= 0
    crop_gt, dis_gt, targets = crop_gt[keep], dis_gt[keep], targets[keep]
    scores = pair_score_matrix(np.asarray(crop_logits)[keep], np.asarray(disease_logits)[keep], pairs)

    order = np.argsort(-scores, axis=1)[:, :min(5, scores.shape[1])]
    top1 = float((order[:, 0] == targets).mean()) if len(targets) else float("nan")
    top5 = float((order == targets[:, None]).any(axis=1).mean()) if len(targets) else float("nan")
    pair_arr = np.asarray(pairs, dtype=np.int64)
    pred_crop, pred_dis = pair_arr[order[:, 0], 0], pair_arr[order[:, 0], 1]
    pair_acc = {}
    for pair in set(zip(crop_gt.tolist(), dis_gt.tolist())):
        mask = (crop_gt == pair[0]) & (dis_gt == pair[1])
        pair_acc[pair] = float(((pred_crop[mask] == pair[0]) & (pred_dis[mask] == pair[1])).mean())
    balanced = float(np.mean([pair_acc[p] for p in unseen_pairs if p in pair_acc])) if unseen_pairs else float("nan")
    return {"n_images": int(keep.sum()), "n_skipped": int((~keep).sum()),
            "top1": top1, "top5": top5,
            "crop_acc": float((pred_crop == crop_gt).mean()) if len(targets) else float("nan"),
            "disease_acc": float((pred_dis == dis_gt).mean()) if len(targets) else float("nan"),
            "class_balanced_acc": balanced}, scores, targets


def harmonic_mean(a, b):
    return 0.0 if a + b == 0 else 2.0 * a * b / (a + b)


def ausuc(seen_scores, seen_targets, unseen_scores, unseen_targets, num_seen, n_points=201):
    """Sweep a constant bias added to the unseen-pair columns (index >= num_seen)."""
    if seen_scores.shape[1] <= num_seen:
        return {"auc": float("nan"), "best_hm": float("nan"), "best_bias": 0.0,
                "seen_at_best": float("nan"), "unseen_at_best": float("nan")}
    all_gaps = np.concatenate([
        seen_scores[:, :num_seen].max(1) - seen_scores[:, num_seen:].max(1),
        unseen_scores[:, :num_seen].max(1) - unseen_scores[:, num_seen:].max(1),
    ])
    biases = np.unique(np.concatenate([np.quantile(all_gaps, np.linspace(0, 1, n_points)),
                                       [0.0, all_gaps.min() - 1e-8, all_gaps.max() + 1e-8]]))
    seen_acc, unseen_acc = [], []
    for bias in biases:
        ss, us = seen_scores.copy(), unseen_scores.copy()
        ss[:, num_seen:] += bias
        us[:, num_seen:] += bias
        seen_acc.append(float((ss.argmax(1) == seen_targets).mean()))
        unseen_acc.append(float((us.argmax(1) == unseen_targets).mean()))
    seen_acc, unseen_acc = np.asarray(seen_acc), np.asarray(unseen_acc)
    hms = np.array([harmonic_mean(s, u) for s, u in zip(seen_acc, unseen_acc)])
    best = int(hms.argmax())
    # Seen acc ascending; ties broken by unseen acc descending, so the curve is the
    # staircase traced by the bias sweep (plain argsort orders ties arbitrarily).
    order = np.lexsort((-unseen_acc, seen_acc))
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # trapz removed in NumPy 2.4
    auc = trapezoid(unseen_acc[order], seen_acc[order])
    return {"auc": float(auc), "best_hm": float(hms[best]), "best_bias": float(biases[best]),
            "seen_at_best": float(seen_acc[best]), "unseen_at_best": float(unseen_acc[best])}
