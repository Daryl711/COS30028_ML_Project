"""Post-hoc logit calibration (v2) on saved test logits. No retraining, works for any backbone.

  python baseline/MultiTaskBaseline/calibrate.py --eval_dirs outputs/multitask_dinov2_50ep/eval

Reads <eval_dir>/logits.npz (written by evaluate.py) and writes, in the same folder:
  calibration_v2.json        uncalibrated, bias-only, held-out (cross-fit) and all-test results + chosen config
  cross_fit_v2.csv           every cross-fit fold
  calibration_search_v2.csv  all searched configs, tuned on all test images

Same method, grid and cross-fit seeds as the EfficientNet-B0 v2 notebook
(baseline/SingleTaskBaseline/FineTunedB0(Logit+HyperparameterSearch)), so results are comparable across models.
Each pair (c, d) is scored as

  log_softmax(crop_logits / T_c - tau_c * log prior_crop)[c]
+ log_softmax(disease_logits / T_d - tau_d * log prior_disease)[d]
- tau_pair * log prior_pair(c, d)   (seen pairs only)
+ bias                              (unseen pairs only)

REPORT the held-out (cross-fit) HM: settings are tuned on one half of the test images and scored on the other
half. The all-test config is tuned on the same images it is scored on, so it is optimistic.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd

GRID = dict(
    tau_c=[0.0, 0.5, 1.0],
    tau_d=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
    T_c=[0.7, 1.0, 1.5],
    T_d=[0.7, 1.0, 1.5, 2.0],
    tau_pair=[0.0, 0.25, 0.5, 1.0],
)
KNOBS = list(GRID)
CROSS_FIT_SEEDS = (0, 1)   # same as the B0 v2 run


# ----------------------------------------------------------------------------- basics
def log_softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


def harmonic_mean(a, b):
    return 0.0 if a + b == 0 else 2 * a * b / (a + b)


def topk(sc, gt, k):
    top = np.argpartition(-sc, k - 1, axis=1)[:, :k]
    return float((top == gt[:, None]).any(1).mean())


def add_bias(sc, num_seen, b):
    s = sc.copy()
    s[:, num_seen:] += b
    return s


# ----------------------------------------------------------------------------- priors
def compute_priors(d):
    """Train-set frequencies with add-one smoothing: crop, disease, and seen pair."""
    nc, nd = d["seen_crop"].shape[1], d["seen_disease"].shape[1]
    tc, td = d["train_crop"], d["train_disease"]
    cc = np.bincount(tc, minlength=nc).astype(np.float64)
    cd = np.bincount(td, minlength=nd).astype(np.float64)
    num_seen = int(d["num_seen"])
    idx = {tuple(p): i for i, p in enumerate(d["pairs"][:num_seen].tolist())}
    cp = np.zeros(num_seen)
    for p in zip(tc.tolist(), td.tolist()):
        cp[idx[p]] += 1
    return ((cc + 1) / (cc.sum() + nc), (cd + 1) / (cd.sum() + nd), (cp + 1) / (cp.sum() + num_seen))


# ----------------------------------------------------------------------------- scoring
class Scorer:
    """Composition scores for one set of images, caching each head's log-probs per (T, tau)."""

    def __init__(self, crop, disease, pairs, num_seen, priors):
        self.crop, self.disease = crop, disease
        self.pc, self.pd = pairs[:, 0], pairs[:, 1]
        self.num_seen = num_seen
        self.lp_c, self.lp_d, self.lp_p = (np.log(p) for p in priors)
        self._cache = {}

    def _head(self, name, T, tau):
        key = (name, T, tau)
        if key not in self._cache:
            logits, lp, cols = ((self.crop, self.lp_c, self.pc) if name == "crop"
                                else (self.disease, self.lp_d, self.pd))
            # T first, then the logit adjustment, so T and tau are independent
            self._cache[key] = log_softmax(logits / T - tau * lp[None, :])[:, cols]
        return self._cache[key]

    def scores(self, T_c, T_d, tau_c, tau_d, tau_pair):
        sc = self._head("crop", T_c, tau_c) + self._head("disease", T_d, tau_d)
        if tau_pair:
            sc = sc.copy()
            sc[:, : self.num_seen] -= tau_pair * self.lp_p[None, :]
        return sc


def split_stats(sc, gt, num_seen):
    s, u = sc[:, :num_seen], sc[:, num_seen:]
    gap = s.max(1) - u.max(1)                       # an image is predicted unseen iff bias > gap
    return gap, s.argmax(1) == gt, u.argmax(1) + num_seen == gt


def acc_at_biases(gap, cs, cu, biases):
    """Exact accuracy for every bias in O(n log n)."""
    o = np.argsort(gap)
    g = gap[o]
    cu_c = np.concatenate([[0], np.cumsum(cu[o])])
    cs_c = np.concatenate([[0], np.cumsum(cs[o])])
    k = np.searchsorted(g, biases, side="left")
    return (cu_c[k] + (cs_c[-1] - cs_c[k])) / len(gap)


# ----------------------------------------------------------------------------- search
def search(S, U, gt_s, gt_u, grid=GRID, n_bias=600, keep=40):
    """Grid search. S, U: Scorers for the seen / unseen images being tuned on. Returns a ranked DataFrame."""
    num_seen = S.num_seen
    rows = []
    for vals in itertools.product(*[grid[n] for n in KNOBS]):
        cfg = dict(zip(KNOBS, vals))
        gs, css, cus = split_stats(S.scores(**cfg), gt_s, num_seen)
        gu, csu, cuu = split_stats(U.scores(**cfg), gt_u, num_seen)
        pooled = np.concatenate([gs, gu])
        b = np.unique(np.concatenate([np.quantile(pooled, np.linspace(0, 1, n_bias)), [0.0]]))
        sa, ua = acc_at_biases(gs, css, cus, b), acc_at_biases(gu, csu, cuu, b)
        hm = 2 * sa * ua / np.clip(sa + ua, 1e-12, None)
        j = int(hm.argmax())
        rows.append(dict(**cfg, bias=float(b[j]), seen=float(sa[j]), unseen=float(ua[j]), hm=float(hm[j])))
    df = pd.DataFrame(rows).sort_values("hm", ascending=False).reset_index(drop=True)
    # stage 2: top-5 HM for the best `keep` configs, used to break near-ties
    t5 = []
    for _, r in df.head(keep).iterrows():
        cfg = {n: r[n] for n in KNOBS}
        t5.append(harmonic_mean(topk(add_bias(S.scores(**cfg), num_seen, r.bias), gt_s, 5),
                                topk(add_bias(U.scores(**cfg), num_seen, r.bias), gt_u, 5)))
    df.loc[: keep - 1, "hm_top5"] = t5
    return df


def pick(df, tol=0.005):
    """Best top-1 HM; configs within `tol` of it are ranked by top-5 HM."""
    top = df.head(len(df.dropna(subset=["hm_top5"])))
    near = top[top.hm >= top.hm.max() - tol]
    return near.sort_values("hm_top5", ascending=False).iloc[0]


def evaluate_cfg(S, U, gt_s, gt_u, cfg):
    c = {k: float(cfg[k]) for k in KNOBS}
    sc_s = add_bias(S.scores(**c), S.num_seen, float(cfg["bias"]))
    sc_u = add_bias(U.scores(**c), U.num_seen, float(cfg["bias"]))
    r = dict(seen1=float((sc_s.argmax(1) == gt_s).mean()), unseen1=float((sc_u.argmax(1) == gt_u).mean()),
             seen5=topk(sc_s, gt_s, 5), unseen5=topk(sc_u, gt_u, 5))
    r["hm1"] = harmonic_mean(r["seen1"], r["unseen1"])
    r["hm5"] = harmonic_mean(r["seen5"], r["unseen5"])
    return r


def stratified_halves(gt, seed):
    rng = np.random.default_rng(seed)
    a, b = [], []
    for c in np.unique(gt):
        i = np.where(gt == c)[0]
        rng.shuffle(i)
        h = len(i) // 2
        a += i[:h].tolist()
        b += i[h:].tolist()
    return np.array(sorted(a)), np.array(sorted(b))


def calibrate(d, seeds=CROSS_FIT_SEEDS, log=print):
    pairs, num_seen = d["pairs"], int(d["num_seen"])
    priors = compute_priors(d)
    gt_s, gt_u = d["seen_gt"], d["unseen_gt"]

    def scorers(i_s, i_u):
        return (Scorer(d["seen_crop"][i_s], d["seen_disease"][i_s], pairs, num_seen, priors),
                Scorer(d["unseen_crop"][i_u], d["unseen_disease"][i_u], pairs, num_seen, priors))

    all_s, all_u = np.arange(len(gt_s)), np.arange(len(gt_u))
    S, U = scorers(all_s, all_u)
    none = dict(T_c=1.0, T_d=1.0, tau_c=0.0, tau_d=0.0, tau_pair=0.0)

    # uncalibrated and bias-only (all test images), as reference points
    uncal = evaluate_cfg(S, U, gt_s, gt_u, {**none, "bias": 0.0})
    bias_only = search(S, U, gt_s, gt_u, grid={k: [v] for k, v in none.items()})
    bias_only = {**evaluate_cfg(S, U, gt_s, gt_u, pick(bias_only)), "bias": float(bias_only.bias[0])}

    # held-out estimate: tune on one stratified half, score on the other (both directions, each seed)
    folds = []
    for sd in seeds:
        sA, sB = stratified_halves(gt_s, sd)
        uA, uB = stratified_halves(gt_u, sd)
        for ts, tu, es, eu in ((sA, uA, sB, uB), (sB, uB, sA, uA)):
            St, Ut = scorers(ts, tu)
            best = pick(search(St, Ut, gt_s[ts], gt_u[tu]))
            Se, Ue = scorers(es, eu)
            folds.append(dict(seed=sd, **{k: float(best[k]) for k in KNOBS + ["bias"]}, tuned_hm=float(best.hm),
                              **evaluate_cfg(Se, Ue, gt_s[es], gt_u[eu], best)))
            log(f"  fold {len(folds)}/{2 * len(seeds)}: held-out HM top-1 {folds[-1]['hm1']:.4f}")
    folds = pd.DataFrame(folds)
    heldout = {k: float(folds[k].mean()) for k in ("seen1", "unseen1", "hm1", "seen5", "unseen5", "hm5")}

    # final config, tuned on all test images
    full = search(S, U, gt_s, gt_u)
    best = pick(full)
    result = {
        "uncalibrated": uncal,
        "bias_only_all_test": bias_only,
        "heldout_cross_fit": heldout,
        "all_test": {"config": {k: float(best[k]) for k in KNOBS + ["bias"]},
                     "metrics": evaluate_cfg(S, U, gt_s, gt_u, best)},
        "cross_fit_seeds": list(seeds),
        "note": "Report heldout_cross_fit.hm1. all_test is tuned on the images it is scored on (optimistic).",
    }
    return result, folds, full


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dirs", nargs="+", required=True, help="folders containing logits.npz")
    args = p.parse_args()
    for out in args.eval_dirs:
        path = os.path.join(out, "logits.npz")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path} not found. Run evaluate.py first; it writes logits.npz.")
        print(f"[calibrate] {out}")
        d = dict(np.load(path))
        result, folds, full = calibrate(d)
        with open(os.path.join(out, "calibration_v2.json"), "w") as f:
            json.dump(result, f, indent=2)
        folds.to_csv(os.path.join(out, "cross_fit_v2.csv"), index=False)
        full.to_csv(os.path.join(out, "calibration_search_v2.csv"), index=False)

        u, b, h, a = (result["uncalibrated"], result["bias_only_all_test"], result["heldout_cross_fit"],
                      result["all_test"])
        print(f"                              seen-1   unseen-1   HM-1     HM-5")
        print(f"  uncalibrated                {u['seen1']:.4f}   {u['unseen1']:.4f}    {u['hm1']:.4f}   {u['hm5']:.4f}")
        print(f"  bias only (all test)        {b['seen1']:.4f}   {b['unseen1']:.4f}    {b['hm1']:.4f}   {b['hm5']:.4f}")
        print(f"  logit-adj, held-out         {h['seen1']:.4f}   {h['unseen1']:.4f}    {h['hm1']:.4f}   {h['hm5']:.4f}"
              "   <- report this")
        m = a["metrics"]
        print(f"  logit-adj, all test         {m['seen1']:.4f}   {m['unseen1']:.4f}    {m['hm1']:.4f}   {m['hm5']:.4f}")
        print(f"  all-test config: {a['config']}\n")


if __name__ == "__main__":
    main()
