#!/usr/bin/env python3
"""
H21b: Balance Ratio Predicts When Direction Matters
=====================================================

FROM 2.2d: At c=1 (balanced), Muon beats SGD by 25-65x even at optimal LR.
At c=100 (unbalanced), Muon's ONLY value is per-layer LR normalization.

HYPOTHESIS:
  The directional value of Muon's polar factor (SV equalization) is
  CONDITIONAL on the network being approximately balanced. Specifically:
  Muon's directional advantage (after controlling for per-layer LR)
  correlates with max(||W_l||_F) / min(||W_l||_F) being BELOW a threshold.

  When layers are balanced, all gradients are comparable in magnitude and
  the SV structure carries signal about the loss landscape geometry. When
  layers are imbalanced, gradient magnitudes vary by orders of magnitude
  and the SV structure of small-gradient layers is dominated by noise.

PROTOCOL:
  4-layer deep linear 32x32. Sweep balance ratios:
    - c in {1, 1.5, 2, 3, 5, 10, 30, 100} (applied as W0 *= c, W3 /= c)
  For each c:
    1. Muon (single LR sweep)
    2. SGD + oracle per-layer LR (grid search)
    3. NormSGD per-layer (G/||G||_F per layer, single LR)
  Compute residual_directional_advantage = loss_normsgd / loss_muon
  This isolates the directional value (both have normalized magnitudes).

  Plot residual_directional_advantage vs balance_ratio (= c).
  Find threshold c_thresh where advantage drops below 2x.

KEY PREDICTION:
  - At c=1: residual advantage ~ 5-20x (direction matters a lot)
  - At c>50: residual advantage < 2x (direction barely matters)
  - Monotone decrease as c increases
"""

import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DIM = 32
N_LAYERS = 4
NUM_STEPS = 300
MOMENTUM = 0.9
NS_ITERS = 5
NUM_SEEDS = 5
BATCH_SIZE = 64

C_VALUES = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0]

LR_MUON = np.logspace(-4, -1, 10)
LR_NORM = np.logspace(-3, 0, 10)


def newton_schulz(M, n_iters=NS_ITERS):
    norm = np.linalg.norm(M, 'fro')
    if norm < 1e-15:
        return M
    X = M / norm
    for _ in range(n_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


def init_weights(seed, c):
    rng = np.random.RandomState(seed)
    weights = [np.eye(DIM) + rng.randn(DIM, DIM) * 0.1 for _ in range(N_LAYERS)]
    weights[0] = weights[0] * c
    weights[-1] = weights[-1] / c
    return weights


def forward(weights, X):
    out = X.copy()
    for W in weights:
        out = W @ out
    return out


def compute_loss(weights, X, Y):
    pred = forward(weights, X)
    return 0.5 * np.mean(np.sum((pred - Y)**2, axis=0))


def compute_gradients(weights, X, Y):
    L = len(weights)
    N = X.shape[1]
    acts = [X.copy()]
    for W in weights:
        acts.append(W @ acts[-1])
    delta = (acts[-1] - Y) / N
    grads = [None] * L
    for l in range(L - 1, -1, -1):
        grads[l] = delta @ acts[l].T
        if l > 0:
            delta = weights[l].T @ delta
    return grads


def make_data(seed):
    rng = np.random.RandomState(seed)
    W_target = rng.randn(DIM, DIM) * 0.5
    X = rng.randn(DIM, BATCH_SIZE) * 0.3
    Y = W_target @ X
    return X, Y


def train(w0, X, Y, lr, opt):
    weights = [W.copy() for W in w0]
    mom = [np.zeros_like(W) for W in weights]
    for step in range(NUM_STEPS):
        if compute_loss(weights, X, Y) > 1e10:
            return float('inf')
        grads = compute_gradients(weights, X, Y)
        for i in range(N_LAYERS):
            if opt == 'muon':
                mom[i] = MOMENTUM * mom[i] + newton_schulz(grads[i])
            elif opt == 'normsgd':
                nrm = np.linalg.norm(grads[i], 'fro')
                mom[i] = MOMENTUM * mom[i] + grads[i] / max(nrm, 1e-15)
            weights[i] -= lr * mom[i]
    return compute_loss(weights, X, Y)


def main():
    seeds = [42 + i * 137 for i in range(NUM_SEEDS)]

    print("=" * 100)
    print("H21b: BALANCE RATIO PREDICTS DIRECTIONAL VALUE")
    print("=" * 100)
    print(f"c values: {C_VALUES}")
    print()

    results = {}

    for c in C_VALUES:
        print(f"\n  c={c:.1f}")
        best = {}

        for opt, grid in [('muon', LR_MUON), ('normsgd', LR_NORM)]:
            best_lr, best_loss = grid[-1], float('inf')
            for lr in grid:
                losses = [train(init_weights(s+5000, c), *make_data(s), lr, opt)
                          for s in seeds[:3]]
                ml = np.mean([l for l in losses if np.isfinite(l)] or [float('inf')])
                if ml < best_loss:
                    best_loss = ml
                    best_lr = lr
            best[opt] = best_lr

        # Full eval
        muon_losses = [train(init_weights(s+5000, c), *make_data(s), best['muon'], 'muon')
                       for s in seeds]
        norm_losses = [train(init_weights(s+5000, c), *make_data(s), best['normsgd'], 'normsgd')
                       for s in seeds]

        muon_mean = np.mean([l for l in muon_losses if np.isfinite(l)] or [float('inf')])
        norm_mean = np.mean([l for l in norm_losses if np.isfinite(l)] or [float('inf')])

        residual_adv = norm_mean / max(muon_mean, 1e-30)
        balance_ratio = c

        # Also measure actual layer norm ratios
        w = init_weights(seeds[0] + 5000, c)
        norms = [np.linalg.norm(W, 'fro') for W in w]
        actual_ratio = max(norms) / min(norms)

        results[c] = {
            'muon': muon_mean, 'normsgd': norm_mean,
            'residual_adv': residual_adv, 'actual_ratio': actual_ratio,
        }
        print(f"    Muon={muon_mean:.4e}  NormSGD={norm_mean:.4e}  "
              f"Residual directional adv: {residual_adv:.1f}x  "
              f"Actual ||W||_F ratio: {actual_ratio:.1f}")

    # =========================================================================
    # RESULTS
    # =========================================================================
    print(f"\n\n{'='*100}")
    print("RESULTS: RESIDUAL DIRECTIONAL ADVANTAGE vs BALANCE")
    print(f"{'='*100}")

    print(f"\n  {'c':>6}  {'||W|| ratio':>12}  {'Residual adv':>14}  {'Direction matters?':>20}")
    print("  " + "-" * 56)
    c_thresh = None
    for c in C_VALUES:
        r = results[c]
        matters = "YES" if r['residual_adv'] > 2.0 else "MARGINAL" if r['residual_adv'] > 1.2 else "NO"
        if c_thresh is None and r['residual_adv'] < 2.0:
            c_thresh = c
        print(f"  {c:>6.1f}  {r['actual_ratio']:>12.1f}  {r['residual_adv']:>14.1f}x  {matters:>20}")

    print(f"\n  Threshold c (directional advantage < 2x): {c_thresh if c_thresh else '>100'}")

    # Correlation
    cs = np.array(C_VALUES)
    advs = np.array([results[c]['residual_adv'] for c in C_VALUES])
    log_cs = np.log(cs)
    log_advs = np.log(np.clip(advs, 1e-10, None))
    r = np.corrcoef(log_cs, log_advs)[0, 1]
    print(f"  Correlation log(c) vs log(residual_adv): r = {r:.3f}")

    t1 = r < -0.5
    print(f"\n  T1: Negative correlation (r < -0.5)?  --> {'PASS' if t1 else 'FAIL'}  (r={r:.3f})")
    t2 = results[1.0]['residual_adv'] > 3 * results[100.0]['residual_adv']
    print(f"  T2: c=1 adv > 3x c=100 adv?           --> {'PASS' if t2 else 'FAIL'}")

    print(f"\n{'='*100}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()
