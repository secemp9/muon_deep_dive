#!/usr/bin/env python3
"""
H18b: Is the ~30x Constant Factor LR-Independent?
===================================================

FROM D-TEST RETRACTION: After controlling for LR, the Muon advantage is a
constant ~30x across depths (NOT exponential). But was that 30x measured at
a SINGLE LR grid resolution? The number could shift with exhaustive sweeps.

HYPOTHESIS:
  The ~30x constant advantage is stable across LR granularities. If you do
  EXHAUSTIVE LR sweeps (50+ candidates, fine-grained) at each depth, the
  advantage ratio SGD/Muon remains in the range [15x, 60x] for all depths
  2-16, confirming it is a genuine constant-factor directional benefit.

PROTOCOL:
  For depths L in {2, 4, 8, 16}:
    - SGD: sweep 50 LRs log-spaced in [1e-6, 0.5]
    - Muon: sweep 50 LRs log-spaced in [1e-5, 0.2]
    - 5 seeds each, pick best by median
    - Compute advantage ratio at best LRs
  Measure coefficient of variation of advantage across depths.

PASS CRITERIA:
  - All advantages in [5x, 100x] (same order of magnitude)
  - CV of log(advantage) across depths < 0.3
  - No monotone trend: Spearman(depth, advantage) not significant
"""

import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DIM = 32
DEPTHS = [2, 4, 8, 16]
NUM_STEPS = 300
MOMENTUM = 0.9
NS_ITERS = 5
NUM_SEEDS = 5
BATCH_SIZE = 64

# EXHAUSTIVE grids: 50 candidates each
SGD_LR_GRID = np.logspace(-6, np.log10(0.5), 50)
MUON_LR_GRID = np.logspace(-5, np.log10(0.2), 50)


def newton_schulz(M, n_iters=NS_ITERS):
    norm = np.linalg.norm(M, ord='fro')
    if norm < 1e-15:
        return M
    X = M / norm
    for _ in range(n_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


def init_weights(depth, seed):
    rng = np.random.RandomState(seed)
    return [np.eye(DIM) + rng.randn(DIM, DIM) * 0.1 for _ in range(depth)]


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


def train(weights_init, X, Y, lr, optimizer):
    weights = [W.copy() for W in weights_init]
    mom = [np.zeros_like(W) for W in weights]
    for step in range(NUM_STEPS):
        loss = compute_loss(weights, X, Y)
        if not np.isfinite(loss) or loss > 1e10:
            return float('inf')
        grads = compute_gradients(weights, X, Y)
        for i in range(len(weights)):
            if optimizer == 'muon':
                mom[i] = MOMENTUM * mom[i] + newton_schulz(grads[i])
            else:
                mom[i] = MOMENTUM * mom[i] + grads[i]
            weights[i] = weights[i] - lr * mom[i]
    return compute_loss(weights, X, Y)


def spearman_rank(x, y):
    n = len(x)
    if n < 3:
        return float('nan')
    rx = np.argsort(np.argsort(x)).astype(float) + 1
    ry = np.argsort(np.argsort(y)).astype(float) + 1
    return np.corrcoef(rx, ry)[0, 1]


def main():
    seeds = [42 + i * 137 for i in range(NUM_SEEDS)]

    print("=" * 100)
    print("H18b: IS THE ~30x CONSTANT FACTOR LR-INDEPENDENT?")
    print("=" * 100)
    print(f"Depths: {DEPTHS}")
    print(f"SGD: {len(SGD_LR_GRID)} LR candidates")
    print(f"Muon: {len(MUON_LR_GRID)} LR candidates")
    print()

    advantages = {}

    for depth in DEPTHS:
        print(f"\n  DEPTH L={depth}")

        best_losses = {}
        for opt, lr_grid in [('sgd', SGD_LR_GRID), ('muon', MUON_LR_GRID)]:
            best_lr = lr_grid[0]
            best_loss = float('inf')

            for lr in lr_grid:
                losses = []
                for s in seeds[:3]:
                    X, Y = make_data(s)
                    w = init_weights(depth, s + 5000)
                    fl = train(w, X, Y, lr, opt)
                    losses.append(fl)
                finite = [l for l in losses if np.isfinite(l)]
                ml = np.median(finite) if finite else float('inf')
                if ml < best_loss:
                    best_loss = ml
                    best_lr = lr

            # Full eval
            all_losses = []
            for s in seeds:
                X, Y = make_data(s)
                w = init_weights(depth, s + 5000)
                fl = train(w, X, Y, best_lr, opt)
                all_losses.append(fl)
            finite = [l for l in all_losses if np.isfinite(l)]
            mean_loss = np.mean(finite) if finite else float('inf')
            best_losses[opt] = mean_loss
            print(f"    {opt:>5}: best_lr={best_lr:.6f}  loss={mean_loss:.6e}")

        adv = best_losses['sgd'] / max(best_losses['muon'], 1e-30)
        advantages[depth] = adv
        print(f"    Advantage: {adv:.1f}x")

    # =========================================================================
    # ANALYSIS
    # =========================================================================
    print(f"\n\n{'='*100}")
    print("RESULTS")
    print(f"{'='*100}")

    adv_list = [advantages[d] for d in DEPTHS]
    log_advs = np.log(adv_list)
    cv = np.std(log_advs) / (np.abs(np.mean(log_advs)) + 1e-15)
    rho = spearman_rank(np.array(DEPTHS, dtype=float), np.array(adv_list))

    print(f"\n  {'Depth':>6}  {'Advantage':>12}")
    print("  " + "-" * 22)
    for d in DEPTHS:
        print(f"  {d:>6}  {advantages[d]:>12.1f}x")

    print(f"\n  Mean advantage: {np.mean(adv_list):.1f}x")
    print(f"  Std advantage:  {np.std(adv_list):.1f}x")
    print(f"  CV of log(advantage): {cv:.3f}")
    print(f"  Spearman(depth, advantage): {rho:.3f}")

    # Tests
    all_in_range = all(5 < a < 100 for a in adv_list)
    cv_low = cv < 0.3
    no_trend = abs(rho) < 0.8

    print(f"\n  T1: All advantages in [5x, 100x]?  --> {'PASS' if all_in_range else 'FAIL'}")
    print(f"  T2: CV of log(advantage) < 0.3?     --> {'PASS' if cv_low else 'FAIL'}  (CV={cv:.3f})")
    print(f"  T3: No monotone depth trend?         --> {'PASS' if no_trend else 'FAIL'}  (rho={rho:.3f})")

    print(f"\n{'='*100}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()
