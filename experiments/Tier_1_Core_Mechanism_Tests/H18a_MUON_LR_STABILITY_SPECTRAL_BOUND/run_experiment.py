#!/usr/bin/env python3
"""
H18a: Muon LR Stability from Spectral-Norm-Bounded Updates
============================================================

FROM D-TEST RETRACTION: The exponential depth scaling was an LR confound.
The REAL depth story is LR robustness: Muon's optimal LR varies ~2x across
depths 2-16, while SGD's varies ~100x (dropping as O(1/L^2)).

HYPOTHESIS:
  Muon's LR stability comes from ||ortho(G)||_op = 1 always. The spectral
  norm of Muon's update is bounded by LR regardless of depth, while SGD's
  effective step size depends on ||G||_op which grows with the product of
  layer condition numbers.

  Specifically: SGD's max stable LR ~ 2 / (L * lambda_max(H)), where
  lambda_max grows with product-of-condition-numbers. For Muon, the update
  is on the Stiefel manifold with bounded operator norm, so the effective
  LR is independent of the per-layer spectrum.

PREDICTION:
  - Muon optimal LR vs depth: slope < 0.05 on log-log (nearly flat)
  - SGD optimal LR vs depth: slope ~ -2.0 on log-log (O(1/L^2))
  - The spectral norm of Muon's actual weight update ||delta_W||_op is
    constant across depths (at optimal LR), while SGD's varies by 100x+

PROTOCOL:
  Depths L in {2, 3, 4, 6, 8, 12, 16}. For each:
    1. Full LR sweep for both SGD and Muon (20 candidates each, log-spaced)
    2. Record optimal LR and ||delta_W||_op at step 1 at that optimal LR
    3. Fit log(optimal_LR) vs log(L) for both optimizers
  5 seeds each. 32x32 deep linear nets.

KEY MEASUREMENTS:
  - log-log slope of optimal_LR vs L for SGD and Muon
  - ||delta_W_l||_op at optimal LR for each layer at each depth
  - Ratio of max/min optimal LR across depths for each optimizer
"""

import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DIM = 32
DEPTHS = [2, 3, 4, 6, 8, 12, 16]
NUM_STEPS = 300
MOMENTUM = 0.9
NS_ITERS = 5
NUM_SEEDS = 5
BATCH_SIZE = 64

# Dense LR grids: 20 candidates each, log-spaced
SGD_LR_GRID = np.logspace(-5, -1, 20)
MUON_LR_GRID = np.logspace(-4, -1, 20)


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


def train(weights_init, X, Y, lr, optimizer, num_steps=NUM_STEPS):
    """Train and return (final_loss, step1_update_op_norms)."""
    weights = [W.copy() for W in weights_init]
    mom = [np.zeros_like(W) for W in weights]
    step1_op_norms = None

    for step in range(num_steps):
        loss = compute_loss(weights, X, Y)
        if not np.isfinite(loss) or loss > 1e10:
            return float('inf'), None
        grads = compute_gradients(weights, X, Y)
        deltas = []
        for i in range(len(weights)):
            if optimizer == 'muon':
                mom[i] = MOMENTUM * mom[i] + newton_schulz(grads[i])
            else:
                mom[i] = MOMENTUM * mom[i] + grads[i]
            delta = lr * mom[i]
            deltas.append(delta)
            weights[i] = weights[i] - delta

        if step == 0:
            step1_op_norms = [np.linalg.svd(d, compute_uv=False)[0] for d in deltas]

    return compute_loss(weights, X, Y), step1_op_norms


def main():
    seeds = [42 + i * 137 for i in range(NUM_SEEDS)]

    print("=" * 100)
    print("H18a: MUON LR STABILITY FROM SPECTRAL-NORM-BOUNDED UPDATES")
    print("=" * 100)
    print(f"Depths: {DEPTHS}")
    print(f"SGD LR grid: {len(SGD_LR_GRID)} candidates [{SGD_LR_GRID[0]:.1e} .. {SGD_LR_GRID[-1]:.1e}]")
    print(f"Muon LR grid: {len(MUON_LR_GRID)} candidates [{MUON_LR_GRID[0]:.1e} .. {MUON_LR_GRID[-1]:.1e}]")
    print()

    results = {}

    for depth in DEPTHS:
        print(f"\n{'='*80}")
        print(f"  DEPTH L={depth}")
        print(f"{'='*80}")

        for opt, lr_grid in [('sgd', SGD_LR_GRID), ('muon', MUON_LR_GRID)]:
            best_lr = lr_grid[0]
            best_loss = float('inf')

            for lr in lr_grid:
                losses = []
                for s in seeds[:3]:
                    X, Y = make_data(s)
                    w = init_weights(depth, s + 5000)
                    fl, _ = train(w, X, Y, lr, opt)
                    losses.append(fl)
                finite = [l for l in losses if np.isfinite(l)]
                ml = np.mean(finite) if finite else float('inf')
                if ml < best_loss:
                    best_loss = ml
                    best_lr = lr

            # Full evaluation at best LR
            all_losses = []
            all_op_norms = []
            for s in seeds:
                X, Y = make_data(s)
                w = init_weights(depth, s + 5000)
                fl, op_norms = train(w, X, Y, best_lr, opt)
                all_losses.append(fl)
                if op_norms is not None:
                    all_op_norms.append(op_norms)

            finite = [l for l in all_losses if np.isfinite(l)]
            mean_loss = np.mean(finite) if finite else float('inf')

            # Mean step-1 operator norm across seeds and layers
            if all_op_norms:
                mean_max_op = np.mean([max(norms) for norms in all_op_norms])
                mean_avg_op = np.mean([np.mean(norms) for norms in all_op_norms])
            else:
                mean_max_op = float('nan')
                mean_avg_op = float('nan')

            results[(depth, opt)] = {
                'best_lr': best_lr,
                'mean_loss': mean_loss,
                'mean_max_op_norm': mean_max_op,
                'mean_avg_op_norm': mean_avg_op,
            }
            print(f"  {opt:>5}: best_lr={best_lr:.6f}  loss={mean_loss:.6e}  "
                  f"max_||dW||_op={mean_max_op:.4e}  avg_||dW||_op={mean_avg_op:.4e}")

    # =========================================================================
    # ANALYSIS: log-log fit of optimal LR vs depth
    # =========================================================================
    print(f"\n\n{'='*100}")
    print("ANALYSIS: OPTIMAL LR VS DEPTH SCALING")
    print(f"{'='*100}")

    for opt in ['sgd', 'muon']:
        depths_arr = np.array(DEPTHS, dtype=float)
        lrs = np.array([results[(d, opt)]['best_lr'] for d in DEPTHS])
        log_d = np.log(depths_arr)
        log_lr = np.log(lrs)

        slope, intercept = np.polyfit(log_d, log_lr, 1)
        predicted = slope * log_d + intercept
        ss_res = np.sum((log_lr - predicted)**2)
        ss_tot = np.sum((log_lr - np.mean(log_lr))**2)
        r2 = 1 - ss_res / (ss_tot + 1e-15) if ss_tot > 1e-15 else 0

        lr_ratio = max(lrs) / min(lrs)

        print(f"\n  {opt.upper()}:")
        print(f"    log-log slope: {slope:.3f}  (O(L^{slope:.2f}))")
        print(f"    R^2: {r2:.4f}")
        print(f"    LR range: [{min(lrs):.6f}, {max(lrs):.6f}]  ratio: {lr_ratio:.1f}x")
        print(f"    Per-depth: ", end="")
        for d in DEPTHS:
            print(f"L={d}:{results[(d, opt)]['best_lr']:.5f}  ", end="")
        print()

    # =========================================================================
    # ANALYSIS: operator norm of updates
    # =========================================================================
    print(f"\n\n{'='*100}")
    print("ANALYSIS: UPDATE OPERATOR NORM VS DEPTH")
    print(f"{'='*100}")

    for opt in ['sgd', 'muon']:
        print(f"\n  {opt.upper()} max ||delta_W||_op at optimal LR:")
        for d in DEPTHS:
            r = results[(d, opt)]
            print(f"    L={d:>2}: {r['mean_max_op_norm']:.4e}")

    # =========================================================================
    # VERDICT
    # =========================================================================
    print(f"\n\n{'='*100}")
    print("VERDICT")
    print(f"{'='*100}")

    sgd_lrs = [results[(d, 'sgd')]['best_lr'] for d in DEPTHS]
    muon_lrs = [results[(d, 'muon')]['best_lr'] for d in DEPTHS]
    sgd_ratio = max(sgd_lrs) / min(sgd_lrs)
    muon_ratio = max(muon_lrs) / min(muon_lrs)

    t1 = muon_ratio < 5.0
    t2 = sgd_ratio > 20.0
    t3 = sgd_ratio / muon_ratio > 10.0

    print(f"\n  T1: Muon LR varies < 5x across depths? ratio={muon_ratio:.1f}x  --> {'PASS' if t1 else 'FAIL'}")
    print(f"  T2: SGD LR varies > 20x across depths?  ratio={sgd_ratio:.1f}x  --> {'PASS' if t2 else 'FAIL'}")
    print(f"  T3: SGD/Muon LR variability ratio > 10?  {sgd_ratio/muon_ratio:.1f}x  --> {'PASS' if t3 else 'FAIL'}")

    print(f"\n{'='*100}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()
