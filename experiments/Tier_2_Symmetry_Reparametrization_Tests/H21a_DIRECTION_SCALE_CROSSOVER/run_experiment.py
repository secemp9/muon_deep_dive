#!/usr/bin/env python3
"""
H21a: DIRECTION vs SCALE CROSSOVER
====================================

MOTIVATION:
  2.2d: Oracle per-layer SGD beats Muon by 8 orders at c=100. But at c=1,
  Muon wins ~30x. There must be a crossover point c* where Muon's directional
  advantage is exactly offset by the per-layer LR advantage of oracle SGD.

PROTOCOL:
  4-layer deep linear 32x32, 300 steps, 3 seeds.
  Sweep c in {1, 2, 5, 10, 20, 50, 100, 200, 500}.
  For each c:
    (a) Muon with single LR sweep
    (b) Oracle per-layer SGD: grid 5 LRs per layer = 625 combos (full grid)
        OR random sample 100 combos if too slow.
  Compute: Muon_loss / Oracle_SGD_loss vs c. Find c* where ratio crosses 1.0.

KEY TEST: find the critical imbalance c* marking the direction-to-scale
  regime transition.

Setup: 4-layer, 32x32, 300 steps, 3 seeds.
"""

import numpy as np
import itertools
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DIM = 32
N_LAYERS = 4
NUM_STEPS = 300
MOMENTUM = 0.9
NS_ITERS = 5
NUM_SEEDS = 3
BATCH_SIZE = 64

C_VALUES = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]

MUON_LR_GRID = [0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.002, 0.001]

# For oracle per-layer: 5 LRs per layer, 5^4 = 625 combos
PER_LAYER_LR_GRID = [0.1, 0.01, 0.001, 0.0001, 0.00001]


# =============================================================================
# NETWORK
# =============================================================================

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
    # Apply c-rescaling: layer 0 *= c, layer -1 *= 1/c
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
    return 0.5 * np.mean(np.sum((pred - Y) ** 2, axis=0))


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


# =============================================================================
# TRAINING
# =============================================================================

def train_muon(w0, X, Y, lr):
    weights = [W.copy() for W in w0]
    mom = [np.zeros_like(W) for W in weights]
    for step in range(NUM_STEPS):
        if compute_loss(weights, X, Y) > 1e10:
            return float('inf')
        grads = compute_gradients(weights, X, Y)
        for i in range(N_LAYERS):
            mom[i] = MOMENTUM * mom[i] + newton_schulz(grads[i])
            weights[i] -= lr * mom[i]
    fl = compute_loss(weights, X, Y)
    return fl if np.isfinite(fl) else float('inf')


def train_sgd_per_layer(w0, X, Y, per_layer_lrs):
    weights = [W.copy() for W in w0]
    mom = [np.zeros_like(W) for W in weights]
    for step in range(NUM_STEPS):
        if compute_loss(weights, X, Y) > 1e10:
            return float('inf')
        grads = compute_gradients(weights, X, Y)
        for i in range(N_LAYERS):
            mom[i] = MOMENTUM * mom[i] + grads[i]
            weights[i] -= per_layer_lrs[i] * mom[i]
    fl = compute_loss(weights, X, Y)
    return fl if np.isfinite(fl) else float('inf')


# =============================================================================
# LR SWEEP
# =============================================================================

def sweep_muon(seeds, c):
    """Sweep LR for Muon, return best LR."""
    best_lr, best_loss = MUON_LR_GRID[-1], float('inf')
    for lr in MUON_LR_GRID:
        losses = []
        for s in seeds:
            X, Y = make_data(s)
            w = init_weights(s + 5000, c)
            fl = train_muon(w, X, Y, lr)
            losses.append(fl)
        finite = [l for l in losses if np.isfinite(l)]
        ml = np.mean(finite) if finite else float('inf')
        if ml < best_loss:
            best_loss = ml
            best_lr = lr
    return best_lr, best_loss


def sweep_oracle_per_layer(seeds, c):
    """
    Full grid search: 5^4 = 625 combos. Return best per-layer LR tuple.
    """
    combos = list(itertools.product(PER_LAYER_LR_GRID, repeat=N_LAYERS))

    best_combo = None
    best_loss = float('inf')

    for combo in combos:
        lrs = list(combo)
        losses = []
        for s in seeds:
            X, Y = make_data(s)
            w = init_weights(s + 5000, c)
            fl = train_sgd_per_layer(w, X, Y, lrs)
            losses.append(fl)
        finite = [l for l in losses if np.isfinite(l)]
        ml = np.mean(finite) if finite else float('inf')
        if ml < best_loss:
            best_loss = ml
            best_combo = lrs

    return best_combo, best_loss


# =============================================================================
# MAIN
# =============================================================================

def main():
    seeds = [42 + i * 137 for i in range(NUM_SEEDS)]

    print("=" * 100)
    print("H21a: DIRECTION vs SCALE CROSSOVER")
    print("=" * 100)
    print(f"Network: {N_LAYERS}-layer deep linear {DIM}x{DIM}")
    print(f"c values: {C_VALUES}")
    print(f"Steps: {NUM_STEPS}, Seeds: {NUM_SEEDS}")
    print(f"Oracle grid: {len(PER_LAYER_LR_GRID)}^{N_LAYERS} = "
          f"{len(PER_LAYER_LR_GRID) ** N_LAYERS} combos")
    print()

    results = {}

    for c in C_VALUES:
        print(f"\n  c={c:.0f}")

        # Muon LR sweep
        best_muon_lr, muon_sweep_loss = sweep_muon(seeds, c)
        print(f"    Muon best LR: {best_muon_lr}, sweep loss: {muon_sweep_loss:.4e}")

        # Oracle per-layer LR
        print(f"    Oracle grid search (625 combos)...", end=" ", flush=True)
        best_oracle_lrs, oracle_sweep_loss = sweep_oracle_per_layer(seeds, c)
        print(f"done. Best LRs: {[f'{lr:.0e}' for lr in best_oracle_lrs]}")

        # Full evaluation
        muon_losses = []
        oracle_losses = []
        for s in seeds:
            X, Y = make_data(s)
            w = init_weights(s + 5000, c)
            muon_losses.append(train_muon(w, X, Y, best_muon_lr))
            oracle_losses.append(train_sgd_per_layer(w, X, Y, best_oracle_lrs))

        muon_mean = np.mean([l for l in muon_losses if np.isfinite(l)] or [float('inf')])
        oracle_mean = np.mean([l for l in oracle_losses if np.isfinite(l)] or [float('inf')])

        ratio = muon_mean / max(oracle_mean, 1e-30)
        regime = "DIRECTION (Muon wins)" if ratio < 1.0 else "SCALE (Oracle wins)"

        results[c] = {
            'muon': muon_mean, 'oracle': oracle_mean,
            'ratio': ratio, 'regime': regime,
            'muon_lr': best_muon_lr, 'oracle_lrs': best_oracle_lrs,
        }
        print(f"    Muon={muon_mean:.4e}  Oracle={oracle_mean:.4e}  "
              f"Muon/Oracle={ratio:.4f}  [{regime}]")

    # =========================================================================
    # RESULTS
    # =========================================================================
    print(f"\n\n{'=' * 100}")
    print("RESULTS: DIRECTION vs SCALE CROSSOVER")
    print(f"{'=' * 100}")

    print(f"\n  {'c':>6}  {'Muon loss':>12}  {'Oracle loss':>12}  {'Muon/Oracle':>12}  {'Regime':>25}")
    print("  " + "-" * 75)

    c_star = None
    prev_ratio = None
    for c in C_VALUES:
        r = results[c]
        marker = ""
        if prev_ratio is not None and prev_ratio < 1.0 and r['ratio'] >= 1.0 and c_star is None:
            c_star = c
            marker = "  <-- crossover"
        elif prev_ratio is not None and prev_ratio >= 1.0 and r['ratio'] < 1.0 and c_star is None:
            c_star = c
            marker = "  <-- crossover"
        prev_ratio = r['ratio']

        print(f"  {c:>6.0f}  {r['muon']:>12.4e}  {r['oracle']:>12.4e}  "
              f"{r['ratio']:>12.4f}  {r['regime']:>25}{marker}")

    # If no crossover found, check if it's all one regime
    if c_star is None:
        ratios = [results[c]['ratio'] for c in C_VALUES]
        if all(r < 1.0 for r in ratios):
            print(f"\n  Muon wins at ALL c values tested. c* > {C_VALUES[-1]}")
        elif all(r >= 1.0 for r in ratios):
            print(f"\n  Oracle wins at ALL c values tested. c* < {C_VALUES[0]}")
        else:
            # Find the transition
            for i in range(1, len(C_VALUES)):
                r_prev = results[C_VALUES[i - 1]]['ratio']
                r_curr = results[C_VALUES[i]]['ratio']
                if (r_prev < 1.0) != (r_curr < 1.0):
                    c_star = (C_VALUES[i - 1] + C_VALUES[i]) / 2
                    break
            print(f"\n  Estimated c* (crossover): ~{c_star:.0f}")
    else:
        print(f"\n  Crossover c* = {c_star:.0f}")

    # Detailed analysis at c=1 (direction regime)
    print(f"\n  === Analysis at c=1 (direction regime) ===")
    r1 = results[1.0]
    print(f"    Muon loss:   {r1['muon']:.4e}")
    print(f"    Oracle loss: {r1['oracle']:.4e}")
    print(f"    Muon/Oracle: {r1['ratio']:.4f}")
    if r1['ratio'] < 1.0:
        print(f"    Muon's DIRECTIONAL advantage: {1.0 / r1['ratio']:.1f}x better than oracle per-layer LR")
    else:
        print(f"    Oracle SGD is already better even at c=1")

    # Trend analysis
    print(f"\n  === Trend: Muon/Oracle ratio vs c ===")
    cs = np.array(C_VALUES)
    ratios = np.array([results[c]['ratio'] for c in C_VALUES])
    log_cs = np.log10(cs)
    log_ratios = np.log10(np.clip(ratios, 1e-30, None))
    if len(log_cs) >= 3:
        slope, intercept = np.polyfit(log_cs, log_ratios, 1)
        print(f"    log10(ratio) = {slope:.3f} * log10(c) + {intercept:.3f}")
        print(f"    Power law: ratio ~ c^{slope:.3f}")
        if slope > 0:
            # Find where ratio = 1 => log_ratio = 0
            c_star_fit = 10 ** (-intercept / slope)
            print(f"    Extrapolated c* from power law: {c_star_fit:.1f}")

    print(f"\n{'=' * 100}")
    print("EXPERIMENT COMPLETE")
    print(f"{'=' * 100}")


if __name__ == '__main__':
    main()
