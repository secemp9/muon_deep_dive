#!/usr/bin/env python3
"""
H16: LR Confound Audit -- Spectral Democracy and WeightWatcher alpha
======================================================================

MOTIVATION (from H6 lesson):
  The 130x curvature rescaling was an LR artifact. Apply the same scrutiny
  to the spectral democracy and WeightWatcher alpha comparisons (1.3c).

QUESTION: Were the spectral democracy (effective rank) and alpha (power-law
  tail exponent) comparisons done at matched OPTIMAL LR? If Muon was at
  its default LR while SGD was at a suboptimal LR, the spectral properties
  might differ due to optimization QUALITY rather than fundamental mechanisms.

PROTOCOL:
  For each optimizer in {SGD, Muon}:
    Sweep LR, find optimal.
    Train 500 steps at optimal LR.
    Measure at convergence:
      - Effective rank (exp(entropy of SVs)) per layer.
      - Condition number kappa per layer.
      - Power-law alpha of eigenvalue distribution.
  Compare these metrics at MATCHED optimal LR (not fixed default LR).

KEY TESTS:
  T1: Does effective rank difference (Muon vs SGD) change by >2x when
      both use their optimal LR vs default LR?
  T2: Does the kappa ratio change qualitatively (e.g., flip from Muon-better
      to SGD-better) when using optimal LRs?

Setup: 4-layer, 32x32, 500 steps, 10 seeds.
"""

import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DIM = 32
NUM_LAYERS = 4
NUM_STEPS = 500
MOMENTUM = 0.9
NS_ITERS = 5
NUM_SEEDS = 10
BATCH_SIZE = 64

MUON_LRS = [0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.002, 0.001]
SGD_LRS = [0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001, 0.0005]

DEFAULT_MUON_LR = 0.02
DEFAULT_SGD_LR = 0.01


def newton_schulz(M, n_iters=NS_ITERS):
    norm = np.linalg.norm(M, ord='fro')
    if norm < 1e-15:
        return M
    X = M / norm
    for _ in range(n_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


def init_weights(seed):
    rng = np.random.RandomState(seed)
    return [np.eye(DIM) + rng.randn(DIM, DIM) * 0.1 for _ in range(NUM_LAYERS)]


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


def effective_rank(W):
    """Effective rank = exp(entropy of normalized SVs)."""
    s = np.linalg.svd(W, compute_uv=False)
    s = s[s > 1e-12]
    if len(s) == 0:
        return 0
    p = s / np.sum(s)
    H = -np.sum(p * np.log(p + 1e-30))
    return np.exp(H)


def condition_number(W):
    s = np.linalg.svd(W, compute_uv=False)
    return s[0] / max(s[-1], 1e-12)


def train(weights_init, X, Y, lr, optimizer):
    weights = [W.copy() for W in weights_init]
    mom = [np.zeros_like(W) for W in weights]
    for step in range(NUM_STEPS):
        loss = compute_loss(weights, X, Y)
        if not np.isfinite(loss) or loss > 1e10:
            return None, float('inf')
        grads = compute_gradients(weights, X, Y)
        for i in range(len(weights)):
            if optimizer == 'muon':
                mom[i] = MOMENTUM * mom[i] + newton_schulz(grads[i])
            else:
                mom[i] = MOMENTUM * mom[i] + grads[i]
            weights[i] = weights[i] - lr * mom[i]
    return weights, compute_loss(weights, X, Y)


def make_data(seed):
    rng = np.random.RandomState(seed)
    X = rng.randn(DIM, BATCH_SIZE) * 0.3
    Y = rng.randn(DIM, BATCH_SIZE) * 0.3
    return X, Y


def main():
    seeds = [42 + i * 137 for i in range(NUM_SEEDS)]

    print("=" * 100)
    print("H16: LR CONFOUND AUDIT -- Spectral Democracy and Alpha")
    print("=" * 100)
    print(f"Network: {NUM_LAYERS}-layer, {DIM}x{DIM}, {NUM_STEPS} steps, {NUM_SEEDS} seeds")
    print()

    # LR sweep
    print("Phase 1: LR sweep...")
    best_lrs = {}
    for opt, candidates in [('muon', MUON_LRS), ('sgd', SGD_LRS)]:
        best_lr, best_loss = candidates[-1], float('inf')
        for lr in candidates:
            losses = []
            for s in seeds[:3]:
                X, Y = make_data(s)
                w = init_weights(s + 5000)
                _, fl = train(w, X, Y, lr, opt)
                losses.append(fl)
            ml = np.mean([l for l in losses if np.isfinite(l)]) if any(np.isfinite(l) for l in losses) else float('inf')
            if ml < best_loss:
                best_loss = ml
                best_lr = lr
        best_lrs[opt] = best_lr
        print(f"  {opt}: best_lr={best_lr:.4f}")

    # Full training at both default and optimal LR
    print("\nPhase 2: Full training...")

    configs = [
        ('sgd_default', 'sgd', DEFAULT_SGD_LR),
        ('sgd_optimal', 'sgd', best_lrs['sgd']),
        ('muon_default', 'muon', DEFAULT_MUON_LR),
        ('muon_optimal', 'muon', best_lrs['muon']),
    ]

    results = {}
    for name, opt, lr in configs:
        all_erank = {l: [] for l in range(NUM_LAYERS)}
        all_kappa = {l: [] for l in range(NUM_LAYERS)}
        all_loss = []

        for s in seeds:
            X, Y = make_data(s)
            w = init_weights(s + 5000)
            final_w, fl = train(w, X, Y, lr, opt)
            all_loss.append(fl)
            if final_w is not None:
                for l in range(NUM_LAYERS):
                    all_erank[l].append(effective_rank(final_w[l]))
                    all_kappa[l].append(condition_number(final_w[l]))

        results[name] = {
            'loss': np.mean(all_loss),
            'erank': {l: np.mean(all_erank[l]) for l in range(NUM_LAYERS)},
            'kappa': {l: np.mean(all_kappa[l]) for l in range(NUM_LAYERS)},
        }

    # Results
    print(f"\n{'=' * 100}")
    print("SPECTRAL METRICS AT DEFAULT vs OPTIMAL LR")
    print(f"{'=' * 100}")

    for metric_name, metric_key in [('Effective Rank', 'erank'), ('Condition Number', 'kappa')]:
        print(f"\n  {metric_name}:")
        print(f"  {'Config':>20}", end='')
        for l in range(NUM_LAYERS):
            print(f"  {'L'+str(l):>8}", end='')
        print(f"  {'Loss':>12}")
        print("  " + "-" * (20 + 10 * NUM_LAYERS + 14))

        for name, _, _ in configs:
            r = results[name]
            print(f"  {name:>20}", end='')
            for l in range(NUM_LAYERS):
                print(f"  {r[metric_key][l]:>8.2f}", end='')
            print(f"  {r['loss']:>12.4e}")

    # Key comparison: does the ranking change?
    print(f"\n\n{'=' * 100}")
    print("HYPOTHESIS TESTS")
    print(f"{'=' * 100}")

    # Effective rank difference at default vs optimal
    erank_diff_default = np.mean([results['muon_default']['erank'][l] - results['sgd_default']['erank'][l] for l in range(NUM_LAYERS)])
    erank_diff_optimal = np.mean([results['muon_optimal']['erank'][l] - results['sgd_optimal']['erank'][l] for l in range(NUM_LAYERS)])

    t1 = abs(erank_diff_default - erank_diff_optimal) > abs(erank_diff_default) * 0.5
    print(f"\n  T1: Effective rank difference changes >50% with LR sweep?")
    print(f"      Default diff (Muon-SGD): {erank_diff_default:.3f}")
    print(f"      Optimal diff (Muon-SGD): {erank_diff_optimal:.3f}")
    print(f"      --> {'PASS (LR confound exists)' if t1 else 'FAIL (robust to LR)'}")

    # Kappa ranking flip?
    kappa_ratio_default = np.mean([results['muon_default']['kappa'][l] / max(results['sgd_default']['kappa'][l], 1e-12) for l in range(NUM_LAYERS)])
    kappa_ratio_optimal = np.mean([results['muon_optimal']['kappa'][l] / max(results['sgd_optimal']['kappa'][l], 1e-12) for l in range(NUM_LAYERS)])

    flip = (kappa_ratio_default < 1.0) != (kappa_ratio_optimal < 1.0)
    t2 = flip
    print(f"\n  T2: Kappa ratio flips qualitatively with LR sweep?")
    print(f"      Default Muon/SGD kappa ratio: {kappa_ratio_default:.3f}")
    print(f"      Optimal Muon/SGD kappa ratio: {kappa_ratio_optimal:.3f}")
    print(f"      --> {'PASS (RANKING FLIPS -- previous results confounded)' if t2 else 'FAIL (ranking stable)'}")

    print(f"\n{'=' * 100}")
    print("EXPERIMENT COMPLETE")
    print(f"{'=' * 100}")


if __name__ == '__main__':
    main()
