#!/usr/bin/env python3
"""
H15b: Explicit SVD Clamping -- Does Improving Conditioning Match Muon's Loss?
==============================================================================

MOTIVATION (from H15 surprise):
  Matrix layers under Muon sometimes have WORSE kappa than SGD (0.8x) despite
  3x better loss. If conditioning improvement is NOT the mechanism, then
  artificially forcing good conditioning should NOT match Muon's loss.

QUESTION: If we add explicit SVD clamping to SGD (clamp sigma_max/sigma_min
  to target kappa after each step), does it match Muon's loss trajectory?
  If NO: confirms direction quality, not conditioning, is the mechanism.
  If YES: conditioning IS the mechanism and H15 results were confounded.

PROTOCOL:
  Optimizers:
    (a) SGD -- baseline
    (b) Muon -- polar factor
    (c) SGD + SVD clamping -- after each SGD step, decompose W=USV^T,
        clamp S so kappa(W) <= target_kappa, recompose W.
    (d) SGD + SVD equalize -- set ALL SVs to mean(S) after each step.
  Sweep target_kappa for (c) in {2, 5, 10, 50}.

KEY TESTS:
  T1: Does SVD-clamped SGD (kappa<=5) match Muon's final loss within 2x?
  T2: Does SVD-equalized SGD (all SVs equal) match Muon?
  T3: If both fail, the conditioning-path explanation is falsified.

Setup: 4-layer, 32x32, 500 steps, 10 seeds, LR swept per method.
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

LR_CANDIDATES = [0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.002, 0.001]
KAPPA_TARGETS = [2, 5, 10, 50]


def newton_schulz(M, n_iters=NS_ITERS):
    norm = np.linalg.norm(M, ord='fro')
    if norm < 1e-15:
        return M
    X = M / norm
    for _ in range(n_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


def svd_clamp(W, target_kappa):
    """Clamp singular values so kappa(W) <= target_kappa."""
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    s_max = s[0]
    s_min_target = s_max / target_kappa
    s_clamped = np.maximum(s, s_min_target)
    return U @ np.diag(s_clamped) @ Vt


def svd_equalize(W):
    """Set all SVs to mean(SVs)."""
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    s_eq = np.full_like(s, np.mean(s))
    return U @ np.diag(s_eq) @ Vt


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


def train(weights_init, X, Y, lr, method, kappa_target=None):
    weights = [W.copy() for W in weights_init]
    mom = [np.zeros((DIM, DIM)) for _ in range(NUM_LAYERS)]
    for step in range(NUM_STEPS):
        loss = compute_loss(weights, X, Y)
        if not np.isfinite(loss) or loss > 1e10:
            return float('inf')
        grads = compute_gradients(weights, X, Y)
        for i in range(NUM_LAYERS):
            if method == 'muon':
                mom[i] = MOMENTUM * mom[i] + newton_schulz(grads[i])
                weights[i] = weights[i] - lr * mom[i]
            elif method == 'sgd':
                mom[i] = MOMENTUM * mom[i] + grads[i]
                weights[i] = weights[i] - lr * mom[i]
            elif method == 'sgd_clamp':
                mom[i] = MOMENTUM * mom[i] + grads[i]
                weights[i] = weights[i] - lr * mom[i]
                weights[i] = svd_clamp(weights[i], kappa_target)
            elif method == 'sgd_equalize':
                mom[i] = MOMENTUM * mom[i] + grads[i]
                weights[i] = weights[i] - lr * mom[i]
                weights[i] = svd_equalize(weights[i])
    return compute_loss(weights, X, Y)


def make_data(seed):
    rng = np.random.RandomState(seed)
    X = rng.randn(DIM, BATCH_SIZE) * 0.3
    Y = rng.randn(DIM, BATCH_SIZE) * 0.3
    return X, Y


def sweep_lr(method, seeds, kappa_target=None):
    best_lr, best_loss = LR_CANDIDATES[-1], float('inf')
    for lr in LR_CANDIDATES:
        losses = []
        for s in seeds:
            X, Y = make_data(s)
            w = init_weights(s + 5000)
            fl = train(w, X, Y, lr, method, kappa_target)
            losses.append(fl)
        ml = np.mean([l for l in losses if np.isfinite(l)]) if any(np.isfinite(l) for l in losses) else float('inf')
        if ml < best_loss:
            best_loss = ml
            best_lr = lr
    return best_lr


def main():
    seeds = [42 + i * 137 for i in range(NUM_SEEDS)]

    print("=" * 100)
    print("H15b: EXPLICIT SVD CLAMPING -- Does Improving kappa Match Muon's Loss?")
    print("=" * 100)
    print(f"Network: {NUM_LAYERS}-layer, {DIM}x{DIM}, {NUM_STEPS} steps, {NUM_SEEDS} seeds")
    print(f"Kappa targets for clamped SGD: {KAPPA_TARGETS}")
    print()

    # LR sweeps
    print("Phase 1: LR sweeps...")
    configs = [('sgd', None), ('muon', None), ('sgd_equalize', None)]
    for kt in KAPPA_TARGETS:
        configs.append(('sgd_clamp', kt))

    best_lrs = {}
    for method, kt in configs:
        name = method if kt is None else f"{method}_k{kt}"
        lr = sweep_lr(method, seeds[:3], kt)
        best_lrs[name] = lr
        print(f"  {name:>20}: best_lr={lr:.4f}")

    # Full training
    print("\nPhase 2: Full training...")
    results = {}
    for method, kt in configs:
        name = method if kt is None else f"{method}_k{kt}"
        lr = best_lrs[name]
        losses = []
        for s in seeds:
            X, Y = make_data(s)
            w = init_weights(s + 5000)
            fl = train(w, X, Y, lr, method, kt)
            losses.append(fl)
        finite = [l for l in losses if np.isfinite(l)]
        results[name] = np.mean(finite) if finite else float('inf')

    # Results
    print(f"\n{'=' * 100}")
    print("RESULTS")
    print(f"{'=' * 100}")

    muon_loss = results['muon']
    sgd_loss = results['sgd']

    print(f"\n  {'Method':>20}  {'Loss':>14}  {'vs Muon':>10}  {'vs SGD':>10}")
    print("  " + "-" * 58)
    for name in ['sgd', 'muon', 'sgd_equalize'] + [f'sgd_clamp_k{kt}' for kt in KAPPA_TARGETS]:
        r = results[name]
        vs_muon = r / max(muon_loss, 1e-30)
        vs_sgd = r / max(sgd_loss, 1e-30)
        print(f"  {name:>20}  {r:>14.6e}  {vs_muon:>10.2f}x  {vs_sgd:>10.2f}x")

    # Hypothesis tests
    print(f"\n\n{'=' * 100}")
    print("HYPOTHESIS TESTS")
    print(f"{'=' * 100}")

    clamp5_loss = results.get('sgd_clamp_k5', float('inf'))
    eq_loss = results.get('sgd_equalize', float('inf'))

    t1 = clamp5_loss < muon_loss * 2
    print(f"\n  T1: SVD-clamped SGD (kappa<=5) within 2x of Muon?")
    print(f"      Clamped: {clamp5_loss:.6e}, Muon: {muon_loss:.6e}, ratio: {clamp5_loss/max(muon_loss,1e-30):.2f}x")
    print(f"      --> {'PASS (conditioning explains it)' if t1 else 'FAIL (conditioning insufficient)'}")

    t2 = eq_loss < muon_loss * 2
    print(f"\n  T2: SVD-equalized SGD within 2x of Muon?")
    print(f"      Equalized: {eq_loss:.6e}, Muon: {muon_loss:.6e}, ratio: {eq_loss/max(muon_loss,1e-30):.2f}x")
    print(f"      --> {'PASS' if t2 else 'FAIL'}")

    t3 = not t1 and not t2
    print(f"\n  T3: If T1 and T2 both fail, conditioning-path explanation is falsified.")
    print(f"      --> {'CONFIRMED: direction quality, not conditioning' if t3 else 'NOT YET FALSIFIED'}")

    print(f"\n{'=' * 100}")
    print("EXPERIMENT COMPLETE")
    print(f"{'=' * 100}")


if __name__ == '__main__':
    main()
