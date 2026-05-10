#!/usr/bin/env python3
"""
H3a: Per-SV Gradient Utilization -- Muon vs Normalized SGD
============================================================

MOTIVATION (from H3 surprise):
  Muon beats normalized SGD by 19x (linear) and 10x (ReLU) at optimal LR.
  The critical difference: scale normalization PRESERVES gradient anisotropy
  (keeps internal SV ratios), while the polar factor EQUALIZES them (all -> 1).
  This is qualitatively different.

QUESTION: How much gradient information does each optimizer USE per singular
  direction? If G = U diag(sigma) V^T, SGD uses sigma_i (favors large SVs),
  normalized SGD uses sigma_i / ||sigma|| (same ratios, just scaled), and
  Muon uses 1 for all i (equal contribution from every SV direction).

PROTOCOL:
  At each training step, decompose gradient G = U * diag(sigma) * V^T.
  Compute for each optimizer's actual update Delta_W:
    - Project Delta_W onto each left/right singular direction pair.
    - Measure "utilization_i" = |<Delta_W, u_i v_i^T>| for each SV direction.
    - Compute utilization entropy: H = -sum(p_i log p_i) where p_i = util_i / sum(util).
  Higher entropy = more democratic use of ALL gradient directions.

KEY TESTS:
  T1: Muon has higher utilization entropy than normalized SGD (>0.5 bits more).
  T2: Muon's utilization is nearly uniform (H > 0.9 * H_max).
  T3: SGD's utilization is dominated by top SV (top-1 fraction > 0.5).

Setup: 4-layer, 32x32, track utilization over 500 steps, 5 seeds.
"""

import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DIM = 32
NUM_LAYERS = 4
NUM_STEPS = 500
MOMENTUM = 0.9
NS_ITERS = 5
NUM_SEEDS = 5
BATCH_SIZE = 64


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


def compute_utilization(update, U, V):
    """
    Compute per-SV utilization of update matrix.
    Returns array of |<update, u_i v_i^T>| for each i.
    """
    k = min(U.shape[1], V.shape[1])
    util = np.zeros(k)
    for i in range(k):
        # Projection onto rank-1 component u_i v_i^T
        util[i] = abs(np.dot(U[:, i], update @ V[:, i]))
    return util


def entropy(probs):
    """Shannon entropy of a probability distribution."""
    p = probs[probs > 1e-30]
    return -np.sum(p * np.log2(p))


def main():
    seeds = [42 + i * 137 for i in range(NUM_SEEDS)]

    print("=" * 100)
    print("H3a: PER-SV GRADIENT UTILIZATION -- Muon vs Normalized SGD")
    print("=" * 100)
    print(f"Network: {NUM_LAYERS}-layer, {DIM}x{DIM}, {NUM_STEPS} steps, {NUM_SEEDS} seeds")
    print()

    # Track utilization entropy for each optimizer
    methods = ['sgd', 'muon', 'norm_sgd']
    all_entropies = {m: [] for m in methods}  # lists of per-step entropies
    all_top1_fracs = {m: [] for m in methods}

    for seed in seeds:
        rng = np.random.RandomState(seed)
        X = rng.randn(DIM, BATCH_SIZE) * 0.3
        Y = rng.randn(DIM, BATCH_SIZE) * 0.3

        for method in methods:
            weights = init_weights(seed + 5000)
            mom = [np.zeros((DIM, DIM)) for _ in range(NUM_LAYERS)]
            step_entropies = []
            step_top1 = []

            for step in range(NUM_STEPS):
                loss = compute_loss(weights, X, Y)
                if not np.isfinite(loss) or loss > 1e10:
                    break
                grads = compute_gradients(weights, X, Y)

                for i in range(NUM_LAYERS):
                    G = grads[i]
                    U_g, sigma_g, Vt_g = np.linalg.svd(G, full_matrices=False)

                    if method == 'sgd':
                        mom[i] = MOMENTUM * mom[i] + G
                        update = mom[i]
                    elif method == 'muon':
                        ortho_g = newton_schulz(G)
                        mom[i] = MOMENTUM * mom[i] + ortho_g
                        update = mom[i]
                    elif method == 'norm_sgd':
                        mom[i] = MOMENTUM * mom[i] + G
                        v_norm = np.linalg.norm(mom[i], 'fro')
                        update = mom[i] / max(v_norm, 1e-12)

                    util = compute_utilization(update, U_g, Vt_g.T)
                    util_sum = np.sum(util)
                    if util_sum > 1e-30:
                        p = util / util_sum
                        step_entropies.append(entropy(p))
                        step_top1.append(p[0])

                    weights[i] = weights[i] - 0.01 * update

            all_entropies[method].extend(step_entropies)
            all_top1_fracs[method].extend(step_top1)

    # Results
    print(f"\n{'=' * 100}")
    print("UTILIZATION ENTROPY (higher = more democratic use of SV directions)")
    print(f"{'=' * 100}")

    H_max = np.log2(DIM)  # maximum entropy for DIM SVs
    print(f"  Maximum possible entropy (uniform over {DIM} SVs): {H_max:.2f} bits")
    print()

    print(f"  {'Method':>15}  {'Mean H':>8}  {'H/H_max':>8}  {'Top-1 frac':>12}  {'Std H':>8}")
    print("  " + "-" * 55)

    for m in methods:
        h = np.array(all_entropies[m])
        t1 = np.array(all_top1_fracs[m])
        print(f"  {m:>15}  {np.mean(h):>8.3f}  {np.mean(h)/H_max:>8.3f}  "
              f"{np.mean(t1):>12.4f}  {np.std(h):>8.3f}")

    # Hypothesis tests
    print(f"\n\n{'=' * 100}")
    print("HYPOTHESIS TESTS")
    print(f"{'=' * 100}")

    muon_H = np.mean(all_entropies['muon'])
    norm_H = np.mean(all_entropies['norm_sgd'])
    sgd_H = np.mean(all_entropies['sgd'])
    sgd_top1 = np.mean(all_top1_fracs['sgd'])

    t1 = muon_H > norm_H + 0.5
    t2 = muon_H > 0.9 * H_max
    t3 = sgd_top1 > 0.5

    print(f"\n  T1: Muon entropy > Norm SGD entropy + 0.5 bits?")
    print(f"      Muon={muon_H:.3f}, NormSGD={norm_H:.3f}, diff={muon_H-norm_H:.3f}")
    print(f"      --> {'PASS' if t1 else 'FAIL'}")

    print(f"\n  T2: Muon utilization nearly uniform (H > 0.9 * H_max)?")
    print(f"      Muon H={muon_H:.3f}, 0.9*H_max={0.9*H_max:.3f}")
    print(f"      --> {'PASS' if t2 else 'FAIL'}")

    print(f"\n  T3: SGD top-1 SV fraction > 0.5?")
    print(f"      SGD top-1 = {sgd_top1:.4f}")
    print(f"      --> {'PASS' if t3 else 'FAIL'}")

    print(f"\n{'=' * 100}")
    print("EXPERIMENT COMPLETE")
    print(f"{'=' * 100}")


if __name__ == '__main__':
    main()
