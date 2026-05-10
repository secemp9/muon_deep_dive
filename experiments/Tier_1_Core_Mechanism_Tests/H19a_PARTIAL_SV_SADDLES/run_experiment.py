#!/usr/bin/env python3
"""
H19a: PARTIAL SV EQUALIZATION CREATES SADDLE POINTS
=====================================================

MOTIVATION:
  H3b: partial SV equalization (k<32) is WORSE than SGD. Not just weaker --
  actively destructive. WHY?

HYPOTHESIS:
  Partial SV equalization creates an inconsistent geometry: equalized SVs
  want to explore uniformly while non-equalized SVs pull toward the original
  anisotropic landscape. This mismatch introduces negative Hessian eigenvalues
  (saddle directions) along directions mixing equalized and non-equalized SVs.

PROTOCOL:
  2-layer 4x4 deep linear (32 params, full Hessian tractable).
  At step 50 (after warmup with SGD):
    For k in {0, 1, 2, 4, 8, 16, 32}: take ONE step with partial-SV-equalized
    gradient. Compute full Hessian AFTER the step. Count negative eigenvalues
    and near-zero eigenvalues.

KEY TEST: negative eigenvalue count peaks at intermediate k.

Setup: 2-layer 4x4, 32 params, 5 seeds.
"""

import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DIM = 4
N_LAYERS = 2
N_PARAMS = N_LAYERS * DIM * DIM  # 32
WARMUP_STEPS = 50
NUM_SEEDS = 5
MOMENTUM = 0.9
LR_WARMUP = 0.01
LR_STEP = 0.01
FD_EPS = 1e-5
NS_ITERS = 5

# For a 4x4 matrix, SVD has 4 SVs. We test k per-layer from 0..4.
# But we also want to test higher k values conceptually by thinking of
# the "flattened" gradient. To keep it concrete: k is per-layer.
# k=0: SGD direction (no equalization)
# k=1..3: partial equalization
# k=4: full equalization (polar factor)
K_PER_LAYER = [0, 1, 2, 3, 4]


# =============================================================================
# NETWORK
# =============================================================================

def pack(Ws):
    return np.concatenate([W.ravel() for W in Ws])


def unpack(theta):
    Ws = []
    idx = 0
    for _ in range(N_LAYERS):
        Ws.append(theta[idx:idx + DIM * DIM].reshape(DIM, DIM))
        idx += DIM * DIM
    return Ws


def forward(Ws, X):
    out = X.copy()
    for W in Ws:
        out = W @ out
    return out


def loss_fn(theta, X, Y):
    Ws = unpack(theta)
    pred = forward(Ws, X)
    return 0.5 * np.mean(np.sum((pred - Y) ** 2, axis=0))


def grad_fn(theta, X, Y):
    Ws = unpack(theta)
    N = X.shape[1]
    acts = [X.copy()]
    for W in Ws:
        acts.append(W @ acts[-1])
    delta = (acts[-1] - Y) / N
    grads = []
    for l in range(N_LAYERS - 1, -1, -1):
        grads.insert(0, delta @ acts[l].T)
        if l > 0:
            delta = Ws[l].T @ delta
    return pack(grads), grads


def hessian_fd(theta, X, Y):
    """Full Hessian via finite differences (32x32 = tractable)."""
    n = len(theta)
    H = np.zeros((n, n))
    for i in range(n):
        tp = theta.copy()
        tp[i] += FD_EPS
        tm = theta.copy()
        tm[i] -= FD_EPS
        gp, _ = grad_fn(tp, X, Y)
        gm, _ = grad_fn(tm, X, Y)
        H[:, i] = (gp - gm) / (2 * FD_EPS)
    return 0.5 * (H + H.T)


# =============================================================================
# PARTIAL SV EQUALIZATION
# =============================================================================

def partial_sv_equalize_layer(M, k):
    """
    Equalize top-k SVs of a layer gradient matrix M (DIM x DIM).
    k=0: return M (no equalization, just norm-matched)
    k=DIM: full polar factor (UV^T)
    k=1..DIM-1: partial equalization
    """
    U, sigma, Vt = np.linalg.svd(M, full_matrices=False)
    d = len(sigma)

    if k == 0:
        # No equalization, but match norm to polar factor for fair comparison
        target_norm = np.sqrt(d)  # ||UV^T||_F = sqrt(d)
        current_norm = np.linalg.norm(sigma)
        if current_norm > 1e-15:
            sigma_scaled = sigma * (target_norm / current_norm)
            return U @ np.diag(sigma_scaled) @ Vt
        return M

    kk = min(k, d)
    if kk >= d:
        # Full polar factor
        return U @ Vt

    # Equalize top-k SVs to their mean
    sigma_new = sigma.copy()
    top_mean = np.mean(sigma[:kk])
    sigma_new[:kk] = top_mean

    # Scale to match polar factor norm
    target_norm = np.sqrt(d)
    current_norm = np.linalg.norm(sigma_new)
    if current_norm > 1e-15:
        sigma_new *= target_norm / current_norm

    return U @ np.diag(sigma_new) @ Vt


def newton_schulz_layer(M, n_iters=NS_ITERS):
    norm = np.linalg.norm(M, 'fro')
    if norm < 1e-15:
        return M
    X = M / norm
    for _ in range(n_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


# =============================================================================
# MAIN
# =============================================================================

def main():
    seeds = [42 + i * 137 for i in range(NUM_SEEDS)]

    print("=" * 100)
    print("H19a: PARTIAL SV EQUALIZATION CREATES SADDLE POINTS?")
    print("=" * 100)
    print(f"Network: {N_LAYERS}-layer deep linear {DIM}x{DIM} ({N_PARAMS} params)")
    print(f"k per layer: {K_PER_LAYER}")
    print(f"Warmup: {WARMUP_STEPS} SGD steps at LR={LR_WARMUP}, then one step at LR={LR_STEP}")
    print(f"Seeds: {NUM_SEEDS}")
    print()

    # Accumulators
    neg_eigs_before_all = []
    neg_eigs_after = {k: [] for k in K_PER_LAYER}
    min_eig_after = {k: [] for k in K_PER_LAYER}
    near_zero_after = {k: [] for k in K_PER_LAYER}
    loss_after = {k: [] for k in K_PER_LAYER}

    for si, seed in enumerate(seeds):
        rng = np.random.RandomState(seed)
        X = rng.randn(DIM, 64) * 0.3
        Y = rng.randn(DIM, 64) * 0.3
        weights = [np.eye(DIM) + rng.randn(DIM, DIM) * 0.1 for _ in range(N_LAYERS)]
        theta = pack(weights)

        # Warmup with SGD + momentum
        mom = np.zeros_like(theta)
        for step in range(WARMUP_STEPS):
            g, _ = grad_fn(theta, X, Y)
            mom = MOMENTUM * mom + g
            theta = theta - LR_WARMUP * mom

        # Hessian BEFORE the partial-SV step
        H_before = hessian_fd(theta, X, Y)
        eigs_before = np.linalg.eigvalsh(H_before)
        n_neg_before = int(np.sum(eigs_before < -1e-8))
        n_near_zero_before = int(np.sum(np.abs(eigs_before) < 1e-6))
        loss_before = loss_fn(theta, X, Y)
        neg_eigs_before_all.append(n_neg_before)

        print(f"\n  Seed {si + 1} (seed={seed}):")
        print(f"    Before: {n_neg_before} neg eigs, {n_near_zero_before} near-zero, "
              f"min={eigs_before[0]:.4e}, loss={loss_before:.6e}")

        # For each k: take one partial-SV step, measure Hessian
        for k in K_PER_LAYER:
            theta_k = theta.copy()
            g, grads_list = grad_fn(theta_k, X, Y)

            # Apply partial SV equalization per layer
            step_layers = []
            for l in range(N_LAYERS):
                step_layers.append(partial_sv_equalize_layer(grads_list[l], k))
            step_vec = pack(step_layers)

            theta_new = theta_k - LR_STEP * step_vec

            # Hessian after step
            H_after = hessian_fd(theta_new, X, Y)
            eigs_after_k = np.linalg.eigvalsh(H_after)
            n_neg = int(np.sum(eigs_after_k < -1e-8))
            min_eig = eigs_after_k[0]
            n_near_zero = int(np.sum(np.abs(eigs_after_k) < 1e-6))
            l_after = loss_fn(theta_new, X, Y)

            neg_eigs_after[k].append(n_neg)
            min_eig_after[k].append(min_eig)
            near_zero_after[k].append(n_near_zero)
            loss_after[k].append(l_after)

            print(f"    k={k}: {n_neg} neg eigs, {n_near_zero} near-zero, "
                  f"min={min_eig:.4e}, loss={l_after:.6e}")

    # =========================================================================
    # RESULTS
    # =========================================================================
    print(f"\n\n{'=' * 100}")
    print("RESULTS: NEGATIVE HESSIAN EIGENVALUES AFTER ONE PARTIAL-SV STEP")
    print(f"{'=' * 100}")

    print(f"\n  {'k':>4}  {'Mean neg eigs':>14}  {'Std neg eigs':>14}  "
          f"{'Mean min eig':>14}  {'Mean near-0':>12}  {'Mean loss':>14}")
    print("  " + "-" * 80)
    for k in K_PER_LAYER:
        print(f"  {k:>4}  {np.mean(neg_eigs_after[k]):>14.1f}  "
              f"{np.std(neg_eigs_after[k]):>14.1f}  "
              f"{np.mean(min_eig_after[k]):>14.4e}  "
              f"{np.mean(near_zero_after[k]):>12.1f}  "
              f"{np.mean(loss_after[k]):>14.6e}")

    # KEY TEST: Does intermediate k produce MORE negative eigenvalues?
    print(f"\n  === KEY TEST: Negative eigenvalue count peaks at intermediate k ===")
    mean_neg_0 = np.mean(neg_eigs_after[0])
    mean_neg_full = np.mean(neg_eigs_after[DIM])
    endpoints_max = max(mean_neg_0, mean_neg_full)

    peak_k = None
    peak_neg = -1
    for k in K_PER_LAYER:
        mn = np.mean(neg_eigs_after[k])
        if mn > peak_neg:
            peak_neg = mn
            peak_k = k

    intermediate_ks = [k for k in K_PER_LAYER if 0 < k < DIM]
    intermediate_worse = any(
        np.mean(neg_eigs_after[k]) > endpoints_max + 0.5
        for k in intermediate_ks
    )

    print(f"    k=0 (SGD) mean neg eigs:       {mean_neg_0:.1f}")
    print(f"    k={DIM} (full Muon) mean neg eigs: {mean_neg_full:.1f}")
    print(f"    Peak at k={peak_k} with {peak_neg:.1f} neg eigs")
    print(f"    Intermediate k MORE saddle dirs: {'CONFIRMED' if intermediate_worse else 'NOT CONFIRMED'}")

    # Secondary analysis: loss degradation
    print(f"\n  === Loss comparison ===")
    for k in K_PER_LAYER:
        ml = np.mean(loss_after[k])
        ml0 = np.mean(loss_after[0])
        ratio = ml / max(ml0, 1e-30)
        status = "WORSE" if ratio > 1.05 else ("BETTER" if ratio < 0.95 else "SIMILAR")
        print(f"    k={k}: loss={ml:.6e}  (ratio vs k=0: {ratio:.4f}x)  [{status}]")

    # Hessian spectrum shape analysis
    print(f"\n  === Mean Hessian eigenvalue spectrum (before step) ===")
    print(f"    Mean neg eigs before step: {np.mean(neg_eigs_before_all):.1f}")

    print(f"\n{'=' * 100}")
    print("EXPERIMENT COMPLETE")
    print(f"{'=' * 100}")


if __name__ == '__main__':
    main()
