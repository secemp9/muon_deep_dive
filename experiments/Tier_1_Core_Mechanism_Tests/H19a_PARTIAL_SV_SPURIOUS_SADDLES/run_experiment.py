#!/usr/bin/env python3
"""
H19a: Partial SV Equalization Creates Spurious Saddle Points
==============================================================

FROM H3b: k<32 is WORSE than SGD (not just weaker -- actively destructive).
This is the most surprising result: partial equalization HURTS.

HYPOTHESIS:
  Partial SV equalization (top-k SVs equalized, rest proportional) creates
  an inconsistent geometry: the equalized SVs want to explore uniformly
  while the non-equalized SVs pull toward the original anisotropic landscape.
  This mismatch introduces negative Hessian eigenvalues (saddle points) along
  directions that mix equalized and non-equalized SVs.

  Full equalization (k=dim) avoids this because ALL directions are treated
  uniformly. k=0 (SGD) avoids it because the landscape is consistent.
  The half-measure creates a Frankensteinian geometry.

PROTOCOL:
  3-layer deep linear 4x4 (48 params, full Hessian tractable).
  At step 50 (after some training), for each k in {0, 4, 8, 16, 32}:
    1. Compute the Hessian at the current point
    2. Apply one partial-SV-equalized step
    3. Compute the Hessian at the new point
    4. Count negative eigenvalues (saddle directions)
    5. Measure the minimum eigenvalue (most negative = sharpest saddle)
  Compare negative eigenvalue counts across k.

KEY PREDICTION:
  Intermediate k (4, 8, 16) produce MORE negative Hessian eigenvalues
  than k=0 (SGD) or k=32 (full Muon).
"""

import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DIM = 4
N_LAYERS = 3
N_PARAMS = N_LAYERS * DIM * DIM  # 48
WARMUP_STEPS = 50
NUM_SEEDS = 5
MOMENTUM = 0.9
LR = 0.01
FD_EPS = 1e-5

K_VALUES = [0, 2, 4, 8, 16]  # 0=SGD, 16=full (DIM=4, so 16=N_PARAMS/3 per layer)
# For DIM=4, per-layer matrix is 4x4, SVD has 4 SVs.
# k values for equalization: 0 (none), 1, 2, 3, 4 (all) per layer
K_PER_LAYER = [0, 1, 2, 3, 4]


def pack(Ws):
    return np.concatenate([W.ravel() for W in Ws])


def unpack(theta):
    Ws = []
    idx = 0
    for _ in range(N_LAYERS):
        Ws.append(theta[idx:idx + DIM*DIM].reshape(DIM, DIM))
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
    return 0.5 * np.mean(np.sum((pred - Y)**2, axis=0))


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
    n = len(theta)
    H = np.zeros((n, n))
    for i in range(n):
        tp = theta.copy(); tp[i] += FD_EPS
        tm = theta.copy(); tm[i] -= FD_EPS
        gp, _ = grad_fn(tp, X, Y)
        gm, _ = grad_fn(tm, X, Y)
        H[:, i] = (gp - gm) / (2 * FD_EPS)
    return 0.5 * (H + H.T)


def partial_sv_equalize_layer(M, k):
    """Equalize top-k SVs of a single layer gradient matrix."""
    if k == 0:
        return M
    U, sigma, Vt = np.linalg.svd(M, full_matrices=False)
    d = len(sigma)
    kk = min(k, d)
    sigma_new = sigma.copy()

    if kk >= d:
        # Full equalization: all SVs to 1 (polar factor)
        return U @ Vt

    # Partial: equalize top-k to their mean
    top_mean = np.mean(sigma[:kk])
    sigma_new[:kk] = top_mean

    # Normalize to match polar factor norm
    target_norm = np.sqrt(d)
    current_norm = np.linalg.norm(sigma_new)
    if current_norm > 1e-15:
        sigma_new *= target_norm / current_norm

    return U @ np.diag(sigma_new) @ Vt


def newton_schulz_layer(M, n_iters=5):
    norm = np.linalg.norm(M, 'fro')
    if norm < 1e-15:
        return M
    X = M / norm
    for _ in range(n_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


def main():
    seeds = [42 + i * 137 for i in range(NUM_SEEDS)]

    print("=" * 100)
    print("H19a: PARTIAL SV EQUALIZATION CREATES SPURIOUS SADDLE POINTS?")
    print("=" * 100)
    print(f"Network: {N_LAYERS}-layer deep linear {DIM}x{DIM} ({N_PARAMS} params)")
    print(f"k per layer: {K_PER_LAYER}")
    print(f"Warmup: {WARMUP_STEPS} SGD steps, then measure Hessian before/after one step")
    print()

    # Accumulators
    neg_eigs_before = {k: [] for k in K_PER_LAYER}
    neg_eigs_after = {k: [] for k in K_PER_LAYER}
    min_eig_after = {k: [] for k in K_PER_LAYER}

    for si, seed in enumerate(seeds):
        rng = np.random.RandomState(seed)
        X = rng.randn(DIM, 64) * 0.3
        Y = rng.randn(DIM, 64) * 0.3
        weights = [np.eye(DIM) + rng.randn(DIM, DIM) * 0.1 for _ in range(N_LAYERS)]
        theta = pack(weights)

        # Warmup with SGD
        mom = np.zeros_like(theta)
        for step in range(WARMUP_STEPS):
            g, _ = grad_fn(theta, X, Y)
            mom = MOMENTUM * mom + g
            theta = theta - LR * mom

        # Hessian at warmup point
        H_before = hessian_fd(theta, X, Y)
        eigs_before = np.linalg.eigvalsh(H_before)
        n_neg_before = np.sum(eigs_before < -1e-8)

        print(f"\n  Seed {si+1}: H before has {n_neg_before} negative eigenvalues, min={eigs_before[0]:.4e}")

        # For each k: take one step, measure Hessian
        for k in K_PER_LAYER:
            theta_k = theta.copy()
            g, grads_list = grad_fn(theta_k, X, Y)

            # Apply partial SV equalization per layer
            step_layers = []
            for l in range(N_LAYERS):
                step_layers.append(partial_sv_equalize_layer(grads_list[l], k))
            step_vec = pack(step_layers)

            theta_new = theta_k - LR * step_vec
            H_after = hessian_fd(theta_new, X, Y)
            eigs_after_k = np.linalg.eigvalsh(H_after)
            n_neg_after = np.sum(eigs_after_k < -1e-8)
            min_eig = eigs_after_k[0]

            neg_eigs_before[k].append(n_neg_before)
            neg_eigs_after[k].append(n_neg_after)
            min_eig_after[k].append(min_eig)

            print(f"    k={k}: after step -> {n_neg_after} neg eigs, min={min_eig:.4e}")

    # =========================================================================
    # RESULTS
    # =========================================================================
    print(f"\n\n{'='*100}")
    print("RESULTS: NEGATIVE HESSIAN EIGENVALUES AFTER ONE STEP")
    print(f"{'='*100}")

    print(f"\n  {'k':>4}  {'Mean neg eigs (after)':>22}  {'Mean min eig':>16}  {'Mean neg eigs (before)':>24}")
    print("  " + "-" * 70)
    for k in K_PER_LAYER:
        print(f"  {k:>4}  {np.mean(neg_eigs_after[k]):>22.1f}  "
              f"{np.mean(min_eig_after[k]):>16.4e}  {np.mean(neg_eigs_before[k]):>24.1f}")

    # Test: intermediate k has more negative eigenvalues than k=0 or k=4
    mean_neg_0 = np.mean(neg_eigs_after[0])
    mean_neg_4 = np.mean(neg_eigs_after[4])
    intermediate_worse = any(
        np.mean(neg_eigs_after[k]) > max(mean_neg_0, mean_neg_4) + 0.5
        for k in [1, 2, 3]
    )

    print(f"\n  k=0 (SGD) mean neg eigs: {mean_neg_0:.1f}")
    print(f"  k=4 (full) mean neg eigs: {mean_neg_4:.1f}")
    print(f"  Intermediate k creates MORE saddle directions: {'CONFIRMED' if intermediate_worse else 'NOT CONFIRMED'}")

    print(f"\n{'='*100}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()
