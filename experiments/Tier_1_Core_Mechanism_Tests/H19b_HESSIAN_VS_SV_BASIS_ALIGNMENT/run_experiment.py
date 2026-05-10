#!/usr/bin/env python3
"""
H19b: Hessian-Basis Partial Eq Works Because It Preserves Curvature Alignment
===============================================================================

FROM H3b + Exp 3.2 PARADOX:
  - H3b: Partial SV equalization (k<32) is WORSE than SGD. All-or-nothing.
  - Exp 3.2: Partial HESSIAN-basis equalization (k=10 of 24) works at 114%.

WHY does Hessian-basis handle partial but SV-basis does not?

HYPOTHESIS:
  SV-basis equalization destroys the alignment between the update direction
  and the Hessian eigenbasis. When you set top-k SVs to their mean, the
  resulting update vector rotates AWAY from high-curvature Hessian directions
  into a hybrid direction that is neither curvature-aligned nor curvature-
  agnostic. Hessian-basis equalization, by construction, preserves alignment
  with the curvature structure -- it only changes magnitudes along already-
  meaningful directions.

PROTOCOL:
  3-layer deep linear 4x4 (48 params). At training step 50:
  1. Compute full Hessian, its eigenbasis V_H
  2. Compute gradient G and its layer-wise SVD basis V_SV
  3. For partial SV eq (k=2 of 4 per layer): compute update, project onto
     V_H, measure how much energy lands in top/middle/bottom Hessian modes
  4. For partial Hessian eq (k=10 of 48): compute update, same projection
  5. Compare curvature-weighted alignment: sum_i lambda_i * |<update, v_i>|^2

KEY PREDICTION:
  Partial SV eq redistributes energy INTO low-curvature Hessian directions
  (wasting update budget). Partial Hessian eq keeps energy in high-curvature
  directions (where it matters for convergence).
"""

import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DIM = 4
N_LAYERS = 3
N_PARAMS = N_LAYERS * DIM * DIM  # 48
WARMUP_STEPS = 50
NUM_SEEDS = 5
LR = 0.01
MOMENTUM = 0.9
FD_EPS = 1e-5


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


def partial_sv_eq_step(grads_list, k_per_layer):
    """Apply partial SV equalization per layer, return flat vector."""
    step_layers = []
    for G in grads_list:
        U, sigma, Vt = np.linalg.svd(G, full_matrices=False)
        d = len(sigma)
        kk = min(k_per_layer, d)
        if kk == 0:
            step_layers.append(G)
        elif kk >= d:
            step_layers.append(U @ Vt)
        else:
            sigma_new = sigma.copy()
            sigma_new[:kk] = np.mean(sigma[:kk])
            target_norm = np.sqrt(d)
            cn = np.linalg.norm(sigma_new)
            if cn > 1e-15:
                sigma_new *= target_norm / cn
            step_layers.append(U @ np.diag(sigma_new) @ Vt)
    return pack(step_layers)


def partial_hessian_eq_step(g_vec, eigvecs, eigvals, k):
    """Equalize top-k + bottom-k Hessian eigenvector components."""
    n = len(g_vec)
    projs = eigvecs.T @ g_vec
    if 2 * k >= n:
        selected = list(range(n))
    else:
        selected = list(range(k)) + list(range(n - k, n))

    eq_projs = projs.copy()
    mean_mag = np.mean(np.abs(projs[selected]))
    for idx in selected:
        eq_projs[idx] = np.sign(projs[idx]) * mean_mag

    result = eigvecs @ eq_projs
    # Normalize to same magnitude as gradient
    rn = np.linalg.norm(result)
    gn = np.linalg.norm(g_vec)
    if rn > 1e-15:
        result *= gn / rn
    return result


def full_muon_step(grads_list):
    """Full polar factor per layer."""
    step_layers = []
    for G in grads_list:
        U, _, Vt = np.linalg.svd(G, full_matrices=False)
        step_layers.append(U @ Vt)
    return pack(step_layers)


def curvature_weighted_alignment(update_vec, eigvecs, eigvals):
    """
    Compute sum_i |lambda_i| * |<update, v_i>|^2 / ||update||^2.
    Higher = more energy in high-curvature directions.
    """
    projs = eigvecs.T @ update_vec
    nrm = np.linalg.norm(update_vec)
    if nrm < 1e-15:
        return 0.0
    projs_normalized = projs / nrm
    return np.sum(np.abs(eigvals) * projs_normalized**2)


def energy_distribution(update_vec, eigvecs, n_params):
    """Return fraction of energy in top/middle/bottom thirds of Hessian spectrum."""
    projs = eigvecs.T @ update_vec
    energy = projs**2
    total = np.sum(energy) + 1e-30
    third = n_params // 3
    bottom = np.sum(energy[:third]) / total
    middle = np.sum(energy[third:2*third]) / total
    top = np.sum(energy[2*third:]) / total
    return bottom, middle, top


def main():
    seeds = [42 + i * 137 for i in range(NUM_SEEDS)]

    print("=" * 100)
    print("H19b: HESSIAN-BASIS vs SV-BASIS PARTIAL EQUALIZATION")
    print("=" * 100)
    print(f"Network: {N_LAYERS}-layer deep linear {DIM}x{DIM} ({N_PARAMS} params)")
    print()

    methods = {
        'SGD': None,
        'SV_partial_k2': 2,
        'SV_full_k4': 4,
        'Hessian_partial_k10': 10,
        'Hessian_full_k24': 24,
        'Muon': None,
    }

    # Accumulators
    curv_alignments = {m: [] for m in methods}
    energy_top = {m: [] for m in methods}
    energy_bot = {m: [] for m in methods}

    for si, seed in enumerate(seeds):
        rng = np.random.RandomState(seed)
        X = rng.randn(DIM, 64) * 0.3
        Y = rng.randn(DIM, 64) * 0.3
        weights = [np.eye(DIM) + rng.randn(DIM, DIM) * 0.1 for _ in range(N_LAYERS)]
        theta = pack(weights)

        # Warmup with SGD
        mom_vec = np.zeros_like(theta)
        for step in range(WARMUP_STEPS):
            g, _ = grad_fn(theta, X, Y)
            mom_vec = MOMENTUM * mom_vec + g
            theta = theta - LR * mom_vec

        # Compute Hessian at this point
        H = hessian_fd(theta, X, Y)
        eigvals, eigvecs = np.linalg.eigh(H)
        g, grads_list = grad_fn(theta, X, Y)

        print(f"\n  Seed {si+1}: loss={loss_fn(theta, X, Y):.6f}, "
              f"kappa(H)={eigvals[-1]/(abs(eigvals[0])+1e-15):.1f}")

        for method_name in methods:
            if method_name == 'SGD':
                update = g
            elif method_name == 'SV_partial_k2':
                update = partial_sv_eq_step(grads_list, 2)
            elif method_name == 'SV_full_k4':
                update = partial_sv_eq_step(grads_list, 4)
            elif method_name == 'Hessian_partial_k10':
                update = partial_hessian_eq_step(g, eigvecs, eigvals, 10)
            elif method_name == 'Hessian_full_k24':
                update = partial_hessian_eq_step(g, eigvecs, eigvals, 24)
            elif method_name == 'Muon':
                update = full_muon_step(grads_list)

            ca = curvature_weighted_alignment(update, eigvecs, eigvals)
            bot, mid, top = energy_distribution(update, eigvecs, N_PARAMS)

            curv_alignments[method_name].append(ca)
            energy_top[method_name].append(top)
            energy_bot[method_name].append(bot)

    # =========================================================================
    # RESULTS
    # =========================================================================
    print(f"\n\n{'='*100}")
    print("RESULTS: CURVATURE-WEIGHTED ALIGNMENT AND ENERGY DISTRIBUTION")
    print(f"{'='*100}")

    print(f"\n  {'Method':<25}  {'Curv Alignment':>15}  {'Energy Top-1/3':>15}  {'Energy Bot-1/3':>15}")
    print("  " + "-" * 75)
    for m in methods:
        ca = np.mean(curv_alignments[m])
        et = np.mean(energy_top[m])
        eb = np.mean(energy_bot[m])
        print(f"  {m:<25}  {ca:>15.4f}  {et:>15.3f}  {eb:>15.3f}")

    # Test: SV partial has more energy in bottom Hessian modes than SGD
    sv_partial_bot = np.mean(energy_bot['SV_partial_k2'])
    sgd_bot = np.mean(energy_bot['SGD'])
    hessian_partial_bot = np.mean(energy_bot['Hessian_partial_k10'])

    print(f"\n  SV partial (k=2) bottom energy: {sv_partial_bot:.3f}")
    print(f"  SGD bottom energy:              {sgd_bot:.3f}")
    print(f"  Hessian partial (k=10) bottom:  {hessian_partial_bot:.3f}")

    t1 = sv_partial_bot > sgd_bot + 0.02
    t2 = hessian_partial_bot < sv_partial_bot
    print(f"\n  T1: SV partial wastes more energy in low-curvature modes than SGD?  --> {'PASS' if t1 else 'FAIL'}")
    print(f"  T2: Hessian partial wastes less than SV partial?                     --> {'PASS' if t2 else 'FAIL'}")

    print(f"\n{'='*100}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()
