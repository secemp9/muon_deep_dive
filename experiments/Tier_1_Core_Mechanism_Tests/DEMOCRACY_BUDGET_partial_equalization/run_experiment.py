#!/usr/bin/env python3
"""
Experiment 3.2: Democracy Budget -- Partial Equalization
=========================================================

FROM 2.11: Full equalization in Hessian basis gives ~150% recovery of Muon's
advantage over SGD. How many eigenvectors ACTUALLY need equalizing?

PROTOCOL:
  Same 3-layer 4x4 deep linear setup as 2.11 (48 params).
  Sweep: equalize only the top-k + bottom-k eigenvectors for
  k in {1, 2, 3, 5, 10, 15, 24(=all)}. Leave the middle ones untouched.

  "Equalize" means: project gradient onto Hessian eigenbasis, set the
  magnitudes of the selected components to the mean magnitude, keep signs.
  Middle components keep their original projection magnitudes.

  Plot recovery % vs k. Find minimum k for >100% recovery (beating Muon).

SETUP:
  - 3-layer deep linear, 4x4, 48 params
  - Ill-conditioned target (kappa=1000)
  - 500 steps, 5 seeds for robustness
  - Hessian recomputed every 50 steps
  - LR individually optimized per method
"""

import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

DIM = 4
N_LAYERS = 3
N_PARAMS = N_LAYERS * DIM * DIM  # 48
N_STEPS = 500
HESSIAN_RECOMPUTE_EVERY = 50
N_SEEDS = 5
K_VALUES = [1, 2, 3, 5, 10, 15, 24]  # 24 = half of 48 = all

LR_CANDIDATES = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]


# =============================================================================
# NETWORK / LOSS / GRAD / HESSIAN
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

def forward(Ws):
    out = Ws[0]
    for W in Ws[1:]:
        out = W @ out
    return out

def loss_fn(theta, target):
    Ws = unpack(theta)
    diff = forward(Ws) - target
    return 0.5 * np.sum(diff ** 2)

def grad_fn(theta, target):
    Ws = unpack(theta)
    prod = forward(Ws)
    R = prod - target
    grads = []
    for k in range(N_LAYERS):
        L = np.eye(DIM)
        for j in range(k + 1, N_LAYERS):
            L = Ws[j] @ L
        Rp = np.eye(DIM)
        for j in range(0, k):
            Rp = Ws[j] @ Rp
        dWk = L.T @ R @ Rp.T
        grads.append(dWk.ravel())
    return np.concatenate(grads)

def grad_matrices(theta, target):
    g = grad_fn(theta, target)
    mats = []
    for k in range(N_LAYERS):
        mats.append(g[k * DIM * DIM:(k + 1) * DIM * DIM].reshape(DIM, DIM))
    return mats

def hessian_fn(theta, target):
    n = len(theta)
    H = np.zeros((n, n))
    eps = 1e-5
    for i in range(n):
        theta_p = theta.copy()
        theta_m = theta.copy()
        theta_p[i] += eps
        theta_m[i] -= eps
        g_p = grad_fn(theta_p, target)
        g_m = grad_fn(theta_m, target)
        H[:, i] = (g_p - g_m) / (2 * eps)
    H = 0.5 * (H + H.T)
    return H


# =============================================================================
# MUON DIRECTION (polar factor per layer)
# =============================================================================

def polar_factor_svd(M):
    U, S, Vt = np.linalg.svd(M, full_matrices=True)
    return U @ Vt

def muon_direction(theta, target):
    gmats = grad_matrices(theta, target)
    polars = []
    for gm in gmats:
        polars.append(polar_factor_svd(gm).ravel())
    return np.concatenate(polars)


# =============================================================================
# PARTIAL DEMOCRATIC DIRECTION
# =============================================================================

def partial_democratic_direction(grad_vec, eigvecs, eigvals, k):
    """
    Equalize only the top-k and bottom-k Hessian eigenvector components.
    Leave the middle ones untouched.

    eigvals are sorted ascending (from np.linalg.eigh).
    Bottom-k = indices 0..k-1 (smallest eigenvalues)
    Top-k = indices (n-k)..n-1 (largest eigenvalues)
    """
    n = len(grad_vec)
    projs = eigvecs.T @ grad_vec  # projections onto each eigenvector

    # Identify which indices to equalize
    if 2 * k >= n:
        # Equalize all
        selected = list(range(n))
    else:
        bottom_k = list(range(k))
        top_k = list(range(n - k, n))
        selected = bottom_k + top_k

    # Compute mean magnitude of the selected components
    selected_mags = np.abs(projs[selected])
    mean_mag = np.mean(selected_mags)

    # Build equalized projection vector
    eq_projs = projs.copy()
    for idx in selected:
        eq_projs[idx] = np.sign(projs[idx]) * mean_mag

    return eigvecs @ eq_projs


def full_democratic_direction(grad_vec, eigvecs):
    """Full equalization: all components get mean magnitude."""
    projs = eigvecs.T @ grad_vec
    signs = np.sign(projs)
    mean_mag = np.mean(np.abs(projs))
    eq_projs = signs * mean_mag
    return eigvecs @ eq_projs


# =============================================================================
# TRAINING ENGINE
# =============================================================================

def run_method(method, lr, theta0, target, k=None, seed_rb=999):
    """Run optimizer and return trajectory of losses."""
    theta = theta0.copy()
    H_eigvecs = None
    H_eigvals = None
    losses = []

    for step in range(N_STEPS):
        L = loss_fn(theta, target)
        losses.append(L)
        if np.isnan(L) or L > 1e8:
            # Pad with last value
            losses.extend([1e8] * (N_STEPS - step - 1))
            break

        g = grad_fn(theta, target)

        if step % HESSIAN_RECOMPUTE_EVERY == 0:
            H = hessian_fn(theta, target)
            H_eigvals, H_eigvecs = np.linalg.eigh(H)

        if method == 'SGD':
            direction = g
        elif method == 'Muon':
            direction = muon_direction(theta, target)
        elif method == 'DemocraticSGD_full':
            direction = full_democratic_direction(g, H_eigvecs)
            # Normalize to same magnitude as gradient
            dn = np.linalg.norm(direction)
            gn = np.linalg.norm(g)
            if dn > 1e-12:
                direction = direction * (gn / dn)
        elif method == 'PartialDemocratic':
            direction = partial_democratic_direction(g, H_eigvecs, H_eigvals, k)
            dn = np.linalg.norm(direction)
            gn = np.linalg.norm(g)
            if dn > 1e-12:
                direction = direction * (gn / dn)

        theta -= lr * direction

    return np.array(losses)


def find_best_lr(method, theta0, target, k=None, seed_rb=999):
    """Grid search for best LR."""
    best_loss = 1e20
    best_lr = 0.001
    for lr in LR_CANDIDATES:
        losses = run_method(method, lr, theta0, target, k=k, seed_rb=seed_rb)
        final_loss = losses[-1]
        if final_loss < best_loss:
            best_loss = final_loss
            best_lr = lr
    return best_lr


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

print("=" * 80)
print("Experiment 3.2: Democracy Budget -- Partial Equalization")
print("=" * 80)
print(f"Network: {N_LAYERS}-layer deep linear {DIM}x{DIM}  ({N_PARAMS} params)")
print(f"Steps: {N_STEPS},  Hessian recompute every {HESSIAN_RECOMPUTE_EVERY}")
print(f"Seeds: {N_SEEDS}")
print(f"k values: {K_VALUES}")
print(f"Total eigenvectors: {N_PARAMS} (equalize top-k + bottom-k)")
print()

# Per-seed results
all_sgd_finals = []
all_muon_finals = []
all_dem_full_finals = []
all_partial_finals = {k: [] for k in K_VALUES}
all_recoveries = {k: [] for k in K_VALUES}
all_dem_full_recoveries = []

for seed_idx in range(N_SEEDS):
    seed = 42 + seed_idx * 7
    rng_init = np.random.RandomState(seed)

    # Ill-conditioned target
    U_t, _ = np.linalg.qr(rng_init.randn(DIM, DIM))
    V_t, _ = np.linalg.qr(rng_init.randn(DIM, DIM))
    sigma_t = np.array([100.0, 10.0, 1.0, 0.1])
    target = U_t @ np.diag(sigma_t) @ V_t
    theta0 = 0.3 * rng_init.randn(N_PARAMS)

    print(f"--- Seed {seed} (target cond={np.linalg.cond(target):.0f}) ---")

    # Find best LRs
    lr_sgd = find_best_lr('SGD', theta0, target)
    lr_muon = find_best_lr('Muon', theta0, target)
    lr_dem_full = find_best_lr('DemocraticSGD_full', theta0, target)

    lrs_partial = {}
    for k in K_VALUES:
        lrs_partial[k] = find_best_lr('PartialDemocratic', theta0, target, k=k)

    print(f"  LRs: SGD={lr_sgd}, Muon={lr_muon}, DemFull={lr_dem_full}")
    print(f"  Partial LRs: {dict((k, lrs_partial[k]) for k in K_VALUES)}")

    # Run all methods
    sgd_losses = run_method('SGD', lr_sgd, theta0, target)
    muon_losses = run_method('Muon', lr_muon, theta0, target)
    dem_full_losses = run_method('DemocraticSGD_full', lr_dem_full, theta0, target)

    sgd_final = sgd_losses[-1]
    muon_final = muon_losses[-1]
    dem_full_final = dem_full_losses[-1]
    gap = sgd_final - muon_final

    all_sgd_finals.append(sgd_final)
    all_muon_finals.append(muon_final)
    all_dem_full_finals.append(dem_full_final)

    if gap > 1e-12:
        dem_full_rec = (sgd_final - dem_full_final) / gap * 100.0
    else:
        dem_full_rec = 0.0
    all_dem_full_recoveries.append(dem_full_rec)

    print(f"  SGD={sgd_final:.4f}, Muon={muon_final:.4f}, DemFull={dem_full_final:.4f} (rec={dem_full_rec:.1f}%)")

    for k in K_VALUES:
        partial_losses = run_method('PartialDemocratic', lrs_partial[k], theta0, target, k=k)
        partial_final = partial_losses[-1]
        all_partial_finals[k].append(partial_final)

        if gap > 1e-12:
            recovery = (sgd_final - partial_final) / gap * 100.0
        else:
            recovery = 0.0
        all_recoveries[k].append(recovery)

    k_rec_str = ", ".join([f"k={k}:{np.mean(all_recoveries[k][-1:]):.1f}%" for k in K_VALUES])
    print(f"  Partial: {k_rec_str}")
    print()


# =============================================================================
# AGGREGATE RESULTS
# =============================================================================

print("\n" + "=" * 80)
print("AGGREGATE RESULTS ACROSS ALL SEEDS")
print("=" * 80)

print(f"\n{'Method':<30} {'Mean Final Loss':>16} {'Std':>10} {'Mean Recovery':>15}")
print("-" * 75)

sgd_mean = np.mean(all_sgd_finals)
muon_mean = np.mean(all_muon_finals)
dem_full_mean = np.mean(all_dem_full_finals)
dem_full_rec_mean = np.mean(all_dem_full_recoveries)

print(f"{'SGD':<30} {sgd_mean:>16.6f} {np.std(all_sgd_finals):>10.6f} {'(baseline)':>15}")
print(f"{'Muon':<30} {muon_mean:>16.6f} {np.std(all_muon_finals):>10.6f} {'(reference)':>15}")
print(f"{'Democratic SGD (full)':<30} {dem_full_mean:>16.6f} {np.std(all_dem_full_finals):>10.6f} {dem_full_rec_mean:>14.1f}%")

for k in K_VALUES:
    mean_final = np.mean(all_partial_finals[k])
    std_final = np.std(all_partial_finals[k])
    mean_rec = np.mean(all_recoveries[k])
    n_equalized = min(2 * k, N_PARAMS)
    pct_equalized = 100.0 * n_equalized / N_PARAMS
    label = f"Partial k={k} ({n_equalized}/{N_PARAMS}={pct_equalized:.0f}%)"
    print(f"{label:<30} {mean_final:>16.6f} {std_final:>10.6f} {mean_rec:>14.1f}%")


# =============================================================================
# RECOVERY vs K TABLE
# =============================================================================

print(f"\n\n{'=' * 80}")
print("RECOVERY % vs k (number of extreme eigenvectors equalized)")
print(f"{'=' * 80}")

print(f"\n{'k':>4} {'Equalized':>10} {'% of N':>8} {'Recovery %':>12} {'Per-seed recoveries'}")
print("-" * 70)

min_k_100 = None
for k in K_VALUES:
    n_eq = min(2 * k, N_PARAMS)
    pct = 100.0 * n_eq / N_PARAMS
    rec = np.mean(all_recoveries[k])
    per_seed = [f"{r:.1f}" for r in all_recoveries[k]]
    print(f"{k:>4} {n_eq:>10} {pct:>7.0f}% {rec:>11.1f}% {'  '.join(per_seed):>30}")

    if min_k_100 is None and rec > 100.0:
        min_k_100 = k


# =============================================================================
# HYPOTHESIS TESTS
# =============================================================================

print(f"\n\n{'=' * 80}")
print("HYPOTHESIS TESTS")
print(f"{'=' * 80}")

# T1: Full equalization should beat Muon (recovery > 100%)
t1 = dem_full_rec_mean > 100.0
print(f"\nT1: Full equalization > 100% recovery?")
print(f"    Mean recovery = {dem_full_rec_mean:.1f}%")
print(f"    --> {'PASS' if t1 else 'FAIL'}")

# T2: Recovery should increase with k
recoveries_by_k = [np.mean(all_recoveries[k]) for k in K_VALUES]
monotonic_count = sum(1 for i in range(1, len(recoveries_by_k)) if recoveries_by_k[i] >= recoveries_by_k[i-1] - 5)
t2 = monotonic_count >= len(K_VALUES) - 2  # allow 1 non-monotonicity
print(f"\nT2: Recovery increases with k (mostly monotonic)?")
print(f"    Monotonic transitions: {monotonic_count}/{len(K_VALUES)-1}")
print(f"    --> {'PASS' if t2 else 'FAIL'}")

# T3: k=1 should give substantially less than full recovery
t3 = np.mean(all_recoveries[1]) < np.mean(all_recoveries[K_VALUES[-1]]) - 10
print(f"\nT3: k=1 substantially less effective than k=all?")
print(f"    k=1: {np.mean(all_recoveries[1]):.1f}%, k={K_VALUES[-1]}: {np.mean(all_recoveries[K_VALUES[-1]]):.1f}%")
print(f"    --> {'PASS' if t3 else 'FAIL'}")

# T4: Minimum k for >100% recovery
if min_k_100 is not None:
    print(f"\nT4: Minimum k for >100% recovery = {min_k_100}")
    print(f"    That is {min(2*min_k_100, N_PARAMS)}/{N_PARAMS} = "
          f"{100.0*min(2*min_k_100, N_PARAMS)/N_PARAMS:.0f}% of eigenvectors")
else:
    print(f"\nT4: No k achieved >100% recovery in means")
    # Check if any k is close
    best_k = K_VALUES[np.argmax(recoveries_by_k)]
    print(f"    Best: k={best_k} with {np.max(recoveries_by_k):.1f}% recovery")


# =============================================================================
# FINAL VERDICT
# =============================================================================

print(f"\n\n{'=' * 80}")
print("FINAL VERDICT: DEMOCRACY BUDGET")
print(f"{'=' * 80}")

print(f"""
  QUESTION: How many Hessian eigenvectors need equalizing to match Muon?

  Full equalization recovery: {dem_full_rec_mean:.1f}%
  Minimum k for >100%:       {min_k_100 if min_k_100 else 'N/A'}
""")

print("  Recovery curve:")
for k in K_VALUES:
    n_eq = min(2 * k, N_PARAMS)
    rec = np.mean(all_recoveries[k])
    bar = '#' * int(rec / 5) if rec > 0 else ''
    marker = " <-- beats Muon" if rec > 100 else ""
    print(f"    k={k:>2} ({n_eq:>2} eigvecs): {rec:>6.1f}% {bar}{marker}")

print()
total_pass = sum([t1, t2, t3])
if total_pass >= 2:
    print("  CONCLUSION: Spectral democracy is a graduated effect.")
    print("  Equalizing extreme eigenvalue components is most impactful.")
else:
    print("  CONCLUSION: Results need further investigation.")

print("=" * 80)
