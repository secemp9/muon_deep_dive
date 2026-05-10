#!/usr/bin/env python3
"""
Experiment 3.1: Hessian-Adaptive Muon 2.0

Background: Experiment 2.11 showed that equalizing gradient projections in the
Hessian eigenbasis gives ~150% of Muon's advantage. The polar factor is a cheap
approximation to this. Now test if we can get CLOSER to that 150% with a cheap
Hessian sketch (Lanczos).

Setup: 3-layer deep linear network (4x4, 48 params), ill-conditioned target
(kappa=1000), 500 steps. Same setup as 2.11 for direct comparison.

6 optimizers:
  (a) SGD baseline
  (b) Standard Muon (polar factor, k=5 Newton-Schulz iterations)
  (c) Full Democratic SGD (equalize in exact Hessian basis -- upper bound)
  (d) Muon 2.0 (Lanczos-10): top-10 + bottom-10 Hessian eigenvectors via
      Lanczos with finite-difference Hessian-vector products, equalize those
      directions, SGD for the rest. Recompute every 50 steps.
  (e) Muon 2.0 (Lanczos-5): same but top-5 + bottom-5 (cheaper)
  (f) Muon + Hessian rescaling: polar factor step rescaled by 1/sqrt(curvature)
      in each singular direction -- cheap natural-gradient approximation.

Tracks: loss curves, matmul counts, final loss, recovery %, Pareto score.
"""

import numpy as np
import time

# ── network / problem setup ──────────────────────────────────────────
DIM = 4
N_LAYERS = 3       # W3 @ W2 @ W1 -> T   (48 params total)
N_PARAMS = N_LAYERS * DIM * DIM
N_STEPS = 500
HESSIAN_RECOMPUTE_EVERY = 50
N_SEEDS = 5
EPS_FD = 1e-5       # finite-difference epsilon for Hessian-vector products

# ── matmul counter ───────────────────────────────────────────────────
_matmul_count = 0

def counted_matmul(A, B):
    """Matrix multiply with counting."""
    global _matmul_count
    _matmul_count += 1
    return A @ B

def reset_matmul_count():
    global _matmul_count
    _matmul_count = 0

def get_matmul_count():
    return _matmul_count

# ── helpers ──────────────────────────────────────────────────────────

def pack(Ws):
    return np.concatenate([W.ravel() for W in Ws])

def unpack(theta):
    Ws = []
    idx = 0
    for _ in range(N_LAYERS):
        Ws.append(theta[idx:idx+DIM*DIM].reshape(DIM, DIM))
        idx += DIM*DIM
    return Ws

def forward(Ws, count=True):
    mm = counted_matmul if count else (lambda a, b: a @ b)
    out = Ws[0]
    for W in Ws[1:]:
        out = mm(W, out)
    return out

def loss_fn(theta, target, count=True):
    Ws = unpack(theta)
    diff = forward(Ws, count=count) - target
    return 0.5 * np.sum(diff**2)

def grad_fn(theta, target, count=True):
    """Compute gradient of loss w.r.t. theta (flat vector)."""
    Ws = unpack(theta)
    mm = counted_matmul if count else (lambda a, b: a @ b)
    prod = forward(Ws, count=count)
    R = prod - target

    grads = []
    for k in range(N_LAYERS):
        L = np.eye(DIM)
        for j in range(k+1, N_LAYERS):
            L = mm(Ws[j], L)
        Rp = np.eye(DIM)
        for j in range(0, k):
            Rp = mm(Ws[j], Rp)
        dWk = mm(L.T, mm(R, Rp.T))
        grads.append(dWk.ravel())
    return np.concatenate(grads)

def grad_matrices(theta, target, count=True):
    g = grad_fn(theta, target, count=count)
    mats = []
    for k in range(N_LAYERS):
        mats.append(g[k*DIM*DIM:(k+1)*DIM*DIM].reshape(DIM, DIM))
    return mats

# ── Hessian (exact, for Full Democratic and verification) ────────────

def hessian_fn(theta, target):
    """Full Hessian via finite differences (no matmul counting -- overhead)."""
    n = len(theta)
    H = np.zeros((n, n))
    for i in range(n):
        theta_p = theta.copy()
        theta_m = theta.copy()
        theta_p[i] += EPS_FD
        theta_m[i] -= EPS_FD
        g_p = grad_fn(theta_p, target, count=False)
        g_m = grad_fn(theta_m, target, count=False)
        H[:, i] = (g_p - g_m) / (2 * EPS_FD)
    H = 0.5 * (H + H.T)
    return H

# ── Hessian-vector product via finite differences ────────────────────

def hvp(theta, target, v):
    """Hessian-vector product Hv via central finite differences.
    Requires 2 gradient evaluations. These ARE counted for matmul cost."""
    theta_p = theta + EPS_FD * v
    theta_m = theta - EPS_FD * v
    g_p = grad_fn(theta_p, target, count=True)
    g_m = grad_fn(theta_m, target, count=True)
    return (g_p - g_m) / (2 * EPS_FD)

# ── Lanczos tridiagonalization ───────────────────────────────────────

def lanczos(theta, target, k):
    """Run k steps of Lanczos to get an approximate eigendecomposition.
    Returns (eigenvalues, eigenvectors_in_full_space) of the top-k and bottom-k
    Ritz vectors.

    Cost: k Hessian-vector products = 2k gradient evaluations.
    """
    n = len(theta)
    # Initialize with a random unit vector (seeded for reproducibility)
    rng = np.random.RandomState(int(np.abs(theta[:4]).sum() * 1000) % (2**31))
    q = rng.randn(n)
    q = q / np.linalg.norm(q)

    Q = np.zeros((n, k))
    alpha = np.zeros(k)
    beta = np.zeros(k)

    Q[:, 0] = q
    for j in range(k):
        v = hvp(theta, target, Q[:, j])
        alpha[j] = Q[:, j] @ v
        if j == 0:
            v = v - alpha[j] * Q[:, j]
        else:
            v = v - alpha[j] * Q[:, j] - beta[j] * Q[:, j-1]

        # Full reorthogonalization for numerical stability
        for jj in range(j+1):
            v = v - (Q[:, jj] @ v) * Q[:, jj]

        beta_next = np.linalg.norm(v)
        if beta_next < 1e-12:
            # Krylov subspace exhausted, pad with zeros
            if j + 1 < k:
                beta[j+1] = 0.0
            break
        if j + 1 < k:
            beta[j+1] = beta_next
            Q[:, j+1] = v / beta_next

    # Build tridiagonal matrix
    T = np.diag(alpha)
    for j in range(k-1):
        T[j, j+1] = beta[j+1]
        T[j+1, j] = beta[j+1]

    # Eigendecompose the tridiagonal
    ritz_vals, ritz_vecs_small = np.linalg.eigh(T)

    # Map back to full space
    ritz_vecs = Q @ ritz_vecs_small

    # Normalize (should already be near-unit, but ensure)
    for i in range(ritz_vecs.shape[1]):
        norm = np.linalg.norm(ritz_vecs[:, i])
        if norm > 1e-12:
            ritz_vecs[:, i] /= norm

    return ritz_vals, ritz_vecs

# ── Optimizer directions ─────────────────────────────────────────────

def polar_factor_ns(M, k=5):
    """Polar factor via Newton-Schulz iterations (k iterations)."""
    # Normalize
    a = np.linalg.norm(M, 'fro')
    if a < 1e-12:
        return M
    X = M / a
    for _ in range(k):
        X = 1.5 * X - 0.5 * counted_matmul(X, counted_matmul(X.T, X))
    return X

def polar_factor_svd(M):
    """Polar factor via SVD (exact, for reference)."""
    U, S, Vt = np.linalg.svd(M, full_matrices=True)
    return counted_matmul(U, Vt)

def muon_direction(theta, target):
    """Standard Muon: polar factor of each layer's gradient matrix."""
    gmats = grad_matrices(theta, target, count=True)
    polars = []
    for gm in gmats:
        polars.append(polar_factor_ns(gm, k=5).ravel())
    return np.concatenate(polars)

def democratic_direction(grad_vec, eigvecs):
    """Full Democratic SGD: equalize projections in Hessian eigenbasis."""
    projs = eigvecs.T @ grad_vec
    signs = np.sign(projs)
    magnitudes = np.abs(projs)
    mean_mag = np.mean(magnitudes)
    eq_projs = signs * mean_mag
    return eigvecs @ eq_projs

def muon2_lanczos_direction(grad_vec, ritz_vecs, ritz_vals):
    """Muon 2.0 (Lanczos): equalize gradient in the Lanczos subspace,
    keep SGD direction in the complement.

    ritz_vecs: (n, 2m) matrix of m top + m bottom Ritz vectors.
    """
    n = len(grad_vec)
    m = ritz_vecs.shape[1]

    # Project gradient onto Lanczos subspace
    projs = ritz_vecs.T @ grad_vec       # shape (m,)

    # Equalize magnitudes in the Lanczos subspace
    signs = np.sign(projs)
    magnitudes = np.abs(projs)
    mean_mag = np.mean(magnitudes) if np.max(magnitudes) > 1e-30 else 0.0
    eq_projs = signs * mean_mag

    # Reconstructed equalized component in Lanczos subspace
    equalized_part = ritz_vecs @ eq_projs

    # Complement: gradient minus its projection onto Lanczos subspace
    grad_in_subspace = ritz_vecs @ projs
    complement = grad_vec - grad_in_subspace

    # Full direction: equalized Lanczos + raw SGD complement
    direction = equalized_part + complement
    return direction

def muon_hessian_rescale_direction(theta, target):
    """Muon + Hessian rescaling: polar factor step rescaled by 1/sqrt(curvature)
    in each singular direction.

    For each layer gradient G = U S V^T:
      - Polar factor gives U V^T
      - Estimate curvature in each singular direction via a cheap Hv product
      - Rescale: sum_i (1/sqrt(|h_i| + eps)) * u_i v_i^T
    """
    gmats = grad_matrices(theta, target, count=True)
    directions = []

    for layer_idx, gm in enumerate(gmats):
        U, S, Vt = np.linalg.svd(gm, full_matrices=True)
        _mc = 3  # SVD counts as ~3 matmuls

        # Estimate curvature along each singular direction
        curvatures = np.zeros(DIM)
        offset = layer_idx * DIM * DIM
        for i in range(DIM):
            # Direction in parameter space corresponding to u_i v_i^T
            direction_mat = np.outer(U[:, i], Vt[i, :])
            v_full = np.zeros(N_PARAMS)
            v_full[offset:offset+DIM*DIM] = direction_mat.ravel()

            # Hessian-vector product
            Hv = hvp(theta, target, v_full)
            curvatures[i] = np.abs(v_full @ Hv) + 1e-8

        # Rescale: polar factor with 1/sqrt(curvature) weighting
        rescaled = np.zeros((DIM, DIM))
        for i in range(DIM):
            weight = 1.0 / np.sqrt(curvatures[i])
            rescaled += weight * np.outer(U[:, i], Vt[i, :])

        directions.append(rescaled.ravel())

    return np.concatenate(directions)

# ── LR sweep helper ─────────────────────────────────────────────────

def run_single(method, lr, theta0, target, lanczos_cache=None):
    """Run optimizer for N_STEPS, return final loss. No matmul counting."""
    theta = theta0.copy()
    H_eigvecs = None
    ritz_vecs_10 = None
    ritz_vals_10 = None
    ritz_vecs_5 = None
    ritz_vals_5 = None

    for step in range(N_STEPS):
        g = grad_fn(theta, target, count=False)

        if step % HESSIAN_RECOMPUTE_EVERY == 0:
            if method in ('FullDemocratic',):
                H = hessian_fn(theta, target)
                _, H_eigvecs = np.linalg.eigh(H)

            if method == 'Muon2_L10':
                ritz_vals_10, ritz_vecs_10 = lanczos_nocost(theta, target, 10)
            elif method == 'Muon2_L5':
                ritz_vals_5, ritz_vecs_5 = lanczos_nocost(theta, target, 5)

        if method == 'SGD':
            theta -= lr * g
        elif method == 'Muon':
            d = muon_direction_nocost(theta, target)
            theta -= lr * d
        elif method == 'FullDemocratic':
            g_dem = democratic_direction(g, H_eigvecs)
            gn = np.linalg.norm(g)
            dn = np.linalg.norm(g_dem)
            if dn > 1e-12:
                g_dem = g_dem * (gn / dn)
            theta -= lr * g_dem
        elif method == 'Muon2_L10':
            d = muon2_lanczos_direction(g, ritz_vecs_10, ritz_vals_10)
            gn = np.linalg.norm(g)
            dn = np.linalg.norm(d)
            if dn > 1e-12:
                d = d * (gn / dn)
            theta -= lr * d
        elif method == 'Muon2_L5':
            d = muon2_lanczos_direction(g, ritz_vecs_5, ritz_vals_5)
            gn = np.linalg.norm(g)
            dn = np.linalg.norm(d)
            if dn > 1e-12:
                d = d * (gn / dn)
            theta -= lr * d
        elif method == 'Muon+Hrescale':
            d = muon_hrescale_nocost(theta, target)
            gn = np.linalg.norm(g)
            dn = np.linalg.norm(d)
            if dn > 1e-12:
                d = d * (gn / dn)
            theta -= lr * d

        L = loss_fn(theta, target, count=False)
        if np.isnan(L) or L > 1e8:
            return 1e8
    return loss_fn(theta, target, count=False)

# Non-counting versions for LR sweep
def muon_direction_nocost(theta, target):
    gmats = grad_matrices(theta, target, count=False)
    polars = []
    for gm in gmats:
        U, S, Vt = np.linalg.svd(gm, full_matrices=True)
        X = gm / (np.linalg.norm(gm, 'fro') + 1e-12)
        for _ in range(5):
            X = 1.5 * X - 0.5 * X @ (X.T @ X)
        polars.append(X.ravel())
    return np.concatenate(polars)

def lanczos_nocost(theta, target, k):
    """Lanczos without matmul counting."""
    n = len(theta)
    rng = np.random.RandomState(int(np.abs(theta[:4]).sum() * 1000) % (2**31))
    q = rng.randn(n)
    q = q / np.linalg.norm(q)

    Q = np.zeros((n, k))
    alpha = np.zeros(k)
    beta = np.zeros(k)
    Q[:, 0] = q

    for j in range(k):
        # HVP without counting
        v_dir = Q[:, j]
        g_p = grad_fn(theta + EPS_FD * v_dir, target, count=False)
        g_m = grad_fn(theta - EPS_FD * v_dir, target, count=False)
        v = (g_p - g_m) / (2 * EPS_FD)

        alpha[j] = Q[:, j] @ v
        if j == 0:
            v = v - alpha[j] * Q[:, j]
        else:
            v = v - alpha[j] * Q[:, j] - beta[j] * Q[:, j-1]
        for jj in range(j+1):
            v = v - (Q[:, jj] @ v) * Q[:, jj]
        beta_next = np.linalg.norm(v)
        if beta_next < 1e-12:
            break
        if j + 1 < k:
            beta[j+1] = beta_next
            Q[:, j+1] = v / beta_next

    T = np.diag(alpha)
    for j in range(k-1):
        T[j, j+1] = beta[j+1]
        T[j+1, j] = beta[j+1]
    ritz_vals, ritz_vecs_small = np.linalg.eigh(T)
    ritz_vecs = Q @ ritz_vecs_small
    for i in range(ritz_vecs.shape[1]):
        norm = np.linalg.norm(ritz_vecs[:, i])
        if norm > 1e-12:
            ritz_vecs[:, i] /= norm
    return ritz_vals, ritz_vecs

def muon_hrescale_nocost(theta, target):
    gmats = grad_matrices(theta, target, count=False)
    directions = []
    for layer_idx, gm in enumerate(gmats):
        U, S, Vt = np.linalg.svd(gm, full_matrices=True)
        curvatures = np.zeros(DIM)
        offset = layer_idx * DIM * DIM
        for i in range(DIM):
            direction_mat = np.outer(U[:, i], Vt[i, :])
            v_full = np.zeros(N_PARAMS)
            v_full[offset:offset+DIM*DIM] = direction_mat.ravel()
            g_p = grad_fn(theta + EPS_FD * v_full, target, count=False)
            g_m = grad_fn(theta - EPS_FD * v_full, target, count=False)
            Hv = (g_p - g_m) / (2 * EPS_FD)
            curvatures[i] = np.abs(v_full @ Hv) + 1e-8
        rescaled = np.zeros((DIM, DIM))
        for i in range(DIM):
            weight = 1.0 / np.sqrt(curvatures[i])
            rescaled += weight * np.outer(U[:, i], Vt[i, :])
        directions.append(rescaled.ravel())
    return np.concatenate(directions)

# ── Full run with counting ───────────────────────────────────────────

def run_full_counted(method, lr, theta0, target):
    """Full run with matmul counting and loss curve."""
    reset_matmul_count()
    theta = theta0.copy()
    losses = []
    H_eigvecs = None
    ritz_vecs = None
    ritz_vals = None

    for step in range(N_STEPS):
        L = loss_fn(theta, target, count=True)
        losses.append(L)
        g = grad_fn(theta, target, count=True)

        if step % HESSIAN_RECOMPUTE_EVERY == 0:
            if method == 'FullDemocratic':
                H = hessian_fn(theta, target)
                _, H_eigvecs = np.linalg.eigh(H)

            if method == 'Muon2_L10':
                ritz_vals, ritz_vecs = lanczos(theta, target, 10)
            elif method == 'Muon2_L5':
                ritz_vals, ritz_vecs = lanczos(theta, target, 5)

        if method == 'SGD':
            theta -= lr * g
        elif method == 'Muon':
            d = muon_direction(theta, target)
            theta -= lr * d
        elif method == 'FullDemocratic':
            g_dem = democratic_direction(g, H_eigvecs)
            gn = np.linalg.norm(g)
            dn = np.linalg.norm(g_dem)
            if dn > 1e-12:
                g_dem = g_dem * (gn / dn)
            theta -= lr * g_dem
        elif method == 'Muon2_L10':
            d = muon2_lanczos_direction(g, ritz_vecs, ritz_vals)
            gn = np.linalg.norm(g)
            dn = np.linalg.norm(d)
            if dn > 1e-12:
                d = d * (gn / dn)
            theta -= lr * d
        elif method == 'Muon2_L5':
            d = muon2_lanczos_direction(g, ritz_vecs, ritz_vals)
            gn = np.linalg.norm(g)
            dn = np.linalg.norm(d)
            if dn > 1e-12:
                d = d * (gn / dn)
            theta -= lr * d
        elif method == 'Muon+Hrescale':
            d = muon_hessian_rescale_direction(theta, target)
            gn = np.linalg.norm(g)
            dn = np.linalg.norm(d)
            if dn > 1e-12:
                d = d * (gn / dn)
            theta -= lr * d

        if np.isnan(losses[-1]) or losses[-1] > 1e8:
            return np.array(losses), 1e8, get_matmul_count()

    final = loss_fn(theta, target, count=True)
    return np.array(losses), final, get_matmul_count()


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

METHOD_NAMES = ['SGD', 'Muon', 'FullDemocratic', 'Muon2_L10', 'Muon2_L5', 'Muon+Hrescale']
DISPLAY_NAMES = {
    'SGD': 'SGD (baseline)',
    'Muon': 'Muon (polar, k=5 NS)',
    'FullDemocratic': 'Full Democratic SGD',
    'Muon2_L10': 'Muon 2.0 (Lanczos-10)',
    'Muon2_L5': 'Muon 2.0 (Lanczos-5)',
    'Muon+Hrescale': 'Muon + Hessian rescale',
}

print("=" * 90)
print("Experiment 3.1: Hessian-Adaptive Muon 2.0")
print("=" * 90)
print(f"Network: {N_LAYERS}-layer deep linear {DIM}x{DIM}  ({N_PARAMS} params)")
print(f"Steps: {N_STEPS},  Hessian/Lanczos recompute every {HESSIAN_RECOMPUTE_EVERY} steps")
print(f"Seeds: {N_SEEDS}")
print(f"Target condition number: 1000")
print()

# LR candidates -- broader range for the new methods
lr_candidates = [0.0001, 0.0003, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]

all_results = {m: {'final_losses': [], 'matmuls': [], 'loss_curves': []} for m in METHOD_NAMES}

t_start = time.time()

for seed_idx in range(N_SEEDS):
    seed = 42 + seed_idx * 7
    rng_init = np.random.RandomState(seed)

    # Ill-conditioned target (kappa = 1000)
    U_t, _ = np.linalg.qr(rng_init.randn(DIM, DIM))
    V_t, _ = np.linalg.qr(rng_init.randn(DIM, DIM))
    sigma_t = np.array([100.0, 10.0, 1.0, 0.1])  # kappa = 100/0.1 = 1000
    target = U_t @ np.diag(sigma_t) @ V_t
    theta0 = 0.3 * rng_init.randn(N_PARAMS)

    kappa = np.linalg.cond(target)
    print(f"--- Seed {seed} (target kappa={kappa:.0f}) ---")

    # LR sweep for each method
    best_lrs = {}
    for method in METHOD_NAMES:
        best_loss = 1e20
        best_lr = 0.001
        for lr in lr_candidates:
            fl = run_single(method, lr, theta0, target)
            if fl < best_loss:
                best_loss = fl
                best_lr = lr
        best_lrs[method] = best_lr

    lr_str = ", ".join([f"{DISPLAY_NAMES[m].split('(')[0].strip()[:8]}={best_lrs[m]}" for m in METHOD_NAMES])
    print(f"  Best LRs: {lr_str}")

    # Full counted run
    for method in METHOD_NAMES:
        losses, final, matmuls = run_full_counted(method, best_lrs[method], theta0, target)
        all_results[method]['final_losses'].append(final)
        all_results[method]['matmuls'].append(matmuls)
        all_results[method]['loss_curves'].append(losses)

    # Quick per-seed summary
    finals = {m: all_results[m]['final_losses'][-1] for m in METHOD_NAMES}
    print(f"  Finals: " + "  ".join([f"{m[:6]}={finals[m]:.4f}" for m in METHOD_NAMES]))
    print()

elapsed = time.time() - t_start
print(f"\nTotal runtime: {elapsed:.1f}s\n")

# ── Aggregate results ────────────────────────────────────────────────
print("=" * 90)
print("AGGREGATE RESULTS")
print("=" * 90)
print()

# Compute statistics
stats = {}
for m in METHOD_NAMES:
    fl = np.array(all_results[m]['final_losses'])
    mm = np.array(all_results[m]['matmuls'])
    stats[m] = {
        'mean_loss': np.mean(fl),
        'std_loss': np.std(fl),
        'median_loss': np.median(fl),
        'mean_matmuls': np.mean(mm),
        'losses': fl,
        'matmuls': mm,
    }

# Recovery %: (loss_SGD - loss_method) / (loss_SGD - loss_FullDemocratic) * 100
# Computed per-seed then averaged
recoveries = {m: [] for m in METHOD_NAMES}
for i in range(N_SEEDS):
    sgd_l = all_results['SGD']['final_losses'][i]
    dem_l = all_results['FullDemocratic']['final_losses'][i]
    gap = sgd_l - dem_l
    for m in METHOD_NAMES:
        ml = all_results[m]['final_losses'][i]
        if abs(gap) > 1e-12:
            rec = (sgd_l - ml) / gap * 100.0
        else:
            rec = 0.0
        recoveries[m].append(rec)

for m in METHOD_NAMES:
    stats[m]['mean_recovery'] = np.mean(recoveries[m])
    stats[m]['std_recovery'] = np.std(recoveries[m])
    stats[m]['recoveries'] = np.array(recoveries[m])

# Pareto score: Recovery% / (matmuls / matmuls_SGD)
# Higher is better: high recovery at low cost
for m in METHOD_NAMES:
    cost_ratio = stats[m]['mean_matmuls'] / stats['SGD']['mean_matmuls']
    stats[m]['cost_ratio'] = cost_ratio
    if cost_ratio > 0:
        stats[m]['pareto_score'] = stats[m]['mean_recovery'] / cost_ratio
    else:
        stats[m]['pareto_score'] = 0.0

# ── Print main results table ─────────────────────────────────────────
print(f"{'Method':<28} {'Final Loss':>12} {'+-Std':>10} {'Matmuls':>10} {'Recovery%':>12} {'Pareto':>10}")
print("-" * 90)
for m in METHOD_NAMES:
    s = stats[m]
    print(f"{DISPLAY_NAMES[m]:<28} {s['mean_loss']:>12.6f} {s['std_loss']:>10.6f} "
          f"{s['mean_matmuls']:>10.0f} {s['mean_recovery']:>11.1f}% {s['pareto_score']:>10.1f}")
print()

# ── Per-seed recovery table ──────────────────────────────────────────
print("Per-seed recovery % (relative to Full Democratic):")
print(f"{'Method':<28}", end="")
for i in range(N_SEEDS):
    print(f" {'Seed'+str(i):>8}", end="")
print(f" {'Mean':>8}")
print("-" * (28 + 9 * (N_SEEDS + 1)))
for m in METHOD_NAMES:
    print(f"{DISPLAY_NAMES[m]:<28}", end="")
    for i in range(N_SEEDS):
        print(f" {recoveries[m][i]:>7.1f}%", end="")
    print(f" {stats[m]['mean_recovery']:>7.1f}%")
print()

# ── Loss trajectory (seed 0) ─────────────────────────────────────────
print("Loss trajectory (seed 0, every 50 steps):")
print(f"{'Step':>5}", end="")
for m in METHOD_NAMES:
    print(f" {DISPLAY_NAMES[m][:14]:>14}", end="")
print()
for s in list(range(0, N_STEPS, 50)) + [N_STEPS - 1]:
    print(f"{s:>5}", end="")
    for m in METHOD_NAMES:
        lc = all_results[m]['loss_curves'][0]
        if s < len(lc):
            print(f" {lc[s]:>14.6f}", end="")
        else:
            print(f" {'N/A':>14}", end="")
    print()
print()

# ── Matmul cost breakdown ────────────────────────────────────────────
print("Matmul cost analysis:")
print(f"{'Method':<28} {'Total matmuls':>14} {'Cost ratio':>12} {'Notes'}")
print("-" * 80)
for m in METHOD_NAMES:
    s = stats[m]
    notes = ""
    if m == 'SGD':
        notes = "baseline (1 grad/step)"
    elif m == 'Muon':
        notes = "1 grad + k=5 NS iters/step"
    elif m == 'FullDemocratic':
        notes = "1 grad + full Hessian every 50 steps (ORACLE)"
    elif m == 'Muon2_L10':
        notes = "1 grad + 10 Lanczos steps every 50 steps"
    elif m == 'Muon2_L5':
        notes = "1 grad + 5 Lanczos steps every 50 steps"
    elif m == 'Muon+Hrescale':
        notes = "Muon + 4 HVPs/layer/step"
    print(f"{DISPLAY_NAMES[m]:<28} {s['mean_matmuls']:>14.0f} {s['cost_ratio']:>11.2f}x  {notes}")
print()

# ── Key hypothesis tests ─────────────────────────────────────────────
print("=" * 90)
print("KEY HYPOTHESIS TESTS")
print("=" * 90)
print()

# T1: Muon 2.0 (Lanczos-10) recovery > 80%?
rec_l10 = stats['Muon2_L10']['mean_recovery']
t1_pass = rec_l10 > 80.0
print(f"T1: Muon 2.0 (Lanczos-10) recovers >80% of Full Democratic advantage?")
print(f"    Mean recovery = {rec_l10:.1f}%  (per-seed: {stats['Muon2_L10']['recoveries'].round(1)})")
print(f"    --> {'PASS' if t1_pass else 'FAIL'}")
print()

# T2: Muon 2.0 (Lanczos-10) beats standard Muon?
rec_muon = stats['Muon']['mean_recovery']
t2_pass = rec_l10 > rec_muon
print(f"T2: Muon 2.0 (Lanczos-10) beats standard Muon?")
print(f"    Muon 2.0 recovery = {rec_l10:.1f}%  vs  Muon recovery = {rec_muon:.1f}%")
print(f"    --> {'PASS' if t2_pass else 'FAIL'}")
print()

# T3: Muon 2.0 (Lanczos-5) -- does even the cheap version help?
rec_l5 = stats['Muon2_L5']['mean_recovery']
t3_pass = rec_l5 > rec_muon
print(f"T3: Muon 2.0 (Lanczos-5) beats standard Muon?")
print(f"    Muon 2.0 (L5) recovery = {rec_l5:.1f}%  vs  Muon recovery = {rec_muon:.1f}%")
print(f"    --> {'PASS' if t3_pass else 'FAIL'}")
print()

# T4: Muon + Hessian rescaling beats standard Muon?
rec_hrescale = stats['Muon+Hrescale']['mean_recovery']
t4_pass = rec_hrescale > rec_muon
print(f"T4: Muon + Hessian rescaling beats standard Muon?")
print(f"    Muon+Hrescale recovery = {rec_hrescale:.1f}%  vs  Muon recovery = {rec_muon:.1f}%")
print(f"    --> {'PASS' if t4_pass else 'FAIL'}")
print()

# T5: Best Pareto-efficient method?
best_pareto = max(METHOD_NAMES, key=lambda m: stats[m]['pareto_score'])
print(f"T5: Best Pareto score (recovery / relative cost)?")
for m in METHOD_NAMES:
    marker = " <-- BEST" if m == best_pareto else ""
    print(f"    {DISPLAY_NAMES[m]:<28} Pareto = {stats[m]['pareto_score']:.1f}{marker}")
print()

# ── Summary verdict ──────────────────────────────────────────────────
print("=" * 90)
print("SUMMARY VERDICT")
print("=" * 90)
print()

n_pass = sum([t1_pass, t2_pass, t3_pass, t4_pass])
print(f"Tests passed: {n_pass}/4")
print()

if t1_pass and t2_pass:
    print("STRONG RESULT: Muon 2.0 (Lanczos-10) recovers >80% of the full democratic")
    print("advantage AND beats standard Muon. The Hessian sketch approach works.")
elif t2_pass:
    print("POSITIVE RESULT: Muon 2.0 (Lanczos-10) beats standard Muon, though it does")
    print("not reach 80% of full democratic advantage. The Lanczos sketch helps but")
    print("more eigenvectors or more frequent updates may be needed.")
elif t1_pass:
    print("MIXED: High recovery but doesn't beat Muon -- standard Muon may already be")
    print("near-optimal for this problem size.")
else:
    print("NEGATIVE: The Lanczos sketch approach did not significantly improve over Muon")
    print("in this setting. Possible reasons: too few eigenvectors, stale Hessian info,")
    print("or the polar factor is already capturing the essential structure.")

print()
if t4_pass:
    print("BONUS: Muon + Hessian rescaling improves over vanilla Muon, suggesting that")
    print("curvature-aware step sizing complements the polar factor's direction choice.")
else:
    print("NOTE: Muon + Hessian rescaling did NOT improve over vanilla Muon. The polar")
    print("factor may already provide sufficient curvature adaptation, or the rescaling")
    print("scheme needs refinement.")

print()
print(f"Best Pareto method: {DISPLAY_NAMES[best_pareto]} (score={stats[best_pareto]['pareto_score']:.1f})")
print()

# Final comparison with 2.11
print("Comparison with Experiment 2.11 results:")
print(f"  2.11 showed Full Democratic SGD at ~150% of Muon's advantage")
print(f"  Here: Muon 2.0 (Lanczos-10) at {rec_l10:.0f}% of Full Democratic = "
      f"{rec_l10 * 1.0:.0f}% of Full Democratic, which is {rec_l10 / 100.0 * 150.0:.0f}% of Muon's advantage")
print(f"  Standard Muon at {rec_muon:.0f}% of Full Democratic (={rec_muon / 100.0 * 150.0:.0f}% of Muon's advantage by 2.11 metric)")
print()
print("=" * 90)
