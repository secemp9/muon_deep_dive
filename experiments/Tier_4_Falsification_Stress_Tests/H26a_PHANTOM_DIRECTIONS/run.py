#!/usr/bin/env python3
"""
H26a: Phantom Directions

Hypothesis: At high target kappa, gradient is low-rank (top few SVs dominate).
Muon's polar factor creates full-rank output. The "phantom" directions (those NOT
in the gradient's top subspace) happen to align with the Newton residual.

Setup: 4-layer 32x32 deep linear, target kappa in {1, 10, 100}. At step 50:
- Gradient SVD -> top-5 right singular vectors = "gradient subspace"
- Newton direction H^{-1}g (or pseudoinverse if H singular)
- Project Newton onto COMPLEMENT of gradient's top-5 subspace = "Newton residual"
- Muon step's projection onto same complement = "phantom component"
- Measure: cos(phantom_component, Newton_residual). Random baseline = 1/sqrt(dim-5)

Key test: does phantom-Newton cos > 0.3 at kappa=100? Does it increase with kappa?
"""

import numpy as np

np.random.seed(42)


def newton_schulz_muon(G, steps=5):
    """Newton-Schulz iteration to approximate polar factor U of G = U S V^T."""
    # Normalize
    a = G / (np.linalg.norm(G, ord='fro') + 1e-12)
    for _ in range(steps):
        a = 1.5 * a - 0.5 * a @ a.T @ a
    return a


def make_target(dim, kappa):
    """Create target matrix with condition number kappa."""
    U, _ = np.linalg.qr(np.random.randn(dim, dim))
    V, _ = np.linalg.qr(np.random.randn(dim, dim))
    svs = np.linspace(1.0, 1.0 / kappa, dim)
    return U @ np.diag(svs) @ V.T


def deep_linear_forward(weights):
    """Product of weight matrices."""
    result = weights[0]
    for W in weights[1:]:
        result = W @ result
    return result


def compute_loss(weights, target):
    """MSE loss between product and target."""
    prod = deep_linear_forward(weights)
    diff = prod - target
    return 0.5 * np.sum(diff ** 2)


def compute_gradient(weights, target, layer_idx):
    """Gradient of loss w.r.t. weights[layer_idx] for deep linear net."""
    L = len(weights)
    prod = deep_linear_forward(weights)
    diff = prod - target  # d(loss)/d(product)

    # For layer l: grad_l = (W_{L-1} ... W_{l+1})^T @ diff @ (W_{l-1} ... W_0)^T
    # Left factor: product of layers above
    left = np.eye(weights[0].shape[0])
    for i in range(L - 1, layer_idx, -1):
        left = weights[i].T @ left

    # Right factor: product of layers below
    right = np.eye(weights[0].shape[1])
    for i in range(layer_idx - 1, -1, -1):
        right = right @ weights[i].T

    grad = left @ diff @ right
    return grad


def flatten_weights(weights):
    """Flatten list of weight matrices to single vector."""
    return np.concatenate([W.flatten() for W in weights])


def unflatten_weights(vec, shapes):
    """Unflatten vector back to list of weight matrices."""
    weights = []
    idx = 0
    for shape in shapes:
        size = shape[0] * shape[1]
        weights.append(vec[idx:idx + size].reshape(shape))
        idx += size
    return weights


def compute_full_gradient(weights, target):
    """Compute gradient as flattened vector."""
    grads = []
    for l in range(len(weights)):
        g = compute_gradient(weights, target, l)
        grads.append(g.flatten())
    return np.concatenate(grads)


def compute_hessian_numerical(weights, target, eps=1e-5):
    """Numerical Hessian via finite differences."""
    shapes = [W.shape for W in weights]
    vec = flatten_weights(weights)
    n = len(vec)
    H = np.zeros((n, n))

    f0 = compute_loss(weights, target)

    for i in range(n):
        vec_p = vec.copy()
        vec_p[i] += eps
        wp = unflatten_weights(vec_p, shapes)
        fp = compute_loss(wp, target)

        vec_m = vec.copy()
        vec_m[i] -= eps
        wm = unflatten_weights(vec_m, shapes)
        fm = compute_loss(wm, target)

        H[i, :] = 0  # will fill column by column below

    # Actually do full Hessian
    for i in range(n):
        for j in range(i, n):
            vec_pp = vec.copy()
            vec_pp[i] += eps
            vec_pp[j] += eps

            vec_pm = vec.copy()
            vec_pm[i] += eps
            vec_pm[j] -= eps

            vec_mp = vec.copy()
            vec_mp[i] -= eps
            vec_mp[j] += eps

            vec_mm = vec.copy()
            vec_mm[i] -= eps
            vec_mm[j] -= eps

            fpp = compute_loss(unflatten_weights(vec_pp, shapes), target)
            fpm = compute_loss(unflatten_weights(vec_pm, shapes), target)
            fmp = compute_loss(unflatten_weights(vec_mp, shapes), target)
            fmm = compute_loss(unflatten_weights(vec_mm, shapes), target)

            H[i, j] = (fpp - fpm - fmp + fmm) / (4 * eps * eps)
            H[j, i] = H[i, j]

    return H


def run_experiment(kappa, dim=32, n_layers=4, lr=0.001, n_steps=50, top_k=5):
    """Run experiment for a given target condition number."""
    print(f"\n{'='*60}")
    print(f"  Target kappa = {kappa}")
    print(f"{'='*60}")

    target = make_target(dim, kappa)

    # Initialize weights close to identity (scaled)
    weights = []
    for _ in range(n_layers):
        W = np.eye(dim) + 0.01 * np.random.randn(dim, dim)
        weights.append(W)

    # Train for n_steps with SGD
    for step in range(n_steps):
        grads = [compute_gradient(weights, target, l) for l in range(n_layers)]
        for l in range(n_layers):
            weights[l] -= lr * grads[l]

    # At step 50, analyze
    print(f"  Loss at step {n_steps}: {compute_loss(weights, target):.6f}")

    # Pick layer 1 for analysis (middle layer)
    layer_idx = 1
    G = compute_gradient(weights, target, layer_idx)

    # Gradient SVD
    U_g, S_g, Vt_g = np.linalg.svd(G, full_matrices=True)
    print(f"  Gradient SV spectrum (top 10): {S_g[:10].round(4)}")
    print(f"  Gradient SV ratio (s1/s5): {S_g[0]/(S_g[4]+1e-12):.2f}")

    # Top-k right singular vectors span the "gradient subspace"
    V_top = Vt_g[:top_k, :].T  # dim x top_k

    # Projector onto gradient's top-k subspace
    P_top = V_top @ V_top.T  # dim x dim
    # Projector onto complement
    P_comp = np.eye(dim) - P_top

    # --- Newton direction (for this layer, treating it as isolated) ---
    # We need the Hessian w.r.t. this layer's parameters
    # For deep linear: use numerical Hessian on flattened layer params
    # But full Hessian is dim^2 x dim^2 = 1024x1024 for 32x32 -- feasible
    vec_l = weights[layer_idx].flatten()
    n_params = len(vec_l)
    eps = 1e-5

    # Compute Hessian w.r.t. this layer only
    H_layer = np.zeros((n_params, n_params))
    for i in range(n_params):
        for j in range(i, n_params):
            w_pp = weights[layer_idx].copy().flatten()
            w_pp[i] += eps
            w_pp[j] += eps
            weights_pp = weights.copy()
            weights_pp[layer_idx] = w_pp.reshape(dim, dim)

            w_pm = weights[layer_idx].copy().flatten()
            w_pm[i] += eps
            w_pm[j] -= eps
            weights_pm = weights.copy()
            weights_pm[layer_idx] = w_pm.reshape(dim, dim)

            w_mp = weights[layer_idx].copy().flatten()
            w_mp[i] -= eps
            w_mp[j] += eps
            weights_mp = weights.copy()
            weights_mp[layer_idx] = w_mp.reshape(dim, dim)

            w_mm = weights[layer_idx].copy().flatten()
            w_mm[i] -= eps
            w_mm[j] -= eps
            weights_mm = weights.copy()
            weights_mm[layer_idx] = w_mm.reshape(dim, dim)

            fpp = compute_loss(weights_pp, target)
            fpm = compute_loss(weights_pm, target)
            fmp = compute_loss(weights_mp, target)
            fmm = compute_loss(weights_mm, target)

            H_layer[i, j] = (fpp - fpm - fmp + fmm) / (4 * eps * eps)
            H_layer[j, i] = H_layer[i, j]

    g_flat = G.flatten()

    # Newton direction: H^{-1} g (pseudoinverse if singular)
    try:
        # Regularize slightly for numerical stability
        H_reg = H_layer + 1e-8 * np.eye(n_params)
        newton_dir = np.linalg.solve(H_reg, g_flat)
    except np.linalg.LinAlgError:
        newton_dir = np.linalg.lstsq(H_layer, g_flat, rcond=None)[0]

    newton_mat = newton_dir.reshape(dim, dim)

    # Muon step
    muon_step = newton_schulz_muon(G)

    # Project Newton direction onto complement (per-row or full matrix)
    # We work in the RIGHT singular vector space (columns of weight matrix)
    # Project each row of newton_mat and muon_step onto complement
    newton_comp = newton_mat @ P_comp
    muon_comp = muon_step @ P_comp

    # Flatten the complement components
    newton_comp_flat = newton_comp.flatten()
    muon_comp_flat = muon_comp.flatten()

    # Cosine between phantom components
    norm_n = np.linalg.norm(newton_comp_flat)
    norm_m = np.linalg.norm(muon_comp_flat)

    if norm_n > 1e-12 and norm_m > 1e-12:
        cos_phantom = np.dot(muon_comp_flat, newton_comp_flat) / (norm_m * norm_n)
    else:
        cos_phantom = 0.0

    # Also measure: fraction of Muon step in complement
    muon_flat = muon_step.flatten()
    frac_phantom_muon = np.linalg.norm(muon_comp_flat) / (np.linalg.norm(muon_flat) + 1e-12)

    # Fraction of Newton in complement
    newton_flat = newton_dir
    frac_phantom_newton = np.linalg.norm(newton_comp_flat) / (np.linalg.norm(newton_flat) + 1e-12)

    # Random baseline
    random_baseline = 1.0 / np.sqrt(dim - top_k)

    print(f"\n  --- Phantom Direction Analysis ---")
    print(f"  Gradient rank concentration (s1/s_total): {S_g[0]/np.sum(S_g):.4f}")
    print(f"  Top-{top_k} SV energy fraction: {np.sum(S_g[:top_k]**2)/np.sum(S_g**2):.4f}")
    print(f"  Muon phantom fraction: {frac_phantom_muon:.4f}")
    print(f"  Newton phantom fraction: {frac_phantom_newton:.4f}")
    print(f"  cos(Muon_phantom, Newton_phantom): {cos_phantom:.4f}")
    print(f"  Random baseline (1/sqrt({dim-top_k})): {random_baseline:.4f}")
    print(f"  Ratio to random: {cos_phantom/random_baseline:.2f}x")

    return {
        'kappa': kappa,
        'cos_phantom': cos_phantom,
        'random_baseline': random_baseline,
        'frac_phantom_muon': frac_phantom_muon,
        'frac_phantom_newton': frac_phantom_newton,
        'top_k_energy': np.sum(S_g[:top_k]**2) / np.sum(S_g**2),
        'sv_ratio': S_g[0] / (S_g[4] + 1e-12),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  H26a: PHANTOM DIRECTIONS")
    print("  Do Muon's phantom components align with Newton residual?")
    print("=" * 60)

    # Use smaller dim for Hessian feasibility
    # 32x32 layer has 1024 params -> Hessian is 1024x1024 -> ~1M entries
    # This is expensive but doable. Use dim=16 for speed, verify trend.
    DIM = 16  # 16x16 = 256 params per layer, Hessian = 256x256

    results = []
    for kappa in [1, 10, 100]:
        r = run_experiment(kappa, dim=DIM, n_layers=4, lr=0.005, n_steps=50, top_k=5)
        results.append(r)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'kappa':<8} {'cos(phantom)':<14} {'baseline':<10} {'ratio':<8} {'top5_energy':<12} {'sv_ratio':<10}")
    print(f"  {'-'*62}")
    for r in results:
        print(f"  {r['kappa']:<8} {r['cos_phantom']:<14.4f} {r['random_baseline']:<10.4f} "
              f"{r['cos_phantom']/r['random_baseline']:<8.2f} {r['top_k_energy']:<12.4f} {r['sv_ratio']:<10.2f}")

    print(f"\n  KEY TESTS:")
    cos_100 = results[2]['cos_phantom']
    print(f"  1. cos(phantom, Newton) at kappa=100: {cos_100:.4f} (threshold: 0.3) -> {'PASS' if cos_100 > 0.3 else 'FAIL'}")
    increases = results[2]['cos_phantom'] > results[0]['cos_phantom']
    print(f"  2. Increases with kappa: {results[0]['cos_phantom']:.4f} -> {results[1]['cos_phantom']:.4f} -> {results[2]['cos_phantom']:.4f} -> {'PASS' if increases else 'FAIL'}")
    low_rank = results[2]['top_k_energy'] > results[0]['top_k_energy']
    print(f"  3. Gradient becomes low-rank with kappa: energy {results[0]['top_k_energy']:.4f} -> {results[2]['top_k_energy']:.4f} -> {'PASS' if low_rank else 'FAIL'}")
