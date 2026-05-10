#!/usr/bin/env python3
"""
H24a: Bifurcation Steps

H20a: compounding is noisy, directional advantage starts negative then positive.

Hypothesis: Muon's advantage spikes at bifurcation points where Hessian eigenvalues
cross zero (curvature sign changes). These are saddle-node or pitchfork bifurcations
in the loss landscape.

Setup: 2-layer 4x4 (32 params, full Hessian feasible), 200 steps.
Every 5 steps: compute full Hessian, get eigenvalues.
Mark "bifurcation steps" where any eigenvalue crosses zero (changes sign).
Also compute per-step cosine advantage (Muon vs NormSGD Newton alignment difference).
Correlate: |d(min_eig)/dt| with cosine_advantage.
If rho > 0.3, bifurcations drive the advantage.
"""

import numpy as np

np.random.seed(42)


def newton_schulz_muon(G, steps=10):
    """Newton-Schulz iteration for polar factor."""
    a = G / (np.linalg.norm(G, ord='fro') + 1e-12)
    for _ in range(steps):
        a = 1.5 * a - 0.5 * a @ a.T @ a
    return a


def deep_linear_forward(weights):
    result = weights[0]
    for W in weights[1:]:
        result = W @ result
    return result


def compute_loss(weights, target):
    prod = deep_linear_forward(weights)
    diff = prod - target
    return 0.5 * np.sum(diff ** 2)


def compute_gradient(weights, target, layer_idx):
    L = len(weights)
    prod = deep_linear_forward(weights)
    diff = prod - target

    left = np.eye(weights[0].shape[0])
    for i in range(L - 1, layer_idx, -1):
        left = weights[i].T @ left

    right = np.eye(weights[0].shape[1])
    for i in range(layer_idx - 1, -1, -1):
        right = right @ weights[i].T

    return left @ diff @ right


def flatten_weights(weights):
    return np.concatenate([W.flatten() for W in weights])


def unflatten_weights(vec, shapes):
    weights = []
    idx = 0
    for shape in shapes:
        size = shape[0] * shape[1]
        weights.append(vec[idx:idx + size].reshape(shape))
        idx += size
    return weights


def compute_full_gradient_vec(weights, target):
    grads = []
    for l in range(len(weights)):
        g = compute_gradient(weights, target, l)
        grads.append(g.flatten())
    return np.concatenate(grads)


def compute_hessian(weights, target, eps=1e-5):
    """Full numerical Hessian."""
    shapes = [W.shape for W in weights]
    vec = flatten_weights(weights)
    n = len(vec)
    H = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            vec_pp = vec.copy(); vec_pp[i] += eps; vec_pp[j] += eps
            vec_pm = vec.copy(); vec_pm[i] += eps; vec_pm[j] -= eps
            vec_mp = vec.copy(); vec_mp[i] -= eps; vec_mp[j] += eps
            vec_mm = vec.copy(); vec_mm[i] -= eps; vec_mm[j] -= eps

            fpp = compute_loss(unflatten_weights(vec_pp, shapes), target)
            fpm = compute_loss(unflatten_weights(vec_pm, shapes), target)
            fmp = compute_loss(unflatten_weights(vec_mp, shapes), target)
            fmm = compute_loss(unflatten_weights(vec_mm, shapes), target)

            H[i, j] = (fpp - fpm - fmp + fmm) / (4 * eps * eps)
            H[j, i] = H[i, j]

    return H


def compute_newton_direction(H, g):
    """Newton direction H^{-1}g with regularization."""
    try:
        H_reg = H + 1e-6 * np.eye(len(g))
        return np.linalg.solve(H_reg, g)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(H, g, rcond=None)[0]


def cosine_sim(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return np.dot(a, b) / (na * nb)


if __name__ == "__main__":
    print("=" * 60)
    print("  H24a: BIFURCATION STEPS")
    print("  Do Hessian eigenvalue zero-crossings drive Muon advantage?")
    print("=" * 60)

    dim = 4
    n_layers = 2
    n_params = n_layers * dim * dim  # 32 params
    n_steps = 200
    lr = 0.01
    hessian_interval = 5

    # Target
    np.random.seed(42)
    U, _ = np.linalg.qr(np.random.randn(dim, dim))
    V, _ = np.linalg.qr(np.random.randn(dim, dim))
    svs = np.array([1.0, 0.7, 0.3, 0.1])
    target = U @ np.diag(svs) @ V.T

    # Initialize
    weights = [np.eye(dim) + 0.1 * np.random.randn(dim, dim) for _ in range(n_layers)]
    shapes = [W.shape for W in weights]

    # Storage
    eigenvalue_history = []
    cosine_advantages = []
    bifurcation_markers = []
    losses = []
    min_eig_history = []
    eig_rate_history = []

    prev_eigs = None

    for step in range(n_steps):
        loss = compute_loss(weights, target)
        losses.append(loss)

        # Compute gradient
        g_vec = compute_full_gradient_vec(weights, target)

        # Muon step (per-layer, then concatenate)
        muon_steps = []
        for l in range(n_layers):
            G_l = compute_gradient(weights, target, l)
            M_l = newton_schulz_muon(G_l)
            muon_steps.append(M_l.flatten())
        muon_vec = np.concatenate(muon_steps)

        # NormSGD step
        g_norm = np.linalg.norm(g_vec)
        if g_norm > 1e-12:
            normsgd_vec = g_vec / g_norm
        else:
            normsgd_vec = g_vec

        # Every hessian_interval steps, compute Hessian
        if step % hessian_interval == 0:
            H = compute_hessian(weights, target)
            eigs = np.linalg.eigvalsh(H)
            eigs_sorted = np.sort(eigs)
            eigenvalue_history.append((step, eigs_sorted))
            min_eig = eigs_sorted[0]
            min_eig_history.append((step, min_eig))

            # Newton direction
            newton_dir = compute_newton_direction(H, g_vec)

            # Cosine advantage: cos(muon, newton) - cos(normsgd, newton)
            cos_muon_newton = cosine_sim(muon_vec, newton_dir)
            cos_sgd_newton = cosine_sim(normsgd_vec, newton_dir)
            advantage = cos_muon_newton - cos_sgd_newton
            cosine_advantages.append((step, advantage))

            # Check for bifurcation (eigenvalue sign change)
            if prev_eigs is not None:
                # Check if any eigenvalue crossed zero
                sign_changes = np.sum((np.sign(prev_eigs) != np.sign(eigs_sorted)) &
                                      (np.abs(prev_eigs) > 1e-8) & (np.abs(eigs_sorted) > 1e-8))
                bifurcation_markers.append((step, sign_changes > 0, sign_changes))

                # Rate of change of minimum eigenvalue
                rate = abs(min_eig - prev_min_eig) / hessian_interval
                eig_rate_history.append((step, rate))

            prev_eigs = eigs_sorted.copy()
            prev_min_eig = min_eig

        # SGD update
        for l in range(n_layers):
            G_l = compute_gradient(weights, target, l)
            weights[l] -= lr * G_l

    # Analysis
    print(f"\n  Training: loss {losses[0]:.4f} -> {losses[-1]:.6f}")
    print(f"  Hessian computed at {len(eigenvalue_history)} steps")

    # Count bifurcations
    n_bifurcations = sum(1 for _, is_bif, _ in bifurcation_markers if is_bif)
    print(f"  Bifurcation events (eigenvalue zero-crossings): {n_bifurcations}")

    # Print eigenvalue evolution
    print(f"\n  Minimum eigenvalue evolution:")
    for step, min_eig in min_eig_history[:10]:
        print(f"    Step {step:3d}: min_eig = {min_eig:.6f}")
    if len(min_eig_history) > 10:
        print(f"    ... ({len(min_eig_history)} total measurements)")
        for step, min_eig in min_eig_history[-5:]:
            print(f"    Step {step:3d}: min_eig = {min_eig:.6f}")

    # Correlation between |d(min_eig)/dt| and cosine advantage
    if len(eig_rate_history) > 5 and len(cosine_advantages) > 5:
        # Align the two series (both start from step hessian_interval)
        rates = np.array([r for _, r in eig_rate_history])
        advantages = np.array([a for s, a in cosine_advantages if s > 0])

        # Make same length
        min_len = min(len(rates), len(advantages))
        rates = rates[:min_len]
        advantages = advantages[:min_len]

        # Pearson correlation
        if np.std(rates) > 1e-12 and np.std(advantages) > 1e-12:
            rho = np.corrcoef(rates, advantages)[0, 1]
        else:
            rho = 0.0

        print(f"\n  CORRELATION ANALYSIS:")
        print(f"  |d(min_eig)/dt| vs cosine_advantage:")
        print(f"    Pearson rho = {rho:.4f}")
        print(f"    Mean rate at bifurcations: ", end="")

        # Rate at bifurcation vs non-bifurcation steps
        bif_steps = set(s for s, is_bif, _ in bifurcation_markers if is_bif)
        rate_at_bif = [r for s, r in eig_rate_history if s in bif_steps]
        rate_not_bif = [r for s, r in eig_rate_history if s not in bif_steps]

        if rate_at_bif:
            print(f"{np.mean(rate_at_bif):.6f}")
        else:
            print("N/A (no bifurcations)")
        if rate_not_bif:
            print(f"    Mean rate at non-bifurcations: {np.mean(rate_not_bif):.6f}")

        # Advantage at bifurcation vs non-bifurcation
        adv_dict = {s: a for s, a in cosine_advantages}
        adv_at_bif = [adv_dict[s] for s, is_bif, _ in bifurcation_markers if is_bif and s in adv_dict]
        adv_not_bif = [adv_dict[s] for s, is_bif, _ in bifurcation_markers if not is_bif and s in adv_dict]

        print(f"\n  Cosine advantage at bifurcation steps: ", end="")
        if adv_at_bif:
            print(f"{np.mean(adv_at_bif):.4f} (n={len(adv_at_bif)})")
        else:
            print("N/A")
        print(f"  Cosine advantage at non-bifurcation steps: ", end="")
        if adv_not_bif:
            print(f"{np.mean(adv_not_bif):.4f} (n={len(adv_not_bif)})")
        else:
            print("N/A")

    # Also look at Spearman rank correlation (manual, no scipy)
    # Manual Spearman
    def rank_array(x):
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(x))
        return ranks

    if len(rates) > 5:
        r_rates = rank_array(rates)
        r_advs = rank_array(advantages)
        spearman = np.corrcoef(r_rates, r_advs)[0, 1]
        print(f"    Spearman rho = {spearman:.4f}")

    print(f"\n  KEY TESTS:")
    print(f"  1. Pearson |d(min_eig)/dt| vs advantage rho = {rho:.4f} (threshold: 0.3) -> {'PASS' if abs(rho) > 0.3 else 'FAIL'}")
    print(f"  2. Bifurcation events detected: {n_bifurcations} -> {'PASS' if n_bifurcations > 0 else 'FAIL'}")
    if adv_at_bif and adv_not_bif:
        diff = np.mean(adv_at_bif) - np.mean(adv_not_bif)
        print(f"  3. Advantage higher at bifurcations: diff = {diff:.4f} -> {'PASS' if diff > 0 else 'FAIL'}")
    else:
        print(f"  3. Advantage comparison: insufficient data")

    print(f"\n  FULL COSINE ADVANTAGE TRAJECTORY:")
    for step, adv in cosine_advantages:
        marker = " <-- BIFURCATION" if step in bif_steps else ""
        print(f"    Step {step:3d}: advantage = {adv:+.4f}{marker}")
