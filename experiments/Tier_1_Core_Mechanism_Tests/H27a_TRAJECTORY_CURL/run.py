#!/usr/bin/env python3
"""
H27a: Trajectory Curl

H3b: partial SV equalization (k<n) is worse than SGD.
H19a: not from single-step saddles.

Hypothesis: Partial equalization creates "wasted rotation" in parameter space --
the optimizer moves a lot but doesn't get far. This shows up as high trajectory curl.

Setup: 3-layer 4x4 deep linear (48 params), 200 steps. Sweep k in {0(SGD), 1, 2, 3, 4(full Muon)}.
Measure trajectory curl: path_length / ||displacement|| - 1, where:
- path_length = sum ||delta_t||_F (sum of step norms)
- displacement = ||W_final - W_init||_F (net movement)
- curl_proxy = path_length/displacement - 1 (excess path = wasted rotation)

Key test: partial k (especially k=2) has HIGHER curl than k=0 (SGD) or k=4 (full Muon).
"""

import numpy as np

np.random.seed(42)


def partial_sv_equalization(G, k):
    """
    Equalize only the top-k singular values, leave the rest as-is.
    k=0: return normalized gradient (SGD-like)
    k=n: full Muon (all SVs equalized to 1)
    """
    if k == 0:
        # Normalized SGD: just normalize the gradient
        norm = np.linalg.norm(G, ord='fro')
        if norm < 1e-12:
            return G
        return G / norm

    U, S, Vt = np.linalg.svd(G, full_matrices=False)
    n = min(G.shape)

    if k >= n:
        # Full Muon: all SVs set to 1
        return U @ Vt
    else:
        # Partial: equalize top-k SVs to 1, keep rest normalized
        S_new = S.copy()
        S_new[:k] = 1.0
        # Normalize the remaining to preserve their relative scale but with unit total
        remaining_norm = np.linalg.norm(S_new[k:])
        if remaining_norm > 1e-12:
            S_new[k:] = S_new[k:] / remaining_norm
        return U @ np.diag(S_new) @ Vt


def newton_schulz_muon(G, steps=10):
    """Full Muon via Newton-Schulz iteration."""
    a = G / (np.linalg.norm(G, ord='fro') + 1e-12)
    for _ in range(steps):
        a = 1.5 * a - 0.5 * a @ a.T @ a
    return a


def deep_linear_forward(weights):
    """Product of weight matrices."""
    result = weights[0]
    for W in weights[1:]:
        result = W @ result
    return result


def compute_loss(weights, target):
    """MSE loss."""
    prod = deep_linear_forward(weights)
    diff = prod - target
    return 0.5 * np.sum(diff ** 2)


def compute_gradient(weights, target, layer_idx):
    """Gradient of loss w.r.t. weights[layer_idx]."""
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


def run_single(k, dim=4, n_layers=3, lr=0.01, n_steps=200, seed=42):
    """Run optimization with partial SV equalization level k."""
    np.random.seed(seed)

    # Target matrix with moderate conditioning
    U, _ = np.linalg.qr(np.random.randn(dim, dim))
    V, _ = np.linalg.qr(np.random.randn(dim, dim))
    svs = np.array([1.0, 0.5, 0.25, 0.1])
    target = U @ np.diag(svs) @ V.T

    # Initialize
    weights = [np.eye(dim) + 0.05 * np.random.randn(dim, dim) for _ in range(n_layers)]
    w_init = flatten_weights(weights)

    path_length = 0.0
    losses = []

    for step in range(n_steps):
        losses.append(compute_loss(weights, target))

        grads = [compute_gradient(weights, target, l) for l in range(n_layers)]

        # Apply partial SV equalization
        if k == dim:
            # Full Muon via Newton-Schulz
            steps_list = [newton_schulz_muon(g) for g in grads]
        else:
            steps_list = [partial_sv_equalization(g, k) for g in grads]

        # Take step and measure path length
        delta_norm_sq = 0.0
        for l in range(n_layers):
            delta = lr * steps_list[l]
            delta_norm_sq += np.sum(delta ** 2)
            weights[l] -= delta

        path_length += np.sqrt(delta_norm_sq)

    w_final = flatten_weights(weights)
    displacement = np.linalg.norm(w_final - w_init)
    final_loss = compute_loss(weights, target)

    curl = (path_length / displacement) - 1.0 if displacement > 1e-12 else float('inf')

    return {
        'k': k,
        'path_length': path_length,
        'displacement': displacement,
        'curl': curl,
        'final_loss': final_loss,
        'init_loss': losses[0],
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  H27a: TRAJECTORY CURL")
    print("  Does partial SV equalization create wasted rotation?")
    print("=" * 60)

    dim = 4
    n_layers = 3
    n_steps = 200
    lr = 0.01
    n_seeds = 5

    k_values = [0, 1, 2, 3, 4]
    k_labels = {0: 'SGD(norm)', 1: 'k=1', 2: 'k=2', 3: 'k=3', 4: 'Muon(full)'}

    all_results = {k: [] for k in k_values}

    for seed in range(n_seeds):
        for k in k_values:
            r = run_single(k, dim=dim, n_layers=n_layers, lr=lr, n_steps=n_steps, seed=seed + 100)
            all_results[k].append(r)

    print(f"\n  {'Method':<12} {'Curl (mean)':<12} {'Curl (std)':<12} {'Final Loss':<12} {'Path':<10} {'Disp':<10}")
    print(f"  {'-'*68}")

    summary = {}
    for k in k_values:
        curls = [r['curl'] for r in all_results[k]]
        losses = [r['final_loss'] for r in all_results[k]]
        paths = [r['path_length'] for r in all_results[k]]
        disps = [r['displacement'] for r in all_results[k]]

        mean_curl = np.mean(curls)
        std_curl = np.std(curls)
        mean_loss = np.mean(losses)
        mean_path = np.mean(paths)
        mean_disp = np.mean(disps)

        summary[k] = {'mean_curl': mean_curl, 'std_curl': std_curl, 'mean_loss': mean_loss}

        print(f"  {k_labels[k]:<12} {mean_curl:<12.4f} {std_curl:<12.4f} {mean_loss:<12.6f} "
              f"{mean_path:<10.4f} {mean_disp:<10.4f}")

    print(f"\n  KEY TESTS:")

    # Test 1: Does k=2 have higher curl than SGD?
    curl_sgd = summary[0]['mean_curl']
    curl_k2 = summary[2]['mean_curl']
    curl_muon = summary[4]['mean_curl']

    test1 = curl_k2 > curl_sgd
    print(f"  1. k=2 curl ({curl_k2:.4f}) > SGD curl ({curl_sgd:.4f}): {'PASS' if test1 else 'FAIL'}")

    # Test 2: Does k=2 have higher curl than full Muon?
    test2 = curl_k2 > curl_muon
    print(f"  2. k=2 curl ({curl_k2:.4f}) > Muon curl ({curl_muon:.4f}): {'PASS' if test2 else 'FAIL'}")

    # Test 3: Is curl non-monotonic in k? (peaks at intermediate k)
    curls_by_k = [summary[k]['mean_curl'] for k in k_values]
    peak_k = k_values[np.argmax(curls_by_k)]
    test3 = peak_k in [1, 2, 3]
    print(f"  3. Peak curl at intermediate k={peak_k}: {'PASS' if test3 else 'FAIL'}")

    # Test 4: Does Muon have lower final loss than partial k?
    test4 = summary[4]['mean_loss'] < summary[2]['mean_loss']
    print(f"  4. Muon loss ({summary[4]['mean_loss']:.6f}) < k=2 loss ({summary[2]['mean_loss']:.6f}): {'PASS' if test4 else 'FAIL'}")

    print(f"\n  INTERPRETATION:")
    if test1 and test2:
        print(f"  Partial SV equalization creates EXCESS trajectory curl.")
        print(f"  The optimizer wastes energy rotating without progress.")
        print(f"  Full Muon avoids this by making the step ORTHOGONAL (no rotation waste).")
    elif test1 and not test2:
        print(f"  Partial k has more curl than SGD but also more than Muon.")
        print(f"  Suggests partial equalization fights the natural gradient flow.")
    else:
        print(f"  Curl hypothesis NOT confirmed. Partial k failure has different cause.")
