#!/usr/bin/env python3
"""
1.3b-ii-B -- Hessian: Null Space Dimension vs Network Symmetry Count

For L layers of width d, the gauge group GL(d)^{L-1} has d^2*(L-1) dimensions.
At a MINIMUM of the loss, these gauge symmetries produce exactly-zero Hessian
eigenvalues, because the loss is constant along gauge orbits.

Key insight: the gauge null space only manifests at CRITICAL POINTS (minima).
At random initialization, gauge directions generically have nonzero curvature.

Approach:
  1. Construct an exact minimum for deep linear net by setting
     W_1 = W_star, W_2 = ... = W_L = I  (trivial factorization)
  2. Then apply random gauge transformations to get a non-trivial factorization
     (to avoid any special structure from having identity matrices)
  3. Compute the full Hessian at this minimum
  4. Count near-zero eigenvalues and compare to d^2*(L-1)

For nonlinear (ReLU) networks:
  - Train to minimum via gradient descent
  - ReLU breaks GL(d) to permutation/scaling => fewer null directions
"""

import numpy as np
import time

# ============================================================================
# Network forward pass and loss
# ============================================================================

def forward_linear(weights, x):
    """Forward pass: y = W_L ... W_2 W_1 x."""
    h = x.copy()
    for W in weights:
        h = W @ h
    return h


def forward_relu(weights, x):
    """Forward pass: y = W_L relu(... relu(W_1 x))."""
    h = x.copy()
    for i, W in enumerate(weights):
        h = W @ h
        if i < len(weights) - 1:
            h = np.maximum(0, h)
    return h


def loss_fn(weights, x, y_target, forward_fn):
    """MSE loss: 0.5 * ||forward(x) - y_target||^2, averaged over samples."""
    y_pred = forward_fn(weights, x)
    return 0.5 * np.mean(np.sum((y_pred - y_target) ** 2, axis=0))


# ============================================================================
# Flatten / unflatten parameters
# ============================================================================

def flatten_weights(weights):
    return np.concatenate([W.ravel() for W in weights])


def unflatten_weights(flat, shapes):
    weights = []
    idx = 0
    for shape in shapes:
        size = shape[0] * shape[1]
        weights.append(flat[idx:idx + size].reshape(shape))
        idx += size
    return weights


# ============================================================================
# Gradient via finite differences
# ============================================================================

def compute_gradient(weights, x, y_target, forward_fn, eps=1e-6):
    shapes = [W.shape for W in weights]
    theta = flatten_weights(weights)
    n = len(theta)
    grad = np.zeros(n)
    for i in range(n):
        theta_p = theta.copy()
        theta_p[i] += eps
        fp = loss_fn(unflatten_weights(theta_p, shapes), x, y_target, forward_fn)
        theta_m = theta.copy()
        theta_m[i] -= eps
        fm = loss_fn(unflatten_weights(theta_m, shapes), x, y_target, forward_fn)
        grad[i] = (fp - fm) / (2 * eps)
    return grad


# ============================================================================
# Full Hessian via finite differences
# ============================================================================

def compute_hessian(weights, x, y_target, forward_fn, eps=1e-5):
    shapes = [W.shape for W in weights]
    theta = flatten_weights(weights)
    n = len(theta)
    H = np.zeros((n, n))

    f0 = loss_fn(weights, x, y_target, forward_fn)

    f_plus = np.zeros(n)
    f_minus = np.zeros(n)
    for i in range(n):
        theta_p = theta.copy()
        theta_p[i] += eps
        f_plus[i] = loss_fn(unflatten_weights(theta_p, shapes), x, y_target, forward_fn)
        theta_m = theta.copy()
        theta_m[i] -= eps
        f_minus[i] = loss_fn(unflatten_weights(theta_m, shapes), x, y_target, forward_fn)

    for i in range(n):
        H[i, i] = (f_plus[i] - 2 * f0 + f_minus[i]) / (eps ** 2)

    for i in range(n):
        for j in range(i + 1, n):
            theta_pp = theta.copy()
            theta_pp[i] += eps
            theta_pp[j] += eps
            f_pp = loss_fn(unflatten_weights(theta_pp, shapes), x, y_target, forward_fn)

            theta_pm = theta.copy()
            theta_pm[i] += eps
            theta_pm[j] -= eps
            f_pm = loss_fn(unflatten_weights(theta_pm, shapes), x, y_target, forward_fn)

            theta_mp = theta.copy()
            theta_mp[i] -= eps
            theta_mp[j] += eps
            f_mp = loss_fn(unflatten_weights(theta_mp, shapes), x, y_target, forward_fn)

            theta_mm = theta.copy()
            theta_mm[i] -= eps
            theta_mm[j] -= eps
            f_mm = loss_fn(unflatten_weights(theta_mm, shapes), x, y_target, forward_fn)

            H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps ** 2)
            H[j, i] = H[i, j]

    return H


# ============================================================================
# Construct exact minimum for deep linear network
# ============================================================================

def construct_linear_minimum(d, L, W_star, scramble=True, seed=99):
    """
    Construct an exact minimum: product W_L...W_1 = W_star.

    Strategy: Start with W_1 = W_star, W_k = I for k > 1.
    If scramble=True, apply random gauge transformations:
      W_l -> W_l G_l^{-1}, W_{l-1} -> G_l W_{l-1}
    This gives a non-trivial factorization that is still an exact minimum.
    """
    weights = [np.eye(d) for _ in range(L)]
    weights[0] = W_star.copy()

    if scramble and L > 1:
        rng = np.random.RandomState(seed)
        for l in range(1, L):
            # Random invertible matrix (gauge transformation)
            G = rng.randn(d, d) * 0.5
            G += np.eye(d)  # Make it close to identity to ensure invertibility
            G_inv = np.linalg.inv(G)

            # Apply: W_l -> W_l @ G_inv, W_{l-1} -> G @ W_{l-1}
            weights[l] = weights[l] @ G_inv
            weights[l - 1] = G @ weights[l - 1]

    # Verify the product
    product = np.eye(d)
    for W in weights:
        product = W @ product
    recon_err = np.linalg.norm(product - W_star) / (np.linalg.norm(W_star) + 1e-15)

    return weights, recon_err


def train_to_minimum_gd(weights, x, y_target, forward_fn, lr=0.01,
                         n_steps=10000, tol=1e-12, verbose=False):
    """Train network to minimum using gradient descent."""
    shapes = [W.shape for W in weights]
    theta = flatten_weights(weights)

    for step in range(n_steps):
        w = unflatten_weights(theta, shapes)
        current_loss = loss_fn(w, x, y_target, forward_fn)

        if current_loss < tol:
            if verbose:
                print(f"    Converged at step {step}, loss={current_loss:.2e}")
            break

        grad = compute_gradient(w, x, y_target, forward_fn)
        grad_norm = np.linalg.norm(grad)

        if verbose and step % 1000 == 0:
            print(f"    Step {step}: loss={current_loss:.2e}, |grad|={grad_norm:.2e}")

        theta -= lr * grad

    final_weights = unflatten_weights(theta, shapes)
    final_loss = loss_fn(final_weights, x, y_target, forward_fn)
    final_grad = compute_gradient(final_weights, x, y_target, forward_fn)

    return final_weights, final_loss, np.linalg.norm(final_grad)


# ============================================================================
# Experiment runner
# ============================================================================

def run_experiment(d, L, forward_fn, net_type, x, y_target, seed=42,
                   at_minimum=True, label=""):
    """Run Hessian null space analysis for a single configuration."""
    np.random.seed(seed)

    total_params = d * d * L
    predicted_gauge_dim = d * d * (L - 1)

    if net_type == "linear" and at_minimum:
        # Construct exact minimum analytically
        W_star = y_target @ x.T @ np.linalg.inv(x @ x.T)
        weights, recon_err = construct_linear_minimum(d, L, W_star, scramble=True, seed=seed)
        loss_val = loss_fn(weights, x, y_target, forward_fn)
        grad = compute_gradient(weights, x, y_target, forward_fn)
        grad_norm = np.linalg.norm(grad)
        print(f"    Constructed minimum: recon_err={recon_err:.2e}, "
              f"loss={loss_val:.2e}, |grad|={grad_norm:.2e}")
    elif net_type == "relu" and at_minimum:
        # Train ReLU to minimum
        weights = []
        for _ in range(L):
            W = np.eye(d) * 0.5 + np.random.randn(d, d) * 0.1
            weights.append(W)
        print(f"    Training ReLU network...", flush=True)
        weights, final_loss, grad_norm = train_to_minimum_gd(
            weights, x, y_target, forward_fn, lr=0.005, n_steps=10000, verbose=True)
        print(f"    Final: loss={final_loss:.2e}, |grad|={grad_norm:.2e}")
    else:
        # Random init
        weights = []
        for _ in range(L):
            W = np.random.randn(d, d) * 0.5 / np.sqrt(d)
            weights.append(W)
        loss_val = loss_fn(weights, x, y_target, forward_fn)
        print(f"    Random init: loss={loss_val:.2e}")

    print(f"    Computing Hessian ({total_params}x{total_params})...", end=" ", flush=True)
    t0 = time.time()
    H = compute_hessian(weights, x, y_target, forward_fn, eps=1e-5)
    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s")

    H = 0.5 * (H + H.T)
    eigenvalues = np.linalg.eigh(H)[0]

    max_abs_eig = np.max(np.abs(eigenvalues))
    if max_abs_eig == 0:
        max_abs_eig = 1.0

    thresholds = [0.01, 0.001, 0.0001]
    null_counts = {}
    for thr in thresholds:
        eps_val = thr * max_abs_eig
        count = int(np.sum(np.abs(eigenvalues) < eps_val))
        null_counts[thr] = count

    return {
        "d": d, "L": L, "net_type": net_type, "label": label,
        "total_params": total_params,
        "predicted_gauge_dim": predicted_gauge_dim,
        "null_counts": null_counts,
        "eigenvalues": eigenvalues,
        "max_abs_eig": max_abs_eig,
        "elapsed": elapsed,
    }


def print_results_table(results, title=""):
    if title:
        print(f"\n{'='*130}")
        print(title)
        print("=" * 130)
    print(f"{'Label':<30} {'Params':<7} {'Predicted':<10} "
          f"{'eps=1%':<8} {'ratio':<7} {'eps=0.1%':<8} {'ratio':<7} "
          f"{'eps=0.01%':<9} {'ratio':<7} {'max|eig|':<12}")
    print("-" * 130)

    for r in results:
        label = r["label"] or f"d={r['d']},L={r['L']},{r['net_type']}"
        pred = r["predicted_gauge_dim"]
        nc = r["null_counts"]
        ratios = {thr: nc[thr] / pred if pred > 0 else 0
                  for thr in [0.01, 0.001, 0.0001]}

        print(f"{label:<30} {r['total_params']:<7} {pred:<10} "
              f"{nc[0.01]:<8} {ratios[0.01]:<7.2f} {nc[0.001]:<8} {ratios[0.001]:<7.2f} "
              f"{nc[0.0001]:<9} {ratios[0.0001]:<7.2f} {r['max_abs_eig']:<12.6f}")


def print_eigenvalue_details(r):
    """Print eigenvalue details for a single result."""
    eigs = np.sort(np.abs(r["eigenvalues"]))
    max_eig = r["max_abs_eig"]
    pred = r["predicted_gauge_dim"]
    n = len(eigs)
    label = r["label"] or f"d={r['d']},L={r['L']},{r['net_type']}"

    print(f"\n  --- {label} ---")
    print(f"  Max |eigenvalue|: {max_eig:.8f}")
    print(f"  Min |eigenvalue|: {eigs[0]:.2e}")

    if n <= 70:
        # Print all eigenvalues grouped
        print(f"  Full spectrum (sorted by |value|):")
        for idx in range(n):
            marker = ""
            if idx == pred:
                marker = " <<<< PREDICTED GAUGE BOUNDARY"
            ratio_to_max = eigs[idx] / max_eig if max_eig > 0 else 0
            print(f"    [{idx:3d}] {eigs[idx]:15.8e}  ({ratio_to_max:.4e}){marker}")
    else:
        print(f"  Smallest 15:")
        for idx in range(min(15, n)):
            print(f"    [{idx:3d}] {eigs[idx]:15.8e}")
        if pred > 0 and pred < n:
            lo = max(0, pred - 5)
            hi = min(n, pred + 5)
            print(f"  Around predicted boundary (idx {pred}):")
            for idx in range(lo, hi):
                marker = " <<<< PREDICTED" if idx == pred else ""
                print(f"    [{idx:3d}] {eigs[idx]:15.8e}{marker}")
        print(f"  Largest 5:")
        for idx in range(max(0, n - 5), n):
            print(f"    [{idx:3d}] {eigs[idx]:15.8e}")

    if pred > 0 and pred < n and eigs[pred - 1] > 0:
        gap = eigs[pred] / eigs[pred - 1]
        print(f"  Spectral gap at boundary: eig[{pred}]/eig[{pred-1}] = {gap:.1f}x")


def main():
    print("=" * 100)
    print("Experiment 1.3b-ii-B: Hessian Null Space Dimension vs Network Symmetry Count")
    print("=" * 100)
    print()
    print("Theory: For L layers of width d, gauge group GL(d)^{L-1} => d^2*(L-1) flat directions.")
    print("At a MINIMUM, these become zero-eigenvalue directions of the Hessian.")
    print()
    print("Approach: Construct exact minima for deep linear networks via")
    print("  W_1 = W*, W_2=...=W_L=I, then apply random gauge transforms.")
    print("  Product is preserved exactly, so we are at a true minimum.")
    print()

    configs = [
        (4, 2),  # 32 params, predicted gauge dim = 16
        (4, 3),  # 48 params, predicted gauge dim = 32
        (4, 4),  # 64 params, predicted gauge dim = 48
        (6, 3),  # 108 params, predicted gauge dim = 72
    ]

    n_samples = 20
    all_results = []

    # ========================================================================
    # PART 1: Linear networks at EXACT minimum
    # ========================================================================
    print("\n" + "#" * 100)
    print("# PART 1: Deep Linear Networks at Exact Minimum")
    print("#" * 100)

    for d, L in configs:
        print(f"\n{'='*80}")
        print(f"d={d}, L={L}: params={d*d*L}, gauge_dim={d*d*(L-1)}, "
              f"physical_dim={d*d}")
        print(f"{'='*80}")

        np.random.seed(100 + d * 10 + L)
        x = np.random.randn(d, n_samples)
        W_target = np.random.randn(d, d) * 0.5
        y_target = W_target @ x

        result = run_experiment(
            d, L, forward_linear, "linear", x, y_target,
            seed=42, at_minimum=True,
            label=f"LINEAR@min d={d},L={L}")
        all_results.append(result)

    # ========================================================================
    # PART 2: Linear networks at random init (control)
    # ========================================================================
    print("\n\n" + "#" * 100)
    print("# PART 2: Deep Linear Networks at Random Init (Control)")
    print("#" * 100)

    for d, L in [(4, 2), (4, 3)]:
        print(f"\n{'='*80}")
        print(f"d={d}, L={L}: random init (NOT at minimum)")
        print(f"{'='*80}")

        np.random.seed(100 + d * 10 + L)
        x = np.random.randn(d, n_samples)
        W_target = np.random.randn(d, d) * 0.5
        y_target = W_target @ x

        result = run_experiment(
            d, L, forward_linear, "lin-rand", x, y_target,
            seed=42, at_minimum=False,
            label=f"LINEAR@rand d={d},L={L}")
        all_results.append(result)

    # ========================================================================
    # PART 3: ReLU networks trained toward minimum
    # ========================================================================
    print("\n\n" + "#" * 100)
    print("# PART 3: ReLU Networks Trained Toward Minimum")
    print("#" * 100)

    for d, L in [(4, 2), (4, 3)]:
        print(f"\n{'='*80}")
        print(f"d={d}, L={L}: ReLU trained")
        print(f"{'='*80}")

        np.random.seed(100 + d * 10 + L)
        x = np.random.randn(d, n_samples)
        W_target = np.random.randn(d, d) * 0.5
        y_target = W_target @ x

        result = run_experiment(
            d, L, forward_relu, "relu", x, y_target,
            seed=42, at_minimum=True,
            label=f"RELU@min d={d},L={L}")
        all_results.append(result)

    # ========================================================================
    # Print results
    # ========================================================================
    linear_min = [r for r in all_results if "LINEAR@min" in r["label"]]
    linear_rand = [r for r in all_results if "LINEAR@rand" in r["label"]]
    relu_min = [r for r in all_results if "RELU" in r["label"]]

    print_results_table(linear_min, "LINEAR NETWORKS AT EXACT MINIMUM")
    print("\n  EIGENVALUE SPECTRA:")
    for r in linear_min:
        print_eigenvalue_details(r)

    print_results_table(linear_rand, "LINEAR NETWORKS AT RANDOM INIT (CONTROL)")
    for r in linear_rand:
        print_eigenvalue_details(r)

    print_results_table(relu_min, "RELU NETWORKS AT MINIMUM")
    for r in relu_min:
        print_eigenvalue_details(r)

    # ========================================================================
    # Analysis
    # ========================================================================
    print("\n\n" + "=" * 100)
    print("ANALYSIS")
    print("=" * 100)

    print("\n--- Linear at minimum: Does null count match d^2*(L-1)? ---")
    print(f"{'Config':<25} {'predicted':<10} {'obs(1%)':<10} {'ratio':<8} {'verdict':<10}")
    print("-" * 63)
    good_count = 0
    for r in linear_min:
        pred = r["predicted_gauge_dim"]
        obs = r["null_counts"][0.01]
        ratio = obs / pred if pred > 0 else 0
        verdict = "MATCH" if 0.8 <= ratio <= 1.2 else (
            "CLOSE" if 0.5 <= ratio <= 1.5 else "MISMATCH")
        if ratio >= 0.8:
            good_count += 1
        print(f"{r['label']:<25} {pred:<10} {obs:<10} {ratio:<8.3f} {verdict:<10}")

    print("\n--- Control: Random init should NOT show null space ---")
    for r in linear_rand:
        pred = r["predicted_gauge_dim"]
        obs = r["null_counts"][0.01]
        ratio = obs / pred if pred > 0 else 0
        print(f"  {r['label']}: predicted={pred}, observed={obs}, ratio={ratio:.3f}")

    print("\n--- ReLU: Should show fewer null directions than linear ---")
    for rr in relu_min:
        rl = [r for r in linear_min if r["d"] == rr["d"] and r["L"] == rr["L"]]
        if rl:
            lin_null = rl[0]["null_counts"][0.01]
            relu_null = rr["null_counts"][0.01]
            print(f"  d={rr['d']},L={rr['L']}: linear_null={lin_null}, "
                  f"relu_null={relu_null}")

    # Key test
    print("\n--- KEY TEST: d=4, L=3 (Linear at minimum) ---")
    key = [r for r in linear_min if r["d"] == 4 and r["L"] == 3]
    if key:
        r = key[0]
        pred = r["predicted_gauge_dim"]
        eigs = np.sort(np.abs(r["eigenvalues"]))
        print(f"  Total params: {r['total_params']}")
        print(f"  Predicted gauge dim: {pred} = 4^2 * 2 = 32")
        print(f"  Non-null (physical) dims: {r['total_params'] - pred} = 16")
        for thr in [0.01, 0.001, 0.0001]:
            obs = r["null_counts"][thr]
            print(f"  Threshold {thr}: null_count={obs}, "
                  f"ratio={obs/pred:.3f}")
        # Show the gap
        if pred < len(eigs) and eigs[pred - 1] > 0:
            print(f"  Spectral gap: eig[{pred}]/eig[{pred-1}] = "
                  f"{eigs[pred]/eigs[pred-1]:.1f}x")

    # Verdict
    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)

    if good_count >= len(linear_min) * 0.75:
        print("\n=> HYPOTHESIS STRONGLY SUPPORTED:")
        print("   Hessian null space dimension matches d^2*(L-1) at minima")
        print("   of deep linear networks. DIRECT evidence of gauge flat directions.")
    elif good_count >= len(linear_min) * 0.5:
        print("\n=> HYPOTHESIS PARTIALLY SUPPORTED:")
        print("   Some configs match, others do not.")
    else:
        print("\n=> RESULTS NEED EXAMINATION:")
        print("   Check eigenvalue spectra for a clear gap at the predicted boundary.")
        print("   The exact null space count depends on threshold choice.")
        print("   Look at the SPECTRAL GAP rather than raw counts.")

    print("\nTheoretical prediction:")
    for d, L in configs:
        total = d * d * L
        gauge = d * d * (L - 1)
        phys = d * d
        print(f"  d={d}, L={L}: {total} params = {gauge} gauge + {phys} physical")


if __name__ == "__main__":
    main()
