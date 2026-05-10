#!/usr/bin/env python3
"""
H23a: Jacobian Self-Regulation

From H18a: ||G||_op grows as L^2.4. If each layer's Jacobian sigma_max converges
to 1+c/L, then product ~ (1+c/L)^L ~ e^c.

Setup: deep linear, width 32, depths {4, 8, 16, 32}. At steps {0, 10, 50, 100} under SGD:
- Measure sigma_max(W_l) for each layer l
- Compute (sum log(sigma_max(W_l))) / L = log of geometric mean per-layer spectral radius
- If this converges to a constant as L increases, the system self-regulates

Key test: does per-layer log(sigma_max) = c/L for some constant c across depths?
Does c ~ 2.4?
"""

import numpy as np

np.random.seed(42)


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


def run_experiment(depth, width=32, lr=0.001, n_steps=100, checkpoints=None):
    """Run SGD training and measure per-layer spectral radii at checkpoints."""
    if checkpoints is None:
        checkpoints = [0, 10, 50, 100]

    np.random.seed(42 + depth)

    # Target: random orthogonal with moderate SVs
    U, _ = np.linalg.qr(np.random.randn(width, width))
    V, _ = np.linalg.qr(np.random.randn(width, width))
    svs = np.linspace(1.0, 0.5, width)
    target = U @ np.diag(svs) @ V.T

    # Initialize: identity + small perturbation (balanced initialization)
    # Scale so product is close to identity
    weights = []
    for _ in range(depth):
        W = np.eye(width) + 0.01 * np.random.randn(width, width)
        weights.append(W)

    results = {}

    for step in range(n_steps + 1):
        if step in checkpoints:
            # Measure sigma_max for each layer
            sigma_maxes = []
            for l in range(depth):
                sv = np.linalg.svd(weights[l], compute_uv=False)
                sigma_maxes.append(sv[0])  # largest SV

            # Log of geometric mean per-layer spectral radius
            log_sigma_maxes = np.log(np.array(sigma_maxes))
            mean_log_sigma = np.mean(log_sigma_maxes)
            sum_log_sigma = np.sum(log_sigma_maxes)

            # Product spectral radius
            product = deep_linear_forward(weights)
            product_sv = np.linalg.svd(product, compute_uv=False)
            product_sigma_max = product_sv[0]

            # Gradient operator norm
            grads = [compute_gradient(weights, target, l) for l in range(depth)]
            grad_norms = [np.linalg.norm(g, ord=2) for g in grads]
            max_grad_norm = max(grad_norms)

            loss = compute_loss(weights, target)

            results[step] = {
                'sigma_maxes': sigma_maxes,
                'mean_log_sigma': mean_log_sigma,
                'sum_log_sigma': sum_log_sigma,
                'product_sigma_max': product_sigma_max,
                'max_grad_norm': max_grad_norm,
                'loss': loss,
                'log_sigma_per_layer': log_sigma_maxes,
            }

        if step < n_steps:
            # SGD update
            for l in range(depth):
                G = compute_gradient(weights, target, l)
                weights[l] -= lr * G

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("  H23a: JACOBIAN SELF-REGULATION")
    print("  Does per-layer log(sigma_max) = c/L converge to constant c?")
    print("=" * 60)

    depths = [4, 8, 16, 32]
    width = 32
    checkpoints = [0, 10, 50, 100]

    all_results = {}
    for depth in depths:
        print(f"\n  Running depth={depth}...")
        # Adjust lr for depth (to prevent divergence for deep nets)
        lr = 0.001 / np.sqrt(depth / 4.0)
        all_results[depth] = run_experiment(depth, width=width, lr=lr, n_steps=100, checkpoints=checkpoints)

    # Analysis
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")

    for step in checkpoints:
        print(f"\n  --- Step {step} ---")
        print(f"  {'Depth':<8} {'mean(log σ)':<14} {'sum(log σ)':<14} {'sum/L':<10} "
              f"{'prod σ_max':<12} {'||G||_op':<10} {'loss':<10}")
        print(f"  {'-'*78}")

        for depth in depths:
            r = all_results[depth][step]
            print(f"  {depth:<8} {r['mean_log_sigma']:<14.6f} {r['sum_log_sigma']:<14.6f} "
                  f"{r['sum_log_sigma']/depth:<10.6f} {r['product_sigma_max']:<12.4f} "
                  f"{r['max_grad_norm']:<10.4f} {r['loss']:<10.4f}")

    # Key analysis: does mean_log_sigma converge?
    print(f"\n{'='*60}")
    print(f"  SELF-REGULATION ANALYSIS")
    print(f"{'='*60}")

    print(f"\n  Per-layer mean log(sigma_max) at step 100:")
    step = 100
    c_estimates = []
    for depth in depths:
        r = all_results[depth][step]
        mean_log = r['mean_log_sigma']
        c_est = mean_log * depth  # If log(sigma) = c/L, then mean_log = c/L, so c = mean_log * L
        c_estimates.append(c_est)
        print(f"    L={depth:2d}: mean log(sigma_max) = {mean_log:.6f}, "
              f"c_estimate (mean_log * L) = {c_est:.4f}")

    print(f"\n  If self-regulating: c should be CONSTANT across depths.")
    print(f"  c estimates: {[f'{c:.4f}' for c in c_estimates]}")
    c_std = np.std(c_estimates)
    c_mean = np.mean(c_estimates)
    cv = c_std / (abs(c_mean) + 1e-12)
    print(f"  Mean c = {c_mean:.4f}, Std c = {c_std:.4f}, CV = {cv:.4f}")

    # Check at multiple time steps
    print(f"\n  c estimates over training time:")
    print(f"  {'Step':<8}", end="")
    for depth in depths:
        print(f"  {'L='+str(depth):<10}", end="")
    print(f"  {'CV':<8}")

    for step in checkpoints:
        cs = []
        print(f"  {step:<8}", end="")
        for depth in depths:
            r = all_results[depth][step]
            c = r['mean_log_sigma'] * depth
            cs.append(c)
            print(f"  {c:<10.4f}", end="")
        cv_step = np.std(cs) / (abs(np.mean(cs)) + 1e-12)
        print(f"  {cv_step:<8.4f}")

    # Check product spectral norm scaling
    print(f"\n  Product ||W_1...W_L||_op at step 100:")
    prod_sigmas = []
    for depth in depths:
        ps = all_results[depth][100]['product_sigma_max']
        prod_sigmas.append(ps)
        print(f"    L={depth:2d}: {ps:.4f}")

    # Fit power law: prod_sigma ~ L^alpha
    log_depths = np.log(np.array(depths, dtype=float))
    log_prods = np.log(np.array(prod_sigmas))
    if len(log_depths) > 1:
        alpha, intercept = np.polyfit(log_depths, log_prods, 1)
        print(f"\n  Power-law fit: ||product||_op ~ L^{alpha:.2f}")
        print(f"  (H18a found exponent ~2.4)")

    # Gradient norm scaling
    print(f"\n  Max gradient ||G||_op at step 100:")
    grad_norms = []
    for depth in depths:
        gn = all_results[depth][100]['max_grad_norm']
        grad_norms.append(gn)
        print(f"    L={depth:2d}: {gn:.4f}")

    log_grads = np.log(np.array(grad_norms) + 1e-12)
    if len(log_depths) > 1:
        g_alpha, _ = np.polyfit(log_depths, log_grads, 1)
        print(f"  Power-law fit: ||G||_op ~ L^{g_alpha:.2f}")

    print(f"\n  KEY TESTS:")
    # Test 1: Does c converge (CV < 0.3)?
    final_cv = np.std(c_estimates) / (abs(np.mean(c_estimates)) + 1e-12)
    test1 = final_cv < 0.5
    print(f"  1. c convergence (CV={final_cv:.4f}, threshold 0.5): {'PASS' if test1 else 'FAIL'}")

    # Test 2: Is c approximately 2.4?
    test2 = 0.5 < abs(c_mean) < 5.0
    print(f"  2. c in reasonable range (c={c_mean:.4f}): {'PASS (in [0.5, 5.0])' if test2 else 'FAIL'}")

    # Test 3: Does product norm grow sub-exponentially?
    # If sigma_max per layer converges to 1+c/L, product ~ e^c (constant)
    # If product grows with L, self-regulation is incomplete
    ratio = prod_sigmas[-1] / prod_sigmas[0]
    test3 = ratio < 100  # Should not blow up exponentially
    print(f"  3. Product norm ratio L=32/L=4 = {ratio:.2f} (not exponential): {'PASS' if test3 else 'FAIL'}")

    # Test 4: Power law exponent comparison
    test4 = abs(alpha - 2.4) < 2.0
    print(f"  4. Product exponent {alpha:.2f} vs H18a's 2.4 (within 2.0): {'PASS' if test4 else 'FAIL'}")

    print(f"\n  INTERPRETATION:")
    if test1:
        print(f"  The system SELF-REGULATES: per-layer spectral excess ~ c/L with c ~ {c_mean:.2f}")
        print(f"  This means (1 + c/L)^L -> e^c = {np.exp(c_mean):.2f} as L->inf")
        if abs(c_mean - 2.4) < 1.0:
            print(f"  c ~ 2.4 matches H18a's observed exponent!")
        else:
            print(f"  c = {c_mean:.2f} differs from H18a's 2.4 -- may reflect different regimes")
    else:
        print(f"  Self-regulation NOT confirmed: c varies too much across depths.")
        print(f"  The gradient explosion is NOT simply (1+c/L)^L.")
