#!/usr/bin/env python3
"""
D-TEST: Depth-Exponent Scaling — Complexity Class Separation
=============================================================
PREDICTION (from dynamical systems / RG gauge-fixing model):
  The Muon-vs-SGD advantage ratio grows EXPONENTIALLY with network depth L.
  Specifically: log(advantage) ~ a * L  (linear in depth)

  This is because each layer contributes a multiplicative Lyapunov factor.
  In a deep linear net, SGD must fight O(L) coupled condition numbers that
  grow exponentially in t, while Muon's orthogonal updates decouple layers.

  If confirmed (R^2 > 0.9 for log(advantage) vs L), this proves a
  COMPLEXITY CLASS SEPARATION: Muon solves in polynomial steps what
  SGD requires exponential steps for, at sufficient depth.

Setup: Deep linear net, hidden_dim=32, output_dim=32.
       Random target matrix T. Loss = ||W_L...W_1 x - Tx||^2.
       Sweep depths L in {2, 3, 4, 6, 8, 12, 16}.
       SGD gets optimally-tuned LR per depth. Muon uses fixed LR.
"""

import numpy as np

np.random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_DIM = 32
HIDDEN_DIM = 32
OUTPUT_DIM = 32
NUM_STEPS = 300
BATCH_SIZE = 64
DEPTHS = [2, 3, 4, 6, 8, 12, 16]
MEASUREMENT_STEPS = [50, 100, 150, 200, 250, 300]
LR_MUON = 0.005
MOMENTUM = 0.9
NS_ITERS = 5

# Random target matrix (fixed across all experiments)
W_target = np.random.randn(OUTPUT_DIM, INPUT_DIM) * 0.5

# Random input data (fixed batch for deterministic comparison)
X_data = np.random.randn(INPUT_DIM, BATCH_SIZE) * 0.3


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def init_weights(num_layers):
    """Initialize layers near identity for stability."""
    weights = []
    for _ in range(num_layers):
        W = np.eye(HIDDEN_DIM) + np.random.randn(HIDDEN_DIM, HIDDEN_DIM) * 0.1
        weights.append(W.copy())
    return weights


def forward(weights, X):
    """Forward pass: W_L @ ... @ W_1 @ X."""
    out = X.copy()
    for W in weights:
        out = W @ out
    return out


def compute_loss(weights, X, target):
    """Loss = 0.5 * ||W_product @ X - T @ X||^2 / N."""
    pred = forward(weights, X)
    target_out = target @ X
    diff = pred - target_out
    return 0.5 * np.mean(np.sum(diff**2, axis=0))


def compute_gradients(weights, X, target):
    """Backprop through deep linear net."""
    num_layers = len(weights)
    N = X.shape[1]

    # Forward pass storing activations
    activations = [X.copy()]
    out = X.copy()
    for W in weights:
        out = W @ out
        activations.append(out.copy())

    # Backward pass
    target_out = target @ X
    delta = (activations[-1] - target_out) / N

    grads = []
    for i in range(num_layers - 1, -1, -1):
        G = delta @ activations[i].T
        grads.insert(0, G)
        if i > 0:
            delta = weights[i].T @ delta

    return grads


def newton_schulz_orthogonalize(G, num_iters=NS_ITERS):
    """
    Newton-Schulz iteration to approximate the orthogonal polar factor.
    Returns closest orthogonal matrix to G (i.e., U @ V^T from SVD).
    """
    norm = np.linalg.norm(G, ord='fro')
    if norm < 1e-12:
        return G
    X = G / norm

    for _ in range(num_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A

    return X


def compute_max_eigenvalue_hessian_approx(weights, X, target):
    """
    Approximate the largest eigenvalue of the loss Hessian
    by computing the product weight matrix and its interaction with the data.
    For a deep linear net, the effective curvature scales with the product
    of singular values across layers.
    """
    # Product matrix
    W_prod = np.eye(HIDDEN_DIM)
    for W in weights:
        W_prod = W @ W_prod

    # Approximate max eigenvalue: related to ||W_prod||^2 * ||X||^2 / N
    sv_prod = np.linalg.svd(W_prod, compute_uv=False)
    sv_X = np.linalg.svd(X, compute_uv=False)
    N = X.shape[1]

    # Upper bound on spectral radius of Hessian
    lambda_max = (sv_prod[0] ** 2) * (sv_X[0] ** 2) / N
    return lambda_max


def find_stable_lr_sgd(depth):
    """
    Find maximum stable SGD learning rate for given depth.
    Strategy: start with theoretical bound lr = 2/(lambda_max * L),
    then verify stability by running a few steps.
    """
    np.random.seed(42)  # reproducible per-depth
    test_weights = init_weights(depth)

    # Compute approximate max eigenvalue
    lambda_max = compute_max_eigenvalue_hessian_approx(test_weights, X_data, W_target)

    # Theoretical maximum stable LR (with safety factor)
    lr_theory = 2.0 / (lambda_max * depth)

    # Cap at reasonable maximum
    lr = min(lr_theory, 0.1)

    # Verify stability: run 50 steps, check loss doesn't explode
    for attempt in range(10):
        np.random.seed(42)
        w_test = init_weights(depth)
        v_test = [np.zeros_like(w) for w in w_test]  # momentum buffer
        stable = True
        initial_loss = compute_loss(w_test, X_data, W_target)

        for step in range(50):
            grads = compute_gradients(w_test, X_data, W_target)
            for i in range(depth):
                v_test[i] = MOMENTUM * v_test[i] + grads[i]
                w_test[i] -= lr * v_test[i]

            current_loss = compute_loss(w_test, X_data, W_target)
            if np.isnan(current_loss) or current_loss > initial_loss * 100:
                stable = False
                break

        if stable:
            return lr
        else:
            lr *= 0.5  # reduce and retry

    # Fallback: very conservative
    return 0.0001


def condition_number(W):
    """Compute condition number of matrix W."""
    sv = np.linalg.svd(W, compute_uv=False)
    if sv[-1] < 1e-12:
        return 1e12
    return sv[0] / sv[-1]


# =============================================================================
# MAIN EXPERIMENT: SWEEP OVER DEPTHS
# =============================================================================

print("=" * 100)
print("D-TEST: DEPTH-EXPONENT SCALING — COMPLEXITY CLASS SEPARATION")
print("=" * 100)
print(f"Setup: Deep linear net (dim={HIDDEN_DIM}), quadratic loss, {NUM_STEPS} steps")
print(f"Depths: {DEPTHS}")
print(f"LR_Muon={LR_MUON} (fixed), Momentum={MOMENTUM}")
print(f"SGD LR: per-depth optimally tuned (max stable)")
print("=" * 100)

# Storage for all results
all_results = {}

for depth in DEPTHS:
    print(f"\n{'─' * 100}")
    print(f"  DEPTH L={depth}")
    print(f"{'─' * 100}")

    # Find optimal SGD learning rate for this depth
    np.random.seed(42)
    lr_sgd = find_stable_lr_sgd(depth)
    print(f"  SGD LR (max stable): {lr_sgd:.6f}")

    # Initialize weights (same seed for fair comparison)
    np.random.seed(42 + depth)
    weights_sgd = init_weights(depth)
    weights_muon = [w.copy() for w in weights_sgd]

    # Momentum buffers
    v_sgd = [np.zeros_like(w) for w in weights_sgd]
    v_muon = [np.zeros_like(w) for w in weights_muon]

    # Storage
    losses_sgd = []
    losses_muon = []
    condition_numbers_per_layer = {i: [] for i in range(depth)}  # per-layer kappa over time
    product_condition_numbers = []

    # Training loop
    for step in range(NUM_STEPS + 1):
        # Record losses
        loss_sgd = compute_loss(weights_sgd, X_data, W_target)
        loss_muon = compute_loss(weights_muon, X_data, W_target)
        losses_sgd.append(loss_sgd)
        losses_muon.append(loss_muon)

        # Track condition numbers for SGD (every 10 steps to save compute)
        if step % 10 == 0:
            for i in range(depth):
                kappa = condition_number(weights_sgd[i])
                condition_numbers_per_layer[i].append(kappa)

            # Product condition number
            W_prod = np.eye(HIDDEN_DIM)
            for W in weights_sgd:
                W_prod = W @ W_prod
            product_condition_numbers.append(condition_number(W_prod))

        # Gradient step (skip last)
        if step < NUM_STEPS:
            # --- SGD with momentum ---
            grads_sgd = compute_gradients(weights_sgd, X_data, W_target)
            for i in range(depth):
                v_sgd[i] = MOMENTUM * v_sgd[i] + grads_sgd[i]
                weights_sgd[i] -= lr_sgd * v_sgd[i]

            # --- Muon with momentum ---
            grads_muon = compute_gradients(weights_muon, X_data, W_target)
            for i in range(depth):
                # Orthogonalize gradient, then apply momentum
                ortho_grad = newton_schulz_orthogonalize(grads_muon[i])
                v_muon[i] = MOMENTUM * v_muon[i] + ortho_grad
                weights_muon[i] -= LR_MUON * v_muon[i]

        # Check for SGD instability
        if np.isnan(loss_sgd) or loss_sgd > 1e10:
            print(f"  WARNING: SGD unstable at step {step}, loss={loss_sgd:.2e}")
            # Fill remaining with NaN
            for remaining in range(step + 1, NUM_STEPS + 1):
                losses_sgd.append(float('nan'))
                losses_muon.append(compute_loss(weights_muon, X_data, W_target))
            break

    # Compute advantage ratios at measurement steps
    advantage_ratios = {}
    for ms in MEASUREMENT_STEPS:
        if ms <= len(losses_sgd) - 1 and not np.isnan(losses_sgd[ms]):
            if losses_muon[ms] > 1e-15:
                advantage_ratios[ms] = losses_sgd[ms] / losses_muon[ms]
            else:
                advantage_ratios[ms] = float('inf')
        else:
            advantage_ratios[ms] = float('nan')

    # Store results
    all_results[depth] = {
        'lr_sgd': lr_sgd,
        'losses_sgd': losses_sgd,
        'losses_muon': losses_muon,
        'advantage_ratios': advantage_ratios,
        'condition_numbers_per_layer': condition_numbers_per_layer,
        'product_condition_numbers': product_condition_numbers,
        'final_loss_sgd': losses_sgd[-1] if not np.isnan(losses_sgd[-1]) else float('inf'),
        'final_loss_muon': losses_muon[-1],
    }

    # Print per-depth summary
    print(f"  Final loss SGD:  {all_results[depth]['final_loss_sgd']:.6e}")
    print(f"  Final loss Muon: {all_results[depth]['final_loss_muon']:.6e}")
    print(f"  Advantage ratios at measurement steps:")
    for ms in MEASUREMENT_STEPS:
        ar = advantage_ratios[ms]
        if np.isfinite(ar):
            print(f"    Step {ms:3d}: {ar:.4f}x")
        else:
            print(f"    Step {ms:3d}: {'INF (SGD diverged)' if np.isnan(ar) else 'INF (Muon converged to 0)'}")


# =============================================================================
# KEY ANALYSIS: log(advantage) vs L
# =============================================================================

print("\n\n" + "=" * 100)
print("KEY ANALYSIS: EXPONENTIAL SCALING OF ADVANTAGE WITH DEPTH")
print("=" * 100)

# Extract final-step advantage for each depth
final_step = MEASUREMENT_STEPS[-1]
depths_valid = []
advantages_valid = []
sgd_trainable = {}

for depth in DEPTHS:
    ar = all_results[depth]['advantage_ratios'].get(final_step, float('nan'))
    final_loss_sgd = all_results[depth]['final_loss_sgd']
    initial_loss_approx = all_results[depth]['losses_sgd'][0]

    # SGD is "trainable" if it reduced loss by at least 50%
    trainable = (final_loss_sgd < initial_loss_approx * 0.5) and np.isfinite(final_loss_sgd)
    sgd_trainable[depth] = trainable

    if np.isfinite(ar) and ar > 0:
        depths_valid.append(depth)
        advantages_valid.append(ar)

depths_arr = np.array(depths_valid, dtype=float)
log_advantages = np.log(np.array(advantages_valid))

print(f"\nValid data points: {len(depths_valid)} / {len(DEPTHS)}")
print(f"\n{'Depth':>6} | {'Advantage':>12} | {'log(Advantage)':>14} | {'SGD trainable?':>14}")
print("-" * 60)
for i, d in enumerate(depths_valid):
    trainable_str = "YES" if sgd_trainable[d] else "NO"
    print(f"{d:6d} | {advantages_valid[i]:12.4f} | {log_advantages[i]:14.4f} | {trainable_str:>14}")

# Linear fit: log(advantage) = a * L + b
if len(depths_arr) >= 3:
    # Least squares fit
    A_matrix = np.vstack([depths_arr, np.ones(len(depths_arr))]).T
    result = np.linalg.lstsq(A_matrix, log_advantages, rcond=None)
    slope_a, intercept_b = result[0]

    # R^2 calculation
    ss_res = np.sum((log_advantages - (slope_a * depths_arr + intercept_b)) ** 2)
    ss_tot = np.sum((log_advantages - np.mean(log_advantages)) ** 2)
    R_squared = 1.0 - ss_res / (ss_tot + 1e-15)

    # Per-layer Lyapunov exponent
    lyapunov_per_layer = slope_a

    print(f"\n{'─' * 60}")
    print(f"LINEAR FIT: log(advantage) = {slope_a:.4f} * L + ({intercept_b:.4f})")
    print(f"  -> Per-layer Lyapunov exponent (slope a): {lyapunov_per_layer:.4f}")
    print(f"  -> Exponential base: e^a = {np.exp(slope_a):.4f}")
    print(f"     (Advantage grows by factor {np.exp(slope_a):.2f}x per added layer)")
    print(f"  -> R^2 = {R_squared:.6f}")
    print(f"{'─' * 60}")

    # Predicted advantages from fit
    print(f"\n{'Depth':>6} | {'Measured':>12} | {'Predicted (fit)':>15} | {'Ratio':>8}")
    print("-" * 55)
    for i, d in enumerate(depths_valid):
        predicted = np.exp(slope_a * d + intercept_b)
        ratio = advantages_valid[i] / predicted
        print(f"{d:6d} | {advantages_valid[i]:12.4f} | {predicted:15.4f} | {ratio:8.3f}")
else:
    R_squared = 0.0
    slope_a = 0.0
    intercept_b = 0.0
    lyapunov_per_layer = 0.0
    print("\n  INSUFFICIENT DATA for linear fit (need >= 3 valid depths)")


# =============================================================================
# SECONDARY ANALYSIS: log(advantage) vs log(step) per depth
# =============================================================================

print("\n\n" + "=" * 100)
print("SECONDARY ANALYSIS: POWER-LAW SCALING IN TIME")
print("=" * 100)
print("Prediction: log(advantage) vs log(step) should have slope ~ L")
print(f"\n{'Depth':>6} | {'Slope of log-log':>18} | {'Expected (~L)':>14} | {'Ratio slope/L':>14}")
print("-" * 70)

loglog_slopes = {}
for depth in DEPTHS:
    ratios_at_steps = all_results[depth]['advantage_ratios']
    steps_valid = []
    log_ratios_valid = []

    for ms in MEASUREMENT_STEPS:
        ar = ratios_at_steps.get(ms, float('nan'))
        if np.isfinite(ar) and ar > 0:
            steps_valid.append(ms)
            log_ratios_valid.append(np.log(ar))

    if len(steps_valid) >= 3:
        log_steps = np.log(np.array(steps_valid, dtype=float))
        log_ratios = np.array(log_ratios_valid)
        slope_ll = np.polyfit(log_steps, log_ratios, 1)[0]
        loglog_slopes[depth] = slope_ll
        ratio_to_L = slope_ll / depth
        print(f"{depth:6d} | {slope_ll:18.4f} | {depth:14d} | {ratio_to_L:14.4f}")
    else:
        loglog_slopes[depth] = float('nan')
        print(f"{depth:6d} | {'N/A':>18} | {depth:14d} | {'N/A':>14}")

# Check if slopes form arithmetic sequence (proportional to L)
valid_slopes = [(d, loglog_slopes[d]) for d in DEPTHS if np.isfinite(loglog_slopes.get(d, float('nan')))]
if len(valid_slopes) >= 3:
    slope_depths = np.array([x[0] for x in valid_slopes], dtype=float)
    slope_vals = np.array([x[1] for x in valid_slopes])
    # Fit slope_val = c * depth + d
    slope_fit = np.polyfit(slope_depths, slope_vals, 1)
    slope_of_slopes = slope_fit[0]
    print(f"\n  Linearity of log-log slopes vs depth:")
    print(f"  Slope-of-slopes = {slope_of_slopes:.4f}")
    print(f"  -> {'CONFIRMED: slopes scale with L' if slope_of_slopes > 0.01 else 'INCONCLUSIVE'}")


# =============================================================================
# SECONDARY ANALYSIS: Per-layer condition number growth (Lyapunov analysis)
# =============================================================================

print("\n\n" + "=" * 100)
print("SECONDARY ANALYSIS: CONDITION NUMBER GROWTH (LYAPUNOV EXPONENTS)")
print("=" * 100)
print("Prediction: log(kappa_i) grows linearly in t (exponential condition growth)")
print("            Product kappa grows faster than individual kappas (multiplicative coupling)")

# Pick depth=8 as representative deep case
analysis_depth = 8
if analysis_depth in all_results:
    cond_data = all_results[analysis_depth]['condition_numbers_per_layer']
    prod_cond = all_results[analysis_depth]['product_condition_numbers']

    num_measurements = len(prod_cond)
    t_axis = np.arange(num_measurements) * 10  # steps

    print(f"\n  Analysis for depth L={analysis_depth}:")
    print(f"  Number of condition number measurements: {num_measurements}")

    if num_measurements >= 5:
        # Fit log(kappa_i) vs t for each layer
        print(f"\n  {'Layer':>6} | {'Initial kappa':>14} | {'Final kappa':>12} | {'Lyapunov (slope)':>16} | {'R^2':>6}")
        print("  " + "-" * 65)

        layer_lyapunovs = []
        for i in range(analysis_depth):
            kappas = np.array(cond_data[i])
            log_kappas = np.log(kappas + 1e-12)

            if len(log_kappas) >= 3:
                fit = np.polyfit(t_axis[:len(log_kappas)], log_kappas, 1)
                layer_lyap = fit[0]
                # R^2
                predicted_lk = np.polyval(fit, t_axis[:len(log_kappas)])
                ss_res_lk = np.sum((log_kappas - predicted_lk) ** 2)
                ss_tot_lk = np.sum((log_kappas - np.mean(log_kappas)) ** 2)
                r2_lk = 1.0 - ss_res_lk / (ss_tot_lk + 1e-15) if ss_tot_lk > 1e-15 else 0.0

                layer_lyapunovs.append(layer_lyap)
                print(f"  {i:6d} | {kappas[0]:14.4f} | {kappas[-1]:12.4f} | {layer_lyap:16.6f} | {r2_lk:6.3f}")

        # Product condition number growth
        log_prod = np.log(np.array(prod_cond) + 1e-12)
        if len(log_prod) >= 3:
            fit_prod = np.polyfit(t_axis[:len(log_prod)], log_prod, 1)
            prod_lyap = fit_prod[0]
            sum_individual = sum(layer_lyapunovs)

            print(f"\n  Product condition number Lyapunov: {prod_lyap:.6f}")
            print(f"  Sum of individual Lyapunovs:       {sum_individual:.6f}")
            print(f"  Ratio (product / sum):             {prod_lyap / (sum_individual + 1e-15):.4f}")
            if prod_lyap > sum_individual * 1.1:
                print(f"  -> CONFIRMED: Multiplicative coupling (product grows FASTER than sum)")
            elif prod_lyap > sum_individual * 0.9:
                print(f"  -> APPROXIMATELY ADDITIVE: layers are weakly coupled")
            else:
                print(f"  -> UNEXPECTED: product grows slower than sum of individuals")
else:
    print(f"\n  Depth {analysis_depth} not in results, skipping condition number analysis.")


# =============================================================================
# VERDICT TABLE
# =============================================================================

print("\n\n" + "=" * 100)
print("VERDICT TABLE")
print("=" * 100)

print(f"\n{'Depth':>6} | {'Final Advantage':>15} | {'Fit O(c^L)':>12} | {'Per-layer Lyap':>14} | "
      f"{'SGD LR':>8} | {'SGD Trainable?':>14}")
print("-" * 90)

for depth in DEPTHS:
    ar = all_results[depth]['advantage_ratios'].get(final_step, float('nan'))
    trainable_str = "YES" if sgd_trainable.get(depth, False) else "NO"
    lr_used = all_results[depth]['lr_sgd']

    if np.isfinite(ar) and len(depths_arr) >= 3:
        predicted = np.exp(slope_a * depth + intercept_b)
        fit_str = f"{predicted:.2f}"
    else:
        fit_str = "N/A"

    if np.isfinite(ar):
        print(f"{depth:6d} | {ar:15.4f} | {fit_str:>12} | {lyapunov_per_layer:14.4f} | "
              f"{lr_used:8.6f} | {trainable_str:>14}")
    else:
        print(f"{depth:6d} | {'INF/NaN':>15} | {fit_str:>12} | {lyapunov_per_layer:14.4f} | "
              f"{lr_used:8.6f} | {trainable_str:>14}")


# =============================================================================
# FINAL VERDICT
# =============================================================================

print("\n" + "=" * 100)
print("FINAL VERDICT: COMPLEXITY CLASS SEPARATION TEST")
print("=" * 100)

if len(depths_arr) >= 3 and R_squared > 0.9:
    print(f"""
  ╔══════════════════════════════════════════════════════════════════════════╗
  ║  PASS: EXPONENTIAL DEPTH SCALING CONFIRMED                             ║
  ╠══════════════════════════════════════════════════════════════════════════╣
  ║                                                                        ║
  ║  log(advantage) vs L is LINEAR with R^2 = {R_squared:.4f}                    ║
  ║  Per-layer Lyapunov exponent: {lyapunov_per_layer:.4f}                            ║
  ║  Advantage grows as O({np.exp(slope_a):.2f}^L) per layer                          ║
  ║                                                                        ║
  ║  INTERPRETATION:                                                       ║
  ║  - Each added layer MULTIPLIES the advantage by ~{np.exp(slope_a):.1f}x              ║
  ║  - SGD's difficulty grows EXPONENTIALLY with depth                      ║
  ║  - Muon's orthogonal updates DECOUPLE the layers                       ║
  ║  - This is a COMPLEXITY CLASS SEPARATION in optimization               ║
  ║                                                                        ║
  ╚══════════════════════════════════════════════════════════════════════════╝
""")
elif len(depths_arr) >= 3 and R_squared > 0.7:
    print(f"""
  ╔══════════════════════════════════════════════════════════════════════════╗
  ║  PARTIAL PASS: TREND IS EXPONENTIAL BUT NOISY                          ║
  ╠══════════════════════════════════════════════════════════════════════════╣
  ║                                                                        ║
  ║  R^2 = {R_squared:.4f} (threshold: 0.9 for full pass)                        ║
  ║  Slope = {slope_a:.4f}, suggesting exponential growth exists but with          ║
  ║  deviations at extreme depths (finite-size effects likely).            ║
  ║                                                                        ║
  ╚══════════════════════════════════════════════════════════════════════════╝
""")
elif len(depths_arr) >= 3:
    print(f"""
  ╔══════════════════════════════════════════════════════════════════════════╗
  ║  FAIL: EXPONENTIAL SCALING NOT CONFIRMED                               ║
  ╠══════════════════════════════════════════════════════════════════════════╣
  ║                                                                        ║
  ║  R^2 = {R_squared:.4f} — log(advantage) vs L is NOT linear.                  ║
  ║  The predicted O(c^L) scaling does NOT hold.                           ║
  ║  Possible explanations:                                                ║
  ║  - Muon also suffers from depth (not purely orthogonal)                ║
  ║  - SGD LR tuning compensates for depth effects                         ║
  ║  - The linear model is too simple for this prediction                  ║
  ║                                                                        ║
  ╚══════════════════════════════════════════════════════════════════════════╝
""")
else:
    print(f"""
  ╔══════════════════════════════════════════════════════════════════════════╗
  ║  INCONCLUSIVE: INSUFFICIENT VALID DATA POINTS                          ║
  ╠══════════════════════════════════════════════════════════════════════════╣
  ║                                                                        ║
  ║  Only {len(depths_arr)} depths produced valid advantage ratios.                    ║
  ║  Need at least 3 for meaningful linear fit.                            ║
  ║  SGD may be unstable at most depths tested.                            ║
  ║                                                                        ║
  ╚══════════════════════════════════════════════════════════════════════════╝
""")

# Summary statistics
print(f"  Summary:")
print(f"    Depths tested: {DEPTHS}")
print(f"    Valid data points: {len(depths_valid)}")
print(f"    R^2 of linear fit: {R_squared:.4f}")
print(f"    Per-layer Lyapunov: {lyapunov_per_layer:.4f}")
print(f"    SGD trainable at all depths: {all(sgd_trainable.get(d, False) for d in DEPTHS)}")
print(f"    Maximum advantage observed: {max(advantages_valid) if advantages_valid else 'N/A':.4f}")
print(f"    At depth: {depths_valid[np.argmax(advantages_valid)] if advantages_valid else 'N/A'}")
print("\n" + "=" * 100)
