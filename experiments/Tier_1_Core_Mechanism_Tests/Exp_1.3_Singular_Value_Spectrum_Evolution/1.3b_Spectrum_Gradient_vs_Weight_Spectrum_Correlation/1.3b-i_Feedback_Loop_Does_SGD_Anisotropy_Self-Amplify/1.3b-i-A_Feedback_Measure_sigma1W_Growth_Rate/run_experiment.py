#!/usr/bin/env python3
"""
1.3b-i-A: Feedback Measure -- sigma_1(W) Growth Rate Under Each Optimizer
=========================================================================

PREDICTION (from anisotropy cascade / RG gauge-fixing model):
  SGD with momentum accumulates updates along the dominant singular direction
  of the weight matrix, creating a positive feedback loop:
    large sigma_1(W) -> gradient aligns with top direction -> update amplifies
    sigma_1(W) further -> exponential growth of sigma_1(W).

  Muon's Newton-Schulz orthogonalization projects the gradient onto the
  orthogonal manifold, meaning every singular value of the update is 1.
  This breaks the feedback loop: the step size in every spectral direction
  is the same, so sigma_1(W) can only grow linearly (bounded step per
  iteration), not exponentially.

HYPOTHESIS:
  - SGD: log(sigma_1) grows LINEARLY in t  (i.e. sigma_1 ~ exp(a*t))
  - Muon: log(sigma_1) grows SUB-LINEARLY in t  (i.e. sigma_1 ~ t^a or bounded)
  - The correlation corr(sigma_1(G_i), sigma_1(W_i)) should be HIGH for SGD
    (gradient tracks weight anisotropy = feedback loop active) and LOWER for Muon
    (orthogonalization decorrelates gradient spectrum from weight spectrum).

CRITICAL CONTEXT:
  - 1.2b-i showed Muon is MORE chaotic (higher Lyapunov) -- but direction is better
  - 1.3a-i showed per-layer erank stays higher for Muon (93.3% vs 89.5%)
  - 1.3a-ii showed Muon's momentum has 2x the effective rank

Setup: 4-layer deep linear net, 32x32, quadratic loss, 500 steps.
"""

import numpy as np
import os

np.random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================

DIM = 32
NUM_LAYERS = 4
NUM_STEPS = 500
BATCH_SIZE = 64
LR_MUON = 0.005
MOMENTUM = 0.9
NS_ITERS = 5

# Random target matrix (fixed)
W_target = np.random.randn(DIM, DIM) * 0.5

# Random input data (fixed batch)
X_data = np.random.randn(DIM, BATCH_SIZE) * 0.3

# Output directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def init_weights(num_layers, seed=42):
    """Initialize layers near identity for stability."""
    rng = np.random.RandomState(seed)
    weights = []
    for _ in range(num_layers):
        W = np.eye(DIM) + rng.randn(DIM, DIM) * 0.1
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
    return 0.5 * np.mean(np.sum(diff ** 2, axis=0))


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


def gini_coefficient(values):
    """
    Compute the Gini coefficient of a 1D array.
    0 = perfect equality, 1 = perfect inequality.
    """
    values = np.sort(np.abs(values))
    n = len(values)
    if n == 0 or np.sum(values) < 1e-30:
        return 0.0
    index = np.arange(1, n + 1)
    return (2.0 * np.sum(index * values) / (n * np.sum(values))) - (n + 1.0) / n


# =============================================================================
# OPTIMIZER STEP FUNCTIONS
# =============================================================================

def find_stable_lr_sgd():
    """Find maximum stable SGD learning rate."""
    candidates = [0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.001]
    for lr in candidates:
        np.random.seed(42)
        weights = init_weights(NUM_LAYERS)
        velocities = [np.zeros((DIM, DIM)) for _ in range(NUM_LAYERS)]
        initial_loss = compute_loss(weights, X_data, W_target)
        stable = True
        for step in range(200):
            grads = compute_gradients(weights, X_data, W_target)
            for i in range(NUM_LAYERS):
                velocities[i] = MOMENTUM * velocities[i] + grads[i]
                weights[i] -= lr * velocities[i]
            loss = compute_loss(weights, X_data, W_target)
            if np.isnan(loss) or loss > initial_loss * 50:
                stable = False
                break
        if stable:
            return lr
    return 0.001


def sgd_step(weights, velocities, lr):
    """One step of SGD with momentum. Returns (weights, velocities, raw_grads)."""
    grads = compute_gradients(weights, X_data, W_target)
    for i in range(len(weights)):
        velocities[i] = MOMENTUM * velocities[i] + grads[i]
        weights[i] = weights[i] - lr * velocities[i]
    return weights, velocities, grads


def muon_step(weights, velocities, lr):
    """One step of Muon with momentum. Returns (weights, velocities, raw_grads)."""
    grads = compute_gradients(weights, X_data, W_target)
    for i in range(len(weights)):
        ortho_grad = newton_schulz_orthogonalize(grads[i])
        velocities[i] = MOMENTUM * velocities[i] + ortho_grad
        weights[i] = weights[i] - lr * velocities[i]
    return weights, velocities, grads


# =============================================================================
# MEASUREMENT ENGINE
# =============================================================================

def run_and_measure(optimizer_name, optimizer_fn, lr, num_steps):
    """
    Run optimizer for num_steps and measure at EVERY step:
      - sigma_1(W_i) for each layer (top singular value)
      - sigma_n(W_i) for each layer (bottom singular value)
      - full SV spectrum at selected steps
      - sigma_1(G_i) for each layer (gradient top SV)
      - correlation between sigma_1(G_i) and sigma_1(W_i)
      - loss
    """
    np.random.seed(42)
    weights = init_weights(NUM_LAYERS)
    velocities = [np.zeros((DIM, DIM)) for _ in range(NUM_LAYERS)]

    # Storage: per step, per layer
    sigma1_W = np.zeros((num_steps + 1, NUM_LAYERS))  # top SV of weight
    sigman_W = np.zeros((num_steps + 1, NUM_LAYERS))   # bottom SV of weight
    sigma1_G = np.zeros((num_steps + 1, NUM_LAYERS))   # top SV of gradient
    losses = np.zeros(num_steps + 1)
    # Full SV spectrum at final step
    final_sv_spectrum = []

    # Measure at step 0
    for i in range(NUM_LAYERS):
        sv = np.linalg.svd(weights[i], compute_uv=False)
        sigma1_W[0, i] = sv[0]
        sigman_W[0, i] = sv[-1]
    losses[0] = compute_loss(weights, X_data, W_target)

    # Compute initial gradients for sigma1_G at step 0
    grads_init = compute_gradients(weights, X_data, W_target)
    for i in range(NUM_LAYERS):
        sv_g = np.linalg.svd(grads_init[i], compute_uv=False)
        sigma1_G[0, i] = sv_g[0]

    for step in range(1, num_steps + 1):
        weights, velocities, grads = optimizer_fn(weights, velocities, lr)

        # Measure
        for i in range(NUM_LAYERS):
            sv = np.linalg.svd(weights[i], compute_uv=False)
            sigma1_W[step, i] = sv[0]
            sigman_W[step, i] = sv[-1]

            sv_g = np.linalg.svd(grads[i], compute_uv=False)
            sigma1_G[step, i] = sv_g[0]

        loss = compute_loss(weights, X_data, W_target)
        losses[step] = loss

        # Check for divergence
        if np.isnan(loss) or loss > 1e10:
            print(f"    WARNING: {optimizer_name} diverged at step {step}!")
            # Fill remaining with NaN
            sigma1_W[step + 1:] = np.nan
            sigman_W[step + 1:] = np.nan
            sigma1_G[step + 1:] = np.nan
            losses[step + 1:] = np.nan
            break

    # Record final SV spectrum
    for i in range(NUM_LAYERS):
        sv = np.linalg.svd(weights[i], compute_uv=False)
        final_sv_spectrum.append(sv)

    return {
        'sigma1_W': sigma1_W,
        'sigman_W': sigman_W,
        'sigma1_G': sigma1_G,
        'losses': losses,
        'final_sv_spectrum': final_sv_spectrum,
    }


# =============================================================================
# GROWTH MODEL FITTING
# =============================================================================

def fit_exponential(steps, log_sigma1):
    """
    Fit log(sigma_1) = a*t + b  (exponential growth model).
    Returns (a, b, R^2).
    """
    # Filter out NaN/Inf
    mask = np.isfinite(log_sigma1)
    t = steps[mask].astype(float)
    y = log_sigma1[mask]
    if len(t) < 3:
        return 0.0, 0.0, 0.0

    # Linear regression: y = a*t + b
    A = np.vstack([t, np.ones(len(t))]).T
    result = np.linalg.lstsq(A, y, rcond=None)
    coeffs = result[0]
    a, b = coeffs[0], coeffs[1]

    y_pred = a * t + b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0

    return a, b, r2


def fit_polynomial(steps, log_sigma1):
    """
    Fit log(sigma_1) = a*log(t) + b  (polynomial/power-law growth model).
    Returns (a, b, R^2).
    Only uses steps > 0 (since log(0) is undefined).
    """
    mask = np.isfinite(log_sigma1) & (steps > 0)
    t = steps[mask].astype(float)
    y = log_sigma1[mask]
    if len(t) < 3:
        return 0.0, 0.0, 0.0

    log_t = np.log(t)

    # Linear regression: y = a*log(t) + b
    A = np.vstack([log_t, np.ones(len(log_t))]).T
    result = np.linalg.lstsq(A, y, rcond=None)
    coeffs = result[0]
    a, b = coeffs[0], coeffs[1]

    y_pred = a * log_t + b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0

    return a, b, r2


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

print("=" * 100)
print("1.3b-i-A: FEEDBACK MEASURE -- sigma_1(W) GROWTH RATE UNDER EACH OPTIMIZER")
print("=" * 100)
print(f"Setup: {NUM_LAYERS}-layer deep linear net (dim={DIM}), quadratic loss, {NUM_STEPS} steps")
print(f"Track: log(sigma_1(W)) vs step for each layer, for SGD and Muon")
print(f"Fit: exponential model log(s1) = a*t + b  vs  polynomial model log(s1) = a*log(t) + b")
print(f"Also: condition number sigma_1/sigma_n, Gini coefficient, corr(sigma_1(G), sigma_1(W))")
print(f"LR_Muon={LR_MUON}, Momentum={MOMENTUM}")
print("=" * 100)

# Find stable SGD learning rate
lr_sgd = find_stable_lr_sgd()
print(f"\nSGD learning rate (max stable): {lr_sgd}")
print(f"Muon learning rate (fixed):     {LR_MUON}")

# Initial loss
np.random.seed(42)
w_test = init_weights(NUM_LAYERS)
loss_init = compute_loss(w_test, X_data, W_target)
print(f"Initial loss: {loss_init:.6e}")

# Run both optimizers
print(f"\n{'=' * 100}")
print("RUNNING OPTIMIZERS AND TRACKING sigma_1 EVOLUTION")
print("=" * 100)

print("\n  Running SGD...", flush=True)
results_sgd = run_and_measure('SGD', sgd_step, lr_sgd, NUM_STEPS)
print(f"    SGD final loss: {results_sgd['losses'][-1]:.6e}")

print("\n  Running Muon...", flush=True)
results_muon = run_and_measure('Muon', muon_step, LR_MUON, NUM_STEPS)
print(f"    Muon final loss: {results_muon['losses'][-1]:.6e}")

steps = np.arange(NUM_STEPS + 1)


# =============================================================================
# ANALYSIS 1: Growth Model Fitting for Each Layer
# =============================================================================

print(f"\n\n{'=' * 100}")
print("TABLE 1: sigma_1(W) GROWTH MODEL FITS PER LAYER")
print("  Exponential model: log(sigma_1) = a*t + b   => sigma_1 ~ exp(a*t)")
print("  Polynomial model:  log(sigma_1) = a*log(t) + b  => sigma_1 ~ t^a")
print("  Better fit = higher R^2")
print("=" * 100)

print(f"\n{'':>3} {'Layer':>5} | {'Exp a':>10} {'Exp R2':>8} | {'Poly a':>10} {'Poly R2':>8} | {'Best Fit':>10}")
print("-" * 75)

sgd_exp_r2_all = []
sgd_poly_r2_all = []
muon_exp_r2_all = []
muon_poly_r2_all = []

print("\n  SGD:")
for layer in range(NUM_LAYERS):
    log_s1 = np.log(results_sgd['sigma1_W'][:, layer] + 1e-30)
    a_exp, b_exp, r2_exp = fit_exponential(steps, log_s1)
    a_poly, b_poly, r2_poly = fit_polynomial(steps, log_s1)
    best = "EXPONENTIAL" if r2_exp > r2_poly else "POLYNOMIAL"
    sgd_exp_r2_all.append(r2_exp)
    sgd_poly_r2_all.append(r2_poly)
    print(f"  {'SGD':>3} {layer:5d} | {a_exp:10.6f} {r2_exp:8.4f} | {a_poly:10.6f} {r2_poly:8.4f} | {best:>10}")

print("\n  Muon:")
for layer in range(NUM_LAYERS):
    log_s1 = np.log(results_muon['sigma1_W'][:, layer] + 1e-30)
    a_exp, b_exp, r2_exp = fit_exponential(steps, log_s1)
    a_poly, b_poly, r2_poly = fit_polynomial(steps, log_s1)
    best = "EXPONENTIAL" if r2_exp > r2_poly else "POLYNOMIAL"
    muon_exp_r2_all.append(r2_exp)
    muon_poly_r2_all.append(r2_poly)
    print(f"  {'Muon':>4} {layer:5d} | {a_exp:10.6f} {r2_exp:8.4f} | {a_poly:10.6f} {r2_poly:8.4f} | {best:>10}")

sgd_mean_exp_r2 = np.mean(sgd_exp_r2_all)
sgd_mean_poly_r2 = np.mean(sgd_poly_r2_all)
muon_mean_exp_r2 = np.mean(muon_exp_r2_all)
muon_mean_poly_r2 = np.mean(muon_poly_r2_all)

print(f"\n  SUMMARY:")
print(f"    SGD  mean R^2:  exponential={sgd_mean_exp_r2:.4f}   polynomial={sgd_mean_poly_r2:.4f}   "
      f"=> {'EXPONENTIAL' if sgd_mean_exp_r2 > sgd_mean_poly_r2 else 'POLYNOMIAL'} fits better")
print(f"    Muon mean R^2:  exponential={muon_mean_exp_r2:.4f}   polynomial={muon_mean_poly_r2:.4f}   "
      f"=> {'EXPONENTIAL' if muon_mean_exp_r2 > muon_mean_poly_r2 else 'POLYNOMIAL'} fits better")


# =============================================================================
# ANALYSIS 2: Condition Number (sigma_1 / sigma_n) Over Training
# =============================================================================

print(f"\n\n{'=' * 100}")
print("TABLE 2: CONDITION NUMBER sigma_1/sigma_n PER LAYER AT KEY STEPS")
print("=" * 100)

report_steps = [0, 50, 100, 200, 300, 500]

print(f"\n  {'Step':>6} | ", end="")
for layer in range(NUM_LAYERS):
    print(f"{'SGD L' + str(layer):>10} {'Muon L' + str(layer):>10} | ", end="")
print()
print("  " + "-" * (8 + (22 + 3) * NUM_LAYERS))

for step in report_steps:
    if step > NUM_STEPS:
        continue
    print(f"  {step:6d} | ", end="")
    for layer in range(NUM_LAYERS):
        s1_sgd = results_sgd['sigma1_W'][step, layer]
        sn_sgd = results_sgd['sigman_W'][step, layer]
        kappa_sgd = s1_sgd / sn_sgd if sn_sgd > 1e-15 else np.inf

        s1_muon = results_muon['sigma1_W'][step, layer]
        sn_muon = results_muon['sigman_W'][step, layer]
        kappa_muon = s1_muon / sn_muon if sn_muon > 1e-15 else np.inf

        k_sgd_str = f"{kappa_sgd:10.2f}" if np.isfinite(kappa_sgd) else f"{'inf':>10}"
        k_muon_str = f"{kappa_muon:10.2f}" if np.isfinite(kappa_muon) else f"{'inf':>10}"
        print(f"{k_sgd_str} {k_muon_str} | ", end="")
    print()


# =============================================================================
# ANALYSIS 3: Gini Coefficient of SV Spectrum at Step 500
# =============================================================================

print(f"\n\n{'=' * 100}")
print("TABLE 3: GINI COEFFICIENT OF SINGULAR VALUE SPECTRUM AT STEP 500")
print("  Gini = 0: all SVs equal (isotropic). Gini -> 1: one SV dominates (anisotropic).")
print("=" * 100)

print(f"\n  {'Layer':>6} | {'SGD Gini':>10} | {'Muon Gini':>11} | {'SGD-Muon':>10}")
print("  " + "-" * 50)

sgd_gini_all = []
muon_gini_all = []

for layer in range(NUM_LAYERS):
    sgd_gini = gini_coefficient(results_sgd['final_sv_spectrum'][layer])
    muon_gini = gini_coefficient(results_muon['final_sv_spectrum'][layer])
    sgd_gini_all.append(sgd_gini)
    muon_gini_all.append(muon_gini)
    print(f"  {layer:6d} | {sgd_gini:10.4f} | {muon_gini:11.4f} | {sgd_gini - muon_gini:+10.4f}")

sgd_mean_gini = np.mean(sgd_gini_all)
muon_mean_gini = np.mean(muon_gini_all)
print("  " + "-" * 50)
print(f"  {'MEAN':>6} | {sgd_mean_gini:10.4f} | {muon_mean_gini:11.4f} | {sgd_mean_gini - muon_mean_gini:+10.4f}")


# =============================================================================
# ANALYSIS 4: Correlation between sigma_1(G_i) and sigma_1(W_i)
# =============================================================================

print(f"\n\n{'=' * 100}")
print("TABLE 4: CORRELATION corr(sigma_1(G), sigma_1(W)) PER LAYER")
print("  High correlation = gradient top direction tracks weight top direction = feedback loop ACTIVE")
print("  Low correlation = gradient spectrum decorrelated from weight spectrum = feedback loop BROKEN")
print("=" * 100)

# Compute Pearson correlation over the training trajectory for each layer
# Use steps 10..500 to avoid initialization transients
start_step = 10

print(f"\n  {'Layer':>6} | {'SGD corr':>10} | {'Muon corr':>11} | {'SGD-Muon':>10}")
print("  " + "-" * 50)

sgd_corr_all = []
muon_corr_all = []

for layer in range(NUM_LAYERS):
    # SGD
    s1_w_sgd = results_sgd['sigma1_W'][start_step:, layer]
    s1_g_sgd = results_sgd['sigma1_G'][start_step:, layer]
    mask_sgd = np.isfinite(s1_w_sgd) & np.isfinite(s1_g_sgd)
    if np.sum(mask_sgd) > 2:
        corr_sgd = np.corrcoef(s1_w_sgd[mask_sgd], s1_g_sgd[mask_sgd])[0, 1]
    else:
        corr_sgd = np.nan
    sgd_corr_all.append(corr_sgd)

    # Muon
    s1_w_muon = results_muon['sigma1_W'][start_step:, layer]
    s1_g_muon = results_muon['sigma1_G'][start_step:, layer]
    mask_muon = np.isfinite(s1_w_muon) & np.isfinite(s1_g_muon)
    if np.sum(mask_muon) > 2:
        corr_muon = np.corrcoef(s1_w_muon[mask_muon], s1_g_muon[mask_muon])[0, 1]
    else:
        corr_muon = np.nan
    muon_corr_all.append(corr_muon)

    print(f"  {layer:6d} | {corr_sgd:10.4f} | {corr_muon:11.4f} | {corr_sgd - corr_muon:+10.4f}")

sgd_mean_corr = np.nanmean(sgd_corr_all)
muon_mean_corr = np.nanmean(muon_corr_all)
print("  " + "-" * 50)
print(f"  {'MEAN':>6} | {sgd_mean_corr:10.4f} | {muon_mean_corr:11.4f} | {sgd_mean_corr - muon_mean_corr:+10.4f}")


# =============================================================================
# ANALYSIS 5: sigma_1 Trajectory Summary
# =============================================================================

print(f"\n\n{'=' * 100}")
print("TABLE 5: sigma_1(W) VALUES AT KEY STEPS")
print("=" * 100)

print(f"\n  {'Step':>6} | ", end="")
for layer in range(NUM_LAYERS):
    print(f"{'SGD L' + str(layer):>8} {'Muon L' + str(layer):>8} | ", end="")
print()
print("  " + "-" * (8 + (18 + 3) * NUM_LAYERS))

for step in report_steps:
    if step > NUM_STEPS:
        continue
    print(f"  {step:6d} | ", end="")
    for layer in range(NUM_LAYERS):
        s1_sgd = results_sgd['sigma1_W'][step, layer]
        s1_muon = results_muon['sigma1_W'][step, layer]
        print(f"{s1_sgd:8.3f} {s1_muon:8.3f} | ", end="")
    print()


# =============================================================================
# ANALYSIS 6: log(sigma_1) growth rate (slope of log(s1) vs t)
# =============================================================================

print(f"\n\n{'=' * 100}")
print("TABLE 6: GROWTH RATE d(log sigma_1)/dt  (exponential fit slope)")
print("  Positive = sigma_1 growing. Larger magnitude = faster growth.")
print("=" * 100)

print(f"\n  {'Layer':>6} | {'SGD slope':>12} | {'Muon slope':>12} | {'Ratio SGD/Muon':>15}")
print("  " + "-" * 55)

sgd_slopes = []
muon_slopes = []

for layer in range(NUM_LAYERS):
    log_s1_sgd = np.log(results_sgd['sigma1_W'][:, layer] + 1e-30)
    a_sgd, _, _ = fit_exponential(steps, log_s1_sgd)
    sgd_slopes.append(a_sgd)

    log_s1_muon = np.log(results_muon['sigma1_W'][:, layer] + 1e-30)
    a_muon, _, _ = fit_exponential(steps, log_s1_muon)
    muon_slopes.append(a_muon)

    ratio = a_sgd / a_muon if abs(a_muon) > 1e-10 else np.inf
    ratio_str = f"{ratio:15.2f}" if np.isfinite(ratio) else f"{'N/A':>15}"
    print(f"  {layer:6d} | {a_sgd:12.6f} | {a_muon:12.6f} | {ratio_str}")

sgd_mean_slope = np.mean(sgd_slopes)
muon_mean_slope = np.mean(muon_slopes)
ratio_mean = sgd_mean_slope / muon_mean_slope if abs(muon_mean_slope) > 1e-10 else np.inf
ratio_str = f"{ratio_mean:15.2f}" if np.isfinite(ratio_mean) else f"{'N/A':>15}"
print("  " + "-" * 55)
print(f"  {'MEAN':>6} | {sgd_mean_slope:12.6f} | {muon_mean_slope:12.6f} | {ratio_str}")


# =============================================================================
# PLOT: log(sigma_1) vs step
# =============================================================================

print(f"\n\n{'=' * 100}")
print("GENERATING PLOTS")
print("=" * 100)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('1.3b-i-A: Feedback Measure -- sigma_1(W) Growth Rate\n'
                 f'{NUM_LAYERS}-layer linear net, dim={DIM}, {NUM_STEPS} steps',
                 fontsize=14, fontweight='bold')

    colors_sgd = ['#1f77b4', '#4a9fd4', '#7ec7f0', '#a8d8f8']
    colors_muon = ['#d62728', '#e74c3c', '#f08080', '#f4a6a6']

    # --- Panel (a): log(sigma_1) vs step ---
    ax = axes[0, 0]
    ax.set_title('(a) log(sigma_1(W)) vs Step')
    for layer in range(NUM_LAYERS):
        log_s1_sgd = np.log(results_sgd['sigma1_W'][:, layer] + 1e-30)
        log_s1_muon = np.log(results_muon['sigma1_W'][:, layer] + 1e-30)
        ax.plot(steps, log_s1_sgd, color=colors_sgd[layer % len(colors_sgd)],
                linewidth=1.5, label=f'SGD L{layer}' if layer == 0 else None, alpha=0.7)
        ax.plot(steps, log_s1_muon, color=colors_muon[layer % len(colors_muon)],
                linewidth=1.5, label=f'Muon L{layer}' if layer == 0 else None,
                linestyle='--', alpha=0.7)
    # Plot mean with bold lines
    sgd_mean_log_s1 = np.mean(np.log(results_sgd['sigma1_W'] + 1e-30), axis=1)
    muon_mean_log_s1 = np.mean(np.log(results_muon['sigma1_W'] + 1e-30), axis=1)
    ax.plot(steps, sgd_mean_log_s1, 'b-', linewidth=3, label='SGD (mean)')
    ax.plot(steps, muon_mean_log_s1, 'r--', linewidth=3, label='Muon (mean)')
    ax.set_xlabel('Step')
    ax.set_ylabel('log(sigma_1)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel (b): sigma_1(W) vs step (linear scale) ---
    ax = axes[0, 1]
    ax.set_title('(b) sigma_1(W) vs Step (linear scale)')
    for layer in range(NUM_LAYERS):
        ax.plot(steps, results_sgd['sigma1_W'][:, layer],
                color=colors_sgd[layer % len(colors_sgd)], linewidth=1.2, alpha=0.7)
        ax.plot(steps, results_muon['sigma1_W'][:, layer],
                color=colors_muon[layer % len(colors_muon)], linewidth=1.2, alpha=0.7,
                linestyle='--')
    sgd_mean_s1 = np.mean(results_sgd['sigma1_W'], axis=1)
    muon_mean_s1 = np.mean(results_muon['sigma1_W'], axis=1)
    ax.plot(steps, sgd_mean_s1, 'b-', linewidth=3, label='SGD (mean)')
    ax.plot(steps, muon_mean_s1, 'r--', linewidth=3, label='Muon (mean)')
    ax.set_xlabel('Step')
    ax.set_ylabel('sigma_1(W)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel (c): Condition number over time ---
    ax = axes[0, 2]
    ax.set_title('(c) Condition Number sigma_1/sigma_n vs Step')
    for layer in range(NUM_LAYERS):
        kappa_sgd = results_sgd['sigma1_W'][:, layer] / np.maximum(results_sgd['sigman_W'][:, layer], 1e-15)
        kappa_muon = results_muon['sigma1_W'][:, layer] / np.maximum(results_muon['sigman_W'][:, layer], 1e-15)
        ax.semilogy(steps, kappa_sgd, color=colors_sgd[layer % len(colors_sgd)],
                     linewidth=1.2, alpha=0.7)
        ax.semilogy(steps, kappa_muon, color=colors_muon[layer % len(colors_muon)],
                     linewidth=1.2, alpha=0.7, linestyle='--')
    # Mean condition number
    kappa_sgd_mean = np.mean(
        results_sgd['sigma1_W'] / np.maximum(results_sgd['sigman_W'], 1e-15), axis=1)
    kappa_muon_mean = np.mean(
        results_muon['sigma1_W'] / np.maximum(results_muon['sigman_W'], 1e-15), axis=1)
    ax.semilogy(steps, kappa_sgd_mean, 'b-', linewidth=3, label='SGD (mean)')
    ax.semilogy(steps, kappa_muon_mean, 'r--', linewidth=3, label='Muon (mean)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Condition Number')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel (d): corr(sigma_1(G), sigma_1(W)) rolling window ---
    ax = axes[1, 0]
    ax.set_title('(d) Rolling corr(sigma_1(G), sigma_1(W))\nwindow=50 steps')
    window = 50
    for layer in range(NUM_LAYERS):
        rolling_corr_sgd = []
        rolling_corr_muon = []
        for t in range(window, NUM_STEPS + 1):
            s1w = results_sgd['sigma1_W'][t - window:t, layer]
            s1g = results_sgd['sigma1_G'][t - window:t, layer]
            if np.std(s1w) > 1e-12 and np.std(s1g) > 1e-12:
                rolling_corr_sgd.append(np.corrcoef(s1w, s1g)[0, 1])
            else:
                rolling_corr_sgd.append(0)

            s1w_m = results_muon['sigma1_W'][t - window:t, layer]
            s1g_m = results_muon['sigma1_G'][t - window:t, layer]
            if np.std(s1w_m) > 1e-12 and np.std(s1g_m) > 1e-12:
                rolling_corr_muon.append(np.corrcoef(s1w_m, s1g_m)[0, 1])
            else:
                rolling_corr_muon.append(0)

        t_axis = np.arange(window, NUM_STEPS + 1)
        ax.plot(t_axis, rolling_corr_sgd,
                color=colors_sgd[layer % len(colors_sgd)], linewidth=1, alpha=0.5)
        ax.plot(t_axis, rolling_corr_muon,
                color=colors_muon[layer % len(colors_muon)], linewidth=1, alpha=0.5,
                linestyle='--')

    # Compute mean rolling corr
    all_rc_sgd = np.zeros(NUM_STEPS + 1 - window)
    all_rc_muon = np.zeros(NUM_STEPS + 1 - window)
    for layer in range(NUM_LAYERS):
        for idx, t in enumerate(range(window, NUM_STEPS + 1)):
            s1w = results_sgd['sigma1_W'][t - window:t, layer]
            s1g = results_sgd['sigma1_G'][t - window:t, layer]
            if np.std(s1w) > 1e-12 and np.std(s1g) > 1e-12:
                all_rc_sgd[idx] += np.corrcoef(s1w, s1g)[0, 1]
            s1w_m = results_muon['sigma1_W'][t - window:t, layer]
            s1g_m = results_muon['sigma1_G'][t - window:t, layer]
            if np.std(s1w_m) > 1e-12 and np.std(s1g_m) > 1e-12:
                all_rc_muon[idx] += np.corrcoef(s1w_m, s1g_m)[0, 1]
    all_rc_sgd /= NUM_LAYERS
    all_rc_muon /= NUM_LAYERS
    t_axis = np.arange(window, NUM_STEPS + 1)
    ax.plot(t_axis, all_rc_sgd, 'b-', linewidth=3, label='SGD (mean)')
    ax.plot(t_axis, all_rc_muon, 'r--', linewidth=3, label='Muon (mean)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Pearson Correlation')
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel (e): Gini coefficient over time ---
    ax = axes[1, 1]
    ax.set_title('(e) Gini Coefficient of SV Spectrum vs Step')
    # Compute Gini at every 10 steps
    gini_steps = np.arange(0, NUM_STEPS + 1, 10)
    sgd_gini_ts = np.zeros((len(gini_steps), NUM_LAYERS))
    muon_gini_ts = np.zeros((len(gini_steps), NUM_LAYERS))

    # We need to re-run to get full SV spectra at intermediate steps
    # Instead, approximate Gini from sigma1/sigman ratio
    # Actually, we only stored sigma1 and sigman, not full spectrum at each step.
    # We can compute Gini from just sigma1 and sigman as an approximation,
    # or we use the tracked data more cleverly.
    # For a cleaner approach: plot sigma1/mean_sigma as a measure of anisotropy
    # which is related to Gini. We have sigma1 and sigman tracked.
    # Let's just plot sigma1/sigman (condition number) which captures the spread.
    # But we already did that in panel (c). Instead let's plot the loss curves.
    ax_loss = ax
    ax_loss.set_title('(e) Loss vs Step')
    ax_loss.semilogy(steps, results_sgd['losses'], 'b-', linewidth=2.5, label='SGD')
    ax_loss.semilogy(steps, results_muon['losses'], 'r--', linewidth=2.5, label='Muon')
    ax_loss.set_xlabel('Step')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend(fontsize=9)
    ax_loss.grid(True, alpha=0.3)

    # --- Panel (f): Final SV spectrum bar chart ---
    ax = axes[1, 2]
    ax.set_title(f'(f) Final SV Spectrum (Layer 0, step {NUM_STEPS})')
    sv_sgd_l0 = results_sgd['final_sv_spectrum'][0]
    sv_muon_l0 = results_muon['final_sv_spectrum'][0]
    x_idx = np.arange(DIM)
    width = 0.35
    ax.bar(x_idx - width / 2, sv_sgd_l0, width, color='blue', alpha=0.6, label='SGD')
    ax.bar(x_idx + width / 2, sv_muon_l0, width, color='red', alpha=0.6, label='Muon')
    ax.set_xlabel('Singular Value Index')
    ax.set_ylabel('Singular Value')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = os.path.join(SCRIPT_DIR, 'sigma1_growth_rate.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved to: {plot_path}")

except ImportError:
    print("\n  WARNING: matplotlib not available, skipping plots.")
    plot_path = None


# =============================================================================
# FINAL VERDICT
# =============================================================================

print(f"\n\n{'=' * 100}")
print("FINAL VERDICT: sigma_1(W) GROWTH RATE FEEDBACK LOOP")
print("=" * 100)

# Test 1: SGD exponential fit R^2 > Muon exponential fit R^2
# (SGD growth is better described by exponential)
test1_pass = sgd_mean_exp_r2 > muon_mean_exp_r2

# Test 2: Muon polynomial fit R^2 > Muon exponential fit R^2
# (Muon growth is better described by sub-linear/polynomial)
test2_pass = muon_mean_poly_r2 > muon_mean_exp_r2

# Test 3: SGD exponential fit R^2 > SGD polynomial fit R^2
# (SGD growth IS exponential, not polynomial)
test3_pass = sgd_mean_exp_r2 > sgd_mean_poly_r2

# Test 4: SGD corr(sigma1_G, sigma1_W) > Muon corr(sigma1_G, sigma1_W)
# (feedback loop is active for SGD, broken for Muon)
test4_pass = sgd_mean_corr > muon_mean_corr

# Test 5: SGD Gini > Muon Gini at final step
# (SGD has more anisotropic spectrum)
test5_pass = sgd_mean_gini > muon_mean_gini

# Composite: SGD exponential > Muon sub-linear
# This is the key claim: R2_exp(SGD) > R2_exp(Muon) AND R2_poly(Muon) > R2_exp(Muon)
composite_pass = test1_pass and test2_pass

tests = [test1_pass, test2_pass, test3_pass, test4_pass, test5_pass]
tests_passed = sum(tests)
tests_total = 5

print(f"""
  MEASURED QUANTITIES:
  ---------------------------------------------------------------
  Growth Model Fit (mean R^2 across layers):
    SGD:   exponential R^2 = {sgd_mean_exp_r2:.4f}   polynomial R^2 = {sgd_mean_poly_r2:.4f}
    Muon:  exponential R^2 = {muon_mean_exp_r2:.4f}   polynomial R^2 = {muon_mean_poly_r2:.4f}

  Feedback Loop Correlation corr(sigma_1(G), sigma_1(W)):
    SGD:   {sgd_mean_corr:.4f}
    Muon:  {muon_mean_corr:.4f}

  Gini Coefficient of SV Spectrum at step {NUM_STEPS}:
    SGD:   {sgd_mean_gini:.4f}
    Muon:  {muon_mean_gini:.4f}

  sigma_1 exponential growth rate (slope of log(s1) vs t):
    SGD:   {sgd_mean_slope:.6f} per step
    Muon:  {muon_mean_slope:.6f} per step
  ---------------------------------------------------------------

  HYPOTHESIS CHECKS:
  ---------------------------------------------------------------
  T1: SGD exp-R^2 > Muon exp-R^2 (SGD growth is more exponential)
      SGD: {sgd_mean_exp_r2:.4f} vs Muon: {muon_mean_exp_r2:.4f}
      -> {"CONFIRMED" if test1_pass else "REJECTED"}

  T2: Muon poly-R^2 > Muon exp-R^2 (Muon growth is sub-linear)
      poly: {muon_mean_poly_r2:.4f} vs exp: {muon_mean_exp_r2:.4f}
      -> {"CONFIRMED" if test2_pass else "REJECTED"}

  T3: SGD exp-R^2 > SGD poly-R^2 (SGD growth IS exponential)
      exp: {sgd_mean_exp_r2:.4f} vs poly: {sgd_mean_poly_r2:.4f}
      -> {"CONFIRMED" if test3_pass else "REJECTED"}

  T4: SGD corr > Muon corr (feedback loop active for SGD, broken for Muon)
      SGD: {sgd_mean_corr:.4f} vs Muon: {muon_mean_corr:.4f}
      -> {"CONFIRMED" if test4_pass else "REJECTED"}

  T5: SGD Gini > Muon Gini (SGD has more anisotropic spectrum)
      SGD: {sgd_mean_gini:.4f} vs Muon: {muon_mean_gini:.4f}
      -> {"CONFIRMED" if test5_pass else "REJECTED"}

  COMPOSITE: SGD exponential > Muon sub-linear
      (T1 AND T2): {"CONFIRMED" if composite_pass else "REJECTED"}
  ---------------------------------------------------------------
""")

if tests_passed >= 4 and composite_pass:
    overall = "PASS"
    detail = (
        f"  {tests_passed}/5 tests pass + composite confirmed.\n"
        "  SGD sigma_1 growth is exponential (positive feedback loop).\n"
        "  Muon sigma_1 growth is sub-linear (bounded step size breaks feedback).\n"
        "  Gradient-weight spectral correlation confirms: SGD feedback loop active,\n"
        "  Muon feedback loop broken by orthogonalization."
    )
elif tests_passed >= 3:
    overall = "PARTIAL PASS"
    detail = (
        f"  {tests_passed}/5 tests pass, composite={'PASS' if composite_pass else 'FAIL'}.\n"
        f"  T1 (SGD exp > Muon exp):    {'PASS' if test1_pass else 'FAIL'}\n"
        f"  T2 (Muon poly > Muon exp):  {'PASS' if test2_pass else 'FAIL'}\n"
        f"  T3 (SGD exp > SGD poly):    {'PASS' if test3_pass else 'FAIL'}\n"
        f"  T4 (SGD corr > Muon corr):  {'PASS' if test4_pass else 'FAIL'}\n"
        f"  T5 (SGD Gini > Muon Gini):  {'PASS' if test5_pass else 'FAIL'}"
    )
elif tests_passed >= 2:
    overall = "WEAK SIGNAL"
    detail = (
        f"  {tests_passed}/5 tests pass, composite={'PASS' if composite_pass else 'FAIL'}.\n"
        f"  T1 (SGD exp > Muon exp):    {'PASS' if test1_pass else 'FAIL'}\n"
        f"  T2 (Muon poly > Muon exp):  {'PASS' if test2_pass else 'FAIL'}\n"
        f"  T3 (SGD exp > SGD poly):    {'PASS' if test3_pass else 'FAIL'}\n"
        f"  T4 (SGD corr > Muon corr):  {'PASS' if test4_pass else 'FAIL'}\n"
        f"  T5 (SGD Gini > Muon Gini):  {'PASS' if test5_pass else 'FAIL'}"
    )
else:
    overall = "FAIL"
    detail = (
        f"  Only {tests_passed}/5 tests pass, composite={'PASS' if composite_pass else 'FAIL'}.\n"
        f"  T1 (SGD exp > Muon exp):    {'PASS' if test1_pass else 'FAIL'}\n"
        f"  T2 (Muon poly > Muon exp):  {'PASS' if test2_pass else 'FAIL'}\n"
        f"  T3 (SGD exp > SGD poly):    {'PASS' if test3_pass else 'FAIL'}\n"
        f"  T4 (SGD corr > Muon corr):  {'PASS' if test4_pass else 'FAIL'}\n"
        f"  T5 (SGD Gini > Muon Gini):  {'PASS' if test5_pass else 'FAIL'}"
    )

print(f"""
  +========================================================================+
  |  VERDICT: {overall:<63}|
  +========================================================================+
  |                                                                        |""")
for line in detail.split('\n'):
    print(f"  |  {line:<70}|")
print(f"""  |                                                                        |
  +========================================================================+
""")

print("=" * 100)
print(f"  Tests passed: {tests_passed}/{tests_total}")
print(f"  Composite (SGD exponential > Muon sub-linear): {'PASS' if composite_pass else 'FAIL'}")
print(f"  Overall: {overall}")
print("=" * 100)
