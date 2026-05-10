#!/usr/bin/env python3
"""
1.2b-i: Lyapunov Exponent of Gauge Direction Under Each Optimizer
=================================================================
PREDICTION (from dynamical systems / RG gauge-fixing model):
  Start from W_0, perturb to W_0' = W_0 @ (I + eps*S/||S||_F) where S is
  random symmetric (gauge/PSD direction), eps=0.001.

  Run N=200 steps of each optimizer from both starting points.
  Compute: lambda = (1/N) * log(d(N) / d(0))
  where d(t) = ||W_t' - W_t||_F.

  This is the Lyapunov exponent along the gauge (PSD) direction.
    lambda < 0  =>  gauge perturbation DECAYS (stable)
    lambda > 0  =>  gauge perturbation GROWS (unstable)
    lambda = 0  =>  neutral

  HYPOTHESIS:
    Muon:  lambda << 0  (strongly negative -- perturbations decay exponentially)
    SGD:   lambda ~ 0   (neutral -- random walk)
    Adam:  lambda slightly negative

  CRITICAL CONTEXT:
    - D-TEST confirmed SGD advantage compounds exponentially with depth
      (per-layer Lyapunov ~0.095)
    - P17 perspective predicted: SGD trajectories diverge (lambda>0),
      Muon converges (lambda<0)
    - This experiment directly measures the Lyapunov exponent predicted
      by the dynamical systems model

  Also compare with PHYSICAL (skew-symmetric/tangent) perturbation direction:
    W_0' = W_0 @ expm(eps*A/||A||_F) where A is skew-symmetric.
    Both optimizers should be ~neutral in the physical direction, since that
    direction changes the actual function computed by the network.

Setup: 4-layer deep linear net, 32x32, quadratic loss.
"""

import numpy as np
import os

np.random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================

DIM = 32
NUM_LAYERS = 4
NUM_STEPS = 200
BATCH_SIZE = 64
LR_MUON = 0.005
MOMENTUM = 0.9
NS_ITERS = 5
EPSILON = 0.001
NUM_PERTURBATIONS = 20

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


def find_stable_lr_sgd():
    """Find maximum stable SGD learning rate."""
    candidates = [0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.001]
    for lr in candidates:
        np.random.seed(42)
        weights = init_weights(NUM_LAYERS)
        velocities = [np.zeros((DIM, DIM)) for _ in range(NUM_LAYERS)]
        initial_loss = compute_loss(weights, X_data, W_target)
        stable = True
        for step in range(80):
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


def random_symmetric(dim, rng):
    """Generate a random symmetric matrix."""
    A = rng.randn(dim, dim)
    return (A + A.T) / 2.0


def random_skew_symmetric(dim, rng):
    """Generate a random skew-symmetric matrix."""
    A = rng.randn(dim, dim)
    return (A - A.T) / 2.0


def matrix_exponential_pade(A, order=6):
    """
    Compute matrix exponential via scaling-and-squaring with Pade approximation.
    For small ||A||, direct Pade is accurate. For larger ||A||, we scale down first.
    """
    norm_A = np.linalg.norm(A, ord='fro')
    if norm_A < 1e-15:
        return np.eye(A.shape[0])

    # Scaling: find s such that ||A/2^s|| < 1
    s = max(0, int(np.ceil(np.log2(norm_A + 1e-15))))
    A_scaled = A / (2 ** s)

    # Pade [order/order] approximant
    I = np.eye(A.shape[0])
    N_pade = I.copy()
    D_pade = I.copy()
    A_power = I.copy()
    c = 1.0

    for k in range(1, order + 1):
        c *= (order - k + 1) / (k * (2 * order - k + 1))
        A_power = A_power @ A_scaled
        N_pade += c * A_power
        D_pade += ((-1) ** k) * c * A_power

    # expm(A_scaled) ~ D^{-1} N
    result = np.linalg.solve(D_pade, N_pade)

    # Squaring phase
    for _ in range(s):
        result = result @ result

    return result


# =============================================================================
# OPTIMIZER STEP FUNCTIONS
# =============================================================================

def sgd_step(weights, velocities, lr):
    """One step of SGD with momentum."""
    grads = compute_gradients(weights, X_data, W_target)
    for i in range(len(weights)):
        velocities[i] = MOMENTUM * velocities[i] + grads[i]
        weights[i] = weights[i] - lr * velocities[i]
    return weights, velocities


def muon_step(weights, velocities, lr):
    """One step of Muon with momentum."""
    grads = compute_gradients(weights, X_data, W_target)
    for i in range(len(weights)):
        ortho_grad = newton_schulz_orthogonalize(grads[i])
        velocities[i] = MOMENTUM * velocities[i] + ortho_grad
        weights[i] = weights[i] - lr * velocities[i]
    return weights, velocities


# =============================================================================
# LYAPUNOV MEASUREMENT ENGINE
# =============================================================================

def run_trajectory(weights_init, optimizer, lr, num_steps):
    """
    Run optimizer for num_steps from weights_init.
    Returns list of weight snapshots at each step (including step 0).
    """
    weights = [w.copy() for w in weights_init]
    velocities = [np.zeros_like(w) for w in weights]

    trajectory = [[w.copy() for w in weights]]

    for step in range(num_steps):
        if optimizer == 'sgd':
            weights, velocities = sgd_step(weights, velocities, lr)
        elif optimizer == 'muon':
            weights, velocities = muon_step(weights, velocities, lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        trajectory.append([w.copy() for w in weights])

        # Check for divergence
        loss = compute_loss(weights, X_data, W_target)
        if np.isnan(loss) or loss > 1e10:
            # Pad remaining with last valid state
            for _ in range(num_steps - step - 1):
                trajectory.append([w.copy() for w in weights])
            break

    return trajectory


def compute_trajectory_distance(traj_a, traj_b):
    """
    Compute d(t) = ||W_t^a - W_t^b||_F (summed over all layers) at each timestep.
    """
    num_steps = min(len(traj_a), len(traj_b))
    distances = []
    for t in range(num_steps):
        d = 0.0
        for i in range(len(traj_a[t])):
            d += np.linalg.norm(traj_a[t][i] - traj_b[t][i], 'fro') ** 2
        distances.append(np.sqrt(d))
    return np.array(distances)


def compute_per_layer_distances(traj_a, traj_b):
    """Compute per-layer distances over time."""
    num_steps = min(len(traj_a), len(traj_b))
    num_layers = len(traj_a[0])
    per_layer = np.zeros((num_layers, num_steps))
    for t in range(num_steps):
        for i in range(num_layers):
            per_layer[i, t] = np.linalg.norm(traj_a[t][i] - traj_b[t][i], 'fro')
    return per_layer


def measure_lyapunov(optimizer, lr, perturbation_type, num_perturbations, seed_base=100):
    """
    Measure Lyapunov exponent for a given optimizer and perturbation type.

    perturbation_type: 'gauge' (symmetric/PSD) or 'physical' (skew-symmetric/tangent)

    Returns:
      lyapunov_exponents: array of shape (num_perturbations,)
      all_distances: list of distance arrays for plotting
      d0_values: initial distances
      dN_values: final distances
    """
    # Initialize base weights
    np.random.seed(42)
    weights_base = init_weights(NUM_LAYERS)

    # Run base trajectory once
    traj_base = run_trajectory(weights_base, optimizer, lr, NUM_STEPS)

    lyapunov_exponents = []
    all_distances = []
    d0_values = []
    dN_values = []

    for p in range(num_perturbations):
        rng_pert = np.random.RandomState(seed_base + p)

        # Create perturbed initial weights
        weights_perturbed = []
        for layer_idx in range(NUM_LAYERS):
            W0 = weights_base[layer_idx].copy()

            if perturbation_type == 'gauge':
                # Gauge direction: W' = W @ (I + eps * S / ||S||_F)
                # S is symmetric => (I + eps*S) is approximately PSD for small eps
                S = random_symmetric(DIM, rng_pert)
                S = S / np.linalg.norm(S, 'fro')
                W_pert = W0 @ (np.eye(DIM) + EPSILON * S)
            elif perturbation_type == 'physical':
                # Physical/tangent direction: W' = W @ expm(eps * A / ||A||_F)
                # A is skew-symmetric => expm(A) is orthogonal
                A = random_skew_symmetric(DIM, rng_pert)
                A = A / np.linalg.norm(A, 'fro')
                R = matrix_exponential_pade(EPSILON * A)
                W_pert = W0 @ R
            else:
                raise ValueError(f"Unknown perturbation type: {perturbation_type}")

            weights_perturbed.append(W_pert)

        # Run perturbed trajectory
        traj_perturbed = run_trajectory(weights_perturbed, optimizer, lr, NUM_STEPS)

        # Compute distances
        distances = compute_trajectory_distance(traj_base, traj_perturbed)
        all_distances.append(distances)

        d0 = distances[0]
        dN = distances[-1]
        d0_values.append(d0)
        dN_values.append(dN)

        # Lyapunov exponent: lambda = (1/N) * log(d(N) / d(0))
        if d0 > 1e-15 and dN > 1e-15:
            lyap = (1.0 / NUM_STEPS) * np.log(dN / d0)
        elif dN < 1e-15:
            lyap = -np.inf  # Perturbation collapsed
        else:
            lyap = np.nan

        lyapunov_exponents.append(lyap)

    return (np.array(lyapunov_exponents), all_distances,
            np.array(d0_values), np.array(dN_values))


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

print("=" * 100)
print("1.2b-i: LYAPUNOV EXPONENT OF GAUGE DIRECTION UNDER EACH OPTIMIZER")
print("=" * 100)
print(f"Setup: {NUM_LAYERS}-layer deep linear net (dim={DIM}), quadratic loss, {NUM_STEPS} steps")
print(f"Perturbation: eps={EPSILON}, {NUM_PERTURBATIONS} random directions")
print(f"LR_Muon={LR_MUON}, Momentum={MOMENTUM}")
print("=" * 100)

# Find stable SGD learning rate
lr_sgd = find_stable_lr_sgd()
print(f"\nSGD learning rate (max stable): {lr_sgd}")
print(f"Muon learning rate (fixed):     {LR_MUON}")

# Verify both optimizers train properly
np.random.seed(42)
w_test = init_weights(NUM_LAYERS)
loss_init = compute_loss(w_test, X_data, W_target)
print(f"\nInitial loss: {loss_init:.6e}")

# Quick check: run both optimizers
for opt_name, opt_type, lr in [('SGD', 'sgd', lr_sgd), ('Muon', 'muon', LR_MUON)]:
    np.random.seed(42)
    w_check = init_weights(NUM_LAYERS)
    traj_check = run_trajectory(w_check, opt_type, lr, NUM_STEPS)
    loss_final = compute_loss(traj_check[-1], X_data, W_target)
    print(f"  {opt_name} final loss after {NUM_STEPS} steps: {loss_final:.6e}")


# =============================================================================
# MEASURE LYAPUNOV EXPONENTS
# =============================================================================

print(f"\n{'=' * 100}")
print("MEASURING LYAPUNOV EXPONENTS")
print("=" * 100)

results = {}

for opt_name, opt_type, lr in [('SGD', 'sgd', lr_sgd), ('Muon', 'muon', LR_MUON)]:
    for pert_name, pert_type in [('gauge', 'gauge'), ('physical', 'physical')]:
        key = f"{opt_name}_{pert_name}"
        print(f"\n  Running {opt_name} with {pert_name} perturbation "
              f"({NUM_PERTURBATIONS} trials)...", flush=True)

        lyaps, dists, d0s, dNs = measure_lyapunov(
            opt_type, lr, pert_type, NUM_PERTURBATIONS
        )

        # Filter out any inf/nan
        valid_mask = np.isfinite(lyaps)
        lyaps_valid = lyaps[valid_mask]

        results[key] = {
            'lyapunov_all': lyaps,
            'lyapunov_valid': lyaps_valid,
            'distances': dists,
            'd0': d0s,
            'dN': dNs,
            'mean_lyap': np.mean(lyaps_valid) if len(lyaps_valid) > 0 else np.nan,
            'std_lyap': np.std(lyaps_valid) if len(lyaps_valid) > 0 else np.nan,
            'median_lyap': np.median(lyaps_valid) if len(lyaps_valid) > 0 else np.nan,
        }

        print(f"    Valid trials: {len(lyaps_valid)}/{NUM_PERTURBATIONS}")
        print(f"    Mean lambda:   {results[key]['mean_lyap']:.6f}")
        print(f"    Median lambda: {results[key]['median_lyap']:.6f}")
        print(f"    Std lambda:    {results[key]['std_lyap']:.6f}")
        print(f"    Mean d(0):     {np.mean(d0s):.6e}")
        print(f"    Mean d(N):     {np.mean(dNs):.6e}")
        print(f"    Ratio d(N)/d(0): {np.mean(dNs)/np.mean(d0s):.4f}")


# =============================================================================
# DETAILED RESULTS TABLE
# =============================================================================

print(f"\n\n{'=' * 100}")
print("DETAILED LYAPUNOV EXPONENT RESULTS")
print("=" * 100)

print(f"\n{'Optimizer':<10} | {'Perturbation':<12} | {'Mean lambda':>12} | {'Median lambda':>14} | "
      f"{'Std':>8} | {'d(N)/d(0)':>10} | {'Sign':>8}")
print("-" * 90)

for opt_name in ['SGD', 'Muon']:
    for pert_name in ['gauge', 'physical']:
        key = f"{opt_name}_{pert_name}"
        r = results[key]
        ratio = np.mean(r['dN']) / np.mean(r['d0']) if np.mean(r['d0']) > 0 else np.nan

        if r['mean_lyap'] < -0.001:
            sign_str = "DECAY"
        elif r['mean_lyap'] > 0.001:
            sign_str = "GROW"
        else:
            sign_str = "NEUTRAL"

        print(f"{opt_name:<10} | {pert_name:<12} | {r['mean_lyap']:12.6f} | "
              f"{r['median_lyap']:14.6f} | {r['std_lyap']:8.6f} | {ratio:10.4f} | {sign_str:>8}")


# =============================================================================
# PER-TRIAL BREAKDOWN
# =============================================================================

print(f"\n\n{'=' * 100}")
print("PER-TRIAL LYAPUNOV EXPONENTS (GAUGE DIRECTION)")
print("=" * 100)

print(f"\n{'Trial':>6} | {'SGD lambda':>12} | {'Muon lambda':>12} | {'SGD d(N)/d(0)':>14} | "
      f"{'Muon d(N)/d(0)':>14}")
print("-" * 75)

sgd_g = results['SGD_gauge']
muon_g = results['Muon_gauge']

for p in range(NUM_PERTURBATIONS):
    sgd_ratio = sgd_g['dN'][p] / sgd_g['d0'][p] if sgd_g['d0'][p] > 0 else np.nan
    muon_ratio = muon_g['dN'][p] / muon_g['d0'][p] if muon_g['d0'][p] > 0 else np.nan
    print(f"{p:6d} | {sgd_g['lyapunov_all'][p]:12.6f} | {muon_g['lyapunov_all'][p]:12.6f} | "
          f"{sgd_ratio:14.6f} | {muon_ratio:14.6f}")


# =============================================================================
# PER-LAYER ANALYSIS (representative trial)
# =============================================================================

print(f"\n\n{'=' * 100}")
print("PER-LAYER DISTANCE EVOLUTION (Trial 0, Gauge Perturbation)")
print("=" * 100)

# Re-run trial 0 with per-layer tracking
np.random.seed(42)
weights_base = init_weights(NUM_LAYERS)
rng_layer = np.random.RandomState(100)

# Create gauge-perturbed weights for trial 0
weights_pert_gauge = []
for layer_idx in range(NUM_LAYERS):
    S = random_symmetric(DIM, rng_layer)
    S = S / np.linalg.norm(S, 'fro')
    W_pert = weights_base[layer_idx] @ (np.eye(DIM) + EPSILON * S)
    weights_pert_gauge.append(W_pert)

for opt_name, opt_type, lr in [('SGD', 'sgd', lr_sgd), ('Muon', 'muon', LR_MUON)]:
    traj_base = run_trajectory(weights_base, opt_type, lr, NUM_STEPS)
    traj_pert = run_trajectory(weights_pert_gauge, opt_type, lr, NUM_STEPS)
    per_layer = compute_per_layer_distances(traj_base, traj_pert)

    print(f"\n  {opt_name}:")
    print(f"  {'Layer':>6} | {'d(0)':>10} | {'d(50)':>10} | {'d(100)':>10} | "
          f"{'d(150)':>10} | {'d(200)':>10} | {'lambda_layer':>12}")
    print("  " + "-" * 80)

    for layer_idx in range(NUM_LAYERS):
        d0_l = per_layer[layer_idx, 0]
        dN_l = per_layer[layer_idx, -1]
        lyap_l = (1.0 / NUM_STEPS) * np.log(dN_l / d0_l) if d0_l > 1e-15 and dN_l > 1e-15 else np.nan
        print(f"  {layer_idx:6d} | {per_layer[layer_idx, 0]:10.6f} | "
              f"{per_layer[layer_idx, 50]:10.6f} | {per_layer[layer_idx, 100]:10.6f} | "
              f"{per_layer[layer_idx, 150]:10.6f} | {per_layer[layer_idx, -1]:10.6f} | "
              f"{lyap_l:12.6f}")


# =============================================================================
# STATISTICAL SIGNIFICANCE
# =============================================================================

print(f"\n\n{'=' * 100}")
print("STATISTICAL SIGNIFICANCE TEST")
print("=" * 100)

sgd_gauge_lyaps = results['SGD_gauge']['lyapunov_valid']
muon_gauge_lyaps = results['Muon_gauge']['lyapunov_valid']

# Two-sample t-test (manual, no scipy)
n1, n2 = len(sgd_gauge_lyaps), len(muon_gauge_lyaps)
mean1, mean2 = np.mean(sgd_gauge_lyaps), np.mean(muon_gauge_lyaps)
var1, var2 = np.var(sgd_gauge_lyaps, ddof=1), np.var(muon_gauge_lyaps, ddof=1)

if n1 > 1 and n2 > 1:
    se = np.sqrt(var1 / n1 + var2 / n2)
    if se > 1e-15:
        t_stat = (mean1 - mean2) / se
        # Welch's degrees of freedom
        df_num = (var1 / n1 + var2 / n2) ** 2
        df_den = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
        df = df_num / (df_den + 1e-15)
    else:
        t_stat = np.inf if mean1 != mean2 else 0.0
        df = min(n1, n2) - 1
else:
    t_stat = np.nan
    df = np.nan

print(f"\n  H0: lambda_gauge_SGD = lambda_gauge_Muon")
print(f"  H1: lambda_gauge_SGD > lambda_gauge_Muon (Muon more stable)")
print(f"\n  SGD gauge Lyapunov:  mean={mean1:.6f}, std={np.sqrt(var1):.6f}, n={n1}")
print(f"  Muon gauge Lyapunov: mean={mean2:.6f}, std={np.sqrt(var2):.6f}, n={n2}")
print(f"  Difference (SGD - Muon): {mean1 - mean2:.6f}")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  Degrees of freedom: {df:.1f}")
print(f"  (t > 2.0 indicates statistical significance at p < 0.05, one-tailed)")

is_significant = t_stat > 2.0 if np.isfinite(t_stat) else False
print(f"  Statistically significant: {'YES' if is_significant else 'NO'}")


# =============================================================================
# GAUGE vs PHYSICAL COMPARISON
# =============================================================================

print(f"\n\n{'=' * 100}")
print("GAUGE vs PHYSICAL PERTURBATION COMPARISON")
print("=" * 100)

for opt_name in ['SGD', 'Muon']:
    gauge_key = f"{opt_name}_gauge"
    phys_key = f"{opt_name}_physical"
    print(f"\n  {opt_name}:")
    print(f"    Gauge lambda:    {results[gauge_key]['mean_lyap']:.6f} +/- {results[gauge_key]['std_lyap']:.6f}")
    print(f"    Physical lambda: {results[phys_key]['mean_lyap']:.6f} +/- {results[phys_key]['std_lyap']:.6f}")
    diff = results[gauge_key]['mean_lyap'] - results[phys_key]['mean_lyap']
    print(f"    Difference (gauge - physical): {diff:.6f}")
    if abs(diff) > 0.001:
        if diff < 0:
            print(f"    -> Gauge direction is MORE STABLE than physical (confirms gauge fixing)")
        else:
            print(f"    -> Physical direction is MORE STABLE than gauge (unexpected)")
    else:
        print(f"    -> Similar stability in both directions")


# =============================================================================
# PLOT: d(t) OVER TIME
# =============================================================================

print(f"\n\n{'=' * 100}")
print("GENERATING PLOTS")
print("=" * 100)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('1.2b-i: Lyapunov Exponent of Gauge Direction\n'
                 f'{NUM_LAYERS}-layer linear net, dim={DIM}, eps={EPSILON}, '
                 f'{NUM_PERTURBATIONS} perturbations',
                 fontsize=14, fontweight='bold')

    t_axis = np.arange(NUM_STEPS + 1)

    # --- Panel (a): d(t) for gauge perturbation, both optimizers ---
    ax = axes[0, 0]
    ax.set_title('(a) Gauge Perturbation: d(t) over time')

    # Plot individual trials (faint) and mean (bold)
    for p in range(NUM_PERTURBATIONS):
        dists_sgd = results['SGD_gauge']['distances'][p]
        dists_muon = results['Muon_gauge']['distances'][p]
        ax.semilogy(t_axis[:len(dists_sgd)], dists_sgd, 'b-', alpha=0.1, linewidth=0.5)
        ax.semilogy(t_axis[:len(dists_muon)], dists_muon, 'r-', alpha=0.1, linewidth=0.5)

    # Mean distance
    sgd_mean_dist = np.mean([d for d in results['SGD_gauge']['distances']], axis=0)
    muon_mean_dist = np.mean([d for d in results['Muon_gauge']['distances']], axis=0)
    ax.semilogy(t_axis[:len(sgd_mean_dist)], sgd_mean_dist, 'b-', linewidth=2.5,
                label=f'SGD (lambda={results["SGD_gauge"]["mean_lyap"]:.4f})')
    ax.semilogy(t_axis[:len(muon_mean_dist)], muon_mean_dist, 'r-', linewidth=2.5,
                label=f'Muon (lambda={results["Muon_gauge"]["mean_lyap"]:.4f})')

    # Reference: d(0) line
    ax.axhline(y=np.mean(results['SGD_gauge']['d0']), color='gray', linestyle='--',
               alpha=0.5, label='d(0)')
    ax.set_xlabel('Step')
    ax.set_ylabel('d(t) = ||W_t\' - W_t||_F')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel (b): d(t) for physical perturbation, both optimizers ---
    ax = axes[0, 1]
    ax.set_title('(b) Physical Perturbation: d(t) over time')

    for p in range(NUM_PERTURBATIONS):
        dists_sgd = results['SGD_physical']['distances'][p]
        dists_muon = results['Muon_physical']['distances'][p]
        ax.semilogy(t_axis[:len(dists_sgd)], dists_sgd, 'b-', alpha=0.1, linewidth=0.5)
        ax.semilogy(t_axis[:len(dists_muon)], dists_muon, 'r-', alpha=0.1, linewidth=0.5)

    sgd_phys_mean = np.mean([d for d in results['SGD_physical']['distances']], axis=0)
    muon_phys_mean = np.mean([d for d in results['Muon_physical']['distances']], axis=0)
    ax.semilogy(t_axis[:len(sgd_phys_mean)], sgd_phys_mean, 'b-', linewidth=2.5,
                label=f'SGD (lambda={results["SGD_physical"]["mean_lyap"]:.4f})')
    ax.semilogy(t_axis[:len(muon_phys_mean)], muon_phys_mean, 'r-', linewidth=2.5,
                label=f'Muon (lambda={results["Muon_physical"]["mean_lyap"]:.4f})')
    ax.axhline(y=np.mean(results['SGD_physical']['d0']), color='gray', linestyle='--',
               alpha=0.5, label='d(0)')
    ax.set_xlabel('Step')
    ax.set_ylabel('d(t) = ||W_t\' - W_t||_F')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel (c): Histogram of Lyapunov exponents ---
    ax = axes[1, 0]
    ax.set_title('(c) Distribution of Lyapunov Exponents (Gauge)')

    bins = np.linspace(
        min(np.min(results['SGD_gauge']['lyapunov_valid']),
            np.min(results['Muon_gauge']['lyapunov_valid'])) - 0.005,
        max(np.max(results['SGD_gauge']['lyapunov_valid']),
            np.max(results['Muon_gauge']['lyapunov_valid'])) + 0.005,
        25
    )
    ax.hist(results['SGD_gauge']['lyapunov_valid'], bins=bins, alpha=0.6, color='blue',
            label=f'SGD (mean={results["SGD_gauge"]["mean_lyap"]:.4f})', edgecolor='navy')
    ax.hist(results['Muon_gauge']['lyapunov_valid'], bins=bins, alpha=0.6, color='red',
            label=f'Muon (mean={results["Muon_gauge"]["mean_lyap"]:.4f})', edgecolor='darkred')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='lambda=0 (neutral)')
    ax.set_xlabel('Lyapunov Exponent (lambda)')
    ax.set_ylabel('Count')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel (d): Gauge vs Physical comparison (bar chart) ---
    ax = axes[1, 1]
    ax.set_title('(d) Gauge vs Physical Lyapunov Exponents')

    categories = ['SGD\nGauge', 'SGD\nPhysical', 'Muon\nGauge', 'Muon\nPhysical']
    means = [
        results['SGD_gauge']['mean_lyap'],
        results['SGD_physical']['mean_lyap'],
        results['Muon_gauge']['mean_lyap'],
        results['Muon_physical']['mean_lyap'],
    ]
    stds = [
        results['SGD_gauge']['std_lyap'],
        results['SGD_physical']['std_lyap'],
        results['Muon_gauge']['std_lyap'],
        results['Muon_physical']['std_lyap'],
    ]
    colors = ['#4477AA', '#88CCEE', '#CC3311', '#EE7733']

    bars = ax.bar(categories, means, yerr=stds, capsize=5, color=colors,
                  edgecolor='black', linewidth=0.8)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
    ax.set_ylabel('Lyapunov Exponent (lambda)')

    # Annotate bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                f'{mean:.4f}', ha='center', va='bottom' if mean >= 0 else 'top',
                fontsize=9, fontweight='bold')

    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = os.path.join(SCRIPT_DIR, 'lyapunov_gauge_direction.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved to: {plot_path}")

except ImportError:
    print("\n  WARNING: matplotlib not available, skipping plots.")
    plot_path = None


# =============================================================================
# ADDITIONAL PLOT: log(d(t)/d(0)) vs t (slope = Lyapunov)
# =============================================================================

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('1.2b-i: log(d(t)/d(0)) vs t -- Slope = Lyapunov Exponent',
                 fontsize=13, fontweight='bold')

    for idx, (pert_name, pert_label) in enumerate([('gauge', 'Gauge (Symmetric/PSD)'),
                                                     ('physical', 'Physical (Skew/Tangent)')]):
        ax = axes[idx]
        ax.set_title(pert_label)

        for opt_name, color in [('SGD', 'blue'), ('Muon', 'red')]:
            key = f"{opt_name}_{pert_name}"
            dists_list = results[key]['distances']

            # Plot individual trials
            for p in range(NUM_PERTURBATIONS):
                d = dists_list[p]
                d0 = d[0]
                if d0 > 1e-15:
                    log_ratio = np.log(d / d0)
                    ax.plot(t_axis[:len(log_ratio)], log_ratio, color=color,
                            alpha=0.1, linewidth=0.5)

            # Mean
            mean_dist = np.mean(dists_list, axis=0)
            d0_mean = mean_dist[0]
            if d0_mean > 1e-15:
                log_ratio_mean = np.log(mean_dist / d0_mean)
                ax.plot(t_axis[:len(log_ratio_mean)], log_ratio_mean, color=color,
                        linewidth=2.5, label=f'{opt_name} (lambda={results[key]["mean_lyap"]:.4f})')

                # Linear fit line
                lyap = results[key]['mean_lyap']
                ax.plot(t_axis, lyap * t_axis, color=color, linestyle='--',
                        linewidth=1.5, alpha=0.7)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('log(d(t) / d(0))')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path2 = os.path.join(SCRIPT_DIR, 'lyapunov_log_ratio.png')
    plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved to: {plot_path2}")

except ImportError:
    pass


# =============================================================================
# VERDICT
# =============================================================================

print(f"\n\n{'=' * 100}")
print("FINAL VERDICT: LYAPUNOV EXPONENT OF GAUGE DIRECTION")
print("=" * 100)

lambda_gauge_sgd = results['SGD_gauge']['mean_lyap']
lambda_gauge_muon = results['Muon_gauge']['mean_lyap']
lambda_phys_sgd = results['SGD_physical']['mean_lyap']
lambda_phys_muon = results['Muon_physical']['mean_lyap']

print(f"""
  MEASURED LYAPUNOV EXPONENTS:
  ---------------------------------------------------------------
  lambda_gauge_SGD    = {lambda_gauge_sgd:+.6f}  ({"DECAY" if lambda_gauge_sgd < -0.001 else "GROW" if lambda_gauge_sgd > 0.001 else "NEUTRAL"})
  lambda_gauge_Muon   = {lambda_gauge_muon:+.6f}  ({"DECAY" if lambda_gauge_muon < -0.001 else "GROW" if lambda_gauge_muon > 0.001 else "NEUTRAL"})
  lambda_phys_SGD     = {lambda_phys_sgd:+.6f}  ({"DECAY" if lambda_phys_sgd < -0.001 else "GROW" if lambda_phys_sgd > 0.001 else "NEUTRAL"})
  lambda_phys_Muon    = {lambda_phys_muon:+.6f}  ({"DECAY" if lambda_phys_muon < -0.001 else "GROW" if lambda_phys_muon > 0.001 else "NEUTRAL"})
  ---------------------------------------------------------------

  HYPOTHESIS CHECK:
  ---------------------------------------------------------------
  H1: lambda_gauge_Muon < lambda_gauge_SGD
      (Muon stabilizes gauge directions more than SGD)
      Muon: {lambda_gauge_muon:+.6f}  vs  SGD: {lambda_gauge_sgd:+.6f}
      Difference: {lambda_gauge_sgd - lambda_gauge_muon:.6f}
      -> {"CONFIRMED" if lambda_gauge_muon < lambda_gauge_sgd else "REJECTED"}
      {"   (statistically significant)" if is_significant else "   (NOT statistically significant)"}

  H2: lambda_gauge_Muon << 0 (strongly negative)
      -> {"CONFIRMED" if lambda_gauge_muon < -0.01 else "PARTIALLY CONFIRMED" if lambda_gauge_muon < -0.001 else "REJECTED"}

  H3: lambda_gauge_SGD ~ 0 (neutral) or > 0 (unstable)
      -> {"CONFIRMED (neutral)" if abs(lambda_gauge_sgd) < 0.005 else "CONFIRMED (unstable)" if lambda_gauge_sgd > 0.005 else "REJECTED (SGD also decays gauge)"}
  ---------------------------------------------------------------
""")

# Determine overall pass/fail
tests_passed = 0
tests_total = 3

# Test 1: lambda_gauge_Muon < lambda_gauge_SGD
test1_pass = lambda_gauge_muon < lambda_gauge_sgd
if test1_pass:
    tests_passed += 1

# Test 2: lambda_gauge_Muon < 0 (Muon decays gauge perturbations)
test2_pass = lambda_gauge_muon < -0.001
if test2_pass:
    tests_passed += 1

# Test 3: Muon's gauge Lyapunov is more negative than its physical Lyapunov
# (Muon specifically targets gauge directions, not just all directions)
test3_pass = lambda_gauge_muon < lambda_phys_muon - 0.001
if test3_pass:
    tests_passed += 1

if tests_passed == 3:
    overall = "PASS"
    detail = (
        "All three tests pass:\n"
        "  1. Muon's gauge Lyapunov < SGD's gauge Lyapunov (Muon more stable)\n"
        "  2. Muon's gauge Lyapunov < 0 (perturbations decay)\n"
        "  3. Muon's gauge Lyapunov < Muon's physical Lyapunov\n"
        "     (Muon SPECIFICALLY stabilizes gauge directions)\n"
        "\n"
        "  This confirms the dynamical systems prediction: Muon acts as a\n"
        "  gauge-fixing mechanism that exponentially suppresses PSD perturbations."
    )
elif tests_passed >= 2:
    overall = "PARTIAL PASS"
    detail = (
        f"  {tests_passed}/3 tests pass.\n"
        f"  Test 1 (Muon < SGD):          {'PASS' if test1_pass else 'FAIL'}\n"
        f"  Test 2 (Muon gauge < 0):      {'PASS' if test2_pass else 'FAIL'}\n"
        f"  Test 3 (gauge < physical):     {'PASS' if test3_pass else 'FAIL'}\n"
        "\n"
        "  The core prediction is partially supported."
    )
elif tests_passed == 1:
    overall = "WEAK SIGNAL"
    detail = (
        f"  Only {tests_passed}/3 tests pass.\n"
        f"  Test 1 (Muon < SGD):          {'PASS' if test1_pass else 'FAIL'}\n"
        f"  Test 2 (Muon gauge < 0):      {'PASS' if test2_pass else 'FAIL'}\n"
        f"  Test 3 (gauge < physical):     {'PASS' if test3_pass else 'FAIL'}"
    )
else:
    overall = "FAIL"
    detail = (
        "  No tests pass. The Lyapunov exponent predictions are not confirmed.\n"
        f"  Test 1 (Muon < SGD):          {'PASS' if test1_pass else 'FAIL'}\n"
        f"  Test 2 (Muon gauge < 0):      {'PASS' if test2_pass else 'FAIL'}\n"
        f"  Test 3 (gauge < physical):     {'PASS' if test3_pass else 'FAIL'}"
    )

print(f"""
  ╔══════════════════════════════════════════════════════════════════════════╗
  ║  VERDICT: {overall:<63}║
  ╠══════════════════════════════════════════════════════════════════════════╣
  ║                                                                        ║""")
for line in detail.split('\n'):
    print(f"  ║  {line:<70}║")
print(f"""  ║                                                                        ║
  ╚══════════════════════════════════════════════════════════════════════════╝
""")

print("=" * 100)
print(f"  Tests passed: {tests_passed}/{tests_total}")
print(f"  Overall: {overall}")
print("=" * 100)
