#!/usr/bin/env python3
"""
2.10: Anti-Sharpness — Muon finds FLATTER minima despite projecting onto sharp Hessian directions
=================================================================================================

HYPOTHESIS: tr(H_Muon) < 0.5 * tr(H_SGD) at convergence

CONTEXT:
  - 1.3b-ii-A showed Muon projects 11x MORE onto gauge Hessian directions
  - Prediction: aggressive movement THROUGH sharp directions helps ESCAPE them,
    leading to flatter minima at convergence

SETUP:
  - 2-layer deep linear net (4x4, only 32 params — can compute full Hessian)
  - Also 3-layer net (4x4, 48 params) — more gauge-flat directions expected
  - Random target, train to convergence (loss < 0.001 * initial)
  - At convergence compute FULL Hessian via finite differences
  - Repeat 10 times with different seeds, report mean +/- std

MEASUREMENTS:
  - tr(H) = sum of eigenvalues (total curvature)
  - lambda_max (sharpness)
  - Effective Hessian rank: # eigenvalues > 0.01 * lambda_max
  - Gauge-flat count: # eigenvalues < 0.001 * lambda_max
  - Condition number kappa(H) = lambda_max / lambda_min_positive
"""

import numpy as np
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# CONFIGURATION
# =============================================================================

DIM = 4                 # 4x4 matrices => 16 params per layer
NUM_STEPS = 3000        # Max training steps
LR_SGD = 0.01
LR_MUON = 0.02
MOMENTUM = 0.9
NS_ITERS = 5
NUM_SEEDS = 10
CONVERGENCE_FACTOR = 0.001  # Stop when loss < this fraction of initial loss
HESSIAN_EPS = 1e-5          # Finite difference step for Hessian

# =============================================================================
# NETWORK UTILITIES
# =============================================================================

def init_weights(num_layers, dim, seed):
    """Initialize layers near identity."""
    rng = np.random.RandomState(seed)
    weights = []
    for _ in range(num_layers):
        W = np.eye(dim) + rng.randn(dim, dim) * 0.1
        weights.append(W.copy())
    return weights


def forward_linear(weights, X):
    """Forward pass through deep linear net."""
    out = X.copy()
    for W in weights:
        out = W @ out
    return out


def compute_loss(weights, X, Y_target):
    """MSE loss."""
    Y_pred = forward_linear(weights, X)
    diff = Y_pred - Y_target
    return 0.5 * np.mean(diff ** 2)


def compute_gradients(weights, X, Y_target):
    """Backprop through deep linear net."""
    num_layers = len(weights)
    batch_size = X.shape[1]

    # Forward pass — store activations
    activations = [X.copy()]
    for W in weights:
        activations.append(W @ activations[-1])

    # Output error
    Y_pred = activations[-1]
    delta = (Y_pred - Y_target) / batch_size  # shape (dim, batch)

    # Backward pass
    grads = [None] * num_layers
    for l in range(num_layers - 1, -1, -1):
        grads[l] = delta @ activations[l].T  # (dim, dim)
        if l > 0:
            delta = weights[l].T @ delta

    return grads


# =============================================================================
# NEWTON-SCHULZ ITERATION (Muon's core)
# =============================================================================

def newton_schulz_orthogonalize(G, num_iters=NS_ITERS):
    """Apply Newton-Schulz iteration to approximate orthogonal polar factor."""
    # Normalize
    norm = np.linalg.norm(G, ord='fro')
    if norm < 1e-12:
        return G
    X = G / norm

    for _ in range(num_iters):
        A = X @ X.T
        # X <- 1.5*X - 0.5*A@X  (Newton-Schulz iteration for polar decomposition)
        X = 1.5 * X - 0.5 * A @ X

    return X


# =============================================================================
# OPTIMIZERS
# =============================================================================

def train_sgd(weights, X, Y_target, lr, num_steps, convergence_threshold):
    """Train with SGD + momentum."""
    velocities = [np.zeros_like(W) for W in weights]
    initial_loss = compute_loss(weights, X, Y_target)
    target_loss = initial_loss * convergence_threshold

    for step in range(num_steps):
        loss = compute_loss(weights, X, Y_target)
        if loss < target_loss:
            return weights, loss, step, True

        grads = compute_gradients(weights, X, Y_target)
        for i in range(len(weights)):
            velocities[i] = MOMENTUM * velocities[i] + grads[i]
            weights[i] = weights[i] - lr * velocities[i]

    final_loss = compute_loss(weights, X, Y_target)
    return weights, final_loss, num_steps, (final_loss < target_loss)


def train_muon(weights, X, Y_target, lr, num_steps, convergence_threshold):
    """Train with Muon (NS-orthogonalized gradients + momentum)."""
    velocities = [np.zeros_like(W) for W in weights]
    initial_loss = compute_loss(weights, X, Y_target)
    target_loss = initial_loss * convergence_threshold

    for step in range(num_steps):
        loss = compute_loss(weights, X, Y_target)
        if loss < target_loss:
            return weights, loss, step, True

        grads = compute_gradients(weights, X, Y_target)
        for i in range(len(weights)):
            # Newton-Schulz orthogonalization of gradient
            G_orth = newton_schulz_orthogonalize(grads[i])
            velocities[i] = MOMENTUM * velocities[i] + G_orth
            weights[i] = weights[i] - lr * velocities[i]

    final_loss = compute_loss(weights, X, Y_target)
    return weights, final_loss, num_steps, (final_loss < target_loss)


# =============================================================================
# FULL HESSIAN COMPUTATION VIA FINITE DIFFERENCES
# =============================================================================

def weights_to_vector(weights):
    """Flatten all weight matrices into a single vector."""
    return np.concatenate([W.flatten() for W in weights])


def vector_to_weights(vec, shapes):
    """Unflatten a vector back into list of weight matrices."""
    weights = []
    idx = 0
    for shape in shapes:
        size = shape[0] * shape[1]
        W = vec[idx:idx + size].reshape(shape)
        weights.append(W)
        idx += size
    return weights


def compute_gradient_vector(weights, X, Y_target):
    """Return gradient as a flat vector."""
    grads = compute_gradients(weights, X, Y_target)
    return np.concatenate([g.flatten() for g in grads])


def compute_full_hessian(weights, X, Y_target, eps=HESSIAN_EPS):
    """Compute full Hessian via central finite differences on the gradient."""
    shapes = [W.shape for W in weights]
    theta = weights_to_vector(weights)
    n_params = len(theta)

    H = np.zeros((n_params, n_params))

    for i in range(n_params):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[i] += eps
        theta_minus[i] -= eps

        w_plus = vector_to_weights(theta_plus, shapes)
        w_minus = vector_to_weights(theta_minus, shapes)

        grad_plus = compute_gradient_vector(w_plus, X, Y_target)
        grad_minus = compute_gradient_vector(w_minus, X, Y_target)

        H[:, i] = (grad_plus - grad_minus) / (2 * eps)

    # Symmetrize
    H = 0.5 * (H + H.T)
    return H


# =============================================================================
# HESSIAN ANALYSIS
# =============================================================================

def analyze_hessian(H):
    """Compute all Hessian statistics."""
    eigenvalues = np.linalg.eigvalsh(H)
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]  # descending

    lambda_max = eigenvalues_sorted[0]
    trace_H = np.sum(eigenvalues)

    # Effective rank: eigenvalues > 0.01 * lambda_max
    if lambda_max > 1e-15:
        eff_rank = np.sum(np.abs(eigenvalues) > 0.01 * lambda_max)
        gauge_flat = np.sum(np.abs(eigenvalues) < 0.001 * lambda_max)
    else:
        eff_rank = 0
        gauge_flat = len(eigenvalues)

    # Condition number: lambda_max / lambda_min_positive
    positive_eigs = eigenvalues[eigenvalues > 1e-12]
    if len(positive_eigs) > 0:
        kappa = lambda_max / np.min(positive_eigs)
    else:
        kappa = np.inf

    # Also track negative eigenvalue count and magnitude
    neg_eigs = eigenvalues[eigenvalues < -1e-12]
    n_negative = len(neg_eigs)
    sum_negative = np.sum(neg_eigs) if n_negative > 0 else 0.0

    return {
        'trace': trace_H,
        'lambda_max': lambda_max,
        'eff_rank': eff_rank,
        'gauge_flat': gauge_flat,
        'kappa': kappa,
        'n_negative': n_negative,
        'sum_negative': sum_negative,
        'eigenvalues': eigenvalues_sorted,
    }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_single_experiment(num_layers, dim, seed):
    """Run one comparison between SGD and Muon for a given architecture and seed."""
    # Generate random target and data
    rng = np.random.RandomState(seed)
    W_target_list = [rng.randn(dim, dim) * 0.3 for _ in range(num_layers)]
    X = rng.randn(dim, 32) * 0.5  # 32 data points

    # Compute target output
    Y_target = X.copy()
    for W in W_target_list:
        Y_target = W @ Y_target

    # Train SGD
    w_sgd = init_weights(num_layers, dim, seed + 1000)
    w_sgd, loss_sgd, steps_sgd, conv_sgd = train_sgd(
        w_sgd, X, Y_target, LR_SGD, NUM_STEPS, CONVERGENCE_FACTOR)

    # Train Muon (start from SAME initialization)
    w_muon = init_weights(num_layers, dim, seed + 1000)
    w_muon, loss_muon, steps_muon, conv_muon = train_muon(
        w_muon, X, Y_target, LR_MUON, NUM_STEPS, CONVERGENCE_FACTOR)

    # Compute full Hessian at convergence
    H_sgd = compute_full_hessian(w_sgd, X, Y_target)
    H_muon = compute_full_hessian(w_muon, X, Y_target)

    stats_sgd = analyze_hessian(H_sgd)
    stats_muon = analyze_hessian(H_muon)

    return {
        'sgd': {**stats_sgd, 'loss': loss_sgd, 'steps': steps_sgd, 'converged': conv_sgd},
        'muon': {**stats_muon, 'loss': loss_muon, 'steps': steps_muon, 'converged': conv_muon},
    }


def print_separator(char='=', width=100):
    print(char * width)


def run_experiment_suite(num_layers, dim, label):
    """Run full experiment suite for a given architecture."""
    n_params = num_layers * dim * dim

    print_separator()
    print(f"  {label}: {num_layers}-LAYER DEEP LINEAR NET ({dim}x{dim}, {n_params} params)")
    print_separator()

    all_results = []
    for i in range(NUM_SEEDS):
        seed = 42 + i * 137
        result = run_single_experiment(num_layers, dim, seed)
        all_results.append(result)
        sgd_c = "YES" if result['sgd']['converged'] else "NO"
        muon_c = "YES" if result['muon']['converged'] else "NO"
        print(f"  Seed {i+1:2d}/{NUM_SEEDS}: SGD conv={sgd_c} (loss={result['sgd']['loss']:.2e}, steps={result['sgd']['steps']})"
              f"  |  Muon conv={muon_c} (loss={result['muon']['loss']:.2e}, steps={result['muon']['steps']})")

    # Aggregate statistics
    metrics = ['trace', 'lambda_max', 'eff_rank', 'gauge_flat', 'kappa']
    metric_labels = {
        'trace': 'tr(H)',
        'lambda_max': 'lambda_max',
        'eff_rank': 'Eff. Hessian rank',
        'gauge_flat': 'Gauge-flat count',
        'kappa': 'Condition number kappa',
    }

    sgd_stats = {m: [] for m in metrics}
    muon_stats = {m: [] for m in metrics}

    for r in all_results:
        for m in metrics:
            sgd_stats[m].append(r['sgd'][m])
            muon_stats[m].append(r['muon'][m])

    # Also gather convergence info
    sgd_losses = [r['sgd']['loss'] for r in all_results]
    muon_losses = [r['muon']['loss'] for r in all_results]
    sgd_conv_count = sum(1 for r in all_results if r['sgd']['converged'])
    muon_conv_count = sum(1 for r in all_results if r['muon']['converged'])

    print()
    print_separator('-')
    print(f"  CONVERGENCE SUMMARY:")
    print(f"    SGD converged:  {sgd_conv_count}/{NUM_SEEDS}  (mean loss = {np.mean(sgd_losses):.2e} +/- {np.std(sgd_losses):.2e})")
    print(f"    Muon converged: {muon_conv_count}/{NUM_SEEDS}  (mean loss = {np.mean(muon_losses):.2e} +/- {np.std(muon_losses):.2e})")
    print_separator('-')
    print()

    # Print comparison table
    print(f"  {'Metric':<28s} | {'SGD (mean +/- std)':<28s} | {'Muon (mean +/- std)':<28s} | {'Ratio Muon/SGD':<16s}")
    print(f"  {'-'*28}-+-{'-'*28}-+-{'-'*28}-+-{'-'*16}")

    ratios = {}
    for m in metrics:
        sgd_mean = np.mean(sgd_stats[m])
        sgd_std = np.std(sgd_stats[m])
        muon_mean = np.mean(muon_stats[m])
        muon_std = np.std(muon_stats[m])

        if abs(sgd_mean) > 1e-15:
            ratio = muon_mean / sgd_mean
            ratio_str = f"{ratio:.4f}"
        else:
            ratio = np.inf
            ratio_str = "N/A"

        ratios[m] = ratio
        label_str = metric_labels[m]

        # Format numbers appropriately
        if m in ['eff_rank', 'gauge_flat']:
            sgd_str = f"{sgd_mean:8.1f} +/- {sgd_std:6.1f}"
            muon_str = f"{muon_mean:8.1f} +/- {muon_std:6.1f}"
        elif m == 'kappa':
            sgd_str = f"{sgd_mean:10.1f} +/- {sgd_std:8.1f}"
            muon_str = f"{muon_mean:10.1f} +/- {muon_std:8.1f}"
        else:
            sgd_str = f"{sgd_mean:12.4e} +/- {sgd_std:.2e}"
            muon_str = f"{muon_mean:12.4e} +/- {muon_std:.2e}"

        print(f"  {label_str:<28s} | {sgd_str:<28s} | {muon_str:<28s} | {ratio_str:<16s}")

    print()

    # KEY HYPOTHESIS TEST
    print_separator('*')
    trace_ratio = ratios['trace']
    print(f"  KEY HYPOTHESIS TEST: tr(H_Muon) < 0.5 * tr(H_SGD)")
    print(f"  Ratio tr(H_Muon)/tr(H_SGD) = {trace_ratio:.4f}")
    if trace_ratio < 0.5:
        print(f"  >>> HYPOTHESIS CONFIRMED: Muon finds significantly flatter minima (ratio = {trace_ratio:.4f} < 0.5)")
    elif trace_ratio < 1.0:
        print(f"  >>> PARTIAL SUPPORT: Muon finds flatter minima (ratio = {trace_ratio:.4f} < 1.0) but not 2x flatter")
    else:
        print(f"  >>> HYPOTHESIS REJECTED: Muon does NOT find flatter minima (ratio = {trace_ratio:.4f} >= 1.0)")
    print_separator('*')
    print()

    # Additional analysis: eigenvalue spectrum comparison
    print(f"  EIGENVALUE SPECTRUM COMPARISON (averaged over {NUM_SEEDS} seeds):")
    print(f"  {'Bin':<30s} | {'SGD count':<14s} | {'Muon count':<14s}")
    print(f"  {'-'*30}-+-{'-'*14}-+-{'-'*14}")

    # Compute averaged eigenvalue distributions
    sgd_all_eigs = np.array([r['sgd']['eigenvalues'] for r in all_results])
    muon_all_eigs = np.array([r['muon']['eigenvalues'] for r in all_results])

    # Bins for eigenvalue magnitudes
    bins = [
        ('|eig| > 1.0', lambda e: np.sum(np.abs(e) > 1.0)),
        ('0.1 < |eig| <= 1.0', lambda e: np.sum((np.abs(e) > 0.1) & (np.abs(e) <= 1.0))),
        ('0.01 < |eig| <= 0.1', lambda e: np.sum((np.abs(e) > 0.01) & (np.abs(e) <= 0.1))),
        ('0.001 < |eig| <= 0.01', lambda e: np.sum((np.abs(e) > 0.001) & (np.abs(e) <= 0.01))),
        ('|eig| <= 0.001', lambda e: np.sum(np.abs(e) <= 0.001)),
        ('eig < 0 (negative)', lambda e: np.sum(e < -1e-12)),
    ]

    for bin_label, bin_fn in bins:
        sgd_counts = [bin_fn(sgd_all_eigs[i]) for i in range(NUM_SEEDS)]
        muon_counts = [bin_fn(muon_all_eigs[i]) for i in range(NUM_SEEDS)]
        print(f"  {bin_label:<30s} | {np.mean(sgd_counts):6.1f} +/- {np.std(sgd_counts):4.1f} | {np.mean(muon_counts):6.1f} +/- {np.std(muon_counts):4.1f}")

    print()

    return all_results, ratios


def main():
    print()
    print_separator('#')
    print("  EXPERIMENT 2.10: ANTI-SHARPNESS — Muon finds FLATTER minima")
    print("  Despite projecting onto sharp Hessian directions")
    print_separator('#')
    print()
    print(f"  Config: {NUM_SEEDS} seeds, max {NUM_STEPS} steps, convergence threshold = {CONVERGENCE_FACTOR}")
    print(f"  SGD lr = {LR_SGD}, Muon lr = {LR_MUON}, momentum = {MOMENTUM}")
    print(f"  Hessian computed via central finite differences (eps = {HESSIAN_EPS})")
    print()

    # ---- Part A: 2-layer net ----
    results_2layer, ratios_2layer = run_experiment_suite(
        num_layers=2, dim=DIM, label="PART A")

    # ---- Part B: 3-layer net ----
    results_3layer, ratios_3layer = run_experiment_suite(
        num_layers=3, dim=DIM, label="PART B")

    # ---- Final Summary ----
    print_separator('#')
    print("  FINAL COMPARATIVE SUMMARY")
    print_separator('#')
    print()
    print(f"  {'Metric':<28s} | {'2-layer Muon/SGD':<20s} | {'3-layer Muon/SGD':<20s}")
    print(f"  {'-'*28}-+-{'-'*20}-+-{'-'*20}")

    metric_labels = {
        'trace': 'tr(H) ratio',
        'lambda_max': 'lambda_max ratio',
        'eff_rank': 'Eff. rank ratio',
        'gauge_flat': 'Gauge-flat ratio',
        'kappa': 'Condition # ratio',
    }

    for m in ['trace', 'lambda_max', 'eff_rank', 'gauge_flat', 'kappa']:
        r2 = ratios_2layer[m]
        r3 = ratios_3layer[m]
        r2s = f"{r2:.4f}" if r2 != np.inf else "N/A"
        r3s = f"{r3:.4f}" if r3 != np.inf else "N/A"
        print(f"  {metric_labels[m]:<28s} | {r2s:<20s} | {r3s:<20s}")

    print()

    # Overall verdict
    t2 = ratios_2layer['trace']
    t3 = ratios_3layer['trace']
    print_separator('*')
    print("  OVERALL VERDICT:")
    print(f"    2-layer: tr(H_Muon)/tr(H_SGD) = {t2:.4f}  {'< 0.5 CONFIRMED' if t2 < 0.5 else '< 1.0 PARTIAL' if t2 < 1.0 else '>= 1.0 REJECTED'}")
    print(f"    3-layer: tr(H_Muon)/tr(H_SGD) = {t3:.4f}  {'< 0.5 CONFIRMED' if t3 < 0.5 else '< 1.0 PARTIAL' if t3 < 1.0 else '>= 1.0 REJECTED'}")
    print()
    if t3 < t2:
        print("    Deeper net shows STRONGER anti-sharpness effect (as predicted: more gauge directions)")
    else:
        print("    Deeper net does NOT show stronger anti-sharpness effect")
    print_separator('*')
    print()


if __name__ == '__main__':
    main()
