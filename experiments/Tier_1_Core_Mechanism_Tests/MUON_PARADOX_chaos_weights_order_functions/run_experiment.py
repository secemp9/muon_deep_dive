#!/usr/bin/env python3
"""
THE MUON PARADOX: Chaotic in Weight Space, Contractive in Function Space
=========================================================================

UNIFYING TEST for the entire research project.

Known results:
  - 1.2b-i: Muon has HIGHER Lyapunov exponent in weight space
            (lambda_weight ~ +0.013 vs +0.002 for SGD)
  - C5vsA1: Muon gives DIVERSE weights but CONSISTENT losses

The paradox has TWO faces:

  FACE 1 (Perturbation Lyapunov):
    Muon amplifies weight perturbations (higher lambda_weight) but the
    RATIO of function-space divergence to weight-space divergence is lower,
    meaning Muon's weight exploration is preferentially in gauge directions.

  FACE 2 (Convergence Basin):
    Run many independent initializations to similar loss levels.
    Muon reaches DIVERSE weights but CONSISTENT functions/losses.
    SGD reaches SIMILAR weights but has MORE function variance.

Both faces arise from the same mechanism: Newton-Schulz orthogonalization
pushes weight exploration into gauge (function-invariant) directions.

Setup:
  - 4-layer deep linear net (32x32) + 4-layer ReLU net (32x32)
  - Quadratic loss
  - 20 random perturbation directions, 200 training steps
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
NUM_TEST_INPUTS = 50
NUM_INDEPENDENT_RUNS = 20  # For convergence basin analysis

# Fixed target and data
W_target = np.random.randn(DIM, DIM) * 0.5
X_data = np.random.randn(DIM, BATCH_SIZE) * 0.3
X_test = np.random.randn(DIM, NUM_TEST_INPUTS) * 0.3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# NETWORK DEFINITIONS
# =============================================================================

def init_weights(num_layers, seed=42):
    """Initialize layers near identity for stability."""
    rng = np.random.RandomState(seed)
    weights = []
    for _ in range(num_layers):
        W = np.eye(DIM) + rng.randn(DIM, DIM) * 0.1
        weights.append(W.copy())
    return weights


# ---- DEEP LINEAR NET ----

def forward_linear(weights, X):
    out = X.copy()
    for W in weights:
        out = W @ out
    return out


def compute_loss_linear(weights, X, target):
    pred = forward_linear(weights, X)
    target_out = target @ X
    diff = pred - target_out
    return 0.5 * np.mean(np.sum(diff ** 2, axis=0))


def compute_gradients_linear(weights, X, target):
    num_layers = len(weights)
    N = X.shape[1]
    activations = [X.copy()]
    out = X.copy()
    for W in weights:
        out = W @ out
        activations.append(out.copy())
    target_out = target @ X
    delta = (activations[-1] - target_out) / N
    grads = []
    for i in range(num_layers - 1, -1, -1):
        G = delta @ activations[i].T
        grads.insert(0, G)
        if i > 0:
            delta = weights[i].T @ delta
    return grads


# ---- RELU NET ----

def forward_relu(weights, X):
    out = X.copy()
    for idx, W in enumerate(weights):
        out = W @ out
        if idx < len(weights) - 1:
            out = np.maximum(0, out)
    return out


def compute_loss_relu(weights, X, target):
    pred = forward_relu(weights, X)
    target_out = target @ X
    diff = pred - target_out
    return 0.5 * np.mean(np.sum(diff ** 2, axis=0))


def compute_gradients_relu(weights, X, target):
    num_layers = len(weights)
    N = X.shape[1]
    pre_activations = []
    post_activations = [X.copy()]
    out = X.copy()
    for idx, W in enumerate(weights):
        pre = W @ out
        pre_activations.append(pre.copy())
        if idx < num_layers - 1:
            out = np.maximum(0, pre)
        else:
            out = pre
        post_activations.append(out.copy())
    target_out = target @ X
    delta = (post_activations[-1] - target_out) / N
    grads = []
    for i in range(num_layers - 1, -1, -1):
        G = delta @ post_activations[i].T
        grads.insert(0, G)
        if i > 0:
            delta = weights[i].T @ delta
            delta = delta * (pre_activations[i - 1] > 0).astype(float)
    return grads


# =============================================================================
# MUON CORE: NEWTON-SCHULZ ORTHOGONALIZATION
# =============================================================================

def newton_schulz_orthogonalize(G, num_iters=NS_ITERS):
    norm = np.linalg.norm(G, ord='fro')
    if norm < 1e-12:
        return G
    X = G / norm
    for _ in range(num_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


# =============================================================================
# OPTIMIZER STEP FUNCTIONS
# =============================================================================

def sgd_step(weights, velocities, grads, lr):
    for i in range(len(weights)):
        velocities[i] = MOMENTUM * velocities[i] + grads[i]
        weights[i] = weights[i] - lr * velocities[i]
    return weights, velocities


def muon_step(weights, velocities, grads, lr):
    for i in range(len(weights)):
        ortho_grad = newton_schulz_orthogonalize(grads[i])
        velocities[i] = MOMENTUM * velocities[i] + ortho_grad
        weights[i] = weights[i] - lr * velocities[i]
    return weights, velocities


# =============================================================================
# LEARNING RATE FINDER
# =============================================================================

def find_stable_lr_sgd(net_type):
    compute_loss_fn = compute_loss_linear if net_type == 'linear' else compute_loss_relu
    compute_grad_fn = compute_gradients_linear if net_type == 'linear' else compute_gradients_relu
    candidates = [0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.001]
    for lr in candidates:
        np.random.seed(42)
        weights = init_weights(NUM_LAYERS)
        velocities = [np.zeros((DIM, DIM)) for _ in range(NUM_LAYERS)]
        initial_loss = compute_loss_fn(weights, X_data, W_target)
        stable = True
        for step in range(80):
            grads = compute_grad_fn(weights, X_data, W_target)
            weights, velocities = sgd_step(weights, velocities, grads, lr)
            loss = compute_loss_fn(weights, X_data, W_target)
            if np.isnan(loss) or loss > initial_loss * 50:
                stable = False
                break
        if stable:
            return lr
    return 0.001


# =============================================================================
# TRAJECTORY ENGINE
# =============================================================================

def run_trajectory(weights_init, optimizer, lr, num_steps, net_type):
    """
    Run optimizer for num_steps from weights_init.
    Returns weight_snapshots, function_outputs (on X_test), losses.
    """
    compute_loss_fn = compute_loss_linear if net_type == 'linear' else compute_loss_relu
    compute_grad_fn = compute_gradients_linear if net_type == 'linear' else compute_gradients_relu
    forward_fn = forward_linear if net_type == 'linear' else forward_relu

    weights = [w.copy() for w in weights_init]
    velocities = [np.zeros_like(w) for w in weights]

    weight_snapshots = [[w.copy() for w in weights]]
    function_outputs = [forward_fn(weights, X_test).copy()]
    losses = [compute_loss_fn(weights, X_data, W_target)]

    for step in range(num_steps):
        grads = compute_grad_fn(weights, X_data, W_target)
        if optimizer == 'sgd':
            weights, velocities = sgd_step(weights, velocities, grads, lr)
        elif optimizer == 'muon':
            weights, velocities = muon_step(weights, velocities, grads, lr)

        weight_snapshots.append([w.copy() for w in weights])
        function_outputs.append(forward_fn(weights, X_test).copy())
        loss = compute_loss_fn(weights, X_data, W_target)
        losses.append(loss)

        if np.isnan(loss) or loss > 1e10:
            for _ in range(num_steps - step - 1):
                weight_snapshots.append([w.copy() for w in weights])
                function_outputs.append(forward_fn(weights, X_test).copy())
                losses.append(loss)
            break

    return weight_snapshots, function_outputs, np.array(losses)


# =============================================================================
# DIVERGENCE MEASUREMENT
# =============================================================================

def compute_weight_divergence(snap_a, snap_b):
    T = min(len(snap_a), len(snap_b))
    distances = np.zeros(T)
    for t in range(T):
        d_sq = 0.0
        for i in range(len(snap_a[t])):
            d_sq += np.linalg.norm(snap_a[t][i] - snap_b[t][i], 'fro') ** 2
        distances[t] = np.sqrt(d_sq)
    return distances


def compute_function_divergence(func_a, func_b):
    T = min(len(func_a), len(func_b))
    x_norm = np.linalg.norm(X_test, 'fro')
    distances = np.zeros(T)
    for t in range(T):
        distances[t] = np.linalg.norm(func_a[t] - func_b[t], 'fro') / x_norm
    return distances


def compute_loss_divergence(loss_a, loss_b):
    T = min(len(loss_a), len(loss_b))
    return np.abs(loss_a[:T] - loss_b[:T])


def compute_lyapunov(d_series, N):
    d0 = d_series[0]
    dN = d_series[min(N, len(d_series) - 1)]
    if d0 > 1e-15 and dN > 1e-15:
        return (1.0 / N) * np.log(dN / d0)
    elif dN < 1e-15:
        return -np.inf
    else:
        return np.nan


# =============================================================================
# FACE 1: PERTURBATION LYAPUNOV ANALYSIS
# =============================================================================

def measure_perturbation_lyapunov(net_type, lr_sgd, lr_muon, num_pert, seed_base=100):
    """
    Measure Lyapunov exponents from initial-condition perturbations.
    Tracks weight-space, function-space, and loss-space divergence.
    """
    print(f"\n  [FACE 1] Perturbation Lyapunov for {net_type.upper()} net")

    np.random.seed(42)
    weights_base = init_weights(NUM_LAYERS)

    # Run base trajectories for each optimizer
    sgd_base_snap, sgd_base_func, sgd_base_loss = run_trajectory(
        weights_base, 'sgd', lr_sgd, NUM_STEPS, net_type)
    muon_base_snap, muon_base_func, muon_base_loss = run_trajectory(
        weights_base, 'muon', lr_muon, NUM_STEPS, net_type)

    print(f"    SGD  final loss: {sgd_base_loss[-1]:.6e}")
    print(f"    Muon final loss: {muon_base_loss[-1]:.6e}")

    results = {
        'sgd': {'lyap_w': [], 'lyap_f': [], 'lyap_l': [],
                'd_w': [], 'd_f': [], 'd_l': []},
        'muon': {'lyap_w': [], 'lyap_f': [], 'lyap_l': [],
                 'd_w': [], 'd_f': [], 'd_l': []},
    }

    for p in range(num_pert):
        rng = np.random.RandomState(seed_base + p)

        # Perturbation: W0' = W0 + epsilon * delta_W (unit-norm random)
        weights_pert = []
        for layer_idx in range(NUM_LAYERS):
            delta_W = rng.randn(DIM, DIM)
            delta_W = delta_W / np.linalg.norm(delta_W, 'fro')
            weights_pert.append(weights_base[layer_idx] + EPSILON * delta_W)

        for opt, lr, base_s, base_f, base_l in [
            ('sgd', lr_sgd, sgd_base_snap, sgd_base_func, sgd_base_loss),
            ('muon', lr_muon, muon_base_snap, muon_base_func, muon_base_loss),
        ]:
            p_snap, p_func, p_loss = run_trajectory(
                weights_pert, opt, lr, NUM_STEPS, net_type)

            d_w = compute_weight_divergence(base_s, p_snap)
            d_f = compute_function_divergence(base_f, p_func)
            d_l = compute_loss_divergence(base_l, p_loss)

            results[opt]['lyap_w'].append(compute_lyapunov(d_w, NUM_STEPS))
            results[opt]['lyap_f'].append(compute_lyapunov(d_f, NUM_STEPS))
            results[opt]['lyap_l'].append(compute_lyapunov(d_l, NUM_STEPS))
            results[opt]['d_w'].append(d_w)
            results[opt]['d_f'].append(d_f)
            results[opt]['d_l'].append(d_l)

        if (p + 1) % 5 == 0:
            print(f"    Completed {p+1}/{num_pert} perturbations", flush=True)

    # Compute statistics
    stats = {}
    for opt in ['sgd', 'muon']:
        for metric in ['lyap_w', 'lyap_f', 'lyap_l']:
            arr = np.array(results[opt][metric])
            valid = arr[np.isfinite(arr)]
            stats[f"{opt}_{metric}_all"] = arr
            stats[f"{opt}_{metric}_mean"] = np.mean(valid) if len(valid) > 0 else np.nan
            stats[f"{opt}_{metric}_std"] = np.std(valid) if len(valid) > 0 else np.nan
        for metric in ['d_w', 'd_f', 'd_l']:
            stats[f"{opt}_{metric}_all"] = results[opt][metric]
            stats[f"{opt}_{metric}_mean_traj"] = np.mean(results[opt][metric], axis=0)

    # Compute the RATIO: function-divergence / weight-divergence at each timestep
    # This is the key diagnostic: for Muon, function divergence should grow
    # SLOWER than weight divergence.
    for opt in ['sgd', 'muon']:
        ratios_over_time = []
        for p in range(num_pert):
            d_w = results[opt]['d_w'][p]
            d_f = results[opt]['d_f'][p]
            # ratio = d_f(t) / d_w(t) -- how much of weight divergence maps to function divergence
            ratio = np.zeros_like(d_w)
            for t in range(len(d_w)):
                if d_w[t] > 1e-15:
                    ratio[t] = d_f[t] / d_w[t]
                else:
                    ratio[t] = np.nan
            ratios_over_time.append(ratio)
        stats[f"{opt}_ratio_f_w_all"] = ratios_over_time
        # Mean ratio at final time
        final_ratios = [r[-1] for r in ratios_over_time if np.isfinite(r[-1])]
        stats[f"{opt}_ratio_f_w_final_mean"] = np.mean(final_ratios) if final_ratios else np.nan
        stats[f"{opt}_ratio_f_w_final_std"] = np.std(final_ratios) if final_ratios else np.nan
        # Mean ratio trajectory
        ratio_arr = np.array(ratios_over_time)
        stats[f"{opt}_ratio_f_w_mean_traj"] = np.nanmean(ratio_arr, axis=0)

    return stats


# =============================================================================
# FACE 2: CONVERGENCE BASIN ANALYSIS
# =============================================================================

def measure_convergence_basin(net_type, lr_sgd, lr_muon, num_runs, num_steps=500):
    """
    Run many independent initializations with each optimizer.
    At the end, measure:
      - Weight diversity: mean pairwise ||W_i - W_j||_F
      - Function diversity: mean pairwise ||f(X_test; W_i) - f(X_test; W_j)||_F
      - Loss diversity: std of final losses

    The paradox: Muon weight diversity > SGD weight diversity
                 Muon function diversity < SGD function diversity
    """
    print(f"\n  [FACE 2] Convergence Basin for {net_type.upper()} net ({num_runs} runs, {num_steps} steps)")

    forward_fn = forward_linear if net_type == 'linear' else forward_relu
    compute_loss_fn = compute_loss_linear if net_type == 'linear' else compute_loss_relu

    results = {}

    for opt_name, opt_type, lr in [('sgd', 'sgd', lr_sgd), ('muon', 'muon', lr_muon)]:
        final_weights_list = []
        final_functions = []
        final_losses = []

        for run_idx in range(num_runs):
            # Different random init for each run
            weights_init = init_weights(NUM_LAYERS, seed=1000 + run_idx)
            _, _, loss_traj = run_trajectory(
                weights_init, opt_type, lr, num_steps, net_type)

            # Re-run to get final weights (less memory than storing snapshots)
            compute_grad_fn = compute_gradients_linear if net_type == 'linear' else compute_gradients_relu
            weights = [w.copy() for w in init_weights(NUM_LAYERS, seed=1000 + run_idx)]
            velocities = [np.zeros_like(w) for w in weights]
            for step in range(num_steps):
                grads = compute_grad_fn(weights, X_data, W_target)
                if opt_type == 'sgd':
                    weights, velocities = sgd_step(weights, velocities, grads, lr)
                else:
                    weights, velocities = muon_step(weights, velocities, grads, lr)
                loss = compute_loss_fn(weights, X_data, W_target)
                if np.isnan(loss) or loss > 1e10:
                    break

            final_weights_list.append([w.copy() for w in weights])
            final_functions.append(forward_fn(weights, X_test).copy())
            final_losses.append(compute_loss_fn(weights, X_data, W_target))

        # Compute pairwise diversity
        n = len(final_weights_list)
        weight_dists = []
        func_dists = []
        for i in range(n):
            for j in range(i + 1, n):
                # Weight distance
                d_w = 0.0
                for k in range(NUM_LAYERS):
                    d_w += np.linalg.norm(final_weights_list[i][k] - final_weights_list[j][k], 'fro') ** 2
                weight_dists.append(np.sqrt(d_w))
                # Function distance
                d_f = np.linalg.norm(final_functions[i] - final_functions[j], 'fro') / np.linalg.norm(X_test, 'fro')
                func_dists.append(d_f)

        results[opt_name] = {
            'weight_diversity_mean': np.mean(weight_dists),
            'weight_diversity_std': np.std(weight_dists),
            'func_diversity_mean': np.mean(func_dists),
            'func_diversity_std': np.std(func_dists),
            'loss_mean': np.mean(final_losses),
            'loss_std': np.std(final_losses),
            'losses': np.array(final_losses),
            'weight_dists': np.array(weight_dists),
            'func_dists': np.array(func_dists),
        }

        print(f"    {opt_name.upper()}: loss={np.mean(final_losses):.6e} +/- {np.std(final_losses):.6e}, "
              f"d_weight={np.mean(weight_dists):.4f}, d_func={np.mean(func_dists):.6f}")

    return results


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

print("=" * 100)
print("THE MUON PARADOX: Chaotic in Weight Space, Contractive in Function Space")
print("=" * 100)
print(f"Setup: {NUM_LAYERS}-layer nets (dim={DIM}), quadratic loss")
print(f"Face 1: eps={EPSILON}, {NUM_PERTURBATIONS} perturbations, {NUM_STEPS} steps")
print(f"Face 2: {NUM_INDEPENDENT_RUNS} independent inits, 500 steps")
print(f"LR_Muon={LR_MUON}, Momentum={MOMENTUM}, NS_iters={NS_ITERS}")
print("=" * 100)

all_face1 = {}
all_face2 = {}

for net_type in ['linear', 'relu']:
    print(f"\n{'#' * 80}")
    print(f"  NETWORK TYPE: {net_type.upper()}")
    print(f"{'#' * 80}")

    lr_sgd = find_stable_lr_sgd(net_type)
    print(f"  SGD lr (max stable): {lr_sgd}")

    # Face 1: Perturbation Lyapunov
    face1 = measure_perturbation_lyapunov(net_type, lr_sgd, LR_MUON, NUM_PERTURBATIONS)
    all_face1[net_type] = face1
    all_face1[f"{net_type}_lr_sgd"] = lr_sgd

    # Face 2: Convergence Basin
    face2 = measure_convergence_basin(net_type, lr_sgd, LR_MUON, NUM_INDEPENDENT_RUNS)
    all_face2[net_type] = face2


# =============================================================================
# FACE 1 RESULTS TABLES
# =============================================================================

for net_type in ['linear', 'relu']:
    f1 = all_face1[net_type]
    lr_sgd = all_face1[f"{net_type}_lr_sgd"]

    print(f"\n\n{'=' * 100}")
    print(f"FACE 1: PERTURBATION LYAPUNOV -- {net_type.upper()} NET  (lr_sgd={lr_sgd}, lr_muon={LR_MUON})")
    print(f"{'=' * 100}")

    print(f"\n  {'Metric':<30} | {'SGD':>14} | {'Muon':>14} | {'Direction':>10}")
    print(f"  {'-' * 76}")

    lw_s = f1['sgd_lyap_w_mean']
    lw_m = f1['muon_lyap_w_mean']
    lf_s = f1['sgd_lyap_f_mean']
    lf_m = f1['muon_lyap_f_mean']
    ll_s = f1['sgd_lyap_l_mean']
    ll_m = f1['muon_lyap_l_mean']

    print(f"  {'lambda_weight':<30} | {lw_s:>+14.6f} | {lw_m:>+14.6f} | {'Muon>SGD' if lw_m > lw_s else 'SGD>Muon':>10}")
    print(f"  {'lambda_function':<30} | {lf_s:>+14.6f} | {lf_m:>+14.6f} | {'Muon<SGD' if lf_m < lf_s else 'Muon>SGD':>10}")
    print(f"  {'lambda_loss':<30} | {ll_s:>+14.6f} | {ll_m:>+14.6f} | {'Muon<SGD' if ll_m < ll_s else 'Muon>SGD':>10}")

    # The key ratio: d_func(T) / d_weight(T) at final time
    rf_s = f1['sgd_ratio_f_w_final_mean']
    rf_m = f1['muon_ratio_f_w_final_mean']
    print(f"  {'d_func/d_weight at T=200':<30} | {rf_s:>14.6f} | {rf_m:>14.6f} | {'Muon<SGD' if rf_m < rf_s else 'Muon>SGD':>10}")

    print(f"\n  Standard deviations:")
    print(f"  {'lambda_weight std':<30} | {f1['sgd_lyap_w_std']:>14.6f} | {f1['muon_lyap_w_std']:>14.6f}")
    print(f"  {'lambda_function std':<30} | {f1['sgd_lyap_f_std']:>14.6f} | {f1['muon_lyap_f_std']:>14.6f}")
    print(f"  {'lambda_loss std':<30} | {f1['sgd_lyap_l_std']:>14.6f} | {f1['muon_lyap_l_std']:>14.6f}")
    print(f"  {'d_func/d_weight std':<30} | {f1['sgd_ratio_f_w_final_std']:>14.6f} | {f1['muon_ratio_f_w_final_std']:>14.6f}")


# =============================================================================
# FACE 2 RESULTS TABLES
# =============================================================================

for net_type in ['linear', 'relu']:
    f2 = all_face2[net_type]

    print(f"\n\n{'=' * 100}")
    print(f"FACE 2: CONVERGENCE BASIN -- {net_type.upper()} NET")
    print(f"{'=' * 100}")

    print(f"\n  {'Metric':<35} | {'SGD':>14} | {'Muon':>14} | {'Paradox?':>10}")
    print(f"  {'-' * 80}")

    wd_s = f2['sgd']['weight_diversity_mean']
    wd_m = f2['muon']['weight_diversity_mean']
    fd_s = f2['sgd']['func_diversity_mean']
    fd_m = f2['muon']['func_diversity_mean']
    ls_s = f2['sgd']['loss_std']
    ls_m = f2['muon']['loss_std']
    lm_s = f2['sgd']['loss_mean']
    lm_m = f2['muon']['loss_mean']

    # Weight diversity: Muon > SGD (more diverse weights)
    p_wd = "YES" if wd_m > wd_s else "no"
    print(f"  {'Weight diversity (pairwise)':<35} | {wd_s:>14.6f} | {wd_m:>14.6f} | {p_wd:>10}")

    # Function diversity: Muon < SGD (less diverse functions)
    p_fd = "YES" if fd_m < fd_s else "no"
    print(f"  {'Function diversity (pairwise)':<35} | {fd_s:>14.6f} | {fd_m:>14.6f} | {p_fd:>10}")

    # Loss spread: Muon < SGD (more consistent)
    p_ls = "YES" if ls_m < ls_s else "no"
    print(f"  {'Loss std across runs':<35} | {ls_s:>14.6e} | {ls_m:>14.6e} | {p_ls:>10}")

    print(f"  {'Mean final loss':<35} | {lm_s:>14.6e} | {lm_m:>14.6e} |")

    # The KEY ratio: func_diversity / weight_diversity
    ratio_s = fd_s / wd_s if wd_s > 1e-15 else np.nan
    ratio_m = fd_m / wd_m if wd_m > 1e-15 else np.nan
    p_ratio = "YES" if ratio_m < ratio_s else "no"
    print(f"  {'Func_div / Weight_div RATIO':<35} | {ratio_s:>14.6f} | {ratio_m:>14.6f} | {p_ratio:>10}")
    print(f"  (Lower ratio = more weight exploration in gauge directions)")


# =============================================================================
# PER-PERTURBATION TABLES (Face 1)
# =============================================================================

for net_type in ['linear', 'relu']:
    f1 = all_face1[net_type]
    print(f"\n\n{'=' * 100}")
    print(f"PER-PERTURBATION LYAPUNOV EXPONENTS: {net_type.upper()} NET")
    print(f"{'=' * 100}")

    print(f"\n  {'Trial':>5} | {'SGD lw':>10} | {'SGD lf':>10} | {'SGD ll':>10} | "
          f"{'Muon lw':>10} | {'Muon lf':>10} | {'Muon ll':>10}")
    print(f"  {'-' * 80}")

    for p in range(NUM_PERTURBATIONS):
        def fmt(v):
            return f"{v:>+10.6f}" if np.isfinite(v) else f"{'inf':>10}"

        print(f"  {p:>5} | {fmt(f1['sgd_lyap_w_all'][p])} | {fmt(f1['sgd_lyap_f_all'][p])} | "
              f"{fmt(f1['sgd_lyap_l_all'][p])} | {fmt(f1['muon_lyap_w_all'][p])} | "
              f"{fmt(f1['muon_lyap_f_all'][p])} | {fmt(f1['muon_lyap_l_all'][p])}")


# =============================================================================
# KEY HYPOTHESIS TESTS
# =============================================================================

print(f"\n\n{'=' * 100}")
print("KEY HYPOTHESIS TESTS")
print("=" * 100)

total_pass = 0
total_tests = 0
test_details = {}

for net_type in ['linear', 'relu']:
    f1 = all_face1[net_type]
    f2 = all_face2[net_type]

    lw_s = f1['sgd_lyap_w_mean']
    lw_m = f1['muon_lyap_w_mean']
    lf_s = f1['sgd_lyap_f_mean']
    lf_m = f1['muon_lyap_f_mean']
    ll_s = f1['sgd_lyap_l_mean']
    ll_m = f1['muon_lyap_l_mean']
    rf_s = f1['sgd_ratio_f_w_final_mean']
    rf_m = f1['muon_ratio_f_w_final_mean']

    wd_s = f2['sgd']['weight_diversity_mean']
    wd_m = f2['muon']['weight_diversity_mean']
    fd_s = f2['sgd']['func_diversity_mean']
    fd_m = f2['muon']['func_diversity_mean']
    basin_ratio_s = fd_s / wd_s if wd_s > 1e-15 else np.nan
    basin_ratio_m = fd_m / wd_m if wd_m > 1e-15 else np.nan

    print(f"\n  --- {net_type.upper()} NET ---")

    # T1: lambda_weight(Muon) > lambda_weight(SGD) -- weight chaos
    t1 = lw_m > lw_s
    total_tests += 1
    total_pass += t1
    print(f"  T1: lambda_weight(Muon) > lambda_weight(SGD)  [weight-space chaos]")
    print(f"      Muon={lw_m:+.6f} vs SGD={lw_s:+.6f}  --> {'PASS' if t1 else 'FAIL'}")

    # T2: d_func/d_weight ratio lower for Muon (more weight exploration is gauge)
    t2 = rf_m < rf_s
    total_tests += 1
    total_pass += t2
    print(f"  T2: d_func/d_weight(Muon) < d_func/d_weight(SGD)  [gauge fraction]")
    print(f"      Muon={rf_m:.6f} vs SGD={rf_s:.6f}  --> {'PASS' if t2 else 'FAIL'}")

    # T3: Convergence basin -- Muon weight diversity > SGD weight diversity
    t3 = wd_m > wd_s
    total_tests += 1
    total_pass += t3
    print(f"  T3: Weight diversity(Muon) > Weight diversity(SGD)  [diverse weights]")
    print(f"      Muon={wd_m:.6f} vs SGD={wd_s:.6f}  --> {'PASS' if t3 else 'FAIL'}")

    # T4: Convergence basin -- Muon function diversity < SGD function diversity
    t4 = fd_m < fd_s
    total_tests += 1
    total_pass += t4
    print(f"  T4: Func diversity(Muon) < Func diversity(SGD)  [consistent functions]")
    print(f"      Muon={fd_m:.6f} vs SGD={fd_s:.6f}  --> {'PASS' if t4 else 'FAIL'}")

    # T5: Basin ratio lower for Muon
    t5 = basin_ratio_m < basin_ratio_s
    total_tests += 1
    total_pass += t5
    print(f"  T5: Basin func/weight ratio(Muon) < ratio(SGD)  [THE PARADOX]")
    print(f"      Muon={basin_ratio_m:.6f} vs SGD={basin_ratio_s:.6f}  --> {'PASS' if t5 else 'FAIL'}")

    # T6: lambda_loss(Muon) < lambda_loss(SGD) OR loss_std(Muon) < loss_std(SGD)
    t6_lyap = ll_m < ll_s
    t6_basin = f2['muon']['loss_std'] < f2['sgd']['loss_std']
    t6 = t6_lyap or t6_basin
    total_tests += 1
    total_pass += t6
    print(f"  T6: Loss stability (either Lyapunov or basin std)  [loss order]")
    print(f"      Lyap: Muon={ll_m:+.6f} vs SGD={ll_s:+.6f} ({'PASS' if t6_lyap else 'FAIL'})")
    print(f"      Std:  Muon={f2['muon']['loss_std']:.6e} vs SGD={f2['sgd']['loss_std']:.6e} ({'PASS' if t6_basin else 'FAIL'})")
    print(f"      --> {'PASS' if t6 else 'FAIL'}")

    net_pass = sum([t1, t2, t3, t4, t5, t6])
    print(f"\n  {net_type.upper()} total: {net_pass}/6 tests passed")

    test_details[net_type] = {'t1': t1, 't2': t2, 't3': t3, 't4': t4, 't5': t5, 't6': t6,
                               'passes': net_pass}

    # Statistical significance for the key Face 2 tests
    # Bootstrap-style: the pairwise distances are our samples
    if len(f2['muon']['func_dists']) > 5 and len(f2['sgd']['func_dists']) > 5:
        fd_muon = f2['muon']['func_dists']
        fd_sgd = f2['sgd']['func_dists']
        n1, n2 = len(fd_sgd), len(fd_muon)
        m1, m2 = np.mean(fd_sgd), np.mean(fd_muon)
        v1, v2 = np.var(fd_sgd, ddof=1), np.var(fd_muon, ddof=1)
        se = np.sqrt(v1 / n1 + v2 / n2)
        if se > 1e-15:
            t_stat = (m1 - m2) / se  # positive if SGD > Muon
            print(f"\n  T4 significance (function diversity):")
            print(f"    SGD  mean={m1:.6f}, n={n1}")
            print(f"    Muon mean={m2:.6f}, n={n2}")
            print(f"    t-stat={t_stat:.4f} (positive => SGD > Muon)")
            print(f"    Significant (t>2.0): {'YES' if t_stat > 2.0 else 'NO'}")


# =============================================================================
# PLOTS
# =============================================================================

print(f"\n\n{'=' * 100}")
print("GENERATING PLOTS")
print("=" * 100)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t_axis = np.arange(NUM_STEPS + 1)

    for net_type in ['linear', 'relu']:
        f1 = all_face1[net_type]
        f2 = all_face2[net_type]
        lr_sgd = all_face1[f"{net_type}_lr_sgd"]

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            f'THE MUON PARADOX ({net_type.upper()} net): '
            f'Chaotic Weights, Ordered Functions\n'
            f'{NUM_LAYERS}-layer, dim={DIM}, lr_sgd={lr_sgd}, lr_muon={LR_MUON}',
            fontsize=14, fontweight='bold')

        # ---- (a) SGD: weight vs function divergence ----
        ax = axes[0, 0]
        ax.set_title('SGD: Weight vs Function Divergence')
        for p in range(NUM_PERTURBATIONS):
            dw = f1['sgd_d_w_all'][p]
            df = f1['sgd_d_f_all'][p]
            ax.semilogy(t_axis[:len(dw)], dw, 'b-', alpha=0.12, linewidth=0.5)
            ax.semilogy(t_axis[:len(df)], df, 'r-', alpha=0.12, linewidth=0.5)
        dw_m = f1['sgd_d_w_mean_traj']
        df_m = f1['sgd_d_f_mean_traj']
        ax.semilogy(t_axis[:len(dw_m)], dw_m, 'b-', linewidth=2.5,
                     label=f'd_weight (lw={f1["sgd_lyap_w_mean"]:+.4f})')
        ax.semilogy(t_axis[:len(df_m)], df_m, 'r-', linewidth=2.5,
                     label=f'd_function (lf={f1["sgd_lyap_f_mean"]:+.4f})')
        ax.set_xlabel('Step')
        ax.set_ylabel('Divergence')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # ---- (b) MUON: weight vs function divergence ----
        ax = axes[0, 1]
        ax.set_title('MUON: Weight vs Function Divergence')
        for p in range(NUM_PERTURBATIONS):
            dw = f1['muon_d_w_all'][p]
            df = f1['muon_d_f_all'][p]
            ax.semilogy(t_axis[:len(dw)], dw, 'b-', alpha=0.12, linewidth=0.5)
            ax.semilogy(t_axis[:len(df)], df, 'r-', alpha=0.12, linewidth=0.5)
        dw_m = f1['muon_d_w_mean_traj']
        df_m = f1['muon_d_f_mean_traj']
        ax.semilogy(t_axis[:len(dw_m)], dw_m, 'b-', linewidth=2.5,
                     label=f'd_weight (lw={f1["muon_lyap_w_mean"]:+.4f})')
        ax.semilogy(t_axis[:len(df_m)], df_m, 'r-', linewidth=2.5,
                     label=f'd_function (lf={f1["muon_lyap_f_mean"]:+.4f})')
        ax.set_xlabel('Step')
        ax.set_ylabel('Divergence')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # ---- (c) THE RATIO: d_func / d_weight over time ----
        ax = axes[0, 2]
        ax.set_title('RATIO d_func/d_weight Over Time\n(Lower = more gauge exploration)')
        sgd_ratio = f1['sgd_ratio_f_w_mean_traj']
        muon_ratio = f1['muon_ratio_f_w_mean_traj']
        ax.plot(t_axis[:len(sgd_ratio)], sgd_ratio, 'b-', linewidth=2.5,
                label=f'SGD (final={f1["sgd_ratio_f_w_final_mean"]:.4f})')
        ax.plot(t_axis[:len(muon_ratio)], muon_ratio, 'r-', linewidth=2.5,
                label=f'Muon (final={f1["muon_ratio_f_w_final_mean"]:.4f})')
        ax.set_xlabel('Step')
        ax.set_ylabel('d_function / d_weight')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # ---- (d) Convergence basin: weight diversity bar chart ----
        ax = axes[1, 0]
        ax.set_title('Convergence Basin: Diversity')
        categories = ['Weight\nDiversity', 'Function\nDiversity']
        sgd_vals = [f2['sgd']['weight_diversity_mean'], f2['sgd']['func_diversity_mean']]
        muon_vals = [f2['muon']['weight_diversity_mean'], f2['muon']['func_diversity_mean']]
        x = np.arange(len(categories))
        width = 0.35
        b1 = ax.bar(x - width / 2, sgd_vals, width, label='SGD', color='#4477AA', edgecolor='black')
        b2 = ax.bar(x + width / 2, muon_vals, width, label='Muon', color='#CC3311', edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylabel('Pairwise Distance (mean)')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        for bars in [b1, b2]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., h,
                        f'{h:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        # ---- (e) Convergence basin: func_div/weight_div ratio ----
        ax = axes[1, 1]
        ax.set_title('THE PARADOX RATIO:\nFunc Diversity / Weight Diversity')
        sgd_ratio_basin = f2['sgd']['func_diversity_mean'] / f2['sgd']['weight_diversity_mean'] if f2['sgd']['weight_diversity_mean'] > 1e-15 else 0
        muon_ratio_basin = f2['muon']['func_diversity_mean'] / f2['muon']['weight_diversity_mean'] if f2['muon']['weight_diversity_mean'] > 1e-15 else 0
        bars = ax.bar(['SGD', 'Muon'], [sgd_ratio_basin, muon_ratio_basin],
                       color=['#4477AA', '#CC3311'], edgecolor='black', width=0.5)
        for bar, val in zip(bars, [sgd_ratio_basin, muon_ratio_basin]):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.set_ylabel('Func Div / Weight Div')
        ax.grid(True, alpha=0.3, axis='y')

        # ---- (f) Lyapunov summary bar chart ----
        ax = axes[1, 2]
        ax.set_title('Lyapunov Exponent Summary')
        categories = ['Weight\nSpace', 'Function\nSpace', 'Loss\nSpace']
        sgd_lyaps = [f1['sgd_lyap_w_mean'], f1['sgd_lyap_f_mean'], f1['sgd_lyap_l_mean']]
        muon_lyaps = [f1['muon_lyap_w_mean'], f1['muon_lyap_f_mean'], f1['muon_lyap_l_mean']]
        x = np.arange(len(categories))
        width = 0.35
        ax.bar(x - width / 2, sgd_lyaps, width, label='SGD', color='#4477AA', edgecolor='black')
        ax.bar(x + width / 2, muon_lyaps, width, label='Muon', color='#CC3311', edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylabel('Lyapunov Exponent')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        for i, (sv, mv) in enumerate(zip(sgd_lyaps, muon_lyaps)):
            ax.text(i - width / 2, sv, f'{sv:+.4f}', ha='center',
                    va='bottom' if sv >= 0 else 'top', fontsize=8, fontweight='bold')
            ax.text(i + width / 2, mv, f'{mv:+.4f}', ha='center',
                    va='bottom' if mv >= 0 else 'top', fontsize=8, fontweight='bold')

        plt.tight_layout()
        plot_path = os.path.join(SCRIPT_DIR, f'muon_paradox_{net_type}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Plot saved: {plot_path}")

    # ---- Combined summary ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('THE MUON PARADOX: Func/Weight Diversity Ratio\n'
                 '(Lower = more weight exploration in gauge directions)',
                 fontsize=14, fontweight='bold')

    for idx, net_type in enumerate(['linear', 'relu']):
        f2 = all_face2[net_type]
        ax = axes[idx]
        ax.set_title(f'{net_type.upper()} Net')

        sgd_r = f2['sgd']['func_diversity_mean'] / f2['sgd']['weight_diversity_mean'] if f2['sgd']['weight_diversity_mean'] > 1e-15 else 0
        muon_r = f2['muon']['func_diversity_mean'] / f2['muon']['weight_diversity_mean'] if f2['muon']['weight_diversity_mean'] > 1e-15 else 0

        bars = ax.bar(['SGD', 'Muon'], [sgd_r, muon_r],
                       color=['#4477AA', '#CC3311'], edgecolor='black', width=0.5)
        for bar, val in zip(bars, [sgd_r, muon_r]):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax.set_ylabel('Func Diversity / Weight Diversity')
        ax.grid(True, alpha=0.3, axis='y')

        # Add text annotation
        ax.text(0.5, 0.85,
                f'Weight div: SGD={f2["sgd"]["weight_diversity_mean"]:.3f}, Muon={f2["muon"]["weight_diversity_mean"]:.3f}\n'
                f'Func div:   SGD={f2["sgd"]["func_diversity_mean"]:.5f}, Muon={f2["muon"]["func_diversity_mean"]:.5f}',
                transform=ax.transAxes, fontsize=9, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    combined_path = os.path.join(SCRIPT_DIR, 'muon_paradox_combined.png')
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Combined plot saved: {combined_path}")

except ImportError:
    print("  WARNING: matplotlib not available, skipping plots.")


# =============================================================================
# FINAL VERDICT
# =============================================================================

print(f"\n\n{'=' * 100}")
print("FINAL VERDICT: THE MUON PARADOX")
print("=" * 100)

print(f"""
  THE MUON PARADOX HYPOTHESIS:
    "Muon is chaotic in weight space but contractive in function space"

  This means Muon's weight exploration preferentially lies in GAUGE DIRECTIONS
  that do not affect the network's input-output mapping. Newton-Schulz
  orthogonalization acts as a gauge-fixing mechanism that redirects gradient
  information into function-relevant directions while allowing weights
  to explore freely in gauge (function-invariant) directions.

  TWO FACES OF THE PARADOX:
    Face 1: Perturbation sensitivity -- the ratio d_func/d_weight is lower for Muon
    Face 2: Convergence basin -- diverse weights, consistent functions
""")

for net_type in ['linear', 'relu']:
    td = test_details[net_type]
    print(f"  {net_type.upper()} NET: {td['passes']}/6 tests passed")
    for tname, tval, desc in [
        ('T1', td['t1'], 'Weight-space chaos (Muon > SGD)'),
        ('T2', td['t2'], 'Gauge fraction (d_f/d_w ratio Muon < SGD)'),
        ('T3', td['t3'], 'Weight diversity (Muon > SGD)'),
        ('T4', td['t4'], 'Function consistency (Muon < SGD)'),
        ('T5', td['t5'], 'Basin paradox ratio (Muon < SGD)'),
        ('T6', td['t6'], 'Loss stability'),
    ]:
        print(f"    {tname}: {'PASS' if tval else 'FAIL'}  -- {desc}")
    print()

if total_pass >= 10:
    verdict = "STRONG PASS"
    detail = "The Muon Paradox is robustly confirmed across network types."
elif total_pass >= 7:
    verdict = "PASS"
    detail = f"The Muon Paradox is confirmed ({total_pass}/{total_tests} tests)."
elif total_pass >= 5:
    verdict = "PARTIAL PASS"
    detail = f"Partial support ({total_pass}/{total_tests} tests). Core mechanism present but not universal."
else:
    verdict = "FAIL"
    detail = f"Not confirmed ({total_pass}/{total_tests} tests). The hypothesis needs revision."

print(f"""
  {'=' * 74}
  ||  VERDICT: {verdict:<60}||
  {'=' * 74}

  {detail}

  Total: {total_pass}/{total_tests} tests passed
  {'=' * 74}
""")
