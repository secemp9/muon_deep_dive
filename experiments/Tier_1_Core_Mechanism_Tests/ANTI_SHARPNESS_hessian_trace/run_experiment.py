#!/usr/bin/env python3
"""
Experiment 2.10: Anti-sharpness toy test — endpoint finite-difference Hessian comparison
=========================================================================================

This file runs a toy endpoint-curvature study on small deep linear networks.
It compares SGD+momentum against Muon (momentum with Newton-Schulz-orthogonalized
gradients), trains both optimizers to a loss threshold, and then computes a full
finite-difference Hessian estimate at the final weights.

What this script measures:
  - final-loss convergence behavior
  - steps-to-threshold
  - endpoint Hessian spectral summaries (trace, lambda_max, effective rank,
    near-zero eigenvalue count proxy, condition number, negative-curvature diagnostics)

What this script does not measure:
  - training-trajectory sharpness or movement through sharp directions
  - a true gauge-subspace dimension
  - generalization

The near-zero count is only a threshold-based proxy; it is not a direct gauge measurement.
"""

import time
import numpy as np


DEFAULT_CONFIG = {
    'dim': 4,
    'num_steps': 3000,
    'lr_sgd': 0.01,
    'lr_muon': 0.02,
    'momentum': 0.9,
    'ns_iters': 5,
    'num_seeds': 10,
    'convergence_factor': 0.001,
    'hessian_eps': 1e-5,
    'batch_size': 32,
    'target_weight_scale': 0.3,
    'input_scale': 0.5,
    'seed_start': 42,
    'seed_stride': 137,
    'init_seed_offset': 1000,
    'eff_rank_threshold': 0.01,
    'near_zero_threshold': 0.001,
    'positive_eig_tol': 1e-12,
    'negative_eig_tol': -1e-12,
}

DEFAULT_ARCHITECTURES = (
    {'key': '2-layer', 'label': 'PART A', 'num_layers': 2},
    {'key': '3-layer', 'label': 'PART B', 'num_layers': 3},
)

DIM = DEFAULT_CONFIG['dim']
NUM_STEPS = DEFAULT_CONFIG['num_steps']
LR_SGD = DEFAULT_CONFIG['lr_sgd']
LR_MUON = DEFAULT_CONFIG['lr_muon']
MOMENTUM = DEFAULT_CONFIG['momentum']
NS_ITERS = DEFAULT_CONFIG['ns_iters']
NUM_SEEDS = DEFAULT_CONFIG['num_seeds']
CONVERGENCE_FACTOR = DEFAULT_CONFIG['convergence_factor']
HESSIAN_EPS = DEFAULT_CONFIG['hessian_eps']

METRIC_DISPLAY = {
    'trace': {'label': 'tr(H)', 'kind': 'scientific'},
    'lambda_max': {'label': 'lambda_max', 'kind': 'scientific'},
    'eff_rank': {'label': 'Eff. Hessian rank', 'kind': 'count'},
    'near_zero_count_proxy': {'label': 'Near-zero count proxy', 'kind': 'count'},
    'kappa': {'label': 'Condition number kappa', 'kind': 'float'},
    'loss': {'label': 'Final loss', 'kind': 'scientific'},
    'steps': {'label': 'Steps to threshold', 'kind': 'count'},
    'n_negative': {'label': 'Negative eigenvalue count', 'kind': 'count'},
    'sum_negative': {'label': 'Sum of negative eigenvalues', 'kind': 'scientific'},
    'negative_mass_ratio': {'label': 'Negative mass ratio', 'kind': 'scientific'},
    'min_eigenvalue': {'label': 'Minimum eigenvalue', 'kind': 'scientific'},
}

PRIMARY_METRICS = ['trace', 'lambda_max', 'eff_rank', 'near_zero_count_proxy', 'kappa']
SECONDARY_METRICS = ['loss', 'steps', 'n_negative', 'sum_negative', 'negative_mass_ratio', 'min_eigenvalue']
SUMMARY_METRICS = PRIMARY_METRICS + SECONDARY_METRICS


# =============================================================================
# CONFIGURATION HELPERS
# =============================================================================


def get_default_config():
    """Return a copy of the default configuration."""
    return dict(DEFAULT_CONFIG)


def resolve_config(config=None, **overrides):
    """Merge a user config with defaults."""
    resolved = get_default_config()
    if config is not None:
        resolved.update(config)
    for key, value in overrides.items():
        if value is not None:
            resolved[key] = value
    return resolved


def build_seed_list(config):
    """Construct the deterministic seed list used by the experiment."""
    return [
        int(config['seed_start'] + i * config['seed_stride'])
        for i in range(config['num_seeds'])
    ]


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


def copy_weights(weights):
    """Deep-copy a list of weight matrices."""
    return [W.copy() for W in weights]


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

    activations = [X.copy()]
    for W in weights:
        activations.append(W @ activations[-1])

    Y_pred = activations[-1]
    delta = (Y_pred - Y_target) / batch_size

    grads = [None] * num_layers
    for l in range(num_layers - 1, -1, -1):
        grads[l] = delta @ activations[l].T
        if l > 0:
            delta = weights[l].T @ delta

    return grads


# =============================================================================
# NEWTON-SCHULZ ITERATION (Muon's core)
# =============================================================================


def newton_schulz_orthogonalize(G, num_iters):
    """Apply Newton-Schulz iteration to approximate the orthogonal polar factor."""
    norm = np.linalg.norm(G, ord='fro')
    if norm < 1e-12:
        return G
    X = G / norm

    for _ in range(num_iters):
        A = X @ X.T
        X = 1.5 * X - 0.5 * A @ X

    return X


# =============================================================================
# OPTIMIZERS
# =============================================================================


def train_sgd(weights, X, Y_target, lr, momentum, num_steps, convergence_threshold):
    """Train with SGD + momentum."""
    velocities = [np.zeros_like(W) for W in weights]
    initial_loss = compute_loss(weights, X, Y_target)
    target_loss = initial_loss * convergence_threshold

    for step in range(num_steps):
        loss = compute_loss(weights, X, Y_target)
        if loss < target_loss:
            return weights, float(loss), int(step), True

        grads = compute_gradients(weights, X, Y_target)
        for i in range(len(weights)):
            velocities[i] = momentum * velocities[i] + grads[i]
            weights[i] = weights[i] - lr * velocities[i]

    final_loss = compute_loss(weights, X, Y_target)
    return weights, float(final_loss), int(num_steps), bool(final_loss < target_loss)


def train_muon(weights, X, Y_target, lr, momentum, ns_iters, num_steps, convergence_threshold):
    """Train with Muon (NS-orthogonalized gradients + momentum)."""
    velocities = [np.zeros_like(W) for W in weights]
    initial_loss = compute_loss(weights, X, Y_target)
    target_loss = initial_loss * convergence_threshold

    for step in range(num_steps):
        loss = compute_loss(weights, X, Y_target)
        if loss < target_loss:
            return weights, float(loss), int(step), True

        grads = compute_gradients(weights, X, Y_target)
        for i in range(len(weights)):
            G_orth = newton_schulz_orthogonalize(grads[i], num_iters=ns_iters)
            velocities[i] = momentum * velocities[i] + G_orth
            weights[i] = weights[i] - lr * velocities[i]

    final_loss = compute_loss(weights, X, Y_target)
    return weights, float(final_loss), int(num_steps), bool(final_loss < target_loss)


# =============================================================================
# FULL HESSIAN COMPUTATION VIA FINITE DIFFERENCES
# =============================================================================


def weights_to_vector(weights):
    """Flatten all weight matrices into a single vector."""
    return np.concatenate([W.flatten() for W in weights])


def vector_to_weights(vec, shapes):
    """Unflatten a vector back into a list of weight matrices."""
    weights = []
    idx = 0
    for shape in shapes:
        size = shape[0] * shape[1]
        W = vec[idx:idx + size].reshape(shape)
        weights.append(W)
        idx += size
    return weights


def compute_gradient_vector(weights, X, Y_target):
    """Return the gradient as a flat vector."""
    grads = compute_gradients(weights, X, Y_target)
    return np.concatenate([g.flatten() for g in grads])


def compute_full_hessian(weights, X, Y_target, eps, return_diagnostics=False):
    """Compute the full Hessian estimate via central finite differences on the gradient."""
    shapes = [W.shape for W in weights]
    theta = weights_to_vector(weights)
    n_params = len(theta)

    H_raw = np.zeros((n_params, n_params))

    for i in range(n_params):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[i] += eps
        theta_minus[i] -= eps

        w_plus = vector_to_weights(theta_plus, shapes)
        w_minus = vector_to_weights(theta_minus, shapes)

        grad_plus = compute_gradient_vector(w_plus, X, Y_target)
        grad_minus = compute_gradient_vector(w_minus, X, Y_target)

        H_raw[:, i] = (grad_plus - grad_minus) / (2 * eps)

    pre_sym_asymmetry = float(np.max(np.abs(H_raw - H_raw.T))) if n_params > 0 else 0.0
    H = 0.5 * (H_raw + H_raw.T)

    if return_diagnostics:
        diagnostics = {
            'eps': float(eps),
            'n_params': int(n_params),
            'pre_symmetry_max_abs_asymmetry': pre_sym_asymmetry,
        }
        return H, diagnostics

    return H


# =============================================================================
# HESSIAN ANALYSIS
# =============================================================================


def analyze_hessian(H, eff_rank_threshold, near_zero_threshold, positive_eig_tol, negative_eig_tol,
                    include_eigenvalues=True):
    """Compute endpoint Hessian statistics from a symmetric Hessian estimate."""
    eigenvalues = np.linalg.eigvalsh(H)
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]

    if len(eigenvalues_sorted) == 0:
        lambda_max = 0.0
        trace_H = 0.0
        eff_rank = 0
        near_zero_count_proxy = 0
        min_eigenvalue = 0.0
        min_positive_eigenvalue = None
        kappa = float('inf')
        n_negative = 0
        sum_negative = 0.0
        negative_mass_ratio = float('nan')
    else:
        lambda_max = float(eigenvalues_sorted[0])
        trace_H = float(np.sum(eigenvalues))

        if lambda_max > 1e-15:
            eff_rank = int(np.sum(np.abs(eigenvalues) > eff_rank_threshold * lambda_max))
            near_zero_count_proxy = int(np.sum(np.abs(eigenvalues) < near_zero_threshold * lambda_max))
        else:
            eff_rank = 0
            near_zero_count_proxy = int(len(eigenvalues))

        positive_eigs = eigenvalues[eigenvalues > positive_eig_tol]
        if len(positive_eigs) > 0:
            min_positive_eigenvalue = float(np.min(positive_eigs))
            kappa = float(lambda_max / min_positive_eigenvalue)
        else:
            min_positive_eigenvalue = None
            kappa = float('inf')

        neg_eigs = eigenvalues[eigenvalues < negative_eig_tol]
        n_negative = int(len(neg_eigs))
        sum_negative = float(np.sum(neg_eigs)) if n_negative > 0 else 0.0
        min_eigenvalue = float(np.min(eigenvalues))
        negative_mass_ratio = float(abs(sum_negative) / abs(trace_H)) if abs(trace_H) > 1e-15 else float('nan')

    result = {
        'trace': trace_H,
        'lambda_max': lambda_max,
        'eff_rank': eff_rank,
        'near_zero_count_proxy': near_zero_count_proxy,
        'kappa': kappa,
        'n_negative': n_negative,
        'sum_negative': sum_negative,
        'negative_mass_ratio': negative_mass_ratio,
        'min_eigenvalue': min_eigenvalue,
        'min_positive_eigenvalue': min_positive_eigenvalue,
    }
    if include_eigenvalues:
        result['eigenvalues'] = eigenvalues_sorted.tolist()
    return result


# =============================================================================
# AGGREGATION AND REPORTING UTILITIES
# =============================================================================


def safe_mean_std(values):
    """Return mean/std over finite values plus the count of non-finite entries."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    nonfinite_count = int(np.sum(~np.isfinite(arr)))
    if finite.size == 0:
        return float('nan'), float('nan'), nonfinite_count
    return float(np.mean(finite)), float(np.std(finite)), nonfinite_count


def summarize_paired_metric(seed_results, metric):
    """Aggregate one metric across seeds for SGD and Muon."""
    sgd_values = np.asarray([result['sgd'][metric] for result in seed_results], dtype=float)
    muon_values = np.asarray([result['muon'][metric] for result in seed_results], dtype=float)

    sgd_mean, sgd_std, sgd_nonfinite = safe_mean_std(sgd_values)
    muon_mean, muon_std, muon_nonfinite = safe_mean_std(muon_values)

    finite_pair_mask = np.isfinite(sgd_values) & np.isfinite(muon_values)
    pairwise_differences = muon_values[finite_pair_mask] - sgd_values[finite_pair_mask]
    paired_mean_difference = float(np.mean(pairwise_differences)) if pairwise_differences.size else float('nan')

    valid_ratio_mask = finite_pair_mask & (np.abs(sgd_values) > 1e-15)
    seedwise_ratios = muon_values[valid_ratio_mask] / sgd_values[valid_ratio_mask]
    finite_ratios = seedwise_ratios[np.isfinite(seedwise_ratios)]
    mean_seedwise_ratio = float(np.mean(finite_ratios)) if finite_ratios.size else float('nan')
    median_seedwise_ratio = float(np.median(finite_ratios)) if finite_ratios.size else float('nan')

    if np.isfinite(sgd_mean) and abs(sgd_mean) > 1e-15 and np.isfinite(muon_mean):
        ratio_of_means = float(muon_mean / sgd_mean)
    else:
        ratio_of_means = float('inf')

    return {
        'metric': metric,
        'n': int(len(seed_results)),
        'sgd': {
            'mean': sgd_mean,
            'std': sgd_std,
            'nonfinite_count': sgd_nonfinite,
        },
        'muon': {
            'mean': muon_mean,
            'std': muon_std,
            'nonfinite_count': muon_nonfinite,
        },
        'paired_mean_difference': paired_mean_difference,
        'ratio_of_means': ratio_of_means,
        'mean_seedwise_ratio': mean_seedwise_ratio,
        'median_seedwise_ratio': median_seedwise_ratio,
        'valid_seedwise_ratio_count': int(finite_ratios.size),
        'muon_lt_sgd_count': int(np.sum(muon_values[finite_pair_mask] < sgd_values[finite_pair_mask])),
        'muon_gt_sgd_count': int(np.sum(muon_values[finite_pair_mask] > sgd_values[finite_pair_mask])),
    }


def summarize_metrics(seed_results):
    """Aggregate all headline metrics across seeds."""
    return {metric: summarize_paired_metric(seed_results, metric) for metric in SUMMARY_METRICS}


def build_trace_hypothesis(metric_summary):
    """Interpret the final-trace hypothesis using the ratio of mean traces."""
    trace_ratio = metric_summary['ratio_of_means']
    mean_seedwise_ratio = metric_summary['mean_seedwise_ratio']
    muon_lower_count = metric_summary['muon_lt_sgd_count']
    n_seeds = metric_summary['n']

    if trace_ratio < 0.5:
        verdict = 'supported'
        verdict_message = (
            f"Hypothesis supported: ratio of mean traces = {trace_ratio:.4f} < 0.5."
        )
    elif trace_ratio < 1.0:
        verdict = 'weaker_support'
        verdict_message = (
            f"Muon has a lower mean trace than SGD (ratio = {trace_ratio:.4f}) but not the pre-registered 2x gap."
        )
    else:
        verdict = 'rejected'
        verdict_message = (
            f"Hypothesis rejected under this metric: ratio of mean traces = {trace_ratio:.4f} >= 1.0."
        )

    return {
        'statement': 'Muon endpoint Hessian trace is less than half of SGD endpoint Hessian trace.',
        'ratio_of_means': trace_ratio,
        'mean_seedwise_ratio': mean_seedwise_ratio,
        'muon_lower_trace_seeds': muon_lower_count,
        'n_seeds': n_seeds,
        'verdict': verdict,
        'verdict_message': verdict_message,
    }


def build_speed_summary(metric_summary):
    """Interpret the steps-to-threshold comparison."""
    steps_ratio = metric_summary['ratio_of_means']
    if steps_ratio < 1.0:
        verdict = 'muon_faster'
        message = f"Muon reaches the threshold faster on average (mean-steps ratio = {steps_ratio:.4f})."
    elif steps_ratio > 1.0:
        verdict = 'sgd_faster'
        message = f"SGD reaches the threshold faster on average (mean-steps ratio = {steps_ratio:.4f})."
    else:
        verdict = 'parity'
        message = f"Both optimizers reach the threshold in the same mean number of steps (ratio = {steps_ratio:.4f})."

    return {
        'statement': 'Muon reaches the convergence threshold in fewer steps than SGD.',
        'ratio_of_means': steps_ratio,
        'mean_seedwise_ratio': metric_summary['mean_seedwise_ratio'],
        'muon_fewer_steps_seeds': metric_summary['muon_lt_sgd_count'],
        'n_seeds': metric_summary['n'],
        'verdict': verdict,
        'verdict_message': message,
    }


def build_decision_outputs(metric_summaries):
    """Build the current test outputs used by the script and notebook."""
    return {
        'trace_hypothesis': build_trace_hypothesis(metric_summaries['trace']),
        'speed_comparison': build_speed_summary(metric_summaries['steps']),
    }


def build_eigenvalue_bin_summary(seed_results, negative_eig_tol):
    """Summarize simple eigenvalue count bins for the stored spectra."""
    bins = [
        ('|eig| > 1.0', lambda e: np.sum(np.abs(e) > 1.0)),
        ('0.1 < |eig| <= 1.0', lambda e: np.sum((np.abs(e) > 0.1) & (np.abs(e) <= 1.0))),
        ('0.01 < |eig| <= 0.1', lambda e: np.sum((np.abs(e) > 0.01) & (np.abs(e) <= 0.1))),
        ('0.001 < |eig| <= 0.01', lambda e: np.sum((np.abs(e) > 0.001) & (np.abs(e) <= 0.01))),
        ('|eig| <= 0.001', lambda e: np.sum(np.abs(e) <= 0.001)),
        (f'eig < {negative_eig_tol:.0e}', lambda e: np.sum(e < negative_eig_tol)),
    ]

    summary = {}
    for label, count_fn in bins:
        sgd_counts = []
        muon_counts = []
        for result in seed_results:
            if 'eigenvalues' not in result['sgd'] or 'eigenvalues' not in result['muon']:
                continue
            sgd_counts.append(count_fn(np.asarray(result['sgd']['eigenvalues'], dtype=float)))
            muon_counts.append(count_fn(np.asarray(result['muon']['eigenvalues'], dtype=float)))
        if sgd_counts and muon_counts:
            summary[label] = {
                'sgd_mean': float(np.mean(sgd_counts)),
                'sgd_std': float(np.std(sgd_counts)),
                'muon_mean': float(np.mean(muon_counts)),
                'muon_std': float(np.std(muon_counts)),
            }
    return summary


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================


def run_single_experiment(num_layers, dim, seed, config, include_eigenvalues=True):
    """Run one SGD-vs-Muon comparison for a given architecture and seed."""
    rng = np.random.RandomState(seed)
    W_target_list = [rng.randn(dim, dim) * config['target_weight_scale'] for _ in range(num_layers)]
    X = rng.randn(dim, config['batch_size']) * config['input_scale']

    Y_target = X.copy()
    for W in W_target_list:
        Y_target = W @ Y_target

    init_seed = seed + config['init_seed_offset']
    initial_weights = init_weights(num_layers, dim, init_seed)
    initial_loss = float(compute_loss(initial_weights, X, Y_target))
    target_loss = float(initial_loss * config['convergence_factor'])

    w_sgd = copy_weights(initial_weights)
    w_muon = copy_weights(initial_weights)

    w_sgd, loss_sgd, steps_sgd, conv_sgd = train_sgd(
        w_sgd,
        X,
        Y_target,
        lr=config['lr_sgd'],
        momentum=config['momentum'],
        num_steps=config['num_steps'],
        convergence_threshold=config['convergence_factor'],
    )
    w_muon, loss_muon, steps_muon, conv_muon = train_muon(
        w_muon,
        X,
        Y_target,
        lr=config['lr_muon'],
        momentum=config['momentum'],
        ns_iters=config['ns_iters'],
        num_steps=config['num_steps'],
        convergence_threshold=config['convergence_factor'],
    )

    H_sgd, hessian_diag_sgd = compute_full_hessian(
        w_sgd, X, Y_target, eps=config['hessian_eps'], return_diagnostics=True
    )
    H_muon, hessian_diag_muon = compute_full_hessian(
        w_muon, X, Y_target, eps=config['hessian_eps'], return_diagnostics=True
    )

    stats_sgd = analyze_hessian(
        H_sgd,
        eff_rank_threshold=config['eff_rank_threshold'],
        near_zero_threshold=config['near_zero_threshold'],
        positive_eig_tol=config['positive_eig_tol'],
        negative_eig_tol=config['negative_eig_tol'],
        include_eigenvalues=include_eigenvalues,
    )
    stats_muon = analyze_hessian(
        H_muon,
        eff_rank_threshold=config['eff_rank_threshold'],
        near_zero_threshold=config['near_zero_threshold'],
        positive_eig_tol=config['positive_eig_tol'],
        negative_eig_tol=config['negative_eig_tol'],
        include_eigenvalues=include_eigenvalues,
    )

    sgd_result = {
        **stats_sgd,
        'loss': float(loss_sgd),
        'steps': int(steps_sgd),
        'converged': bool(conv_sgd),
        'hessian_fd': hessian_diag_sgd,
    }
    muon_result = {
        **stats_muon,
        'loss': float(loss_muon),
        'steps': int(steps_muon),
        'converged': bool(conv_muon),
        'hessian_fd': hessian_diag_muon,
    }

    return {
        'seed': int(seed),
        'initialization_seed': int(init_seed),
        'initial_loss': initial_loss,
        'target_loss': target_loss,
        'sgd': sgd_result,
        'muon': muon_result,
    }


def run_architecture_experiment(num_layers, config, label, key=None, include_eigenvalues=True):
    """Run the experiment suite for one architecture and return structured results."""
    dim = int(config['dim'])
    n_params = int(num_layers * dim * dim)
    seed_list = build_seed_list(config)

    seed_results = [
        run_single_experiment(
            num_layers=num_layers,
            dim=dim,
            seed=seed,
            config=config,
            include_eigenvalues=include_eigenvalues,
        )
        for seed in seed_list
    ]

    metric_summaries = summarize_metrics(seed_results)
    decision_outputs = build_decision_outputs(metric_summaries)
    eigenvalue_bin_summary = build_eigenvalue_bin_summary(seed_results, config['negative_eig_tol'])

    return {
        'key': key or f'{num_layers}-layer',
        'label': label,
        'num_layers': int(num_layers),
        'dim': dim,
        'n_params': n_params,
        'seed_results': seed_results,
        'metric_summaries': metric_summaries,
        'eigenvalue_bin_summary': eigenvalue_bin_summary,
        'decision_outputs': decision_outputs,
    }


def build_depth_comparison(architectures):
    """Summarize the 2-layer vs 3-layer trace comparison without overclaiming."""
    if '2-layer' not in architectures or '3-layer' not in architectures:
        return {
            'available': False,
            'interpretation': 'Depth comparison not available because both 2-layer and 3-layer results are required.',
        }

    trace_2 = architectures['2-layer']['metric_summaries']['trace']['ratio_of_means']
    trace_3 = architectures['3-layer']['metric_summaries']['trace']['ratio_of_means']

    if trace_3 < trace_2:
        if trace_2 >= 1.0 and trace_3 >= 1.0:
            interpretation = (
                'The 3-layer case is closer to parity than the 2-layer case, '
                'but neither architecture supports the anti-sharpness trace hypothesis.'
            )
        else:
            interpretation = (
                'The 3-layer case shows a lower Muon/SGD trace ratio than the 2-layer case; '
                'interpretation still depends on whether either ratio is actually below 1.'
            )
    elif trace_3 > trace_2:
        interpretation = (
            'The 3-layer case does not reduce the Muon/SGD trace ratio relative to the 2-layer case.'
        )
    else:
        interpretation = 'The 2-layer and 3-layer trace ratios are equal at the reported precision.'

    return {
        'available': True,
        'trace_ratio_2_layer': trace_2,
        'trace_ratio_3_layer': trace_3,
        'interpretation': interpretation,
    }


def build_overall_summary(architectures, runtime_seconds):
    """Construct overall summaries spanning all requested architectures."""
    trace_supported_architectures = [
        key for key, arch in architectures.items()
        if arch['decision_outputs']['trace_hypothesis']['verdict'] == 'supported'
    ]
    lower_trace_architectures = [
        key for key, arch in architectures.items()
        if arch['decision_outputs']['trace_hypothesis']['ratio_of_means'] < 1.0
    ]
    faster_architectures = [
        key for key, arch in architectures.items()
        if arch['decision_outputs']['speed_comparison']['verdict'] == 'muon_faster'
    ]

    if not lower_trace_architectures:
        summary_statement = (
            'Muon reaches the threshold faster in this toy setting, but the final Hessian-trace '
            'hypothesis is not supported for any tested architecture.'
        )
    else:
        summary_statement = (
            'Muon is faster on at least one architecture and has a lower final trace on at least one '
            'architecture, but support should be judged architecture-by-architecture.'
        )

    return {
        'runtime_seconds': float(runtime_seconds),
        'trace_supported_architectures': trace_supported_architectures,
        'lower_trace_architectures': lower_trace_architectures,
        'faster_architectures': faster_architectures,
        'depth_comparison': build_depth_comparison(architectures),
        'summary_statement': summary_statement,
    }


def run_experiment(config=None, architectures=None, include_eigenvalues=True, verbose=False):
    """Run the full experiment and return structured results suitable for scripting or notebooks."""
    resolved_config = resolve_config(config)
    architecture_specs = tuple(architectures or DEFAULT_ARCHITECTURES)

    start_time = time.perf_counter()
    architecture_results = {}
    for spec in architecture_specs:
        architecture_result = run_architecture_experiment(
            num_layers=spec['num_layers'],
            config=resolved_config,
            label=spec.get('label', spec['key']),
            key=spec['key'],
            include_eigenvalues=include_eigenvalues,
        )
        architecture_results[spec['key']] = architecture_result
    runtime_seconds = time.perf_counter() - start_time

    results = {
        'study_id': 'Experiment 2.10',
        'title': 'Anti-sharpness toy test: final finite-difference Hessian comparison',
        'question': 'Does Muon converge to a lower endpoint Hessian trace than SGD in this toy deep-linear setting?',
        'scope_notes': [
            'This is a final-endpoint curvature study, not a training-trajectory sharpness analysis.',
            'The Hessian is estimated via central finite differences of the gradient.',
            'The near-zero eigenvalue count is only a proxy and should not be interpreted as a direct gauge measurement.',
        ],
        'config': resolved_config,
        'seeds': build_seed_list(resolved_config),
        'architecture_order': [spec['key'] for spec in architecture_specs],
        'architecture_specs': [
            {
                'key': spec['key'],
                'label': spec.get('label', spec['key']),
                'num_layers': int(spec['num_layers']),
            }
            for spec in architecture_specs
        ],
        'architectures': architecture_results,
        'overall': build_overall_summary(architecture_results, runtime_seconds),
    }

    if verbose:
        print_experiment_report(results)

    return results


# =============================================================================
# CLI REPORTING
# =============================================================================


def print_separator(char='=', width=110):
    print(char * width)


def format_ratio(value):
    if np.isfinite(value):
        return f"{value:.4f}"
    return 'N/A'


def format_mean_std(mean, std, kind):
    if not np.isfinite(mean):
        return 'N/A'
    if kind == 'count':
        return f"{mean:8.1f} +/- {std:6.1f}"
    if kind == 'float':
        return f"{mean:10.1f} +/- {std:8.1f}"
    return f"{mean:12.4e} +/- {std:.2e}"


def print_architecture_report(architecture_result, config):
    """Pretty-print one architecture block from structured results."""
    print_separator()
    print(
        f"  {architecture_result['label']}: {architecture_result['num_layers']}-LAYER DEEP LINEAR NET "
        f"({architecture_result['dim']}x{architecture_result['dim']}, {architecture_result['n_params']} params)"
    )
    print_separator()

    for index, result in enumerate(architecture_result['seed_results'], start=1):
        sgd_c = 'YES' if result['sgd']['converged'] else 'NO'
        muon_c = 'YES' if result['muon']['converged'] else 'NO'
        print(
            f"  Seed {index:2d}/{len(architecture_result['seed_results'])}: "
            f"SGD conv={sgd_c} (loss={result['sgd']['loss']:.2e}, steps={result['sgd']['steps']})  |  "
            f"Muon conv={muon_c} (loss={result['muon']['loss']:.2e}, steps={result['muon']['steps']})"
        )

    print()
    print_separator('-')
    print('  CONVERGENCE SUMMARY:')
    loss_summary = architecture_result['metric_summaries']['loss']
    step_summary = architecture_result['metric_summaries']['steps']
    sgd_conv_count = sum(1 for result in architecture_result['seed_results'] if result['sgd']['converged'])
    muon_conv_count = sum(1 for result in architecture_result['seed_results'] if result['muon']['converged'])
    n_seeds = len(architecture_result['seed_results'])
    print(
        f"    SGD converged:  {sgd_conv_count}/{n_seeds}  "
        f"(mean loss = {loss_summary['sgd']['mean']:.2e} +/- {loss_summary['sgd']['std']:.2e}, "
        f"mean steps = {step_summary['sgd']['mean']:.1f})"
    )
    print(
        f"    Muon converged: {muon_conv_count}/{n_seeds}  "
        f"(mean loss = {loss_summary['muon']['mean']:.2e} +/- {loss_summary['muon']['std']:.2e}, "
        f"mean steps = {step_summary['muon']['mean']:.1f})"
    )
    print_separator('-')
    print()

    print(
        f"  {'Metric':<28s} | {'SGD (mean +/- std)':<28s} | {'Muon (mean +/- std)':<28s} | {'Ratio Muon/SGD':<16s}"
    )
    print(f"  {'-' * 28}-+-{'-' * 28}-+-{'-' * 28}-+-{'-' * 16}")
    for metric in PRIMARY_METRICS:
        summary = architecture_result['metric_summaries'][metric]
        display = METRIC_DISPLAY[metric]
        sgd_str = format_mean_std(summary['sgd']['mean'], summary['sgd']['std'], display['kind'])
        muon_str = format_mean_std(summary['muon']['mean'], summary['muon']['std'], display['kind'])
        ratio_str = format_ratio(summary['ratio_of_means'])
        print(f"  {display['label']:<28s} | {sgd_str:<28s} | {muon_str:<28s} | {ratio_str:<16s}")
    print()

    print('  NEGATIVE-CURVATURE DIAGNOSTICS:')
    for metric in ['n_negative', 'sum_negative', 'negative_mass_ratio', 'min_eigenvalue']:
        summary = architecture_result['metric_summaries'][metric]
        display = METRIC_DISPLAY[metric]
        sgd_str = format_mean_std(summary['sgd']['mean'], summary['sgd']['std'], display['kind'])
        muon_str = format_mean_std(summary['muon']['mean'], summary['muon']['std'], display['kind'])
        ratio_str = format_ratio(summary['ratio_of_means'])
        print(f"    {display['label']:<26s} | SGD {sgd_str:<24s} | Muon {muon_str:<24s} | ratio {ratio_str}")
    print()

    trace_test = architecture_result['decision_outputs']['trace_hypothesis']
    speed_test = architecture_result['decision_outputs']['speed_comparison']
    print_separator('*')
    print('  TRACE HYPOTHESIS TEST (ratio of means): tr(H_Muon) < 0.5 * tr(H_SGD)')
    print(f"    ratio of mean traces      = {trace_test['ratio_of_means']:.4f}")
    print(f"    mean seedwise trace ratio = {trace_test['mean_seedwise_ratio']:.4f}")
    print(f"    Muon lower trace on       = {trace_test['muon_lower_trace_seeds']}/{trace_test['n_seeds']} seeds")
    print(f"    verdict                   = {trace_test['verdict_message']}")
    print()
    print('  SPEED COMPARISON:')
    print(f"    ratio of mean steps       = {speed_test['ratio_of_means']:.4f}")
    print(f"    mean seedwise step ratio  = {speed_test['mean_seedwise_ratio']:.4f}")
    print(f"    Muon fewer steps on       = {speed_test['muon_fewer_steps_seeds']}/{speed_test['n_seeds']} seeds")
    print(f"    verdict                   = {speed_test['verdict_message']}")
    print_separator('*')
    print()

    if architecture_result['eigenvalue_bin_summary']:
        print('  EIGENVALUE SPECTRUM BIN COUNTS (averaged over seeds):')
        print(f"  {'Bin':<30s} | {'SGD count':<14s} | {'Muon count':<14s}")
        print(f"  {'-' * 30}-+-{'-' * 14}-+-{'-' * 14}")
        for label, stats in architecture_result['eigenvalue_bin_summary'].items():
            print(
                f"  {label:<30s} | {stats['sgd_mean']:6.1f} +/- {stats['sgd_std']:4.1f} | "
                f"{stats['muon_mean']:6.1f} +/- {stats['muon_std']:4.1f}"
            )
        print()

    asymmetry_sgd = [
        result['sgd']['hessian_fd']['pre_symmetry_max_abs_asymmetry']
        for result in architecture_result['seed_results']
    ]
    asymmetry_muon = [
        result['muon']['hessian_fd']['pre_symmetry_max_abs_asymmetry']
        for result in architecture_result['seed_results']
    ]
    print('  FINITE-DIFFERENCE HESSIAN QUALITY CHECK:')
    print(f"    SGD  pre-symmetry max |H-H^T|: mean={np.mean(asymmetry_sgd):.2e}, std={np.std(asymmetry_sgd):.2e}")
    print(f"    Muon pre-symmetry max |H-H^T|: mean={np.mean(asymmetry_muon):.2e}, std={np.std(asymmetry_muon):.2e}")
    print(f"    eps = {config['hessian_eps']}, near-zero threshold = {config['near_zero_threshold']} * lambda_max")
    print()


def print_final_summary(results):
    """Pretty-print the overall architecture comparison."""
    print_separator('#')
    print('  FINAL COMPARATIVE SUMMARY (ratio of means, Muon/SGD)')
    print_separator('#')
    print()
    print(f"  {'Metric':<28s} | {'2-layer':<20s} | {'3-layer':<20s}")
    print(f"  {'-' * 28}-+-{'-' * 20}-+-{'-' * 20}")
    for metric in PRIMARY_METRICS:
        label = METRIC_DISPLAY[metric]['label']
        ratio_2 = results['architectures']['2-layer']['metric_summaries'][metric]['ratio_of_means']
        ratio_3 = results['architectures']['3-layer']['metric_summaries'][metric]['ratio_of_means']
        print(f"  {label:<28s} | {format_ratio(ratio_2):<20s} | {format_ratio(ratio_3):<20s}")
    print()

    depth_comparison = results['overall']['depth_comparison']
    print_separator('*')
    print('  OVERALL VERDICT:')
    for key in ['2-layer', '3-layer']:
        trace_test = results['architectures'][key]['decision_outputs']['trace_hypothesis']
        speed_test = results['architectures'][key]['decision_outputs']['speed_comparison']
        print(
            f"    {key}: trace verdict = {trace_test['verdict']} "
            f"(ratio={trace_test['ratio_of_means']:.4f}), speed verdict = {speed_test['verdict']} "
            f"(ratio={speed_test['ratio_of_means']:.4f})"
        )
    print()
    print(f"    {results['overall']['summary_statement']}")
    print(f"    Depth comparison: {depth_comparison['interpretation']}")
    print_separator('*')
    print()


def print_experiment_report(results):
    """Pretty-print the full CLI report from structured results."""
    print()
    print_separator('#')
    print('  EXPERIMENT 2.10: ANTI-SHARPNESS TOY TEST — FINAL FINITE-DIFFERENCE HESSIAN COMPARISON')
    print('  Question: does Muon reach a lower endpoint Hessian trace than SGD?')
    print_separator('#')
    print()
    print(f"  Runtime: {results['overall']['runtime_seconds']:.2f} seconds")
    print(
        f"  Config: {results['config']['num_seeds']} seeds, max {results['config']['num_steps']} steps, "
        f"convergence threshold = {results['config']['convergence_factor']}"
    )
    print(
        f"  SGD lr = {results['config']['lr_sgd']}, Muon lr = {results['config']['lr_muon']}, "
        f"momentum = {results['config']['momentum']}, NS iters = {results['config']['ns_iters']}"
    )
    print(
        f"  Hessian: full central finite-difference estimate at converged endpoints "
        f"(eps = {results['config']['hessian_eps']})"
    )
    print(f"  Seeds: {results['seeds']}")
    print('  Scope notes:')
    for note in results['scope_notes']:
        print(f"    - {note}")
    print()

    for spec in DEFAULT_ARCHITECTURES:
        print_architecture_report(results['architectures'][spec['key']], results['config'])

    print_final_summary(results)


def main():
    run_experiment(verbose=True)


if __name__ == '__main__':
    main()
