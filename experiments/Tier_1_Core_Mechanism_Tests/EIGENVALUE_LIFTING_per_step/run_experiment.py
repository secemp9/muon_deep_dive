#!/usr/bin/env python3
"""
Experiment 3.16: Eigenvalue Lifting -- one-step λ_min^+ from matched SGD checkpoints
====================================================================================

This experiment measures a one-step curvature effect from matched SGD warmup
checkpoints in a tiny deep linear network where the full finite-difference
Hessian is tractable.

Question
--------
From the same SGD checkpoint, does one Muon step increase the smallest positive
Hessian eigenvalue (λ_min^+) more than one SGD step?

Important scope notes
---------------------
- The Hessian is computed by central finite differences on the gradient; it is
  not an analytic Hessian derivation.
- The Hessian is generally indefinite in this model, so the experiment tracks
  λ_min^+ (the smallest *positive* eigenvalue), not the algebraic minimum
  eigenvalue.
- The setup compares one-step interventions from SGD checkpoints. It does not,
  by itself, run a cumulative Muon trajectory decomposition.
"""

import time
import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

DIM = 4
NUM_LAYERS = 2
N_PARAMS = NUM_LAYERS * DIM * DIM  # 32
HESSIAN_EPS = 1e-5
DATA_POINTS = 32
MOMENTUM = 0.9
LR_SGD = 0.01
LR_MUON = 0.02
NS_ITERS = 5
WARMUP_MAX_STEPS = 1500
NUM_SEEDS = 5
POS_EIG_THRESHOLD = 1e-12
SEED_START = 42
SEED_STRIDE = 137

# Measurement points: step indices during warmup at which to snapshot
MEASUREMENT_STEPS = [50, 100, 150, 200, 300, 400, 600, 800, 1000, 1300]


# =============================================================================
# NETWORK UTILITIES
# =============================================================================


def init_weights(dim, num_layers, seed):
    """Initialize layers near identity."""
    rng = np.random.RandomState(seed)
    weights = []
    for _ in range(num_layers):
        W = np.eye(dim) + rng.randn(dim, dim) * 0.1
        weights.append(W.copy())
    return weights



def forward_linear(weights, X):
    out = X.copy()
    for W in weights:
        out = W @ out
    return out



def compute_loss(weights, X, Y_target):
    Y_pred = forward_linear(weights, X)
    diff = Y_pred - Y_target
    return 0.5 * np.mean(diff ** 2)



def compute_gradients(weights, X, Y_target):
    num_layers = len(weights)
    batch_size = X.shape[1]

    activations = [X.copy()]
    for W in weights:
        activations.append(W @ activations[-1])

    delta = (activations[-1] - Y_target) / batch_size

    grads = [None] * num_layers
    for l in range(num_layers - 1, -1, -1):
        grads[l] = delta @ activations[l].T
        if l > 0:
            delta = weights[l].T @ delta

    return grads


# =============================================================================
# NEWTON-SCHULZ ITERATION
# =============================================================================


def newton_schulz_orthogonalize(G, num_iters=5):
    """Quintic Newton-Schulz: a=3.4445, b=-4.7750, c=2.0315."""
    a, b, c = 3.4445, -4.7750, 2.0315
    norm = np.linalg.norm(G, 'fro')
    if norm < 1e-12:
        return G
    X = G / norm

    for _ in range(num_iters):
        XtX = X.T @ X
        X_XtX = X @ XtX
        XtX2 = XtX @ XtX
        X_XtX2 = X @ XtX2
        X = a * X + b * X_XtX + c * X_XtX2

    return X


# =============================================================================
# FULL HESSIAN COMPUTATION
# =============================================================================


def weights_to_vector(weights):
    return np.concatenate([W.flatten() for W in weights])



def vector_to_weights(vec, shapes):
    weights = []
    idx = 0
    for shape in shapes:
        size = shape[0] * shape[1]
        W = vec[idx:idx + size].reshape(shape)
        weights.append(W)
        idx += size
    return weights



def compute_gradient_vector(weights, X, Y_target):
    grads = compute_gradients(weights, X, Y_target)
    return np.concatenate([g.flatten() for g in grads])



def compute_full_hessian(weights, X, Y_target, eps=HESSIAN_EPS):
    """Full Hessian via central finite differences on the gradient."""
    shapes = [W.shape for W in weights]
    theta = weights_to_vector(weights)
    n = len(theta)

    H = np.zeros((n, n))
    for i in range(n):
        theta_p = theta.copy()
        theta_m = theta.copy()
        theta_p[i] += eps
        theta_m[i] -= eps

        g_p = compute_gradient_vector(vector_to_weights(theta_p, shapes), X, Y_target)
        g_m = compute_gradient_vector(vector_to_weights(theta_m, shapes), X, Y_target)

        H[:, i] = (g_p - g_m) / (2 * eps)

    H = 0.5 * (H + H.T)
    return H



def analyze_hessian(H, pos_eig_threshold=POS_EIG_THRESHOLD):
    """Compute key Hessian statistics, including signed-spectrum diagnostics."""
    eigenvalues = np.linalg.eigvalsh(H)
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]

    lambda_max = eigenvalues_sorted[0]
    lambda_min_raw = eigenvalues[0]
    trace_H = np.sum(eigenvalues)

    pos_mask = eigenvalues > pos_eig_threshold
    neg_mask = eigenvalues < -pos_eig_threshold
    n_positive = int(np.sum(pos_mask))
    n_negative = int(np.sum(neg_mask))
    n_near_zero = int(len(eigenvalues) - n_positive - n_negative)

    pos_eigs = eigenvalues[pos_mask]
    if len(pos_eigs) > 0:
        lambda_min_pos = np.min(pos_eigs)
    else:
        lambda_min_pos = None

    if lambda_min_pos is not None and lambda_min_pos > 1e-15:
        kappa = lambda_max / lambda_min_pos
    else:
        kappa = np.inf

    return {
        'eigenvalues': eigenvalues_sorted,
        'lambda_max': lambda_max,
        'lambda_min_raw': lambda_min_raw,
        'lambda_min_pos': lambda_min_pos,
        'trace': trace_H,
        'kappa': kappa,
        'n_positive': n_positive,
        'n_negative': n_negative,
        'n_near_zero': n_near_zero,
    }


# =============================================================================
# SINGLE-STEP COMPARISON
# =============================================================================


def one_step_sgd(weights, X, Y_target, lr, momentum_state=None):
    """Take one SGD+momentum step. Returns new weights."""
    grads = compute_gradients(weights, X, Y_target)
    new_weights = []
    for i in range(len(weights)):
        if momentum_state is not None:
            vel = MOMENTUM * momentum_state[i] + grads[i]
        else:
            vel = grads[i]
        W_new = weights[i] - lr * vel
        new_weights.append(W_new)
    return new_weights



def one_step_muon(weights, X, Y_target, lr, momentum_state=None):
    """Take one Muon step. Returns new weights."""
    grads = compute_gradients(weights, X, Y_target)
    new_weights = []
    for i in range(len(weights)):
        G_orth = newton_schulz_orthogonalize(grads[i], num_iters=NS_ITERS)
        if momentum_state is not None:
            vel = MOMENTUM * momentum_state[i] + G_orth
        else:
            vel = G_orth
        W_new = weights[i] - lr * vel
        new_weights.append(W_new)
    return new_weights


# =============================================================================
# WARMUP TRAINING (SGD to reach measurement points)
# =============================================================================


def warmup_sgd(weights, X, Y_target, max_steps, measurement_steps):
    """
    Train with SGD, collecting weight snapshots and momentum states
    at specified measurement steps.
    """
    velocities = [np.zeros_like(W) for W in weights]
    snapshots = {}

    for step in range(max_steps):
        if step in measurement_steps:
            snapshots[step] = {
                'weights': [W.copy() for W in weights],
                'velocities': [v.copy() for v in velocities],
                'loss': compute_loss(weights, X, Y_target),
            }

        grads = compute_gradients(weights, X, Y_target)
        for i in range(len(weights)):
            velocities[i] = MOMENTUM * velocities[i] + grads[i]
            weights[i] = weights[i] - LR_SGD * velocities[i]

    return snapshots


# =============================================================================
# RESULT HELPERS
# =============================================================================


def get_seed_list(num_seeds=NUM_SEEDS, seed_start=SEED_START, seed_stride=SEED_STRIDE):
    return [seed_start + s * seed_stride for s in range(num_seeds)]



def _finite_array(values):
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]



def summarize_numeric(values):
    arr = _finite_array(values)
    if arr.size == 0:
        return {
            'n': 0,
            'mean': np.nan,
            'median': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
        }
    return {
        'n': int(arr.size),
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
    }



def _compute_lift_ratio(lift_muon, lift_sgd):
    if np.isfinite(lift_muon) and np.isfinite(lift_sgd) and lift_sgd > 1e-15:
        return lift_muon / lift_sgd
    return np.nan



def _safe_ratio(numerator, denominator):
    if denominator is None:
        return np.nan
    try:
        denominator = float(denominator)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(denominator) or abs(denominator) <= 1e-15:
        return np.nan
    if numerator is None:
        return np.nan
    try:
        numerator = float(numerator)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(numerator):
        return np.nan
    return numerator / denominator



def _extract_state_stats(prefix, stats):
    return {
        f'lambda_min_pos_{prefix}': stats['lambda_min_pos'],
        f'lambda_min_raw_{prefix}': stats['lambda_min_raw'],
        f'lambda_max_{prefix}': stats['lambda_max'],
        f'trace_{prefix}': stats['trace'],
        f'kappa_{prefix}': stats['kappa'],
        f'n_negative_{prefix}': stats['n_negative'],
        f'n_positive_{prefix}': stats['n_positive'],
        f'n_near_zero_{prefix}': stats['n_near_zero'],
    }


# =============================================================================
# MAIN EXPERIMENT LOGIC
# =============================================================================


def run_single_seed(seed):
    """Run the full per-step λ_min^+ lifting test for one seed."""
    rng = np.random.RandomState(seed)

    W_target = [rng.randn(DIM, DIM) * 0.3 for _ in range(NUM_LAYERS)]
    X = rng.randn(DIM, DATA_POINTS) * 0.5
    Y_target = X.copy()
    for W in W_target:
        Y_target = W @ Y_target

    weights_init = init_weights(DIM, NUM_LAYERS, seed + 1000)
    snapshots = warmup_sgd(
        [W.copy() for W in weights_init], X, Y_target,
        max_steps=WARMUP_MAX_STEPS,
        measurement_steps=MEASUREMENT_STEPS,
    )

    rows = []
    for mstep in MEASUREMENT_STEPS:
        if mstep not in snapshots:
            continue

        snap = snapshots[mstep]
        w_snap = snap['weights']
        v_snap = snap['velocities']
        loss_at = snap['loss']

        H_before = compute_full_hessian(w_snap, X, Y_target)
        stats_before = analyze_hessian(H_before)

        w_after_sgd = one_step_sgd(w_snap, X, Y_target, LR_SGD, v_snap)
        H_after_sgd = compute_full_hessian(w_after_sgd, X, Y_target)
        stats_after_sgd = analyze_hessian(H_after_sgd)

        w_after_muon = one_step_muon(w_snap, X, Y_target, LR_MUON, v_snap)
        H_after_muon = compute_full_hessian(w_after_muon, X, Y_target)
        stats_after_muon = analyze_hessian(H_after_muon)

        lambda_min_pos_before = stats_before['lambda_min_pos']
        lambda_min_pos_after_sgd = stats_after_sgd['lambda_min_pos']
        lambda_min_pos_after_muon = stats_after_muon['lambda_min_pos']

        if lambda_min_pos_before is not None and lambda_min_pos_before > 1e-15:
            lift_sgd = lambda_min_pos_after_sgd / lambda_min_pos_before if lambda_min_pos_after_sgd is not None else 0.0
            lift_muon = lambda_min_pos_after_muon / lambda_min_pos_before if lambda_min_pos_after_muon is not None else 0.0
        else:
            lift_sgd = np.nan
            lift_muon = np.nan

        kappa_before = stats_before['kappa']
        kappa_after_sgd = stats_after_sgd['kappa']
        kappa_after_muon = stats_after_muon['kappa']
        kappa_ratio_sgd = _safe_ratio(kappa_after_sgd, kappa_before)
        kappa_ratio_muon = _safe_ratio(kappa_after_muon, kappa_before)

        trace_before = stats_before['trace']
        trace_after_sgd = stats_after_sgd['trace']
        trace_after_muon = stats_after_muon['trace']
        trace_ratio_sgd = _safe_ratio(trace_after_sgd, trace_before)
        trace_ratio_muon = _safe_ratio(trace_after_muon, trace_before)

        lambda_max_before = stats_before['lambda_max']
        lambda_max_after_sgd = stats_after_sgd['lambda_max']
        lambda_max_after_muon = stats_after_muon['lambda_max']
        lambda_max_ratio_sgd = _safe_ratio(lambda_max_after_sgd, lambda_max_before)
        lambda_max_ratio_muon = _safe_ratio(lambda_max_after_muon, lambda_max_before)

        muon_to_sgd_lift_ratio = _compute_lift_ratio(lift_muon, lift_sgd)

        row = {
            'seed': int(seed),
            'step': int(mstep),
            'loss': float(loss_at),
            **_extract_state_stats('before', stats_before),
            **_extract_state_stats('after_sgd', stats_after_sgd),
            **_extract_state_stats('after_muon', stats_after_muon),
            'lambda_min_pos_lift_sgd': lift_sgd,
            'lambda_min_pos_lift_muon': lift_muon,
            'muon_to_sgd_lift_ratio': muon_to_sgd_lift_ratio,
            'kappa_ratio_sgd': kappa_ratio_sgd,
            'kappa_ratio_muon': kappa_ratio_muon,
            'trace_ratio_sgd': trace_ratio_sgd,
            'trace_ratio_muon': trace_ratio_muon,
            'lambda_max_ratio_sgd': lambda_max_ratio_sgd,
            'lambda_max_ratio_muon': lambda_max_ratio_muon,
            # Legacy aliases for continuity with older labels.
            'lmin_before': lambda_min_pos_before,
            'lmin_sgd': lambda_min_pos_after_sgd,
            'lmin_muon': lambda_min_pos_after_muon,
            'lift_sgd': lift_sgd,
            'lift_muon': lift_muon,
            'kappa_before': kappa_before,
            'tr_ratio_sgd': trace_ratio_sgd,
            'tr_ratio_muon': trace_ratio_muon,
            'lmax_before': lambda_max_before,
            'lmax_sgd': lambda_max_after_sgd,
            'lmax_muon': lambda_max_after_muon,
        }
        rows.append(row)

    return rows



def aggregate_by_step(rows, measurement_steps=MEASUREMENT_STEPS):
    """Aggregate per-seed rows into per-step summaries."""
    summary_rows = []

    for mstep in measurement_steps:
        step_rows = [row for row in rows if row['step'] == mstep]
        if not step_rows:
            continue

        summary = {
            'step': int(mstep),
            'n_rows': int(len(step_rows)),
        }

        metric_names = [
            'loss',
            'lambda_min_pos_before',
            'lambda_min_pos_after_sgd',
            'lambda_min_pos_after_muon',
            'lambda_min_pos_lift_sgd',
            'lambda_min_pos_lift_muon',
            'muon_to_sgd_lift_ratio',
            'kappa_before',
            'kappa_after_sgd',
            'kappa_after_muon',
            'kappa_ratio_sgd',
            'kappa_ratio_muon',
            'trace_before',
            'trace_after_sgd',
            'trace_after_muon',
            'trace_ratio_sgd',
            'trace_ratio_muon',
            'lambda_max_before',
            'lambda_max_after_sgd',
            'lambda_max_after_muon',
            'lambda_max_ratio_sgd',
            'lambda_max_ratio_muon',
            'lambda_min_raw_before',
            'lambda_min_raw_after_sgd',
            'lambda_min_raw_after_muon',
            'n_negative_before',
            'n_negative_after_sgd',
            'n_negative_after_muon',
            'n_positive_before',
            'n_positive_after_sgd',
            'n_positive_after_muon',
            'n_near_zero_before',
            'n_near_zero_after_sgd',
            'n_near_zero_after_muon',
        ]

        for metric_name in metric_names:
            stats = summarize_numeric([row[metric_name] for row in step_rows])
            summary[f'{metric_name}_mean'] = stats['mean']
            summary[f'{metric_name}_median'] = stats['median']
            summary[f'{metric_name}_std'] = stats['std']

        lift_sgd_mean = summary['lambda_min_pos_lift_sgd_mean']
        lift_muon_mean = summary['lambda_min_pos_lift_muon_mean']
        if np.isfinite(lift_sgd_mean) and lift_sgd_mean > 1e-15:
            ratio_of_means = lift_muon_mean / lift_sgd_mean
        else:
            ratio_of_means = np.nan

        summary['muon_to_sgd_ratio_of_means'] = ratio_of_means
        summary['muon_better_count'] = int(sum(
            1 for row in step_rows
            if np.isfinite(row['muon_to_sgd_lift_ratio']) and row['muon_to_sgd_lift_ratio'] > 1.0
        ))
        summary['muon_better_fraction'] = summary['muon_better_count'] / len(step_rows)

        summary_rows.append(summary)

    return summary_rows



def evaluate_decision_rules(per_step_summary, rows):
    """Compute the legacy T1-T5 descriptive checks plus extra diagnostics."""
    valid_step_ratios = [
        row['muon_to_sgd_ratio_of_means']
        for row in per_step_summary
        if np.isfinite(row['muon_to_sgd_ratio_of_means'])
    ]

    muon_wins_count = sum(1 for ratio in valid_step_ratios if ratio > 1.0)
    total_points = len(valid_step_ratios)
    t1_pass = muon_wins_count > total_points * 0.5

    mean_ratio = float(np.mean(valid_step_ratios)) if valid_step_ratios else np.nan
    median_ratio = float(np.median(valid_step_ratios)) if valid_step_ratios else np.nan
    t2_pass = bool(mean_ratio > 3.0) if np.isfinite(mean_ratio) else False

    kappa_comparisons = [
        row for row in per_step_summary
        if np.isfinite(row['kappa_ratio_sgd_mean']) and np.isfinite(row['kappa_ratio_muon_mean'])
    ]
    kappa_muon_better = sum(1 for row in kappa_comparisons if row['kappa_ratio_muon_mean'] < row['kappa_ratio_sgd_mean'])
    kappa_total = len(kappa_comparisons)
    t3_pass = kappa_muon_better > kappa_total * 0.5

    if valid_step_ratios and abs(mean_ratio) > 1e-15:
        cv = float(np.std(valid_step_ratios) / mean_ratio)
        t4_pass = cv < 1.0
    else:
        cv = np.nan
        t4_pass = False

    late_step_ratios = [
        row['muon_to_sgd_ratio_of_means']
        for row in per_step_summary
        if row['step'] >= 600 and np.isfinite(row['muon_to_sgd_ratio_of_means'])
    ]
    mean_late = float(np.mean(late_step_ratios)) if late_step_ratios else np.nan
    t5_pass = bool(mean_late > 1.0) if np.isfinite(mean_late) else False

    pairwise_ratios = [row['muon_to_sgd_lift_ratio'] for row in rows if np.isfinite(row['muon_to_sgd_lift_ratio'])]
    pairwise_muon_better_count = sum(1 for ratio in pairwise_ratios if ratio > 1.0)
    positive_pairwise_ratios = [ratio for ratio in pairwise_ratios if ratio > 0]
    geometric_mean_pairwise_ratio = (
        float(np.exp(np.mean(np.log(positive_pairwise_ratios))))
        if positive_pairwise_ratios else np.nan
    )

    step_100_rows = [row for row in rows if row['step'] == 100 and np.isfinite(row['muon_to_sgd_lift_ratio'])]
    step_100_muon_better_count = sum(1 for row in step_100_rows if row['muon_to_sgd_lift_ratio'] > 1.0)

    return {
        'T1_majority_steps_muon_better': {
            'description': 'Muon/SGD ratio of step means > 1 at a majority of measurement steps.',
            'pass': bool(t1_pass),
            'muon_wins_count': int(muon_wins_count),
            'total_points': int(total_points),
        },
        'T2_mean_step_ratio_gt_3': {
            'description': 'Mean of per-step Muon/SGD ratios of step means exceeds 3.',
            'pass': bool(t2_pass),
            'mean_ratio': mean_ratio,
            'median_ratio': median_ratio,
        },
        'T3_majority_steps_kappa_better': {
            'description': 'Muon kappa ratio is smaller than SGD at a majority of measurement steps.',
            'pass': bool(t3_pass),
            'muon_better_count': int(kappa_muon_better),
            'total_points': int(kappa_total),
        },
        'T4_step_ratio_cv_lt_1': {
            'description': 'Coefficient of variation of per-step Muon/SGD ratios is below 1.',
            'pass': bool(t4_pass),
            'cv': cv,
        },
        'T5_late_steps_mean_ratio_gt_1': {
            'description': 'Mean Muon/SGD ratio of step means is > 1 for late checkpoints (step >= 600).',
            'pass': bool(t5_pass),
            'mean_late_ratio': mean_late,
        },
        'pairwise_ratio_summary': {
            'description': 'Per-seed/per-step Muon-to-SGD lift ratio summary.',
            'valid_pairwise_count': int(len(pairwise_ratios)),
            'muon_better_count': int(pairwise_muon_better_count),
            'muon_better_fraction': (
                pairwise_muon_better_count / len(pairwise_ratios) if pairwise_ratios else np.nan
            ),
            'mean_ratio': float(np.mean(pairwise_ratios)) if pairwise_ratios else np.nan,
            'median_ratio': float(np.median(pairwise_ratios)) if pairwise_ratios else np.nan,
            'geometric_mean_ratio': geometric_mean_pairwise_ratio,
        },
        'step_100_reversal_check': {
            'description': 'Muon-vs-SGD result at the representative checkpoint step 100.',
            'step': 100,
            'muon_better_count': int(step_100_muon_better_count),
            'total_points': int(len(step_100_rows)),
        },
    }



def run_experiment(seed_list=None, progress=False):
    """Run the full experiment and return structured results."""
    if seed_list is None:
        seed_list = get_seed_list()

    start_time = time.time()
    per_seed_results = []
    rows = []

    for seed_index, seed in enumerate(seed_list, start=1):
        if progress:
            print(f"  Running seed {seed_index}/{len(seed_list)} (seed={seed})...", flush=True)
        seed_rows = run_single_seed(seed)
        per_seed_results.append({
            'seed': int(seed),
            'rows': seed_rows,
        })
        rows.extend(seed_rows)

    per_step_summary = aggregate_by_step(rows, MEASUREMENT_STEPS)
    decision_rules = evaluate_decision_rules(per_step_summary, rows)

    runtime_seconds = time.time() - start_time
    results = {
        'identity': {
            'experiment': 'Experiment 3.16',
            'title': 'Eigenvalue Lifting -- one-step λ_min^+ from matched SGD checkpoints',
            'script_path': __file__,
            'question': 'From the same SGD checkpoint, does one Muon step lift λ_min^+ more than one SGD step?',
            'scope_note': 'One-step interventions from SGD checkpoints; not a cumulative Muon trajectory experiment.',
        },
        'config': {
            'dim': DIM,
            'num_layers': NUM_LAYERS,
            'n_params': N_PARAMS,
            'hessian_eps': HESSIAN_EPS,
            'data_points': DATA_POINTS,
            'momentum': MOMENTUM,
            'lr_sgd': LR_SGD,
            'lr_muon': LR_MUON,
            'ns_iters': NS_ITERS,
            'warmup_max_steps': WARMUP_MAX_STEPS,
            'num_seeds': len(seed_list),
            'measurement_steps': list(MEASUREMENT_STEPS),
            'pos_eig_threshold': POS_EIG_THRESHOLD,
            'seed_start': SEED_START,
            'seed_stride': SEED_STRIDE,
            'hessian_method': 'full central finite-difference Hessian on the gradient',
            'reported_floor_metric': 'lambda_min_pos (smallest positive Hessian eigenvalue)',
        },
        'seed_list': list(seed_list),
        'per_seed_results': per_seed_results,
        'rows': rows,
        'per_step_summary': per_step_summary,
        'decision_rules': decision_rules,
        'runtime_seconds': runtime_seconds,
    }
    return results


# =============================================================================
# REPORTING
# =============================================================================


def _fmt_num(value, width=10, precision=4, scientific=False):
    if value is None:
        return f"{'None':>{width}}"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return f"{str(value):>{width}}"
    if np.isnan(value):
        return f"{'nan':>{width}}"
    if np.isposinf(value):
        return f"{'inf':>{width}}"
    if np.isneginf(value):
        return f"{'-inf':>{width}}"
    if scientific:
        return f"{value:>{width}.{precision}e}"
    return f"{value:>{width}.{precision}f}"



def print_report(results):
    config = results['config']
    per_step_summary = results['per_step_summary']
    decision_rules = results['decision_rules']
    seed_list = results['seed_list']
    rows = results['rows']

    print()
    print("=" * 110)
    print("  Experiment 3.16: Eigenvalue Lifting -- one-step λ_min^+ from matched SGD checkpoints")
    print("=" * 110)
    print()
    print("  QUESTION: From the same SGD checkpoint, does one Muon step lift λ_min^+ more than one SGD step?")
    print("  SCOPE: one-step interventions from SGD warmup checkpoints only; not a cumulative Muon trajectory study.")
    print("  HESSIAN: full central finite-difference Hessian on the gradient (indefinite Hessians are allowed).")
    print()
    print(f"  Config: {config['num_layers']}-layer {config['dim']}x{config['dim']} deep linear net ({config['n_params']} params)")
    print(f"  LR_SGD={config['lr_sgd']}, LR_MUON={config['lr_muon']}, momentum={config['momentum']}, NS_iters={config['ns_iters']}")
    print(f"  Measurement points: {config['measurement_steps']}")
    print(f"  Seeds ({len(seed_list)}): {seed_list}")
    print(f"  Runtime: {_fmt_num(results['runtime_seconds'], width=8, precision=2)} s")
    print()

    print("=" * 110)
    print("  PER-STEP λ_min^+ LIFTING RESULTS (averaged over seeds)")
    print("=" * 110)
    print()
    print(
        f"  {'Step':>6} {'Loss':>10} {'lmin+_bef':>10} {'lift_SGD':>10} {'lift_Muon':>10} "
        f"{'Muon/SGD':>10} {'kR_SGD':>10} {'kR_Muon':>10} {'neg_bef':>8}"
    )
    print(
        f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}"
    )

    for row in per_step_summary:
        print(
            f"  {row['step']:>6} "
            f"{_fmt_num(row['loss_mean'], width=10, precision=4, scientific=True)} "
            f"{_fmt_num(row['lambda_min_pos_before_mean'], width=10, precision=4, scientific=True)} "
            f"{_fmt_num(row['lambda_min_pos_lift_sgd_mean'], width=10, precision=4)} "
            f"{_fmt_num(row['lambda_min_pos_lift_muon_mean'], width=10, precision=4)} "
            f"{_fmt_num(row['muon_to_sgd_ratio_of_means'], width=10, precision=4)} "
            f"{_fmt_num(row['kappa_ratio_sgd_mean'], width=10, precision=4)} "
            f"{_fmt_num(row['kappa_ratio_muon_mean'], width=10, precision=4)} "
            f"{_fmt_num(row['n_negative_before_mean'], width=8, precision=2)}"
        )

    print()
    print("  Legend:")
    print("    lift_SGD/Muon = λ_min^+(H_after) / λ_min^+(H_before)  [>1 = positive-floor lifting]")
    print("    Muon/SGD      = lift_Muon / lift_SGD using per-step means  [>1 = Muon lifts more]")
    print("    kR            = κ(H_after) / κ(H_before)  [<1 = conditioning improves]")
    print("    neg_bef       = mean number of negative Hessian eigenvalues before the one-step intervention")

    rep_step = 100
    print()
    print("=" * 110)
    print(f"  DETAILED PER-SEED RESULTS AT STEP {rep_step}")
    print("=" * 110)
    print()
    print(
        f"  {'Seed':>6} {'Loss':>10} {'lmin+_bef':>10} {'lmin+_SGD':>10} {'lmin+_Muon':>11} "
        f"{'lift_SGD':>10} {'lift_Muon':>10} {'Muon>SGD?':>10}"
    )
    print(
        f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*11} {'-'*10} {'-'*10} {'-'*10}"
    )
    for row in rows:
        if row['step'] == rep_step:
            muon_wins = 'YES' if row['muon_to_sgd_lift_ratio'] > 1.0 else 'NO'
            print(
                f"  {row['seed']:>6} "
                f"{_fmt_num(row['loss'], width=10, precision=4, scientific=True)} "
                f"{_fmt_num(row['lambda_min_pos_before'], width=10, precision=4, scientific=True)} "
                f"{_fmt_num(row['lambda_min_pos_after_sgd'], width=10, precision=4, scientific=True)} "
                f"{_fmt_num(row['lambda_min_pos_after_muon'], width=11, precision=4, scientific=True)} "
                f"{_fmt_num(row['lambda_min_pos_lift_sgd'], width=10, precision=4)} "
                f"{_fmt_num(row['lambda_min_pos_lift_muon'], width=10, precision=4)} "
                f"{muon_wins:>10}"
            )

    print()
    print("=" * 110)
    print("  SIGNED-SPECTRUM DIAGNOSTICS (before one-step intervention)")
    print("=" * 110)
    print()
    print(f"  {'Step':>6} {'λ_min(raw)':>12} {'neg eigs':>10} {'pos eigs':>10} {'near-zero':>10}")
    print(f"  {'-'*6} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for row in per_step_summary:
        print(
            f"  {row['step']:>6} "
            f"{_fmt_num(row['lambda_min_raw_before_mean'], width=12, precision=4, scientific=True)} "
            f"{_fmt_num(row['n_negative_before_mean'], width=10, precision=2)} "
            f"{_fmt_num(row['n_positive_before_mean'], width=10, precision=2)} "
            f"{_fmt_num(row['n_near_zero_before_mean'], width=10, precision=2)}"
        )

    print()
    print("=" * 110)
    print("  λ_max (sharpness) PER-STEP CHANGE")
    print("=" * 110)
    print()
    print(
        f"  {'Step':>6} {'lmax_bef':>12} {'lmax_SGD':>12} {'lmax_Muon':>12} {'SGD ratio':>12} {'Muon ratio':>12}"
    )
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    for row in per_step_summary:
        print(
            f"  {row['step']:>6} "
            f"{_fmt_num(row['lambda_max_before_mean'], width=12, precision=4, scientific=True)} "
            f"{_fmt_num(row['lambda_max_after_sgd_mean'], width=12, precision=4, scientific=True)} "
            f"{_fmt_num(row['lambda_max_after_muon_mean'], width=12, precision=4, scientific=True)} "
            f"{_fmt_num(row['lambda_max_ratio_sgd_mean'], width=12, precision=4)} "
            f"{_fmt_num(row['lambda_max_ratio_muon_mean'], width=12, precision=4)}"
        )

    print()
    print("=" * 110)
    print("  LEGACY T1-T5 DECISION RULES (descriptive, not inferential tests)")
    print("=" * 110)
    print()

    t1 = decision_rules['T1_majority_steps_muon_better']
    print("  T1: Muon/SGD ratio of step means > 1 at a majority of measurement steps?")
    print(f"      Points where Muon/SGD > 1: {t1['muon_wins_count']}/{t1['total_points']}")
    print(f"      {'PASS' if t1['pass'] else 'FAIL'}")
    print()

    t2 = decision_rules['T2_mean_step_ratio_gt_3']
    print("  T2: Mean per-step Muon/SGD ratio of step means > 3?")
    print(f"      Mean lift_Muon/lift_SGD = {_fmt_num(t2['mean_ratio'], width=10, precision=4)}")
    print(f"      Median                  = {_fmt_num(t2['median_ratio'], width=10, precision=4)}")
    print(f"      {'PASS' if t2['pass'] else 'FAIL'}")
    print()

    t3 = decision_rules['T3_majority_steps_kappa_better']
    print("  T3: Muon reduces κ more than SGD at a majority of measurement steps?")
    print(f"      Points where Muon κ-ratio < SGD κ-ratio: {t3['muon_better_count']}/{t3['total_points']}")
    print(f"      {'PASS' if t3['pass'] else 'FAIL'}")
    print()

    t4 = decision_rules['T4_step_ratio_cv_lt_1']
    print("  T4: Stepwise Muon/SGD ratio has CV < 1.0?")
    print(f"      Coefficient of variation = {_fmt_num(t4['cv'], width=10, precision=4)}")
    print(f"      {'PASS' if t4['pass'] else 'FAIL'}")
    print()

    t5 = decision_rules['T5_late_steps_mean_ratio_gt_1']
    print("  T5: Mean per-step Muon/SGD ratio remains > 1 at late checkpoints (step >= 600)?")
    print(f"      Mean late-step ratio = {_fmt_num(t5['mean_late_ratio'], width=10, precision=4)}")
    print(f"      {'PASS' if t5['pass'] else 'FAIL'}")
    print()

    pairwise = decision_rules['pairwise_ratio_summary']
    print("  Pairwise seed-step ratio summary:")
    print(f"      Muon better on {pairwise['muon_better_count']}/{pairwise['valid_pairwise_count']} valid seed-step pairs")
    print(f"      Pairwise mean ratio      = {_fmt_num(pairwise['mean_ratio'], width=10, precision=4)}")
    print(f"      Pairwise median ratio    = {_fmt_num(pairwise['median_ratio'], width=10, precision=4)}")
    print(f"      Pairwise geometric mean  = {_fmt_num(pairwise['geometric_mean_ratio'], width=10, precision=4)}")
    print()

    reversal = decision_rules['step_100_reversal_check']
    print("  Step-100 reversal check:")
    print(f"      Muon better count at step 100 = {reversal['muon_better_count']}/{reversal['total_points']}")
    print()

    print("=" * 110)
    print("  INTERPRETATION")
    print("=" * 110)
    print()
    if t2['pass']:
        print("  Supported: Muon often produces much larger one-step increases in λ_min^+ than SGD,")
        print(f"             with a legacy mean per-step Muon/SGD ratio of {t2['mean_ratio']:.2f}.")
    elif t1['pass']:
        print("  Supported more weakly: Muon beats SGD on the legacy stepwise ratio at most checkpoints,")
        print(f"                        but the mean ratio is only {t2['mean_ratio']:.2f}.")
    else:
        print("  Not supported on the legacy stepwise ratio: Muon does not beat SGD at most checkpoints.")

    print("  Important caveats:")
    print("    - The tracked floor is λ_min^+ (smallest positive eigenvalue), not the algebraic minimum eigenvalue.")
    print("    - Hessians are typically indefinite, as shown by the negative-eigenvalue counts above.")
    print("    - Step 100 is a complete reversal in the default run, so the effect is not uniform.")
    print("    - LR_MUON and LR_SGD differ, and Muon reuses the SGD momentum buffer from the warmup trajectory.")
    print("    - This is a one-step intervention study in a toy deep linear model, not a full causal decomposition of cumulative behavior.")
    print()
    print("=" * 110)
    print("  EXPERIMENT COMPLETE")
    print("=" * 110)



def main():
    results = run_experiment(progress=True)
    print_report(results)


if __name__ == '__main__':
    main()
