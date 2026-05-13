#!/usr/bin/env python3
"""
Experiment H6: LR Artifact Check
================================

Scope
-----
This is a toy final-training-loss benchmark on the same 2-layer deep-linear
regression problem used in Experiment 3.4. It does not directly measure
curvature, generalization, or a universal optimizer advantage.

The narrower question addressed here is whether the large final-loss gap
between vanilla Muon at lr=0.02 and curvature-rescaled Muon at lr=0.02 can be
largely explained by learning-rate choice in this setup.
"""

from pathlib import Path
from time import perf_counter

import numpy as np


EXPERIMENT_RELATIVE_PATH = (
    'experiments/Tier_1_Core_Mechanism_Tests/H6_LR_ARTIFACT_CHECK/run_experiment.py'
)
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
NOTEBOOK_PATH = SCRIPT_DIR / 'run_experiment.ipynb'

# =============================================================================
# CONFIGURATION
# =============================================================================

DIM = 4
NUM_LAYERS = 2
NUM_STEPS = 500
NS_ITERS = 5
MOMENTUM = 0.9
GAMMA = 1.0
SCALE_MIN = 0.1
SCALE_MAX = 10.0
NUM_SEEDS = 10
DATA_POINTS = 32

# LR sweep values
VANILLA_LRS = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
SGD_LRS = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
ORIGINAL_LR = 0.02  # the 3.4 default reference point

SCOPE_NOTE = (
    'Reports final training loss after a fixed 500-step budget on a toy '
    'deep-linear regression problem. This benchmark does not directly measure '
    'curvature or establish a general optimizer advantage.'
)


# =============================================================================
# NETWORK UTILITIES (same mathematical setup as 3.4)
# =============================================================================


def clone_weights(weights):
    """Return a defensive copy of a list of weight matrices."""
    return [W.copy() for W in weights]



def init_weights(dim, num_layers, seed):
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
    """Mean-squared error loss."""
    Y_pred = forward_linear(weights, X)
    diff = Y_pred - Y_target
    return 0.5 * np.mean(diff ** 2)



def compute_gradients(weights, X, Y_target):
    """Backpropagation through the deep linear net."""
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
# NEWTON-SCHULZ ITERATION (same polynomial as 3.4)
# =============================================================================


def newton_schulz_orthogonalize(G, num_iters=NS_ITERS):
    """Muon's quintic Newton-Schulz iteration."""
    a, b, c = 3.4445, -4.7750, 2.0315
    norm = np.linalg.norm(G, 'fro')
    if norm < 1e-12:
        return G
    X = G / norm

    for _ in range(num_iters):
        XtX = X.T @ X
        X = a * X + b * (X @ XtX) + c * (X @ (XtX @ XtX))

    return X


# =============================================================================
# OPTIMIZERS
# =============================================================================


def _pad_list(values, target_len, pad_value):
    values = list(values)
    if len(values) < target_len:
        values.extend([pad_value] * (target_len - len(values)))
    return values



def train_muon(
    weights,
    X,
    Y_target,
    lr,
    num_steps,
    ns_iters=NS_ITERS,
    rescale_mode='none',
    gamma=1.0,
    scale_min=0.1,
    scale_max=10.0,
    momentum=0.9,
):
    """
    Muon optimizer with optional curvature rescaling.
    Returns (loss_history, scale_history, final_weights).
    """
    weights = clone_weights(weights)
    num_layers = len(weights)
    velocities = [np.zeros_like(W) for W in weights]
    losses = []
    scales_used = []

    for _step in range(num_steps):
        loss = float(compute_loss(weights, X, Y_target))
        losses.append(loss)

        if not np.isfinite(loss) or loss > 1e10:
            losses = _pad_list(losses, num_steps + 1, float('inf'))
            scales_used = _pad_list(scales_used, num_steps, 1.0)
            return losses, scales_used, weights

        grads = compute_gradients(weights, X, Y_target)

        step_scales = []
        for i in range(num_layers):
            G = grads[i]
            G_norm = np.linalg.norm(G, 'fro')

            G_orth = newton_schulz_orthogonalize(G, num_iters=ns_iters)
            G_orth_norm = np.linalg.norm(G_orth, 'fro')

            if rescale_mode == 'curvature':
                if G_orth_norm > 1e-12:
                    scale = np.clip(G_norm / G_orth_norm * gamma, scale_min, scale_max)
                else:
                    scale = 1.0
                G_orth = G_orth * scale
            else:
                scale = 1.0

            step_scales.append(float(scale))
            velocities[i] = momentum * velocities[i] + G_orth
            weights[i] = weights[i] - lr * velocities[i]

        scales_used.append(float(np.mean(step_scales)))

    final_loss = float(compute_loss(weights, X, Y_target))
    losses.append(final_loss)
    return losses, scales_used, weights



def train_sgd(weights, X, Y_target, lr, num_steps, momentum=0.9):
    """SGD with momentum."""
    weights = clone_weights(weights)
    num_layers = len(weights)
    velocities = [np.zeros_like(W) for W in weights]
    losses = []

    for _step in range(num_steps):
        loss = float(compute_loss(weights, X, Y_target))
        losses.append(loss)

        if not np.isfinite(loss) or loss > 1e10:
            losses = _pad_list(losses, num_steps + 1, float('inf'))
            return losses, weights

        grads = compute_gradients(weights, X, Y_target)

        for i in range(num_layers):
            velocities[i] = momentum * velocities[i] + grads[i]
            weights[i] = weights[i] - lr * velocities[i]

    final_loss = float(compute_loss(weights, X, Y_target))
    losses.append(final_loss)
    return losses, weights


# =============================================================================
# DATA GENERATION (same seed scheme as 3.4)
# =============================================================================


def make_problem(seed):
    """Generate target and data for a single seed."""
    rng = np.random.RandomState(seed)
    W_target = [rng.randn(DIM, DIM) * 0.3 for _ in range(NUM_LAYERS)]
    X = rng.randn(DIM, DATA_POINTS) * 0.5
    Y_target = X.copy()
    for W in W_target:
        Y_target = W @ Y_target
    return X, Y_target


# =============================================================================
# SUMMARIES AND STRUCTURED RESULTS
# =============================================================================


def summarize_losses(final_losses):
    """Summary statistics for a vector of final losses."""
    arr = np.asarray(final_losses, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            'count': int(arr.size),
            'finite_count': 0,
            'finite_fraction': 0.0,
            'mean': float('inf'),
            'median': float('inf'),
            'std': float('nan'),
            'min': float('inf'),
            'max': float('inf'),
        }

    return {
        'count': int(arr.size),
        'finite_count': int(finite.size),
        'finite_fraction': float(finite.size / arr.size),
        'mean': float(np.mean(finite)),
        'median': float(np.median(finite)),
        'std': float(np.std(finite)),
        'min': float(np.min(finite)),
        'max': float(np.max(finite)),
    }



def summarize_scales(scales, scale_min=SCALE_MIN, scale_max=SCALE_MAX):
    """Summary statistics for flattened scale histories."""
    arr = np.asarray(scales, dtype=float)
    if arr.size == 0:
        return {
            'count': 0,
            'mean': float('nan'),
            'median': float('nan'),
            'std': float('nan'),
            'min': float('nan'),
            'max': float('nan'),
            'min_clamp_fraction': float('nan'),
            'max_clamp_fraction': float('nan'),
        }

    return {
        'count': int(arr.size),
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'min_clamp_fraction': float(np.mean(arr <= scale_min + 1e-8)),
        'max_clamp_fraction': float(np.mean(arr >= scale_max - 1e-8)),
    }



def stable_ratio(numerator, denominator):
    """Return numerator/denominator when that ratio is numerically meaningful."""
    numerator = float(numerator)
    denominator = float(denominator)
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return float('nan')
    if abs(denominator) < 1e-300:
        return float('inf') if numerator > 0 else float('nan')
    return float(numerator / denominator)



def stable_log10_ratio(numerator, denominator):
    """Return log10(numerator/denominator) for positive finite ratios."""
    ratio = stable_ratio(numerator, denominator)
    if np.isfinite(ratio) and ratio > 0:
        return float(np.log10(ratio))
    return float('nan')



def add_display_name(condition, display_name):
    """Shallow-copy a condition dictionary and attach a display name."""
    enriched = dict(condition)
    enriched['display_name'] = display_name
    return enriched



def select_best_tested_lr(results_by_lr, lr_grid):
    """Choose the best tested LR by mean finite final loss."""
    best_lr = None
    best_mean = float('inf')

    for lr in lr_grid:
        lr = float(lr)
        summary = results_by_lr[lr]['summary']
        mean_loss = summary['mean']
        if np.isfinite(mean_loss) and mean_loss < best_mean:
            best_mean = mean_loss
            best_lr = lr

    if best_lr is None:
        raise RuntimeError('No finite losses found in LR sweep.')

    min_lr = float(min(lr_grid))
    max_lr = float(max(lr_grid))
    return {
        'lr': float(best_lr),
        'mean': float(best_mean),
        'on_lower_boundary': bool(np.isclose(best_lr, min_lr)),
        'on_upper_boundary': bool(np.isclose(best_lr, max_lr)),
    }



def run_muon_condition(seeds, lr, rescale_mode='none'):
    """Run a Muon condition across seeds and retain histories."""
    final_losses = []
    loss_histories = []
    scale_histories = []

    for seed in seeds:
        X, Y_target = make_problem(seed)
        w_init = init_weights(DIM, NUM_LAYERS, seed + 1000)
        losses, scales, _ = train_muon(
            w_init,
            X,
            Y_target,
            lr=lr,
            num_steps=NUM_STEPS,
            ns_iters=NS_ITERS,
            rescale_mode=rescale_mode,
            gamma=GAMMA,
            scale_min=SCALE_MIN,
            scale_max=SCALE_MAX,
            momentum=MOMENTUM,
        )
        final_losses.append(float(losses[-1]))
        loss_histories.append(np.asarray(losses, dtype=float))
        if rescale_mode == 'curvature':
            scale_histories.append(np.asarray(scales, dtype=float))

    condition = {
        'method': 'Muon',
        'rescale_mode': rescale_mode,
        'lr': float(lr),
        'final_losses': np.asarray(final_losses, dtype=float),
        'loss_histories': np.asarray(loss_histories, dtype=float),
        'summary': summarize_losses(final_losses),
    }

    if rescale_mode == 'curvature':
        scales = np.asarray(scale_histories, dtype=float)
        flat_scales = scales.reshape(-1) if scales.size else np.asarray([], dtype=float)
        condition['scale_histories'] = scales
        condition['flattened_scales'] = flat_scales
        condition['scale_summary'] = summarize_scales(flat_scales)

    return condition



def run_sgd_condition(seeds, lr):
    """Run an SGD condition across seeds and retain histories."""
    final_losses = []
    loss_histories = []

    for seed in seeds:
        X, Y_target = make_problem(seed)
        w_init = init_weights(DIM, NUM_LAYERS, seed + 1000)
        losses, _ = train_sgd(
            w_init,
            X,
            Y_target,
            lr=lr,
            num_steps=NUM_STEPS,
            momentum=MOMENTUM,
        )
        final_losses.append(float(losses[-1]))
        loss_histories.append(np.asarray(losses, dtype=float))

    return {
        'method': 'SGD',
        'lr': float(lr),
        'final_losses': np.asarray(final_losses, dtype=float),
        'loss_histories': np.asarray(loss_histories, dtype=float),
        'summary': summarize_losses(final_losses),
    }



def build_tests(best_vanilla_lr, vanilla_best, vanilla_original, rescaled_original, rescaled_best):
    """Compute the current T1/T2/T3 operational checks."""
    expected_lr = ORIGINAL_LR * SCALE_MIN
    vanilla_best_mean = vanilla_best['summary']['mean']
    vanilla_orig_mean = vanilla_original['summary']['mean']
    rescaled_orig_mean = rescaled_original['summary']['mean']
    rescaled_best_mean = rescaled_best['summary']['mean']

    t1_ratio = stable_ratio(best_vanilla_lr, expected_lr)
    t1_pass = bool(np.isfinite(t1_ratio) and 0.5 <= t1_ratio <= 2.0)
    t1_exact = bool(abs(best_vanilla_lr - expected_lr) < 1e-12)

    t2_ratio = stable_ratio(vanilla_best_mean, rescaled_orig_mean)
    t2_log10_ratio = stable_log10_ratio(vanilla_best_mean, rescaled_orig_mean)
    t2_parity_gap_pct = float(abs(t2_ratio - 1.0) * 100.0) if np.isfinite(t2_ratio) else float('inf')
    t2_pass = bool(np.isfinite(t2_parity_gap_pct) and t2_parity_gap_pct < 5.0)

    t3_ratio = stable_ratio(vanilla_best_mean, rescaled_best_mean)
    t3_log10_ratio = stable_log10_ratio(vanilla_best_mean, rescaled_best_mean)
    t3_rescaled_over_vanilla = stable_ratio(rescaled_best_mean, vanilla_best_mean)
    t3_pass = bool(np.isfinite(rescaled_best_mean) and rescaled_best_mean < 0.95 * vanilla_best_mean)

    return {
        'T1': {
            'label': 'T1',
            'question': 'Is the best tested vanilla Muon LR near 0.02 x 0.1?',
            'definition': 'Pass if best_tested_lr / expected_lr is within [0.5, 2.0].',
            'expected_lr': float(expected_lr),
            'best_tested_lr': float(best_vanilla_lr),
            'ratio_best_to_expected': float(t1_ratio),
            'exact_match': bool(t1_exact),
            'pass': bool(t1_pass),
        },
        'T2': {
            'label': 'T2',
            'question': 'Does vanilla Muon at the best tested LR match rescaled Muon at lr=0.02 within 5%?',
            'definition': 'Pass if |vanilla_best / rescaled_original - 1| < 0.05.',
            'vanilla_best_mean': float(vanilla_best_mean),
            'rescaled_original_mean': float(rescaled_orig_mean),
            'vanilla_best_over_rescaled_original_ratio': float(t2_ratio),
            'log10_vanilla_best_over_rescaled_original_ratio': float(t2_log10_ratio),
            'parity_gap_percent': float(t2_parity_gap_pct),
            'pass': bool(t2_pass),
        },
        'T3': {
            'label': 'T3',
            'question': 'Does rescaled Muon at the vanilla best tested LR improve on vanilla Muon there?',
            'definition': 'Pass if rescaled_best_mean < 0.95 * vanilla_best_mean.',
            'vanilla_best_mean': float(vanilla_best_mean),
            'rescaled_best_mean': float(rescaled_best_mean),
            'vanilla_best_over_rescaled_best_ratio': float(t3_ratio),
            'log10_vanilla_best_over_rescaled_best_ratio': float(t3_log10_ratio),
            'rescaled_best_over_vanilla_best_ratio': float(t3_rescaled_over_vanilla),
            'pass': bool(t3_pass),
        },
        'reference_values': {
            'vanilla_original_mean': float(vanilla_orig_mean),
            'vanilla_best_mean': float(vanilla_best_mean),
            'rescaled_original_mean': float(rescaled_orig_mean),
            'rescaled_best_mean': float(rescaled_best_mean),
        },
    }



def compute_seedwise_winners(conditions, condition_order, seeds):
    """Determine per-seed winners across selected conditions."""
    winner_counts = {key: 0 for key in condition_order}
    rows = []

    for i, seed in enumerate(seeds):
        losses = {key: float(conditions[key]['final_losses'][i]) for key in condition_order}
        winner_key = min(
            condition_order,
            key=lambda key: losses[key] if np.isfinite(losses[key]) else float('inf'),
        )
        winner_counts[winner_key] += 1
        rows.append(
            {
                'seed': int(seed),
                'init_seed': int(seed + 1000),
                'winner_key': winner_key,
                'winner_display_name': conditions[winner_key]['display_name'],
                'losses': losses,
            }
        )

    return rows, winner_counts



def classify_verdict(tests, vanilla_selection, sgd_selection, comparisons):
    """Assign a calibrated overall conclusion for the current benchmark."""
    t1 = tests['T1']
    t2 = tests['T2']
    t3 = tests['T3']

    default_vs_rescaled = comparisons['ratios']['vanilla_original_over_rescaled_original']
    default_vs_vanilla_best = comparisons['ratios']['vanilla_original_over_vanilla_best']
    vanilla_best_mean = tests['reference_values']['vanilla_best_mean']
    rescaled_orig_mean = tests['reference_values']['rescaled_original_mean']
    rescaled_best_mean = tests['reference_values']['rescaled_best_mean']

    boundary_notes = []
    if vanilla_selection['on_lower_boundary'] or vanilla_selection['on_upper_boundary']:
        boundary_notes.append(
            'Vanilla Muon achieves its best tested LR on the sweep boundary, so this '
            'run identifies only a best tested LR, not a localized optimum.'
        )
    if sgd_selection['on_lower_boundary'] or sgd_selection['on_upper_boundary']:
        boundary_notes.append(
            'SGD also achieves its best tested LR on the sweep boundary, so its '
            'reported best LR is likewise boundary-limited.'
        )

    if t1['pass'] and t2['pass'] and not t3['pass']:
        category = 'default-lr artifact'
        summary = (
            'Within the tested grid, the rescaled-vs-vanilla gap at lr=0.02 is '
            'consistent with a learning-rate artifact, and rescaling does not add '
            'further value at the vanilla best tested LR.'
        )
        bullets = [
            f"Vanilla Muon improves over its lr=0.02 baseline by {default_vs_vanilla_best:.1f}x when moved to the best tested LR.",
            f"Vanilla(best tested) / Rescaled(0.02) = {t2['vanilla_best_over_rescaled_original_ratio']:.4f}.",
            'Applying rescaling at the vanilla best tested LR does not produce a >5% gain.',
        ]
    elif (not t3['pass']) and np.isfinite(vanilla_best_mean) and vanilla_best_mean < rescaled_orig_mean:
        category = 'tuned vanilla beats rescaling'
        summary = (
            'A narrower default-lr-artifact story is too weak: tuned vanilla Muon '
            'already beats rescaled Muon at lr=0.02, and adding rescaling at that '
            'best tested vanilla LR makes the final loss worse.'
        )
        bullets = [
            f"Vanilla(0.02) / Rescaled(0.02) = {default_vs_rescaled:.1f}x, so the original default-LR comparison is still highly sensitive to LR choice.",
            f"Vanilla(0.02) / Vanilla(best tested) = {default_vs_vanilla_best:.1f}x.",
            f"Vanilla(best tested) / Rescaled(best tested LR) = {t3['vanilla_best_over_rescaled_best_ratio']:.4e} (log10 ratio = {t3['log10_vanilla_best_over_rescaled_best_ratio']:.2f}).",
        ]
    elif t3['pass']:
        category = 'rescaling adds value beyond LR tuning'
        summary = (
            'In this benchmark, rescaling still improves on vanilla Muon even after '
            'moving vanilla to its best tested LR.'
        )
        bullets = [
            f"Vanilla(0.02) / Rescaled(0.02) = {default_vs_rescaled:.1f}x.",
            f"Vanilla(best tested) / Rescaled(best tested LR) = {t3['vanilla_best_over_rescaled_best_ratio']:.4f}.",
            'The rescaled condition improves final loss by more than 5% at the vanilla best tested LR.',
        ]
    else:
        category = 'mixed / unresolved'
        summary = (
            'The discrete sweep does not support a clean single-sentence story. The '
            'T1/T2/T3 checks should be read together with the boundary caveats.'
        )
        bullets = [
            f"T1={'PASS' if t1['pass'] else 'FAIL'}, T2={'PASS' if t2['pass'] else 'FAIL'}, T3={'PASS' if t3['pass'] else 'FAIL'}.",
            f"Vanilla(0.02) / Rescaled(0.02) = {default_vs_rescaled:.1f}x.",
            f"Vanilla(0.02) / Vanilla(best tested) = {default_vs_vanilla_best:.1f}x.",
        ]

    return {
        'category': category,
        'summary': summary,
        'bullet_points': bullets,
        'boundary_notes': boundary_notes,
    }


# =============================================================================
# MAIN REUSABLE EXPERIMENT
# =============================================================================


def run_experiment():
    """Run the full H6 benchmark and return structured results."""
    start = perf_counter()
    seeds = [42 + i * 137 for i in range(NUM_SEEDS)]
    init_seeds = [seed + 1000 for seed in seeds]

    vanilla_by_lr = {
        float(lr): run_muon_condition(seeds, float(lr), rescale_mode='none')
        for lr in VANILLA_LRS
    }
    vanilla_selection = select_best_tested_lr(vanilla_by_lr, VANILLA_LRS)
    best_vanilla_lr = vanilla_selection['lr']

    sgd_by_lr = {float(lr): run_sgd_condition(seeds, float(lr)) for lr in SGD_LRS}
    sgd_selection = select_best_tested_lr(sgd_by_lr, SGD_LRS)
    best_sgd_lr = sgd_selection['lr']

    rescaled_original = run_muon_condition(seeds, ORIGINAL_LR, rescale_mode='curvature')
    rescaled_best = run_muon_condition(seeds, best_vanilla_lr, rescale_mode='curvature')

    conditions = {
        'vanilla_original': add_display_name(
            vanilla_by_lr[float(ORIGINAL_LR)],
            f'Vanilla Muon @ lr={ORIGINAL_LR}',
        ),
        'vanilla_best': add_display_name(
            vanilla_by_lr[best_vanilla_lr],
            f'Vanilla Muon @ best tested lr={best_vanilla_lr}',
        ),
        'rescaled_original': add_display_name(
            rescaled_original,
            f'Rescaled Muon @ lr={ORIGINAL_LR}',
        ),
        'rescaled_best': add_display_name(
            rescaled_best,
            f'Rescaled Muon @ vanilla best tested lr={best_vanilla_lr}',
        ),
        'sgd_best': add_display_name(
            sgd_by_lr[best_sgd_lr],
            f'SGD @ best tested lr={best_sgd_lr}',
        ),
    }

    paired_condition_order = ['vanilla_best', 'rescaled_original', 'rescaled_best', 'sgd_best']
    trajectory_condition_order = [
        'vanilla_original',
        'vanilla_best',
        'rescaled_original',
        'rescaled_best',
        'sgd_best',
    ]

    per_seed_winners, winner_counts = compute_seedwise_winners(
        conditions,
        paired_condition_order,
        seeds,
    )

    tests = build_tests(
        best_vanilla_lr,
        conditions['vanilla_best'],
        conditions['vanilla_original'],
        conditions['rescaled_original'],
        conditions['rescaled_best'],
    )

    comparisons = {
        'conditions': conditions,
        'paired_condition_order': paired_condition_order,
        'trajectory_condition_order': trajectory_condition_order,
        'per_seed_key_condition_losses': {
            key: conditions[key]['final_losses'] for key in paired_condition_order
        },
        'per_seed_key_condition_histories': {
            key: conditions[key]['loss_histories'] for key in paired_condition_order
        },
        'per_seed_winners': per_seed_winners,
        'winner_counts': winner_counts,
        'winner_counts_display': {
            conditions[key]['display_name']: int(count) for key, count in winner_counts.items()
        },
        'ratios': {
            'vanilla_original_over_rescaled_original': stable_ratio(
                conditions['vanilla_original']['summary']['mean'],
                conditions['rescaled_original']['summary']['mean'],
            ),
            'vanilla_original_over_vanilla_best': stable_ratio(
                conditions['vanilla_original']['summary']['mean'],
                conditions['vanilla_best']['summary']['mean'],
            ),
            'vanilla_best_over_sgd_best': stable_ratio(
                conditions['vanilla_best']['summary']['mean'],
                conditions['sgd_best']['summary']['mean'],
            ),
        },
    }

    verdict = classify_verdict(tests, vanilla_selection, sgd_selection, comparisons)
    runtime_seconds = perf_counter() - start

    return {
        'experiment_id': 'H6_LR_ARTIFACT_CHECK',
        'title': 'Experiment H6: learning-rate artifact check for final training loss',
        'scope': SCOPE_NOTE,
        'paths': {
            'script': str(SCRIPT_PATH),
            'notebook': str(NOTEBOOK_PATH),
        },
        'reproducibility': {
            'script_command': f'python {EXPERIMENT_RELATIVE_PATH}',
        },
        'config': {
            'dim': DIM,
            'num_layers': NUM_LAYERS,
            'num_steps': NUM_STEPS,
            'ns_iters': NS_ITERS,
            'momentum': MOMENTUM,
            'gamma': GAMMA,
            'scale_min': SCALE_MIN,
            'scale_max': SCALE_MAX,
            'num_seeds': NUM_SEEDS,
            'data_points': DATA_POINTS,
            'vanilla_lrs': [float(lr) for lr in VANILLA_LRS],
            'sgd_lrs': [float(lr) for lr in SGD_LRS],
            'original_lr': float(ORIGINAL_LR),
        },
        'seeds': seeds,
        'init_seeds': init_seeds,
        'selected_lrs': {
            'original_lr': float(ORIGINAL_LR),
            'expected_lr_from_min_clamp': float(ORIGINAL_LR * SCALE_MIN),
            'vanilla_best_tested': float(best_vanilla_lr),
            'sgd_best_tested': float(best_sgd_lr),
        },
        'vanilla': {
            'lr_grid': [float(lr) for lr in VANILLA_LRS],
            'by_lr': vanilla_by_lr,
            'best_tested_lr': float(best_vanilla_lr),
            'best_tested_summary': vanilla_by_lr[best_vanilla_lr]['summary'],
            'best_on_lower_boundary': bool(vanilla_selection['on_lower_boundary']),
            'best_on_upper_boundary': bool(vanilla_selection['on_upper_boundary']),
        },
        'sgd': {
            'lr_grid': [float(lr) for lr in SGD_LRS],
            'by_lr': sgd_by_lr,
            'best_tested_lr': float(best_sgd_lr),
            'best_tested_summary': sgd_by_lr[best_sgd_lr]['summary'],
            'best_on_lower_boundary': bool(sgd_selection['on_lower_boundary']),
            'best_on_upper_boundary': bool(sgd_selection['on_upper_boundary']),
        },
        'rescaled': {
            'original': add_display_name(rescaled_original, f'Rescaled Muon @ lr={ORIGINAL_LR}'),
            'best_vanilla_lr': add_display_name(
                rescaled_best,
                f'Rescaled Muon @ vanilla best tested lr={best_vanilla_lr}',
            ),
        },
        'comparisons': comparisons,
        'tests': tests,
        'verdict': verdict,
        'runtime_seconds': float(runtime_seconds),
    }


# =============================================================================
# CLI REPORTING
# =============================================================================


def _format_float(value, fmt='.6e'):
    value = float(value)
    if np.isfinite(value):
        return format(value, fmt)
    return 'inf'



def print_report(results):
    """Pretty-print the structured results for CLI usage."""
    config = results['config']
    tests = results['tests']
    comparisons = results['comparisons']

    print()
    print('=' * 110)
    print(f"  {results['title']}")
    print('=' * 110)
    print()
    print(f"  Scope: {results['scope']}")
    print(f"  Counterpart notebook: {results['paths']['notebook']}")
    print(f"  Reproduce with: {results['reproducibility']['script_command']}")
    print()
    print(
        f"  Setup: {config['num_layers']}-layer {config['dim']}x{config['dim']} deep linear, "
        f"{config['num_steps']} steps, {config['num_seeds']} seeds"
    )
    print(f"  Seeds: {results['seeds']}")
    print(f"  Original reference LR: {config['original_lr']}")
    print(f"  Expected pure min-clamp LR reference: {results['selected_lrs']['expected_lr_from_min_clamp']}")
    print()

    print('-' * 110)
    print('  PHASE 1: Vanilla Muon LR sweep (reported as best tested LR, not a claimed optimum)')
    print('-' * 110)
    for lr in results['vanilla']['lr_grid']:
        summary = results['vanilla']['by_lr'][lr]['summary']
        print(
            f"    lr={lr:<8.4f}  mean={summary['mean']:12.6e}  median={summary['median']:12.6e}  "
            f"std={summary['std']:12.2e}  finite={100.0 * summary['finite_fraction']:.0f}%"
        )
    print()
    print(
        f"    >>> Best tested vanilla Muon LR: {results['vanilla']['best_tested_lr']} "
        f"(mean final loss = {results['vanilla']['best_tested_summary']['mean']:.6e})"
    )
    if results['vanilla']['best_on_lower_boundary'] or results['vanilla']['best_on_upper_boundary']:
        print('    >>> Boundary caveat: the best tested vanilla LR lies on the sweep edge.')
    print()

    print('-' * 110)
    print('  PHASE 2: SGD LR sweep (reported as best tested LR, not a claimed optimum)')
    print('-' * 110)
    for lr in results['sgd']['lr_grid']:
        summary = results['sgd']['by_lr'][lr]['summary']
        print(
            f"    lr={lr:<8.4f}  mean={summary['mean']:12.6e}  median={summary['median']:12.6e}  "
            f"std={summary['std']:12.2e}  finite={100.0 * summary['finite_fraction']:.0f}%"
        )
    print()
    print(
        f"    >>> Best tested SGD LR: {results['sgd']['best_tested_lr']} "
        f"(mean final loss = {results['sgd']['best_tested_summary']['mean']:.6e})"
    )
    if results['sgd']['best_on_lower_boundary'] or results['sgd']['best_on_upper_boundary']:
        print('    >>> Boundary caveat: the best tested SGD LR lies on the sweep edge.')
    print()

    print('-' * 110)
    print('  PHASE 3: Curvature-rescaled Muon reference conditions')
    print('-' * 110)
    for key in ['original', 'best_vanilla_lr']:
        condition = results['rescaled'][key]
        summary = condition['summary']
        scale_summary = condition['scale_summary']
        print(f"    {condition['display_name']}")
        print(
            f"      mean final loss={summary['mean']:.6e}  median={summary['median']:.6e}  "
            f"std={summary['std']:.2e}"
        )
        print(
            f"      scale mean={scale_summary['mean']:.4f}  median={scale_summary['median']:.4f}  "
            f"min={scale_summary['min']:.4f}  max={scale_summary['max']:.4f}"
        )
        print(
            f"      min-clamp fraction={100.0 * scale_summary['min_clamp_fraction']:.1f}%  "
            f"max-clamp fraction={100.0 * scale_summary['max_clamp_fraction']:.1f}%"
        )
        print()

    print('=' * 110)
    print('  KEY CONDITION SUMMARY')
    print('=' * 110)
    print()
    header = f"  {'Condition':<42} {'Mean final loss':>16} {'Median':>16} {'Std':>12} {'Finite':>8}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for key in comparisons['paired_condition_order']:
        condition = comparisons['conditions'][key]
        summary = condition['summary']
        print(
            f"  {condition['display_name']:<42} {_format_float(summary['mean']):>16} "
            f"{_format_float(summary['median']):>16} {summary['std']:12.2e} "
            f"{100.0 * summary['finite_fraction']:7.0f}%"
        )
    print()

    print('=' * 110)
    print('  T1 / T2 / T3 OPERATIONAL CHECKS')
    print('=' * 110)
    print()
    print('  These are decision criteria on final-loss summaries, not formal hypothesis tests.')
    print()
    print(
        f"  T1: best tested vanilla LR / expected min-clamp LR = "
        f"{tests['T1']['ratio_best_to_expected']:.4f}  --> {'PASS' if tests['T1']['pass'] else 'FAIL'}"
    )
    print(f"      best tested vanilla LR = {tests['T1']['best_tested_lr']}")
    print(f"      expected reference LR  = {tests['T1']['expected_lr']}")
    print()
    print(
        f"  T2: vanilla(best tested) / rescaled(0.02) = "
        f"{tests['T2']['vanilla_best_over_rescaled_original_ratio']:.4e} "
        f"(log10 ratio = {tests['T2']['log10_vanilla_best_over_rescaled_original_ratio']:.2f}) "
        f"--> {'PASS' if tests['T2']['pass'] else 'FAIL'}"
    )
    print(f"      parity gap = {tests['T2']['parity_gap_percent']:.1f}%")
    print()
    print(
        f"  T3: vanilla(best tested) / rescaled(best tested LR) = "
        f"{tests['T3']['vanilla_best_over_rescaled_best_ratio']:.4e} "
        f"(log10 ratio = {tests['T3']['log10_vanilla_best_over_rescaled_best_ratio']:.2f}) "
        f"--> {'PASS' if tests['T3']['pass'] else 'FAIL'}"
    )
    print(
        f"      rescaled(best tested LR) / vanilla(best tested) = "
        f"{tests['T3']['rescaled_best_over_vanilla_best_ratio']:.4e}"
    )
    print()

    print('=' * 110)
    print('  PER-SEED WIN COUNTS ACROSS THE PAIRED COMPARISON SET')
    print('=' * 110)
    print()
    for key in comparisons['paired_condition_order']:
        display_name = comparisons['conditions'][key]['display_name']
        print(f"  {display_name:<42} : {comparisons['winner_counts'][key]}")
    print()

    print('=' * 110)
    print('  OVERALL VERDICT')
    print('=' * 110)
    print()
    print(f"  Category: {results['verdict']['category']}")
    print(f"  Summary: {results['verdict']['summary']}")
    print()
    for bullet in results['verdict']['bullet_points']:
        print(f"  - {bullet}")
    if results['verdict']['boundary_notes']:
        print()
        print('  Boundary notes:')
        for note in results['verdict']['boundary_notes']:
            print(f"  - {note}")
    print()
    print(f"  Runtime: {results['runtime_seconds']:.2f}s")
    print()



def main():
    """CLI entrypoint."""
    results = run_experiment()
    print_report(results)
    return results


if __name__ == '__main__':
    main()
