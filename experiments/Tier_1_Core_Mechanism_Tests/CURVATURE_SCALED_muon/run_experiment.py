#!/usr/bin/env python3
"""
Experiment 3.4 (directory name retained for continuity):
Clamped scalar norm-ratio rescaling for Muon on a toy deep-linear problem.

Important framing:
- The directory name `CURVATURE_SCALED_muon` is retained from an earlier draft.
- This implementation does NOT measure Hessians, explicit curvature, or Newton-step
  alignment.
- The main dynamic rescaling rule is a clipped scalar Frobenius-norm ratio,
      scale = clip(||G||_F / ||ortho(G)||_F * gamma, scale_min, scale_max)
  applied after Newton-Schulz orthogonalization.
- A fixed-scale control is included to test whether the observed gain is specific
  to the dynamic norm-ratio heuristic or is largely explainable by simple damping.
"""

import math
from typing import Any, Dict, List, Optional

import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG: Dict[str, Any] = {
    'dim': 4,
    'num_layers': 2,
    'num_steps': 500,
    'lr': 0.02,
    'momentum': 0.9,
    'gamma': 1.0,
    'scale_min': 0.1,
    'scale_max': 10.0,
    'fixed_scale_control': 0.1,
    'num_seeds': 10,
    'data_points': 32,
    'seed_base': 42,
    'seed_stride': 137,
    'bootstrap_resamples': 10000,
    'bootstrap_confidence_level': 95.0,
    'bootstrap_seed': 20260511,
}

CORE_VARIANT_KEYS: List[str] = ['a', 'b', 'c', 'e', 'f']


def get_core_pairwise_plan() -> List[Dict[str, str]]:
    """Meaningful paired comparisons among the core non-momentum variants."""
    return [
        {'key': 'b_minus_a', 'lhs': 'b', 'rhs': 'a', 'headline': 'Plain k=20 minus plain k=5'},
        {'key': 'c_minus_b', 'lhs': 'c', 'rhs': 'b', 'headline': 'Dynamic k=20 minus plain k=20'},
        {'key': 'c_minus_a', 'lhs': 'c', 'rhs': 'a', 'headline': 'Dynamic k=20 minus plain k=5'},
        {'key': 'e_minus_a', 'lhs': 'e', 'rhs': 'a', 'headline': 'Dynamic k=5 minus plain k=5'},
        {'key': 'f_minus_b', 'lhs': 'f', 'rhs': 'b', 'headline': 'Fixed-scale k=20 control minus plain k=20'},
        {'key': 'f_minus_a', 'lhs': 'f', 'rhs': 'a', 'headline': 'Fixed-scale k=20 control minus plain k=5'},
        {'key': 'e_minus_b', 'lhs': 'e', 'rhs': 'b', 'headline': 'Dynamic k=5 minus plain k=20'},
        {'key': 'e_minus_f', 'lhs': 'e', 'rhs': 'f', 'headline': 'Dynamic k=5 minus fixed-scale k=20 control'},
        {'key': 'c_minus_f', 'lhs': 'c', 'rhs': 'f', 'headline': 'Dynamic k=20 minus fixed-scale k=20 control'},
        {'key': 'c_minus_e', 'lhs': 'c', 'rhs': 'e', 'headline': 'Dynamic k=20 minus dynamic k=5'},
    ]


def get_default_config() -> Dict[str, Any]:
    """Return a copy of the default experiment configuration."""
    return dict(DEFAULT_CONFIG)


# =============================================================================
# NETWORK UTILITIES
# =============================================================================


def init_weights(dim: int, num_layers: int, seed: int) -> List[np.ndarray]:
    """Initialize layers near identity."""
    rng = np.random.RandomState(seed)
    weights = []
    for _ in range(num_layers):
        W = np.eye(dim) + rng.randn(dim, dim) * 0.1
        weights.append(W.copy())
    return weights



def forward_linear(weights: List[np.ndarray], X: np.ndarray) -> np.ndarray:
    """Forward pass through deep linear net."""
    out = X.copy()
    for W in weights:
        out = W @ out
    return out



def compute_loss(weights: List[np.ndarray], X: np.ndarray, Y_target: np.ndarray) -> float:
    """MSE loss."""
    Y_pred = forward_linear(weights, X)
    diff = Y_pred - Y_target
    return float(0.5 * np.mean(diff ** 2))



def compute_gradients(weights: List[np.ndarray], X: np.ndarray, Y_target: np.ndarray) -> List[np.ndarray]:
    """Backprop through deep linear net."""
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


def newton_schulz_orthogonalize(G: np.ndarray, num_iters: int = 5) -> np.ndarray:
    """Newton-Schulz iteration with Muon's quintic coefficients.

    a=3.4445, b=-4.7750, c=2.0315
    X_{k+1} = a*X + b*X@(X^T@X) + c*X@(X^T@X)^2
    """
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
# VARIANTS AND OPTIMIZER
# =============================================================================


def build_variant_specs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Variant definitions for the toy study."""
    gamma = float(config['gamma'])
    fixed_scale = float(config['fixed_scale_control'])
    return [
        {
            'key': 'a',
            'label': '(a) Muon k=5',
            'short_label': '(a)',
            'description': 'Baseline Muon with 5 Newton-Schulz iterations and no post-orthogonalization rescaling.',
            'ns_iters': 5,
            'rescale_mode': 'none',
        },
        {
            'key': 'b',
            'label': '(b) Muon k=20',
            'short_label': '(b)',
            'description': 'Higher-iteration Muon baseline with no post-orthogonalization rescaling.',
            'ns_iters': 20,
            'rescale_mode': 'none',
        },
        {
            'key': 'c',
            'label': '(c) k=20 + norm-ratio rescale',
            'short_label': '(c)',
            'description': 'k=20 with clipped scalar rescale based on ||G||_F / ||ortho(G)||_F. This is not an explicit Hessian/curvature estimate.',
            'ns_iters': 20,
            'rescale_mode': 'norm_ratio',
            'gamma': gamma,
        },
        {
            'key': 'd',
            'label': '(d) k=20 + momentum-norm rescale',
            'short_label': '(d)',
            'description': 'k=20 with clipped rescale based on the previous velocity norm (before incorporating the current step).',
            'ns_iters': 20,
            'rescale_mode': 'momentum',
        },
        {
            'key': 'e',
            'label': '(e) k=5 + norm-ratio rescale',
            'short_label': '(e)',
            'description': 'k=5 with the same clipped scalar norm-ratio rescale.',
            'ns_iters': 5,
            'rescale_mode': 'norm_ratio',
            'gamma': gamma,
        },
        {
            'key': 'f',
            'label': f'(f) k=20 + fixed scale {fixed_scale:.1f}',
            'short_label': '(f)',
            'description': 'Mechanistic control: k=20 with a constant post-orthogonalization scale equal to the lower clamp boundary.',
            'ns_iters': 20,
            'rescale_mode': 'fixed',
            'fixed_scale': fixed_scale,
        },
    ]



def train_muon(
    weights: List[np.ndarray],
    X: np.ndarray,
    Y_target: np.ndarray,
    lr: float,
    num_steps: int,
    ns_iters: int = 5,
    rescale_mode: str = 'none',
    gamma: float = 1.0,
    scale_min: float = 0.1,
    scale_max: float = 10.0,
    momentum: float = 0.9,
    fixed_scale: Optional[float] = None,
) -> Dict[str, Any]:
    """Train with the Muon-style update rule used in this toy experiment.

    rescale_mode:
      'none'       -- standard Muon (no post-orthogonalization rescaling)
      'norm_ratio' -- scale = clip(||G||_F / ||ortho(G)||_F * gamma, min, max)
      'momentum'   -- scale = clip(||velocity||_F, min, max), using the previous
                      velocity norm before the current update is incorporated
      'fixed'      -- scale = fixed_scale
    """
    weights = [W.copy() for W in weights]
    num_layers = len(weights)
    velocities = [np.zeros_like(W) for W in weights]

    losses: List[float] = []
    scale_history: List[List[float]] = []
    unclipped_scale_history: List[List[float]] = []
    min_clamp_hits: List[List[bool]] = []
    max_clamp_hits: List[List[bool]] = []
    velocity_norm_history: List[List[float]] = []
    update_norm_history: List[List[float]] = []

    for step in range(num_steps):
        loss = compute_loss(weights, X, Y_target)
        if not np.isfinite(loss):
            raise FloatingPointError(f'Non-finite loss encountered at step {step}: {loss}')
        losses.append(float(loss))

        grads = compute_gradients(weights, X, Y_target)

        step_scales = []
        step_unclipped_scales = []
        step_min_hits = []
        step_max_hits = []
        step_velocity_norms = []
        step_update_norms = []

        for i in range(num_layers):
            G = grads[i]
            G_norm = np.linalg.norm(G, 'fro')
            G_orth = newton_schulz_orthogonalize(G, num_iters=ns_iters)
            G_orth_norm = np.linalg.norm(G_orth, 'fro')

            unclipped_scale = 1.0
            scale = 1.0
            hit_min = False
            hit_max = False

            if rescale_mode == 'norm_ratio':
                if G_orth_norm > 1e-12:
                    unclipped_scale = G_norm / G_orth_norm * gamma
                else:
                    unclipped_scale = 1.0
                scale = float(np.clip(unclipped_scale, scale_min, scale_max))
                hit_min = unclipped_scale < scale_min
                hit_max = unclipped_scale > scale_max
            elif rescale_mode == 'momentum':
                vel_norm = np.linalg.norm(velocities[i], 'fro')
                if vel_norm > 1e-12 and step > 0:
                    unclipped_scale = float(vel_norm)
                else:
                    unclipped_scale = 1.0
                scale = float(np.clip(unclipped_scale, scale_min, scale_max))
                hit_min = unclipped_scale < scale_min
                hit_max = unclipped_scale > scale_max
            elif rescale_mode == 'fixed':
                if fixed_scale is None:
                    raise ValueError('fixed_scale must be provided when rescale_mode="fixed"')
                unclipped_scale = float(fixed_scale)
                scale = float(fixed_scale)
            elif rescale_mode == 'none':
                unclipped_scale = 1.0
                scale = 1.0
            else:
                raise ValueError(f'Unknown rescale_mode: {rescale_mode}')

            G_rescaled = G_orth * scale

            step_scales.append(float(scale))
            step_unclipped_scales.append(float(unclipped_scale))
            step_min_hits.append(bool(hit_min))
            step_max_hits.append(bool(hit_max))

            velocities[i] = momentum * velocities[i] + G_rescaled
            weights[i] = weights[i] - lr * velocities[i]

            if not np.all(np.isfinite(weights[i])):
                raise FloatingPointError(
                    f'Non-finite weights encountered at step {step}, layer {i}, mode={rescale_mode}'
                )

            step_velocity_norms.append(float(np.linalg.norm(velocities[i], 'fro')))
            step_update_norms.append(float(np.linalg.norm(lr * velocities[i], 'fro')))

        scale_history.append(step_scales)
        unclipped_scale_history.append(step_unclipped_scales)
        min_clamp_hits.append(step_min_hits)
        max_clamp_hits.append(step_max_hits)
        velocity_norm_history.append(step_velocity_norms)
        update_norm_history.append(step_update_norms)

    final_loss = compute_loss(weights, X, Y_target)
    if not np.isfinite(final_loss):
        raise FloatingPointError(f'Non-finite final loss encountered: {final_loss}')
    losses.append(float(final_loss))

    scale_history_arr = np.array(scale_history, dtype=float)
    unclipped_scale_history_arr = np.array(unclipped_scale_history, dtype=float)
    min_clamp_hits_arr = np.array(min_clamp_hits, dtype=bool)
    max_clamp_hits_arr = np.array(max_clamp_hits, dtype=bool)
    velocity_norm_history_arr = np.array(velocity_norm_history, dtype=float)
    update_norm_history_arr = np.array(update_norm_history, dtype=float)

    if scale_history_arr.size == 0:
        step_mean_scales = np.array([], dtype=float)
        step_mean_unclipped_scales = np.array([], dtype=float)
        step_min_clamp_fraction = np.array([], dtype=float)
        step_max_clamp_fraction = np.array([], dtype=float)
        step_mean_velocity_norms = np.array([], dtype=float)
        step_mean_update_norms = np.array([], dtype=float)
        overall_min_clamp_fraction = 0.0
        overall_max_clamp_fraction = 0.0
    else:
        step_mean_scales = np.mean(scale_history_arr, axis=1)
        step_mean_unclipped_scales = np.mean(unclipped_scale_history_arr, axis=1)
        step_min_clamp_fraction = np.mean(min_clamp_hits_arr, axis=1)
        step_max_clamp_fraction = np.mean(max_clamp_hits_arr, axis=1)
        step_mean_velocity_norms = np.mean(velocity_norm_history_arr, axis=1)
        step_mean_update_norms = np.mean(update_norm_history_arr, axis=1)
        overall_min_clamp_fraction = float(np.mean(min_clamp_hits_arr))
        overall_max_clamp_fraction = float(np.mean(max_clamp_hits_arr))

    return {
        'losses': np.array(losses, dtype=float),
        'final_loss': float(final_loss),
        'scale_history': scale_history_arr,
        'step_mean_scales': step_mean_scales,
        'unclipped_scale_history': unclipped_scale_history_arr,
        'step_mean_unclipped_scales': step_mean_unclipped_scales,
        'min_clamp_hits': min_clamp_hits_arr,
        'max_clamp_hits': max_clamp_hits_arr,
        'step_min_clamp_fraction': step_min_clamp_fraction,
        'step_max_clamp_fraction': step_max_clamp_fraction,
        'clamp_hit_fractions': {
            'min': overall_min_clamp_fraction,
            'max': overall_max_clamp_fraction,
        },
        'velocity_norm_history': velocity_norm_history_arr,
        'update_norm_history': update_norm_history_arr,
        'step_mean_velocity_norms': step_mean_velocity_norms,
        'step_mean_update_norms': step_mean_update_norms,
        'rescale_mode': rescale_mode,
        'ns_iters': ns_iters,
    }


# =============================================================================
# EXPERIMENT RUNNERS
# =============================================================================


def make_problem(seed: int, config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate the target map and training data for one seed."""
    rng = np.random.RandomState(seed)
    dim = int(config['dim'])
    num_layers = int(config['num_layers'])
    data_points = int(config['data_points'])

    W_target = [rng.randn(dim, dim) * 0.3 for _ in range(num_layers)]
    X = rng.randn(dim, data_points) * 0.5
    Y_target = X.copy()
    for W in W_target:
        Y_target = W @ Y_target

    return {
        'W_target': W_target,
        'X': X,
        'Y_target': Y_target,
    }



def run_single_seed(seed: int, config: Dict[str, Any], variant_specs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run all variants for a single seed and return per-variant results."""
    problem = make_problem(seed, config)
    X = problem['X']
    Y_target = problem['Y_target']

    dim = int(config['dim'])
    num_layers = int(config['num_layers'])
    num_steps = int(config['num_steps'])
    lr = float(config['lr'])
    momentum = float(config['momentum'])
    scale_min = float(config['scale_min'])
    scale_max = float(config['scale_max'])

    variants = {}
    for spec in variant_specs:
        w_init = init_weights(dim, num_layers, seed + 1000)
        train_result = train_muon(
            w_init,
            X,
            Y_target,
            lr=lr,
            num_steps=num_steps,
            ns_iters=int(spec['ns_iters']),
            rescale_mode=spec['rescale_mode'],
            gamma=float(spec.get('gamma', config['gamma'])),
            scale_min=scale_min,
            scale_max=scale_max,
            momentum=momentum,
            fixed_scale=spec.get('fixed_scale'),
        )
        variants[spec['key']] = train_result

    return {
        'seed': seed,
        'variants': variants,
    }



def aggregate_results(
    per_seed_results: List[Dict[str, Any]],
    variant_specs: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Aggregate per-seed results into summary arrays and statistics."""
    aggregates: Dict[str, Dict[str, Any]] = {}

    for spec in variant_specs:
        key = spec['key']
        variant_runs = [seed_result['variants'][key] for seed_result in per_seed_results]

        final_losses = np.array([run['final_loss'] for run in variant_runs], dtype=float)
        loss_curves = np.stack([run['losses'] for run in variant_runs], axis=0)
        scale_curves = np.stack([run['step_mean_scales'] for run in variant_runs], axis=0)
        unclipped_scale_curves = np.stack(
            [run['step_mean_unclipped_scales'] for run in variant_runs], axis=0
        )
        min_clamp_step_curves = np.stack(
            [run['step_min_clamp_fraction'] for run in variant_runs], axis=0
        )
        max_clamp_step_curves = np.stack(
            [run['step_max_clamp_fraction'] for run in variant_runs], axis=0
        )
        velocity_norm_curves = np.stack(
            [run['step_mean_velocity_norms'] for run in variant_runs], axis=0
        )
        update_norm_curves = np.stack(
            [run['step_mean_update_norms'] for run in variant_runs], axis=0
        )

        aggregates[key] = {
            'key': key,
            'label': spec['label'],
            'short_label': spec['short_label'],
            'description': spec['description'],
            'rescale_mode': spec['rescale_mode'],
            'ns_iters': int(spec['ns_iters']),
            'final_losses': final_losses,
            'loss_curves': loss_curves,
            'scale_curves': scale_curves,
            'unclipped_scale_curves': unclipped_scale_curves,
            'step_min_clamp_fractions': min_clamp_step_curves,
            'step_max_clamp_fractions': max_clamp_step_curves,
            'velocity_norm_curves': velocity_norm_curves,
            'update_norm_curves': update_norm_curves,
            'mean_final_loss': float(np.mean(final_losses)),
            'std_final_loss': float(np.std(final_losses)),
            'median_final_loss': float(np.median(final_losses)),
            'mean_loss_curve': np.mean(loss_curves, axis=0),
            'std_loss_curve': np.std(loss_curves, axis=0),
            'mean_scale_curve': np.mean(scale_curves, axis=0),
            'mean_unclipped_scale_curve': np.mean(unclipped_scale_curves, axis=0),
            'mean_step_min_clamp_fraction': np.mean(min_clamp_step_curves, axis=0),
            'mean_step_max_clamp_fraction': np.mean(max_clamp_step_curves, axis=0),
            'overall_min_clamp_fraction': float(np.mean(min_clamp_step_curves)),
            'overall_max_clamp_fraction': float(np.mean(max_clamp_step_curves)),
            'mean_velocity_norm_curve': np.mean(velocity_norm_curves, axis=0),
            'mean_update_norm_curve': np.mean(update_norm_curves, axis=0),
        }

    return aggregates



def safe_relative_change_pct(lhs_value: float, rhs_value: float) -> float:
    """Percent change of lhs relative to rhs, guarding tiny denominators."""
    if abs(rhs_value) <= 1e-15:
        return float('nan')
    return float((lhs_value - rhs_value) / rhs_value * 100.0)



def exact_two_sided_sign_test_pvalue(lhs_wins: int, rhs_wins: int) -> float:
    """Exact two-sided sign-test p-value, ignoring ties."""
    non_ties = int(lhs_wins) + int(rhs_wins)
    if non_ties <= 0:
        return 1.0
    k = min(int(lhs_wins), int(rhs_wins))
    tail_prob = sum(math.comb(non_ties, i) for i in range(k + 1)) / (2 ** non_ties)
    return float(min(1.0, 2.0 * tail_prob))



def paired_bootstrap_mean_delta_ci(
    delta: np.ndarray,
    num_resamples: int,
    confidence_level: float,
    seed: int,
) -> Dict[str, Any]:
    """Percentile bootstrap CI for the paired mean delta."""
    if delta.size == 0:
        return {
            'num_resamples': int(num_resamples),
            'confidence_level': float(confidence_level),
            'lower': float('nan'),
            'upper': float('nan'),
            'excludes_zero': False,
            'prob_mean_lt_zero': float('nan'),
            'prob_mean_gt_zero': float('nan'),
        }

    rng = np.random.RandomState(seed)
    sample_count = int(delta.size)
    resample_indices = rng.randint(0, sample_count, size=(int(num_resamples), sample_count))
    resampled_means = np.mean(delta[resample_indices], axis=1)

    alpha = max(0.0, min(50.0, 0.5 * (100.0 - float(confidence_level))))
    lower, upper = np.percentile(resampled_means, [alpha, 100.0 - alpha])
    excludes_zero = bool((upper < 0.0) or (lower > 0.0))

    return {
        'num_resamples': int(num_resamples),
        'confidence_level': float(confidence_level),
        'lower': float(lower),
        'upper': float(upper),
        'excludes_zero': excludes_zero,
        'prob_mean_lt_zero': float(np.mean(resampled_means < 0.0)),
        'prob_mean_gt_zero': float(np.mean(resampled_means > 0.0)),
    }



def build_paired_summary(
    lhs: np.ndarray,
    rhs: np.ndarray,
    lhs_label: str,
    rhs_label: str,
    num_resamples: int,
    confidence_level: float,
    bootstrap_seed: int,
) -> Dict[str, Any]:
    """Return a strengthened paired comparison summary for final-loss arrays."""
    delta = lhs - rhs
    lhs_wins = int(np.sum(lhs < rhs))
    rhs_wins = int(np.sum(rhs < lhs))
    ties = int(np.sum(lhs == rhs))
    mean_lhs = float(np.mean(lhs))
    mean_rhs = float(np.mean(rhs))
    bootstrap_ci = paired_bootstrap_mean_delta_ci(
        delta,
        num_resamples=num_resamples,
        confidence_level=confidence_level,
        seed=bootstrap_seed,
    )
    return {
        'lhs_label': lhs_label,
        'rhs_label': rhs_label,
        'delta_values': delta,
        'mean_lhs': mean_lhs,
        'mean_rhs': mean_rhs,
        'mean_delta': float(np.mean(delta)),
        'std_delta': float(np.std(delta)),
        'median_delta': float(np.median(delta)),
        'relative_mean_pct': safe_relative_change_pct(mean_lhs, mean_rhs),
        'lhs_wins': lhs_wins,
        'rhs_wins': rhs_wins,
        'ties': ties,
        'non_ties': int(lhs_wins + rhs_wins),
        'exact_sign_test_pvalue': exact_two_sided_sign_test_pvalue(lhs_wins, rhs_wins),
        'bootstrap_mean_delta_ci': bootstrap_ci,
    }



def format_pvalue(pvalue: float) -> str:
    """Compact p-value formatting for text reports."""
    if not np.isfinite(pvalue):
        return 'nan'
    if pvalue < 1e-4:
        return f'{pvalue:.2e}'
    return f'{pvalue:.4f}'



def format_bootstrap_ci(ci: Dict[str, Any]) -> str:
    """Compact bootstrap CI formatting for text reports."""
    level = ci['confidence_level']
    return f"{level:.0f}% CI [{ci['lower']:.3e}, {ci['upper']:.3e}]"



def evaluate_tests(
    aggregates: Dict[str, Dict[str, Any]],
    num_seeds: int,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate pass/fail checks and paired comparisons."""
    losses_a = aggregates['a']['final_losses']
    losses_b = aggregates['b']['final_losses']
    losses_c = aggregates['c']['final_losses']
    losses_d = aggregates['d']['final_losses']
    losses_e = aggregates['e']['final_losses']
    losses_f = aggregates['f']['final_losses']

    mean_a = float(np.mean(losses_a))
    mean_b = float(np.mean(losses_b))
    mean_c = float(np.mean(losses_c))
    mean_d = float(np.mean(losses_d))
    mean_e = float(np.mean(losses_e))
    mean_f = float(np.mean(losses_f))

    bootstrap_resamples = int(config['bootstrap_resamples'])
    bootstrap_confidence_level = float(config['bootstrap_confidence_level'])
    bootstrap_seed = int(config['bootstrap_seed'])

    loss_lookup = {
        'a': losses_a,
        'b': losses_b,
        'c': losses_c,
        'd': losses_d,
        'e': losses_e,
        'f': losses_f,
    }
    label_lookup = {key: aggregates[key]['label'] for key in aggregates.keys()}
    core_pairwise_plan = get_core_pairwise_plan()

    paired_comparisons: Dict[str, Dict[str, Any]] = {}
    for idx, plan_item in enumerate(core_pairwise_plan):
        comp = build_paired_summary(
            loss_lookup[plan_item['lhs']],
            loss_lookup[plan_item['rhs']],
            label_lookup[plan_item['lhs']],
            label_lookup[plan_item['rhs']],
            num_resamples=bootstrap_resamples,
            confidence_level=bootstrap_confidence_level,
            bootstrap_seed=bootstrap_seed + idx + 1,
        )
        comp['key'] = plan_item['key']
        comp['lhs_key'] = plan_item['lhs']
        comp['rhs_key'] = plan_item['rhs']
        comp['headline'] = plan_item['headline']
        paired_comparisons[plan_item['key']] = comp

    paired_comparisons['d_minus_b'] = build_paired_summary(
        losses_d,
        losses_b,
        aggregates['d']['label'],
        aggregates['b']['label'],
        num_resamples=bootstrap_resamples,
        confidence_level=bootstrap_confidence_level,
        bootstrap_seed=bootstrap_seed + len(core_pairwise_plan) + 1,
    )
    paired_comparisons['d_minus_b']['key'] = 'd_minus_b'
    paired_comparisons['d_minus_b']['lhs_key'] = 'd'
    paired_comparisons['d_minus_b']['rhs_key'] = 'b'
    paired_comparisons['d_minus_b']['headline'] = 'Momentum-norm k=20 minus plain k=20'

    paired_b_a = paired_comparisons['b_minus_a']
    paired_c_b = paired_comparisons['c_minus_b']
    paired_c_a = paired_comparisons['c_minus_a']
    paired_e_a = paired_comparisons['e_minus_a']
    paired_d_b = paired_comparisons['d_minus_b']
    paired_c_f = paired_comparisons['c_minus_f']

    tests: Dict[str, Any] = {}

    pct_b_vs_a = safe_relative_change_pct(mean_b, mean_a)
    tests['t1_k20_worse_than_k5'] = {
        'description': 'Plain k=20 is worse than plain k=5 (confirms the original toy problem).',
        'pass': bool(mean_b > mean_a),
        'mean_lhs': mean_b,
        'mean_rhs': mean_a,
        'wins': paired_b_a['rhs_wins'],
        'num_seeds': num_seeds,
        'details': [
            f"mean(b)={mean_b:.6e} vs mean(a)={mean_a:.6e} ({pct_b_vs_a:+.1f}%)",
            f"paired wins for (a) over (b): {paired_b_a['rhs_wins']}/{num_seeds}",
            f"{format_bootstrap_ci(paired_b_a['bootstrap_mean_delta_ci'])} for mean(b - a)",
            f"exact sign-test p={format_pvalue(paired_b_a['exact_sign_test_pvalue'])}",
        ],
    }

    pct_c_vs_b = safe_relative_change_pct(mean_c, mean_b)
    tests['t2_norm_ratio_improves_k20'] = {
        'description': 'Does clipped norm-ratio rescaling improve k=20 relative to plain k=20?',
        'pass': bool(mean_c < mean_b),
        'mean_lhs': mean_c,
        'mean_rhs': mean_b,
        'wins': paired_c_b['lhs_wins'],
        'num_seeds': num_seeds,
        'details': [
            f"mean(c)={mean_c:.6e} vs mean(b)={mean_b:.6e} ({pct_c_vs_b:+.1f}%)",
            f"paired wins for (c) over (b): {paired_c_b['lhs_wins']}/{num_seeds}",
            f"{format_bootstrap_ci(paired_c_b['bootstrap_mean_delta_ci'])} for mean(c - b)",
            f"exact sign-test p={format_pvalue(paired_c_b['exact_sign_test_pvalue'])}",
        ],
    }

    pct_d_vs_b = safe_relative_change_pct(mean_d, mean_b)
    tests['t3_momentum_improves_k20'] = {
        'description': 'Does momentum-norm rescaling improve k=20 relative to plain k=20?',
        'pass': bool(mean_d < mean_b),
        'mean_lhs': mean_d,
        'mean_rhs': mean_b,
        'wins': paired_d_b['lhs_wins'],
        'num_seeds': num_seeds,
        'details': [
            f"mean(d)={mean_d:.6e} vs mean(b)={mean_b:.6e} ({pct_d_vs_b:+.1f}%)",
            f"paired wins for (d) over (b): {paired_d_b['lhs_wins']}/{num_seeds}",
            f"{format_bootstrap_ci(paired_d_b['bootstrap_mean_delta_ci'])} for mean(d - b)",
            f"exact sign-test p={format_pvalue(paired_d_b['exact_sign_test_pvalue'])}",
        ],
    }

    pct_c_vs_a = safe_relative_change_pct(mean_c, mean_a)
    tests['t4_norm_ratio_k20_matches_k5'] = {
        'description': 'Does clipped norm-ratio rescaling make k=20 match or beat k=5 within 5%?',
        'pass': bool(mean_c <= mean_a * 1.05),
        'mean_lhs': mean_c,
        'mean_rhs': mean_a,
        'wins': paired_c_a['lhs_wins'],
        'num_seeds': num_seeds,
        'details': [
            f"mean(c)={mean_c:.6e} vs mean(a)={mean_a:.6e} ({pct_c_vs_a:+.1f}%)",
            f"paired wins for (c) over (a): {paired_c_a['lhs_wins']}/{num_seeds}",
            f"{format_bootstrap_ci(paired_c_a['bootstrap_mean_delta_ci'])} for mean(c - a)",
            f"exact sign-test p={format_pvalue(paired_c_a['exact_sign_test_pvalue'])}",
        ],
    }

    pct_e_vs_a = safe_relative_change_pct(mean_e, mean_a)
    tests['t5_norm_ratio_k5_improves'] = {
        'description': 'Does clipped norm-ratio rescaling improve the already-strong k=5 baseline?',
        'pass': bool(mean_e < mean_a),
        'mean_lhs': mean_e,
        'mean_rhs': mean_a,
        'wins': paired_e_a['lhs_wins'],
        'num_seeds': num_seeds,
        'details': [
            f"mean(e)={mean_e:.6e} vs mean(a)={mean_a:.6e} ({pct_e_vs_a:+.1f}%)",
            f"paired wins for (e) over (a): {paired_e_a['lhs_wins']}/{num_seeds}",
            f"{format_bootstrap_ci(paired_e_a['bootstrap_mean_delta_ci'])} for mean(e - a)",
            f"exact sign-test p={format_pvalue(paired_e_a['exact_sign_test_pvalue'])}",
        ],
    }

    pct_c_vs_f = safe_relative_change_pct(mean_c, mean_f)
    similar_within_10pct = bool(abs(pct_c_vs_f) <= 10.0) if np.isfinite(pct_c_vs_f) else False
    tests['t6_norm_ratio_vs_fixed_control'] = {
        'description': 'Does the dynamic k=20 norm-ratio rule outperform the fixed-scale 0.1 control?',
        'pass': bool(mean_c < mean_f),
        'mean_lhs': mean_c,
        'mean_rhs': mean_f,
        'wins': paired_c_f['lhs_wins'],
        'num_seeds': num_seeds,
        'pct_vs_fixed': pct_c_vs_f,
        'similar_within_10pct': similar_within_10pct,
        'paired_summary': paired_c_f,
        'details': [
            f"mean(c)={mean_c:.6e} vs mean(f)={mean_f:.6e} ({pct_c_vs_f:+.1f}%)",
            f"paired wins for (c) over (f): {paired_c_f['lhs_wins']}/{num_seeds} with {paired_c_f['ties']} ties",
            f"{format_bootstrap_ci(paired_c_f['bootstrap_mean_delta_ci'])} for mean(c - f)",
            f"exact sign-test p={format_pvalue(paired_c_f['exact_sign_test_pvalue'])}",
            f"bootstrap P(mean(c - f) < 0)={paired_c_f['bootstrap_mean_delta_ci']['prob_mean_lt_zero']:.3f}",
            'Interpretation: if (c) and (f) are very similar, the result is consistent with simple damping at the clamp floor.',
        ],
    }

    variant_means = {
        key: aggregates[key]['mean_final_loss']
        for key in aggregates.keys()
    }
    best_key = min(variant_means, key=variant_means.get)
    tests['t7_best_variant'] = {
        'description': 'Best variant by mean final loss.',
        'best_key': best_key,
        'best_label': aggregates[best_key]['label'],
        'best_mean': float(variant_means[best_key]),
        'details': [
            f"best variant = {aggregates[best_key]['label']} with mean final loss {variant_means[best_key]:.6e}",
        ],
    }

    tests['paired_comparisons'] = paired_comparisons

    return tests



def build_verdict_lines(result: Dict[str, Any]) -> List[str]:
    """Construct a compact verdict summary from aggregated results."""
    tests = result['tests']
    aggregates = result['aggregates']

    lines: List[str] = []

    if tests['t1_k20_worse_than_k5']['pass']:
        lines.append('Plain k=20 remains worse than plain k=5 in this toy setup.')
    else:
        lines.append('Plain k=20 is not worse than plain k=5 in this run, so the original degradation is not reproduced cleanly.')

    if tests['t2_norm_ratio_improves_k20']['pass']:
        lines.append('Clipped norm-ratio rescaling improves k=20 relative to plain k=20.')
    else:
        lines.append('Clipped norm-ratio rescaling does not improve k=20 relative to plain k=20 in this run.')

    if tests['t3_momentum_improves_k20']['pass']:
        lines.append('Momentum-norm rescaling also improves k=20 relative to plain k=20.')
    else:
        lines.append('Momentum-norm rescaling does not rescue k=20 here.')

    c_min_frac = aggregates['c']['overall_min_clamp_fraction']
    e_min_frac = aggregates['e']['overall_min_clamp_fraction']
    if c_min_frac > 0.5:
        lines.append(
            f"The winning k=20 norm-ratio variant spends most layer-steps at the lower clamp ({c_min_frac * 100:.1f}%), so mechanistic claims must remain cautious."
        )
    if e_min_frac > 0.5:
        lines.append(
            f"The k=5 norm-ratio variant also spends most layer-steps at the lower clamp ({e_min_frac * 100:.1f}%)."
        )

    control_test = tests['t6_norm_ratio_vs_fixed_control']
    if control_test['similar_within_10pct']:
        lines.append('The fixed-scale control is numerically similar to the dynamic norm-ratio rule, so this pair does not isolate a distinct norm-ratio mechanism beyond damping.')
    elif control_test['pass']:
        ci = control_test['paired_summary']['bootstrap_mean_delta_ci']
        lines.append(
            'The dynamic norm-ratio rule beats the fixed-scale control on most seeds, '
            f"with {format_bootstrap_ci(ci)} for mean(c - f), so the toy data suggest some value beyond constant damping."
        )
    else:
        lines.append('The fixed-scale control matches or beats the dynamic norm-ratio rule, so a simple damping interpretation remains plausible.')
    if c_min_frac > 0.9:
        lines.append('Because the k=20 dynamic rule is almost always at the lower clamp, the c-vs-f control comparison should be read as testing small departures from the clamp floor, not curvature restoration.')

    best = tests['t7_best_variant']
    lines.append(f"Best mean final loss in this run: {best['best_label']} ({best['best_mean']:.6e}).")
    lines.append('No explicit Hessian or curvature quantity is measured here; this pair supports empirical optimization claims only.')
    return lines



def format_experiment_report(result: Dict[str, Any]) -> str:
    """Format a human-readable report for CLI use."""
    config = result['config']
    variant_specs = result['variant_specs']
    aggregates = result['aggregates']
    tests = result['tests']
    variant_order = [spec['key'] for spec in variant_specs]

    lines: List[str] = []
    sep = '=' * 100

    lines.append(sep)
    lines.append('  Experiment 3.4 (directory retained): Clamped scalar norm-ratio rescaling for Muon')
    lines.append(sep)
    lines.append('')
    lines.append('  Honest scope note: this study uses a clipped scalar norm-ratio heuristic after')
    lines.append('  Newton-Schulz orthogonalization. It does not compute Hessians or explicit curvature.')
    lines.append('')
    lines.append(
        f"  Config: {config['num_layers']}-layer {config['dim']}x{config['dim']} deep linear, "
        f"{config['num_steps']} steps, lr={config['lr']}, momentum={config['momentum']}"
    )
    lines.append(
        f"  Dynamic rule: scale = clip(||G||_F / ||ortho(G)||_F * gamma, {config['scale_min']}, {config['scale_max']})"
    )
    lines.append(
        f"  Fixed control: scale = {config['fixed_scale_control']} after orthogonalization at k=20"
    )
    lines.append(
        f"  Seeds: {result['seeds']}"
    )
    lines.append(
        f"  Paired bootstrap summaries: {config['bootstrap_resamples']} resamples at {config['bootstrap_confidence_level']:.0f}% confidence"
    )
    lines.append('')

    lines.append(sep)
    lines.append('  VARIANTS')
    lines.append(sep)
    lines.append('')
    for spec in variant_specs:
        lines.append(f"  {spec['label']}: {spec['description']}")
    lines.append('')

    lines.append(sep)
    lines.append('  SUMMARY TABLE (final loss over seeds)')
    lines.append(sep)
    lines.append('')
    ref_mean = aggregates['a']['mean_final_loss']
    lines.append(
        f"  {'Variant':<35s} {'Mean':>14s} {'Std':>12s} {'vs (a)':>12s} {'Median':>14s}"
    )
    lines.append(
        f"  {'-' * 35} {'-' * 14} {'-' * 12} {'-' * 12} {'-' * 14}"
    )
    for key in variant_order:
        agg = aggregates[key]
        if ref_mean > 1e-15:
            pct = (agg['mean_final_loss'] - ref_mean) / ref_mean * 100
            pct_str = f"{pct:+.1f}%"
        else:
            pct_str = 'N/A'
        lines.append(
            f"  {agg['label']:<35s} {agg['mean_final_loss']:14.6e} {agg['std_final_loss']:12.2e} {pct_str:>12s} {agg['median_final_loss']:14.6e}"
        )
    lines.append('')

    lines.append(sep)
    lines.append('  RESCALING AND CLAMP DIAGNOSTICS')
    lines.append(sep)
    lines.append('')
    for key in variant_order:
        agg = aggregates[key]
        if agg['rescale_mode'] == 'none':
            continue
        scale_curve = agg['mean_scale_curve']
        unclipped_curve = agg['mean_unclipped_scale_curve']
        min_clamp_curve = agg['mean_step_min_clamp_fraction']
        max_clamp_curve = agg['mean_step_max_clamp_fraction']
        lines.append(f"  {agg['label']}:")
        lines.append(f"    Mean applied scale step 0   : {scale_curve[0]:.4f}")
        lines.append(f"    Mean applied scale step 100 : {scale_curve[min(100, len(scale_curve)-1)]:.4f}")
        lines.append(f"    Mean applied scale step 250 : {scale_curve[min(250, len(scale_curve)-1)]:.4f}")
        lines.append(f"    Mean applied scale step 499 : {scale_curve[-1]:.4f}")
        lines.append(f"    Mean unclipped scale step 0 : {unclipped_curve[0]:.4f}")
        lines.append(f"    Mean unclipped scale step 499: {unclipped_curve[-1]:.4f}")
        lines.append(
            f"    Min-clamp occupancy 0/250/499: {min_clamp_curve[0] * 100:.1f}% / {min_clamp_curve[min(250, len(min_clamp_curve)-1)] * 100:.1f}% / {min_clamp_curve[-1] * 100:.1f}%"
        )
        lines.append(
            f"    Max-clamp occupancy 0/250/499: {max_clamp_curve[0] * 100:.1f}% / {max_clamp_curve[min(250, len(max_clamp_curve)-1)] * 100:.1f}% / {max_clamp_curve[-1] * 100:.1f}%"
        )
        lines.append(
            f"    Clamp-hit fractions         : min={agg['overall_min_clamp_fraction'] * 100:.1f}%  max={agg['overall_max_clamp_fraction'] * 100:.1f}%"
        )
        lines.append('')

    lines.append(sep)
    lines.append('  LOSS CURVES AT KEY STEPS (mean over seeds)')
    lines.append(sep)
    lines.append('')
    check_steps = [0, 50, 100, 200, 300, 400, config['num_steps']]
    header = f"  {'Step':>6}"
    for key in variant_order:
        header += f"  {aggregates[key]['short_label']:>16s}"
    lines.append(header)
    lines.append(f"  {'-' * 6}" + f"  {'-' * 16}" * len(variant_order))
    for step in check_steps:
        row = f"  {step:>6}"
        for key in variant_order:
            curve = aggregates[key]['mean_loss_curve']
            row += f"  {curve[step]:16.6e}"
        lines.append(row)
    lines.append('')

    lines.append(sep)
    lines.append('  CORE FIVE PAIRED COMPARISONS (a, b, c, e, f)')
    lines.append(sep)
    lines.append('')
    ci_label = f"{config['bootstrap_confidence_level']:.0f}% CI"
    lines.append(
        f"  {'Comparison':<44s} {'Mean delta':>14s} {ci_label:>27s} {'Sign p':>10s} {'P(mean<0)':>10s} {'LHS':>5s} {'RHS':>5s} {'Tie':>5s}"
    )
    lines.append(
        f"  {'-' * 44} {'-' * 14} {'-' * 27} {'-' * 10} {'-' * 10} {'-' * 5} {'-' * 5} {'-' * 5}"
    )
    for plan_item in result['core_pairwise_plan']:
        comp = tests['paired_comparisons'][plan_item['key']]
        ci = comp['bootstrap_mean_delta_ci']
        ci_str = f"[{ci['lower']:.3e}, {ci['upper']:.3e}]"
        lines.append(
            f"  {plan_item['headline']:<44s} {comp['mean_delta']:14.6e} {ci_str:>27s} {format_pvalue(comp['exact_sign_test_pvalue']):>10s} {ci['prob_mean_lt_zero']:10.3f} {comp['lhs_wins']:5d} {comp['rhs_wins']:5d} {comp['ties']:5d}"
        )
    lines.append('')
    lines.append('  Additional momentum failure check:')
    comp = tests['paired_comparisons']['d_minus_b']
    ci = comp['bootstrap_mean_delta_ci']
    ci_str = f"[{ci['lower']:.3e}, {ci['upper']:.3e}]"
    lines.append(
        f"    {comp['headline']}: mean delta={comp['mean_delta']:.6e}, {ci_label}={ci_str}, sign p={format_pvalue(comp['exact_sign_test_pvalue'])}, P(mean<0)={ci['prob_mean_lt_zero']:.3f}, wins={comp['lhs_wins']}/{comp['non_ties']}"
    )
    lines.append('')

    lines.append(sep)
    lines.append('  HYPOTHESIS / CONTROL CHECKS')
    lines.append(sep)
    lines.append('')
    ordered_tests = [
        't1_k20_worse_than_k5',
        't2_norm_ratio_improves_k20',
        't3_momentum_improves_k20',
        't4_norm_ratio_k20_matches_k5',
        't5_norm_ratio_k5_improves',
        't6_norm_ratio_vs_fixed_control',
        't7_best_variant',
    ]
    for test_key in ordered_tests:
        test = tests[test_key]
        lines.append(f"  {test_key}: {test['description']}")
        for detail in test.get('details', []):
            lines.append(f"      {detail}")
        if 'pass' in test:
            lines.append(f"      Outcome: {'PASS' if test['pass'] else 'FAIL'}")
        lines.append('')

    lines.append(sep)
    lines.append('  PER-SEED FINAL LOSSES')
    lines.append(sep)
    lines.append('')
    header = f"  {'Seed':>6}"
    for key in variant_order:
        header += f"  {aggregates[key]['short_label']:>16s}"
    header += f"  {'Best':>8}"
    lines.append(header)
    lines.append(f"  {'-' * 6}" + f"  {'-' * 16}" * len(variant_order) + f"  {'-' * 8}")
    for idx, seed in enumerate(result['seeds']):
        row = f"  {seed:>6}"
        seed_losses = {}
        for key in variant_order:
            fl = aggregates[key]['final_losses'][idx]
            row += f"  {fl:16.6e}"
            seed_losses[aggregates[key]['short_label']] = fl
        best = min(seed_losses, key=seed_losses.get)
        row += f"  {best:>8s}"
        lines.append(row)
    lines.append('')

    lines.append(sep)
    lines.append('  OVERALL VERDICT')
    lines.append(sep)
    lines.append('')
    for verdict in result['verdict_lines']:
        lines.append(f"  - {verdict}")
    lines.append('')
    lines.append(sep)
    lines.append('  EXPERIMENT COMPLETE')
    lines.append(sep)
    return '\n'.join(lines)



def run_experiment(
    config_overrides: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run the full toy study and return structured results."""
    config = get_default_config()
    if config_overrides:
        config.update(config_overrides)

    variant_specs = build_variant_specs(config)
    seeds = [
        int(config['seed_base']) + i * int(config['seed_stride'])
        for i in range(int(config['num_seeds']))
    ]

    per_seed_results = []
    for idx, seed in enumerate(seeds):
        seed_result = run_single_seed(seed, config, variant_specs)
        per_seed_results.append(seed_result)
        if verbose:
            parts = []
            for spec in variant_specs:
                final_loss = seed_result['variants'][spec['key']]['final_loss']
                parts.append(f"{spec['short_label']}={final_loss:.2e}")
            print(f"  Seed {idx + 1:2d}/{len(seeds)} (seed={seed}): " + '  '.join(parts))

    aggregates = aggregate_results(per_seed_results, variant_specs)
    tests = evaluate_tests(aggregates, num_seeds=len(seeds), config=config)

    core_pairwise_plan = get_core_pairwise_plan()
    result = {
        'experiment_id': 'experiment_3_4_clamped_norm_ratio_muon',
        'title': 'Experiment 3.4 (directory retained): Clamped scalar norm-ratio rescaling for Muon',
        'path_note': 'Directory name CURVATURE_SCALED_muon is retained for continuity from an earlier draft.',
        'scope_note': 'This study does not compute Hessians or explicit curvature; it tests scalar post-orthogonalization rescaling heuristics in a toy deep-linear setting.',
        'config': config,
        'variant_specs': variant_specs,
        'core_variant_keys': list(CORE_VARIANT_KEYS),
        'core_pairwise_plan': core_pairwise_plan,
        'core_pairwise_order': [item['key'] for item in core_pairwise_plan],
        'seeds': seeds,
        'per_seed_results': per_seed_results,
        'aggregates': aggregates,
        'tests': tests,
    }
    result['verdict_lines'] = build_verdict_lines(result)
    result['report_text'] = format_experiment_report(result)
    result['report_lines'] = result['report_text'].splitlines()
    return result



def main() -> None:
    print()
    print('=' * 100)
    print('  Experiment 3.4 (directory retained): Clamped scalar norm-ratio rescaling for Muon')
    print('=' * 100)
    print('  Running toy deep-linear study; detailed summary will print after execution.')
    print()
    result = run_experiment(verbose=True)
    print()
    print(result['report_text'])


if __name__ == '__main__':
    main()
