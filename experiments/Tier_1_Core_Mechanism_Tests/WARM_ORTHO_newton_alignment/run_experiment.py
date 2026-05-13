#!/usr/bin/env python3
"""
Warm-orthogonal / Newton alignment toy study
===========================================

This experiment measures a *directional alignment proxy* in a tiny deep linear
network. At several SGD training checkpoints, it compares

    cos(-d_k, d_newton)

where:
  - d_k is the concatenated, layerwise Newton-Schulz-transformed gradient
    direction after k iterations, and
  - d_newton is a pseudoinverse Newton direction from the full finite-
    difference Hessian.

Important scope notes:
  - This is *not* a natural-gradient computation.
  - This does *not* directly test gauge-removal or gauge-null projection.
  - k=20 is treated as a high-k Newton-Schulz proxy, not asserted to be
    "exact orthogonalization" unless orthogonality diagnostics support that.
  - The Adam baseline is evaluated on the same SGD-trained trajectory; it is
    not an independently Adam-trained run.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

DIM = 4
NUM_LAYERS = 2
N_PARAMS = NUM_LAYERS * DIM * DIM
HESSIAN_EPS = 1e-5
NS_K_VALUES = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20]
MEASUREMENT_STEPS = [10, 20, 50, 100, 200, 300, 500, 750, 1000, 1500]
TRAIN_LR = 0.005
MOMENTUM = 0.9
NUM_SEEDS = 5

ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8

SIGNIFICANT_EIGENVALUE_FRACTION = 0.01
HESSIAN_NONZERO_TOL = 1e-12
HESSIAN_SIGN_TOL_FRACTION = 1e-8
HESSIAN_SIGN_TOL_MIN = 1e-10


# =============================================================================
# UTILITY HELPERS
# =============================================================================


def get_default_seeds(num_seeds: int = NUM_SEEDS, start: int = 42, stride: int = 137) -> List[int]:
    """Default deterministic seed schedule used by the study."""
    return [start + seed_idx * stride for seed_idx in range(num_seeds)]


def print_separator(char: str = '=', width: int = 108) -> None:
    print(char * width)


def summarize_array(values: Iterable[float]) -> Dict[str, Any]:
    """Return mean/std/SEM/95% CI summary statistics."""
    arr = np.asarray(list(values), dtype=float)
    count = int(arr.size)
    if count == 0:
        nan = float('nan')
        return {
            'count': 0,
            'mean': nan,
            'std': nan,
            'sem': nan,
            'ci95_low': nan,
            'ci95_high': nan,
            'ci95_halfwidth': nan,
            'min': nan,
            'max': nan,
        }

    mean = float(np.mean(arr))
    std = float(np.std(arr))
    sem = float(std / np.sqrt(count))
    ci95_halfwidth = float(1.96 * sem)
    return {
        'count': count,
        'mean': mean,
        'std': std,
        'sem': sem,
        'ci95_low': float(mean - ci95_halfwidth),
        'ci95_high': float(mean + ci95_halfwidth),
        'ci95_halfwidth': ci95_halfwidth,
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
    }


def describe_direction_k(k: int, max_k: Optional[int] = None) -> str:
    """Human-readable label for an NS iteration count used in reports/notebooks."""
    if k == 0:
        return "k=0 (layerwise normalized gradient)"
    if max_k is not None and k == max_k:
        return f"k={k} (high-k Newton-Schulz proxy)"
    return f"k={k}"


def build_direction_metadata(ns_k_values: Iterable[int]) -> Dict[str, Any]:
    """Return explicit semantic labels and notes for directions/baselines."""
    ns_k_values = list(ns_k_values)
    max_k = max(ns_k_values)
    selected_k_values_for_reporting = [k for k in [0, 5, 20] if k in ns_k_values]
    if not selected_k_values_for_reporting:
        selected_k_values_for_reporting = ns_k_values[: min(3, len(ns_k_values))]

    return {
        'direction_labels_by_k': {
            k: describe_direction_k(k, max_k=max_k)
            for k in ns_k_values
        },
        'selected_k_values_for_reporting': selected_k_values_for_reporting,
        'baseline_labels': {
            'sgd': 'SGD baseline (flat gradient direction)',
            'adam': 'Adam snapshot baseline on SGD trajectory',
        },
        'direction_notes': {
            'k_zero_definition': (
                'k=0 is the layerwise Frobenius-normalized gradient with zero '
                'Newton-Schulz updates; it is not the raw flat SGD direction.'
            ),
            'high_k_proxy_definition': (
                f'k={max_k} is the largest finite Newton-Schulz iteration count '
                'in this run and is treated only as a high-k proxy, not as '
                'proven exact orthogonalization.'
            ),
        },
    }


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
    """Backprop through deep linear net. Returns list of gradient matrices."""
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
# WEIGHT VECTOR UTILITIES
# =============================================================================


def weights_to_vector(weights: List[np.ndarray]) -> np.ndarray:
    """Flatten all weight matrices into a single vector."""
    return np.concatenate([W.flatten() for W in weights])


def vector_to_weights(vec: np.ndarray, dim: int, num_layers: int) -> List[np.ndarray]:
    """Unflatten a vector back into list of weight matrices."""
    weights = []
    idx = 0
    for _ in range(num_layers):
        size = dim * dim
        W = vec[idx:idx + size].reshape(dim, dim)
        weights.append(W)
        idx += size
    return weights


def grads_to_vector(grads: List[np.ndarray]) -> np.ndarray:
    """Flatten gradient matrices into a single vector."""
    return np.concatenate([g.flatten() for g in grads])


# =============================================================================
# NEWTON-SCHULZ ITERATION
# =============================================================================


def newton_schulz_orthogonalize(G: np.ndarray, num_iters: int) -> np.ndarray:
    """Apply Newton-Schulz iteration to approximate the orthogonal polar factor.

    k=0 returns the Frobenius-normalized matrix, i.e. no Newton-Schulz updates.
    """
    norm = np.linalg.norm(G, ord='fro')
    if norm < 1e-12:
        return G.copy()

    X = G / norm
    for _ in range(num_iters):
        A = X @ X.T
        X = 1.5 * X - 0.5 * A @ X
    return X


def orthogonality_residual(X: np.ndarray) -> float:
    """Normalized Frobenius residual for row-orthogonality: ||XX^T - I||_F / sqrt(dim)."""
    if X.ndim != 2 or X.shape[0] != X.shape[1]:
        raise ValueError("orthogonality_residual expects a square matrix")
    identity = np.eye(X.shape[0], dtype=X.dtype)
    return float(np.linalg.norm(X @ X.T - identity, ord='fro') / np.sqrt(X.shape[0]))


# =============================================================================
# FULL HESSIAN COMPUTATION VIA FINITE DIFFERENCES
# =============================================================================


def compute_gradient_vector(weights: List[np.ndarray], X: np.ndarray, Y_target: np.ndarray) -> np.ndarray:
    """Return gradient as a flat vector."""
    grads = compute_gradients(weights, X, Y_target)
    return grads_to_vector(grads)


def compute_full_hessian(
    weights: List[np.ndarray],
    X: np.ndarray,
    Y_target: np.ndarray,
    eps: float = HESSIAN_EPS,
) -> tuple[np.ndarray, float]:
    """Compute full Hessian via central finite differences on the gradient.

    Returns the symmetrized Hessian and the relative asymmetry of the raw finite-
    difference estimate before symmetrization.
    """
    theta = weights_to_vector(weights)
    n_params = len(theta)

    H_raw = np.zeros((n_params, n_params))

    for i in range(n_params):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[i] += eps
        theta_minus[i] -= eps

        w_plus = vector_to_weights(theta_plus, DIM, NUM_LAYERS)
        w_minus = vector_to_weights(theta_minus, DIM, NUM_LAYERS)

        grad_plus = compute_gradient_vector(w_plus, X, Y_target)
        grad_minus = compute_gradient_vector(w_minus, X, Y_target)

        H_raw[:, i] = (grad_plus - grad_minus) / (2 * eps)

    raw_norm = np.linalg.norm(H_raw, ord='fro')
    symmetry_residual_before_sym = float(
        np.linalg.norm(H_raw - H_raw.T, ord='fro') / max(raw_norm, 1e-15)
    )

    H = 0.5 * (H_raw + H_raw.T)
    return H, symmetry_residual_before_sym


# =============================================================================
# COSINE SIMILARITY AND DIRECTIONS
# =============================================================================


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Returns 0 if either is zero."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-15 or nb < 1e-15:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_muon_direction(grads: List[np.ndarray], k: int) -> tuple[np.ndarray, float]:
    """Apply k NS iterations to each gradient matrix, then flatten and concatenate.

    Returns:
      - flattened direction vector
      - mean orthogonality residual across layers
    """
    direction_parts = []
    residuals = []
    for G in grads:
        G_ns = newton_schulz_orthogonalize(G, num_iters=k)
        direction_parts.append(G_ns.flatten())
        residuals.append(orthogonality_residual(G_ns))
    return np.concatenate(direction_parts), float(np.mean(residuals))


def compute_adam_direction(m_state: np.ndarray, v_state: np.ndarray, t: int) -> np.ndarray:
    """Compute Adam update direction from current moment estimates."""
    m_hat = m_state / (1 - ADAM_BETA1 ** t)
    v_hat = v_state / (1 - ADAM_BETA2 ** t)
    return m_hat / (np.sqrt(v_hat) + ADAM_EPS)


# =============================================================================
# MAIN MEASUREMENT AT ONE TRAINING CHECKPOINT
# =============================================================================


def measure_alignment(
    weights: List[np.ndarray],
    X: np.ndarray,
    Y_target: np.ndarray,
    adam_m: np.ndarray,
    adam_v: np.ndarray,
    adam_t: int,
    ns_k_values: Optional[Iterable[int]] = None,
) -> Optional[Dict[str, Any]]:
    """Measure alignment between NS-transformed gradients and a Newton-like step.

    The Newton-like step is d_newton = -pinv(H) g, where H is the full finite-
    difference Hessian at the current checkpoint.
    """
    ns_k_values = list(NS_K_VALUES if ns_k_values is None else ns_k_values)

    grads = compute_gradients(weights, X, Y_target)
    g = grads_to_vector(grads)
    g_norm = float(np.linalg.norm(g))

    if g_norm < 1e-15:
        return None

    H, symmetry_residual_before_sym = compute_full_hessian(weights, X, Y_target)

    H_pinv = np.linalg.pinv(H, rcond=1e-10)
    d_newton = -H_pinv @ g
    d_newton_norm = float(np.linalg.norm(d_newton))
    if d_newton_norm < 1e-15:
        return None

    cos_by_k = {}
    orth_resid_by_k = {}
    for k in ns_k_values:
        d_muon_k, orth_resid = compute_muon_direction(grads, k)
        cos_by_k[k] = cosine_sim(-d_muon_k, d_newton)
        orth_resid_by_k[k] = orth_resid

    cos_sgd = cosine_sim(-g, d_newton)

    if adam_t > 0:
        adam_dir = compute_adam_direction(adam_m, adam_v, adam_t)
        cos_adam = cosine_sim(-adam_dir, d_newton)
    else:
        cos_adam = cos_sgd

    eigenvalues = np.linalg.eigvalsh(H)
    abs_eigenvalues = np.abs(eigenvalues)
    lambda_max_abs = float(np.max(abs_eigenvalues)) if abs_eigenvalues.size else 0.0

    if lambda_max_abs > 1e-15:
        significant_tol = SIGNIFICANT_EIGENVALUE_FRACTION * lambda_max_abs
        hessian_effective_rank_1pct = int(np.sum(abs_eigenvalues > significant_tol))
        sign_tol = max(HESSIAN_SIGN_TOL_MIN, HESSIAN_SIGN_TOL_FRACTION * lambda_max_abs)
    else:
        hessian_effective_rank_1pct = 0
        sign_tol = HESSIAN_SIGN_TOL_MIN

    nonzero_abs_eigs = abs_eigenvalues[abs_eigenvalues > HESSIAN_NONZERO_TOL]
    if nonzero_abs_eigs.size > 0:
        abs_spectral_cond_proxy = float(lambda_max_abs / max(float(np.min(nonzero_abs_eigs)), 1e-15))
    else:
        abs_spectral_cond_proxy = float('inf')

    hessian_n_pos = int(np.sum(eigenvalues > sign_tol))
    hessian_n_neg = int(np.sum(eigenvalues < -sign_tol))
    hessian_n_zero_tol = int(eigenvalues.size - hessian_n_pos - hessian_n_neg)

    return {
        'cos_by_k': cos_by_k,
        'orth_resid_by_k': orth_resid_by_k,
        'cos_sgd': float(cos_sgd),
        'cos_adam': float(cos_adam),
        'loss': compute_loss(weights, X, Y_target),
        'grad_norm': g_norm,
        'newton_norm': d_newton_norm,
        'abs_spectral_cond_proxy': abs_spectral_cond_proxy,
        'hessian_effective_rank_1pct': hessian_effective_rank_1pct,
        'hessian_lambda_max_abs': lambda_max_abs,
        'hessian_n_pos': hessian_n_pos,
        'hessian_n_neg': hessian_n_neg,
        'hessian_n_zero_tol': hessian_n_zero_tol,
        'hessian_sign_tol': float(sign_tol),
        'hessian_symmetry_residual_before_sym': symmetry_residual_before_sym,
    }


# =============================================================================
# TRAINING LOOP WITH MEASUREMENT
# =============================================================================


def run_single_seed(
    seed: int,
    measurement_steps: Optional[Iterable[int]] = None,
    ns_k_values: Optional[Iterable[int]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train with SGD+momentum, measure alignment at requested steps."""
    measurement_steps = list(MEASUREMENT_STEPS if measurement_steps is None else measurement_steps)
    ns_k_values = list(NS_K_VALUES if ns_k_values is None else ns_k_values)
    requested_steps = set(measurement_steps)

    rng = np.random.RandomState(seed)

    W_target = [rng.randn(DIM, DIM) * 0.5 for _ in range(NUM_LAYERS)]
    X = rng.randn(DIM, 32) * 0.5

    Y_target = X.copy()
    for W in W_target:
        Y_target = W @ Y_target

    weights = init_weights(DIM, NUM_LAYERS, seed + 1000)
    velocities = [np.zeros_like(W) for W in weights]

    adam_m = np.zeros(N_PARAMS)
    adam_v = np.zeros(N_PARAMS)
    adam_t = 0

    initial_loss = compute_loss(weights, X, Y_target)
    measurements = []
    max_step = max(measurement_steps)

    if verbose:
        print(f"    initial_loss={initial_loss:.4e}")

    for step in range(1, max_step + 1):
        grads = compute_gradients(weights, X, Y_target)
        g_flat = grads_to_vector(grads)

        adam_t += 1
        adam_m = ADAM_BETA1 * adam_m + (1 - ADAM_BETA1) * g_flat
        adam_v = ADAM_BETA2 * adam_v + (1 - ADAM_BETA2) * (g_flat ** 2)

        for i in range(len(weights)):
            velocities[i] = MOMENTUM * velocities[i] + grads[i]
            weights[i] = weights[i] - TRAIN_LR * velocities[i]

        if step not in requested_steps:
            continue

        loss_now = compute_loss(weights, X, Y_target)
        if loss_now < 1e-12:
            if verbose:
                print(f"    step {step:5d}: loss={loss_now:.2e} (too small, skipping)")
            continue

        result = measure_alignment(weights, X, Y_target, adam_m, adam_v, adam_t, ns_k_values=ns_k_values)
        if result is None:
            if verbose:
                print(f"    step {step:5d}: gradient or Newton direction vanished, skipping")
            continue

        result['step'] = int(step)
        result['seed'] = int(seed)
        measurements.append(result)

        if verbose:
            k5_text = result['cos_by_k'].get(5, float('nan'))
            k20_text = result['cos_by_k'].get(20, float('nan'))
            print(
                f"    step {step:5d}: loss={result['loss']:.4e}, "
                f"|g|={result['grad_norm']:.4e}, "
                f"H_effRank={result['hessian_effective_rank_1pct']:2d}, "
                f"cos(k=5,Newt)={k5_text:+.4f}, "
                f"cos(k=20,Newt)={k20_text:+.4f}"
            )

    final_loss = compute_loss(weights, X, Y_target)
    return {
        'seed': int(seed),
        'initial_loss': float(initial_loss),
        'final_loss': float(final_loss),
        'measurement_steps_requested': list(measurement_steps),
        'measurement_steps_observed': [m['step'] for m in measurements],
        'num_measurements': len(measurements),
        'measurements': measurements,
    }


# =============================================================================
# AGGREGATION AND VERDICT HELPERS
# =============================================================================


def _normalize_seed_results(seed_results: Iterable[Any]) -> List[Dict[str, Any]]:
    """Accept either new-style seed result dicts or legacy list-of-measurements."""
    normalized = []
    for item in seed_results:
        if isinstance(item, dict) and 'measurements' in item:
            normalized.append(item)
        else:
            normalized.append({
                'seed': None,
                'initial_loss': float('nan'),
                'final_loss': float('nan'),
                'measurement_steps_requested': list(MEASUREMENT_STEPS),
                'measurement_steps_observed': [m['step'] for m in item],
                'num_measurements': len(item),
                'measurements': list(item),
            })
    return normalized


def flatten_measurements(
    seed_results: Iterable[Any],
    ns_k_values: Optional[Iterable[int]] = None,
) -> List[Dict[str, Any]]:
    """Flatten nested per-seed measurement data into record dictionaries."""
    ns_k_values = list(NS_K_VALUES if ns_k_values is None else ns_k_values)
    normalized_seed_results = _normalize_seed_results(seed_results)

    records = []
    for seed_result in normalized_seed_results:
        seed = seed_result.get('seed')
        for measurement in seed_result['measurements']:
            record = {
                'seed': seed,
                'step': int(measurement['step']),
                'loss': float(measurement['loss']),
                'grad_norm': float(measurement['grad_norm']),
                'newton_norm': float(measurement['newton_norm']),
                'cos_sgd': float(measurement['cos_sgd']),
                'cos_adam': float(measurement['cos_adam']),
                'abs_spectral_cond_proxy': float(measurement['abs_spectral_cond_proxy']),
                'hessian_effective_rank_1pct': int(measurement['hessian_effective_rank_1pct']),
                'hessian_lambda_max_abs': float(measurement['hessian_lambda_max_abs']),
                'hessian_n_pos': int(measurement['hessian_n_pos']),
                'hessian_n_neg': int(measurement['hessian_n_neg']),
                'hessian_n_zero_tol': int(measurement['hessian_n_zero_tol']),
                'hessian_sign_tol': float(measurement['hessian_sign_tol']),
                'hessian_symmetry_residual_before_sym': float(measurement['hessian_symmetry_residual_before_sym']),
            }
            for k in ns_k_values:
                record[f'cos_k_{k}'] = float(measurement['cos_by_k'][k])
                record[f'orth_resid_k_{k}'] = float(measurement['orth_resid_by_k'][k])
            records.append(record)
    return records


def aggregate_results(
    seed_results: Iterable[Any],
    ns_k_values: Optional[Iterable[int]] = None,
) -> Dict[str, Any]:
    """Aggregate cosine and diagnostic measurements across seeds and steps."""
    ns_k_values = list(NS_K_VALUES if ns_k_values is None else ns_k_values)
    records = flatten_measurements(seed_results, ns_k_values=ns_k_values)

    overall_by_k = {
        k: summarize_array(record[f'cos_k_{k}'] for record in records)
        for k in ns_k_values
    }
    overall_orthogonality_by_k = {
        k: summarize_array(record[f'orth_resid_k_{k}'] for record in records)
        for k in ns_k_values
    }
    overall_baselines = {
        'sgd': summarize_array(record['cos_sgd'] for record in records),
        'adam': summarize_array(record['cos_adam'] for record in records),
    }
    overall_diagnostics = {
        'loss': summarize_array(record['loss'] for record in records),
        'grad_norm': summarize_array(record['grad_norm'] for record in records),
        'newton_norm': summarize_array(record['newton_norm'] for record in records),
        'abs_spectral_cond_proxy': summarize_array(record['abs_spectral_cond_proxy'] for record in records),
        'hessian_effective_rank_1pct': summarize_array(record['hessian_effective_rank_1pct'] for record in records),
        'hessian_lambda_max_abs': summarize_array(record['hessian_lambda_max_abs'] for record in records),
        'hessian_n_pos': summarize_array(record['hessian_n_pos'] for record in records),
        'hessian_n_neg': summarize_array(record['hessian_n_neg'] for record in records),
        'hessian_n_zero_tol': summarize_array(record['hessian_n_zero_tol'] for record in records),
        'hessian_symmetry_residual_before_sym': summarize_array(
            record['hessian_symmetry_residual_before_sym'] for record in records
        ),
    }

    per_step = {}
    for step in sorted({record['step'] for record in records}):
        step_records = [record for record in records if record['step'] == step]
        per_step[step] = {
            'n_measurements': len(step_records),
            'by_k': {
                k: summarize_array(record[f'cos_k_{k}'] for record in step_records)
                for k in ns_k_values
            },
            'orthogonality_by_k': {
                k: summarize_array(record[f'orth_resid_k_{k}'] for record in step_records)
                for k in ns_k_values
            },
            'baselines': {
                'sgd': summarize_array(record['cos_sgd'] for record in step_records),
                'adam': summarize_array(record['cos_adam'] for record in step_records),
            },
            'diagnostics': {
                'loss': summarize_array(record['loss'] for record in step_records),
                'grad_norm': summarize_array(record['grad_norm'] for record in step_records),
                'newton_norm': summarize_array(record['newton_norm'] for record in step_records),
                'abs_spectral_cond_proxy': summarize_array(record['abs_spectral_cond_proxy'] for record in step_records),
                'hessian_effective_rank_1pct': summarize_array(
                    record['hessian_effective_rank_1pct'] for record in step_records
                ),
                'hessian_n_pos': summarize_array(record['hessian_n_pos'] for record in step_records),
                'hessian_n_neg': summarize_array(record['hessian_n_neg'] for record in step_records),
                'hessian_n_zero_tol': summarize_array(record['hessian_n_zero_tol'] for record in step_records),
                'hessian_symmetry_residual_before_sym': summarize_array(
                    record['hessian_symmetry_residual_before_sym'] for record in step_records
                ),
            },
        }

    return {
        'overall_by_k': overall_by_k,
        'overall_orthogonality_by_k': overall_orthogonality_by_k,
        'overall_baselines': overall_baselines,
        'overall_diagnostics': overall_diagnostics,
        'per_step': per_step,
        'n_measurements': len(records),
    }


def compute_verdict(summary: Dict[str, Any], ns_k_values: Optional[Iterable[int]] = None) -> Dict[str, Any]:
    """Compute compact verdict components for reporting and notebook use."""
    ns_k_values = list(NS_K_VALUES if ns_k_values is None else ns_k_values)
    mean_by_k = {k: summary['overall_by_k'][k]['mean'] for k in ns_k_values}

    pooled_peak_k = max(ns_k_values, key=lambda k: mean_by_k[k])
    pooled_peak_mean_cosine = float(mean_by_k[pooled_peak_k])
    pooled_k5_mean_cosine = float(mean_by_k.get(5, float('nan')))
    pooled_k20_mean_cosine = float(mean_by_k.get(20, float('nan')))
    pooled_k5_minus_k20 = float(pooled_k5_mean_cosine - pooled_k20_mean_cosine)

    pooled_cos_sgd = float(summary['overall_baselines']['sgd']['mean'])
    pooled_cos_adam = float(summary['overall_baselines']['adam']['mean'])
    pooled_k5_minus_sgd = float(pooled_k5_mean_cosine - pooled_cos_sgd)
    pooled_k5_minus_adam = float(pooled_k5_mean_cosine - pooled_cos_adam)

    max_k = max(ns_k_values)
    pooled_final_mean_cosine = float(mean_by_k[max_k])
    if pooled_peak_k < max_k:
        decline_after_peak_observed = bool(pooled_final_mean_cosine < pooled_peak_mean_cosine)
    else:
        decline_after_peak_observed = None

    pooled_peak_in_predicted_range = bool(3 <= pooled_peak_k <= 5)
    inverted_u_story_supported = bool(
        pooled_peak_in_predicted_range
        and pooled_k5_mean_cosine > pooled_k20_mean_cosine
        and decline_after_peak_observed is True
    )

    if inverted_u_story_supported:
        current_result_statement = (
            "Current pooled results support the originally expected inverted-U story."
        )
    else:
        current_result_statement = (
            f"Current pooled results do not support the originally expected inverted-U story; "
            f"the best pooled cosine occurs at k={pooled_peak_k}."
        )

    return {
        'predicted_peak_range': [3, 5],
        'pooled_peak_k': int(pooled_peak_k),
        'pooled_peak_mean_cosine': pooled_peak_mean_cosine,
        'pooled_peak_in_predicted_range': pooled_peak_in_predicted_range,
        'pooled_final_mean_cosine_at_max_k': pooled_final_mean_cosine,
        'decline_after_peak_observed': decline_after_peak_observed,
        'pooled_k5_mean_cosine': pooled_k5_mean_cosine,
        'pooled_k20_mean_cosine': pooled_k20_mean_cosine,
        'pooled_k5_minus_k20': pooled_k5_minus_k20,
        'pooled_k5_beats_k20': bool(pooled_k5_mean_cosine > pooled_k20_mean_cosine),
        'pooled_cos_sgd': pooled_cos_sgd,
        'pooled_cos_adam': pooled_cos_adam,
        'pooled_k5_minus_sgd': pooled_k5_minus_sgd,
        'pooled_k5_minus_adam': pooled_k5_minus_adam,
        'orthogonality_residual_k5_mean': float(summary['overall_orthogonality_by_k'].get(5, {}).get('mean', float('nan'))),
        'orthogonality_residual_k20_mean': float(summary['overall_orthogonality_by_k'].get(20, {}).get('mean', float('nan'))),
        'inverted_u_story_supported': inverted_u_story_supported,
        'current_result_statement': current_result_statement,
    }


# =============================================================================
# TOP-LEVEL RUNNER AND REPORT
# =============================================================================


def run_experiment(
    seeds: Optional[Iterable[int]] = None,
    measurement_steps: Optional[Iterable[int]] = None,
    ns_k_values: Optional[Iterable[int]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the full toy study and return structured results."""
    seeds = list(get_default_seeds() if seeds is None else seeds)
    measurement_steps = list(MEASUREMENT_STEPS if measurement_steps is None else measurement_steps)
    ns_k_values = list(NS_K_VALUES if ns_k_values is None else ns_k_values)

    if not seeds:
        raise ValueError("run_experiment requires at least one seed")
    if not measurement_steps:
        raise ValueError("run_experiment requires at least one measurement step")
    if not ns_k_values:
        raise ValueError("run_experiment requires at least one NS iteration count")

    direction_metadata = build_direction_metadata(ns_k_values)
    start_time = time.perf_counter()
    seed_results = []

    for seed_idx, seed in enumerate(seeds, start=1):
        if verbose:
            print_separator('-')
            print(f"  Seed {seed_idx}/{len(seeds)} (seed={seed})")
            print_separator('-')
        seed_result = run_single_seed(
            seed,
            measurement_steps=measurement_steps,
            ns_k_values=ns_k_values,
            verbose=verbose,
        )
        seed_results.append(seed_result)
        if verbose:
            print()

    measurement_records = flatten_measurements(seed_results, ns_k_values=ns_k_values)
    summary = aggregate_results(seed_results, ns_k_values=ns_k_values)
    verdict = compute_verdict(summary, ns_k_values=ns_k_values)
    runtime_sec = float(time.perf_counter() - start_time)

    return {
        'config': {
            'script_path': os.path.abspath(__file__),
            'dim': DIM,
            'num_layers': NUM_LAYERS,
            'n_params': N_PARAMS,
            'hessian_eps': HESSIAN_EPS,
            'ns_k_values': ns_k_values,
            'measurement_steps': measurement_steps,
            'train_lr': TRAIN_LR,
            'momentum': MOMENTUM,
            'adam_beta1': ADAM_BETA1,
            'adam_beta2': ADAM_BETA2,
            'adam_eps': ADAM_EPS,
            'num_seeds': len(seeds),
            'significant_eigenvalue_fraction': SIGNIFICANT_EIGENVALUE_FRACTION,
            'hessian_nonzero_tol': HESSIAN_NONZERO_TOL,
            'hessian_sign_tol_fraction': HESSIAN_SIGN_TOL_FRACTION,
            'hessian_sign_tol_min': HESSIAN_SIGN_TOL_MIN,
        },
        'seeds': seeds,
        'direction_metadata': direction_metadata,
        'seed_results': seed_results,
        'measurement_records': measurement_records,
        'summary': summary,
        'verdict': verdict,
        'runtime_sec': runtime_sec,
    }


def print_report(results: Dict[str, Any]) -> None:
    """Pretty-print a calibrated report for CLI use."""
    config = results['config']
    summary = results['summary']
    verdict = results['verdict']
    ns_k_values = config['ns_k_values']
    per_step = summary['per_step']
    direction_metadata = results.get('direction_metadata', build_direction_metadata(ns_k_values))
    direction_labels_by_k = direction_metadata['direction_labels_by_k']

    print()
    print_separator('#')
    print("  WARM-ORTHOGONAL / NEWTON ALIGNMENT TOY STUDY")
    print_separator('#')
    print()
    print("  Scope: cosine alignment between layerwise NS-transformed gradients and a")
    print("         pseudoinverse Newton direction in a tiny 2-layer 4x4 deep linear net.")
    print("  Caveat: this is a directional proxy only; it does NOT compute the natural")
    print("          gradient and does NOT directly demonstrate gauge removal.")
    print()
    print(f"  Network: {config['num_layers']}-layer deep linear, dim={config['dim']} ({config['n_params']} params)")
    print(f"  Training trajectory: SGD+momentum (lr={config['train_lr']}, momentum={config['momentum']})")
    print(f"  Measurement steps: {config['measurement_steps']}")
    print(f"  NS iteration counts: {ns_k_values}")
    print(f"  Seeds: {results['seeds']}")
    print(f"  Finite-difference Hessian eps: {config['hessian_eps']}")
    print(f"  Runtime: {results['runtime_sec']:.2f} s")
    print()
    print(f"  Direction note: {direction_metadata['direction_notes']['k_zero_definition']}")
    print(f"  Direction note: {direction_metadata['direction_notes']['high_k_proxy_definition']}")
    print("  Baseline note: Adam is evaluated as a snapshot direction on the SGD-trained trajectory.")
    print()

    print_separator('=')
    print("  OVERALL ALIGNMENT SUMMARY (pooled over all measured seeds x steps)")
    print_separator('=')
    print()
    print(
        f"  {'k':<6s} | {'label':<38s} | {'mean cos':>10s} | {'std':>10s} | {'95% CI hw':>10s} | {'mean orth resid':>16s}"
    )
    print(f"  {'-'*6}-+-{'-'*38}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*16}")
    for k in ns_k_values:
        stats = summary['overall_by_k'][k]
        orth_stats = summary['overall_orthogonality_by_k'][k]
        marker = '  <<< pooled peak' if k == verdict['pooled_peak_k'] else ''
        print(
            f"  k={k:<3d} | "
            f"{direction_labels_by_k[k]:<38s} | "
            f"{stats['mean']:>+10.6f} | "
            f"{stats['std']:>10.6f} | "
            f"{stats['ci95_halfwidth']:>10.6f} | "
            f"{orth_stats['mean']:>16.6f}{marker}"
        )
    print()
    print("  Baselines:")
    for name in ['sgd', 'adam']:
        stats = summary['overall_baselines'][name]
        baseline_label = direction_metadata['baseline_labels'].get(name, name.upper())
        print(
            f"    {name.upper():<4s}: mean={stats['mean']:+.6f}, std={stats['std']:.6f}, "
            f"95% CI half-width={stats['ci95_halfwidth']:.6f} | {baseline_label}"
        )
    print()

    print_separator('=')
    print("  PER-STEP SUMMARY (means over seeds)")
    print_separator('=')
    print()
    selected_k_values = direction_metadata['selected_k_values_for_reporting']
    selected_labels = [f"k={k}" for k in selected_k_values]
    while len(selected_labels) < 3:
        selected_labels.append('')
        selected_k_values.append(selected_k_values[-1])
    print(
        f"  {'step':<6s} | {selected_labels[0]:>8s} | {selected_labels[1]:>8s} | {selected_labels[2]:>8s} | {'SGD':>8s} | {'Adam':>8s} | {'H_effRank':>10s}"
    )
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")
    for step in sorted(per_step):
        step_summary = per_step[step]
        print(
            f"  {step:<6d} | "
            f"{step_summary['by_k'][selected_k_values[0]]['mean']:+8.4f} | "
            f"{step_summary['by_k'][selected_k_values[1]]['mean']:+8.4f} | "
            f"{step_summary['by_k'][selected_k_values[2]]['mean']:+8.4f} | "
            f"{step_summary['baselines']['sgd']['mean']:+8.4f} | "
            f"{step_summary['baselines']['adam']['mean']:+8.4f} | "
            f"{step_summary['diagnostics']['hessian_effective_rank_1pct']['mean']:10.2f}"
        )
    print()

    print_separator('*')
    print("  VERDICT")
    print_separator('*')
    print()
    print(f"  Peak pooled cosine occurs at k={verdict['pooled_peak_k']} with mean {verdict['pooled_peak_mean_cosine']:+.6f}.")
    print(
        f"  Predicted low-k peak range [3, 5]: "
        f"{'YES' if verdict['pooled_peak_in_predicted_range'] else 'NO'}"
    )
    print()
    print(f"  k=5 pooled mean cosine  = {verdict['pooled_k5_mean_cosine']:+.6f}")
    print(f"  k=20 pooled mean cosine = {verdict['pooled_k20_mean_cosine']:+.6f}")
    print(f"  Difference (k=5 - k=20) = {verdict['pooled_k5_minus_k20']:+.6f}")
    print("  Note: k=20 here is a high-k Newton-Schulz proxy, not claimed exact orthogonalization.")
    print()
    print(f"  SGD pooled mean cosine  = {verdict['pooled_cos_sgd']:+.6f}")
    print(f"  Adam pooled mean cosine = {verdict['pooled_cos_adam']:+.6f}")
    print(f"  Difference (k=5 - SGD)  = {verdict['pooled_k5_minus_sgd']:+.6f}")
    print(f"  Difference (k=5 - Adam) = {verdict['pooled_k5_minus_adam']:+.6f}")
    print()
    print(f"  Mean orthogonality residual at k=5  = {verdict['orthogonality_residual_k5_mean']:.6f}")
    print(f"  Mean orthogonality residual at k=20 = {verdict['orthogonality_residual_k20_mean']:.6f}")
    print()
    print(f"  {verdict['current_result_statement']}")
    if verdict['inverted_u_story_supported']:
        print("  On this metric, the current pooled run is consistent with a low-k sweet spot.")
    else:
        print("  On this metric, the current pooled run does not justify the originally stronger")
        print("  inverted-U / exact-ortho-hurts narrative without additional evidence.")
    print()

    print_separator('#')
    print("  END OF REPORT")
    print_separator('#')
    print()


def main() -> Dict[str, Any]:
    """CLI entrypoint."""
    results = run_experiment(verbose=True)
    print_report(results)
    return results


if __name__ == '__main__':
    main()
