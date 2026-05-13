#!/usr/bin/env python3
"""
1.3b-i-A: sigma_1(W) Growth Under SGD vs Muon
=============================================

First completion-pass scope
---------------------------
This file keeps the original deterministic deep-linear toy study but tightens
its execution model and claim discipline.

What is measured
- Per-state sigma_1(W), sigma_n(W), condition numbers, and losses.
- Final singular-value spectra for every layer.
- Two fit families for log(sigma_1(W)):
    * exponential proxy: log(sigma_1) = a * t + b
    * polynomial/power-law proxy: log(sigma_1) = a * log(t) + b
- A retained legacy scalar coupling proxy:
    corr( sigma_1(raw gradient used for the update into W_t), sigma_1(W_t) )
- A small direct directional diagnostic:
    mean absolute overlap of the top left/right singular vectors of W with the
    raw gradient and with the actual momentum update.

Important limitations
- The scalar correlation proxy is NOT a singular-vector alignment measure.
  It only tracks co-variation of top singular-value magnitudes.
- Muon has three distinct objects:
    raw gradient -> orthogonalized gradient -> momentum update.
  Only the orthogonalized gradient is approximately spectrum-flattened.
  The actual momentum update is generally not orthogonal.
- This is still a single-seed deterministic toy study, not a statistical claim.
- Strong wording about “exponential self-amplification” is only justified when
  the within-SGD exponential fit actually beats the polynomial alternative.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "dim": 32,
    "num_layers": 4,
    "num_steps": 500,
    "batch_size": 64,
    "lr_muon": 0.005,
    "momentum": 0.9,
    "ns_iters": 5,
    "sgd_lr_candidates": [0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.001],
    "lr_search_steps": 200,
    "lr_search_divergence_factor": 50.0,
    "corr_start_step": 10,
    "report_steps": [0, 50, 100, 200, 300, 500],
    "rolling_corr_window": 50,
}


# =============================================================================
# BASIC HELPERS
# =============================================================================


def merge_config(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Merge user config with defaults."""
    merged = dict(DEFAULT_CONFIG)
    if config:
        merged.update(config)
    merged["sgd_lr_candidates"] = list(merged["sgd_lr_candidates"])
    merged["report_steps"] = list(merged["report_steps"])
    return merged


def safe_scalar_nanmean(values: np.ndarray | List[float]) -> float:
    """Mean over finite values only; returns NaN if no finite values exist."""
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def safe_columnwise_nanmean(values: np.ndarray) -> np.ndarray:
    """Column-wise finite-only mean for 2D arrays."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError("safe_columnwise_nanmean expects a 2D array")
    out = np.full(arr.shape[1], np.nan, dtype=float)
    for col in range(arr.shape[1]):
        finite = arr[:, col][np.isfinite(arr[:, col])]
        if finite.size:
            out[col] = finite.mean()
    return out


def summarize_finite_vector(values: np.ndarray | List[float]) -> Dict[str, Any]:
    """Summary stats over finite values, including SEM and a normal-approx 95% CI."""
    arr = np.asarray(values, dtype=float).ravel()
    finite = arr[np.isfinite(arr)]
    n_finite = int(finite.size)
    if n_finite == 0:
        return {
            "n_finite": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "sem": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }

    mean = float(np.mean(finite))
    std = float(np.std(finite, ddof=1)) if n_finite > 1 else 0.0
    sem = float(std / np.sqrt(n_finite)) if n_finite > 1 else 0.0
    ci95_half_width = 1.96 * sem if n_finite > 1 else 0.0
    return {
        "n_finite": n_finite,
        "mean": mean,
        "std": std,
        "sem": sem,
        "median": float(np.median(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "ci95_low": mean - ci95_half_width,
        "ci95_high": mean + ci95_half_width,
    }


def safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation guarded against NaN / constant inputs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def to_serializable(obj: Any) -> Any:
    """Convert numpy / Path heavy structures into JSON-safe Python objects."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


# =============================================================================
# PROBLEM SETUP
# =============================================================================


def build_problem(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create the fixed target matrix and fixed batch used by the toy study."""
    rng = np.random.RandomState(config["seed"])
    dim = config["dim"]
    batch_size = config["batch_size"]

    w_target = rng.randn(dim, dim) * 0.5
    x_data = rng.randn(dim, batch_size) * 0.3
    target_spectrum = np.linalg.svd(w_target, compute_uv=False)

    return {
        "W_target": w_target,
        "X_data": x_data,
        "target_spectrum": target_spectrum,
        "target_fro_norm": float(np.linalg.norm(w_target, ord="fro")),
        "target_condition_number": float(target_spectrum[0] / max(target_spectrum[-1], 1e-30)),
        "input_mean_abs": float(np.mean(np.abs(x_data))),
        "input_std": float(np.std(x_data)),
    }


# =============================================================================
# CORE FUNCTIONS
# =============================================================================


def init_weights(num_layers: int, dim: int, seed: int = 42) -> List[np.ndarray]:
    """Initialize each layer near identity for stable deep-linear dynamics."""
    rng = np.random.RandomState(seed)
    weights = []
    for _ in range(num_layers):
        weights.append(np.eye(dim) + rng.randn(dim, dim) * 0.1)
    return weights


def forward(weights: List[np.ndarray], x_data: np.ndarray) -> np.ndarray:
    """Forward pass through the deep linear network."""
    out = x_data.copy()
    for weight in weights:
        out = weight @ out
    return out


def compute_loss(weights: List[np.ndarray], x_data: np.ndarray, target: np.ndarray) -> float:
    """Quadratic loss between network output and target-transformed data."""
    pred = forward(weights, x_data)
    target_out = target @ x_data
    diff = pred - target_out
    return float(0.5 * np.mean(np.sum(diff ** 2, axis=0)))


def compute_gradients(weights: List[np.ndarray], x_data: np.ndarray, target: np.ndarray) -> List[np.ndarray]:
    """Exact backpropagation through the deep linear network."""
    num_layers = len(weights)
    n_samples = x_data.shape[1]

    activations = [x_data.copy()]
    out = x_data.copy()
    for weight in weights:
        out = weight @ out
        activations.append(out.copy())

    target_out = target @ x_data
    delta = (activations[-1] - target_out) / n_samples

    grads: List[np.ndarray] = []
    for layer in range(num_layers - 1, -1, -1):
        grad = delta @ activations[layer].T
        grads.insert(0, grad)
        if layer > 0:
            delta = weights[layer].T @ delta

    return grads


def newton_schulz_orthogonalize(grad: np.ndarray, num_iters: int) -> np.ndarray:
    """Approximate the orthogonal polar factor of grad via Newton-Schulz."""
    norm = np.linalg.norm(grad, ord="fro")
    if norm < 1e-12:
        return grad.copy()

    x = grad / norm
    for _ in range(num_iters):
        a = x.T @ x
        x = 1.5 * x - 0.5 * x @ a
    return x


def gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient of a 1D array of magnitudes."""
    values = np.sort(np.abs(np.asarray(values, dtype=float)))
    n = len(values)
    total = np.sum(values)
    if n == 0 or total < 1e-30:
        return 0.0
    index = np.arange(1, n + 1)
    gini = (2.0 * np.sum(index * values) / (n * total)) - (n + 1.0) / n
    return float(gini)


def top_singular_vector_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """
    Mean absolute overlap of top left/right singular vectors.

    This is a small direct directional diagnostic. It is still a low-dimensional
    summary, but unlike the scalar sigma_1 correlation proxy it actually uses
    singular-vector information.
    """
    norm_a = np.linalg.norm(a, ord="fro")
    norm_b = np.linalg.norm(b, ord="fro")
    if norm_a < 1e-12 or norm_b < 1e-12:
        return float("nan")

    ua, _, vha = np.linalg.svd(a, full_matrices=False)
    ub, _, vhb = np.linalg.svd(b, full_matrices=False)
    left_overlap = abs(float(np.dot(ua[:, 0], ub[:, 0])))
    right_overlap = abs(float(np.dot(vha[0, :], vhb[0, :])))
    return 0.5 * (left_overlap + right_overlap)


# =============================================================================
# OPTIMIZER HELPERS
# =============================================================================


def find_stable_lr_sgd(config: Dict[str, Any], x_data: np.ndarray, w_target: np.ndarray) -> float:
    """Find the largest stable SGD learning rate from the configured candidate list."""
    dim = config["dim"]
    num_layers = config["num_layers"]
    seed = config["seed"]
    momentum = config["momentum"]

    for lr in config["sgd_lr_candidates"]:
        weights = init_weights(num_layers, dim, seed=seed)
        velocities = [np.zeros((dim, dim)) for _ in range(num_layers)]
        initial_loss = compute_loss(weights, x_data, w_target)
        stable = True

        for _ in range(config["lr_search_steps"]):
            grads = compute_gradients(weights, x_data, w_target)
            for layer in range(num_layers):
                velocities[layer] = momentum * velocities[layer] + grads[layer]
                weights[layer] = weights[layer] - lr * velocities[layer]
            loss = compute_loss(weights, x_data, w_target)
            if np.isnan(loss) or loss > initial_loss * config["lr_search_divergence_factor"]:
                stable = False
                break

        if stable:
            return float(lr)

    return float(config["sgd_lr_candidates"][-1])


def compute_optimizer_diagnostics(
    optimizer_name: str,
    weights: List[np.ndarray],
    velocities: List[np.ndarray],
    x_data: np.ndarray,
    w_target: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute raw-grad, transformed-grad, and actual-update diagnostics at one state."""
    num_layers = config["num_layers"]
    momentum = config["momentum"]

    grads = compute_gradients(weights, x_data, w_target)
    transformed_grads: List[np.ndarray] = []
    updates: List[np.ndarray] = []

    raw_grad_sigma1 = np.zeros(num_layers, dtype=float)
    transformed_grad_sigma1 = np.zeros(num_layers, dtype=float)
    update_sigma1 = np.zeros(num_layers, dtype=float)
    weight_raw_grad_alignment = np.zeros(num_layers, dtype=float)
    weight_update_alignment = np.zeros(num_layers, dtype=float)

    for layer in range(num_layers):
        raw_grad = grads[layer]
        if optimizer_name == "SGD":
            transformed_grad = raw_grad.copy()
        elif optimizer_name == "Muon":
            transformed_grad = newton_schulz_orthogonalize(raw_grad, num_iters=config["ns_iters"])
        else:
            raise ValueError(f"Unknown optimizer_name={optimizer_name!r}")

        update = momentum * velocities[layer] + transformed_grad
        transformed_grads.append(transformed_grad)
        updates.append(update)

        raw_grad_sigma1[layer] = np.linalg.svd(raw_grad, compute_uv=False)[0]
        transformed_grad_sigma1[layer] = np.linalg.svd(transformed_grad, compute_uv=False)[0]
        update_sigma1[layer] = np.linalg.svd(update, compute_uv=False)[0]
        weight_raw_grad_alignment[layer] = top_singular_vector_overlap(weights[layer], raw_grad)
        weight_update_alignment[layer] = top_singular_vector_overlap(weights[layer], update)

    return {
        "raw_grads": grads,
        "transformed_grads": transformed_grads,
        "updates": updates,
        "raw_grad_sigma1": raw_grad_sigma1,
        "transformed_grad_sigma1": transformed_grad_sigma1,
        "update_sigma1": update_sigma1,
        "weight_raw_grad_alignment": weight_raw_grad_alignment,
        "weight_update_alignment": weight_update_alignment,
    }


# =============================================================================
# MEASUREMENT ENGINE
# =============================================================================


def fit_exponential(steps: np.ndarray, log_sigma1: np.ndarray) -> tuple[float, float, float]:
    """Fit log(sigma_1) = a * t + b."""
    steps = np.asarray(steps, dtype=float)
    log_sigma1 = np.asarray(log_sigma1, dtype=float)
    mask = np.isfinite(log_sigma1)
    t = steps[mask]
    y = log_sigma1[mask]
    if len(t) < 3:
        return 0.0, 0.0, 0.0

    design = np.vstack([t, np.ones(len(t))]).T
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    a, b = float(coeffs[0]), float(coeffs[1])
    y_pred = a * t + b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0
    return a, b, float(r2)


def fit_polynomial(steps: np.ndarray, log_sigma1: np.ndarray) -> tuple[float, float, float]:
    """Fit log(sigma_1) = a * log(t) + b."""
    steps = np.asarray(steps, dtype=float)
    log_sigma1 = np.asarray(log_sigma1, dtype=float)
    mask = np.isfinite(log_sigma1) & (steps > 0)
    t = steps[mask]
    y = log_sigma1[mask]
    if len(t) < 3:
        return 0.0, 0.0, 0.0

    log_t = np.log(t)
    design = np.vstack([log_t, np.ones(len(log_t))]).T
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    a, b = float(coeffs[0]), float(coeffs[1])
    y_pred = a * log_t + b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0
    return a, b, float(r2)


def measure_optimizer_run(
    optimizer_name: str,
    lr: float,
    config: Dict[str, Any],
    x_data: np.ndarray,
    w_target: np.ndarray,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run one optimizer and record per-state spectral diagnostics."""
    dim = config["dim"]
    num_layers = config["num_layers"]
    num_steps = config["num_steps"]
    seed = config["seed"]

    weights = init_weights(num_layers, dim, seed=seed)
    velocities = [np.zeros((dim, dim)) for _ in range(num_layers)]

    sigma1_w = np.full((num_steps + 1, num_layers), np.nan, dtype=float)
    sigman_w = np.full((num_steps + 1, num_layers), np.nan, dtype=float)
    losses = np.full(num_steps + 1, np.nan, dtype=float)

    raw_grad_sigma1 = np.full((num_steps + 1, num_layers), np.nan, dtype=float)
    transformed_grad_sigma1 = np.full((num_steps + 1, num_layers), np.nan, dtype=float)
    update_sigma1 = np.full((num_steps + 1, num_layers), np.nan, dtype=float)
    weight_raw_grad_alignment = np.full((num_steps + 1, num_layers), np.nan, dtype=float)
    weight_update_alignment = np.full((num_steps + 1, num_layers), np.nan, dtype=float)

    diverged_at_step = None

    def record_weight_state(state_idx: int) -> None:
        for layer, weight in enumerate(weights):
            sv = np.linalg.svd(weight, compute_uv=False)
            sigma1_w[state_idx, layer] = sv[0]
            sigman_w[state_idx, layer] = sv[-1]
        losses[state_idx] = compute_loss(weights, x_data, w_target)

    record_weight_state(0)

    for state_idx in range(num_steps):
        diagnostics = compute_optimizer_diagnostics(
            optimizer_name=optimizer_name,
            weights=weights,
            velocities=velocities,
            x_data=x_data,
            w_target=w_target,
            config=config,
        )
        raw_grad_sigma1[state_idx] = diagnostics["raw_grad_sigma1"]
        transformed_grad_sigma1[state_idx] = diagnostics["transformed_grad_sigma1"]
        update_sigma1[state_idx] = diagnostics["update_sigma1"]
        weight_raw_grad_alignment[state_idx] = diagnostics["weight_raw_grad_alignment"]
        weight_update_alignment[state_idx] = diagnostics["weight_update_alignment"]

        velocities = [update.copy() for update in diagnostics["updates"]]
        weights = [weight - lr * update for weight, update in zip(weights, velocities)]
        record_weight_state(state_idx + 1)

        if np.isnan(losses[state_idx + 1]) or losses[state_idx + 1] > 1e10:
            diverged_at_step = state_idx + 1
            if verbose:
                print(f"    WARNING: {optimizer_name} diverged at state {diverged_at_step}")
            break

    final_state_index = num_steps if diverged_at_step is None else diverged_at_step
    final_diagnostics = compute_optimizer_diagnostics(
        optimizer_name=optimizer_name,
        weights=weights,
        velocities=velocities,
        x_data=x_data,
        w_target=w_target,
        config=config,
    )
    raw_grad_sigma1[final_state_index] = final_diagnostics["raw_grad_sigma1"]
    transformed_grad_sigma1[final_state_index] = final_diagnostics["transformed_grad_sigma1"]
    update_sigma1[final_state_index] = final_diagnostics["update_sigma1"]
    weight_raw_grad_alignment[final_state_index] = final_diagnostics["weight_raw_grad_alignment"]
    weight_update_alignment[final_state_index] = final_diagnostics["weight_update_alignment"]

    final_sv_spectrum = np.stack([np.linalg.svd(weight, compute_uv=False) for weight in weights], axis=0)

    return {
        "optimizer_name": optimizer_name,
        "lr": float(lr),
        "sigma1_W": sigma1_w,
        "sigman_W": sigman_w,
        "losses": losses,
        "raw_grad_sigma1": raw_grad_sigma1,
        "transformed_grad_sigma1": transformed_grad_sigma1,
        "update_sigma1": update_sigma1,
        "weight_raw_grad_alignment": weight_raw_grad_alignment,
        "weight_update_alignment": weight_update_alignment,
        "final_sv_spectrum": final_sv_spectrum,
        "diverged_at_step": diverged_at_step,
        "metric_notes": {
            "sigma1_W": "Top singular value of the layer weight at state t (after t updates).",
            "sigman_W": "Bottom singular value of the layer weight at state t (after t updates).",
            "raw_grad_sigma1": "Top singular value of the raw backprop gradient evaluated at the same state t.",
            "transformed_grad_sigma1": (
                "Top singular value of the optimizer-transformed gradient evaluated at the same state t. "
                "For SGD this equals the raw gradient; for Muon it is the Newton-Schulz orthogonalized gradient."
            ),
            "update_sigma1": (
                "Top singular value of the actual momentum update v_t = m v_{t-1} + transformed_grad_t, "
                "evaluated at the same state t before applying the next parameter update."
            ),
            "weight_raw_grad_alignment": (
                "Mean absolute overlap of the top left/right singular vectors of W and the raw gradient at the same state t."
            ),
            "weight_update_alignment": (
                "Mean absolute overlap of the top left/right singular vectors of W and the actual momentum update at the same state t."
            ),
        },
    }


# =============================================================================
# SUMMARY ANALYSIS
# =============================================================================


def build_optimizer_summary(
    optimizer_results: Dict[str, Any],
    steps: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute fits, Gini summaries, scalar proxy summaries, and alignment summaries."""
    num_layers = config["num_layers"]
    corr_start_step = config["corr_start_step"]
    report_steps = [step for step in config["report_steps"] if step <= config["num_steps"]]

    fit_by_layer = []
    exp_r2_all = []
    poly_r2_all = []
    exp_slope_all = []
    poly_exponent_all = []
    fit_r2_gap_all = []

    for layer in range(num_layers):
        log_sigma1 = np.log(optimizer_results["sigma1_W"][:, layer] + 1e-30)
        exp_a, exp_b, exp_r2 = fit_exponential(steps, log_sigma1)
        poly_a, poly_b, poly_r2 = fit_polynomial(steps, log_sigma1)
        r2_gap = float(exp_r2 - poly_r2)
        exp_r2_all.append(exp_r2)
        poly_r2_all.append(poly_r2)
        exp_slope_all.append(exp_a)
        poly_exponent_all.append(poly_a)
        fit_r2_gap_all.append(r2_gap)
        fit_by_layer.append({
            "layer": layer,
            "exp_a": exp_a,
            "exp_b": exp_b,
            "exp_r2": exp_r2,
            "poly_a": poly_a,
            "poly_b": poly_b,
            "poly_r2": poly_r2,
            "r2_gap_exp_minus_poly": r2_gap,
            "best_fit": "exponential" if exp_r2 > poly_r2 else "polynomial",
        })

    gini_by_layer = [
        gini_coefficient(optimizer_results["final_sv_spectrum"][layer])
        for layer in range(num_layers)
    ]

    lagged_proxy_sigma1 = np.full_like(optimizer_results["raw_grad_sigma1"], np.nan)
    lagged_proxy_sigma1[0] = optimizer_results["raw_grad_sigma1"][0]
    lagged_proxy_sigma1[1:] = optimizer_results["raw_grad_sigma1"][:-1]
    lagged_corr_proxy_by_layer = []
    for layer in range(num_layers):
        corr_value = safe_corrcoef(
            optimizer_results["sigma1_W"][corr_start_step:, layer],
            lagged_proxy_sigma1[corr_start_step:, layer],
        )
        lagged_corr_proxy_by_layer.append(corr_value)

    same_state_raw_grad_corr_by_layer = []
    for layer in range(num_layers):
        corr_value = safe_corrcoef(
            optimizer_results["sigma1_W"][corr_start_step:, layer],
            optimizer_results["raw_grad_sigma1"][corr_start_step:, layer],
        )
        same_state_raw_grad_corr_by_layer.append(corr_value)

    raw_grad_alignment_by_layer = safe_columnwise_nanmean(
        optimizer_results["weight_raw_grad_alignment"][corr_start_step:]
    )
    update_alignment_by_layer = safe_columnwise_nanmean(
        optimizer_results["weight_update_alignment"][corr_start_step:]
    )
    update_minus_raw_alignment_by_layer = update_alignment_by_layer - raw_grad_alignment_by_layer

    sigma1_key_steps = {
        str(step): optimizer_results["sigma1_W"][step].tolist() for step in report_steps
    }
    condition_number_key_steps = {
        str(step): (
            optimizer_results["sigma1_W"][step] / np.maximum(optimizer_results["sigman_W"][step], 1e-15)
        ).tolist()
        for step in report_steps
    }

    exp_r2_stats = summarize_finite_vector(np.asarray(exp_r2_all))
    poly_r2_stats = summarize_finite_vector(np.asarray(poly_r2_all))
    exp_slope_stats = summarize_finite_vector(np.asarray(exp_slope_all))
    poly_exponent_stats = summarize_finite_vector(np.asarray(poly_exponent_all))
    fit_r2_gap_stats = summarize_finite_vector(np.asarray(fit_r2_gap_all))
    raw_grad_alignment_stats = summarize_finite_vector(raw_grad_alignment_by_layer)
    update_alignment_stats = summarize_finite_vector(update_alignment_by_layer)
    update_minus_raw_alignment_stats = summarize_finite_vector(update_minus_raw_alignment_by_layer)

    fit_means = {
        "exp_r2_mean": exp_r2_stats["mean"],
        "poly_r2_mean": poly_r2_stats["mean"],
        "exp_slope_mean": exp_slope_stats["mean"],
        "poly_exponent_mean": poly_exponent_stats["mean"],
        "r2_gap_exp_minus_poly_mean": fit_r2_gap_stats["mean"],
        "best_fit_by_mean": "exponential"
        if exp_r2_stats["mean"] > poly_r2_stats["mean"]
        else "polynomial",
    }
    fit_uncertainty = {
        "exp_r2": exp_r2_stats,
        "poly_r2": poly_r2_stats,
        "exp_slope": exp_slope_stats,
        "poly_exponent": poly_exponent_stats,
        "r2_gap_exp_minus_poly": fit_r2_gap_stats,
        "layers_preferring_exponential": int(sum(row["best_fit"] == "exponential" for row in fit_by_layer)),
        "layers_preferring_polynomial": int(sum(row["best_fit"] == "polynomial" for row in fit_by_layer)),
    }
    alignment_uncertainty = {
        "weight_raw_grad_alignment": raw_grad_alignment_stats,
        "weight_update_alignment": update_alignment_stats,
        "weight_update_minus_raw_alignment": update_minus_raw_alignment_stats,
    }

    return {
        "fit_by_layer": fit_by_layer,
        "fit_means": fit_means,
        "fit_uncertainty": fit_uncertainty,
        "alignment_uncertainty": alignment_uncertainty,
        "gini_by_layer": gini_by_layer,
        "gini_mean": safe_scalar_nanmean(np.asarray(gini_by_layer)),
        "lagged_corr_proxy_by_layer": lagged_corr_proxy_by_layer,
        "lagged_corr_proxy_mean": safe_scalar_nanmean(np.asarray(lagged_corr_proxy_by_layer)),
        "same_state_raw_grad_corr_by_layer": same_state_raw_grad_corr_by_layer,
        "same_state_raw_grad_corr_mean": safe_scalar_nanmean(np.asarray(same_state_raw_grad_corr_by_layer)),
        "weight_raw_grad_alignment_by_layer_mean": raw_grad_alignment_by_layer.tolist(),
        "weight_raw_grad_alignment_mean": safe_scalar_nanmean(raw_grad_alignment_by_layer),
        "weight_update_alignment_by_layer_mean": update_alignment_by_layer.tolist(),
        "weight_update_alignment_mean": safe_scalar_nanmean(update_alignment_by_layer),
        "weight_update_minus_raw_alignment_by_layer_mean": update_minus_raw_alignment_by_layer.tolist(),
        "weight_update_minus_raw_alignment_mean": safe_scalar_nanmean(update_minus_raw_alignment_by_layer),
        "loss_initial": float(optimizer_results["losses"][0]),
        "loss_final": float(optimizer_results["losses"][-1]),
        "sigma1_key_steps": sigma1_key_steps,
        "condition_number_key_steps": condition_number_key_steps,
    }


def build_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Create optimizer summaries and calibrated verdict components."""
    config = results["config"]
    steps = results["steps"]

    optimizer_summaries = {
        name: build_optimizer_summary(opt_results, steps, config)
        for name, opt_results in results["optimizers"].items()
    }

    sgd = optimizer_summaries["SGD"]
    muon = optimizer_summaries["Muon"]

    test1 = sgd["fit_means"]["exp_r2_mean"] > muon["fit_means"]["exp_r2_mean"]
    test2 = muon["fit_means"]["poly_r2_mean"] > muon["fit_means"]["exp_r2_mean"]
    test2b = 0.0 <= muon["fit_means"]["poly_exponent_mean"] < 1.0
    test3 = sgd["fit_means"]["exp_r2_mean"] > sgd["fit_means"]["poly_r2_mean"]
    test4 = sgd["lagged_corr_proxy_mean"] > muon["lagged_corr_proxy_mean"]
    test5 = sgd["gini_mean"] > muon["gini_mean"]
    direct_alignment_advantage = (
        sgd["weight_update_alignment_mean"] > muon["weight_update_alignment_mean"]
    )
    slope_advantage = sgd["fit_means"]["exp_slope_mean"] > muon["fit_means"]["exp_slope_mean"]

    strong_story_supported = bool(test1 and test2 and test2b and test3)
    weaker_sgd_growth_story_supported = bool(slope_advantage and test5)

    sgd_fit_gap_stats = sgd["fit_uncertainty"]["r2_gap_exp_minus_poly"]
    muon_fit_gap_stats = muon["fit_uncertainty"]["r2_gap_exp_minus_poly"]
    t3_margin = float(sgd["fit_means"]["exp_r2_mean"] - sgd["fit_means"]["poly_r2_mean"])
    t3_layer_votes = {
        "layers_preferring_exponential": sgd["fit_uncertainty"]["layers_preferring_exponential"],
        "layers_preferring_polynomial": sgd["fit_uncertainty"]["layers_preferring_polynomial"],
    }
    t3_diagnostic = {
        "pass": bool(test3),
        "mean_r2_gap_exp_minus_poly": t3_margin,
        "mean_r2_gap_direction": "exponential_better" if t3_margin > 0 else "polynomial_better_or_tied",
        "layer_vote_counts": t3_layer_votes,
        "layerwise_r2_gap_summary": sgd_fit_gap_stats,
        "layerwise_r2_gap_values": [row["r2_gap_exp_minus_poly"] for row in sgd["fit_by_layer"]],
        "interpretation": (
            "T3 asks whether SGD's exponential proxy beats its polynomial proxy on mean R^2. "
            "Negative mean gap means the current run favors the polynomial family instead."
        ),
    }
    alignment_diagnostic = {
        "SGD": {
            "weight_raw_grad_alignment": sgd["weight_raw_grad_alignment_mean"],
            "weight_update_alignment": sgd["weight_update_alignment_mean"],
            "weight_update_minus_raw_alignment": sgd["weight_update_minus_raw_alignment_mean"],
            "weight_raw_grad_alignment_summary": sgd["alignment_uncertainty"]["weight_raw_grad_alignment"],
            "weight_update_alignment_summary": sgd["alignment_uncertainty"]["weight_update_alignment"],
            "weight_update_minus_raw_alignment_summary": sgd["alignment_uncertainty"]["weight_update_minus_raw_alignment"],
        },
        "Muon": {
            "weight_raw_grad_alignment": muon["weight_raw_grad_alignment_mean"],
            "weight_update_alignment": muon["weight_update_alignment_mean"],
            "weight_update_minus_raw_alignment": muon["weight_update_minus_raw_alignment_mean"],
            "weight_raw_grad_alignment_summary": muon["alignment_uncertainty"]["weight_raw_grad_alignment"],
            "weight_update_alignment_summary": muon["alignment_uncertainty"]["weight_update_alignment"],
            "weight_update_minus_raw_alignment_summary": muon["alignment_uncertainty"]["weight_update_minus_raw_alignment"],
        },
        "comparison": {
            "optimizer_with_larger_raw_grad_alignment_mean": (
                "SGD" if sgd["weight_raw_grad_alignment_mean"] > muon["weight_raw_grad_alignment_mean"] else "Muon"
            ),
            "optimizer_with_larger_update_alignment_mean": (
                "SGD" if sgd["weight_update_alignment_mean"] > muon["weight_update_alignment_mean"] else "Muon"
            ),
            "interpretation": (
                "Direct top-vector overlaps should be read separately from the scalar sigma1 correlation proxies. "
                "Comparing raw-gradient overlap to update overlap also helps show how momentum and Muon's gradient transformation alter the actual update direction."
            ),
        },
    }

    notes = []
    if not test3:
        notes.append(
            "Within SGD, the polynomial fit beats the exponential fit on mean R^2, so this run does not justify calling SGD sigma_1(W) growth exponential."
        )
        if (
            sgd_fit_gap_stats["ci95_high"] < 0
            and t3_layer_votes["layers_preferring_exponential"] == 0
        ):
            notes.append(
                "The layerwise SGD T3 diagnostic is uniformly unfavorable to the exponential story in this run: all layers prefer the polynomial family and the descriptive layer-level 95% interval for the SGD R^2 gap stays below zero."
            )
    if test2 and not test2b:
        notes.append(
            "Muon looks better fit by the polynomial family than the exponential family, but the fitted power-law exponent does not independently justify the word 'sub-linear'."
        )
    notes.append(
        "The retained scalar correlation proxy tracks top-singular-value magnitudes only; it is not a singular-vector alignment metric."
    )
    if not direct_alignment_advantage:
        notes.append(
            "The small direct W/update top-vector overlap diagnostic does not favor SGD in this run, so the scalar proxy advantage should not be read as directional-alignment evidence."
        )
    notes.append(
        "For Muon, the orthogonalized gradient and the actual momentum update are distinct objects; only the former is approximately spectrum-flattened."
    )

    if strong_story_supported:
        overall_label = "toy-scale support for an exponential-vs-slower-growth separation"
    elif weaker_sgd_growth_story_supported:
        overall_label = "weaker support only: SGD grows faster / ends more anisotropic, but the strong exponential-self-amplification story is not established"
    else:
        overall_label = "mixed evidence in this deterministic toy setup"

    verdict_components = {
        "T1_sgd_exp_r2_gt_muon_exp_r2": {
            "pass": bool(test1),
            "description": "SGD has the larger mean exponential-fit R^2.",
            "sgd_exp_r2_mean": sgd["fit_means"]["exp_r2_mean"],
            "muon_exp_r2_mean": muon["fit_means"]["exp_r2_mean"],
        },
        "T2_muon_poly_r2_gt_muon_exp_r2": {
            "pass": bool(test2),
            "description": "Muon is better fit by the polynomial/power-law family than by the exponential family.",
            "muon_poly_r2_mean": muon["fit_means"]["poly_r2_mean"],
            "muon_exp_r2_mean": muon["fit_means"]["exp_r2_mean"],
        },
        "T2b_muon_poly_exponent_in_[0,1)": {
            "pass": bool(test2b),
            "description": "Muon's mean fitted power-law exponent is in the sub-linear range [0, 1).",
            "muon_poly_exponent_mean": muon["fit_means"]["poly_exponent_mean"],
        },
        "T3_sgd_exp_r2_gt_sgd_poly_r2": {
            "pass": bool(test3),
            "description": "Within SGD, the exponential family beats the polynomial family on mean R^2.",
            "sgd_exp_r2_mean": sgd["fit_means"]["exp_r2_mean"],
            "sgd_poly_r2_mean": sgd["fit_means"]["poly_r2_mean"],
            "sgd_r2_gap_exp_minus_poly_mean": sgd["fit_means"]["r2_gap_exp_minus_poly_mean"],
            "sgd_layers_preferring_exponential": sgd["fit_uncertainty"]["layers_preferring_exponential"],
            "sgd_layers_preferring_polynomial": sgd["fit_uncertainty"]["layers_preferring_polynomial"],
        },
        "T4_sgd_lagged_scalar_proxy_corr_gt_muon": {
            "pass": bool(test4),
            "description": "SGD has the larger retained lagged scalar proxy corr(sigma1(raw grad used for step t), sigma1(W_t)).",
            "sgd_lagged_corr_proxy_mean": sgd["lagged_corr_proxy_mean"],
            "muon_lagged_corr_proxy_mean": muon["lagged_corr_proxy_mean"],
        },
        "T5_sgd_final_gini_gt_muon": {
            "pass": bool(test5),
            "description": "SGD ends with the more unequal final singular-value spectrum on average.",
            "sgd_gini_mean": sgd["gini_mean"],
            "muon_gini_mean": muon["gini_mean"],
        },
        "D1_sgd_update_alignment_gt_muon": {
            "pass": bool(direct_alignment_advantage),
            "description": "Small direct directional diagnostic: SGD has the larger mean top-singular-vector overlap between W and the actual momentum update.",
            "sgd_weight_update_alignment_mean": sgd["weight_update_alignment_mean"],
            "muon_weight_update_alignment_mean": muon["weight_update_alignment_mean"],
        },
        "D2_sgd_exp_slope_gt_muon": {
            "pass": bool(slope_advantage),
            "description": "SGD has the larger mean slope in the exponential proxy fit log(sigma1) = a t + b.",
            "sgd_exp_slope_mean": sgd["fit_means"]["exp_slope_mean"],
            "muon_exp_slope_mean": muon["fit_means"]["exp_slope_mean"],
        },
    }

    assessment = {
        "overall_label": overall_label,
        "supports_weaker_sgd_growth_story": weaker_sgd_growth_story_supported,
        "supports_strong_exponential_self_amplification_story": strong_story_supported,
        "notes": notes,
    }

    comparative_fit_summary = {
        "SGD": {
            "exp_r2": sgd["fit_uncertainty"]["exp_r2"],
            "poly_r2": sgd["fit_uncertainty"]["poly_r2"],
            "r2_gap_exp_minus_poly": sgd_fit_gap_stats,
        },
        "Muon": {
            "exp_r2": muon["fit_uncertainty"]["exp_r2"],
            "poly_r2": muon["fit_uncertainty"]["poly_r2"],
            "r2_gap_exp_minus_poly": muon_fit_gap_stats,
        },
    }

    return {
        "optimizer_summaries": optimizer_summaries,
        "verdict_components": verdict_components,
        "assessment": assessment,
        "comparative_fit_summary": comparative_fit_summary,
        "t3_diagnostic": t3_diagnostic,
        "alignment_diagnostic": alignment_diagnostic,
        "metric_caveats": {
            "lagged_scalar_proxy": (
                "corr( sigma1(raw gradient used for the update into W_t), sigma1(W_t) ). This is retained for continuity with the original script but is not a singular-vector alignment measure."
            ),
            "same_state_raw_grad_corr": (
                "corr( sigma1(raw gradient at state t), sigma1(W_t) ). This is a same-state scalar magnitude proxy, still not a direction measure."
            ),
            "direct_alignment": (
                "Mean absolute overlap of top left/right singular vectors between W and either the raw gradient or the actual momentum update at the same state."
            ),
        },
    }


# =============================================================================
# PUBLIC RUN FUNCTION
# =============================================================================


def run_experiment(config: Dict[str, Any] | None = None, *, verbose: bool = True) -> Dict[str, Any]:
    """Run the full default toy experiment and return structured results."""
    merged_config = merge_config(config)
    start_time = time.time()

    if verbose:
        print("=" * 100)
        print("1.3b-i-A: sigma_1(W) GROWTH UNDER SGD VS MUON")
        print("=" * 100)
        print("Scope: deterministic single-seed deep-linear toy study.")
        print("Caveat: scalar sigma_1 correlation proxies are not singular-vector alignment measures.")
        print(
            f"Config: dim={merged_config['dim']}, layers={merged_config['num_layers']}, "
            f"steps={merged_config['num_steps']}, batch={merged_config['batch_size']}, seed={merged_config['seed']}"
        )

    problem = build_problem(merged_config)
    lr_sgd = find_stable_lr_sgd(merged_config, problem["X_data"], problem["W_target"])
    lr_muon = float(merged_config["lr_muon"])

    if verbose:
        print(f"Chosen learning rates: SGD={lr_sgd}, Muon={lr_muon}")
        print(f"Initial target condition number: {problem['target_condition_number']:.4f}")
        print("Running SGD...")

    results_sgd = measure_optimizer_run(
        optimizer_name="SGD",
        lr=lr_sgd,
        config=merged_config,
        x_data=problem["X_data"],
        w_target=problem["W_target"],
        verbose=verbose,
    )

    if verbose:
        print("Running Muon...")

    results_muon = measure_optimizer_run(
        optimizer_name="Muon",
        lr=lr_muon,
        config=merged_config,
        x_data=problem["X_data"],
        w_target=problem["W_target"],
        verbose=verbose,
    )

    results = {
        "config": merged_config,
        "steps": np.arange(merged_config["num_steps"] + 1),
        "problem": {
            "target_spectrum": problem["target_spectrum"],
            "target_fro_norm": problem["target_fro_norm"],
            "target_condition_number": problem["target_condition_number"],
            "input_mean_abs": problem["input_mean_abs"],
            "input_std": problem["input_std"],
            "W_target": problem["W_target"],
            "X_data": problem["X_data"],
        },
        "learning_rates": {"SGD": lr_sgd, "Muon": lr_muon},
        "optimizers": {"SGD": results_sgd, "Muon": results_muon},
    }
    results["summary"] = build_summary(results)
    results["runtime_seconds"] = float(time.time() - start_time)
    return results


# =============================================================================
# PLOTTING / SAVING / REPORTING
# =============================================================================


def rolling_corr_series(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """Rolling correlation series with NaN where undefined."""
    if len(x) != len(y):
        raise ValueError("rolling_corr_series expects arrays with the same length")
    out = np.full(len(x) - window + 1, np.nan, dtype=float)
    for idx in range(window - 1, len(x)):
        x_slice = x[idx - window + 1: idx + 1]
        y_slice = y[idx - window + 1: idx + 1]
        out[idx - window + 1] = safe_corrcoef(x_slice, y_slice)
    return out


def make_plots(
    results: Dict[str, Any],
    output_dir: Path | str | None = None,
    filename: str = "sigma1_growth_rate.png",
    *,
    show: bool = False,
):
    """Generate a summary figure. Returns (fig, save_path_or_None)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None, None

    config = results["config"]
    steps = results["steps"]
    sgd = results["optimizers"]["SGD"]
    muon = results["optimizers"]["Muon"]

    proxy_sgd = np.full_like(sgd["raw_grad_sigma1"], np.nan)
    proxy_muon = np.full_like(muon["raw_grad_sigma1"], np.nan)
    proxy_sgd[0] = sgd["raw_grad_sigma1"][0]
    proxy_muon[0] = muon["raw_grad_sigma1"][0]
    proxy_sgd[1:] = sgd["raw_grad_sigma1"][:-1]
    proxy_muon[1:] = muon["raw_grad_sigma1"][:-1]

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle(
        "1.3b-i-A: sigma_1(W) growth under SGD vs Muon\n"
        "Deterministic deep-linear toy study; scalar correlation panel is a magnitude proxy, not vector alignment",
        fontsize=13,
        fontweight="bold",
    )

    colors = {"SGD": "#1f77b4", "Muon": "#d62728"}

    # (a) log sigma1(W)
    ax = axes[0, 0]
    ax.set_title("(a) log sigma_1(W) trajectories")
    for layer in range(config["num_layers"]):
        ax.plot(steps, np.log(sgd["sigma1_W"][:, layer] + 1e-30), color=colors["SGD"], alpha=0.25)
        ax.plot(steps, np.log(muon["sigma1_W"][:, layer] + 1e-30), color=colors["Muon"], alpha=0.25, linestyle="--")
    ax.plot(steps, np.mean(np.log(sgd["sigma1_W"] + 1e-30), axis=1), color=colors["SGD"], linewidth=3, label="SGD mean")
    ax.plot(steps, np.mean(np.log(muon["sigma1_W"] + 1e-30), axis=1), color=colors["Muon"], linewidth=3, linestyle="--", label="Muon mean")
    ax.set_xlabel("Step")
    ax.set_ylabel("log(sigma_1)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) sigma1(W)
    ax = axes[0, 1]
    ax.set_title("(b) sigma_1(W) trajectories")
    for layer in range(config["num_layers"]):
        ax.plot(steps, sgd["sigma1_W"][:, layer], color=colors["SGD"], alpha=0.25)
        ax.plot(steps, muon["sigma1_W"][:, layer], color=colors["Muon"], alpha=0.25, linestyle="--")
    ax.plot(steps, np.mean(sgd["sigma1_W"], axis=1), color=colors["SGD"], linewidth=3, label="SGD mean")
    ax.plot(steps, np.mean(muon["sigma1_W"], axis=1), color=colors["Muon"], linewidth=3, linestyle="--", label="Muon mean")
    ax.set_xlabel("Step")
    ax.set_ylabel("sigma_1(W)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) condition number
    ax = axes[0, 2]
    ax.set_title("(c) Mean condition number sigma_1 / sigma_n")
    kappa_sgd = sgd["sigma1_W"] / np.maximum(sgd["sigman_W"], 1e-15)
    kappa_muon = muon["sigma1_W"] / np.maximum(muon["sigman_W"], 1e-15)
    ax.semilogy(steps, np.mean(kappa_sgd, axis=1), color=colors["SGD"], linewidth=3, label="SGD mean")
    ax.semilogy(steps, np.mean(kappa_muon, axis=1), color=colors["Muon"], linewidth=3, linestyle="--", label="Muon mean")
    ax.set_xlabel("Step")
    ax.set_ylabel("Condition number")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) rolling lagged scalar proxy
    ax = axes[1, 0]
    ax.set_title("(d) Rolling lagged scalar proxy corr\nraw grad used for step t vs sigma_1(W_t)")
    window = min(config["rolling_corr_window"], len(steps))
    corr_axis = steps[window - 1:]
    sgd_rolls = []
    muon_rolls = []
    for layer in range(config["num_layers"]):
        sgd_roll = rolling_corr_series(sgd["sigma1_W"][:, layer], proxy_sgd[:, layer], window)
        muon_roll = rolling_corr_series(muon["sigma1_W"][:, layer], proxy_muon[:, layer], window)
        sgd_rolls.append(sgd_roll)
        muon_rolls.append(muon_roll)
        ax.plot(corr_axis, sgd_roll, color=colors["SGD"], alpha=0.20)
        ax.plot(corr_axis, muon_roll, color=colors["Muon"], alpha=0.20, linestyle="--")
    ax.plot(corr_axis, np.nanmean(np.vstack(sgd_rolls), axis=0), color=colors["SGD"], linewidth=3, label="SGD mean")
    ax.plot(corr_axis, np.nanmean(np.vstack(muon_rolls), axis=0), color=colors["Muon"], linewidth=3, linestyle="--", label="Muon mean")
    ax.axhline(0.0, color="gray", linestyle=":", alpha=0.6)
    ax.set_xlabel("Step")
    ax.set_ylabel("Rolling corr")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (e) loss
    ax = axes[1, 1]
    ax.set_title("(e) Loss vs step")
    ax.semilogy(steps, sgd["losses"], color=colors["SGD"], linewidth=2.5, label="SGD")
    ax.semilogy(steps, muon["losses"], color=colors["Muon"], linewidth=2.5, linestyle="--", label="Muon")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (f) final singular spectra
    ax = axes[1, 2]
    ax.set_title("(f) Final singular-value spectra\nmean across layers")
    sv_idx = np.arange(1, config["dim"] + 1)
    ax.plot(sv_idx, np.mean(sgd["final_sv_spectrum"], axis=0), color=colors["SGD"], linewidth=2.5, marker="o", markersize=3, label="SGD mean spectrum")
    ax.plot(sv_idx, np.mean(muon["final_sv_spectrum"], axis=0), color=colors["Muon"], linewidth=2.5, linestyle="--", marker="s", markersize=3, label="Muon mean spectrum")
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Singular value")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = None
    if output_dir is not None:
        save_dir = Path(output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / filename
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig, save_path


def save_results_artifacts(
    results: Dict[str, Any],
    output_dir: Path | str,
    plot_path: Path | None = None,
) -> Dict[str, Any]:
    """Save structured summary JSON and compressed raw-array NPZ artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = output_dir / "sigma1_growth_rate_summary.json"
    raw_npz_path = output_dir / "sigma1_growth_rate_raw.npz"

    summary_payload = {
        "config": results["config"],
        "learning_rates": results["learning_rates"],
        "runtime_seconds": results["runtime_seconds"],
        "problem": {
            "target_spectrum": results["problem"]["target_spectrum"],
            "target_fro_norm": results["problem"]["target_fro_norm"],
            "target_condition_number": results["problem"]["target_condition_number"],
            "input_mean_abs": results["problem"]["input_mean_abs"],
            "input_std": results["problem"]["input_std"],
        },
        "summary": results["summary"],
        "artifact_paths": {
            "plot": plot_path,
            "summary_json": summary_json_path,
            "raw_npz": raw_npz_path,
        },
    }
    summary_json_path.write_text(json.dumps(to_serializable(summary_payload), indent=2))

    npz_payload: Dict[str, Any] = {
        "steps": results["steps"],
        "W_target": results["problem"]["W_target"],
        "X_data": results["problem"]["X_data"],
        "target_spectrum": results["problem"]["target_spectrum"],
    }
    for name, optimizer_results in results["optimizers"].items():
        prefix = name.lower()
        npz_payload[f"{prefix}_sigma1_W"] = optimizer_results["sigma1_W"]
        npz_payload[f"{prefix}_sigman_W"] = optimizer_results["sigman_W"]
        npz_payload[f"{prefix}_losses"] = optimizer_results["losses"]
        npz_payload[f"{prefix}_raw_grad_sigma1"] = optimizer_results["raw_grad_sigma1"]
        npz_payload[f"{prefix}_transformed_grad_sigma1"] = optimizer_results["transformed_grad_sigma1"]
        npz_payload[f"{prefix}_update_sigma1"] = optimizer_results["update_sigma1"]
        npz_payload[f"{prefix}_weight_raw_grad_alignment"] = optimizer_results["weight_raw_grad_alignment"]
        npz_payload[f"{prefix}_weight_update_alignment"] = optimizer_results["weight_update_alignment"]
        npz_payload[f"{prefix}_final_sv_spectrum"] = optimizer_results["final_sv_spectrum"]
    np.savez_compressed(raw_npz_path, **npz_payload)

    return {
        "plot": plot_path,
        "summary_json": summary_json_path,
        "raw_npz": raw_npz_path,
    }


def print_report(results: Dict[str, Any]) -> None:
    """Print a concise calibrated text report for normal script usage."""
    summary = results["summary"]
    config = results["config"]
    sgd = summary["optimizer_summaries"]["SGD"]
    muon = summary["optimizer_summaries"]["Muon"]
    t3 = summary["t3_diagnostic"]

    print("\n" + "=" * 100)
    print("CALIBRATED SUMMARY REPORT")
    print("=" * 100)
    print(
        f"Runtime: {results['runtime_seconds']:.2f}s | Seed: {config['seed']} | "
        f"LRs: SGD={results['learning_rates']['SGD']}, Muon={results['learning_rates']['Muon']}"
    )
    print(
        f"Initial / final loss: SGD {sgd['loss_initial']:.6e} -> {sgd['loss_final']:.6e} | "
        f"Muon {muon['loss_initial']:.6e} -> {muon['loss_final']:.6e}"
    )
    print("\nFit means (higher R^2 = better within-family fit):")
    print(
        f"  SGD : exp R^2={sgd['fit_means']['exp_r2_mean']:.4f}, poly R^2={sgd['fit_means']['poly_r2_mean']:.4f}, "
        f"gap(exp-poly)={sgd['fit_means']['r2_gap_exp_minus_poly_mean']:.4f}, "
        f"exp slope={sgd['fit_means']['exp_slope_mean']:.6f}, poly exponent={sgd['fit_means']['poly_exponent_mean']:.4f}"
    )
    print(
        f"  Muon: exp R^2={muon['fit_means']['exp_r2_mean']:.4f}, poly R^2={muon['fit_means']['poly_r2_mean']:.4f}, "
        f"gap(exp-poly)={muon['fit_means']['r2_gap_exp_minus_poly_mean']:.4f}, "
        f"exp slope={muon['fit_means']['exp_slope_mean']:.6f}, poly exponent={muon['fit_means']['poly_exponent_mean']:.4f}"
    )
    print("\nLayer-level uncertainty summaries (n = layers):")
    print(
        f"  SGD exp R^2  mean±95%CI ≈ {sgd['fit_uncertainty']['exp_r2']['mean']:.4f} "
        f"[{sgd['fit_uncertainty']['exp_r2']['ci95_low']:.4f}, {sgd['fit_uncertainty']['exp_r2']['ci95_high']:.4f}]"
    )
    print(
        f"  SGD poly R^2 mean±95%CI ≈ {sgd['fit_uncertainty']['poly_r2']['mean']:.4f} "
        f"[{sgd['fit_uncertainty']['poly_r2']['ci95_low']:.4f}, {sgd['fit_uncertainty']['poly_r2']['ci95_high']:.4f}]"
    )
    print(
        f"  Muon exp R^2  mean±95%CI ≈ {muon['fit_uncertainty']['exp_r2']['mean']:.4f} "
        f"[{muon['fit_uncertainty']['exp_r2']['ci95_low']:.4f}, {muon['fit_uncertainty']['exp_r2']['ci95_high']:.4f}]"
    )
    print(
        f"  Muon poly R^2 mean±95%CI ≈ {muon['fit_uncertainty']['poly_r2']['mean']:.4f} "
        f"[{muon['fit_uncertainty']['poly_r2']['ci95_low']:.4f}, {muon['fit_uncertainty']['poly_r2']['ci95_high']:.4f}]"
    )
    print("\nT3 diagnostic (the key honesty check for strong exponential wording):")
    print(
        f"  pass={t3['pass']} | mean R^2 gap exp-poly={t3['mean_r2_gap_exp_minus_poly']:.4f} | "
        f"layer votes exp/poly={t3['layer_vote_counts']['layers_preferring_exponential']}/"
        f"{t3['layer_vote_counts']['layers_preferring_polynomial']}"
    )
    print(
        f"  descriptive layer-level 95% CI for SGD gap: "
        f"[{t3['layerwise_r2_gap_summary']['ci95_low']:.4f}, {t3['layerwise_r2_gap_summary']['ci95_high']:.4f}]"
    )
    print(
        "  SGD layerwise R^2 gaps exp-poly: "
        + ", ".join(f"{value:.4f}" for value in t3["layerwise_r2_gap_values"])
    )

    print("\nFinal-spectrum / proxy / alignment summaries:")
    print(
        f"  Final Gini mean: SGD={sgd['gini_mean']:.4f}, Muon={muon['gini_mean']:.4f}"
    )
    print(
        f"  Lagged scalar proxy corr mean: SGD={sgd['lagged_corr_proxy_mean']:.4f}, Muon={muon['lagged_corr_proxy_mean']:.4f}"
    )
    print(
        f"  Same-state raw-grad corr mean: SGD={sgd['same_state_raw_grad_corr_mean']:.4f}, Muon={muon['same_state_raw_grad_corr_mean']:.4f}"
    )
    print(
        f"  Direct W/raw-grad top-vector overlap mean: SGD={sgd['weight_raw_grad_alignment_mean']:.4f}, Muon={muon['weight_raw_grad_alignment_mean']:.4f}"
    )
    print(
        f"  Direct W/update top-vector overlap mean: SGD={sgd['weight_update_alignment_mean']:.4f}, Muon={muon['weight_update_alignment_mean']:.4f}"
    )
    print(
        f"  Update-minus-raw overlap shift mean: SGD={sgd['weight_update_minus_raw_alignment_mean']:.4f}, Muon={muon['weight_update_minus_raw_alignment_mean']:.4f}"
    )

    print("\nVerdict components:")
    for name, info in summary["verdict_components"].items():
        status = "PASS" if info["pass"] else "FAIL"
        print(f"  - {name}: {status} -- {info['description']}")

    print("\nAssessment:")
    print(f"  {summary['assessment']['overall_label']}")
    for note in summary["assessment"]["notes"]:
        print(f"    * {note}")
    print("=" * 100)


# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================


def main() -> None:
    """Run experiment, save figure/artifacts, and print calibrated report."""
    results = run_experiment(verbose=True)
    _, plot_path = make_plots(results, output_dir=SCRIPT_DIR, filename="sigma1_growth_rate.png", show=False)
    artifact_paths = save_results_artifacts(results, output_dir=SCRIPT_DIR, plot_path=plot_path)
    results["artifact_paths"] = artifact_paths

    if plot_path is not None:
        print(f"Saved plot: {plot_path}")
    print(f"Saved summary JSON: {artifact_paths['summary_json']}")
    print(f"Saved raw NPZ: {artifact_paths['raw_npz']}")

    print_report(results)


if __name__ == "__main__":
    main()
