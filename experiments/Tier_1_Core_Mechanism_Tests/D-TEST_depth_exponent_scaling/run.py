#!/usr/bin/env python3
"""
D-TEST: deterministic deep-linear depth-scaling proxy
====================================================

This script runs a deterministic toy deep-linear experiment comparing
momentum SGD against Muon-style momentum updates with Newton-Schulz
gradient orthogonalization.

What it actually measures:
- a fixed-horizon loss comparison over 300 optimization steps,
- a semilog fit of the final-step Muon/SGD loss advantage versus depth,
- a heuristic stable learning-rate search for SGD at each depth,
- condition-number diagnostics for a representative depth,
- and simple steps-to-threshold speed metrics derived from the loss curves.

What it does not establish:
- a complexity-theoretic separation,
- an asymptotic proof,
- uncertainty estimates across seeds,
- or a true Hessian eigendecomposition.

The goal is narrower and empirical: assess whether this deterministic toy
setting exhibits a strong semilog depth-fit of fixed-horizon Muon-vs-SGD
advantage, and present the result honestly.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


EXPERIMENT_NAME = "D-TEST depth-exponent scaling"
EXPERIMENT_SCOPE = (
    "Deterministic deep-linear Muon vs heuristic stable-LR SGD depth-scaling proxy."
)

DEFAULT_CONFIG: Dict[str, Any] = {
    "input_dim": 32,
    "hidden_dim": 32,
    "output_dim": 32,
    "num_steps": 300,
    "batch_size": 64,
    "depths": [2, 3, 4, 6, 8, 12, 16],
    "measurement_steps": [50, 100, 150, 200, 250, 300],
    "lr_muon": 0.005,
    "momentum": 0.9,
    "ns_iters": 5,
    "problem_seed": 42,
    "lr_search_seed": 42,
    "init_seed_offset": 42,
    "condition_measurement_interval": 10,
    "analysis_depth": 8,
    "threshold_fractions": [0.5, 0.1, 0.01],
    "sgd_lr_cap": 0.1,
    "lr_search_validation_steps": 50,
    "lr_search_max_attempts": 10,
    "lr_search_instability_factor": 100.0,
    "main_instability_threshold": 1e10,
}


def get_default_config() -> Dict[str, Any]:
    """Return a deep copy of the default deterministic experiment config."""
    return copy.deepcopy(DEFAULT_CONFIG)


def make_problem_data(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Generate the fixed target matrix and fixed input batch deterministically."""
    rng = np.random.RandomState(config["problem_seed"])
    w_target = rng.randn(config["output_dim"], config["input_dim"]) * 0.5
    x_data = rng.randn(config["input_dim"], config["batch_size"]) * 0.3
    return w_target, x_data


def init_weights(num_layers: int, hidden_dim: int, rng: np.random.RandomState) -> List[np.ndarray]:
    """Initialize square layers near identity for stability."""
    weights: List[np.ndarray] = []
    for _ in range(num_layers):
        w = np.eye(hidden_dim) + rng.randn(hidden_dim, hidden_dim) * 0.1
        weights.append(w.copy())
    return weights


def forward(weights: Sequence[np.ndarray], x: np.ndarray) -> np.ndarray:
    """Forward pass: W_L @ ... @ W_1 @ X."""
    out = x.copy()
    for w in weights:
        out = w @ out
    return out


def compute_loss(weights: Sequence[np.ndarray], x: np.ndarray, target: np.ndarray) -> float:
    """Loss = 0.5 * ||W_product @ X - target @ X||^2 / N."""
    pred = forward(weights, x)
    target_out = target @ x
    diff = pred - target_out
    return float(0.5 * np.mean(np.sum(diff**2, axis=0)))


def compute_gradients(weights: Sequence[np.ndarray], x: np.ndarray, target: np.ndarray) -> List[np.ndarray]:
    """Backpropagation through the deep linear network."""
    num_layers = len(weights)
    num_examples = x.shape[1]

    activations = [x.copy()]
    out = x.copy()
    for w in weights:
        out = w @ out
        activations.append(out.copy())

    target_out = target @ x
    delta = (activations[-1] - target_out) / num_examples

    grads: List[np.ndarray] = []
    for layer_idx in range(num_layers - 1, -1, -1):
        grad = delta @ activations[layer_idx].T
        grads.insert(0, grad)
        if layer_idx > 0:
            delta = weights[layer_idx].T @ delta

    return grads


def newton_schulz_orthogonalize(grad: np.ndarray, num_iters: int) -> np.ndarray:
    """
    Approximate the orthogonal polar factor of a matrix via Newton-Schulz.

    For a matrix G with SVD U S V^T, this targets U V^T after a fixed number
    of iterations, without explicitly computing the SVD during optimization.
    """
    norm = np.linalg.norm(grad, ord="fro")
    if norm < 1e-12:
        return grad

    x = grad / norm
    for _ in range(num_iters):
        a = x.T @ x
        x = 1.5 * x - 0.5 * x @ a
    return x


def compute_curvature_proxy(weights: Sequence[np.ndarray], x: np.ndarray, hidden_dim: int) -> float:
    """
    Compute the product-norm curvature proxy used by the SGD LR heuristic.

    This is not a Hessian eigendecomposition. It is the same proxy used in the
    original script: sigma_max(W_prod)^2 * sigma_max(X)^2 / N.
    """
    w_prod = np.eye(hidden_dim)
    for w in weights:
        w_prod = w @ w_prod

    sv_prod = np.linalg.svd(w_prod, compute_uv=False)
    sv_x = np.linalg.svd(x, compute_uv=False)
    num_examples = x.shape[1]
    return float((sv_prod[0] ** 2) * (sv_x[0] ** 2) / num_examples)


def find_stable_lr_sgd(depth: int, config: Dict[str, Any], x_data: np.ndarray) -> Dict[str, Any]:
    """
    Heuristically search for a stable SGD learning rate at a given depth.

    Procedure preserved from the original experiment:
    - start from lr_theory = 2 / (lambda_proxy * depth), capped at 0.1,
    - validate stability for 50 steps,
    - halve on failure up to 10 attempts,
    - fall back to 1e-4 if needed.
    """
    hidden_dim = config["hidden_dim"]
    target, _ = make_problem_data(config)

    rng_proxy = np.random.RandomState(config["lr_search_seed"])
    proxy_weights = init_weights(depth, hidden_dim, rng_proxy)
    lambda_proxy = compute_curvature_proxy(proxy_weights, x_data, hidden_dim)

    lr_theory = 2.0 / (lambda_proxy * depth)
    lr = min(lr_theory, config["sgd_lr_cap"])
    attempt_history: List[Dict[str, Any]] = []

    for attempt in range(config["lr_search_max_attempts"]):
        rng_attempt = np.random.RandomState(config["lr_search_seed"])
        w_test = init_weights(depth, hidden_dim, rng_attempt)
        v_test = [np.zeros_like(w) for w in w_test]
        initial_loss = compute_loss(w_test, x_data, target)
        stable = True
        instability_step: Optional[int] = None

        for step in range(config["lr_search_validation_steps"]):
            grads = compute_gradients(w_test, x_data, target)
            for layer_idx in range(depth):
                v_test[layer_idx] = config["momentum"] * v_test[layer_idx] + grads[layer_idx]
                w_test[layer_idx] -= lr * v_test[layer_idx]

            current_loss = compute_loss(w_test, x_data, target)
            if np.isnan(current_loss) or current_loss > initial_loss * config["lr_search_instability_factor"]:
                stable = False
                instability_step = step + 1
                break

        attempt_history.append(
            {
                "attempt_index": attempt,
                "lr_tested": float(lr),
                "stable": bool(stable),
                "instability_step": instability_step,
            }
        )

        if stable:
            return {
                "lr": float(lr),
                "lambda_proxy": float(lambda_proxy),
                "lr_theory": float(lr_theory),
                "attempts": attempt_history,
                "fallback_used": False,
            }

        lr *= 0.5

    return {
        "lr": 0.0001,
        "lambda_proxy": float(lambda_proxy),
        "lr_theory": float(lr_theory),
        "attempts": attempt_history,
        "fallback_used": True,
    }


def condition_number(matrix: np.ndarray) -> float:
    """Compute a numerically clipped condition number."""
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values[-1] < 1e-12:
        return 1e12
    return float(singular_values[0] / singular_values[-1])


def _record_condition_snapshot(weights: Sequence[np.ndarray]) -> Dict[str, Any]:
    """Return per-layer and product condition-number diagnostics."""
    per_layer = [condition_number(w) for w in weights]
    product = np.eye(weights[0].shape[0])
    for w in weights:
        product = w @ product
    product_kappa = condition_number(product)
    sum_log_layers = float(np.sum(np.log(np.array(per_layer) + 1e-12)))
    return {
        "per_layer": per_layer,
        "product": float(product_kappa),
        "sum_log_layers": sum_log_layers,
    }


def compute_advantage_trajectory(losses_sgd: Sequence[float], losses_muon: Sequence[float]) -> List[float]:
    """Compute loss_sgd / loss_muon at each recorded step."""
    trajectory: List[float] = []
    for loss_sgd, loss_muon in zip(losses_sgd, losses_muon):
        if not np.isfinite(loss_sgd) or not np.isfinite(loss_muon):
            trajectory.append(float("nan"))
        elif loss_muon <= 1e-15:
            trajectory.append(float("inf"))
        else:
            trajectory.append(float(loss_sgd / loss_muon))
    return trajectory


def first_step_below_threshold(losses: Sequence[float], threshold: float) -> Optional[int]:
    """Return the first step at which the trajectory is at or below the threshold."""
    for step, loss in enumerate(losses):
        if np.isfinite(loss) and loss <= threshold:
            return step
    return None


def compute_threshold_metrics(
    losses_sgd: Sequence[float],
    losses_muon: Sequence[float],
    threshold_fractions: Sequence[float],
) -> List[Dict[str, Any]]:
    """Compute steps-to-threshold metrics relative to the initial loss."""
    initial_loss = float(losses_sgd[0])
    metrics: List[Dict[str, Any]] = []
    for fraction in threshold_fractions:
        threshold = initial_loss * fraction
        metrics.append(
            {
                "fraction": float(fraction),
                "absolute_loss": float(threshold),
                "sgd_step": first_step_below_threshold(losses_sgd, threshold),
                "muon_step": first_step_below_threshold(losses_muon, threshold),
            }
        )
    return metrics


def run_single_depth(
    depth: int,
    config: Dict[str, Any],
    target: np.ndarray,
    x_data: np.ndarray,
) -> Dict[str, Any]:
    """Run the deterministic training comparison at one depth."""
    lr_search = find_stable_lr_sgd(depth, config, x_data)
    lr_sgd = lr_search["lr"]

    rng_init = np.random.RandomState(config["init_seed_offset"] + depth)
    weights_sgd = init_weights(depth, config["hidden_dim"], rng_init)
    weights_muon = [w.copy() for w in weights_sgd]

    v_sgd = [np.zeros_like(w) for w in weights_sgd]
    v_muon = [np.zeros_like(w) for w in weights_muon]

    losses_sgd: List[float] = []
    losses_muon: List[float] = []
    condition_history = {
        "measurement_steps": [],
        "sgd": {"per_layer": [[] for _ in range(depth)], "product": [], "sum_log_layers": []},
        "muon": {"per_layer": [[] for _ in range(depth)], "product": [], "sum_log_layers": []},
    }

    sgd_active = True
    muon_active = True
    sgd_instability_step: Optional[int] = None
    muon_instability_step: Optional[int] = None

    for step in range(config["num_steps"] + 1):
        loss_sgd = compute_loss(weights_sgd, x_data, target) if sgd_active else float("nan")
        loss_muon = compute_loss(weights_muon, x_data, target) if muon_active else float("nan")
        losses_sgd.append(float(loss_sgd))
        losses_muon.append(float(loss_muon))

        if step % config["condition_measurement_interval"] == 0:
            condition_history["measurement_steps"].append(step)
            for optimizer_name, weights, active in (
                ("sgd", weights_sgd, sgd_active),
                ("muon", weights_muon, muon_active),
            ):
                if active:
                    snapshot = _record_condition_snapshot(weights)
                    for layer_idx, value in enumerate(snapshot["per_layer"]):
                        condition_history[optimizer_name]["per_layer"][layer_idx].append(float(value))
                    condition_history[optimizer_name]["product"].append(float(snapshot["product"]))
                    condition_history[optimizer_name]["sum_log_layers"].append(float(snapshot["sum_log_layers"]))
                else:
                    for layer_idx in range(depth):
                        condition_history[optimizer_name]["per_layer"][layer_idx].append(float("nan"))
                    condition_history[optimizer_name]["product"].append(float("nan"))
                    condition_history[optimizer_name]["sum_log_layers"].append(float("nan"))

        if step == config["num_steps"]:
            continue

        if sgd_active:
            grads_sgd = compute_gradients(weights_sgd, x_data, target)
            for layer_idx in range(depth):
                v_sgd[layer_idx] = config["momentum"] * v_sgd[layer_idx] + grads_sgd[layer_idx]
                weights_sgd[layer_idx] -= lr_sgd * v_sgd[layer_idx]
            next_loss_sgd = compute_loss(weights_sgd, x_data, target)
            if np.isnan(next_loss_sgd) or next_loss_sgd > config["main_instability_threshold"]:
                sgd_active = False
                sgd_instability_step = step + 1

        if muon_active:
            grads_muon = compute_gradients(weights_muon, x_data, target)
            for layer_idx in range(depth):
                ortho_grad = newton_schulz_orthogonalize(grads_muon[layer_idx], config["ns_iters"])
                v_muon[layer_idx] = config["momentum"] * v_muon[layer_idx] + ortho_grad
                weights_muon[layer_idx] -= config["lr_muon"] * v_muon[layer_idx]
            next_loss_muon = compute_loss(weights_muon, x_data, target)
            if np.isnan(next_loss_muon) or next_loss_muon > config["main_instability_threshold"]:
                muon_active = False
                muon_instability_step = step + 1

    advantage_trajectory = compute_advantage_trajectory(losses_sgd, losses_muon)
    advantage_ratios = {
        int(step): float(advantage_trajectory[step]) if step < len(advantage_trajectory) else float("nan")
        for step in config["measurement_steps"]
    }
    threshold_metrics = compute_threshold_metrics(
        losses_sgd,
        losses_muon,
        config["threshold_fractions"],
    )

    final_loss_sgd = float(losses_sgd[-1]) if np.isfinite(losses_sgd[-1]) else float("inf")
    final_loss_muon = float(losses_muon[-1]) if np.isfinite(losses_muon[-1]) else float("inf")

    return {
        "depth": int(depth),
        "init_seed": int(config["init_seed_offset"] + depth),
        "lr_search": lr_search,
        "lr_sgd": float(lr_sgd),
        "losses_sgd": losses_sgd,
        "losses_muon": losses_muon,
        "advantage_trajectory": advantage_trajectory,
        "advantage_ratios": advantage_ratios,
        "condition_numbers": condition_history,
        "threshold_steps": threshold_metrics,
        "instability": {
            "sgd_step": sgd_instability_step,
            "muon_step": muon_instability_step,
        },
        "initial_loss_sgd": float(losses_sgd[0]),
        "initial_loss_muon": float(losses_muon[0]),
        "final_loss_sgd": final_loss_sgd,
        "final_loss_muon": final_loss_muon,
    }


def _linear_fit(x_values: Sequence[float], y_values: Sequence[float]) -> Dict[str, Any]:
    """Least-squares line fit with residuals and R^2."""
    x_arr = np.array(x_values, dtype=float)
    y_arr = np.array(y_values, dtype=float)
    a_matrix = np.vstack([x_arr, np.ones(len(x_arr))]).T
    fit_result = np.linalg.lstsq(a_matrix, y_arr, rcond=None)
    slope, intercept = fit_result[0]
    fitted = slope * x_arr + intercept
    residuals = y_arr - fitted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    r_squared = 1.0 - ss_res / (ss_tot + 1e-15)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "fitted_y": [float(v) for v in fitted],
        "residuals": [float(v) for v in residuals],
        "ss_res": ss_res,
        "ss_tot": ss_tot,
    }


def analyze_primary_depth_fit(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze log(final-step advantage) versus depth."""
    config = results["config"]
    final_step = config["measurement_steps"][-1]
    depths_valid: List[int] = []
    advantages_valid: List[float] = []
    trainable_by_depth: Dict[int, bool] = {}

    for depth in config["depths"]:
        depth_result = results["per_depth"][depth]
        advantage = depth_result["advantage_ratios"].get(final_step, float("nan"))
        final_loss_sgd = depth_result["final_loss_sgd"]
        initial_loss = depth_result["initial_loss_sgd"]
        trainable = bool(np.isfinite(final_loss_sgd) and final_loss_sgd < initial_loss * 0.5)
        trainable_by_depth[depth] = trainable

        if np.isfinite(advantage) and advantage > 0:
            depths_valid.append(depth)
            advantages_valid.append(float(advantage))

    if len(depths_valid) < 3:
        return {
            "final_measurement_step": int(final_step),
            "valid_depths": depths_valid,
            "valid_advantages": advantages_valid,
            "log_advantages": [],
            "predicted_advantages": [],
            "residuals": [],
            "slope": 0.0,
            "intercept": 0.0,
            "r_squared": 0.0,
            "exp_base_per_layer": 1.0,
            "num_valid_points": len(depths_valid),
            "trainable_by_depth": trainable_by_depth,
            "monotone_non_decreasing": False,
            "fit_available": False,
        }

    log_advantages = np.log(np.array(advantages_valid))
    fit = _linear_fit(depths_valid, log_advantages)
    predicted_advantages = np.exp(np.array(fit["fitted_y"]))
    monotone_non_decreasing = all(
        advantages_valid[idx + 1] >= advantages_valid[idx] - 1e-12
        for idx in range(len(advantages_valid) - 1)
    )

    return {
        "final_measurement_step": int(final_step),
        "valid_depths": [int(d) for d in depths_valid],
        "valid_advantages": [float(v) for v in advantages_valid],
        "log_advantages": [float(v) for v in log_advantages],
        "predicted_advantages": [float(v) for v in predicted_advantages],
        "residuals": fit["residuals"],
        "slope": fit["slope"],
        "intercept": fit["intercept"],
        "r_squared": fit["r_squared"],
        "exp_base_per_layer": float(np.exp(fit["slope"])),
        "num_valid_points": len(depths_valid),
        "trainable_by_depth": trainable_by_depth,
        "monotone_non_decreasing": bool(monotone_non_decreasing),
        "fit_available": True,
    }


def analyze_time_scaling(results: Dict[str, Any]) -> Dict[str, Any]:
    """Secondary diagnostic: fit log advantage against log step at each depth."""
    config = results["config"]
    per_depth: Dict[int, Dict[str, Any]] = {}
    valid_pairs: List[Tuple[int, float]] = []

    for depth in config["depths"]:
        ratios = results["per_depth"][depth]["advantage_ratios"]
        steps_valid: List[int] = []
        log_ratios_valid: List[float] = []
        for step in config["measurement_steps"]:
            ratio = ratios.get(step, float("nan"))
            if np.isfinite(ratio) and ratio > 0:
                steps_valid.append(step)
                log_ratios_valid.append(float(np.log(ratio)))

        if len(steps_valid) >= 3:
            log_steps = np.log(np.array(steps_valid, dtype=float))
            fit = _linear_fit(log_steps, log_ratios_valid)
            slope = fit["slope"]
            per_depth[depth] = {
                "steps": [int(s) for s in steps_valid],
                "log_ratios": log_ratios_valid,
                "slope": float(slope),
                "ratio_to_depth": float(slope / depth),
                "r_squared": fit["r_squared"],
            }
            valid_pairs.append((depth, slope))
        else:
            per_depth[depth] = {
                "steps": [int(s) for s in steps_valid],
                "log_ratios": log_ratios_valid,
                "slope": float("nan"),
                "ratio_to_depth": float("nan"),
                "r_squared": float("nan"),
            }

    slope_of_slopes = float("nan")
    if len(valid_pairs) >= 3:
        depth_values = [pair[0] for pair in valid_pairs]
        slope_values = [pair[1] for pair in valid_pairs]
        slope_of_slopes = _linear_fit(depth_values, slope_values)["slope"]

    return {
        "per_depth": per_depth,
        "slope_of_slopes": slope_of_slopes,
        "note": (
            "Secondary diagnostic only. This fixed-horizon toy analysis does not by itself "
            "support complexity claims."
        ),
    }


def _fit_condition_series(measurement_steps: Sequence[int], series: Sequence[float]) -> Optional[Dict[str, Any]]:
    """Fit a condition-number-derived series if enough finite points exist."""
    x_values: List[int] = []
    y_values: List[float] = []
    for step, value in zip(measurement_steps, series):
        if np.isfinite(value):
            x_values.append(int(step))
            y_values.append(float(value))

    if len(x_values) < 3:
        return None

    fit = _linear_fit(x_values, y_values)
    fit["steps_used"] = x_values
    fit["y_used"] = y_values
    return fit


def analyze_condition_numbers(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Representative condition-number diagnostic with correct submultiplicative interpretation.

    For any product W_L ... W_1, kappa(product) <= prod_l kappa(W_l), hence
    log kappa(product) <= sum_l log kappa(W_l). The diagnostic here checks the
    actual trajectories and compares SGD/Muon growth cautiously.
    """
    config = results["config"]
    analysis_depth = config["analysis_depth"]
    if analysis_depth not in results["per_depth"]:
        return {
            "analysis_depth": analysis_depth,
            "available": False,
            "message": f"Depth {analysis_depth} not available in results.",
        }

    depth_result = results["per_depth"][analysis_depth]
    cond = depth_result["condition_numbers"]
    measurement_steps = cond["measurement_steps"]
    optimizers: Dict[str, Dict[str, Any]] = {}

    for optimizer_name in ("sgd", "muon"):
        optimizer_cond = cond[optimizer_name]
        product = np.array(optimizer_cond["product"], dtype=float)
        log_product = np.log(product + 1e-12)
        sum_log_layers = np.array(optimizer_cond["sum_log_layers"], dtype=float)
        upper_bound_gap = sum_log_layers - log_product
        finite_gap = upper_bound_gap[np.isfinite(upper_bound_gap)]
        upper_bound_respected = bool(np.all(finite_gap >= -1e-8)) if finite_gap.size else False

        layer_fits = []
        for layer_idx, series in enumerate(optimizer_cond["per_layer"]):
            log_series = np.log(np.array(series, dtype=float) + 1e-12)
            fit = _fit_condition_series(measurement_steps, log_series)
            layer_fits.append(
                {
                    "layer": int(layer_idx),
                    "fit": fit,
                    "initial_kappa": float(series[0]) if len(series) else float("nan"),
                    "final_kappa": float(series[-1]) if len(series) else float("nan"),
                }
            )

        product_fit = _fit_condition_series(measurement_steps, log_product)
        sum_fit = _fit_condition_series(measurement_steps, sum_log_layers)

        optimizers[optimizer_name] = {
            "per_layer": optimizer_cond["per_layer"],
            "product": optimizer_cond["product"],
            "sum_log_layers": optimizer_cond["sum_log_layers"],
            "log_product": [float(v) for v in log_product],
            "upper_bound_gap": [float(v) for v in upper_bound_gap],
            "upper_bound_respected": upper_bound_respected,
            "product_fit": product_fit,
            "sum_log_layers_fit": sum_fit,
            "layer_fits": layer_fits,
        }

    return {
        "analysis_depth": int(analysis_depth),
        "available": True,
        "measurement_steps": [int(step) for step in measurement_steps],
        "optimizers": optimizers,
        "interpretation": (
            "Because log kappa(product) is upper-bounded by the sum of per-layer "
            "log condition numbers, a product-growth slope smaller than the summed "
            "layer slope is expected and not a failure mode."
        ),
    }


def summarize_verdict(primary_fit: Dict[str, Any]) -> Dict[str, Any]:
    """Convert the primary fit into a sober experiment verdict."""
    if not primary_fit["fit_available"]:
        return {
            "label": "INCONCLUSIVE",
            "summary": "Insufficient valid depth points for the semilog fit.",
        }

    r_squared = primary_fit["r_squared"]
    slope = primary_fit["slope"]
    base = primary_fit["exp_base_per_layer"]

    if r_squared > 0.9:
        label = "STRONG SEMILOG DEPTH-FIT"
        summary = (
            f"The deterministic toy run shows a strong semilog fit of final-step advantage vs depth "
            f"(slope={slope:.4f}, base≈{base:.4f}, R^2={r_squared:.4f})."
        )
    elif r_squared > 0.7:
        label = "SUGGESTIVE BUT NOISY SEMILOG DEPTH-FIT"
        summary = (
            f"The deterministic toy run suggests semilog growth, but the fit is not especially clean "
            f"(slope={slope:.4f}, base≈{base:.4f}, R^2={r_squared:.4f})."
        )
    else:
        label = "NO STRONG SEMILOG DEPTH-FIT"
        summary = (
            f"The deterministic toy run does not show a strong semilog fit under this metric "
            f"(slope={slope:.4f}, base≈{base:.4f}, R^2={r_squared:.4f})."
        )

    return {"label": label, "summary": summary}


def run_full_experiment(config: Optional[Dict[str, Any]] = None, verbose: bool = False) -> Dict[str, Any]:
    """Run the full deterministic depth sweep and all analyses."""
    cfg = get_default_config()
    if config is not None:
        cfg.update(copy.deepcopy(config))

    target, x_data = make_problem_data(cfg)
    per_depth: Dict[int, Dict[str, Any]] = {}

    for depth in cfg["depths"]:
        per_depth[depth] = run_single_depth(depth, cfg, target, x_data)

    results: Dict[str, Any] = {
        "experiment_name": EXPERIMENT_NAME,
        "scope": EXPERIMENT_SCOPE,
        "config": cfg,
        "seeds": {
            "problem_seed": cfg["problem_seed"],
            "lr_search_seed": cfg["lr_search_seed"],
            "init_seed_offset": cfg["init_seed_offset"],
            "init_seed_rule": "depth_init_seed = init_seed_offset + depth",
        },
        "problem_data": {
            "target_shape": list(target.shape),
            "input_shape": list(x_data.shape),
            "target_fro_norm": float(np.linalg.norm(target, ord="fro")),
            "input_fro_norm": float(np.linalg.norm(x_data, ord="fro")),
        },
        "per_depth": per_depth,
    }

    primary_fit = analyze_primary_depth_fit(results)
    time_scaling = analyze_time_scaling(results)
    condition_numbers = analyze_condition_numbers(results)
    verdict = summarize_verdict(primary_fit)

    results["analyses"] = {
        "primary_depth_fit": primary_fit,
        "time_scaling": time_scaling,
        "condition_numbers": condition_numbers,
        "verdict": verdict,
    }

    if verbose:
        print_text_report(results)

    return results


def _format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return "NaN"
        if np.isinf(value):
            return "INF"
        magnitude = abs(float(value))
        if magnitude >= 1e4 or (magnitude > 0 and magnitude < 1e-3):
            return f"{float(value):.{digits}e}"
        return f"{float(value):.{digits}f}"
    return str(value)


def print_text_report(results: Dict[str, Any]) -> None:
    """Render a sober text report matching the notebook's scope."""
    config = results["config"]
    primary = results["analyses"]["primary_depth_fit"]
    time_scaling = results["analyses"]["time_scaling"]
    condition_analysis = results["analyses"]["condition_numbers"]
    verdict = results["analyses"]["verdict"]

    print("=" * 100)
    print("D-TEST: DETERMINISTIC DEEP-LINEAR DEPTH-SCALING PROXY")
    print("=" * 100)
    print(results["scope"])
    print(f"Setup: deep linear net (dim={config['hidden_dim']}), quadratic loss, {config['num_steps']} steps")
    print(f"Depths: {config['depths']}")
    print(f"Muon LR={config['lr_muon']} (fixed), Momentum={config['momentum']}, NS iters={config['ns_iters']}")
    print("SGD LR: heuristic max-stable search from a product-norm curvature proxy")
    print("Caveat: this is a deterministic toy experiment, not a complexity proof.")

    for depth in config["depths"]:
        depth_result = results["per_depth"][depth]
        print(f"\n{'─' * 100}")
        print(f"  DEPTH L={depth}")
        print(f"{'─' * 100}")
        print(f"  SGD LR (heuristic max stable): {depth_result['lr_sgd']:.6f}")
        print(f"  Curvature proxy: {depth_result['lr_search']['lambda_proxy']:.6f}")
        print(f"  Theory LR before cap: {depth_result['lr_search']['lr_theory']:.6f}")
        print(f"  Final loss SGD:  {depth_result['final_loss_sgd']:.6e}")
        print(f"  Final loss Muon: {depth_result['final_loss_muon']:.6e}")
        print("  Advantage ratios at measurement steps:")
        for measurement_step in config["measurement_steps"]:
            advantage = depth_result["advantage_ratios"].get(measurement_step, float("nan"))
            print(f"    Step {measurement_step:3d}: {_format_float(advantage)}x")
        print("  Steps to relative-loss thresholds:")
        for entry in depth_result["threshold_steps"]:
            print(
                f"    <= {entry['fraction']:.0%} of initial loss: "
                f"SGD step={_format_float(entry['sgd_step'])}, "
                f"Muon step={_format_float(entry['muon_step'])}"
            )

    print("\n\n" + "=" * 100)
    print("PRIMARY ANALYSIS: SEMILOG DEPTH-FIT OF FINAL-STEP ADVANTAGE")
    print("=" * 100)
    final_step = primary["final_measurement_step"]
    print(f"Final measurement step used: {final_step}")
    print(f"Valid data points: {primary['num_valid_points']} / {len(config['depths'])}")
    print(f"\n{'Depth':>6} | {'Advantage':>12} | {'log(Advantage)':>14} | {'SGD trainable?':>14}")
    print("-" * 60)
    for depth, advantage, log_advantage in zip(
        primary["valid_depths"],
        primary["valid_advantages"],
        primary["log_advantages"],
    ):
        trainable_str = "YES" if primary["trainable_by_depth"][depth] else "NO"
        print(f"{depth:6d} | {advantage:12.4f} | {log_advantage:14.4f} | {trainable_str:>14}")

    if primary["fit_available"]:
        print(f"\nLinear fit: log(advantage) = {primary['slope']:.4f} * L + ({primary['intercept']:.4f})")
        print(f"  Exponential base per added layer: e^a = {primary['exp_base_per_layer']:.4f}")
        print(f"  R^2 = {primary['r_squared']:.6f}")
        print(f"  Monotone non-decreasing by depth? {primary['monotone_non_decreasing']}")
        print(f"\n{'Depth':>6} | {'Measured':>12} | {'Predicted (fit)':>15} | {'Residual log-space':>18}")
        print("-" * 62)
        for depth, measured, predicted, residual in zip(
            primary["valid_depths"],
            primary["valid_advantages"],
            primary["predicted_advantages"],
            primary["residuals"],
        ):
            print(f"{depth:6d} | {measured:12.4f} | {predicted:15.4f} | {residual:18.4f}")
    else:
        print("Insufficient valid points for the semilog fit.")

    print("\n\n" + "=" * 100)
    print("SECONDARY DIAGNOSTIC: LOG(ADVANTAGE) VS LOG(STEP)")
    print("=" * 100)
    print(time_scaling["note"])
    print(f"\n{'Depth':>6} | {'Slope':>10} | {'Slope / depth':>14} | {'R^2':>8}")
    print("-" * 50)
    for depth in config["depths"]:
        entry = time_scaling["per_depth"][depth]
        print(
            f"{depth:6d} | {_format_float(entry['slope']):>10} | "
            f"{_format_float(entry['ratio_to_depth']):>14} | {_format_float(entry['r_squared']):>8}"
        )
    print(f"Slope-of-slopes vs depth: {_format_float(time_scaling['slope_of_slopes'])}")

    print("\n\n" + "=" * 100)
    print("CONDITION-NUMBER DIAGNOSTIC")
    print("=" * 100)
    if condition_analysis["available"]:
        analysis_depth = condition_analysis["analysis_depth"]
        print(f"Representative depth: L={analysis_depth}")
        print(condition_analysis["interpretation"])
        for optimizer_name in ("sgd", "muon"):
            entry = condition_analysis["optimizers"][optimizer_name]
            product_fit = entry["product_fit"]
            sum_fit = entry["sum_log_layers_fit"]
            ratio = float("nan")
            if product_fit and sum_fit and abs(sum_fit["slope"]) > 1e-15:
                ratio = product_fit["slope"] / sum_fit["slope"]
            print(f"\n  Optimizer: {optimizer_name.upper()}")
            print(f"    Upper-bound respected? {entry['upper_bound_respected']}")
            if product_fit:
                print(f"    log kappa(product) slope: {_format_float(product_fit['slope'], 6)}")
                print(f"    log kappa(product) R^2:   {_format_float(product_fit['r_squared'], 6)}")
            if sum_fit:
                print(f"    sum log kappa(layer) slope: {_format_float(sum_fit['slope'], 6)}")
                print(f"    sum log kappa(layer) R^2:   {_format_float(sum_fit['r_squared'], 6)}")
                print(f"    slope ratio product/sum:    {_format_float(ratio, 6)}")
    else:
        print(condition_analysis["message"])

    print("\n\n" + "=" * 100)
    print("DEPTH SUMMARY TABLE")
    print("=" * 100)
    print(
        f"\n{'Depth':>6} | {'Final advantage':>15} | {'SGD LR':>10} | {'Final SGD':>12} | {'Final Muon':>12} | {'SGD trainable?':>14}"
    )
    print("-" * 88)
    for depth in config["depths"]:
        depth_result = results["per_depth"][depth]
        final_advantage = depth_result["advantage_ratios"].get(primary["final_measurement_step"], float("nan"))
        trainable = primary["trainable_by_depth"].get(depth, False)
        print(
            f"{depth:6d} | {_format_float(final_advantage):>15} | {depth_result['lr_sgd']:10.6f} | "
            f"{depth_result['final_loss_sgd']:12.4e} | {depth_result['final_loss_muon']:12.4e} | {str(trainable):>14}"
        )

    print("\n" + "=" * 100)
    print("FINAL VERDICT")
    print("=" * 100)
    print(f"{verdict['label']}: {verdict['summary']}")
    print("Limitations:")
    print("  - single deterministic target/data draw and deterministic initialization scheme")
    print("  - fixed 300-step horizon")
    print("  - heuristic stable-LR baseline for SGD, not globally optimized SGD")
    print("  - product-norm curvature proxy rather than a true Hessian spectral calculation")
    print("  - no uncertainty estimates or asymptotic proof")
    print("=" * 100)


def _to_serializable(obj: Any) -> Any:
    """Recursively convert numpy-heavy structures to JSON-serializable values."""
    if isinstance(obj, dict):
        return {str(key): _to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(value) for value in obj]
    if isinstance(obj, tuple):
        return [_to_serializable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def save_results_json(results: Dict[str, Any], output_path: Path) -> None:
    """Save structured experiment results as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_to_serializable(results), indent=2))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path for structured JSON results.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Run the experiment without printing the text report.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    results = run_full_experiment(verbose=not args.quiet)
    if args.save_json is not None:
        save_results_json(results, args.save_json)
        if not args.quiet:
            print(f"Saved structured results to {args.save_json}")
    return results


if __name__ == "__main__":
    main()
