#!/usr/bin/env python3
"""
1.3c-i: Isolate lambda_max from alpha -- single-seed per-layer spectrum proxy study
===================================================================================

This experiment compares SGD-with-momentum and Muon on a fixed deep linear toy problem.
It tracks two quantities separately for each layer's W^T W spectrum:

  1. lambda_max and lambda_max / lambda_median
  2. a simple rank-decay alpha proxy

Important scope note:
  - The alpha reported here is NOT a full WeightWatcher fit.
  - It is a crude but deterministic proxy obtained by regressing
    log(eigenvalue) against log(rank) on the per-layer W^T W eigenvalues.
  - Results should therefore be read as a single-seed per-layer spectrum proxy
    study, not as a calibrated heavy-tail or generalization analysis.

Default setup (preserved from the legacy script as closely as possible):
  - 4-layer deep linear net
  - 32 x 32 layers
  - fixed synthetic target matrix and fixed input batch
  - 500 training steps
  - SGD with momentum vs Muon
  - measurements every 25 steps
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SEED = 42
DEFAULT_TABLE_STEPS = [0, 100, 200, 300, 500]


def get_default_config() -> Dict[str, Any]:
    """Return the preserved default configuration for the experiment."""
    return {
        "experiment_id": "1.3c-i_Isolate_lambda_max_from_alpha",
        "dim": 32,
        "num_layers": 4,
        "num_steps": 500,
        "batch_size": 64,
        "lr_muon": 0.005,
        "momentum": 0.9,
        "ns_iters": 5,
        "measure_every": 25,
        "target_scale": 0.5,
        "input_scale": 0.3,
        "init_noise_scale": 0.1,
        "table_steps": list(DEFAULT_TABLE_STEPS),
        "sgd_lr_candidates": [0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.001],
        "sgd_lr_search_steps": 200,
        "sgd_divergence_factor": 50.0,
    }


def normalize_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Merge user config into defaults and normalize a few derived fields."""
    merged = get_default_config()
    if config:
        merged.update(config)

    merged["table_steps"] = sorted(
        {
            int(step)
            for step in merged.get("table_steps", [])
            if 0 <= int(step) <= int(merged["num_steps"])
        }
    )
    if int(merged["num_steps"]) not in merged["table_steps"]:
        merged["table_steps"].append(int(merged["num_steps"]))
        merged["table_steps"] = sorted(set(merged["table_steps"]))

    return merged


def make_fixed_problem(config: Dict[str, Any], seed: int = DEFAULT_SEED) -> Tuple[np.ndarray, np.ndarray]:
    """Create the fixed target matrix and input batch, matching legacy seeding behavior."""
    rng = np.random.RandomState(seed)
    dim = int(config["dim"])
    batch_size = int(config["batch_size"])
    w_target = rng.randn(dim, dim) * float(config["target_scale"])
    x_data = rng.randn(dim, batch_size) * float(config["input_scale"])
    return w_target, x_data


def init_weights(
    num_layers: int,
    dim: int,
    seed: int = DEFAULT_SEED,
    init_noise_scale: float = 0.1,
) -> List[np.ndarray]:
    """Initialize all layers near the identity for stability."""
    rng = np.random.RandomState(seed)
    weights: List[np.ndarray] = []
    for _ in range(num_layers):
        w = np.eye(dim) + rng.randn(dim, dim) * init_noise_scale
        weights.append(w.copy())
    return weights


def forward(weights: List[np.ndarray], x: np.ndarray) -> np.ndarray:
    """Forward pass through the deep linear network."""
    out = x.copy()
    for w in weights:
        out = w @ out
    return out


def compute_loss(weights: List[np.ndarray], x: np.ndarray, target: np.ndarray) -> float:
    """Quadratic loss = 0.5 * ||W_product @ X - T @ X||^2 / N."""
    pred = forward(weights, x)
    target_out = target @ x
    diff = pred - target_out
    return 0.5 * np.mean(np.sum(diff**2, axis=0))


def compute_gradients(weights: List[np.ndarray], x: np.ndarray, target: np.ndarray) -> List[np.ndarray]:
    """Backpropagation for the deep linear network."""
    num_layers = len(weights)
    batch_size = x.shape[1]

    activations = [x.copy()]
    out = x.copy()
    for w in weights:
        out = w @ out
        activations.append(out.copy())

    target_out = target @ x
    delta = (activations[-1] - target_out) / batch_size

    grads: List[np.ndarray] = []
    for idx in range(num_layers - 1, -1, -1):
        grad = delta @ activations[idx].T
        grads.insert(0, grad)
        if idx > 0:
            delta = weights[idx].T @ delta

    return grads


def newton_schulz_orthogonalize(grad: np.ndarray, num_iters: int) -> np.ndarray:
    """Approximate the orthogonal polar factor of grad via Newton-Schulz iteration."""
    norm = np.linalg.norm(grad, ord="fro")
    if norm < 1e-12:
        return grad

    x = grad / norm
    for _ in range(num_iters):
        a = x.T @ x
        x = 1.5 * x - 0.5 * x @ a
    return x


# =============================================================================
# Alpha proxy fitting
# =============================================================================


def fit_power_law_alpha(eigenvalues: np.ndarray) -> float:
    """
    Fit a simple rank-decay alpha proxy from the eigenvalues of W^T W.

    Method:
      - sort eigenvalues descending
      - regress log(eigenvalue) against log(rank)
      - return alpha_proxy = -slope

    This is intentionally described as a proxy rather than a full WeightWatcher fit.
    """
    eigs = np.sort(np.asarray(eigenvalues))[::-1]
    eigs = eigs[eigs > 1e-30]
    n = len(eigs)
    if n < 3:
        return float("nan")

    ranks = np.arange(1, n + 1, dtype=float)
    log_rank = np.log(ranks)
    log_eig = np.log(eigs)

    design = np.vstack([log_rank, np.ones(n)]).T
    slope = np.linalg.lstsq(design, log_eig, rcond=None)[0][0]
    return float(-slope)


def fit_power_law_alpha_clipped(eigenvalues: np.ndarray) -> float:
    """
    Same proxy as fit_power_law_alpha, but excluding the top eigenvalue.

    This is only a clip-xmax-inspired diagnostic, not a full WeightWatcher clip_xmax analysis.
    """
    eigs = np.sort(np.asarray(eigenvalues))[::-1]
    eigs = eigs[1:]
    eigs = eigs[eigs > 1e-30]
    n = len(eigs)
    if n < 3:
        return float("nan")

    ranks = np.arange(1, n + 1, dtype=float)
    log_rank = np.log(ranks)
    log_eig = np.log(eigs)

    design = np.vstack([log_rank, np.ones(n)]).T
    slope = np.linalg.lstsq(design, log_eig, rcond=None)[0][0]
    return float(-slope)


# =============================================================================
# Optimizer helpers
# =============================================================================


def sgd_step(
    weights: List[np.ndarray],
    velocities: List[np.ndarray],
    lr: float,
    x_data: np.ndarray,
    w_target: np.ndarray,
    momentum: float,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """One step of SGD with standard momentum (not Nesterov)."""
    grads = compute_gradients(weights, x_data, w_target)
    for idx in range(len(weights)):
        velocities[idx] = momentum * velocities[idx] + grads[idx]
        weights[idx] = weights[idx] - lr * velocities[idx]
    return weights, velocities


def muon_step(
    weights: List[np.ndarray],
    velocities: List[np.ndarray],
    lr: float,
    x_data: np.ndarray,
    w_target: np.ndarray,
    momentum: float,
    ns_iters: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """One step of Muon with standard momentum plus orthogonalized gradients."""
    grads = compute_gradients(weights, x_data, w_target)
    for idx in range(len(weights)):
        ortho_grad = newton_schulz_orthogonalize(grads[idx], num_iters=ns_iters)
        velocities[idx] = momentum * velocities[idx] + ortho_grad
        weights[idx] = weights[idx] - lr * velocities[idx]
    return weights, velocities


def find_stable_lr_sgd(
    config: Dict[str, Any],
    x_data: np.ndarray,
    w_target: np.ndarray,
    seed: int = DEFAULT_SEED,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Heuristic learning-rate scan for SGD.

    This preserves the legacy behavior: choose the first candidate that does not diverge
    within a short fixed-length run. It is useful for a toy comparison, but it should not be
    over-interpreted as a definitive fairness guarantee.
    """
    dim = int(config["dim"])
    num_layers = int(config["num_layers"])
    momentum = float(config["momentum"])
    search_steps = int(config["sgd_lr_search_steps"])
    divergence_factor = float(config["sgd_divergence_factor"])
    init_noise_scale = float(config["init_noise_scale"])

    scan: List[Dict[str, Any]] = []

    for lr in config["sgd_lr_candidates"]:
        weights = init_weights(num_layers, dim, seed=seed, init_noise_scale=init_noise_scale)
        velocities = [np.zeros((dim, dim)) for _ in range(num_layers)]
        initial_loss = compute_loss(weights, x_data, w_target)
        stable = True
        steps_completed = 0
        max_loss_seen = initial_loss

        for step in range(1, search_steps + 1):
            weights, velocities = sgd_step(weights, velocities, lr, x_data, w_target, momentum)
            loss = compute_loss(weights, x_data, w_target)
            max_loss_seen = max(max_loss_seen, loss)
            steps_completed = step
            if np.isnan(loss) or loss > initial_loss * divergence_factor:
                stable = False
                break

        scan.append(
            {
                "lr": float(lr),
                "stable": bool(stable),
                "steps_completed": int(steps_completed),
                "initial_loss": float(initial_loss),
                "max_loss_seen": float(max_loss_seen),
            }
        )

        if stable:
            return float(lr), scan

    return float(config["sgd_lr_candidates"][-1]), scan


# =============================================================================
# Measurement engine
# =============================================================================


def compute_layer_spectrum_stats(weight_matrix: np.ndarray) -> Dict[str, Any]:
    """Compute per-layer W^T W eigenvalue statistics."""
    wtw = weight_matrix.T @ weight_matrix
    eigs = np.linalg.eigvalsh(wtw)[::-1]
    eigs = np.maximum(eigs, 0.0)

    lambda_max = float(eigs[0])
    lambda_min = float(eigs[-1])
    lambda_median = float(np.median(eigs))
    alpha = fit_power_law_alpha(eigs)
    alpha_clipped = fit_power_law_alpha_clipped(eigs)
    outlier_ratio = float(lambda_max / lambda_median) if lambda_median > 1e-30 else float("inf")

    return {
        "eigenvalues": eigs,
        "lambda_max": lambda_max,
        "lambda_median": lambda_median,
        "lambda_min": lambda_min,
        "alpha": alpha,
        "alpha_clipped": alpha_clipped,
        "outlier_ratio": outlier_ratio,
    }


def collect_layer_snapshot(weights: List[np.ndarray]) -> Dict[str, Any]:
    """Collect per-layer spectrum stats into stacked numeric arrays."""
    layer_stats = [compute_layer_spectrum_stats(weight_matrix) for weight_matrix in weights]
    return {
        "layer_stats": layer_stats,
        "eigenvalues": np.stack([stat["eigenvalues"] for stat in layer_stats], axis=0),
        "alpha": np.array([stat["alpha"] for stat in layer_stats], dtype=float),
        "alpha_clipped": np.array([stat["alpha_clipped"] for stat in layer_stats], dtype=float),
        "lambda_max": np.array([stat["lambda_max"] for stat in layer_stats], dtype=float),
        "lambda_median": np.array([stat["lambda_median"] for stat in layer_stats], dtype=float),
        "lambda_min": np.array([stat["lambda_min"] for stat in layer_stats], dtype=float),
        "outlier_ratio": np.array([stat["outlier_ratio"] for stat in layer_stats], dtype=float),
    }


def summarize_optimizer_run(run_result: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Compute layer-mean trajectories for the recorded metrics."""
    return {
        "alpha_mean": np.nanmean(run_result["alpha"], axis=1),
        "alpha_clipped_mean": np.nanmean(run_result["alpha_clipped"], axis=1),
        "lambda_max_mean": np.nanmean(run_result["lambda_max"], axis=1),
        "lambda_median_mean": np.nanmean(run_result["lambda_median"], axis=1),
        "lambda_min_mean": np.nanmean(run_result["lambda_min"], axis=1),
        "outlier_ratio_mean": np.nanmean(run_result["outlier_ratio"], axis=1),
        "clip_effect_mean": np.nanmean(np.abs(run_result["alpha"] - run_result["alpha_clipped"]), axis=1),
        "losses": run_result["losses"],
    }


def run_and_measure(
    optimizer_name: str,
    optimizer_kind: str,
    lr: float,
    config: Dict[str, Any],
    x_data: np.ndarray,
    w_target: np.ndarray,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """Run one optimizer and record per-step per-layer spectrum proxy statistics."""
    dim = int(config["dim"])
    num_layers = int(config["num_layers"])
    num_steps = int(config["num_steps"])
    measure_every = int(config["measure_every"])
    momentum = float(config["momentum"])
    ns_iters = int(config["ns_iters"])
    init_noise_scale = float(config["init_noise_scale"])

    weights = init_weights(num_layers, dim, seed=seed, init_noise_scale=init_noise_scale)
    initial_weights = np.stack([weight.copy() for weight in weights], axis=0)
    velocities = [np.zeros((dim, dim)) for _ in range(num_layers)]

    measure_steps = list(range(0, num_steps + 1, measure_every))
    if num_steps not in measure_steps:
        measure_steps.append(num_steps)
    measure_steps = np.array(sorted(set(measure_steps)), dtype=int)
    n_measures = len(measure_steps)

    shape = (n_measures, num_layers)
    alpha_all = np.full(shape, np.nan)
    alpha_clipped_all = np.full(shape, np.nan)
    lambda_max_all = np.full(shape, np.nan)
    lambda_median_all = np.full(shape, np.nan)
    lambda_min_all = np.full(shape, np.nan)
    outlier_ratio_all = np.full(shape, np.nan)
    losses = np.full(n_measures, np.nan)

    def store_snapshot(measure_index: int, snapshot: Dict[str, Any], loss_value: float) -> None:
        alpha_all[measure_index] = snapshot["alpha"]
        alpha_clipped_all[measure_index] = snapshot["alpha_clipped"]
        lambda_max_all[measure_index] = snapshot["lambda_max"]
        lambda_median_all[measure_index] = snapshot["lambda_median"]
        lambda_min_all[measure_index] = snapshot["lambda_min"]
        outlier_ratio_all[measure_index] = snapshot["outlier_ratio"]
        losses[measure_index] = loss_value

    run_start = time.perf_counter()
    initial_snapshot = collect_layer_snapshot(weights)
    initial_loss = compute_loss(weights, x_data, w_target)
    store_snapshot(0, initial_snapshot, initial_loss)

    measure_index = 1
    diverged = False
    divergence_step: Optional[int] = None
    latest_loss = initial_loss

    for step in range(1, num_steps + 1):
        if optimizer_kind.lower() == "sgd":
            weights, velocities = sgd_step(weights, velocities, lr, x_data, w_target, momentum)
        elif optimizer_kind.lower() == "muon":
            weights, velocities = muon_step(weights, velocities, lr, x_data, w_target, momentum, ns_iters)
        else:
            raise ValueError(f"Unknown optimizer_kind={optimizer_kind!r}")

        latest_loss = compute_loss(weights, x_data, w_target)
        if np.isnan(latest_loss) or latest_loss > 1e10:
            diverged = True
            divergence_step = step
            break

        if measure_index < n_measures and step == int(measure_steps[measure_index]):
            snapshot = collect_layer_snapshot(weights)
            store_snapshot(measure_index, snapshot, latest_loss)
            measure_index += 1

    runtime_seconds = time.perf_counter() - run_start

    final_snapshot = collect_layer_snapshot(weights)
    final_weights = np.stack([weight.copy() for weight in weights], axis=0)

    run_result = {
        "optimizer_name": optimizer_name,
        "optimizer_kind": optimizer_kind,
        "learning_rate": float(lr),
        "measure_steps": measure_steps,
        "alpha": alpha_all,
        "alpha_clipped": alpha_clipped_all,
        "lambda_max": lambda_max_all,
        "lambda_median": lambda_median_all,
        "lambda_min": lambda_min_all,
        "outlier_ratio": outlier_ratio_all,
        "losses": losses,
        "diverged": bool(diverged),
        "divergence_step": divergence_step,
        "steps_completed": int(divergence_step if diverged else num_steps),
        "runtime_seconds": float(runtime_seconds),
        "initial_weights": initial_weights,
        "final_weights": final_weights,
        "initial_layer_stats": initial_snapshot["layer_stats"],
        "final_layer_stats": final_snapshot["layer_stats"],
        "initial_eigenvalues": initial_snapshot["eigenvalues"],
        "final_eigenvalues": final_snapshot["eigenvalues"],
        "initial_loss": float(initial_loss),
        "final_loss": float(latest_loss),
    }
    run_result["trajectory_summary"] = summarize_optimizer_run(run_result)
    return run_result


def step_idx(measure_steps: np.ndarray, step_value: int) -> Optional[int]:
    """Return the index of step_value inside measure_steps, or None if absent."""
    idx = np.where(np.asarray(measure_steps) == step_value)[0]
    return int(idx[0]) if len(idx) > 0 else None


# =============================================================================
# Evaluation and reporting helpers
# =============================================================================


def _test_row(
    label: str,
    claim: str,
    metric_name: str,
    sgd_value: float,
    muon_value: float,
    passed: bool,
    supportive_direction: str,
    note: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    outcome_text = "SUPPORTED" if passed else "NOT SUPPORTED"
    if supportive_direction == "sgd_gt_muon":
        comparison_text = f"SGD > Muon is supportive; observed {sgd_value:.4f} vs {muon_value:.4f}."
    elif supportive_direction == "muon_lt_sgd":
        comparison_text = f"Muon < SGD is supportive; observed {muon_value:.4f} vs {sgd_value:.4f}."
    else:
        comparison_text = f"Observed SGD={sgd_value:.4f}, Muon={muon_value:.4f}."

    row = {
        "label": label,
        "claim": claim,
        "metric_name": metric_name,
        "sgd_value": float(sgd_value),
        "muon_value": float(muon_value),
        "passed": bool(passed),
        "outcome_text": outcome_text,
        "comparison_text": comparison_text,
        "note": note,
    }
    if extra:
        row.update(extra)
    return row


def evaluate_checks(results_sgd: Dict[str, Any], results_muon: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the four deterministic checks used by the legacy script."""
    sgd_summary = results_sgd["trajectory_summary"]
    muon_summary = results_muon["trajectory_summary"]

    alpha_mean_sgd = sgd_summary["alpha_mean"]
    alpha_mean_muon = muon_summary["alpha_mean"]
    alpha_clipped_mean_sgd = sgd_summary["alpha_clipped_mean"]
    alpha_clipped_mean_muon = muon_summary["alpha_clipped_mean"]
    outlier_mean_sgd = sgd_summary["outlier_ratio_mean"]
    outlier_mean_muon = muon_summary["outlier_ratio_mean"]

    alpha_drift_sgd = float(abs(alpha_mean_sgd[-1] - alpha_mean_sgd[0]))
    alpha_drift_muon = float(abs(alpha_mean_muon[-1] - alpha_mean_muon[0]))
    alpha_change_sgd = float(alpha_mean_sgd[-1] - alpha_mean_sgd[0])
    alpha_change_muon = float(alpha_mean_muon[-1] - alpha_mean_muon[0])

    alpha_final_sgd = float(alpha_mean_sgd[-1])
    alpha_final_muon = float(alpha_mean_muon[-1])

    lmax_growth_sgd = float(sgd_summary["lambda_max_mean"][-1] / sgd_summary["lambda_max_mean"][0])
    lmax_growth_muon = float(muon_summary["lambda_max_mean"][-1] / muon_summary["lambda_max_mean"][0])

    clip_effect_sgd = float(sgd_summary["clip_effect_mean"][-1])
    clip_effect_muon = float(muon_summary["clip_effect_mean"][-1])

    test1 = bool(alpha_drift_sgd > alpha_drift_muon)
    test2 = bool(alpha_final_muon < alpha_final_sgd)
    test3 = bool(lmax_growth_sgd > lmax_growth_muon)
    test4 = bool(clip_effect_sgd > clip_effect_muon)

    tests = {
        "T1": _test_row(
            label="T1",
            claim="SGD alpha proxy drifts more than Muon over training.",
            metric_name="|alpha_proxy(final) - alpha_proxy(init)|",
            sgd_value=alpha_drift_sgd,
            muon_value=alpha_drift_muon,
            passed=test1,
            supportive_direction="sgd_gt_muon",
            note="Single-seed drift comparison on the layer-mean rank-decay alpha proxy.",
            extra={
                "sgd_signed_change": alpha_change_sgd,
                "muon_signed_change": alpha_change_muon,
                "sgd_initial": float(alpha_mean_sgd[0]),
                "muon_initial": float(alpha_mean_muon[0]),
                "sgd_final": alpha_final_sgd,
                "muon_final": alpha_final_muon,
            },
        ),
        "T2": _test_row(
            label="T2",
            claim="Muon finishes with a flatter/lower alpha proxy than SGD.",
            metric_name="alpha_proxy(final)",
            sgd_value=alpha_final_sgd,
            muon_value=alpha_final_muon,
            passed=test2,
            supportive_direction="muon_lt_sgd",
            note="In this proxy, lower alpha means a flatter rank-decay fit, but it is not a calibrated WeightWatcher regime label.",
        ),
        "T3": _test_row(
            label="T3",
            claim="SGD shows faster lambda_max growth than Muon.",
            metric_name="lambda_max(final) / lambda_max(init)",
            sgd_value=lmax_growth_sgd,
            muon_value=lmax_growth_muon,
            passed=test3,
            supportive_direction="sgd_gt_muon",
            note="Layer-mean growth factor of the top W^T W eigenvalue.",
        ),
        "T4": _test_row(
            label="T4",
            claim="Removing the top eigenvalue changes SGD's alpha proxy more than Muon's.",
            metric_name="|alpha_proxy - alpha_proxy_clipped| at final step",
            sgd_value=clip_effect_sgd,
            muon_value=clip_effect_muon,
            passed=test4,
            supportive_direction="sgd_gt_muon",
            note="Clip-xmax-inspired diagnostic only; not a full WeightWatcher clipping analysis.",
        ),
    }

    tests_passed = int(sum(test["passed"] for test in tests.values()))
    tests_total = len(tests)
    overall = "PASS" if tests_passed >= 3 else "PARTIAL PASS" if tests_passed >= 2 else "FAIL"

    if tests["T1"]["passed"] and tests["T2"]["passed"] and tests["T3"]["passed"] and not tests["T4"]["passed"]:
        calibrated_conclusion = (
            "In this single-seed per-layer spectrum proxy run, the evidence supports the T1/T2/T3 story: "
            "Muon shows less alpha-proxy drift than SGD, finishes with a lower/flatter alpha proxy, and "
            "exhibits slightly slower lambda_max growth. However, the stronger T4 clipping-effect story is "
            "not supported here: removing the top eigenvalue changes Muon's final alpha proxy more than SGD's."
        )
    else:
        supported = [label for label, test in tests.items() if test["passed"]]
        unsupported = [label for label, test in tests.items() if not test["passed"]]
        supported_text = ", ".join(supported) if supported else "none"
        unsupported_text = ", ".join(unsupported) if unsupported else "none"
        calibrated_conclusion = (
            "This single-seed per-layer spectrum proxy run supports checks "
            f"{supported_text} and does not support checks {unsupported_text}. "
            "Interpret the result as a toy deterministic comparison rather than as a full WeightWatcher or "
            "generalization claim."
        )

    return {
        "tests": tests,
        "tests_in_order": [tests[key] for key in ["T1", "T2", "T3", "T4"]],
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "overall": overall,
        "legacy_pass_rule": "PASS if at least 3 of 4 deterministic checks pass.",
        "alpha_mean_sgd": alpha_mean_sgd,
        "alpha_mean_muon": alpha_mean_muon,
        "alpha_clipped_mean_sgd": alpha_clipped_mean_sgd,
        "alpha_clipped_mean_muon": alpha_clipped_mean_muon,
        "outlier_mean_sgd": outlier_mean_sgd,
        "outlier_mean_muon": outlier_mean_muon,
        "calibrated_conclusion": calibrated_conclusion,
    }


def run_experiment(config: Optional[Dict[str, Any]] = None, seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """Compute the full experiment and return structured results for scripts or notebooks."""
    normalized = normalize_config(config)
    overall_start = time.perf_counter()

    w_target, x_data = make_fixed_problem(normalized, seed=seed)
    initial_weights = init_weights(
        int(normalized["num_layers"]),
        int(normalized["dim"]),
        seed=seed,
        init_noise_scale=float(normalized["init_noise_scale"]),
    )
    initial_loss = compute_loss(initial_weights, x_data, w_target)

    chosen_lr_sgd, lr_scan = find_stable_lr_sgd(normalized, x_data, w_target, seed=seed)
    results_sgd = run_and_measure("SGD", "sgd", chosen_lr_sgd, normalized, x_data, w_target, seed=seed)
    results_muon = run_and_measure(
        "Muon",
        "muon",
        float(normalized["lr_muon"]),
        normalized,
        x_data,
        w_target,
        seed=seed,
    )

    checks = evaluate_checks(results_sgd, results_muon)
    total_runtime = time.perf_counter() - overall_start

    problem_stats = {
        "target_fro_norm": float(np.linalg.norm(w_target, ord="fro")),
        "target_spectral_norm": float(np.linalg.norm(w_target, ord=2)),
        "input_fro_norm": float(np.linalg.norm(x_data, ord="fro")),
        "input_column_norm_mean": float(np.mean(np.linalg.norm(x_data, axis=0))),
        "input_column_norm_std": float(np.std(np.linalg.norm(x_data, axis=0))),
    }

    experiment = {
        "metadata": {
            "experiment_id": normalized["experiment_id"],
            "title": "1.3c-i: Isolate lambda_max from alpha",
            "study_scope": "single-seed per-layer W^T W spectrum proxy study",
            "alpha_scope_note": (
                "alpha is a rank-decay proxy from log(eigenvalue)-vs-log(rank) regression, "
                "not a full WeightWatcher fit"
            ),
            "script_path": str(SCRIPT_DIR / "run_experiment.py"),
            "counterpart_notebook": str(SCRIPT_DIR / "run_experiment.ipynb"),
            "runtime_seconds": float(total_runtime),
        },
        "seed": int(seed),
        "config": normalized,
        "problem_stats": problem_stats,
        "learning_rate_search": {
            "method": "candidate-grid non-divergence heuristic on SGD only",
            "candidates": [float(val) for val in normalized["sgd_lr_candidates"]],
            "search_steps": int(normalized["sgd_lr_search_steps"]),
            "divergence_factor": float(normalized["sgd_divergence_factor"]),
            "chosen_sgd_lr": float(chosen_lr_sgd),
            "scan": lr_scan,
        },
        "initial_loss": float(initial_loss),
        "optimizers": {
            "SGD": results_sgd,
            "Muon": results_muon,
        },
        "checks": checks,
        "summary": {
            "overall": checks["overall"],
            "tests_passed": checks["tests_passed"],
            "tests_total": checks["tests_total"],
            "calibrated_conclusion": checks["calibrated_conclusion"],
        },
    }
    return experiment


def create_summary_figure(
    experiment: Dict[str, Any],
    save_path: Optional[Path] = None,
    show: bool = False,
    close: bool = False,
    use_agg: bool = False,
):
    """Create the legacy-style multi-panel summary figure from structured results."""
    if use_agg:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = experiment["config"]
    results_sgd = experiment["optimizers"]["SGD"]
    results_muon = experiment["optimizers"]["Muon"]
    checks = experiment["checks"]

    steps_sgd = results_sgd["measure_steps"]
    steps_muon = results_muon["measure_steps"]

    alpha_mean_sgd = checks["alpha_mean_sgd"]
    alpha_mean_muon = checks["alpha_mean_muon"]
    alpha_clipped_mean_sgd = checks["alpha_clipped_mean_sgd"]
    alpha_clipped_mean_muon = checks["alpha_clipped_mean_muon"]
    outlier_mean_sgd = checks["outlier_mean_sgd"]
    outlier_mean_muon = checks["outlier_mean_muon"]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(
        "1.3c-i: Isolate lambda_max from alpha\n"
        "Single-seed per-layer W^T W spectrum proxy study"
        f"\n{config['num_layers']}-layer linear net, dim={config['dim']}, {config['num_steps']} steps",
        fontsize=14,
        fontweight="bold",
    )

    ax = axes[0, 0]
    ax.set_title("(a) alpha proxy vs step (mean across layers)")
    ax.plot(steps_sgd, alpha_mean_sgd, "b-o", linewidth=2.5, markersize=3, label="SGD")
    ax.plot(steps_muon, alpha_mean_muon, "r--s", linewidth=2.5, markersize=3, label="Muon")
    ax.set_xlabel("Step")
    ax.set_ylabel("alpha proxy")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.set_title("(b) clipped alpha proxy vs step")
    ax.plot(steps_sgd, alpha_clipped_mean_sgd, "b-o", linewidth=2.5, markersize=3, label="SGD clipped")
    ax.plot(steps_muon, alpha_clipped_mean_muon, "r--s", linewidth=2.5, markersize=3, label="Muon clipped")
    ax.plot(steps_sgd, alpha_mean_sgd, "b:", linewidth=1.0, alpha=0.5, label="SGD full")
    ax.plot(steps_muon, alpha_mean_muon, "r:", linewidth=1.0, alpha=0.5, label="Muon full")
    ax.set_xlabel("Step")
    ax.set_ylabel("alpha proxy")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.set_title("(c) outlier ratio vs step")
    ax.plot(steps_sgd, outlier_mean_sgd, "b-o", linewidth=2.5, markersize=3, label="SGD")
    ax.plot(steps_muon, outlier_mean_muon, "r--s", linewidth=2.5, markersize=3, label="Muon")
    ax.set_xlabel("Step")
    ax.set_ylabel("lambda_max / lambda_median")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.set_title("(d) lambda_max vs step (mean across layers)")
    ax.plot(steps_sgd, results_sgd["trajectory_summary"]["lambda_max_mean"], "b-o", linewidth=2.5, markersize=3, label="SGD")
    ax.plot(steps_muon, results_muon["trajectory_summary"]["lambda_max_mean"], "r--s", linewidth=2.5, markersize=3, label="Muon")
    ax.set_xlabel("Step")
    ax.set_ylabel("lambda_max")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.set_title("(e) clipping effect vs step")
    ax.plot(steps_sgd, results_sgd["trajectory_summary"]["clip_effect_mean"], "b-o", linewidth=2.5, markersize=3, label="SGD")
    ax.plot(steps_muon, results_muon["trajectory_summary"]["clip_effect_mean"], "r--s", linewidth=2.5, markersize=3, label="Muon")
    ax.set_xlabel("Step")
    ax.set_ylabel("|alpha proxy - alpha proxy clipped|")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.set_title("(f) loss vs step")
    ax.semilogy(steps_sgd, results_sgd["losses"], "b-o", linewidth=2.5, markersize=3, label="SGD")
    ax.semilogy(steps_muon, results_muon["losses"], "r--s", linewidth=2.5, markersize=3, label="Muon")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    if close:
        plt.close(fig)

    return fig, axes


def make_json_safe(obj: Any) -> Any:
    """Recursively convert numpy-heavy experiment results into JSON-safe objects."""
    if isinstance(obj, dict):
        return {str(key): make_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_results_bundle(experiment: Dict[str, Any], output_dir: Path = SCRIPT_DIR) -> Dict[str, str]:
    """Save the plot, a compact JSON summary, and a raw NPZ bundle."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_path = output_dir / "alpha_vs_lambda_max.png"
    summary_path = output_dir / "alpha_vs_lambda_max_summary.json"
    raw_path = output_dir / "alpha_vs_lambda_max_raw.npz"

    create_summary_figure(experiment, save_path=plot_path, close=True, use_agg=True)

    sgd = experiment["optimizers"]["SGD"]
    muon = experiment["optimizers"]["Muon"]
    config = experiment["config"]

    np.savez(
        raw_path,
        seed=np.array(experiment["seed"]),
        dim=np.array(config["dim"]),
        num_layers=np.array(config["num_layers"]),
        num_steps=np.array(config["num_steps"]),
        batch_size=np.array(config["batch_size"]),
        lr_muon=np.array(config["lr_muon"]),
        lr_sgd=np.array(experiment["learning_rate_search"]["chosen_sgd_lr"]),
        sgd_measure_steps=sgd["measure_steps"],
        sgd_alpha=sgd["alpha"],
        sgd_alpha_clipped=sgd["alpha_clipped"],
        sgd_lambda_max=sgd["lambda_max"],
        sgd_lambda_median=sgd["lambda_median"],
        sgd_lambda_min=sgd["lambda_min"],
        sgd_outlier_ratio=sgd["outlier_ratio"],
        sgd_losses=sgd["losses"],
        sgd_initial_eigenvalues=sgd["initial_eigenvalues"],
        sgd_final_eigenvalues=sgd["final_eigenvalues"],
        muon_measure_steps=muon["measure_steps"],
        muon_alpha=muon["alpha"],
        muon_alpha_clipped=muon["alpha_clipped"],
        muon_lambda_max=muon["lambda_max"],
        muon_lambda_median=muon["lambda_median"],
        muon_lambda_min=muon["lambda_min"],
        muon_outlier_ratio=muon["outlier_ratio"],
        muon_losses=muon["losses"],
        muon_initial_eigenvalues=muon["initial_eigenvalues"],
        muon_final_eigenvalues=muon["final_eigenvalues"],
    )

    summary_payload = {
        "metadata": experiment["metadata"],
        "seed": experiment["seed"],
        "config": experiment["config"],
        "problem_stats": experiment["problem_stats"],
        "learning_rate_search": experiment["learning_rate_search"],
        "initial_loss": experiment["initial_loss"],
        "checks": experiment["checks"],
        "summary": experiment["summary"],
        "artifacts": {
            "plot_path": str(plot_path),
            "summary_path": str(summary_path),
            "raw_path": str(raw_path),
        },
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(make_json_safe(summary_payload), handle, indent=2)

    artifact_paths = {
        "plot_path": str(plot_path),
        "summary_path": str(summary_path),
        "raw_path": str(raw_path),
    }
    experiment["artifacts"] = artifact_paths
    return artifact_paths


def print_report(experiment: Dict[str, Any]) -> None:
    """Print a calibrated text report from structured experiment results."""
    config = experiment["config"]
    results_sgd = experiment["optimizers"]["SGD"]
    results_muon = experiment["optimizers"]["Muon"]
    checks = experiment["checks"]

    steps_sgd = results_sgd["measure_steps"]
    steps_muon = results_muon["measure_steps"]
    table_steps = config["table_steps"]

    print("=" * 100)
    print("1.3c-i: ISOLATE lambda_max FROM alpha")
    print("=" * 100)
    print("Scope: single-seed per-layer W^T W spectrum proxy study")
    print("Alpha note: the reported alpha is a rank-decay proxy, not a full WeightWatcher fit.")
    print(
        f"Setup: {config['num_layers']}-layer deep linear net (dim={config['dim']}), "
        f"batch={config['batch_size']}, {config['num_steps']} steps"
    )
    print(
        f"Measure every {config['measure_every']} steps | Momentum={config['momentum']} | "
        f"Muon LR={config['lr_muon']}"
    )
    print("Learning-rate note: SGD LR is chosen by a short candidate-grid stability heuristic.")
    print("=" * 100)

    print(f"\nSeed: {experiment['seed']}")
    print(f"Initial loss: {experiment['initial_loss']:.6e}")
    print(f"Chosen SGD LR: {experiment['learning_rate_search']['chosen_sgd_lr']}")
    print(f"Final SGD loss:  {results_sgd['final_loss']:.6e}")
    print(f"Final Muon loss: {results_muon['final_loss']:.6e}")
    print(f"Total runtime:   {experiment['metadata']['runtime_seconds']:.2f} s")

    print(f"\n{'=' * 100}")
    print("TABLE 1: alpha proxy vs step (mean across layers)")
    print("  Lower alpha proxy = flatter rank-decay fit in this toy metric.")
    print("=" * 100)
    print(
        f"\n  {'Step':>6} | {'SGD alpha':>10} | {'Muon alpha':>11} | {'SGD-Muon':>10} | "
        f"{'SGD alpha_c':>12} | {'Muon alpha_c':>13} | {'SGD-Muon clip':>14}"
    )
    print("  " + "-" * 95)
    for table_step in table_steps:
        idx_s = step_idx(steps_sgd, table_step)
        idx_m = step_idx(steps_muon, table_step)
        if idx_s is None or idx_m is None:
            continue
        alpha_sgd = float(np.nanmean(results_sgd["alpha"][idx_s]))
        alpha_muon = float(np.nanmean(results_muon["alpha"][idx_m]))
        alpha_clip_sgd = float(np.nanmean(results_sgd["alpha_clipped"][idx_s]))
        alpha_clip_muon = float(np.nanmean(results_muon["alpha_clipped"][idx_m]))
        print(
            f"  {table_step:6d} | {alpha_sgd:10.4f} | {alpha_muon:11.4f} | {alpha_sgd - alpha_muon:+10.4f} | "
            f"{alpha_clip_sgd:12.4f} | {alpha_clip_muon:13.4f} | {alpha_clip_sgd - alpha_clip_muon:+14.4f}"
        )

    print(f"\n\n{'=' * 100}")
    print("TABLE 2: outlier ratio lambda_max / lambda_median (mean across layers)")
    print("=" * 100)
    print(f"\n  {'Step':>6} | {'SGD ratio':>10} | {'Muon ratio':>11} | {'SGD/Muon':>10}")
    print("  " + "-" * 52)
    for table_step in table_steps:
        idx_s = step_idx(steps_sgd, table_step)
        idx_m = step_idx(steps_muon, table_step)
        if idx_s is None or idx_m is None:
            continue
        ratio_sgd = float(np.nanmean(results_sgd["outlier_ratio"][idx_s]))
        ratio_muon = float(np.nanmean(results_muon["outlier_ratio"][idx_m]))
        ratio_string = f"{ratio_sgd / ratio_muon:.4f}" if ratio_muon > 1e-10 else "N/A"
        print(f"  {table_step:6d} | {ratio_sgd:10.4f} | {ratio_muon:11.4f} | {ratio_string:>10}")

    print(f"\n\n{'=' * 100}")
    print("TABLE 3: per-layer alpha proxy at key steps")
    print("=" * 100)
    print(f"\n  {'Step':>6} | ", end="")
    for layer in range(int(config["num_layers"])):
        print(f"{'SGD L' + str(layer):>8} {'Muon L' + str(layer):>8} | ", end="")
    print()
    print("  " + "-" * (8 + (18 + 3) * int(config["num_layers"])))
    for table_step in table_steps:
        idx_s = step_idx(steps_sgd, table_step)
        idx_m = step_idx(steps_muon, table_step)
        if idx_s is None or idx_m is None:
            continue
        print(f"  {table_step:6d} | ", end="")
        for layer in range(int(config["num_layers"])):
            alpha_sgd = float(results_sgd["alpha"][idx_s, layer])
            alpha_muon = float(results_muon["alpha"][idx_m, layer])
            print(f"{alpha_sgd:8.4f} {alpha_muon:8.4f} | ", end="")
        print()

    print(f"\n\n{'=' * 100}")
    print("TABLE 4: eigenvalue summary (layer means) at key steps")
    print("=" * 100)
    print(
        f"\n  {'Step':>6} | {'SGD lmax':>10} {'SGD lmed':>10} {'SGD lmin':>10} | "
        f"{'Muon lmax':>10} {'Muon lmed':>10} {'Muon lmin':>10}"
    )
    print("  " + "-" * 85)
    for table_step in table_steps:
        idx_s = step_idx(steps_sgd, table_step)
        idx_m = step_idx(steps_muon, table_step)
        if idx_s is None or idx_m is None:
            continue
        lmax_sgd = float(np.nanmean(results_sgd["lambda_max"][idx_s]))
        lmed_sgd = float(np.nanmean(results_sgd["lambda_median"][idx_s]))
        lmin_sgd = float(np.nanmean(results_sgd["lambda_min"][idx_s]))
        lmax_muon = float(np.nanmean(results_muon["lambda_max"][idx_m]))
        lmed_muon = float(np.nanmean(results_muon["lambda_median"][idx_m]))
        lmin_muon = float(np.nanmean(results_muon["lambda_min"][idx_m]))
        print(
            f"  {table_step:6d} | {lmax_sgd:10.4f} {lmed_sgd:10.4f} {lmin_sgd:10.4f} | "
            f"{lmax_muon:10.4f} {lmed_muon:10.4f} {lmin_muon:10.4f}"
        )

    print(f"\n\n{'=' * 100}")
    print("TABLE 5: clipping effect |alpha proxy - alpha proxy clipped| (mean across layers)")
    print("=" * 100)
    print(f"\n  {'Step':>6} | {'SGD |da|':>10} | {'Muon |da|':>11} | {'SGD-Muon':>10}")
    print("  " + "-" * 52)
    for table_step in table_steps:
        idx_s = step_idx(steps_sgd, table_step)
        idx_m = step_idx(steps_muon, table_step)
        if idx_s is None or idx_m is None:
            continue
        delta_sgd = float(np.nanmean(np.abs(results_sgd["alpha"][idx_s] - results_sgd["alpha_clipped"][idx_s])))
        delta_muon = float(np.nanmean(np.abs(results_muon["alpha"][idx_m] - results_muon["alpha_clipped"][idx_m])))
        print(f"  {table_step:6d} | {delta_sgd:10.4f} | {delta_muon:11.4f} | {delta_sgd - delta_muon:+10.4f}")

    print(f"\n\n{'=' * 100}")
    print("DETERMINISTIC CHECKS (single seed)")
    print("=" * 100)
    for test in checks["tests_in_order"]:
        print(f"\n  {test['label']}: {test['claim']}")
        print(f"      Metric: {test['metric_name']}")
        print(f"      SGD value:  {test['sgd_value']:.4f}")
        print(f"      Muon value: {test['muon_value']:.4f}")
        print(f"      Result: {test['outcome_text']}")
        print(f"      Note: {test['note']}")
        if test['label'] == 'T1':
            print(f"      Signed alpha-proxy change: SGD {test['sgd_signed_change']:+.4f}, Muon {test['muon_signed_change']:+.4f}")

    print(f"\n{'=' * 100}")
    print("FINAL VERDICT")
    print("=" * 100)
    print(f"Legacy rule: {checks['legacy_pass_rule']}")
    print(f"Tests passed: {checks['tests_passed']}/{checks['tests_total']}")
    print(f"Overall: {checks['overall']}")
    print(f"Conclusion: {checks['calibrated_conclusion']}")
    if "artifacts" in experiment:
        print("Artifacts:")
        for key, path in experiment["artifacts"].items():
            print(f"  {key}: {path}")
    print("=" * 100)


def main() -> None:
    """Entry point preserving normal script behavior while keeping import safety."""
    experiment = run_experiment()
    artifact_paths = save_results_bundle(experiment, output_dir=SCRIPT_DIR)
    experiment["artifacts"] = artifact_paths
    print_report(experiment)


if __name__ == "__main__":
    main()
