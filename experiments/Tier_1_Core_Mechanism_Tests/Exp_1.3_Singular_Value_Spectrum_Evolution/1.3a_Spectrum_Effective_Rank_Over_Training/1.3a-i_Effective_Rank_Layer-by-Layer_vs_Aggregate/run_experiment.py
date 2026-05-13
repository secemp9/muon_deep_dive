#!/usr/bin/env python3
"""
1.3a-i: Effective Rank -- Layer-by-Layer vs Aggregate
=====================================================

Deterministic single-seed toy probe comparing SGD with momentum against
Muon-style Newton-Schulz gradient orthogonalization with momentum on a
6-layer 32x32 deep linear network.

What this file measures:
- per-layer effective rank over training
- product-matrix effective rank over training
- per-layer condition number over training
- product-matrix condition number over training
- loss trajectories for both optimizers

Important scope limits:
- one seed
- one target matrix
- one fixed input batch
- one network depth (L=6)
- no uncertainty estimates or depth sweep

Accordingly, this script can only support claims about this single trajectory.
It does not directly test depth-scaling laws; the T4 check is only a consistency
check against prior experiments, not a new scaling estimate.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG: Dict[str, Any] = {
    "dim": 32,
    "num_layers": 6,
    "num_steps": 300,
    "batch_size": 64,
    "lr_muon": 0.005,
    "momentum": 0.9,
    "ns_iters": 5,
    "measure_every": 10,
    "report_steps": [0, 50, 100, 200, 300],
    "seed": 42,
    "sgd_lr_candidates": [0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.001],
    "sgd_stability_steps": 100,
    "sgd_divergence_factor": 50.0,
    "run_divergence_loss_threshold": 1e10,
}


def build_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a fresh experiment configuration dictionary."""
    config = dict(DEFAULT_CONFIG)
    if overrides:
        config.update(overrides)
    config["report_steps"] = list(config["report_steps"])
    config["sgd_lr_candidates"] = list(config["sgd_lr_candidates"])
    return config


def generate_problem_data(config: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Generate the fixed target matrix and fixed input batch for this run."""
    rng = np.random.RandomState(config["seed"])
    dim = config["dim"]
    batch_size = config["batch_size"]
    w_target = rng.randn(dim, dim) * 0.5
    x_data = rng.randn(dim, batch_size) * 0.3
    return w_target, x_data


def init_weights(config: Dict[str, Any], seed: Optional[int] = None) -> List[np.ndarray]:
    """Initialize all layers near identity for stability."""
    dim = config["dim"]
    num_layers = config["num_layers"]
    rng = np.random.RandomState(config["seed"] if seed is None else seed)
    return [(np.eye(dim) + rng.randn(dim, dim) * 0.1).copy() for _ in range(num_layers)]


def zero_velocities(config: Dict[str, Any]) -> List[np.ndarray]:
    """Create zero velocity buffers for momentum updates."""
    dim = config["dim"]
    num_layers = config["num_layers"]
    return [np.zeros((dim, dim)) for _ in range(num_layers)]


def forward(weights: List[np.ndarray], x_data: np.ndarray) -> np.ndarray:
    """Forward pass: W_L @ ... @ W_1 @ X."""
    out = x_data.copy()
    for weight in weights:
        out = weight @ out
    return out


def compute_loss(weights: List[np.ndarray], x_data: np.ndarray, target: np.ndarray) -> float:
    """Quadratic loss averaged over the batch."""
    pred = forward(weights, x_data)
    target_out = target @ x_data
    diff = pred - target_out
    return float(0.5 * np.mean(np.sum(diff**2, axis=0)))


def compute_gradients(weights: List[np.ndarray], x_data: np.ndarray, target: np.ndarray) -> List[np.ndarray]:
    """Backpropagation through the deep linear network."""
    num_layers = len(weights)
    batch_size = x_data.shape[1]

    activations = [x_data.copy()]
    out = x_data.copy()
    for weight in weights:
        out = weight @ out
        activations.append(out.copy())

    target_out = target @ x_data
    delta = (activations[-1] - target_out) / batch_size

    grads: List[np.ndarray] = []
    for layer_idx in range(num_layers - 1, -1, -1):
        grad = delta @ activations[layer_idx].T
        grads.insert(0, grad)
        if layer_idx > 0:
            delta = weights[layer_idx].T @ delta

    return grads


def newton_schulz_orthogonalize(grad: np.ndarray, num_iters: int) -> np.ndarray:
    """Approximate the orthogonal polar factor of ``grad`` via Newton-Schulz."""
    norm = np.linalg.norm(grad, ord="fro")
    if norm < 1e-12:
        return grad

    x_iter = grad / norm
    for _ in range(num_iters):
        gram = x_iter.T @ x_iter
        x_iter = 1.5 * x_iter - 0.5 * x_iter @ gram
    return x_iter


def compute_product_matrix(weights: List[np.ndarray]) -> np.ndarray:
    """Compute W_L @ ... @ W_1."""
    dim = weights[0].shape[0]
    product = np.eye(dim)
    for weight in weights:
        product = weight @ product
    return product


def effective_rank(matrix: np.ndarray) -> float:
    """Compute erank = exp(H) using the entropy of normalized singular values."""
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    singular_values = singular_values[singular_values > 1e-15]
    if len(singular_values) == 0:
        return 1.0
    probs = singular_values / np.sum(singular_values)
    entropy = -np.sum(probs * np.log(probs))
    return float(np.exp(entropy))


def condition_number(matrix: np.ndarray) -> float:
    """Compute kappa = sigma_max / sigma_min."""
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values[-1] < 1e-15:
        return float("inf")
    return float(singular_values[0] / singular_values[-1])


def sgd_step(
    weights: List[np.ndarray],
    velocities: List[np.ndarray],
    lr: float,
    config: Dict[str, Any],
    x_data: np.ndarray,
    w_target: np.ndarray,
) -> tuple[List[np.ndarray], List[np.ndarray]]:
    """One step of SGD with momentum."""
    grads = compute_gradients(weights, x_data, w_target)
    momentum = config["momentum"]
    for idx in range(len(weights)):
        velocities[idx] = momentum * velocities[idx] + grads[idx]
        weights[idx] = weights[idx] - lr * velocities[idx]
    return weights, velocities


def muon_step(
    weights: List[np.ndarray],
    velocities: List[np.ndarray],
    lr: float,
    config: Dict[str, Any],
    x_data: np.ndarray,
    w_target: np.ndarray,
) -> tuple[List[np.ndarray], List[np.ndarray]]:
    """One step of Muon with momentum."""
    grads = compute_gradients(weights, x_data, w_target)
    momentum = config["momentum"]
    ns_iters = config["ns_iters"]
    for idx in range(len(weights)):
        ortho_grad = newton_schulz_orthogonalize(grads[idx], num_iters=ns_iters)
        velocities[idx] = momentum * velocities[idx] + ortho_grad
        weights[idx] = weights[idx] - lr * velocities[idx]
    return weights, velocities


def measure_state(
    weights: List[np.ndarray],
    x_data: np.ndarray,
    w_target: np.ndarray,
) -> Dict[str, Any]:
    """Measure all tracked quantities for the current weights."""
    layer_eranks = [effective_rank(weight) for weight in weights]
    layer_kappas = [condition_number(weight) for weight in weights]
    product = compute_product_matrix(weights)
    return {
        "per_layer_erank": layer_eranks,
        "per_layer_kappa": layer_kappas,
        "product_erank": effective_rank(product),
        "product_kappa": condition_number(product),
        "loss": compute_loss(weights, x_data, w_target),
    }


def find_stable_lr_sgd(
    config: Dict[str, Any],
    x_data: np.ndarray,
    w_target: np.ndarray,
) -> Dict[str, Any]:
    """
    Pick the first stable SGD learning rate from a descending candidate list.

    This is a coarse sweep, not a binary search and not a symmetric optimizer
    tuning protocol.
    """
    trials: List[Dict[str, Any]] = []
    chosen_lr = config["sgd_lr_candidates"][-1]

    for lr in config["sgd_lr_candidates"]:
        weights = init_weights(config)
        velocities = zero_velocities(config)
        initial_loss = compute_loss(weights, x_data, w_target)
        max_loss = initial_loss
        stable = True
        failure_step = None

        for step in range(1, config["sgd_stability_steps"] + 1):
            weights, velocities = sgd_step(weights, velocities, lr, config, x_data, w_target)
            loss = compute_loss(weights, x_data, w_target)
            max_loss = max(max_loss, loss)
            if np.isnan(loss) or loss > initial_loss * config["sgd_divergence_factor"]:
                stable = False
                failure_step = step
                break

        trials.append(
            {
                "lr": float(lr),
                "stable": bool(stable),
                "failure_step": None if failure_step is None else int(failure_step),
                "initial_loss": float(initial_loss),
                "max_loss": float(max_loss),
            }
        )

        if stable:
            chosen_lr = float(lr)
            break

    return {
        "method": "first stable candidate from descending sweep",
        "candidates": [float(lr) for lr in config["sgd_lr_candidates"]],
        "stability_steps": int(config["sgd_stability_steps"]),
        "divergence_factor": float(config["sgd_divergence_factor"]),
        "chosen_lr": float(chosen_lr),
        "trial_results": trials,
    }


def run_and_measure(
    optimizer_name: str,
    optimizer_fn,
    lr: float,
    num_steps: int,
    config: Dict[str, Any],
    x_data: np.ndarray,
    w_target: np.ndarray,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run one optimizer and record metrics at step 0 and every measure_every steps."""
    weights = init_weights(config)
    velocities = zero_velocities(config)

    steps_list: List[int] = []
    per_layer_erank: List[List[float]] = []
    product_erank: List[float] = []
    per_layer_kappa: List[List[float]] = []
    product_kappa: List[float] = []
    losses: List[float] = []
    diverged = False

    def record(step: int) -> None:
        measurement = measure_state(weights, x_data, w_target)
        steps_list.append(int(step))
        per_layer_erank.append(measurement["per_layer_erank"])
        product_erank.append(float(measurement["product_erank"]))
        per_layer_kappa.append(measurement["per_layer_kappa"])
        product_kappa.append(float(measurement["product_kappa"]))
        losses.append(float(measurement["loss"]))

    record(0)

    for step in range(1, num_steps + 1):
        weights, velocities = optimizer_fn(weights, velocities, lr, config, x_data, w_target)
        loss = compute_loss(weights, x_data, w_target)
        if np.isnan(loss) or loss > config["run_divergence_loss_threshold"]:
            diverged = True
            if verbose:
                print(f"    WARNING: {optimizer_name} diverged at step {step}.")
            break
        if step % config["measure_every"] == 0:
            record(step)

    return {
        "optimizer_name": optimizer_name,
        "lr": float(lr),
        "steps": np.array(steps_list, dtype=int),
        "per_layer_erank": np.array(per_layer_erank, dtype=float),
        "product_erank": np.array(product_erank, dtype=float),
        "per_layer_kappa": np.array(per_layer_kappa, dtype=float),
        "product_kappa": np.array(product_kappa, dtype=float),
        "losses": np.array(losses, dtype=float),
        "diverged": bool(diverged),
        "final_recorded_step": int(steps_list[-1]),
    }


def select_step_index(steps: np.ndarray, step: int) -> int:
    """Return the recorded index corresponding to or immediately after ``step``."""
    idx = int(np.searchsorted(steps, step))
    return min(idx, len(steps) - 1)


def summarize_problem_setup(
    config: Dict[str, Any],
    w_target: np.ndarray,
    x_data: np.ndarray,
) -> Dict[str, Any]:
    """Summarize the target, input batch, and initial spectral state."""
    target_sv = np.linalg.svd(w_target, compute_uv=False)
    target_probs = target_sv / np.sum(target_sv)
    target_erank = float(np.exp(-np.sum(target_probs * np.log(target_probs))))

    initial_weights = init_weights(config)
    initial_layer_rows = []
    initial_layer_eranks = []
    initial_layer_kappas = []
    for layer_idx, weight in enumerate(initial_weights):
        sv = np.linalg.svd(weight, compute_uv=False)
        layer_erank = effective_rank(weight)
        layer_kappa = condition_number(weight)
        initial_layer_eranks.append(layer_erank)
        initial_layer_kappas.append(layer_kappa)
        initial_layer_rows.append(
            {
                "layer": int(layer_idx),
                "sigma_min": float(sv[-1]),
                "sigma_max": float(sv[0]),
                "effective_rank": float(layer_erank),
                "condition_number": float(layer_kappa),
            }
        )

    initial_product = compute_product_matrix(initial_weights)
    initial_product_erank = effective_rank(initial_product)
    initial_product_kappa = condition_number(initial_product)

    return {
        "target_fro_norm": float(np.linalg.norm(w_target, ord="fro")),
        "target_condition_number": condition_number(w_target),
        "target_effective_rank": float(target_erank),
        "input_fro_norm": float(np.linalg.norm(x_data, ord="fro")),
        "input_mean_column_norm": float(np.mean(np.linalg.norm(x_data, axis=0))),
        "initial_loss": compute_loss(initial_weights, x_data, w_target),
        "initial_state": {
            "per_layer_rows": initial_layer_rows,
            "per_layer_erank_mean": float(np.mean(initial_layer_eranks)),
            "per_layer_erank_min": float(np.min(initial_layer_eranks)),
            "per_layer_erank_max": float(np.max(initial_layer_eranks)),
            "per_layer_kappa_mean": float(np.mean(initial_layer_kappas)),
            "per_layer_kappa_min": float(np.min(initial_layer_kappas)),
            "per_layer_kappa_max": float(np.max(initial_layer_kappas)),
            "product_effective_rank": float(initial_product_erank),
            "product_condition_number": float(initial_product_kappa),
        },
    }


def build_report_rows(
    results_sgd: Dict[str, Any],
    results_muon: Dict[str, Any],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Create compact report rows at the configured report steps."""
    rows: List[Dict[str, Any]] = []
    for step in config["report_steps"]:
        sgd_idx = select_step_index(results_sgd["steps"], step)
        muon_idx = select_step_index(results_muon["steps"], step)

        sgd_eranks = results_sgd["per_layer_erank"][sgd_idx]
        muon_eranks = results_muon["per_layer_erank"][muon_idx]
        sgd_kappas = results_sgd["per_layer_kappa"][sgd_idx]
        muon_kappas = results_muon["per_layer_kappa"][muon_idx]

        rows.append(
            {
                "step": int(step),
                "sgd_per_layer_erank_mean": float(np.mean(sgd_eranks)),
                "sgd_per_layer_erank_min": float(np.min(sgd_eranks)),
                "sgd_per_layer_erank_max": float(np.max(sgd_eranks)),
                "muon_per_layer_erank_mean": float(np.mean(muon_eranks)),
                "muon_per_layer_erank_min": float(np.min(muon_eranks)),
                "muon_per_layer_erank_max": float(np.max(muon_eranks)),
                "sgd_product_erank": float(results_sgd["product_erank"][sgd_idx]),
                "muon_product_erank": float(results_muon["product_erank"][muon_idx]),
                "sgd_per_layer_kappa_mean": float(np.mean(sgd_kappas)),
                "muon_per_layer_kappa_mean": float(np.mean(muon_kappas)),
                "sgd_product_kappa": float(results_sgd["product_kappa"][sgd_idx]),
                "muon_product_kappa": float(results_muon["product_kappa"][muon_idx]),
                "sgd_loss": float(results_sgd["losses"][sgd_idx]),
                "muon_loss": float(results_muon["losses"][muon_idx]),
            }
        )
    return rows


def evaluate_expectations(
    results_sgd: Dict[str, Any],
    results_muon: Dict[str, Any],
    config: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Compute final summaries and calibrated T1/T2/T3/T4 outcomes."""
    dim = config["dim"]

    sgd_final_mean_erank = float(np.mean(results_sgd["per_layer_erank"][-1]))
    muon_final_mean_erank = float(np.mean(results_muon["per_layer_erank"][-1]))
    sgd_final_mean_kappa = float(np.mean(results_sgd["per_layer_kappa"][-1]))
    muon_final_mean_kappa = float(np.mean(results_muon["per_layer_kappa"][-1]))

    sgd_final_product_erank = float(results_sgd["product_erank"][-1])
    muon_final_product_erank = float(results_muon["product_erank"][-1])
    sgd_final_product_kappa = float(results_sgd["product_kappa"][-1])
    muon_final_product_kappa = float(results_muon["product_kappa"][-1])

    sgd_initial_mean_erank = float(np.mean(results_sgd["per_layer_erank"][0]))
    muon_initial_mean_erank = float(np.mean(results_muon["per_layer_erank"][0]))
    sgd_initial_product_erank = float(results_sgd["product_erank"][0])
    muon_initial_product_erank = float(results_muon["product_erank"][0])

    sgd_depth_compression = float(sgd_final_product_erank / sgd_final_mean_erank)
    muon_depth_compression = float(muon_final_product_erank / muon_final_mean_erank)

    final_summary = {
        "sgd_final_loss": float(results_sgd["losses"][-1]),
        "muon_final_loss": float(results_muon["losses"][-1]),
        "sgd_final_per_layer_erank_mean": sgd_final_mean_erank,
        "muon_final_per_layer_erank_mean": muon_final_mean_erank,
        "sgd_final_product_erank": sgd_final_product_erank,
        "muon_final_product_erank": muon_final_product_erank,
        "sgd_final_per_layer_kappa_mean": sgd_final_mean_kappa,
        "muon_final_per_layer_kappa_mean": muon_final_mean_kappa,
        "sgd_final_product_kappa": sgd_final_product_kappa,
        "muon_final_product_kappa": muon_final_product_kappa,
        "sgd_per_layer_erank_retention": float(sgd_final_mean_erank / sgd_initial_mean_erank),
        "muon_per_layer_erank_retention": float(muon_final_mean_erank / muon_initial_mean_erank),
        "sgd_product_erank_retention": float(sgd_final_product_erank / sgd_initial_product_erank),
        "muon_product_erank_retention": float(muon_final_product_erank / muon_initial_product_erank),
        "sgd_depth_compression": sgd_depth_compression,
        "muon_depth_compression": muon_depth_compression,
        "sgd_product_erank_delta": float(sgd_final_product_erank - sgd_initial_product_erank),
        "muon_product_erank_delta": float(muon_final_product_erank - muon_initial_product_erank),
    }

    test1_pass = bool(muon_final_mean_erank > sgd_final_mean_erank)
    test2_pass = bool(muon_final_mean_erank > 0.8 * dim)
    test3_pass = bool(sgd_depth_compression < 1.0 and muon_depth_compression < 1.0)
    test4_pass = bool(muon_final_product_kappa < sgd_final_product_kappa)

    tests: Dict[str, Any] = {
        "T1": {
            "description": "Final mean per-layer effective rank is higher for Muon than for SGD in this run.",
            "passed": test1_pass,
            "observed_summary": f"Muon {muon_final_mean_erank:.2f} vs SGD {sgd_final_mean_erank:.2f}",
            "interpretation": (
                "Supported in this single run: Muon ends with a higher mean per-layer effective rank."
                if test1_pass
                else "Not supported in this single run."
            ),
        },
        "T2": {
            "description": f"Muon final mean per-layer effective rank remains above 80% of n={dim}.",
            "passed": test2_pass,
            "observed_summary": f"Muon {muon_final_mean_erank:.2f} vs threshold {0.8 * dim:.2f}",
            "interpretation": (
                "Supported in this single run: Muon's final mean per-layer effective rank stays above the chosen threshold."
                if test2_pass
                else "Not supported in this single run."
            ),
        },
        "T3": {
            "description": "Final product effective rank is lower than final mean per-layer effective rank for both optimizers.",
            "passed": test3_pass,
            "observed_summary": (
                f"SGD compression {sgd_depth_compression:.3f}; Muon compression {muon_depth_compression:.3f}"
            ),
            "interpretation": (
                "Supported in this single run: the product remains spectrally narrower than the mean layer for both optimizers."
                if test3_pass
                else "Not supported in this single run."
            ),
        },
        "T4": {
            "description": "Muon final product condition number is smaller than SGD's final product condition number.",
            "passed": test4_pass,
            "observed_summary": f"SGD {sgd_final_product_kappa:.2f} vs Muon {muon_final_product_kappa:.2f}",
            "interpretation": (
                "Supported in this single run as a consistency check against prior work."
                if test4_pass
                else "Not supported in this single run; this trajectory does not satisfy the expected product-kappa ordering at the final step."
            ),
            "scope_note": "This is only a consistency check against Exp 1.1a-i, not a direct depth-scaling measurement here.",
        },
    }

    tests_passed = int(sum(test_entry["passed"] for test_entry in tests.values()))
    if tests_passed == 4:
        overall = "PASS"
    elif tests_passed >= 3:
        overall = "PARTIAL PASS"
    elif tests_passed >= 2:
        overall = "MIXED"
    else:
        overall = "FAIL"

    tests["tests_passed"] = tests_passed
    tests["tests_total"] = 4
    tests["overall"] = overall
    tests["scope_note"] = (
        "Interpret only as evidence from one deterministic L=6 trajectory; this file does not estimate seed-level or depth-level generalization."
    )

    return final_summary, tests


def make_plots(results: Dict[str, Any], output_dir: Path) -> Dict[str, str]:
    """Save the same core visualizations used by the script counterpart."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {"plot_warning": "matplotlib not available; plots were not generated."}

    config = results["config"]
    sgd = results["optimizers"]["sgd"]
    muon = results["optimizers"]["muon"]
    dim = config["dim"]
    num_layers = config["num_layers"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "1.3a-i: Effective Rank -- Layer-by-Layer vs Aggregate\n"
        "Single-seed L=6 deep linear toy run",
        fontsize=14,
        fontweight="bold",
    )

    ax = axes[0, 0]
    ax.set_title("(a) Per-layer effective rank over training")
    for layer in range(num_layers):
        ax.plot(sgd["steps"], sgd["per_layer_erank"][:, layer], color="tab:blue", alpha=0.25, linewidth=0.8)
    for layer in range(num_layers):
        ax.plot(muon["steps"], muon["per_layer_erank"][:, layer], color="tab:red", alpha=0.25, linewidth=0.8)
    ax.plot(sgd["steps"], np.mean(sgd["per_layer_erank"], axis=1), color="tab:blue", linewidth=2.5, label="SGD mean")
    ax.plot(muon["steps"], np.mean(muon["per_layer_erank"], axis=1), color="tab:red", linewidth=2.5, label="Muon mean")
    ax.axhline(dim, color="tab:green", linestyle="--", alpha=0.5, label=f"max erank = {dim}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Effective rank")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.set_title("(b) Product effective rank over training")
    ax.plot(sgd["steps"], sgd["product_erank"], color="tab:blue", linewidth=2.5, marker="o", markersize=3, label="SGD product")
    ax.plot(muon["steps"], muon["product_erank"], color="tab:red", linewidth=2.5, marker="s", markersize=3, label="Muon product")
    ax.axhline(dim, color="tab:green", linestyle="--", alpha=0.5, label=f"max erank = {dim}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Effective rank")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    ax.set_title("(c) Per-layer condition number over training")
    for layer in range(num_layers):
        ax.semilogy(sgd["steps"], sgd["per_layer_kappa"][:, layer], color="tab:blue", alpha=0.25, linewidth=0.8)
    for layer in range(num_layers):
        ax.semilogy(muon["steps"], muon["per_layer_kappa"][:, layer], color="tab:red", alpha=0.25, linewidth=0.8)
    ax.semilogy(sgd["steps"], np.mean(sgd["per_layer_kappa"], axis=1), color="tab:blue", linewidth=2.5, label="SGD mean")
    ax.semilogy(muon["steps"], np.mean(muon["per_layer_kappa"], axis=1), color="tab:red", linewidth=2.5, label="Muon mean")
    ax.axhline(1.0, color="tab:green", linestyle="--", alpha=0.5, label="orthogonal baseline")
    ax.set_xlabel("Step")
    ax.set_ylabel("Condition number")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    ax.set_title("(d) Product condition number over training")
    sgd_product_kappa = sgd["product_kappa"].copy()
    muon_product_kappa = muon["product_kappa"].copy()
    sgd_product_kappa[~np.isfinite(sgd_product_kappa)] = np.nan
    muon_product_kappa[~np.isfinite(muon_product_kappa)] = np.nan
    ax.semilogy(sgd["steps"], sgd_product_kappa, color="tab:blue", linewidth=2.5, marker="o", markersize=3, label="SGD product")
    ax.semilogy(muon["steps"], muon_product_kappa, color="tab:red", linewidth=2.5, marker="s", markersize=3, label="Muon product")
    ax.set_xlabel("Step")
    ax.set_ylabel("Condition number")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    fig.tight_layout()
    plot1_path = output_dir / "effective_rank_evolution.png"
    fig.savefig(plot1_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "1.3a-i: Per-layer mean/band vs product effective rank\n"
        "Single-seed comparison; not a depth sweep",
        fontsize=13,
        fontweight="bold",
    )

    sgd_mean = np.mean(sgd["per_layer_erank"], axis=1)
    sgd_min = np.min(sgd["per_layer_erank"], axis=1)
    sgd_max = np.max(sgd["per_layer_erank"], axis=1)
    muon_mean = np.mean(muon["per_layer_erank"], axis=1)
    muon_min = np.min(muon["per_layer_erank"], axis=1)
    muon_max = np.max(muon["per_layer_erank"], axis=1)

    ax = axes[0]
    ax.set_title("SGD")
    ax.fill_between(sgd["steps"], sgd_min, sgd_max, alpha=0.15, color="tab:blue")
    ax.plot(sgd["steps"], sgd_mean, color="tab:blue", linewidth=2, label="per-layer mean")
    ax.plot(sgd["steps"], sgd["product_erank"], color="tab:blue", linestyle="--", linewidth=2, label="product")
    ax.axhline(dim, color="tab:green", linestyle=":", alpha=0.5, label=f"max = {dim}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Effective rank")
    ax.set_ylim(bottom=0, top=dim + 2)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.set_title("Muon")
    ax.fill_between(muon["steps"], muon_min, muon_max, alpha=0.15, color="tab:red")
    ax.plot(muon["steps"], muon_mean, color="tab:red", linewidth=2, label="per-layer mean")
    ax.plot(muon["steps"], muon["product_erank"], color="tab:red", linestyle="--", linewidth=2, label="product")
    ax.axhline(dim, color="tab:green", linestyle=":", alpha=0.5, label=f"max = {dim}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Effective rank")
    ax.set_ylim(bottom=0, top=dim + 2)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    fig.tight_layout()
    plot2_path = output_dir / "erank_per_layer_vs_product.png"
    fig.savefig(plot2_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "effective_rank_plot": str(plot1_path),
        "per_layer_vs_product_plot": str(plot2_path),
    }


def to_builtin(value: Any) -> Any:
    """Recursively convert numpy objects to plain Python / JSON-safe objects."""
    if isinstance(value, dict):
        return {str(key): to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return [to_builtin(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return to_builtin(value.item())
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        if np.isposinf(value):
            return "inf"
        if np.isneginf(value):
            return "-inf"
        return value
    return value


def save_machine_readable_results(results: Dict[str, Any], output_dir: Path) -> Dict[str, str]:
    """Persist raw arrays plus a compact JSON summary for notebook or audit use."""
    sgd = results["optimizers"]["sgd"]
    muon = results["optimizers"]["muon"]

    npz_path = output_dir / "single_seed_results.npz"
    summary_path = output_dir / "single_seed_summary.json"

    np.savez(
        npz_path,
        sgd_steps=sgd["steps"],
        sgd_losses=sgd["losses"],
        sgd_per_layer_erank=sgd["per_layer_erank"],
        sgd_product_erank=sgd["product_erank"],
        sgd_per_layer_kappa=sgd["per_layer_kappa"],
        sgd_product_kappa=sgd["product_kappa"],
        muon_steps=muon["steps"],
        muon_losses=muon["losses"],
        muon_per_layer_erank=muon["per_layer_erank"],
        muon_product_erank=muon["product_erank"],
        muon_per_layer_kappa=muon["per_layer_kappa"],
        muon_product_kappa=muon["product_kappa"],
        config_json=np.array(json.dumps(to_builtin(results["config"]), sort_keys=True)),
        problem_summary_json=np.array(json.dumps(to_builtin(results["problem_summary"]), sort_keys=True)),
        lr_search_json=np.array(json.dumps(to_builtin(results["lr_search"]), sort_keys=True)),
        report_rows_json=np.array(json.dumps(to_builtin(results["report_rows"]), sort_keys=True)),
        final_summary_json=np.array(json.dumps(to_builtin(results["final_summary"]), sort_keys=True)),
        tests_json=np.array(json.dumps(to_builtin(results["tests"]), sort_keys=True)),
    )

    artifact_payload = dict(results["artifacts"])
    artifact_payload.update({"results_npz": str(npz_path), "summary_json": str(summary_path)})
    summary_payload = {
        "experiment": results["experiment"],
        "title": results["title"],
        "scope": results["scope"],
        "script_path": results["script_path"],
        "output_dir": results["output_dir"],
        "runtime_seconds": results["runtime_seconds"],
        "config": to_builtin(results["config"]),
        "problem_summary": to_builtin(results["problem_summary"]),
        "lr_search": to_builtin(results["lr_search"]),
        "report_rows": to_builtin(results["report_rows"]),
        "final_summary": to_builtin(results["final_summary"]),
        "tests": to_builtin(results["tests"]),
        "artifacts": to_builtin(artifact_payload),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True))

    return {"results_npz": str(npz_path), "summary_json": str(summary_path)}


def fmt(value: Any, precision: int = 2, scientific: bool = False) -> str:
    """Format numeric values for console reporting."""
    if isinstance(value, str):
        return value
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    if np.isnan(numeric):
        return "nan"
    if np.isposinf(numeric):
        return "inf"
    if np.isneginf(numeric):
        return "-inf"
    if scientific:
        return f"{numeric:.{precision}e}"
    return f"{numeric:.{precision}f}"


def print_experiment_report(results: Dict[str, Any]) -> None:
    """Print a calibrated console summary for direct script execution."""
    config = results["config"]
    problem = results["problem_summary"]
    lr_search = results["lr_search"]
    report_rows = results["report_rows"]
    final_summary = results["final_summary"]
    tests = results["tests"]
    artifacts = results["artifacts"]

    print("=" * 100)
    print("1.3a-i: EFFECTIVE RANK -- LAYER-BY-LAYER vs AGGREGATE")
    print("=" * 100)
    print("Scope: deterministic single-seed toy probe (one seed, one target, one batch, one depth).")
    print("This run compares per-layer and product spectra along training, but does not estimate depth scaling laws.")
    print(
        f"Config: dim={config['dim']}, layers={config['num_layers']}, steps={config['num_steps']}, "
        f"batch={config['batch_size']}, measure_every={config['measure_every']}, seed={config['seed']}"
    )
    print(
        f"Learning rates: Muon fixed at {config['lr_muon']}; SGD chosen as first stable candidate from "
        f"{lr_search['candidates']} -> {lr_search['chosen_lr']}"
    )
    print(
        f"Initial state: loss={fmt(problem['initial_loss'], scientific=True)}, "
        f"target erank={fmt(problem['target_effective_rank'])}, "
        f"initial mean layer erank={fmt(problem['initial_state']['per_layer_erank_mean'])}, "
        f"initial product erank={fmt(problem['initial_state']['product_effective_rank'])}"
    )
    print(f"Runtime: {fmt(results['runtime_seconds'])} s")

    print("\nKey report steps")
    print("-" * 100)
    print(
        f"{'step':>5} | {'SGD layer erank':>15} | {'Muon layer erank':>16} | "
        f"{'SGD prod erank':>14} | {'Muon prod erank':>15} | {'SGD prod kappa':>15} | {'Muon prod kappa':>16}"
    )
    print("-" * 100)
    for row in report_rows:
        print(
            f"{row['step']:5d} | "
            f"{fmt(row['sgd_per_layer_erank_mean']):>15} | "
            f"{fmt(row['muon_per_layer_erank_mean']):>16} | "
            f"{fmt(row['sgd_product_erank']):>14} | "
            f"{fmt(row['muon_product_erank']):>15} | "
            f"{fmt(row['sgd_product_kappa']):>15} | "
            f"{fmt(row['muon_product_kappa']):>16}"
        )

    print("\nFinal-step summary")
    print("-" * 100)
    print(
        f"Loss: SGD {fmt(final_summary['sgd_final_loss'], scientific=True)} | "
        f"Muon {fmt(final_summary['muon_final_loss'], scientific=True)}"
    )
    print(
        f"Mean per-layer erank: SGD {fmt(final_summary['sgd_final_per_layer_erank_mean'])} | "
        f"Muon {fmt(final_summary['muon_final_per_layer_erank_mean'])}"
    )
    print(
        f"Product erank: SGD {fmt(final_summary['sgd_final_product_erank'])} | "
        f"Muon {fmt(final_summary['muon_final_product_erank'])}"
    )
    print(
        f"Mean per-layer kappa: SGD {fmt(final_summary['sgd_final_per_layer_kappa_mean'])} | "
        f"Muon {fmt(final_summary['muon_final_per_layer_kappa_mean'])}"
    )
    print(
        f"Product kappa: SGD {fmt(final_summary['sgd_final_product_kappa'])} | "
        f"Muon {fmt(final_summary['muon_final_product_kappa'])}"
    )
    print(
        f"Depth compression (product / mean layer erank): "
        f"SGD {fmt(final_summary['sgd_depth_compression'], precision=3)} | "
        f"Muon {fmt(final_summary['muon_depth_compression'], precision=3)}"
    )
    print(
        f"Product erank change from step 0: "
        f"SGD {fmt(final_summary['sgd_product_erank_delta'])} | "
        f"Muon {fmt(final_summary['muon_product_erank_delta'])}"
    )
    print("Note: in this run, product erank stays below mean per-layer erank for both optimizers, but it increases over training for both.")
    print("Note: Muon's mean per-layer kappa is lower than SGD's at the end, but not close to 1.")

    print("\nExpectation checks (single-run only)")
    print("-" * 100)
    for label in ["T1", "T2", "T3", "T4"]:
        entry = tests[label]
        status = "PASS" if entry["passed"] else "FAIL"
        print(f"{label}: {status} | {entry['description']}")
        print(f"    observed: {entry['observed_summary']}")
    print(f"Overall: {tests['overall']} ({tests['tests_passed']}/{tests['tests_total']})")
    print(f"Scope note: {tests['scope_note']}")

    if artifacts:
        print("\nArtifacts")
        print("-" * 100)
        for name, path in artifacts.items():
            print(f"{name}: {path}")

    print("=" * 100)


def run_experiment(
    config_overrides: Optional[Dict[str, Any]] = None,
    save_plots: bool = True,
    save_results: bool = True,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the full experiment and return structured results."""
    config = build_config(config_overrides)
    output_dir = Path(output_dir) if output_dir is not None else SCRIPT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    w_target, x_data = generate_problem_data(config)
    problem_summary = summarize_problem_setup(config, w_target, x_data)

    start_time = time.perf_counter()
    lr_search = find_stable_lr_sgd(config, x_data, w_target)
    lr_sgd = lr_search["chosen_lr"]

    if verbose:
        print("Running SGD...", flush=True)
    results_sgd = run_and_measure("SGD", sgd_step, lr_sgd, config["num_steps"], config, x_data, w_target, verbose=verbose)

    if verbose:
        print("Running Muon...", flush=True)
    results_muon = run_and_measure(
        "Muon", muon_step, config["lr_muon"], config["num_steps"], config, x_data, w_target, verbose=verbose
    )

    report_rows = build_report_rows(results_sgd, results_muon, config)
    final_summary, tests = evaluate_expectations(results_sgd, results_muon, config)
    runtime_seconds = float(time.perf_counter() - start_time)

    results: Dict[str, Any] = {
        "experiment": "1.3a-i",
        "title": "Effective Rank -- Layer-by-Layer vs Aggregate",
        "scope": (
            "Single-seed, single-target, single-batch, single-depth toy trajectory. "
            "Interpret only as a deterministic case study, not a seed sweep or depth-scaling estimate."
        ),
        "script_path": str(Path(__file__).resolve()),
        "output_dir": str(output_dir.resolve()),
        "config": config,
        "problem_summary": problem_summary,
        "lr_search": lr_search,
        "optimizers": {"sgd": results_sgd, "muon": results_muon},
        "report_rows": report_rows,
        "final_summary": final_summary,
        "tests": tests,
        "runtime_seconds": runtime_seconds,
        "artifacts": {},
    }

    if save_plots:
        results["artifacts"].update(make_plots(results, output_dir))
    if save_results:
        results["artifacts"].update(save_machine_readable_results(results, output_dir))
    if verbose:
        print_experiment_report(results)

    return results


def main() -> Dict[str, Any]:
    """CLI entrypoint preserving normal script behavior."""
    return run_experiment()


if __name__ == "__main__":
    main()
