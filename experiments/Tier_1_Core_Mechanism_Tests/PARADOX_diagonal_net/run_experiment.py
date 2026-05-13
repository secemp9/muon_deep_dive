#!/usr/bin/env python3
"""
Experiment 3.18: Paradox on Diagonal Factorized Nets
====================================================

This benchmark compares a diagonal-factorized deep linear control against a
full-matrix deep linear reference. It preserves the original paradox protocol as
closely as possible while narrowing the interpretation:

- The diagonal-factorized model removes inter-layer orthogonal gauge freedom,
  but it does NOT establish a completely gauge-free or fully physical
  parameterization. Continuous reparameterization symmetries of the factorized
  product remain.
- The diagonal "Muon" update is a finite-iteration Newton-Schulz transform of
  the diagonal gradient. It is sign-like, but not exact coordinate-wise sign
  normalization after only a small number of iterations.
- Sampled output diversity is measured on a fixed finite test set X_test. This
  is a useful surrogate for functional diversity, but it is not by itself an
  exact operator comparison.
- To make the benchmark more honest, the script also reports exact end-to-end
  operator diversity computed from the trained weights.

The heuristic decision rules retained at the end of the script are benchmark
sanity checks, not formal statistical hypothesis tests.
"""

from __future__ import annotations

import copy
import time
from typing import Dict, Iterable, List, Tuple

import numpy as np


DEFAULT_CONFIG = {
    "global_seed": 42,
    "dim": 32,
    "num_layers": 4,
    "num_steps": 500,
    "batch_size": 64,
    "momentum": 0.9,
    "ns_iters": 5,
    "num_independent_runs": 20,
    "num_test_inputs": 50,
    "data_scale": 0.3,
    "warmup_steps": 100,
    "divergence_multiplier": 50.0,
    "divergence_loss_cap": 1e10,
    "init_seed_base": 1000,
    "diag_lr_candidates": [0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001],
    "full_lr_candidates": [0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.001],
}


def build_config(overrides: Dict | None = None) -> Dict:
    """Return a config dict with optional overrides."""
    config = copy.deepcopy(DEFAULT_CONFIG)
    if overrides:
        config.update(overrides)
    return config


# =============================================================================
# DATA GENERATION
# =============================================================================


def generate_problem(config: Dict) -> Dict[str, np.ndarray]:
    """Generate the fixed target and train/test inputs for a given config."""
    rng = np.random.RandomState(config["global_seed"])
    dim = config["dim"]
    batch_size = config["batch_size"]
    num_test_inputs = config["num_test_inputs"]
    scale = config["data_scale"]

    d_target = rng.randn(dim)
    X_data = rng.randn(dim, batch_size) * scale
    X_test = rng.randn(dim, num_test_inputs) * scale

    return {
        "d_target": d_target,
        "X_data": X_data,
        "X_test": X_test,
        "W_target_full": np.diag(d_target),
    }


# =============================================================================
# DIAGONAL NETWORK
# =============================================================================


def init_diag(num_layers: int, dim: int, seed: int = 42) -> List[np.ndarray]:
    """Initialize diagonal layers near ones for stability."""
    rng = np.random.RandomState(seed)
    diags = []
    for _ in range(num_layers):
        diags.append(np.ones(dim) + rng.randn(dim) * 0.1)
    return diags


def forward_diag(diags: List[np.ndarray], X: np.ndarray) -> np.ndarray:
    """Forward pass: multiply X by the product of diagonals."""
    prod = np.ones_like(diags[0])
    for d in diags:
        prod = prod * d
    return prod[:, None] * X


def effective_operator_diag(diags: List[np.ndarray]) -> np.ndarray:
    """Return the exact end-to-end diagonal operator as a vector of diagonal entries."""
    prod = np.ones_like(diags[0])
    for d in diags:
        prod = prod * d
    return prod


def compute_loss_diag(diags: List[np.ndarray], X: np.ndarray, target_diag: np.ndarray) -> float:
    """Quadratic loss for the diagonal network."""
    pred = forward_diag(diags, X)
    target_out = target_diag[:, None] * X
    diff = pred - target_out
    return 0.5 * np.mean(np.sum(diff ** 2, axis=0))


def compute_gradients_diag(diags: List[np.ndarray], X: np.ndarray, target_diag: np.ndarray) -> List[np.ndarray]:
    """Backpropagation for the diagonal network."""
    prod = effective_operator_diag(diags)
    err = prod - target_diag
    x_sq_mean = np.mean(X ** 2, axis=1)
    dL_dprod = err * x_sq_mean

    grads = []
    for d in diags:
        with np.errstate(divide="ignore", invalid="ignore"):
            grad_k = dL_dprod * prod / (d + 1e-30)
        grads.append(grad_k)
    return grads


def newton_schulz_diagonal(g_vec: np.ndarray, num_iters: int) -> np.ndarray:
    """
    Finite-iteration Newton-Schulz transform for a diagonal gradient.

    For a diagonal matrix, the fixed points are sign-like (+/-1) entries, but
    with only a finite number of iterations the output is generally not exactly
    coordinate-wise sign(g_vec).
    """
    norm = np.linalg.norm(g_vec)
    if norm < 1e-12:
        return g_vec.copy()
    x = g_vec / norm
    for _ in range(num_iters):
        x = 1.5 * x - 0.5 * x ** 3
    return x


# =============================================================================
# FULL-MATRIX NETWORK
# =============================================================================


def init_weights_full(num_layers: int, dim: int, seed: int = 42) -> List[np.ndarray]:
    """Initialize full matrices near identity for stability."""
    rng = np.random.RandomState(seed)
    weights = []
    for _ in range(num_layers):
        weights.append(np.eye(dim) + rng.randn(dim, dim) * 0.1)
    return weights


def forward_full(weights: List[np.ndarray], X: np.ndarray) -> np.ndarray:
    """Forward pass for the full-matrix network."""
    out = X.copy()
    for W in weights:
        out = W @ out
    return out


def effective_operator_full(weights: List[np.ndarray]) -> np.ndarray:
    """Return the exact end-to-end linear operator W_L ... W_1."""
    out = np.eye(weights[0].shape[0])
    for W in weights:
        out = W @ out
    return out


def compute_loss_full(weights: List[np.ndarray], X: np.ndarray, W_target: np.ndarray) -> float:
    """Quadratic loss for the full-matrix network."""
    pred = forward_full(weights, X)
    target_out = W_target @ X
    diff = pred - target_out
    return 0.5 * np.mean(np.sum(diff ** 2, axis=0))


def compute_gradients_full(weights: List[np.ndarray], X: np.ndarray, W_target: np.ndarray) -> List[np.ndarray]:
    """Backpropagation for the full-matrix network."""
    num_layers = len(weights)
    batch_size = X.shape[1]

    activations = [X.copy()]
    out = X.copy()
    for W in weights:
        out = W @ out
        activations.append(out.copy())

    target_out = W_target @ X
    delta = (activations[-1] - target_out) / batch_size

    grads = []
    for i in range(num_layers - 1, -1, -1):
        grad = delta @ activations[i].T
        grads.insert(0, grad)
        if i > 0:
            delta = weights[i].T @ delta
    return grads


def newton_schulz_full(G: np.ndarray, num_iters: int) -> np.ndarray:
    """Finite-iteration Newton-Schulz transform for a full matrix gradient."""
    norm = np.linalg.norm(G, ord="fro")
    if norm < 1e-12:
        return G.copy()
    X = G / norm
    for _ in range(num_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


# =============================================================================
# OPTIMIZER STEPS
# =============================================================================


def sgd_step_diag(
    diags: List[np.ndarray],
    velocities: List[np.ndarray],
    grads: List[np.ndarray],
    lr: float,
    momentum: float,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    for i in range(len(diags)):
        velocities[i] = momentum * velocities[i] + grads[i]
        diags[i] = diags[i] - lr * velocities[i]
    return diags, velocities


def muon_step_diag(
    diags: List[np.ndarray],
    velocities: List[np.ndarray],
    grads: List[np.ndarray],
    lr: float,
    momentum: float,
    ns_iters: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    for i in range(len(diags)):
        ortho_grad = newton_schulz_diagonal(grads[i], ns_iters)
        velocities[i] = momentum * velocities[i] + ortho_grad
        diags[i] = diags[i] - lr * velocities[i]
    return diags, velocities


def sgd_step_full(
    weights: List[np.ndarray],
    velocities: List[np.ndarray],
    grads: List[np.ndarray],
    lr: float,
    momentum: float,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    for i in range(len(weights)):
        velocities[i] = momentum * velocities[i] + grads[i]
        weights[i] = weights[i] - lr * velocities[i]
    return weights, velocities


def muon_step_full(
    weights: List[np.ndarray],
    velocities: List[np.ndarray],
    grads: List[np.ndarray],
    lr: float,
    momentum: float,
    ns_iters: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    for i in range(len(weights)):
        ortho_grad = newton_schulz_full(grads[i], ns_iters)
        velocities[i] = momentum * velocities[i] + ortho_grad
        weights[i] = weights[i] - lr * velocities[i]
    return weights, velocities


# =============================================================================
# LEARNING-RATE SEARCH
# =============================================================================


def find_stable_lr_diag(optimizer_name: str, candidates: Iterable[float], config: Dict, problem: Dict) -> Dict:
    """Find the first stable diagonal learning rate and record trial diagnostics."""
    dim = config["dim"]
    num_layers = config["num_layers"]
    momentum = config["momentum"]
    ns_iters = config["ns_iters"]
    warmup_steps = config["warmup_steps"]
    divergence_multiplier = config["divergence_multiplier"]
    target_diag = problem["d_target"]
    X_data = problem["X_data"]

    trials = []
    selected_lr = None

    for lr in candidates:
        diags = init_diag(num_layers, dim, seed=config["global_seed"])
        velocities = [np.zeros(dim) for _ in range(num_layers)]
        initial_loss = compute_loss_diag(diags, X_data, target_diag)
        stable = True
        steps_completed = 0
        loss = initial_loss

        for step in range(1, warmup_steps + 1):
            grads = compute_gradients_diag(diags, X_data, target_diag)
            if optimizer_name == "SGD":
                diags, velocities = sgd_step_diag(diags, velocities, grads, lr, momentum)
            else:
                diags, velocities = muon_step_diag(diags, velocities, grads, lr, momentum, ns_iters)
            loss = compute_loss_diag(diags, X_data, target_diag)
            steps_completed = step
            if np.isnan(loss) or loss > initial_loss * divergence_multiplier:
                stable = False
                break

        trials.append(
            {
                "lr": float(lr),
                "stable": bool(stable),
                "initial_loss": float(initial_loss),
                "final_loss": float(loss),
                "steps_completed": int(steps_completed),
            }
        )

        if stable:
            selected_lr = float(lr)
            break

    if selected_lr is None:
        selected_lr = float(list(candidates)[-1])

    return {
        "optimizer": optimizer_name,
        "candidates": list(candidates),
        "trials": trials,
        "selected_lr": selected_lr,
    }


def find_stable_lr_full(optimizer_name: str, candidates: Iterable[float], config: Dict, problem: Dict) -> Dict:
    """Find the first stable full-matrix learning rate and record trial diagnostics."""
    dim = config["dim"]
    num_layers = config["num_layers"]
    momentum = config["momentum"]
    ns_iters = config["ns_iters"]
    warmup_steps = config["warmup_steps"]
    divergence_multiplier = config["divergence_multiplier"]
    W_target_full = problem["W_target_full"]
    X_data = problem["X_data"]

    trials = []
    selected_lr = None

    for lr in candidates:
        weights = init_weights_full(num_layers, dim, seed=config["global_seed"])
        velocities = [np.zeros((dim, dim)) for _ in range(num_layers)]
        initial_loss = compute_loss_full(weights, X_data, W_target_full)
        stable = True
        steps_completed = 0
        loss = initial_loss

        for step in range(1, warmup_steps + 1):
            grads = compute_gradients_full(weights, X_data, W_target_full)
            if optimizer_name == "SGD":
                weights, velocities = sgd_step_full(weights, velocities, grads, lr, momentum)
            else:
                weights, velocities = muon_step_full(weights, velocities, grads, lr, momentum, ns_iters)
            loss = compute_loss_full(weights, X_data, W_target_full)
            steps_completed = step
            if np.isnan(loss) or loss > initial_loss * divergence_multiplier:
                stable = False
                break

        trials.append(
            {
                "lr": float(lr),
                "stable": bool(stable),
                "initial_loss": float(initial_loss),
                "final_loss": float(loss),
                "steps_completed": int(steps_completed),
            }
        )

        if stable:
            selected_lr = float(lr)
            break

    if selected_lr is None:
        selected_lr = float(list(candidates)[-1])

    return {
        "optimizer": optimizer_name,
        "candidates": list(candidates),
        "trials": trials,
        "selected_lr": selected_lr,
    }


# =============================================================================
# TRAINING LOOPS + RAW RESULT COLLECTION
# =============================================================================


def compute_pairwise_distances(objects: List[np.ndarray], distance_fn) -> Tuple[np.ndarray, np.ndarray]:
    """Return pair indices and pairwise distances for a list of objects."""
    pair_indices = []
    distances = []
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            pair_indices.append((i, j))
            distances.append(distance_fn(objects[i], objects[j]))
    return np.asarray(pair_indices, dtype=int), np.asarray(distances, dtype=float)


def summarize_metric(values: np.ndarray) -> Tuple[float, float]:
    """Mean/std summary for a 1D metric array."""
    return float(np.mean(values)), float(np.std(values))


def run_optimizer_diag(optimizer_name: str, lr: float, config: Dict, problem: Dict) -> Dict:
    """Run many independent diagonal-network trainings for one optimizer."""
    dim = config["dim"]
    num_layers = config["num_layers"]
    num_runs = config["num_independent_runs"]
    num_steps = config["num_steps"]
    momentum = config["momentum"]
    ns_iters = config["ns_iters"]
    init_seed_base = config["init_seed_base"]
    divergence_loss_cap = config["divergence_loss_cap"]

    d_target = problem["d_target"]
    X_data = problem["X_data"]
    X_test = problem["X_test"]
    X_test_norm = np.linalg.norm(X_test, ord="fro")

    final_parameters = []
    sampled_outputs = []
    effective_operators = []
    final_losses = []
    run_seeds = []
    steps_completed = []
    diverged = []
    loss_histories = np.full((num_runs, num_steps + 1), np.nan, dtype=float)

    for run_idx in range(num_runs):
        seed = init_seed_base + run_idx
        run_seeds.append(seed)

        diags = init_diag(num_layers, dim, seed=seed)
        velocities = [np.zeros(dim) for _ in range(num_layers)]

        initial_loss = compute_loss_diag(diags, X_data, d_target)
        loss_histories[run_idx, 0] = initial_loss
        run_diverged = False
        last_completed_step = 0

        for step in range(1, num_steps + 1):
            grads = compute_gradients_diag(diags, X_data, d_target)
            if optimizer_name == "SGD":
                diags, velocities = sgd_step_diag(diags, velocities, grads, lr, momentum)
            else:
                diags, velocities = muon_step_diag(diags, velocities, grads, lr, momentum, ns_iters)

            loss = compute_loss_diag(diags, X_data, d_target)
            loss_histories[run_idx, step] = loss
            last_completed_step = step

            if np.isnan(loss) or loss > divergence_loss_cap:
                run_diverged = True
                break

        final_loss = compute_loss_diag(diags, X_data, d_target)
        final_parameters.append(np.stack([d.copy() for d in diags], axis=0))
        sampled_outputs.append(forward_diag(diags, X_test).copy())
        effective_operators.append(effective_operator_diag(diags).copy())
        final_losses.append(final_loss)
        steps_completed.append(last_completed_step)
        diverged.append(run_diverged)

    pair_indices, weight_dists = compute_pairwise_distances(
        final_parameters,
        lambda a, b: np.linalg.norm(a - b),
    )
    _, sampled_output_dists = compute_pairwise_distances(
        sampled_outputs,
        lambda a, b: np.linalg.norm(a - b, ord="fro") / X_test_norm,
    )
    _, operator_dists = compute_pairwise_distances(
        effective_operators,
        lambda a, b: np.linalg.norm(a - b),
    )

    weight_mean, weight_std = summarize_metric(weight_dists)
    output_mean, output_std = summarize_metric(sampled_output_dists)
    operator_mean, operator_std = summarize_metric(operator_dists)
    loss_mean, loss_std = summarize_metric(np.asarray(final_losses, dtype=float))

    return {
        "optimizer": optimizer_name,
        "lr": float(lr),
        "run_seeds": np.asarray(run_seeds, dtype=int),
        "final_parameters": np.asarray(final_parameters, dtype=float),
        "sampled_outputs": np.asarray(sampled_outputs, dtype=float),
        "effective_operators": np.asarray(effective_operators, dtype=float),
        "final_losses": np.asarray(final_losses, dtype=float),
        "losses": np.asarray(final_losses, dtype=float),
        "loss_histories": loss_histories,
        "steps_completed": np.asarray(steps_completed, dtype=int),
        "diverged": np.asarray(diverged, dtype=bool),
        "pair_indices": pair_indices,
        "pairwise_weight_distances": weight_dists,
        "pairwise_sampled_output_distances": sampled_output_dists,
        "pairwise_function_distances": sampled_output_dists,
        "pairwise_operator_distances": operator_dists,
        "weight_diversity_mean": weight_mean,
        "weight_diversity_std": weight_std,
        "sampled_output_diversity_mean": output_mean,
        "sampled_output_diversity_std": output_std,
        "func_diversity_mean": output_mean,
        "func_diversity_std": output_std,
        "operator_diversity_mean": operator_mean,
        "operator_diversity_std": operator_std,
        "loss_mean": loss_mean,
        "loss_std": loss_std,
    }


def run_optimizer_full(optimizer_name: str, lr: float, config: Dict, problem: Dict) -> Dict:
    """Run many independent full-matrix trainings for one optimizer."""
    dim = config["dim"]
    num_layers = config["num_layers"]
    num_runs = config["num_independent_runs"]
    num_steps = config["num_steps"]
    momentum = config["momentum"]
    ns_iters = config["ns_iters"]
    init_seed_base = config["init_seed_base"]
    divergence_loss_cap = config["divergence_loss_cap"]

    W_target_full = problem["W_target_full"]
    X_data = problem["X_data"]
    X_test = problem["X_test"]
    X_test_norm = np.linalg.norm(X_test, ord="fro")

    final_parameters = []
    sampled_outputs = []
    effective_operators = []
    final_losses = []
    run_seeds = []
    steps_completed = []
    diverged = []
    loss_histories = np.full((num_runs, num_steps + 1), np.nan, dtype=float)

    for run_idx in range(num_runs):
        seed = init_seed_base + run_idx
        run_seeds.append(seed)

        weights = init_weights_full(num_layers, dim, seed=seed)
        velocities = [np.zeros((dim, dim)) for _ in range(num_layers)]

        initial_loss = compute_loss_full(weights, X_data, W_target_full)
        loss_histories[run_idx, 0] = initial_loss
        run_diverged = False
        last_completed_step = 0

        for step in range(1, num_steps + 1):
            grads = compute_gradients_full(weights, X_data, W_target_full)
            if optimizer_name == "SGD":
                weights, velocities = sgd_step_full(weights, velocities, grads, lr, momentum)
            else:
                weights, velocities = muon_step_full(weights, velocities, grads, lr, momentum, ns_iters)

            loss = compute_loss_full(weights, X_data, W_target_full)
            loss_histories[run_idx, step] = loss
            last_completed_step = step

            if np.isnan(loss) or loss > divergence_loss_cap:
                run_diverged = True
                break

        final_loss = compute_loss_full(weights, X_data, W_target_full)
        final_parameters.append(np.stack([w.copy() for w in weights], axis=0))
        sampled_outputs.append(forward_full(weights, X_test).copy())
        effective_operators.append(effective_operator_full(weights).copy())
        final_losses.append(final_loss)
        steps_completed.append(last_completed_step)
        diverged.append(run_diverged)

    pair_indices, weight_dists = compute_pairwise_distances(
        final_parameters,
        lambda a, b: np.linalg.norm(a - b),
    )
    _, sampled_output_dists = compute_pairwise_distances(
        sampled_outputs,
        lambda a, b: np.linalg.norm(a - b, ord="fro") / X_test_norm,
    )
    _, operator_dists = compute_pairwise_distances(
        effective_operators,
        lambda a, b: np.linalg.norm(a - b, ord="fro"),
    )

    weight_mean, weight_std = summarize_metric(weight_dists)
    output_mean, output_std = summarize_metric(sampled_output_dists)
    operator_mean, operator_std = summarize_metric(operator_dists)
    loss_mean, loss_std = summarize_metric(np.asarray(final_losses, dtype=float))

    return {
        "optimizer": optimizer_name,
        "lr": float(lr),
        "run_seeds": np.asarray(run_seeds, dtype=int),
        "final_parameters": np.asarray(final_parameters, dtype=float),
        "sampled_outputs": np.asarray(sampled_outputs, dtype=float),
        "effective_operators": np.asarray(effective_operators, dtype=float),
        "final_losses": np.asarray(final_losses, dtype=float),
        "losses": np.asarray(final_losses, dtype=float),
        "loss_histories": loss_histories,
        "steps_completed": np.asarray(steps_completed, dtype=int),
        "diverged": np.asarray(diverged, dtype=bool),
        "pair_indices": pair_indices,
        "pairwise_weight_distances": weight_dists,
        "pairwise_sampled_output_distances": sampled_output_dists,
        "pairwise_function_distances": sampled_output_dists,
        "pairwise_operator_distances": operator_dists,
        "weight_diversity_mean": weight_mean,
        "weight_diversity_std": weight_std,
        "sampled_output_diversity_mean": output_mean,
        "sampled_output_diversity_std": output_std,
        "func_diversity_mean": output_mean,
        "func_diversity_std": output_std,
        "operator_diversity_mean": operator_mean,
        "operator_diversity_std": operator_std,
        "loss_mean": loss_mean,
        "loss_std": loss_std,
    }


# =============================================================================
# SUMMARY + HEURISTICS
# =============================================================================


def safe_ratio(numerator: float, denominator: float) -> float:
    """Numerically safe ratio."""
    return float(numerator / (denominator + 1e-30))


def summarize_architecture(optimizer_results: Dict[str, Dict]) -> Dict:
    """Compute ratio summaries for one architecture."""
    sgd = optimizer_results["SGD"]
    muon = optimizer_results["Muon"]

    sampled_ratio_sgd = safe_ratio(sgd["sampled_output_diversity_mean"], sgd["weight_diversity_mean"])
    sampled_ratio_muon = safe_ratio(muon["sampled_output_diversity_mean"], muon["weight_diversity_mean"])
    operator_ratio_sgd = safe_ratio(sgd["operator_diversity_mean"], sgd["weight_diversity_mean"])
    operator_ratio_muon = safe_ratio(muon["operator_diversity_mean"], muon["weight_diversity_mean"])

    return {
        "sampled_output_ratio_sgd": sampled_ratio_sgd,
        "sampled_output_ratio_muon": sampled_ratio_muon,
        "sampled_output_paradox_strength": safe_ratio(sampled_ratio_sgd, sampled_ratio_muon),
        "operator_ratio_sgd": operator_ratio_sgd,
        "operator_ratio_muon": operator_ratio_muon,
        "operator_paradox_strength": safe_ratio(operator_ratio_sgd, operator_ratio_muon),
    }


def evaluate_heuristics(diagonal_results: Dict[str, Dict], full_results: Dict[str, Dict]) -> Dict:
    """Retain the original heuristic benchmark checks with narrower wording."""
    diagonal_summary = summarize_architecture(diagonal_results)
    full_summary = summarize_architecture(full_results)

    full_ratio_sgd = full_summary["sampled_output_ratio_sgd"]
    full_ratio_muon = full_summary["sampled_output_ratio_muon"]
    diag_ratio_sgd = diagonal_summary["sampled_output_ratio_sgd"]
    diag_ratio_muon = diagonal_summary["sampled_output_ratio_muon"]
    full_paradox = full_summary["sampled_output_paradox_strength"]
    diag_paradox = diagonal_summary["sampled_output_paradox_strength"]

    diag_muon_higher_wd = (
        diagonal_results["Muon"]["weight_diversity_mean"] > diagonal_results["SGD"]["weight_diversity_mean"]
    )
    diag_muon_lower_output = (
        diagonal_results["Muon"]["sampled_output_diversity_mean"]
        < diagonal_results["SGD"]["sampled_output_diversity_mean"]
    )
    diag_full_paradox_pattern = diag_muon_higher_wd and diag_muon_lower_output

    tests = {
        "T1_full_matrix_shows_sampled_output_paradox": {
            "description": "Full-matrix reference shows the paradox pattern (Muon sampled-output ratio < SGD).",
            "passed": bool(full_ratio_muon < full_ratio_sgd),
            "observed": {
                "Muon_ratio": full_ratio_muon,
                "SGD_ratio": full_ratio_sgd,
            },
        },
        "T2_diagonal_strength_near_one": {
            "description": "Diagonal-factorized control does not strongly differ between optimizers (strength near 1.0).",
            "passed": bool(abs(diag_paradox - 1.0) < 1.0),
            "observed": {
                "diagonal_paradox_strength": diag_paradox,
            },
        },
        "T3_full_stronger_than_diagonal": {
            "description": "Full-matrix reference has stronger paradox signature than the diagonal-factorized control.",
            "passed": bool(full_paradox > diag_paradox),
            "observed": {
                "full_paradox_strength": full_paradox,
                "diagonal_paradox_strength": diag_paradox,
            },
        },
        "T4_diagonal_avoids_full_paradox_pattern": {
            "description": "Diagonal-factorized control does not exhibit Muon higher weight diversity together with lower sampled-output diversity.",
            "passed": bool(not diag_full_paradox_pattern),
            "observed": {
                "muon_higher_weight_diversity": bool(diag_muon_higher_wd),
                "muon_lower_sampled_output_diversity": bool(diag_muon_lower_output),
                "full_paradox_pattern_in_diagonal": bool(diag_full_paradox_pattern),
            },
        },
    }

    total_pass = int(sum(test["passed"] for test in tests.values()))

    if total_pass >= 3:
        verdict_label = "HEURISTIC SUPPORT"
        verdict_message = (
            "Within this benchmark, the full-matrix reference shows a stronger paradox signature than the "
            "diagonal-factorized control. This is suggestive, not conclusive, about gauge-related mechanisms."
        )
    elif total_pass >= 2:
        verdict_label = "MIXED / INCONCLUSIVE"
        verdict_message = (
            "The benchmark gives mixed heuristic evidence. The diagonal-factorized control does not cleanly "
            "eliminate the paradox signal, so interpretation should remain cautious."
        )
    else:
        verdict_label = "HEURISTIC REJECTION"
        verdict_message = (
            "In this benchmark, paradox-like behavior persists in the diagonal-factorized control, so this "
            "probe does not support the claim that the observed paradox requires orthogonal gauge freedom."
        )

    return {
        "tests": tests,
        "total_pass": total_pass,
        "diagonal_sampled_output_paradox_strength": diag_paradox,
        "full_sampled_output_paradox_strength": full_paradox,
        "diagonal_sampled_output_ratio_sgd": diag_ratio_sgd,
        "diagonal_sampled_output_ratio_muon": diag_ratio_muon,
        "full_sampled_output_ratio_sgd": full_ratio_sgd,
        "full_sampled_output_ratio_muon": full_ratio_muon,
        "verdict_label": verdict_label,
        "verdict_message": verdict_message,
    }


def build_summary_rows(results: Dict) -> List[Dict[str, float]]:
    """Create a flat table of the main summary rows for printing or notebook use."""
    rows = []
    for architecture_key, architecture_label in [
        ("diagonal", "Diagonal factorized"),
        ("full_matrix", "Full matrix"),
    ]:
        ratios = results[architecture_key]["summary"]
        for optimizer_name in ["SGD", "Muon"]:
            r = results[architecture_key]["optimizers"][optimizer_name]
            rows.append(
                {
                    "architecture": architecture_label,
                    "optimizer": optimizer_name,
                    "lr": r["lr"],
                    "loss_mean": r["loss_mean"],
                    "loss_std": r["loss_std"],
                    "weight_diversity_mean": r["weight_diversity_mean"],
                    "sampled_output_diversity_mean": r["sampled_output_diversity_mean"],
                    "operator_diversity_mean": r["operator_diversity_mean"],
                    "sampled_output_over_weight": (
                        ratios["sampled_output_ratio_sgd"] if optimizer_name == "SGD" else ratios["sampled_output_ratio_muon"]
                    ),
                    "operator_over_weight": (
                        ratios["operator_ratio_sgd"] if optimizer_name == "SGD" else ratios["operator_ratio_muon"]
                    ),
                    "num_diverged_runs": int(np.sum(r["diverged"])),
                }
            )
    return rows


def print_summary(results: Dict) -> None:
    """Print a compact but honest CLI summary."""
    config = results["config"]
    heuristics = results["heuristics"]

    print("=" * 100)
    print("Experiment 3.18: Paradox on Diagonal Factorized Nets")
    print("=" * 100)
    print(
        "Scope: diagonal-factorized control vs full-matrix reference for the original paradox benchmark."
    )
    print(
        "Caveat: diagonal factorization removes inter-layer orthogonal gauge freedom but retains"
    )
    print(
        "        factorization/reparameterization symmetries; this is a control probe, not a proof of gauge removal."
    )
    print(
        "Muon variant: finite-iteration Newton-Schulz gradient transform (sign-like on diagonals, not exact sign)."
    )
    print()
    print(
        f"Config: dim={config['dim']}, layers={config['num_layers']}, runs={config['num_independent_runs']}, "
        f"steps={config['num_steps']}, batch={config['batch_size']}, ns_iters={config['ns_iters']}"
    )
    print(
        f"Seeds: global={config['global_seed']}, per-run={config['init_seed_base']}.."
        f"{config['init_seed_base'] + config['num_independent_runs'] - 1}"
    )
    print()
    print("Learning-rate selection:")
    print(
        f"  Diagonal factorized: SGD={results['learning_rates']['diagonal']['SGD']['selected_lr']}, "
        f"Muon={results['learning_rates']['diagonal']['Muon']['selected_lr']}"
    )
    print(
        f"  Full matrix:         SGD={results['learning_rates']['full_matrix']['SGD']['selected_lr']}, "
        f"Muon={results['learning_rates']['full_matrix']['Muon']['selected_lr']}"
    )
    print()
    print("Main summary metrics:")
    print(
        f"{'Architecture':<22} {'Opt':<8} {'Loss mean':>12} {'Loss std':>12} {'Weight div':>12} "
        f"{'Sampled out div':>16} {'Operator div':>14} {'Out/W':>10} {'Op/W':>10}"
    )
    print("-" * 128)
    for row in build_summary_rows(results):
        print(
            f"{row['architecture']:<22} {row['optimizer']:<8} {row['loss_mean']:>12.6e} "
            f"{row['loss_std']:>12.6e} {row['weight_diversity_mean']:>12.6f} "
            f"{row['sampled_output_diversity_mean']:>16.6f} {row['operator_diversity_mean']:>14.6f} "
            f"{row['sampled_output_over_weight']:>10.6f} {row['operator_over_weight']:>10.6f}"
        )

    print()
    print("Sampled-output paradox strengths (SGD ratio / Muon ratio):")
    print(
        f"  Diagonal factorized: {results['diagonal']['summary']['sampled_output_paradox_strength']:.4f}"
    )
    print(f"  Full matrix:         {results['full_matrix']['summary']['sampled_output_paradox_strength']:.4f}")
    print("Operator paradox strengths (SGD ratio / Muon ratio):")
    print(
        f"  Diagonal factorized: {results['diagonal']['summary']['operator_paradox_strength']:.4f}"
    )
    print(f"  Full matrix:         {results['full_matrix']['summary']['operator_paradox_strength']:.4f}")

    print()
    print("Heuristic benchmark checks (not formal statistical tests):")
    for key, test in heuristics["tests"].items():
        print(f"  {key}: {'PASS' if test['passed'] else 'FAIL'}")
        print(f"    {test['description']}")
        observed_parts = [f"{name}={value}" for name, value in test["observed"].items()]
        print(f"    observed: {', '.join(observed_parts)}")

    print()
    print(f"Heuristic tally: {heuristics['total_pass']}/4")
    print(f"Verdict: {heuristics['verdict_label']}")
    print(f"{heuristics['verdict_message']}")
    print(f"Runtime: {results['runtime_seconds']:.2f}s")
    print("=" * 100)


# =============================================================================
# TOP-LEVEL RUNNER
# =============================================================================


def run_experiment(config: Dict | None = None, emit_summary: bool = False) -> Dict:
    """Run the full benchmark and return structured raw results."""
    config = build_config(config)
    problem = generate_problem(config)

    start_time = time.time()

    diagonal_lr_search = {
        "SGD": find_stable_lr_diag("SGD", config["diag_lr_candidates"], config, problem),
        "Muon": find_stable_lr_diag("Muon", config["diag_lr_candidates"], config, problem),
    }
    diagonal_results = {
        "SGD": run_optimizer_diag("SGD", diagonal_lr_search["SGD"]["selected_lr"], config, problem),
        "Muon": run_optimizer_diag("Muon", diagonal_lr_search["Muon"]["selected_lr"], config, problem),
    }

    full_lr_search = {
        "SGD": find_stable_lr_full("SGD", config["full_lr_candidates"], config, problem),
        "Muon": find_stable_lr_full("Muon", config["full_lr_candidates"], config, problem),
    }
    full_results = {
        "SGD": run_optimizer_full("SGD", full_lr_search["SGD"]["selected_lr"], config, problem),
        "Muon": run_optimizer_full("Muon", full_lr_search["Muon"]["selected_lr"], config, problem),
    }

    results = {
        "config": config,
        "problem": problem,
        "learning_rates": {
            "diagonal": diagonal_lr_search,
            "full_matrix": full_lr_search,
        },
        "selected_lrs": {
            "diagonal": {opt: diagonal_lr_search[opt]["selected_lr"] for opt in ["SGD", "Muon"]},
            "full_matrix": {opt: full_lr_search[opt]["selected_lr"] for opt in ["SGD", "Muon"]},
        },
        "diagonal": {
            "name": "Diagonal factorized",
            "optimizers": diagonal_results,
            "summary": summarize_architecture(diagonal_results),
        },
        "full_matrix": {
            "name": "Full matrix",
            "optimizers": full_results,
            "summary": summarize_architecture(full_results),
        },
    }
    results["summary_rows"] = build_summary_rows(results)
    results["heuristics"] = evaluate_heuristics(diagonal_results, full_results)
    results["runtime_seconds"] = float(time.time() - start_time)

    if emit_summary:
        print_summary(results)

    return results


def main() -> Dict:
    """CLI entrypoint preserving script-style behavior."""
    return run_experiment(emit_summary=True)


if __name__ == "__main__":
    main()
