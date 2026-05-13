#!/usr/bin/env python3
"""
1.2b-i: Finite-Time Sensitivity to Gauge-Preserving vs. Non-Gauge Perturbations
=============================================================================

This second-pass strengthening keeps the original toy deep-linear training study but fixes the
largest remaining conceptual flaw from the first pass: the primary "gauge" condition is now a
true coupled gauge-preserving reparameterization of the deep linear product at t=0.

What this script computes
-------------------------
A 4-layer deep linear network is trained on a fixed quadratic objective under:
  - SGD with classical momentum
  - a Muon-style variant that orthogonalizes each gradient via Newton-Schulz

For each optimizer, the script compares a base trajectory against trajectories started from small
nearby perturbations and reports finite-time growth rates in three metrics:
  - lambda_W: raw stacked-weight distance growth
  - lambda_F: effective end-to-end map distance growth
  - lambda_L: mean per-layer distance growth

Implemented perturbation families
---------------------------------
  - "gauge"              : coupled gauge-preserving reparameterization
                            W_1' = G_1 W_1
                            W_i' = G_i W_i G_{i-1}^{-1}   for 1 < i < L
                            W_L' = W_L G_{L-1}^{-1}
                            with G_i = I + eps * S_i / ||S_i||_F
                            so W_eff' = W_eff exactly up to numerical roundoff.

  - "physical"           : independent skew right-multiplicative control perturbation
                            W_i' = W_i @ exp(eps * A_i / ||A_i||_F)

  - "symmetric_legacy"   : independent symmetric right-multiplicative perturbation retained as a
                            non-gauge legacy comparison
                            W_i' = W_i @ (I + eps * S_i / ||S_i||_F)

Important scope caveats
-----------------------
- This remains a finite-time perturbation sensitivity study, not an asymptotic maximal Lyapunov
  exponent estimator.
- The new primary "gauge" condition is function-preserving at t=0 by construction.
- The skew and independent symmetric conditions remain controls/comparators rather than quotient-
  space physical tangent constructions.
- For the true gauge-preserving condition, lambda_F is often undefined because d_F(0) is
  essentially zero; this is expected and is reported honestly rather than hidden.
"""

from __future__ import annotations

import copy
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


MAP_PRESERVATION_ABS_TOL = 1e-10
MAP_PRESERVATION_REL_TOL = 1e-12

DEFAULT_CONFIG: Dict[str, Any] = {
    "dim": 32,
    "num_layers": 4,
    "num_steps": 200,
    "batch_size": 64,
    "lr_muon": 0.005,
    "momentum": 0.9,
    "ns_iters": 5,
    "epsilon": 0.001,
    "num_perturbations": 20,
    "seed_global": 42,
    "seed_init": 42,
    "seed_perturb_base": 100,
    "sgd_lr_candidates": [0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.001],
    "lr_stability_steps": 80,
    "lr_stability_multiplier": 50.0,
    "divergence_loss_threshold": 1e10,
}

PERTURBATION_METADATA: Dict[str, Dict[str, Any]] = {
    "gauge": {
        "short_label": "coupled gauge",
        "display_label": "coupled gauge-preserving reparameterization (primary)",
        "is_true_gauge": True,
        "construction": (
            "Sample hidden symmetric generators S_k, set G_k = I + eps * S_k / ||S_k||_F, and "
            "apply W_1' = G_1 W_1, W_i' = G_i W_i G_{i-1}^{-1}, W_L' = W_L G_{L-1}^{-1}."
        ),
        "honesty_note": (
            "True internal reparameterization of the deep linear product at t=0. Any later "
            "effective-map drift reflects optimizer dynamics, not an initial function mismatch."
        ),
    },
    "physical": {
        "short_label": "skew control",
        "display_label": "independent skew right-multiplicative control",
        "is_true_gauge": False,
        "construction": "Apply W_i' = W_i @ exp(eps * A_i / ||A_i||_F) with random skew A_i independently.",
        "honesty_note": (
            "Skew right-multiplicative control perturbation. This is a control direction, not a "
            "quotient-space physical tangent construction."
        ),
    },
    "symmetric_legacy": {
        "short_label": "legacy symmetric",
        "display_label": "independent symmetric right-multiplicative (legacy non-gauge)",
        "is_true_gauge": False,
        "construction": "Apply W_i' = W_i @ (I + eps * S_i / ||S_i||_F) with random symmetric S_i independently.",
        "honesty_note": (
            "Retained legacy comparison. This changes the effective map at t=0, so it should not "
            "be interpreted as a literal gauge-orbit perturbation."
        ),
    },
}

OPTIMIZER_METADATA: Dict[str, Dict[str, str]] = {
    "sgd": {"display_name": "SGD", "label": "SGD with classical momentum"},
    "muon": {"display_name": "Muon", "label": "Muon-style momentum + Newton-Schulz orthogonalization"},
}


def get_default_config() -> Dict[str, Any]:
    """Return a fresh copy of the default configuration."""
    return copy.deepcopy(DEFAULT_CONFIG)


def get_perturbation_catalog() -> Dict[str, Dict[str, Any]]:
    """Return a fresh copy of the perturbation metadata catalog."""
    return copy.deepcopy(PERTURBATION_METADATA)


def merge_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Merge configuration overrides into the defaults."""
    config = get_default_config()
    if overrides:
        config.update(overrides)
    return config


def build_problem(config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Construct the fixed target matrix and data batch using the legacy seed order."""
    rng = np.random.RandomState(config["seed_global"])
    dim = config["dim"]
    batch_size = config["batch_size"]
    w_target = rng.randn(dim, dim) * 0.5
    x_data = rng.randn(dim, batch_size) * 0.3
    return {"W_target": w_target, "X_data": x_data}


def init_weights(num_layers: int, dim: int, seed: int = 42) -> List[np.ndarray]:
    """Initialize each layer near identity for stability."""
    rng = np.random.RandomState(seed)
    weights: List[np.ndarray] = []
    for _ in range(num_layers):
        weights.append(np.eye(dim) + rng.randn(dim, dim) * 0.1)
    return weights


def forward(weights: List[np.ndarray], x_data: np.ndarray) -> np.ndarray:
    """Forward pass: W_L @ ... @ W_1 @ X for the stored layer order."""
    out = x_data.copy()
    for weight in weights:
        out = weight @ out
    return out


def effective_weight(weights: List[np.ndarray]) -> np.ndarray:
    """Compute the end-to-end effective matrix W_eff = W_L ... W_1."""
    dim = weights[0].shape[0]
    w_eff = np.eye(dim)
    for weight in weights:
        w_eff = weight @ w_eff
    return w_eff


def compute_loss(weights: List[np.ndarray], x_data: np.ndarray, target: np.ndarray) -> float:
    """Quadratic loss on the fixed batch."""
    pred = forward(weights, x_data)
    target_out = target @ x_data
    diff = pred - target_out
    return float(0.5 * np.mean(np.sum(diff ** 2, axis=0)))


def compute_gradients(weights: List[np.ndarray], x_data: np.ndarray, target: np.ndarray) -> List[np.ndarray]:
    """Backpropagate through the deep linear network."""
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
    for idx in range(num_layers - 1, -1, -1):
        grad = delta @ activations[idx].T
        grads.insert(0, grad)
        if idx > 0:
            delta = weights[idx].T @ delta
    return grads


def newton_schulz_orthogonalize(grad: np.ndarray, num_iters: int = 5) -> np.ndarray:
    """Approximate the orthogonal polar factor of grad via Newton-Schulz iteration."""
    norm = np.linalg.norm(grad, ord="fro")
    if norm < 1e-12:
        return grad.copy()

    x_mat = grad / norm
    for _ in range(num_iters):
        a_mat = x_mat.T @ x_mat
        x_mat = 1.5 * x_mat - 0.5 * x_mat @ a_mat
    return x_mat


def random_symmetric(dim: int, rng: np.random.RandomState) -> np.ndarray:
    """Generate a random symmetric matrix."""
    mat = rng.randn(dim, dim)
    return 0.5 * (mat + mat.T)


def random_skew_symmetric(dim: int, rng: np.random.RandomState) -> np.ndarray:
    """Generate a random skew-symmetric matrix."""
    mat = rng.randn(dim, dim)
    return 0.5 * (mat - mat.T)


def matrix_exponential_pade(mat: np.ndarray, order: int = 6) -> np.ndarray:
    """Matrix exponential by scaling-and-squaring with a [order/order] Pade approximant."""
    norm_mat = np.linalg.norm(mat, ord="fro")
    if norm_mat < 1e-15:
        return np.eye(mat.shape[0])

    scale = max(0, int(np.ceil(np.log2(norm_mat + 1e-15))))
    mat_scaled = mat / (2 ** scale)

    ident = np.eye(mat.shape[0])
    numer = ident.copy()
    denom = ident.copy()
    mat_power = ident.copy()
    coeff = 1.0

    for k_idx in range(1, order + 1):
        coeff *= (order - k_idx + 1) / (k_idx * (2 * order - k_idx + 1))
        mat_power = mat_power @ mat_scaled
        numer += coeff * mat_power
        denom += ((-1) ** k_idx) * coeff * mat_power

    result = np.linalg.solve(denom, numer)
    for _ in range(scale):
        result = result @ result
    return result


def sgd_step(
    weights: List[np.ndarray],
    velocities: List[np.ndarray],
    lr: float,
    x_data: np.ndarray,
    target: np.ndarray,
    momentum: float,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """One step of SGD with classical momentum."""
    grads = compute_gradients(weights, x_data, target)
    for idx in range(len(weights)):
        velocities[idx] = momentum * velocities[idx] + grads[idx]
        weights[idx] = weights[idx] - lr * velocities[idx]
    return weights, velocities


def muon_step(
    weights: List[np.ndarray],
    velocities: List[np.ndarray],
    lr: float,
    x_data: np.ndarray,
    target: np.ndarray,
    momentum: float,
    ns_iters: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """One step of the Muon-style update."""
    grads = compute_gradients(weights, x_data, target)
    for idx in range(len(weights)):
        ortho_grad = newton_schulz_orthogonalize(grads[idx], num_iters=ns_iters)
        velocities[idx] = momentum * velocities[idx] + ortho_grad
        weights[idx] = weights[idx] - lr * velocities[idx]
    return weights, velocities


def run_trajectory(
    weights_init: List[np.ndarray],
    optimizer: str,
    lr: float,
    num_steps: int,
    x_data: np.ndarray,
    target: np.ndarray,
    momentum: float,
    ns_iters: int,
    divergence_loss_threshold: float,
) -> Dict[str, Any]:
    """Run an optimizer trajectory and retain all weight snapshots and losses."""
    weights = [weight.copy() for weight in weights_init]
    velocities = [np.zeros_like(weight) for weight in weights]

    trajectory: List[List[np.ndarray]] = [[weight.copy() for weight in weights]]
    losses: List[float] = [compute_loss(weights, x_data, target)]
    diverged = False

    for step in range(num_steps):
        if optimizer == "sgd":
            weights, velocities = sgd_step(weights, velocities, lr, x_data, target, momentum)
        elif optimizer == "muon":
            weights, velocities = muon_step(weights, velocities, lr, x_data, target, momentum, ns_iters)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        loss = compute_loss(weights, x_data, target)
        losses.append(loss)
        trajectory.append([weight.copy() for weight in weights])

        if np.isnan(loss) or loss > divergence_loss_threshold:
            diverged = True
            for _ in range(num_steps - step - 1):
                losses.append(loss)
                trajectory.append([weight.copy() for weight in weights])
            break

    return {
        "trajectory": trajectory,
        "losses": np.asarray(losses, dtype=float),
        "diverged": diverged,
    }


def compute_weight_distance(traj_a: List[List[np.ndarray]], traj_b: List[List[np.ndarray]]) -> np.ndarray:
    """Compute stacked-weight Frobenius distance d_W(t)."""
    num_times = min(len(traj_a), len(traj_b))
    distances: List[float] = []
    for time_idx in range(num_times):
        accum = 0.0
        for layer_idx in range(len(traj_a[time_idx])):
            accum += np.linalg.norm(traj_a[time_idx][layer_idx] - traj_b[time_idx][layer_idx], ord="fro") ** 2
        distances.append(math.sqrt(accum))
    return np.asarray(distances, dtype=float)


def compute_effective_distance(traj_a: List[List[np.ndarray]], traj_b: List[List[np.ndarray]]) -> np.ndarray:
    """Compute effective-map Frobenius distance d_F(t) = ||W_eff' - W_eff||_F."""
    num_times = min(len(traj_a), len(traj_b))
    distances: List[float] = []
    for time_idx in range(num_times):
        w_eff_a = effective_weight(traj_a[time_idx])
        w_eff_b = effective_weight(traj_b[time_idx])
        distances.append(float(np.linalg.norm(w_eff_a - w_eff_b, ord="fro")))
    return np.asarray(distances, dtype=float)


def compute_per_layer_distances(traj_a: List[List[np.ndarray]], traj_b: List[List[np.ndarray]]) -> np.ndarray:
    """Compute per-layer Frobenius distances over time."""
    num_times = min(len(traj_a), len(traj_b))
    num_layers = len(traj_a[0])
    per_layer = np.zeros((num_layers, num_times), dtype=float)
    for time_idx in range(num_times):
        for layer_idx in range(num_layers):
            per_layer[layer_idx, time_idx] = np.linalg.norm(
                traj_a[time_idx][layer_idx] - traj_b[time_idx][layer_idx], ord="fro"
            )
    return per_layer


def finite_time_growth(
    distances: np.ndarray,
    num_steps: int,
    d0_floor: float = 1e-15,
    dN_floor: float = 1e-15,
) -> float:
    """Compute the finite-time log growth rate (1/N) log(d_N / d_0) when the endpoints are meaningful."""
    d0 = float(distances[0])
    dN = float(distances[-1])
    if d0 > d0_floor and dN > dN_floor:
        return float((1.0 / num_steps) * np.log(dN / d0))
    if dN <= dN_floor:
        return float(-np.inf)
    return float(np.nan)


def summarize_samples(samples: np.ndarray) -> Dict[str, Any]:
    """Summarize a vector of samples, ignoring non-finite entries in moment estimates."""
    sample_array = np.asarray(samples, dtype=float)
    finite = sample_array[np.isfinite(sample_array)]
    summary: Dict[str, Any] = {
        "n_total": int(sample_array.size),
        "n_valid": int(finite.size),
        "n_invalid": int(sample_array.size - finite.size),
    }
    if finite.size == 0:
        summary.update(
            {
                "mean": float("nan"),
                "median": float("nan"),
                "std": float("nan"),
                "ci95": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
            }
        )
        return summary

    std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    ci95 = float(1.96 * std / math.sqrt(finite.size)) if finite.size > 1 else 0.0
    summary.update(
        {
            "mean": float(np.mean(finite)),
            "median": float(np.median(finite)),
            "std": std,
            "ci95": ci95,
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
        }
    )
    return summary


def classify_growth(value: float, tol: float = 1e-3) -> str:
    """Qualitative sign label for a finite-time exponent estimate."""
    if not np.isfinite(value):
        return "INVALID"
    if value < -tol:
        return "DECAY"
    if value > tol:
        return "GROW"
    return "NEUTRAL"


def _normalize_fro(mat: np.ndarray) -> np.ndarray:
    """Return mat scaled to unit Frobenius norm when possible."""
    return mat / max(np.linalg.norm(mat, ord="fro"), 1e-12)


def coupled_gauge_preserving_perturbation(
    weights_base: List[np.ndarray],
    epsilon: float,
    rng: np.random.RandomState,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Apply an exact hidden-basis reparameterization that preserves the deep linear product."""
    dim = weights_base[0].shape[0]
    num_layers = len(weights_base)
    ident = np.eye(dim)

    transforms: List[np.ndarray] = []
    inverses: List[np.ndarray] = []
    condition_numbers: List[float] = []
    generator_norms: List[float] = []

    for _ in range(num_layers - 1):
        sym = _normalize_fro(random_symmetric(dim, rng))
        gauge_transform = ident + epsilon * sym
        gauge_inverse = np.linalg.inv(gauge_transform)
        transforms.append(gauge_transform)
        inverses.append(gauge_inverse)
        condition_numbers.append(float(np.linalg.cond(gauge_transform)))
        generator_norms.append(float(np.linalg.norm(sym, ord="fro")))

    perturbed: List[np.ndarray] = []
    for idx, weight in enumerate(weights_base):
        left = transforms[idx] if idx < num_layers - 1 else ident
        right = inverses[idx - 1] if idx > 0 else ident
        perturbed.append(left @ weight @ right)

    return perturbed, {
        "family": "coupled_hidden_basis_change",
        "num_hidden_transforms": int(num_layers - 1),
        "transform_condition_numbers": condition_numbers,
        "generator_fro_norms": generator_norms,
    }


def independent_symmetric_legacy_perturbation(
    weights_base: List[np.ndarray],
    epsilon: float,
    rng: np.random.RandomState,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Apply the original non-gauge independent symmetric right-multiplicative perturbation."""
    dim = weights_base[0].shape[0]
    perturbed: List[np.ndarray] = []
    generator_norms: List[float] = []
    for weight in weights_base:
        sym = _normalize_fro(random_symmetric(dim, rng))
        perturbed_weight = weight @ (np.eye(dim) + epsilon * sym)
        perturbed.append(perturbed_weight)
        generator_norms.append(float(np.linalg.norm(sym, ord="fro")))
    return perturbed, {
        "family": "independent_right_multiplicative_symmetric",
        "generator_fro_norms": generator_norms,
    }


def independent_skew_control_perturbation(
    weights_base: List[np.ndarray],
    epsilon: float,
    rng: np.random.RandomState,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Apply the independent skew right-multiplicative control perturbation."""
    dim = weights_base[0].shape[0]
    perturbed: List[np.ndarray] = []
    generator_norms: List[float] = []
    orthogonality_errors: List[float] = []
    for weight in weights_base:
        skew = _normalize_fro(random_skew_symmetric(dim, rng))
        rotation = matrix_exponential_pade(epsilon * skew)
        perturbed_weight = weight @ rotation
        perturbed.append(perturbed_weight)
        generator_norms.append(float(np.linalg.norm(skew, ord="fro")))
        orthogonality_errors.append(float(np.linalg.norm(rotation.T @ rotation - np.eye(dim), ord="fro")))
    return perturbed, {
        "family": "independent_right_multiplicative_skew",
        "generator_fro_norms": generator_norms,
        "rotation_orthogonality_errors": orthogonality_errors,
    }


def perturb_weights(
    weights_base: List[np.ndarray],
    perturbation_type: str,
    epsilon: float,
    rng: np.random.RandomState,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Construct a perturbed copy of the base weights and lightweight construction diagnostics."""
    if perturbation_type == "gauge":
        return coupled_gauge_preserving_perturbation(weights_base, epsilon, rng)
    if perturbation_type == "physical":
        return independent_skew_control_perturbation(weights_base, epsilon, rng)
    if perturbation_type == "symmetric_legacy":
        return independent_symmetric_legacy_perturbation(weights_base, epsilon, rng)
    raise ValueError(f"Unknown perturbation type: {perturbation_type}")


def find_stable_lr_sgd(config: Dict[str, Any], problem: Dict[str, np.ndarray]) -> Tuple[float, List[Dict[str, Any]]]:
    """Reproduce the legacy search for the largest empirically stable SGD learning rate."""
    diagnostics: List[Dict[str, Any]] = []
    dim = config["dim"]
    num_layers = config["num_layers"]
    momentum = config["momentum"]
    x_data = problem["X_data"]
    target = problem["W_target"]

    for lr in config["sgd_lr_candidates"]:
        weights = init_weights(num_layers, dim, seed=config["seed_init"])
        velocities = [np.zeros((dim, dim)) for _ in range(num_layers)]
        initial_loss = compute_loss(weights, x_data, target)
        stable = True
        final_loss = initial_loss

        for _ in range(config["lr_stability_steps"]):
            weights, velocities = sgd_step(weights, velocities, lr, x_data, target, momentum)
            final_loss = compute_loss(weights, x_data, target)
            if np.isnan(final_loss) or final_loss > initial_loss * config["lr_stability_multiplier"]:
                stable = False
                break

        diagnostics.append(
            {
                "lr": float(lr),
                "stable": bool(stable),
                "initial_loss": float(initial_loss),
                "final_loss": float(final_loss),
            }
        )
        if stable:
            return float(lr), diagnostics

    fallback = float(config["sgd_lr_candidates"][-1])
    return fallback, diagnostics


def run_training_checks(config: Dict[str, Any], problem: Dict[str, np.ndarray], lr_sgd: float) -> Dict[str, Any]:
    """Run a training sanity check for each optimizer using the default trajectory length."""
    checks: Dict[str, Any] = {}
    dim = config["dim"]
    num_layers = config["num_layers"]
    x_data = problem["X_data"]
    target = problem["W_target"]

    for optimizer_key, lr in [("sgd", lr_sgd), ("muon", config["lr_muon"] )]:
        weights_init = init_weights(num_layers, dim, seed=config["seed_init"])
        run = run_trajectory(
            weights_init=weights_init,
            optimizer=optimizer_key,
            lr=lr,
            num_steps=config["num_steps"],
            x_data=x_data,
            target=target,
            momentum=config["momentum"],
            ns_iters=config["ns_iters"],
            divergence_loss_threshold=config["divergence_loss_threshold"],
        )
        losses = np.asarray(run["losses"], dtype=float)
        checks[optimizer_key] = {
            "optimizer_key": optimizer_key,
            "optimizer_name": OPTIMIZER_METADATA[optimizer_key]["display_name"],
            "learning_rate": float(lr),
            "initial_loss": float(losses[0]),
            "final_loss": float(losses[-1]),
            "loss_ratio": float(losses[-1] / losses[0]),
            "diverged": bool(run["diverged"]),
            "losses": losses,
        }
    return checks


def build_base_run(
    optimizer_key: str,
    lr: float,
    config: Dict[str, Any],
    problem: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Run and cache the unperturbed baseline trajectory for one optimizer."""
    weights_base = init_weights(config["num_layers"], config["dim"], seed=config["seed_init"])
    base_run = run_trajectory(
        weights_init=weights_base,
        optimizer=optimizer_key,
        lr=lr,
        num_steps=config["num_steps"],
        x_data=problem["X_data"],
        target=problem["W_target"],
        momentum=config["momentum"],
        ns_iters=config["ns_iters"],
        divergence_loss_threshold=config["divergence_loss_threshold"],
    )
    return {"weights_base": weights_base, "base_run": base_run}


def measure_condition(
    optimizer_key: str,
    lr: float,
    perturbation_type: str,
    config: Dict[str, Any],
    problem: Dict[str, np.ndarray],
    weights_base: List[np.ndarray],
    base_run: Dict[str, Any],
) -> Dict[str, Any]:
    """Measure finite-time perturbation sensitivity for one optimizer/perturbation pair."""
    num_layers = config["num_layers"]
    num_steps = config["num_steps"]
    num_perturbations = config["num_perturbations"]
    epsilon = config["epsilon"]
    x_data = problem["X_data"]
    target = problem["W_target"]

    traj_base = base_run["trajectory"]
    base_losses = np.asarray(base_run["losses"], dtype=float)
    base_initial_loss = float(base_losses[0])
    base_effective = effective_weight(weights_base)
    base_effective_norm = float(np.linalg.norm(base_effective, ord="fro"))
    base_output = base_effective @ x_data
    base_output_norm = float(np.linalg.norm(base_output, ord="fro"))

    lambda_w_all: List[float] = []
    lambda_f_all: List[float] = []
    lambda_l_all: List[float] = []
    layer_lambda_all: List[np.ndarray] = []
    distances_w_all: List[np.ndarray] = []
    distances_f_all: List[np.ndarray] = []
    d0_w: List[float] = []
    dN_w: List[float] = []
    d0_f: List[float] = []
    dN_f: List[float] = []
    init_eff_abs_all: List[float] = []
    init_eff_rel_all: List[float] = []
    init_output_rel_all: List[float] = []
    init_loss_gap_all: List[float] = []
    trial_records: List[Dict[str, Any]] = []
    representative_trial: Optional[Dict[str, Any]] = None

    for trial_idx in range(num_perturbations):
        trial_seed = int(config["seed_perturb_base"] + trial_idx)
        rng = np.random.RandomState(trial_seed)
        weights_perturbed, construction_diagnostics = perturb_weights(weights_base, perturbation_type, epsilon, rng)

        perturbed_effective = effective_weight(weights_perturbed)
        init_eff_abs = float(np.linalg.norm(perturbed_effective - base_effective, ord="fro"))
        init_eff_rel = float(init_eff_abs / max(base_effective_norm, 1e-12))
        init_output_rel = float(
            np.linalg.norm((perturbed_effective - base_effective) @ x_data, ord="fro") / max(base_output_norm, 1e-12)
        )
        perturbed_initial_loss = compute_loss(weights_perturbed, x_data, target)
        init_loss_gap = float(abs(perturbed_initial_loss - base_initial_loss))

        perturbed_run = run_trajectory(
            weights_init=weights_perturbed,
            optimizer=optimizer_key,
            lr=lr,
            num_steps=num_steps,
            x_data=x_data,
            target=target,
            momentum=config["momentum"],
            ns_iters=config["ns_iters"],
            divergence_loss_threshold=config["divergence_loss_threshold"],
        )
        traj_perturbed = perturbed_run["trajectory"]

        dist_w = compute_weight_distance(traj_base, traj_perturbed)
        dist_f = compute_effective_distance(traj_base, traj_perturbed)
        per_layer_dist = compute_per_layer_distances(traj_base, traj_perturbed)
        layer_lambdas = np.asarray(
            [finite_time_growth(per_layer_dist[layer_idx], num_steps) for layer_idx in range(num_layers)],
            dtype=float,
        )
        finite_layers = layer_lambdas[np.isfinite(layer_lambdas)]
        lambda_l = float(np.mean(finite_layers)) if finite_layers.size > 0 else float("nan")

        lambda_w = finite_time_growth(dist_w, num_steps)
        lambda_f = finite_time_growth(
            dist_f,
            num_steps,
            d0_floor=MAP_PRESERVATION_ABS_TOL if PERTURBATION_METADATA[perturbation_type]["is_true_gauge"] else 1e-15,
        )

        lambda_w_all.append(lambda_w)
        lambda_f_all.append(lambda_f)
        lambda_l_all.append(lambda_l)
        layer_lambda_all.append(layer_lambdas)
        distances_w_all.append(dist_w)
        distances_f_all.append(dist_f)
        d0_w.append(float(dist_w[0]))
        dN_w.append(float(dist_w[-1]))
        d0_f.append(float(dist_f[0]))
        dN_f.append(float(dist_f[-1]))
        init_eff_abs_all.append(init_eff_abs)
        init_eff_rel_all.append(init_eff_rel)
        init_output_rel_all.append(init_output_rel)
        init_loss_gap_all.append(init_loss_gap)

        trial_records.append(
            {
                "trial_index": int(trial_idx),
                "seed": trial_seed,
                "lambda_W": float(lambda_w),
                "lambda_F": float(lambda_f),
                "lambda_L": float(lambda_l),
                "lambda_L_per_layer": layer_lambdas.copy(),
                "d0_W": float(dist_w[0]),
                "dN_W": float(dist_w[-1]),
                "d0_F": float(dist_f[0]),
                "dN_F": float(dist_f[-1]),
                "initial_effective_mismatch_abs": init_eff_abs,
                "initial_effective_mismatch_rel": init_eff_rel,
                "initial_output_mismatch_rel": init_output_rel,
                "initial_loss_gap_abs": init_loss_gap,
                "construction_diagnostics": construction_diagnostics,
            }
        )

        if representative_trial is None:
            representative_trial = {
                "trial_index": int(trial_idx),
                "seed": trial_seed,
                "distance_W": dist_w,
                "distance_F": dist_f,
                "per_layer_distances": per_layer_dist,
                "lambda_L_per_layer": layer_lambdas,
                "initial_effective_mismatch_abs": init_eff_abs,
                "initial_effective_mismatch_rel": init_eff_rel,
                "initial_output_mismatch_rel": init_output_rel,
                "initial_loss_gap_abs": init_loss_gap,
                "construction_diagnostics": construction_diagnostics,
            }

    lambda_w_arr = np.asarray(lambda_w_all, dtype=float)
    lambda_f_arr = np.asarray(lambda_f_all, dtype=float)
    lambda_l_arr = np.asarray(lambda_l_all, dtype=float)
    layer_lambda_arr = np.asarray(layer_lambda_all, dtype=float)
    d0_w_arr = np.asarray(d0_w, dtype=float)
    dN_w_arr = np.asarray(dN_w, dtype=float)
    d0_f_arr = np.asarray(d0_f, dtype=float)
    dN_f_arr = np.asarray(dN_f, dtype=float)
    init_eff_abs_arr = np.asarray(init_eff_abs_all, dtype=float)
    init_eff_rel_arr = np.asarray(init_eff_rel_all, dtype=float)
    init_output_rel_arr = np.asarray(init_output_rel_all, dtype=float)
    init_loss_gap_arr = np.asarray(init_loss_gap_all, dtype=float)

    summary = {
        "lambda_W": summarize_samples(lambda_w_arr),
        "lambda_F": summarize_samples(lambda_f_arr),
        "lambda_L": summarize_samples(lambda_l_arr),
        "lambda_W_class": classify_growth(float(np.nanmean(lambda_w_arr[np.isfinite(lambda_w_arr)])))
        if np.any(np.isfinite(lambda_w_arr))
        else "INVALID",
        "lambda_F_class": classify_growth(float(np.nanmean(lambda_f_arr[np.isfinite(lambda_f_arr)])))
        if np.any(np.isfinite(lambda_f_arr))
        else "INVALID",
        "lambda_L_class": classify_growth(float(np.nanmean(lambda_l_arr[np.isfinite(lambda_l_arr)])))
        if np.any(np.isfinite(lambda_l_arr))
        else "INVALID",
        "mean_d0_W": float(np.mean(d0_w_arr)),
        "mean_dN_W": float(np.mean(dN_w_arr)),
        "mean_d0_F": float(np.mean(d0_f_arr)),
        "mean_dN_F": float(np.mean(dN_f_arr)),
        "mean_ratio_W": float(np.mean(dN_w_arr) / np.mean(d0_w_arr)) if np.mean(d0_w_arr) > 0 else float("nan"),
        "mean_ratio_F": float(np.mean(dN_f_arr) / np.mean(d0_f_arr)) if np.mean(d0_f_arr) > 0 else float("nan"),
    }

    initial_diagnostics = {
        "base_effective_norm": base_effective_norm,
        "base_output_norm": base_output_norm,
        "base_initial_loss": base_initial_loss,
        "initial_effective_mismatch_abs": summarize_samples(init_eff_abs_arr),
        "initial_effective_mismatch_rel": summarize_samples(init_eff_rel_arr),
        "initial_output_mismatch_rel": summarize_samples(init_output_rel_arr),
        "initial_loss_gap_abs": summarize_samples(init_loss_gap_arr),
        "max_initial_effective_mismatch_abs": float(np.max(init_eff_abs_arr)),
        "max_initial_effective_mismatch_rel": float(np.max(init_eff_rel_arr)),
        "map_preservation_abs_tol": MAP_PRESERVATION_ABS_TOL,
        "map_preservation_rel_tol": MAP_PRESERVATION_REL_TOL,
        "preserves_effective_map_within_tol": bool(
            np.max(init_eff_abs_arr) <= MAP_PRESERVATION_ABS_TOL
            and np.max(init_eff_rel_arr) <= MAP_PRESERVATION_REL_TOL
        ),
    }

    return {
        "optimizer_key": optimizer_key,
        "optimizer_name": OPTIMIZER_METADATA[optimizer_key]["display_name"],
        "optimizer_label": OPTIMIZER_METADATA[optimizer_key]["label"],
        "learning_rate": float(lr),
        "perturbation_type": perturbation_type,
        "perturbation_metadata": copy.deepcopy(PERTURBATION_METADATA[perturbation_type]),
        "lambda_W_all": lambda_w_arr,
        "lambda_F_all": lambda_f_arr,
        "lambda_L_all": lambda_l_arr,
        "layer_lambda_all": layer_lambda_arr,
        "distances_W": distances_w_all,
        "distances_F": distances_f_all,
        "d0_W": d0_w_arr,
        "dN_W": dN_w_arr,
        "d0_F": d0_f_arr,
        "dN_F": dN_f_arr,
        "initial_effective_mismatch_abs_all": init_eff_abs_arr,
        "initial_effective_mismatch_rel_all": init_eff_rel_arr,
        "initial_output_mismatch_rel_all": init_output_rel_arr,
        "initial_loss_gap_abs_all": init_loss_gap_arr,
        "initial_diagnostics": initial_diagnostics,
        "summary": summary,
        "trial_records": trial_records,
        "base_losses": base_losses,
        "representative_trial": representative_trial,
    }


def welch_t_test(sample_a: np.ndarray, sample_b: np.ndarray) -> Dict[str, Any]:
    """Compute a Welch-style t-statistic without requiring SciPy."""
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    result: Dict[str, Any] = {
        "n_a": int(a.size),
        "n_b": int(b.size),
        "mean_a": float(np.mean(a)) if a.size else float("nan"),
        "mean_b": float(np.mean(b)) if b.size else float("nan"),
    }
    if a.size > 1:
        result["std_a"] = float(np.std(a, ddof=1))
    else:
        result["std_a"] = float("nan")
    if b.size > 1:
        result["std_b"] = float(np.std(b, ddof=1))
    else:
        result["std_b"] = float("nan")

    if a.size <= 1 or b.size <= 1:
        result.update(
            {
                "difference": float("nan"),
                "t_stat": float("nan"),
                "degrees_freedom": float("nan"),
                "heuristic_one_tailed_significant": False,
            }
        )
        return result

    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    se = math.sqrt(var_a / a.size + var_b / b.size)

    if se > 1e-15:
        t_stat = (mean_a - mean_b) / se
        df_num = (var_a / a.size + var_b / b.size) ** 2
        df_den = ((var_a / a.size) ** 2) / (a.size - 1) + ((var_b / b.size) ** 2) / (b.size - 1)
        degrees_freedom = df_num / (df_den + 1e-15)
    else:
        t_stat = float("inf") if mean_a != mean_b else 0.0
        degrees_freedom = float(min(a.size, b.size) - 1)

    result.update(
        {
            "difference": float(mean_a - mean_b),
            "t_stat": float(t_stat),
            "degrees_freedom": float(degrees_freedom),
            "heuristic_one_tailed_significant": bool(np.isfinite(t_stat) and t_stat > 2.0),
        }
    )
    return result


def build_verdict(
    condition_results: Dict[str, Dict[str, Any]],
    primary_stats: Dict[str, Any],
    legacy_stats: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate the primary gauge-preserving story plus legacy comparison context."""
    lambda_gauge_sgd = condition_results["SGD_gauge"]["summary"]["lambda_W"]["mean"]
    lambda_gauge_muon = condition_results["Muon_gauge"]["summary"]["lambda_W"]["mean"]
    lambda_phys_sgd = condition_results["SGD_physical"]["summary"]["lambda_W"]["mean"]
    lambda_phys_muon = condition_results["Muon_physical"]["summary"]["lambda_W"]["mean"]
    lambda_legacy_sgd = condition_results["SGD_symmetric_legacy"]["summary"]["lambda_W"]["mean"]
    lambda_legacy_muon = condition_results["Muon_symmetric_legacy"]["summary"]["lambda_W"]["mean"]

    gauge_abs_max = max(
        condition_results["SGD_gauge"]["initial_diagnostics"]["max_initial_effective_mismatch_abs"],
        condition_results["Muon_gauge"]["initial_diagnostics"]["max_initial_effective_mismatch_abs"],
    )
    gauge_rel_max = max(
        condition_results["SGD_gauge"]["initial_diagnostics"]["max_initial_effective_mismatch_rel"],
        condition_results["Muon_gauge"]["initial_diagnostics"]["max_initial_effective_mismatch_rel"],
    )
    gauge_construction_pass = bool(
        condition_results["SGD_gauge"]["initial_diagnostics"]["preserves_effective_map_within_tol"]
        and condition_results["Muon_gauge"]["initial_diagnostics"]["preserves_effective_map_within_tol"]
    )

    gauge_construction_check = {
        "description": (
            "The primary gauge condition should preserve the effective end-to-end map at t=0 up to "
            "numerical roundoff."
        ),
        "pass": gauge_construction_pass,
        "tolerance_abs": MAP_PRESERVATION_ABS_TOL,
        "tolerance_rel": MAP_PRESERVATION_REL_TOL,
        "max_abs_observed": float(gauge_abs_max),
        "max_rel_observed": float(gauge_rel_max),
    }

    primary_tests = {
        "muon_gauge_more_stable_than_sgd": {
            "description": "Muon has smaller lambda_W than SGD under the true gauge-preserving perturbation.",
            "metric": "lambda_W",
            "pass": bool(lambda_gauge_muon < lambda_gauge_sgd),
            "lhs": float(lambda_gauge_muon),
            "rhs": float(lambda_gauge_sgd),
        },
        "muon_gauge_contracts": {
            "description": "Muon's gauge-perturbation lambda_W is negative.",
            "metric": "lambda_W",
            "pass": bool(lambda_gauge_muon < -0.001),
            "lhs": float(lambda_gauge_muon),
            "rhs": -0.001,
        },
        "muon_gauge_more_stable_than_muon_skew": {
            "description": "Muon is more stable under the gauge perturbation than under the skew control.",
            "metric": "lambda_W",
            "pass": bool(lambda_gauge_muon < lambda_phys_muon - 0.001),
            "lhs": float(lambda_gauge_muon),
            "rhs": float(lambda_phys_muon - 0.001),
        },
    }

    legacy_tests = {
        "muon_legacy_symmetric_more_stable_than_sgd": {
            "description": "Muon has smaller lambda_W than SGD under the retained legacy symmetric perturbation.",
            "metric": "lambda_W",
            "pass": bool(lambda_legacy_muon < lambda_legacy_sgd),
            "lhs": float(lambda_legacy_muon),
            "rhs": float(lambda_legacy_sgd),
        },
        "muon_legacy_symmetric_contracts": {
            "description": "Muon's legacy symmetric lambda_W is negative.",
            "metric": "lambda_W",
            "pass": bool(lambda_legacy_muon < -0.001),
            "lhs": float(lambda_legacy_muon),
            "rhs": -0.001,
        },
        "muon_legacy_more_stable_than_muon_skew": {
            "description": "Muon is more stable under the legacy symmetric perturbation than under the skew control.",
            "metric": "lambda_W",
            "pass": bool(lambda_legacy_muon < lambda_phys_muon - 0.001),
            "lhs": float(lambda_legacy_muon),
            "rhs": float(lambda_phys_muon - 0.001),
        },
    }

    primary_tests_passed = int(sum(test["pass"] for test in primary_tests.values()))
    legacy_tests_passed = int(sum(test["pass"] for test in legacy_tests.values()))

    if not gauge_construction_pass:
        overall = "INVALID"
        interpretation = (
            "The nominal gauge condition did not preserve the effective map tightly enough at t=0, "
            "so the intended gauge-direction story would remain compromised."
        )
    elif primary_tests_passed == 3:
        overall = "PASS"
        interpretation = (
            "The gauge-preserving construction is numerically validated at t=0 and all three retained "
            "lambda_W stability checks pass under that primary condition."
        )
    elif primary_tests_passed == 2:
        overall = "PARTIAL PASS"
        interpretation = (
            "The pair now tests a true gauge-preserving perturbation at t=0, but only two of three "
            "primary lambda_W stability checks pass."
        )
    elif primary_tests_passed == 1:
        overall = "WEAK SIGNAL"
        interpretation = (
            "The pair now supports a literal gauge-preserving initialization test, but only one of three "
            "primary lambda_W stability checks passes, so evidence for the stability story is weak."
        )
    else:
        overall = "FAIL"
        interpretation = (
            "The pair now supports a literal gauge-preserving initialization test, but under the default "
            "run Muon does not show lower finite-time weight-space sensitivity than SGD in that gauge "
            "direction story."
        )

    return {
        "overall": overall,
        "gauge_construction_check": gauge_construction_check,
        "primary_tests_passed": primary_tests_passed,
        "primary_tests_total": len(primary_tests),
        "primary_tests": primary_tests,
        "legacy_tests_passed": legacy_tests_passed,
        "legacy_tests_total": len(legacy_tests),
        "legacy_tests": legacy_tests,
        "study_scope": (
            "Finite-time perturbation sensitivity study in raw weight space (lambda_W), with additional "
            "effective-map (lambda_F) and mean per-layer (lambda_L) diagnostics."
        ),
        "scope_caveat": (
            "Only the primary gauge condition is guaranteed to preserve the effective map at t=0. The "
            "skew and independent symmetric conditions remain controls/comparators rather than physical "
            "quotient-space perturbations."
        ),
        "primary_shortform": {
            "lambda_gauge_SGD": float(lambda_gauge_sgd),
            "lambda_gauge_Muon": float(lambda_gauge_muon),
            "lambda_phys_SGD": float(lambda_phys_sgd),
            "lambda_phys_Muon": float(lambda_phys_muon),
            "welch_t_statistic_sgd_minus_muon": float(primary_stats["t_stat"]),
            "welch_df": float(primary_stats["degrees_freedom"]),
            "heuristic_significant": bool(primary_stats["heuristic_one_tailed_significant"]),
        },
        "legacy_shortform": {
            "lambda_legacy_symmetric_SGD": float(lambda_legacy_sgd),
            "lambda_legacy_symmetric_Muon": float(lambda_legacy_muon),
            "welch_t_statistic_sgd_minus_muon": float(legacy_stats["t_stat"]),
            "welch_df": float(legacy_stats["degrees_freedom"]),
            "heuristic_significant": bool(legacy_stats["heuristic_one_tailed_significant"]),
        },
        "interpretation": interpretation,
    }


def _format_scalar(value: float, digits: int = 6) -> str:
    """Human-readable formatter that handles non-finite values."""
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def print_summary(results: Dict[str, Any]) -> None:
    """Pretty-print the main outputs of the experiment."""
    config = results["config"]
    condition_results = results["condition_results"]
    verdict = results["verdict"]
    stats = results["statistical_comparison"]
    legacy_stats = results["legacy_statistical_comparison"]

    print("=" * 108)
    print("1.2b-i: FINITE-TIME SENSITIVITY TO GAUGE-PRESERVING VS. NON-GAUGE PERTURBATIONS")
    print("=" * 108)
    print(results["study_title"])
    print(f"Scope note: {results['scope_note']}")
    print("-" * 108)
    print(
        f"Config: layers={config['num_layers']}, dim={config['dim']}, steps={config['num_steps']}, "
        f"batch={config['batch_size']}, epsilon={config['epsilon']}, perturbations={config['num_perturbations']}"
    )
    print(
        f"Seeds: global={config['seed_global']}, init={config['seed_init']}, "
        f"perturb_base={config['seed_perturb_base']}"
    )
    print(
        f"Learning rates: SGD={results['learning_rates']['sgd']}, Muon={results['learning_rates']['muon']}"
    )
    print(f"Runtime: {results['runtime_seconds']:.2f}s")
    print("-" * 108)
    print("Perturbation constructions:")
    for key in results["perturbation_order"]:
        meta = results["perturbation_catalog"][key]
        print(f"  {key:<18} true_gauge={meta['is_true_gauge']!s:<5} | {meta['display_label']}")
    print("-" * 108)
    print("Training sanity check:")
    for optimizer_key in ["sgd", "muon"]:
        check = results["training_checks"][optimizer_key]
        print(
            f"  {check['optimizer_name']:<5} lr={check['learning_rate']:<7g} "
            f"initial_loss={check['initial_loss']:.6e} final_loss={check['final_loss']:.6e} "
            f"diverged={check['diverged']}"
        )

    print("\nCondition summary (means over perturbation trials):")
    print(
        f"{'Condition':<24} | {'lambda_W':>10} | {'lambda_F':>10} | {'lambda_L':>10} | "
        f"{'mean init dF':>12} | {'map-pres?':>9}"
    )
    print("-" * 108)
    for key in results["condition_order"]:
        summary = condition_results[key]["summary"]
        diagnostics = condition_results[key]["initial_diagnostics"]
        print(
            f"{key:<24} | {_format_scalar(summary['lambda_W']['mean']):>10} | "
            f"{_format_scalar(summary['lambda_F']['mean']):>10} | {_format_scalar(summary['lambda_L']['mean']):>10} | "
            f"{diagnostics['initial_effective_mismatch_abs']['mean']:.3e} | "
            f"{str(diagnostics['preserves_effective_map_within_tol']):>9}"
        )

    gauge_check = verdict["gauge_construction_check"]
    print("\nGauge-preservation verification:")
    print(
        f"  pass={gauge_check['pass']} | max_abs={gauge_check['max_abs_observed']:.3e} "
        f"(tol {gauge_check['tolerance_abs']:.1e}) | max_rel={gauge_check['max_rel_observed']:.3e} "
        f"(tol {gauge_check['tolerance_rel']:.1e})"
    )

    print("\nWelch comparison on primary gauge lambda_W samples (SGD minus Muon):")
    print(
        f"  t={stats['t_stat']:.4f}, df={stats['degrees_freedom']:.2f}, "
        f"heuristic_one_tailed_significant={stats['heuristic_one_tailed_significant']}"
    )
    print("Welch comparison on retained legacy symmetric lambda_W samples (SGD minus Muon):")
    print(
        f"  t={legacy_stats['t_stat']:.4f}, df={legacy_stats['degrees_freedom']:.2f}, "
        f"heuristic_one_tailed_significant={legacy_stats['heuristic_one_tailed_significant']}"
    )

    print("\nPrimary verdict tests:")
    for test_name, test in verdict["primary_tests"].items():
        print(f"  [{'PASS' if test['pass'] else 'FAIL'}] {test_name}: {test['description']}")
    print(
        f"  Overall: {verdict['overall']} "
        f"({verdict['primary_tests_passed']}/{verdict['primary_tests_total']} primary tests passed)"
    )
    print(f"  Interpretation: {verdict['interpretation']}")

    if results["plot_paths"]:
        print("\nSaved figures:")
        for name, path in results["plot_paths"].items():
            print(f"  {name}: {path}")
    print("=" * 108)


def make_plots(results: Dict[str, Any], output_dir: Optional[Path] = None, verbose: bool = True) -> Dict[str, str]:
    """Generate summary figures for the experiment."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        if verbose:
            print("matplotlib not available; skipping figure generation.")
        return {}

    output_path = Path(output_dir or results["paths"]["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)
    config = results["config"]
    condition_results = results["condition_results"]
    perturbation_order = results["perturbation_order"]
    time_axis = np.arange(config["num_steps"] + 1)
    floor = 1e-18

    def mean_curve(condition_key: str, field: str) -> np.ndarray:
        curves = [np.asarray(curve, dtype=float) for curve in condition_results[condition_key][field]]
        return np.mean(np.stack(curves, axis=0), axis=0)

    def positive_curve(curve: np.ndarray) -> np.ndarray:
        arr = np.asarray(curve, dtype=float)
        return np.maximum(arr, floor)

    def metric_label(value: float) -> str:
        return f"{value:.4f}" if np.isfinite(value) else "n/a"

    color_map = {"SGD": "#4477AA", "Muon": "#CC3311"}

    fig, axes = plt.subplots(2, len(perturbation_order), figsize=(6 * len(perturbation_order), 10))
    fig.suptitle(
        "1.2b-i finite-time sensitivity study\n"
        "Primary true gauge-preserving condition plus non-gauge controls/comparators",
        fontsize=14,
        fontweight="bold",
    )

    for col_idx, perturbation_type in enumerate(perturbation_order):
        sgd_key, muon_key = results["condition_groups"][perturbation_type]
        meta = results["perturbation_catalog"][perturbation_type]

        top_axis = axes[0, col_idx]
        top_axis.set_title(f"{meta['short_label']}: d_W(t)")
        for trial_curve in condition_results[sgd_key]["distances_W"]:
            top_axis.semilogy(time_axis[: len(trial_curve)], positive_curve(trial_curve), color=color_map["SGD"], alpha=0.12, linewidth=0.6)
        for trial_curve in condition_results[muon_key]["distances_W"]:
            top_axis.semilogy(time_axis[: len(trial_curve)], positive_curve(trial_curve), color=color_map["Muon"], alpha=0.12, linewidth=0.6)
        sgd_mean = mean_curve(sgd_key, "distances_W")
        muon_mean = mean_curve(muon_key, "distances_W")
        top_axis.semilogy(
            time_axis[: len(sgd_mean)],
            positive_curve(sgd_mean),
            color=color_map["SGD"],
            linewidth=2.4,
            label=f"SGD (lambda_W={metric_label(condition_results[sgd_key]['summary']['lambda_W']['mean'])})",
        )
        top_axis.semilogy(
            time_axis[: len(muon_mean)],
            positive_curve(muon_mean),
            color=color_map["Muon"],
            linewidth=2.4,
            label=f"Muon (lambda_W={metric_label(condition_results[muon_key]['summary']['lambda_W']['mean'])})",
        )
        top_axis.set_xlabel("Step")
        top_axis.set_ylabel("d_W(t)")
        top_axis.grid(True, alpha=0.3)
        top_axis.legend(fontsize=9)

        bottom_axis = axes[1, col_idx]
        bottom_axis.set_title(f"{meta['short_label']}: d_F(t)")
        for trial_curve in condition_results[sgd_key]["distances_F"]:
            bottom_axis.semilogy(time_axis[: len(trial_curve)], positive_curve(trial_curve), color=color_map["SGD"], alpha=0.12, linewidth=0.6)
        for trial_curve in condition_results[muon_key]["distances_F"]:
            bottom_axis.semilogy(time_axis[: len(trial_curve)], positive_curve(trial_curve), color=color_map["Muon"], alpha=0.12, linewidth=0.6)
        sgd_mean_f = mean_curve(sgd_key, "distances_F")
        muon_mean_f = mean_curve(muon_key, "distances_F")
        bottom_axis.semilogy(
            time_axis[: len(sgd_mean_f)],
            positive_curve(sgd_mean_f),
            color=color_map["SGD"],
            linewidth=2.4,
            label=f"SGD (lambda_F={metric_label(condition_results[sgd_key]['summary']['lambda_F']['mean'])})",
        )
        bottom_axis.semilogy(
            time_axis[: len(muon_mean_f)],
            positive_curve(muon_mean_f),
            color=color_map["Muon"],
            linewidth=2.4,
            label=f"Muon (lambda_F={metric_label(condition_results[muon_key]['summary']['lambda_F']['mean'])})",
        )
        if perturbation_type == "gauge":
            bottom_axis.text(
                0.04,
                0.04,
                "d_F(0) ≈ 0 by construction",
                transform=bottom_axis.transAxes,
                fontsize=9,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )
        bottom_axis.set_xlabel("Step")
        bottom_axis.set_ylabel("d_F(t)")
        bottom_axis.grid(True, alpha=0.3)
        bottom_axis.legend(fontsize=9)

    plt.tight_layout()
    plot1 = output_path / "lyapunov_gauge_direction.png"
    fig.savefig(plot1, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(perturbation_order), figsize=(6 * len(perturbation_order), 5))
    fig.suptitle(
        "Finite-time log-ratio view in weight space\n"
        "Dashed reference slopes correspond to the lambda_W summary metric",
        fontsize=13,
        fontweight="bold",
    )

    for axis, perturbation_type in zip(np.atleast_1d(axes), perturbation_order):
        axis.set_title(results["perturbation_catalog"][perturbation_type]["short_label"])
        for optimizer_name, color in [("SGD", color_map["SGD"]), ("Muon", color_map["Muon"] )]:
            key = f"{optimizer_name}_{perturbation_type}"
            dist_list = condition_results[key]["distances_W"]
            for dist in dist_list:
                d0 = max(float(dist[0]), floor)
                axis.plot(time_axis[: len(dist)], np.log(positive_curve(dist) / d0), color=color, alpha=0.1, linewidth=0.5)
            mean_dist = mean_curve(key, "distances_W")
            d0_mean = max(float(mean_dist[0]), floor)
            axis.plot(
                time_axis[: len(mean_dist)],
                np.log(positive_curve(mean_dist) / d0_mean),
                color=color,
                linewidth=2.4,
                label=f"{optimizer_name} (lambda_W={metric_label(condition_results[key]['summary']['lambda_W']['mean'])})",
            )
            axis.plot(
                time_axis,
                condition_results[key]["summary"]["lambda_W"]["mean"] * time_axis,
                color=color,
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
            )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xlabel("Step")
        axis.set_ylabel("log(d_W(t) / d_W(0))")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=9)

    plt.tight_layout()
    plot2 = output_path / "lyapunov_log_ratio.png"
    fig.savefig(plot2, dpi=150, bbox_inches="tight")
    plt.close(fig)

    plot_paths = {
        "summary_panel": str(plot1),
        "log_ratio_panel": str(plot2),
    }
    if verbose:
        for name, path in plot_paths.items():
            print(f"Saved {name}: {path}")
    return plot_paths


def run_experiment(
    config_overrides: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None,
    generate_plots: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the full experiment and return structured results for scripts or notebooks."""
    start = time.time()
    config = merge_config(config_overrides)
    script_path = Path(__file__).resolve()
    resolved_output_dir = Path(output_dir) if output_dir is not None else script_path.parent

    problem = build_problem(config)
    lr_sgd, sgd_lr_search = find_stable_lr_sgd(config, problem)
    training_checks = run_training_checks(config, problem, lr_sgd)

    perturbation_order = ["gauge", "physical", "symmetric_legacy"]
    condition_order = [f"{optimizer_name}_{perturbation_type}" for perturbation_type in perturbation_order for optimizer_name in ("SGD", "Muon")]
    condition_groups = {
        perturbation_type: [f"SGD_{perturbation_type}", f"Muon_{perturbation_type}"] for perturbation_type in perturbation_order
    }

    base_runs: Dict[str, Dict[str, Any]] = {}
    for optimizer_key, lr in [("sgd", lr_sgd), ("muon", config["lr_muon"] )]:
        base_runs[optimizer_key] = build_base_run(optimizer_key, lr, config, problem)

    condition_results: Dict[str, Dict[str, Any]] = {}
    for optimizer_name, optimizer_key, lr in [("SGD", "sgd", lr_sgd), ("Muon", "muon", config["lr_muon"] )]:
        weights_base = base_runs[optimizer_key]["weights_base"]
        base_run = base_runs[optimizer_key]["base_run"]
        for perturbation_type in perturbation_order:
            condition_key = f"{optimizer_name}_{perturbation_type}"
            condition_results[condition_key] = measure_condition(
                optimizer_key=optimizer_key,
                lr=lr,
                perturbation_type=perturbation_type,
                config=config,
                problem=problem,
                weights_base=weights_base,
                base_run=base_run,
            )

    primary_stats = welch_t_test(
        condition_results["SGD_gauge"]["lambda_W_all"],
        condition_results["Muon_gauge"]["lambda_W_all"],
    )
    legacy_stats = welch_t_test(
        condition_results["SGD_symmetric_legacy"]["lambda_W_all"],
        condition_results["Muon_symmetric_legacy"]["lambda_W_all"],
    )
    verdict = build_verdict(condition_results, primary_stats, legacy_stats)

    results: Dict[str, Any] = {
        "experiment_id": "1.2b-i",
        "study_title": "Finite-time sensitivity study for a true gauge-preserving perturbation plus non-gauge controls",
        "scope_note": (
            "The primary gauge condition now preserves the deep-linear effective map at t=0 by construction. "
            "The skew and independent symmetric conditions remain control/comparison perturbations rather than "
            "physical quotient-space directions."
        ),
        "config": config,
        "environment": {
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
        },
        "paths": {
            "script_path": str(script_path),
            "output_dir": str(resolved_output_dir),
        },
        "learning_rates": {"sgd": float(lr_sgd), "muon": float(config["lr_muon"])},
        "sgd_lr_search": sgd_lr_search,
        "training_checks": training_checks,
        "perturbation_order": perturbation_order,
        "perturbation_catalog": get_perturbation_catalog(),
        "condition_order": condition_order,
        "condition_groups": condition_groups,
        "condition_results": condition_results,
        "statistical_comparison": primary_stats,
        "legacy_statistical_comparison": legacy_stats,
        "verdict": verdict,
        "runtime_seconds": float(time.time() - start),
        "plot_paths": {},
    }

    if generate_plots:
        results["plot_paths"] = make_plots(results, output_dir=resolved_output_dir, verbose=verbose)

    if verbose:
        print_summary(results)
    return results


def main() -> Dict[str, Any]:
    """CLI entrypoint preserving normal script behavior."""
    return run_experiment(generate_plots=True, verbose=True)


if __name__ == "__main__":
    main()
