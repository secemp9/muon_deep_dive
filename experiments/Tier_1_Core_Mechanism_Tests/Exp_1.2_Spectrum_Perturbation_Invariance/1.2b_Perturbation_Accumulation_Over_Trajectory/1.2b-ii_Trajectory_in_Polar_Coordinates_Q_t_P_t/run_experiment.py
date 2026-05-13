#!/usr/bin/env python3
"""
1.2b-ii: Polar-factor trajectory proxy analysis in (Q_t, P_t) coordinates
============================================================================

This experiment tracks the *per-layer* polar decomposition W_t = Q_t P_t during
training of a deep linear network. For each layer and optimization step it
records:

  - ||ΔQ|| / ||ΔW|| : change in the orthogonal polar factor relative to the
    weight-step norm
  - ||ΔP|| / ||ΔW|| : change in the symmetric positive-semidefinite polar factor
    relative to the weight-step norm
  - cumulative drifts ||Q_t - Q_0||, ||P_t - P_0||, ||W_t - W_0||

Important scope limitations:
  - These are per-layer polar-factor diagnostics for a toy deep linear system.
    They are *not* direct coordinates on the full deep-linear gauge orbit.
  - ||ΔQ|| / ||ΔW|| and ||ΔP|| / ||ΔW|| are descriptive ratios, not additive
    shares of a decomposition of ΔW. They can exceed 1.
  - Raw means of these ratios can become noisy when ||ΔW|| is very small, so the
    script also reports aggregate ratios Σ||ΔQ|| / Σ||ΔW|| and
    Σ||ΔP|| / Σ||ΔW|| as more stable summaries.
  - This default run is a single-seed, fixed-batch study.

Default setup is preserved from the prior version:
  4-layer deep linear net, 32x32, quadratic loss, 300 steps.
"""

from __future__ import annotations

import copy
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_CONFIG: Dict[str, Any] = {
    "SEED": 42,
    "DIM": 32,
    "NUM_LAYERS": 4,
    "NUM_STEPS": 300,
    "BATCH_SIZE": 64,
    "LR_MUON": 0.005,
    "MOMENTUM": 0.9,
    "NS_ITERS": 5,
    "REPORT_STEPS": [50, 100, 200, 300],
    "WINDOW_RADIUS": 5,
    "LR_CANDIDATES_SGD": [0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.001],
    "STABILITY_STEPS": 100,
    "STABILITY_LOSS_MULTIPLIER": 50.0,
    "TARGET_SCALE": 0.5,
    "INPUT_SCALE": 0.3,
    "RATIO_DENOM_EPS": 1e-15,
    "SMALL_DW_EPS": 1e-12,
}


# =============================================================================
# CONFIG / PROBLEM SETUP
# =============================================================================

def get_default_config() -> Dict[str, Any]:
    """Return a deep copy of the default experiment configuration."""
    return copy.deepcopy(DEFAULT_CONFIG)


def prepare_config(config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Merge user overrides into the default config."""
    config = get_default_config()
    if config_overrides:
        for key, value in config_overrides.items():
            config[key] = copy.deepcopy(value)
    return config


def build_problem(config: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Construct the fixed target matrix and fixed input batch."""
    rng = np.random.RandomState(config["SEED"])
    dim = config["DIM"]
    batch_size = config["BATCH_SIZE"]
    target = rng.randn(dim, dim) * config["TARGET_SCALE"]
    x_data = rng.randn(dim, batch_size) * config["INPUT_SCALE"]
    return target, x_data


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def init_weights(num_layers: int, dim: int, seed: int = 42) -> List[np.ndarray]:
    """Initialize layers near identity for stability."""
    rng = np.random.RandomState(seed)
    weights = []
    for _ in range(num_layers):
        w = np.eye(dim) + rng.randn(dim, dim) * 0.1
        weights.append(w.copy())
    return weights


def forward(weights: List[np.ndarray], x: np.ndarray) -> np.ndarray:
    """Forward pass: W_L @ ... @ W_1 @ X."""
    out = x.copy()
    for w in weights:
        out = w @ out
    return out


def compute_loss(weights: List[np.ndarray], x: np.ndarray, target: np.ndarray) -> float:
    """Loss = 0.5 * ||W_product @ X - T @ X||^2 / N."""
    pred = forward(weights, x)
    target_out = target @ x
    diff = pred - target_out
    return float(0.5 * np.mean(np.sum(diff ** 2, axis=0)))


def compute_gradients(weights: List[np.ndarray], x: np.ndarray, target: np.ndarray) -> List[np.ndarray]:
    """Backprop through a deep linear network."""
    num_layers = len(weights)
    n = x.shape[1]

    activations = [x.copy()]
    out = x.copy()
    for w in weights:
        out = w @ out
        activations.append(out.copy())

    target_out = target @ x
    delta = (activations[-1] - target_out) / n

    grads: List[np.ndarray] = []
    for i in range(num_layers - 1, -1, -1):
        grad = delta @ activations[i].T
        grads.insert(0, grad)
        if i > 0:
            delta = weights[i].T @ delta

    return grads


def newton_schulz_orthogonalize(g: np.ndarray, num_iters: int) -> np.ndarray:
    """
    Approximate the orthogonal polar factor of g via Newton-Schulz iteration.
    """
    norm = np.linalg.norm(g, ord="fro")
    if norm < 1e-12:
        return g

    x = g / norm
    for _ in range(num_iters):
        a = x.T @ x
        x = 1.5 * x - 0.5 * x @ a
    return x


def polar_decomposition(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute W = QP using an SVD-based polar decomposition."""
    u, s, vt = np.linalg.svd(w, full_matrices=True)
    q = u @ vt
    p = vt.T @ np.diag(s) @ vt
    return q, p


# =============================================================================
# OPTIMIZER HELPERS
# =============================================================================

def find_stable_lr_sgd(config: Dict[str, Any], x_data: np.ndarray, target: np.ndarray) -> float:
    """Find the maximum stable SGD learning rate among predefined candidates."""
    dim = config["DIM"]
    num_layers = config["NUM_LAYERS"]
    seed = config["SEED"]
    momentum = config["MOMENTUM"]
    stability_steps = config["STABILITY_STEPS"]
    loss_multiplier = config["STABILITY_LOSS_MULTIPLIER"]

    for lr in config["LR_CANDIDATES_SGD"]:
        weights = init_weights(num_layers, dim, seed=seed)
        velocities = [np.zeros((dim, dim)) for _ in range(num_layers)]
        initial_loss = compute_loss(weights, x_data, target)
        stable = True
        for _ in range(stability_steps):
            grads = compute_gradients(weights, x_data, target)
            for i in range(num_layers):
                velocities[i] = momentum * velocities[i] + grads[i]
                weights[i] -= lr * velocities[i]
            loss = compute_loss(weights, x_data, target)
            if np.isnan(loss) or loss > initial_loss * loss_multiplier:
                stable = False
                break
        if stable:
            return float(lr)

    return float(config["LR_CANDIDATES_SGD"][-1])


def sgd_step(
    weights: List[np.ndarray],
    velocities: List[np.ndarray],
    lr: float,
    x_data: np.ndarray,
    target: np.ndarray,
    momentum: float,
) -> tuple[List[np.ndarray], List[np.ndarray]]:
    """One step of SGD with momentum."""
    grads = compute_gradients(weights, x_data, target)
    for i in range(len(weights)):
        velocities[i] = momentum * velocities[i] + grads[i]
        weights[i] = weights[i] - lr * velocities[i]
    return weights, velocities


def muon_step(
    weights: List[np.ndarray],
    velocities: List[np.ndarray],
    lr: float,
    x_data: np.ndarray,
    target: np.ndarray,
    momentum: float,
    ns_iters: int,
) -> tuple[List[np.ndarray], List[np.ndarray]]:
    """One step of Muon with momentum using orthogonalized gradients."""
    grads = compute_gradients(weights, x_data, target)
    for i in range(len(weights)):
        ortho_grad = newton_schulz_orthogonalize(grads[i], num_iters=ns_iters)
        velocities[i] = momentum * velocities[i] + ortho_grad
        weights[i] = weights[i] - lr * velocities[i]
    return weights, velocities


# =============================================================================
# TRACKING ENGINE
# =============================================================================

def run_and_track_polar(
    optimizer: str,
    lr: float,
    num_steps: int,
    config: Dict[str, Any],
    x_data: np.ndarray,
    target: np.ndarray,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run an optimizer for num_steps and track layerwise polar-factor diagnostics.
    """
    dim = config["DIM"]
    num_layers = config["NUM_LAYERS"]
    seed = config["SEED"]
    momentum = config["MOMENTUM"]
    ns_iters = config["NS_ITERS"]
    ratio_eps = config["RATIO_DENOM_EPS"]

    weights = init_weights(num_layers, dim, seed=seed)
    velocities = [np.zeros_like(w) for w in weights]

    q_prev: List[np.ndarray] = []
    p_prev: List[np.ndarray] = []
    q_init: List[np.ndarray] = []
    p_init: List[np.ndarray] = []
    for w in weights:
        q, p = polar_decomposition(w)
        q_prev.append(q.copy())
        p_prev.append(p.copy())
        q_init.append(q.copy())
        p_init.append(p.copy())

    dQ_ratio = np.zeros((num_layers, num_steps))
    dP_ratio = np.zeros((num_layers, num_steps))
    dQ_norm = np.zeros((num_layers, num_steps))
    dP_norm = np.zeros((num_layers, num_steps))
    dW_norm = np.zeros((num_layers, num_steps))
    cum_Q_drift = np.zeros((num_layers, num_steps))
    cum_P_drift = np.zeros((num_layers, num_steps))
    cum_W_drift = np.zeros((num_layers, num_steps))

    losses = np.zeros(num_steps + 1)
    losses[0] = compute_loss(weights, x_data, target)

    w_prev = [w.copy() for w in weights]
    w_init = [w.copy() for w in weights]
    diverged_at: Optional[int] = None

    for step in range(num_steps):
        if optimizer == "sgd":
            weights, velocities = sgd_step(weights, velocities, lr, x_data, target, momentum)
        elif optimizer == "muon":
            weights, velocities = muon_step(weights, velocities, lr, x_data, target, momentum, ns_iters)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        losses[step + 1] = compute_loss(weights, x_data, target)

        if np.isnan(losses[step + 1]) or losses[step + 1] > 1e10:
            diverged_at = step + 1
            if verbose:
                print(f"    WARNING: {optimizer} diverged at step {diverged_at}")
            dQ_ratio[:, step:] = np.nan
            dP_ratio[:, step:] = np.nan
            dQ_norm[:, step:] = np.nan
            dP_norm[:, step:] = np.nan
            dW_norm[:, step:] = np.nan
            cum_Q_drift[:, step:] = np.nan
            cum_P_drift[:, step:] = np.nan
            cum_W_drift[:, step:] = np.nan
            losses[step + 1 :] = np.nan
            break

        for i in range(num_layers):
            q_curr, p_curr = polar_decomposition(weights[i])

            delta_w = weights[i] - w_prev[i]
            delta_q = q_curr - q_prev[i]
            delta_p = p_curr - p_prev[i]

            norm_dW = np.linalg.norm(delta_w, ord="fro")
            norm_dQ = np.linalg.norm(delta_q, ord="fro")
            norm_dP = np.linalg.norm(delta_p, ord="fro")

            dW_norm[i, step] = norm_dW
            dQ_norm[i, step] = norm_dQ
            dP_norm[i, step] = norm_dP

            if norm_dW > ratio_eps:
                dQ_ratio[i, step] = norm_dQ / norm_dW
                dP_ratio[i, step] = norm_dP / norm_dW
            else:
                dQ_ratio[i, step] = np.nan
                dP_ratio[i, step] = np.nan

            cum_Q_drift[i, step] = np.linalg.norm(q_curr - q_init[i], ord="fro")
            cum_P_drift[i, step] = np.linalg.norm(p_curr - p_init[i], ord="fro")
            cum_W_drift[i, step] = np.linalg.norm(weights[i] - w_init[i], ord="fro")

            q_prev[i] = q_curr.copy()
            p_prev[i] = p_curr.copy()

        w_prev = [w.copy() for w in weights]

    return {
        "optimizer": optimizer,
        "lr": float(lr),
        "diverged_at": diverged_at,
        "dQ_ratio": dQ_ratio,
        "dP_ratio": dP_ratio,
        "dQ_norm": dQ_norm,
        "dP_norm": dP_norm,
        "dW_norm": dW_norm,
        "cum_Q_drift": cum_Q_drift,
        "cum_P_drift": cum_P_drift,
        "cum_W_drift": cum_W_drift,
        "losses": losses,
    }


# =============================================================================
# SUMMARY / ANALYSIS HELPERS
# =============================================================================

def safe_ratio(numerator: Any, denominator: Any, eps: float = 1e-15) -> Any:
    """Safe division that returns NaN when the denominator is too small."""
    numer_arr = np.asarray(numerator, dtype=float)
    denom_arr = np.asarray(denominator, dtype=float)
    numer_b, denom_b = np.broadcast_arrays(numer_arr, denom_arr)
    out = np.full(numer_b.shape, np.nan, dtype=float)
    mask = np.abs(denom_b) > eps
    out[mask] = numer_b[mask] / denom_b[mask]
    if out.shape == ():
        return float(out)
    return out


def nan_stats(array: np.ndarray) -> Dict[str, Any]:
    """Basic finite-value summary stats for a NumPy array."""
    flat = np.asarray(array, dtype=float).ravel()
    flat = flat[~np.isnan(flat)]
    if flat.size == 0:
        return {
            "count": 0,
            "min": np.nan,
            "median": np.nan,
            "mean": np.nan,
            "max": np.nan,
        }
    return {
        "count": int(flat.size),
        "min": float(np.min(flat)),
        "median": float(np.median(flat)),
        "mean": float(np.mean(flat)),
        "max": float(np.max(flat)),
    }


def report_window_bounds(step: int, num_steps: int, radius: int) -> tuple[int, int]:
    """10-step style window centered on a 1-indexed report step, matching prior logic."""
    lo = max(0, step - radius)
    hi = min(num_steps, step + radius)
    return lo, hi


def summarize_single_optimizer(results: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Build stable summaries and diagnostics for one optimizer run."""
    ratio_eps = config["RATIO_DENOM_EPS"]
    small_dw_eps = config["SMALL_DW_EPS"]

    dQ_ratio = results["dQ_ratio"]
    dP_ratio = results["dP_ratio"]
    dQ_norm = results["dQ_norm"]
    dP_norm = results["dP_norm"]
    dW_norm = results["dW_norm"]
    cum_Q_drift = results["cum_Q_drift"]
    cum_P_drift = results["cum_P_drift"]
    cum_W_drift = results["cum_W_drift"]
    losses = results["losses"]

    total_dQ = float(np.nansum(dQ_norm))
    total_dP = float(np.nansum(dP_norm))
    total_dW = float(np.nansum(dW_norm))

    step_dQ = np.nansum(dQ_norm, axis=0)
    step_dP = np.nansum(dP_norm, axis=0)
    step_dW = np.nansum(dW_norm, axis=0)

    summary = {
        "aggregate_dQ_over_dW": safe_ratio(total_dQ, total_dW, eps=ratio_eps),
        "aggregate_dP_over_dW": safe_ratio(total_dP, total_dW, eps=ratio_eps),
        "raw_mean_dQ_ratio": float(np.nanmean(dQ_ratio)),
        "raw_mean_dP_ratio": float(np.nanmean(dP_ratio)),
        "stepwise_aggregate_dQ_over_dW": safe_ratio(step_dQ, step_dW, eps=ratio_eps),
        "stepwise_aggregate_dP_over_dW": safe_ratio(step_dP, step_dW, eps=ratio_eps),
        "stepwise_raw_mean_dQ_ratio": np.nanmean(dQ_ratio, axis=0),
        "stepwise_raw_mean_dP_ratio": np.nanmean(dP_ratio, axis=0),
        "mean_cum_Q_final": float(np.nanmean(cum_Q_drift[:, -1])),
        "mean_cum_P_final": float(np.nanmean(cum_P_drift[:, -1])),
        "mean_cum_W_final": float(np.nanmean(cum_W_drift[:, -1])),
        "mean_cum_Q_fraction_final": safe_ratio(
            np.nanmean(cum_Q_drift[:, -1]),
            np.nanmean(cum_Q_drift[:, -1]) + np.nanmean(cum_P_drift[:, -1]),
            eps=ratio_eps,
        ),
        "mean_cum_Q_over_P_final": safe_ratio(
            np.nanmean(cum_Q_drift[:, -1]),
            np.nanmean(cum_P_drift[:, -1]),
            eps=ratio_eps,
        ),
        "final_loss": float(losses[-1]),
        "dW_stats": nan_stats(dW_norm),
        "dQ_ratio_stats": nan_stats(dQ_ratio),
        "dP_ratio_stats": nan_stats(dP_ratio),
        "small_dW_count": int(np.sum((dW_norm < small_dw_eps) & ~np.isnan(dW_norm))),
        "small_dW_fraction": float(
            np.sum((dW_norm < small_dw_eps) & ~np.isnan(dW_norm))
            / max(1, np.sum(~np.isnan(dW_norm)))
        ),
        "valid_ratio_count": int(np.sum(~np.isnan(dQ_ratio))),
        "total_dQ_norm": total_dQ,
        "total_dP_norm": total_dP,
        "total_dW_norm": total_dW,
    }
    return summary


def build_windowed_ratio_rows(
    results_sgd: Dict[str, Any],
    results_muon: Dict[str, Any],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Per-report-step summaries for robust aggregate ratios and legacy raw means."""
    rows: List[Dict[str, Any]] = []
    num_steps = config["NUM_STEPS"]
    radius = config["WINDOW_RADIUS"]
    eps = config["RATIO_DENOM_EPS"]

    for step in config["REPORT_STEPS"]:
        lo, hi = report_window_bounds(step, num_steps=num_steps, radius=radius)
        rows.append(
            {
                "step": int(step),
                "window": [int(lo + 1), int(hi)],
                "sgd_aggregate_dQ_over_dW": safe_ratio(
                    np.nansum(results_sgd["dQ_norm"][:, lo:hi]),
                    np.nansum(results_sgd["dW_norm"][:, lo:hi]),
                    eps=eps,
                ),
                "sgd_aggregate_dP_over_dW": safe_ratio(
                    np.nansum(results_sgd["dP_norm"][:, lo:hi]),
                    np.nansum(results_sgd["dW_norm"][:, lo:hi]),
                    eps=eps,
                ),
                "muon_aggregate_dQ_over_dW": safe_ratio(
                    np.nansum(results_muon["dQ_norm"][:, lo:hi]),
                    np.nansum(results_muon["dW_norm"][:, lo:hi]),
                    eps=eps,
                ),
                "muon_aggregate_dP_over_dW": safe_ratio(
                    np.nansum(results_muon["dP_norm"][:, lo:hi]),
                    np.nansum(results_muon["dW_norm"][:, lo:hi]),
                    eps=eps,
                ),
                "sgd_raw_mean_dQ_ratio": float(np.nanmean(results_sgd["dQ_ratio"][:, lo:hi])),
                "sgd_raw_mean_dP_ratio": float(np.nanmean(results_sgd["dP_ratio"][:, lo:hi])),
                "muon_raw_mean_dQ_ratio": float(np.nanmean(results_muon["dQ_ratio"][:, lo:hi])),
                "muon_raw_mean_dP_ratio": float(np.nanmean(results_muon["dP_ratio"][:, lo:hi])),
            }
        )
    return rows


def build_cumulative_drift_rows(
    results_sgd: Dict[str, Any],
    results_muon: Dict[str, Any],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Per-report-step cumulative drift summaries."""
    rows: List[Dict[str, Any]] = []
    eps = config["RATIO_DENOM_EPS"]

    for step in config["REPORT_STEPS"]:
        idx = step - 1
        sgd_q = float(np.nanmean(results_sgd["cum_Q_drift"][:, idx]))
        sgd_p = float(np.nanmean(results_sgd["cum_P_drift"][:, idx]))
        sgd_w = float(np.nanmean(results_sgd["cum_W_drift"][:, idx]))
        muon_q = float(np.nanmean(results_muon["cum_Q_drift"][:, idx]))
        muon_p = float(np.nanmean(results_muon["cum_P_drift"][:, idx]))
        muon_w = float(np.nanmean(results_muon["cum_W_drift"][:, idx]))
        rows.append(
            {
                "step": int(step),
                "sgd_cum_Q": sgd_q,
                "sgd_cum_P": sgd_p,
                "sgd_cum_W": sgd_w,
                "muon_cum_Q": muon_q,
                "muon_cum_P": muon_p,
                "muon_cum_W": muon_w,
                "sgd_Q_over_P": safe_ratio(sgd_q, sgd_p, eps=eps),
                "muon_Q_over_P": safe_ratio(muon_q, muon_p, eps=eps),
                "sgd_Q_fraction": safe_ratio(sgd_q, sgd_q + sgd_p, eps=eps),
                "muon_Q_fraction": safe_ratio(muon_q, muon_q + muon_p, eps=eps),
            }
        )
    return rows


def build_per_layer_rows(results: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per-layer final summaries for one optimizer."""
    rows: List[Dict[str, Any]] = []
    eps = config["RATIO_DENOM_EPS"]
    num_layers = config["NUM_LAYERS"]

    for layer in range(num_layers):
        total_dW = float(np.nansum(results["dW_norm"][layer, :]))
        total_dQ = float(np.nansum(results["dQ_norm"][layer, :]))
        total_dP = float(np.nansum(results["dP_norm"][layer, :]))
        q_final = float(results["cum_Q_drift"][layer, -1])
        p_final = float(results["cum_P_drift"][layer, -1])
        rows.append(
            {
                "layer": int(layer),
                "aggregate_dQ_over_dW": safe_ratio(total_dQ, total_dW, eps=eps),
                "aggregate_dP_over_dW": safe_ratio(total_dP, total_dW, eps=eps),
                "raw_mean_dQ_ratio": float(np.nanmean(results["dQ_ratio"][layer, :])),
                "raw_mean_dP_ratio": float(np.nanmean(results["dP_ratio"][layer, :])),
                "cum_Q_final": q_final,
                "cum_P_final": p_final,
                "Q_fraction_final": safe_ratio(q_final, q_final + p_final, eps=eps),
            }
        )
    return rows


def build_phase_rows(
    results_sgd: Dict[str, Any],
    results_muon: Dict[str, Any],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Early vs late comparisons for robust and raw summaries."""
    eps = config["RATIO_DENOM_EPS"]
    phase_specs = {
        "early": slice(0, 50),
        "late": slice(max(0, config["NUM_STEPS"] - 50), config["NUM_STEPS"]),
    }
    rows: List[Dict[str, Any]] = []
    for name, sl in phase_specs.items():
        rows.append(
            {
                "phase": name,
                "sgd_aggregate_dQ_over_dW": safe_ratio(
                    np.nansum(results_sgd["dQ_norm"][:, sl]),
                    np.nansum(results_sgd["dW_norm"][:, sl]),
                    eps=eps,
                ),
                "sgd_aggregate_dP_over_dW": safe_ratio(
                    np.nansum(results_sgd["dP_norm"][:, sl]),
                    np.nansum(results_sgd["dW_norm"][:, sl]),
                    eps=eps,
                ),
                "muon_aggregate_dQ_over_dW": safe_ratio(
                    np.nansum(results_muon["dQ_norm"][:, sl]),
                    np.nansum(results_muon["dW_norm"][:, sl]),
                    eps=eps,
                ),
                "muon_aggregate_dP_over_dW": safe_ratio(
                    np.nansum(results_muon["dP_norm"][:, sl]),
                    np.nansum(results_muon["dW_norm"][:, sl]),
                    eps=eps,
                ),
                "sgd_raw_mean_dQ_ratio": float(np.nanmean(results_sgd["dQ_ratio"][:, sl])),
                "sgd_raw_mean_dP_ratio": float(np.nanmean(results_sgd["dP_ratio"][:, sl])),
                "muon_raw_mean_dQ_ratio": float(np.nanmean(results_muon["dQ_ratio"][:, sl])),
                "muon_raw_mean_dP_ratio": float(np.nanmean(results_muon["dP_ratio"][:, sl])),
            }
        )
    return rows


def evaluate_assessment(
    summary_sgd: Dict[str, Any],
    summary_muon: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Return continuity tests plus a more calibrated single-run conclusion."""
    num_steps = config["NUM_STEPS"]

    legacy_tests = {
        "T1": {
            "description": "Muon raw mean ||dQ||/||dW|| > SGD raw mean ||dQ||/||dW||",
            "muon": summary_muon["raw_mean_dQ_ratio"],
            "sgd": summary_sgd["raw_mean_dQ_ratio"],
            "passed": bool(summary_muon["raw_mean_dQ_ratio"] > summary_sgd["raw_mean_dQ_ratio"]),
        },
        "T2": {
            "description": "Muon raw mean ||dP||/||dW|| < SGD raw mean ||dP||/||dW||",
            "muon": summary_muon["raw_mean_dP_ratio"],
            "sgd": summary_sgd["raw_mean_dP_ratio"],
            "passed": bool(summary_muon["raw_mean_dP_ratio"] < summary_sgd["raw_mean_dP_ratio"]),
        },
        "T3": {
            "description": f"Muon cumulative Q/(Q+P) > SGD cumulative Q/(Q+P) at step {num_steps}",
            "muon": summary_muon["mean_cum_Q_fraction_final"],
            "sgd": summary_sgd["mean_cum_Q_fraction_final"],
            "passed": bool(summary_muon["mean_cum_Q_fraction_final"] > summary_sgd["mean_cum_Q_fraction_final"]),
        },
        "T4": {
            "description": "Muon cumulative Q/P ratio > SGD cumulative Q/P ratio",
            "muon": summary_muon["mean_cum_Q_over_P_final"],
            "sgd": summary_sgd["mean_cum_Q_over_P_final"],
            "passed": bool(summary_muon["mean_cum_Q_over_P_final"] > summary_sgd["mean_cum_Q_over_P_final"]),
        },
        "T5": {
            "description": f"Muon ||P-P0|| < SGD ||P-P0|| at step {num_steps}",
            "muon": summary_muon["mean_cum_P_final"],
            "sgd": summary_sgd["mean_cum_P_final"],
            "passed": bool(summary_muon["mean_cum_P_final"] < summary_sgd["mean_cum_P_final"]),
        },
    }

    robust_tests = {
        "R1": {
            "description": "Muon aggregate Σ||dQ|| / Σ||dW|| > SGD aggregate Σ||dQ|| / Σ||dW||",
            "muon": summary_muon["aggregate_dQ_over_dW"],
            "sgd": summary_sgd["aggregate_dQ_over_dW"],
            "passed": bool(summary_muon["aggregate_dQ_over_dW"] > summary_sgd["aggregate_dQ_over_dW"]),
        },
        "R2": {
            "description": "Muon aggregate Σ||dP|| / Σ||dW|| < SGD aggregate Σ||dP|| / Σ||dW||",
            "muon": summary_muon["aggregate_dP_over_dW"],
            "sgd": summary_sgd["aggregate_dP_over_dW"],
            "passed": bool(summary_muon["aggregate_dP_over_dW"] < summary_sgd["aggregate_dP_over_dW"]),
        },
    }

    legacy_score = int(sum(test["passed"] for test in legacy_tests.values()))
    similar_profiles = abs(
        summary_muon["mean_cum_Q_fraction_final"] - summary_sgd["mean_cum_Q_fraction_final"]
    ) < 0.05

    if legacy_score >= 4:
        legacy_label = "STRONG SUPPORT"
    elif legacy_score >= 3:
        legacy_label = "MODERATE SUPPORT"
    elif legacy_score >= 2:
        legacy_label = "WEAK SIGNAL"
    else:
        legacy_label = "NO SUPPORT / SURPRISING"
    if similar_profiles:
        legacy_label += " (SIMILAR PROFILES)"

    supported: List[str] = []
    not_supported: List[str] = []

    if robust_tests["R1"]["passed"]:
        supported.append("Muon has a larger aggregate per-step ||ΔQ||/||ΔW|| ratio than SGD.")
    else:
        not_supported.append(
            "Muon does not have a larger aggregate per-step ||ΔQ||/||ΔW|| ratio than SGD."
        )

    if legacy_tests["T1"]["passed"]:
        supported.append("Muon also exceeds SGD under the legacy raw-mean ||ΔQ||/||ΔW|| metric.")
    else:
        not_supported.append(
            "Muon does not exceed SGD under the legacy raw-mean ||ΔQ||/||ΔW|| metric."
        )

    if robust_tests["R2"]["passed"] and legacy_tests["T2"]["passed"]:
        supported.append("Muon has lower per-step P-factor motion than SGD under both aggregate and raw summaries.")
    elif robust_tests["R2"]["passed"]:
        supported.append("Muon has lower aggregate per-step P-factor motion than SGD.")
    else:
        not_supported.append("Muon does not show lower per-step P-factor motion than SGD.")

    if legacy_tests["T3"]["passed"]:
        supported.append("Muon ends with a higher cumulative orientation fraction Q/(Q+P) than SGD.")
    else:
        not_supported.append("Muon does not end with a higher cumulative orientation fraction than SGD.")

    if legacy_tests["T5"]["passed"]:
        supported.append("Muon ends with lower cumulative P drift than SGD.")
    else:
        not_supported.append("Muon does not end with lower cumulative P drift than SGD.")

    if summary_muon["final_loss"] < summary_sgd["final_loss"]:
        supported.append("Muon reaches a lower final loss in this single fixed-batch run.")
    else:
        not_supported.append("Muon does not reach a lower final loss in this single fixed-batch run.")

    q_support = robust_tests["R1"]["passed"] and legacy_tests["T1"]["passed"]
    p_support = robust_tests["R2"]["passed"]
    cumulative_support = legacy_tests["T3"]["passed"] and legacy_tests["T5"]["passed"]

    if q_support and p_support and cumulative_support:
        calibrated_label = "CONSISTENT SINGLE-RUN SUPPORT"
        calibrated_detail = (
            "Within this toy per-layer polar-factor proxy, Muon shows more Q-directed motion, "
            "less P-directed motion, and a higher cumulative orientation fraction than SGD. "
            "This remains single-seed, fixed-batch evidence rather than a direct measurement "
            "of deep-linear gauge coordinates."
        )
    elif p_support and cumulative_support:
        calibrated_label = "MIXED SINGLE-RUN EVIDENCE"
        calibrated_detail = (
            "Muon shows less P-directed motion and a somewhat higher cumulative orientation "
            "fraction than SGD, but the stronger claim of larger per-step Q motion is not "
            "supported across the available summaries."
        )
    elif cumulative_support:
        calibrated_label = "CUMULATIVE-ONLY SIGNAL"
        calibrated_detail = (
            "Differences appear mainly in cumulative drifts, while the per-step evidence is weak "
            "or inconsistent. Interpret this as suggestive rather than strong support."
        )
    else:
        calibrated_label = "NO CLEAR SUPPORT"
        calibrated_detail = (
            "This run does not provide clear support for the expectation that Muon is more "
            "orientation-dominated than SGD in these polar-factor diagnostics."
        )

    return {
        "legacy_tests": legacy_tests,
        "robust_tests": robust_tests,
        "legacy_score": legacy_score,
        "legacy_label": legacy_label,
        "calibrated_conclusion": {
            "label": calibrated_label,
            "detail": calibrated_detail,
            "supported_expectations": supported,
            "unsupported_expectations": not_supported,
            "limitations": [
                "Single seed only.",
                "Fixed input batch only.",
                "Per-layer polar factors are proxy diagnostics, not direct deep-linear gauge coordinates.",
                "Raw mean ratio tests are sensitive to tiny ||ΔW|| denominators.",
            ],
        },
    }


def compute_problem_diagnostics(
    config: Dict[str, Any],
    x_data: np.ndarray,
    target: np.ndarray,
) -> Dict[str, Any]:
    """Diagnostics for the fixed problem instance and initial weights."""
    dim = config["DIM"]
    num_layers = config["NUM_LAYERS"]
    seed = config["SEED"]

    w_test = init_weights(num_layers, dim, seed=seed)
    initial_loss = compute_loss(w_test, x_data, target)
    target_singular_values = np.linalg.svd(target, compute_uv=False)
    target_condition_number = float(target_singular_values[0] / target_singular_values[-1])

    polar_checks = []
    for layer, w in enumerate(w_test):
        q, p = polar_decomposition(w)
        polar_checks.append(
            {
                "layer": int(layer),
                "recon_err": float(np.linalg.norm(w - q @ p, ord="fro") / np.linalg.norm(w, ord="fro")),
                "Q_orth_err": float(np.linalg.norm(q.T @ q - np.eye(dim), ord="fro")),
                "P_sym_err": float(np.linalg.norm(p - p.T, ord="fro")),
                "P_min_eig": float(np.min(np.linalg.eigvalsh(p))),
            }
        )

    return {
        "initial_loss": float(initial_loss),
        "target_singular_values": target_singular_values,
        "target_condition_number": target_condition_number,
        "polar_checks": polar_checks,
    }


# =============================================================================
# PLOTTING
# =============================================================================

def make_overview_plot(
    results_sgd: Dict[str, Any],
    results_muon: Dict[str, Any],
    summary_sgd: Dict[str, Any],
    summary_muon: Dict[str, Any],
    config: Dict[str, Any],
    lr_sgd: float,
    out_dir: Path,
) -> Optional[str]:
    """Save the main six-panel overview plot and return its path."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    num_layers = config["NUM_LAYERS"]
    num_steps = config["NUM_STEPS"]
    dim = config["DIM"]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(
        "1.2b-ii: Polar-factor trajectory proxy analysis\n"
        f"{num_layers}-layer linear net, dim={dim}, {num_steps} steps",
        fontsize=14,
        fontweight="bold",
    )

    t_axis = np.arange(1, num_steps + 1)

    ax = axes[0, 0]
    ax.set_title("(a) Cumulative orientation drift ||Q_t - Q_0||")
    for layer in range(num_layers):
        ax.plot(t_axis, results_sgd["cum_Q_drift"][layer, :], "b-", alpha=0.3, linewidth=0.8)
        ax.plot(t_axis, results_muon["cum_Q_drift"][layer, :], "r-", alpha=0.3, linewidth=0.8)
    ax.plot(t_axis, np.nanmean(results_sgd["cum_Q_drift"], axis=0), "b-", linewidth=2.5, label="SGD (avg)")
    ax.plot(t_axis, np.nanmean(results_muon["cum_Q_drift"], axis=0), "r-", linewidth=2.5, label="Muon (avg)")
    ax.set_xlabel("Step")
    ax.set_ylabel("||Q_t - Q_0||_F")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.set_title("(b) Cumulative spectrum drift ||P_t - P_0||")
    for layer in range(num_layers):
        ax.plot(t_axis, results_sgd["cum_P_drift"][layer, :], "b-", alpha=0.3, linewidth=0.8)
        ax.plot(t_axis, results_muon["cum_P_drift"][layer, :], "r-", alpha=0.3, linewidth=0.8)
    ax.plot(t_axis, np.nanmean(results_sgd["cum_P_drift"], axis=0), "b-", linewidth=2.5, label="SGD (avg)")
    ax.plot(t_axis, np.nanmean(results_muon["cum_P_drift"], axis=0), "r-", linewidth=2.5, label="Muon (avg)")
    ax.set_xlabel("Step")
    ax.set_ylabel("||P_t - P_0||_F")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.set_title("(c) Cumulative orientation fraction Q/(Q+P)")
    sgd_q_avg = np.nanmean(results_sgd["cum_Q_drift"], axis=0)
    sgd_p_avg = np.nanmean(results_sgd["cum_P_drift"], axis=0)
    muon_q_avg = np.nanmean(results_muon["cum_Q_drift"], axis=0)
    muon_p_avg = np.nanmean(results_muon["cum_P_drift"], axis=0)
    ax.plot(t_axis, safe_ratio(sgd_q_avg, sgd_q_avg + sgd_p_avg), "b-", linewidth=2.5, label="SGD")
    ax.plot(t_axis, safe_ratio(muon_q_avg, muon_q_avg + muon_p_avg), "r-", linewidth=2.5, label="Muon")
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Q=P")
    ax.set_xlabel("Step")
    ax.set_ylabel("Q/(Q+P)")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.set_title("(d) Aggregate per-step Σ||ΔQ|| / Σ||ΔW|| across layers")
    ax.plot(t_axis, summary_sgd["stepwise_aggregate_dQ_over_dW"], "b-", linewidth=2, label="SGD")
    ax.plot(t_axis, summary_muon["stepwise_aggregate_dQ_over_dW"], "r-", linewidth=2, label="Muon")
    ax.set_xlabel("Step")
    ax.set_ylabel("Aggregate ||ΔQ|| / ||ΔW||")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.set_title("(e) Aggregate per-step Σ||ΔP|| / Σ||ΔW|| across layers")
    ax.plot(t_axis, summary_sgd["stepwise_aggregate_dP_over_dW"], "b-", linewidth=2, label="SGD")
    ax.plot(t_axis, summary_muon["stepwise_aggregate_dP_over_dW"], "r-", linewidth=2, label="Muon")
    ax.set_xlabel("Step")
    ax.set_ylabel("Aggregate ||ΔP|| / ||ΔW||")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.set_title("(f) Training loss")
    ax.semilogy(np.arange(num_steps + 1), results_sgd["losses"], "b-", linewidth=2, label=f"SGD (lr={lr_sgd})")
    ax.semilogy(
        np.arange(num_steps + 1),
        results_muon["losses"],
        "r-",
        linewidth=2,
        label=f"Muon (lr={config['LR_MUON']})",
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = out_dir / "polar_trajectory_decomposition.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(plot_path)


def make_phase_portrait(
    results_sgd: Dict[str, Any],
    results_muon: Dict[str, Any],
    config: Dict[str, Any],
    out_dir: Path,
) -> Optional[str]:
    """Save the phase portrait plot and return its path."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    num_layers = config["NUM_LAYERS"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "1.2b-ii: Phase portrait of cumulative polar-factor drifts",
        fontsize=13,
        fontweight="bold",
    )

    ax = axes[0]
    ax.set_title("(a) Per-layer trajectories")
    for layer in range(num_layers):
        ax.plot(
            results_sgd["cum_P_drift"][layer, :],
            results_sgd["cum_Q_drift"][layer, :],
            "b-",
            alpha=0.5,
            linewidth=1,
            label="SGD" if layer == 0 else None,
        )
        ax.plot(
            results_muon["cum_P_drift"][layer, :],
            results_muon["cum_Q_drift"][layer, :],
            "r-",
            alpha=0.5,
            linewidth=1,
            label="Muon" if layer == 0 else None,
        )
        ax.plot(results_sgd["cum_P_drift"][layer, -1], results_sgd["cum_Q_drift"][layer, -1], "bx", markersize=8)
        ax.plot(results_muon["cum_P_drift"][layer, -1], results_muon["cum_Q_drift"][layer, -1], "rx", markersize=8)

    max_val = max(
        float(np.nanmax(results_sgd["cum_Q_drift"])),
        float(np.nanmax(results_sgd["cum_P_drift"])),
        float(np.nanmax(results_muon["cum_Q_drift"])),
        float(np.nanmax(results_muon["cum_P_drift"])),
    )
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.4, label="Q=P")
    ax.set_xlabel("Cumulative ||P_t - P_0||_F")
    ax.set_ylabel("Cumulative ||Q_t - Q_0||_F")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.set_title("(b) Layer-averaged trajectory")
    sgd_q_avg = np.nanmean(results_sgd["cum_Q_drift"], axis=0)
    sgd_p_avg = np.nanmean(results_sgd["cum_P_drift"], axis=0)
    muon_q_avg = np.nanmean(results_muon["cum_Q_drift"], axis=0)
    muon_p_avg = np.nanmean(results_muon["cum_P_drift"], axis=0)
    ax.plot(sgd_p_avg, sgd_q_avg, "b-", linewidth=2.5, label="SGD")
    ax.plot(muon_p_avg, muon_q_avg, "r-", linewidth=2.5, label="Muon")
    ax.plot(sgd_p_avg[-1], sgd_q_avg[-1], "bx", markersize=12, markeredgewidth=3)
    ax.plot(muon_p_avg[-1], muon_q_avg[-1], "rx", markersize=12, markeredgewidth=3)
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.4, label="Q=P")
    ax.set_xlabel("Cumulative ||P_t - P_0||_F")
    ax.set_ylabel("Cumulative ||Q_t - Q_0||_F")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = out_dir / "polar_phase_portrait.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(plot_path)


# =============================================================================
# REPORTING
# =============================================================================

def print_header(config: Dict[str, Any]) -> None:
    print("=" * 100)
    print("1.2b-ii: POLAR-FACTOR TRAJECTORY PROXY ANALYSIS -- (Q_t, P_t) DECOMPOSITION")
    print("=" * 100)
    print(
        f"Setup: {config['NUM_LAYERS']}-layer deep linear net (dim={config['DIM']}), "
        f"quadratic loss, {config['NUM_STEPS']} steps"
    )
    print(f"LR_Muon={config['LR_MUON']}, Momentum={config['MOMENTUM']}, NS iters={config['NS_ITERS']}")
    print(f"Report at steps: {config['REPORT_STEPS']}")
    print("Scope note: per-layer polar-factor proxy only; not a direct deep-linear gauge-coordinate measurement.")
    print("Ratio note: raw ||ΔQ||/||ΔW|| and ||ΔP||/||ΔW|| can be noisy when ||ΔW|| is tiny.")
    print("=" * 100)


def print_problem_diagnostics(problem: Dict[str, Any], lr_sgd: float, config: Dict[str, Any]) -> None:
    print(f"\nSGD learning rate (max stable candidate): {lr_sgd}")
    print(f"Muon learning rate (fixed default):       {config['LR_MUON']}")
    print(f"\nInitial loss: {problem['initial_loss']:.6e}")
    print(f"Target condition number: {problem['target_condition_number']:.4f}")
    print("\nPolar decomposition checks on initial weights:")
    for row in problem["polar_checks"]:
        print(
            f"  Layer {row['layer']}: recon_err={row['recon_err']:.2e}, "
            f"Q_orth_err={row['Q_orth_err']:.2e}, P_sym_err={row['P_sym_err']:.2e}, "
            f"P_min_eig={row['P_min_eig']:.4f}"
        )


def print_windowed_ratio_table(rows: List[Dict[str, Any]]) -> None:
    print(f"\n\n{'=' * 100}")
    print("TABLE 1A: WINDOWED AGGREGATE PER-STEP RATIOS")
    print("          Ratios are Σ||ΔQ||/Σ||ΔW|| and Σ||ΔP||/Σ||ΔW|| over a 10-step window across all layers")
    print("=" * 100)
    print(
        f"\n{'Step':>6} | {'Window':>11} | {'SGD agg dQ/dW':>14} | {'SGD agg dP/dW':>14} | "
        f"{'Muon agg dQ/dW':>15} | {'Muon agg dP/dW':>15}"
    )
    print("-" * 94)
    for row in rows:
        print(
            f"{row['step']:6d} | {str(tuple(row['window'])):>11} | "
            f"{row['sgd_aggregate_dQ_over_dW']:14.6f} | {row['sgd_aggregate_dP_over_dW']:14.6f} | "
            f"{row['muon_aggregate_dQ_over_dW']:15.6f} | {row['muon_aggregate_dP_over_dW']:15.6f}"
        )

    print(f"\n\n{'=' * 100}")
    print("TABLE 1B: SUPPLEMENTAL RAW-MEAN PER-STEP RATIOS")
    print("          Legacy continuity metric; can spike when ||ΔW|| is very small")
    print("=" * 100)
    print(
        f"\n{'Step':>6} | {'SGD raw dQ/dW':>14} | {'SGD raw dP/dW':>14} | "
        f"{'Muon raw dQ/dW':>15} | {'Muon raw dP/dW':>15}"
    )
    print("-" * 80)
    for row in rows:
        print(
            f"{row['step']:6d} | {row['sgd_raw_mean_dQ_ratio']:14.6f} | {row['sgd_raw_mean_dP_ratio']:14.6f} | "
            f"{row['muon_raw_mean_dQ_ratio']:15.6f} | {row['muon_raw_mean_dP_ratio']:15.6f}"
        )


def print_cumulative_table(rows: List[Dict[str, Any]]) -> None:
    print(f"\n\n{'=' * 100}")
    print("TABLE 2: CUMULATIVE DRIFT FROM INITIALIZATION (layer-averaged)")
    print("=" * 100)
    print(
        f"\n{'Step':>6} | {'SGD ||Q-Q0||':>14} | {'SGD ||P-P0||':>14} | {'SGD ||W-W0||':>14} | "
        f"{'Muon ||Q-Q0||':>14} | {'Muon ||P-P0||':>14} | {'Muon ||W-W0||':>14}"
    )
    print("-" * 105)
    for row in rows:
        print(
            f"{row['step']:6d} | {row['sgd_cum_Q']:14.6f} | {row['sgd_cum_P']:14.6f} | {row['sgd_cum_W']:14.6f} | "
            f"{row['muon_cum_Q']:14.6f} | {row['muon_cum_P']:14.6f} | {row['muon_cum_W']:14.6f}"
        )

    print(f"\n\n{'=' * 100}")
    print("TABLE 3: CUMULATIVE ORIENTATION-vs-SPECTRUM BALANCE (layer-averaged)")
    print("=" * 100)
    print(
        f"\n{'Step':>6} | {'SGD Q/P':>10} | {'Muon Q/P':>10} | {'SGD Q/(Q+P)':>12} | {'Muon Q/(Q+P)':>13}"
    )
    print("-" * 64)
    for row in rows:
        print(
            f"{row['step']:6d} | {row['sgd_Q_over_P']:10.4f} | {row['muon_Q_over_P']:10.4f} | "
            f"{row['sgd_Q_fraction']:12.4f} | {row['muon_Q_fraction']:13.4f}"
        )


def print_per_layer_table(per_layer: Dict[str, List[Dict[str, Any]]], lr_sgd: float, config: Dict[str, Any]) -> None:
    print(f"\n\n{'=' * 100}")
    print("TABLE 4: PER-LAYER BREAKDOWN AT FINAL STEP")
    print("=" * 100)

    for optimizer_name, lr_label in [("sgd", lr_sgd), ("muon", config["LR_MUON"] )]:
        label = "SGD" if optimizer_name == "sgd" else "Muon"
        print(f"\n  {label} (lr={lr_label}):")
        print(
            f"  {'Layer':>6} | {'agg dQ/dW':>10} | {'agg dP/dW':>10} | {'raw dQ/dW':>10} | {'raw dP/dW':>10} | "
            f"{'cum ||Q-Q0||':>12} | {'cum ||P-P0||':>12} | {'Q/(Q+P)':>8}"
        )
        print("  " + "-" * 102)
        for row in per_layer[optimizer_name]:
            print(
                f"  {row['layer']:6d} | {row['aggregate_dQ_over_dW']:10.6f} | {row['aggregate_dP_over_dW']:10.6f} | "
                f"{row['raw_mean_dQ_ratio']:10.6f} | {row['raw_mean_dP_ratio']:10.6f} | "
                f"{row['cum_Q_final']:12.6f} | {row['cum_P_final']:12.6f} | {row['Q_fraction_final']:8.4f}"
            )


def print_phase_table(rows: List[Dict[str, Any]]) -> None:
    print(f"\n\n{'=' * 100}")
    print("TABLE 5: EARLY vs LATE TRAINING DYNAMICS")
    print("=" * 100)
    print(
        f"\n{'Phase':>10} | {'SGD agg dQ/dW':>14} | {'SGD agg dP/dW':>14} | "
        f"{'Muon agg dQ/dW':>15} | {'Muon agg dP/dW':>15}"
    )
    print("-" * 84)
    for row in rows:
        print(
            f"{row['phase']:>10} | {row['sgd_aggregate_dQ_over_dW']:14.6f} | {row['sgd_aggregate_dP_over_dW']:14.6f} | "
            f"{row['muon_aggregate_dQ_over_dW']:15.6f} | {row['muon_aggregate_dP_over_dW']:15.6f}"
        )

    print(f"\n{'Phase':>10} | {'SGD raw dQ/dW':>14} | {'SGD raw dP/dW':>14} | {'Muon raw dQ/dW':>15} | {'Muon raw dP/dW':>15}")
    print("-" * 84)
    for row in rows:
        print(
            f"{row['phase']:>10} | {row['sgd_raw_mean_dQ_ratio']:14.6f} | {row['sgd_raw_mean_dP_ratio']:14.6f} | "
            f"{row['muon_raw_mean_dQ_ratio']:15.6f} | {row['muon_raw_mean_dP_ratio']:15.6f}"
        )


def print_diagnostics(summary_sgd: Dict[str, Any], summary_muon: Dict[str, Any]) -> None:
    print(f"\n\n{'=' * 100}")
    print("RATIO DIAGNOSTICS")
    print("=" * 100)
    for label, summary in [("SGD", summary_sgd), ("Muon", summary_muon)]:
        print(f"\n{label}:")
        print(
            f"  Aggregate Σ||ΔQ||/Σ||ΔW|| = {summary['aggregate_dQ_over_dW']:.6f}, "
            f"Aggregate Σ||ΔP||/Σ||ΔW|| = {summary['aggregate_dP_over_dW']:.6f}"
        )
        print(
            f"  Raw mean ||ΔQ||/||ΔW|| = {summary['raw_mean_dQ_ratio']:.6f}, "
            f"Raw mean ||ΔP||/||ΔW|| = {summary['raw_mean_dP_ratio']:.6f}"
        )
        print(
            f"  ||ΔW|| stats: min={summary['dW_stats']['min']:.3e}, median={summary['dW_stats']['median']:.3e}, "
            f"mean={summary['dW_stats']['mean']:.3e}, max={summary['dW_stats']['max']:.3e}"
        )
        print(
            f"  Raw ||ΔQ||/||ΔW|| stats: min={summary['dQ_ratio_stats']['min']:.3e}, "
            f"median={summary['dQ_ratio_stats']['median']:.3e}, max={summary['dQ_ratio_stats']['max']:.3e}"
        )
        print(
            f"  Raw ||ΔP||/||ΔW|| stats: min={summary['dP_ratio_stats']['min']:.3e}, "
            f"median={summary['dP_ratio_stats']['median']:.3e}, max={summary['dP_ratio_stats']['max']:.3e}"
        )
        print(
            f"  Small ||ΔW|| count (<1e-12): {summary['small_dW_count']} / {summary['valid_ratio_count']} "
            f"({summary['small_dW_fraction']:.2%})"
        )


def print_assessment(assessment: Dict[str, Any], summary_sgd: Dict[str, Any], summary_muon: Dict[str, Any], config: Dict[str, Any]) -> None:
    print(f"\n\n{'=' * 100}")
    print("FINAL ANALYSIS")
    print("=" * 100)
    print(
        f"\nFinal losses: SGD={summary_sgd['final_loss']:.6e}, Muon={summary_muon['final_loss']:.6e}"
    )
    print(
        f"Final cumulative Q/(Q+P): SGD={summary_sgd['mean_cum_Q_fraction_final']:.4f}, "
        f"Muon={summary_muon['mean_cum_Q_fraction_final']:.4f}"
    )

    print("\nLegacy continuity tests (raw-metric-heavy; kept for comparability):")
    for test_id, test in assessment["legacy_tests"].items():
        outcome = "YES" if test["passed"] else "NO"
        print(
            f"  {test_id}: {test['description']}\n"
            f"      Muon={test['muon']:.6f} vs SGD={test['sgd']:.6f} -> {outcome}"
        )

    print("\nRobust supplemental tests (aggregate norm ratios):")
    for test_id, test in assessment["robust_tests"].items():
        outcome = "YES" if test["passed"] else "NO"
        print(
            f"  {test_id}: {test['description']}\n"
            f"      Muon={test['muon']:.6f} vs SGD={test['sgd']:.6f} -> {outcome}"
        )

    print(f"\nLegacy heuristic score: {assessment['legacy_score']}/5 -> {assessment['legacy_label']}")
    print("Caution: the legacy label should not be treated as a strong inferential claim.")

    conclusion = assessment["calibrated_conclusion"]
    print(f"\nCalibrated conclusion: {conclusion['label']}")
    print(f"  {conclusion['detail']}")

    print("\nSupported in this run:")
    for item in conclusion["supported_expectations"]:
        print(f"  - {item}")

    print("\nNot supported in this run:")
    for item in conclusion["unsupported_expectations"]:
        print(f"  - {item}")

    print("\nLimitations:")
    for item in conclusion["limitations"]:
        print(f"  - {item}")

    print("=" * 100)
    print(f"Overall calibrated conclusion: {conclusion['label']}")
    print("=" * 100)


# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================

def run_experiment(
    config_overrides: Optional[Dict[str, Any]] = None,
    make_plots: bool = True,
    out_dir: Optional[os.PathLike[str] | str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the full experiment and return structured raw results plus summaries.

    Returns a dictionary with:
      - config
      - metadata
      - problem diagnostics
      - raw optimizer results for SGD and Muon
      - summary tables and diagnostics
      - legacy continuity tests and a calibrated conclusion
      - plot paths (if plotting succeeded)
    """
    start_time = time.time()
    config = prepare_config(config_overrides)
    output_dir = Path(out_dir) if out_dir is not None else SCRIPT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print_header(config)

    target, x_data = build_problem(config)
    problem = compute_problem_diagnostics(config, x_data, target)
    lr_sgd = find_stable_lr_sgd(config, x_data, target)

    if verbose:
        print_problem_diagnostics(problem, lr_sgd, config)
        print(f"\n{'=' * 100}")
        print("RUNNING OPTIMIZERS AND TRACKING POLAR FACTORS")
        print("=" * 100)
        print("\n  Running SGD...", flush=True)

    results_sgd = run_and_track_polar("sgd", lr_sgd, config["NUM_STEPS"], config, x_data, target, verbose=verbose)

    if verbose:
        print(f"    Final loss: {results_sgd['losses'][-1]:.6e}")
        print("\n  Running Muon...", flush=True)

    results_muon = run_and_track_polar("muon", config["LR_MUON"], config["NUM_STEPS"], config, x_data, target, verbose=verbose)

    if verbose:
        print(f"    Final loss: {results_muon['losses'][-1]:.6e}")

    summary_sgd = summarize_single_optimizer(results_sgd, config)
    summary_muon = summarize_single_optimizer(results_muon, config)

    tables = {
        "windowed_ratios": build_windowed_ratio_rows(results_sgd, results_muon, config),
        "cumulative_drift": build_cumulative_drift_rows(results_sgd, results_muon, config),
        "per_layer_final": {
            "sgd": build_per_layer_rows(results_sgd, config),
            "muon": build_per_layer_rows(results_muon, config),
        },
        "phase_comparison": build_phase_rows(results_sgd, results_muon, config),
    }

    assessment = evaluate_assessment(summary_sgd, summary_muon, config)

    plot_paths = {"overview": None, "phase_portrait": None}
    if make_plots:
        plot_paths["overview"] = make_overview_plot(
            results_sgd,
            results_muon,
            summary_sgd,
            summary_muon,
            config,
            lr_sgd,
            output_dir,
        )
        plot_paths["phase_portrait"] = make_phase_portrait(results_sgd, results_muon, config, output_dir)

    runtime_sec = time.time() - start_time

    report = {
        "config": config,
        "metadata": {
            "script_path": str(Path(__file__).resolve()),
            "out_dir": str(output_dir.resolve()),
            "runtime_sec": float(runtime_sec),
            "scope": "Single-seed fixed-batch toy study using per-layer polar-factor proxy diagnostics.",
            "limitations": [
                "Per-layer Q/P diagnostics are not direct deep-linear gauge coordinates.",
                "Raw ratio metrics can spike when ||ΔW|| is tiny.",
                "Single-seed fixed-batch evidence only.",
            ],
        },
        "problem": problem,
        "learning_rates": {
            "sgd": float(lr_sgd),
            "muon": float(config["LR_MUON"]),
        },
        "optimizers": {
            "sgd": {
                "name": "SGD+momentum",
                "lr": float(lr_sgd),
                "results": results_sgd,
                "summary": summary_sgd,
            },
            "muon": {
                "name": "Muon+momentum",
                "lr": float(config["LR_MUON"]),
                "results": results_muon,
                "summary": summary_muon,
            },
        },
        "tables": tables,
        "assessment": assessment,
        "plots": plot_paths,
    }

    if verbose:
        print_windowed_ratio_table(tables["windowed_ratios"])
        print_cumulative_table(tables["cumulative_drift"])
        print_per_layer_table(tables["per_layer_final"], lr_sgd, config)
        print_phase_table(tables["phase_comparison"])
        print_diagnostics(summary_sgd, summary_muon)
        if make_plots:
            print(f"\n\n{'=' * 100}")
            print("PLOTS")
            print("=" * 100)
            if plot_paths["overview"] is not None:
                print(f"Overview plot saved to:      {plot_paths['overview']}")
            else:
                print("Overview plot skipped (matplotlib unavailable).")
            if plot_paths["phase_portrait"] is not None:
                print(f"Phase-portrait plot saved to: {plot_paths['phase_portrait']}")
            else:
                print("Phase-portrait plot skipped (matplotlib unavailable).")
        print_assessment(assessment, summary_sgd, summary_muon, config)
        print(f"Runtime: {runtime_sec:.2f}s")

    return report


def main() -> None:
    """CLI entrypoint preserving normal script behavior."""
    run_experiment(make_plots=True, out_dir=SCRIPT_DIR, verbose=True)


if __name__ == "__main__":
    main()
