#!/usr/bin/env python3
"""
H3: NORMALIZED SGD vs MUON
==========================

Toy-scope synthetic comparison of Muon's Newton-Schulz orthogonalized updates
against several magnitude-discarding baselines:
  - SGD
  - Muon-style orthogonalized updates
  - Frobenius-normalized SGD
  - Spectral-norm-normalized SGD
  - Sign SGD

This script keeps the original 4-layer 32x32 convergence-basin experiment, but
reports it more carefully:
  - descriptive statistics, not formal statistical tests
  - heuristic comparison rules, not universal adjudication
  - architecture-specific conclusions for deep linear vs ReLU nets

Default behavior remains script-friendly: running
    python run_experiment.py
executes the full experiment and saves figures next to this file unless an
explicit output directory is provided.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


EXPERIMENT_ID = "H3_NORMALIZED_SGD_vs_MUON"
EXPERIMENT_TITLE = "H3: Normalized SGD vs Muon"
SCOPE_NOTE = (
    "Controlled synthetic basin experiment on fixed 4-layer 32x32 networks; "
    "results are descriptive and architecture-dependent."
)

DIM = 32
NUM_LAYERS = 4
NUM_STEPS = 500
BATCH_SIZE = 64
MOMENTUM = 0.9
NS_ITERS = 5
NUM_INDEPENDENT_RUNS = 20
NUM_TEST_INPUTS = 50
LR_SWEEP_STEPS = 200
DATA_SEED = 42
RUN_SEED_START = 1000
DIVERGENCE_MULTIPLIER = 50.0
DIVERGENCE_ABS_THRESHOLD = 1e10
EPS = 1e-12

ARCHITECTURES = ("linear", "relu")
OPTIMIZER_NAMES = ("sgd", "muon", "norm_sgd", "spectral_sgd", "sign_sgd")
OPTIMIZER_LABELS = {
    "sgd": "SGD",
    "muon": "Muon (UV^T)",
    "norm_sgd": "Normalized SGD (Frob)",
    "spectral_sgd": "Spectral-Norm SGD",
    "sign_sgd": "Sign SGD",
}
OPTIMIZER_COLORS = {
    "sgd": "#4477AA",
    "muon": "#CC3311",
    "norm_sgd": "#228B22",
    "spectral_sgd": "#9933CC",
    "sign_sgd": "#FF8800",
}
LR_CANDIDATE_GRIDS = {
    "sgd": [0.1, 0.07, 0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.001],
    "muon": [0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.002, 0.001],
    "norm_sgd": [0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.002, 0.001, 0.0005],
    "spectral_sgd": [0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.002, 0.001, 0.0005],
    "sign_sgd": [0.005, 0.003, 0.002, 0.001, 0.0007, 0.0005, 0.0003, 0.0002, 0.0001, 0.00005],
}

SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DEFAULT_OUTPUT_DIR = SCRIPT_DIR


def generate_fixed_problem(seed=DATA_SEED):
    rng = np.random.RandomState(seed)
    return {
        "W_target": rng.randn(DIM, DIM) * 0.5,
        "X_data": rng.randn(DIM, BATCH_SIZE) * 0.3,
        "X_test": rng.randn(DIM, NUM_TEST_INPUTS) * 0.3,
    }


PROBLEM = generate_fixed_problem()
W_target = PROBLEM["W_target"]
X_data = PROBLEM["X_data"]
X_test = PROBLEM["X_test"]


# =============================================================================
# Network definitions
# =============================================================================

def init_weights(num_layers, seed=DATA_SEED):
    rng = np.random.RandomState(seed)
    weights = []
    for _ in range(num_layers):
        W = np.eye(DIM) + rng.randn(DIM, DIM) * 0.1
        weights.append(W.copy())
    return weights


def forward_linear(weights, X):
    out = X.copy()
    for W in weights:
        out = W @ out
    return out


def compute_loss_linear(weights, X, target):
    pred = forward_linear(weights, X)
    target_out = target @ X
    diff = pred - target_out
    return 0.5 * np.mean(np.sum(diff ** 2, axis=0))


def compute_gradients_linear(weights, X, target):
    num_layers = len(weights)
    N = X.shape[1]
    activations = [X.copy()]
    out = X.copy()
    for W in weights:
        out = W @ out
        activations.append(out.copy())
    target_out = target @ X
    delta = (activations[-1] - target_out) / N
    grads = []
    for i in range(num_layers - 1, -1, -1):
        G = delta @ activations[i].T
        grads.insert(0, G)
        if i > 0:
            delta = weights[i].T @ delta
    return grads


def forward_relu(weights, X):
    out = X.copy()
    for idx, W in enumerate(weights):
        out = W @ out
        if idx < len(weights) - 1:
            out = np.maximum(0, out)
    return out


def compute_loss_relu(weights, X, target):
    pred = forward_relu(weights, X)
    target_out = target @ X
    diff = pred - target_out
    return 0.5 * np.mean(np.sum(diff ** 2, axis=0))


def compute_gradients_relu(weights, X, target):
    num_layers = len(weights)
    N = X.shape[1]
    pre_activations = []
    post_activations = [X.copy()]
    out = X.copy()
    for idx, W in enumerate(weights):
        pre = W @ out
        pre_activations.append(pre.copy())
        if idx < num_layers - 1:
            out = np.maximum(0, pre)
        else:
            out = pre
        post_activations.append(out.copy())
    target_out = target @ X
    delta = (post_activations[-1] - target_out) / N
    grads = []
    for i in range(num_layers - 1, -1, -1):
        G = delta @ post_activations[i].T
        grads.insert(0, G)
        if i > 0:
            delta = weights[i].T @ delta
            delta = delta * (pre_activations[i - 1] > 0).astype(float)
    return grads


# =============================================================================
# Optimizer step functions
# =============================================================================

def newton_schulz_orthogonalize(G, num_iters=NS_ITERS):
    norm = np.linalg.norm(G, ord="fro")
    if norm < EPS:
        return G
    X = G / norm
    for _ in range(num_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


def optimizer_step(weights, velocities, grads, lr, method):
    for i in range(len(weights)):
        if method == "sgd":
            velocities[i] = MOMENTUM * velocities[i] + grads[i]
            weights[i] = weights[i] - lr * velocities[i]

        elif method == "muon":
            ortho_grad = newton_schulz_orthogonalize(grads[i])
            velocities[i] = MOMENTUM * velocities[i] + ortho_grad
            weights[i] = weights[i] - lr * velocities[i]

        elif method == "norm_sgd":
            velocities[i] = MOMENTUM * velocities[i] + grads[i]
            v_norm = np.linalg.norm(velocities[i], ord="fro")
            step = velocities[i] / v_norm if v_norm > EPS else velocities[i]
            weights[i] = weights[i] - lr * step

        elif method == "spectral_sgd":
            velocities[i] = MOMENTUM * velocities[i] + grads[i]
            spec_norm = np.linalg.norm(velocities[i], ord=2)
            step = velocities[i] / spec_norm if spec_norm > EPS else velocities[i]
            weights[i] = weights[i] - lr * step

        elif method == "sign_sgd":
            velocities[i] = MOMENTUM * velocities[i] + grads[i]
            step = np.sign(velocities[i])
            weights[i] = weights[i] - lr * step

        else:
            raise ValueError(f"Unknown optimizer method: {method}")

    return weights, velocities


# =============================================================================
# Learning-rate sweep and training engine
# =============================================================================

def find_best_lr(method, net_type, num_steps=LR_SWEEP_STEPS):
    compute_loss_fn = compute_loss_linear if net_type == "linear" else compute_loss_relu
    compute_grad_fn = compute_gradients_linear if net_type == "linear" else compute_gradients_relu

    candidates = list(LR_CANDIDATE_GRIDS[method])
    best_lr = candidates[-1]
    best_loss = float("inf")
    trial_rows = []

    for idx, lr_cand in enumerate(candidates):
        np.random.seed(DATA_SEED)
        weights = init_weights(NUM_LAYERS)
        velocities = [np.zeros((DIM, DIM)) for _ in range(NUM_LAYERS)]
        initial_loss = compute_loss_fn(weights, X_data, W_target)
        stable = True
        final_loss = initial_loss
        break_step = None

        for step in range(num_steps):
            grads = compute_grad_fn(weights, X_data, W_target)
            weights, velocities = optimizer_step(weights, velocities, grads, lr_cand, method)
            loss = compute_loss_fn(weights, X_data, W_target)
            if np.isnan(loss) or loss > initial_loss * DIVERGENCE_MULTIPLIER:
                stable = False
                final_loss = loss
                break_step = step
                break
            final_loss = loss

        if stable and final_loss < best_loss:
            best_loss = final_loss
            best_lr = lr_cand

        trial_rows.append(
            {
                "candidate_index": idx,
                "lr": lr_cand,
                "stable": bool(stable),
                "final_loss": float(final_loss),
                "break_step": break_step,
            }
        )

    best_index = candidates.index(best_lr)
    hit_grid_boundary = best_index in (0, len(candidates) - 1)
    if best_index == 0:
        boundary_side = "highest"
    elif best_index == len(candidates) - 1:
        boundary_side = "lowest"
    else:
        boundary_side = None

    return {
        "method": method,
        "net_type": net_type,
        "num_steps": num_steps,
        "candidates": np.array(candidates, dtype=float),
        "trials": trial_rows,
        "best_lr": float(best_lr),
        "best_loss": float(best_loss),
        "best_index": int(best_index),
        "hit_grid_boundary": bool(hit_grid_boundary),
        "boundary_side": boundary_side,
    }


def run_training(weights_init, method, lr, num_steps, net_type):
    compute_loss_fn = compute_loss_linear if net_type == "linear" else compute_loss_relu
    compute_grad_fn = compute_gradients_linear if net_type == "linear" else compute_gradients_relu

    weights = [w.copy() for w in weights_init]
    velocities = [np.zeros_like(w) for w in weights]
    losses = [compute_loss_fn(weights, X_data, W_target)]
    diverged = False
    divergence_step = None

    for step in range(num_steps):
        grads = compute_grad_fn(weights, X_data, W_target)
        weights, velocities = optimizer_step(weights, velocities, grads, lr, method)
        loss = compute_loss_fn(weights, X_data, W_target)
        losses.append(loss)
        if np.isnan(loss) or loss > DIVERGENCE_ABS_THRESHOLD:
            diverged = True
            divergence_step = step
            for _ in range(num_steps - step - 1):
                losses.append(np.nan)
            break

    return {
        "loss_curve": np.array(losses, dtype=float),
        "final_weights": weights,
        "diverged": bool(diverged),
        "divergence_step": divergence_step,
    }


def _summarize_loss_curves(loss_curve_matrix):
    valid = np.isfinite(loss_curve_matrix)
    counts = valid.sum(axis=0)
    sums = np.where(valid, loss_curve_matrix, 0.0).sum(axis=0)
    means = np.divide(sums, counts, out=np.full(loss_curve_matrix.shape[1], np.nan), where=counts > 0)

    centered = np.where(valid, loss_curve_matrix - means, 0.0)
    sq_sums = (centered ** 2).sum(axis=0)
    stds = np.sqrt(
        np.divide(sq_sums, counts, out=np.full(loss_curve_matrix.shape[1], np.nan), where=counts > 0)
    )
    sems = np.divide(stds, np.sqrt(counts), out=np.full(loss_curve_matrix.shape[1], np.nan), where=counts > 0)
    return means, stds, sems, counts


def measure_convergence_basin(method, lr, net_type, num_runs=NUM_INDEPENDENT_RUNS, num_steps=NUM_STEPS):
    forward_fn = forward_linear if net_type == "linear" else forward_relu
    compute_loss_fn = compute_loss_linear if net_type == "linear" else compute_loss_relu

    final_weights_list = []
    final_functions = []
    final_losses = []
    loss_curves = []
    condition_numbers = []
    diverged_flags = []
    divergence_steps = []
    run_seeds = []

    for run_idx in range(num_runs):
        seed = RUN_SEED_START + run_idx
        run_seeds.append(seed)
        weights_init = init_weights(NUM_LAYERS, seed=seed)
        train_result = run_training(weights_init, method, lr, num_steps, net_type)
        loss_curve = train_result["loss_curve"]
        final_weights = train_result["final_weights"]

        loss_curves.append(loss_curve)
        final_weights_list.append(final_weights)
        final_functions.append(forward_fn(final_weights, X_test).copy())
        final_losses.append(compute_loss_fn(final_weights, X_data, W_target))
        diverged_flags.append(train_result["diverged"])
        divergence_steps.append(train_result["divergence_step"])

        cond_per_layer = []
        for W in final_weights:
            svs = np.linalg.svd(W, compute_uv=False)
            if svs[-1] > 1e-15:
                cond_per_layer.append(svs[0] / svs[-1])
            else:
                cond_per_layer.append(np.inf)
        condition_numbers.append(cond_per_layer)

    n = len(final_weights_list)
    weight_dists = []
    func_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d_w = 0.0
            for k in range(NUM_LAYERS):
                d_w += np.linalg.norm(final_weights_list[i][k] - final_weights_list[j][k], ord="fro") ** 2
            weight_dists.append(np.sqrt(d_w))
            d_f = np.linalg.norm(final_functions[i] - final_functions[j], ord="fro") / np.linalg.norm(X_test, ord="fro")
            func_dists.append(d_f)

    cond_arr = np.array(condition_numbers, dtype=float)
    mean_cond = np.mean(cond_arr, axis=0)
    geom_cond_by_run = np.exp(np.mean(np.log(np.clip(cond_arr, 1e-15, None)), axis=1))

    max_len = max(len(curve) for curve in loss_curves)
    padded = np.full((num_runs, max_len), np.nan)
    for idx, curve in enumerate(loss_curves):
        padded[idx, : len(curve)] = curve

    mean_loss_curve, std_loss_curve, sem_loss_curve, valid_counts = _summarize_loss_curves(padded)
    weight_dists = np.array(weight_dists, dtype=float)
    func_dists = np.array(func_dists, dtype=float)
    final_losses = np.array(final_losses, dtype=float)

    weight_diversity_mean = np.mean(weight_dists)
    func_diversity_mean = np.mean(func_dists)
    function_over_weight_ratio = (
        func_diversity_mean / weight_diversity_mean if weight_diversity_mean > 1e-15 else np.nan
    )

    return {
        "lr": float(lr),
        "net_type": net_type,
        "method": method,
        "loss_mean": float(np.mean(final_losses)),
        "loss_std": float(np.std(final_losses)),
        "losses": final_losses,
        "loss_curve_matrix": padded,
        "loss_curve_mean": mean_loss_curve,
        "loss_curve_std": std_loss_curve,
        "loss_curve_sem": sem_loss_curve,
        "loss_curve_valid_counts": valid_counts,
        "weight_diversity_mean": float(weight_diversity_mean),
        "weight_diversity_std": float(np.std(weight_dists)),
        "func_diversity_mean": float(func_diversity_mean),
        "func_diversity_std": float(np.std(func_dists)),
        "function_over_weight_ratio": float(function_over_weight_ratio),
        "weight_dists": weight_dists,
        "func_dists": func_dists,
        "mean_cond_per_layer": mean_cond,
        "cond_all": cond_arr,
        "condition_geom_by_run": geom_cond_by_run,
        "condition_geom_mean": float(np.mean(geom_cond_by_run)),
        "condition_geom_std": float(np.std(geom_cond_by_run)),
        "run_seeds": run_seeds,
        "num_diverged_runs": int(np.sum(diverged_flags)),
        "stable_run_fraction": float(1.0 - np.mean(diverged_flags)),
        "diverged_flags": np.array(diverged_flags, dtype=bool),
        "divergence_steps": divergence_steps,
    }


# =============================================================================
# Heuristic interpretation helpers
# =============================================================================

def _safe_ratio(numerator, denominator):
    if denominator <= 1e-15:
        return np.nan
    return numerator / denominator


def _comparable_ratio(a, b):
    if a > 1e-15 and b > 1e-15:
        return min(a, b) / max(a, b)
    return np.nan


def compute_heuristic_tests(net_results):
    def paradox_ratio(method):
        return net_results[method]["function_over_weight_ratio"]

    def paradox_strength(method):
        result = net_results[method]
        if result["func_diversity_mean"] > 1e-15 and result["loss_std"] > 1e-20:
            return result["weight_diversity_mean"] / (result["func_diversity_mean"] * result["loss_std"])
        return 0.0

    sgd_ratio = paradox_ratio("sgd")
    norm_ratio = paradox_ratio("norm_sgd")
    muon_ratio = paradox_ratio("muon")

    t1 = bool(norm_ratio < sgd_ratio)

    muon_curve = net_results["muon"]["loss_curve_mean"]
    norm_curve = net_results["norm_sgd"]["loss_curve_mean"]
    half_idx = len(muon_curve) // 2
    muon_half = muon_curve[half_idx] if half_idx < len(muon_curve) else np.nan
    norm_half = norm_curve[half_idx] if half_idx < len(norm_curve) else np.nan
    muon_final = net_results["muon"]["loss_mean"]
    norm_final = net_results["norm_sgd"]["loss_mean"]
    half_ratio = _comparable_ratio(muon_half, norm_half)
    final_ratio = _comparable_ratio(muon_final, norm_final)
    t2_half = bool(half_ratio > 0.33) if np.isfinite(half_ratio) else False
    t2_final = bool(final_ratio > 0.33) if np.isfinite(final_ratio) else False
    t2 = bool(t2_half and t2_final)

    t3 = bool(muon_final < norm_final)

    norm_methods = ["norm_sgd", "spectral_sgd", "sign_sgd", "muon"]
    ratios = {method: paradox_ratio(method) for method in norm_methods}
    strengths = {method: paradox_strength(method) for method in norm_methods}
    best_ratio_method = min(ratios, key=lambda method: ratios[method])
    best_strength_method = max(strengths, key=lambda method: strengths[method])
    t4 = bool(best_ratio_method == "muon")

    norm_matches_paradox = bool(norm_ratio < sgd_ratio * 0.8)
    norm_matches_loss = bool(abs(norm_final - muon_final) / max(muon_final, 1e-15) < 0.5)
    spec_ratio = paradox_ratio("spectral_sgd")
    spec_loss = net_results["spectral_sgd"]["loss_mean"]
    spec_matches_paradox = bool(spec_ratio < sgd_ratio * 0.8)
    spec_matches_loss = bool(abs(spec_loss - muon_final) / max(muon_final, 1e-15) < 0.5)
    any_matches_both = (norm_matches_paradox and norm_matches_loss) or (
        spec_matches_paradox and spec_matches_loss
    )
    any_matches_paradox_only = (norm_matches_paradox or spec_matches_paradox) and not any_matches_both

    if any_matches_both:
        critical_conclusion = (
            "On this toy architecture, simple normalization can reproduce both a stronger paradox-style "
            "F/W signature and Muon-comparable loss. That still does not establish universal necessity or "
            "non-necessity of the polar factor beyond this setting."
        )
    elif any_matches_paradox_only or (norm_matches_paradox and not norm_matches_loss) or (
        spec_matches_paradox and not spec_matches_loss
    ):
        critical_conclusion = (
            "On this toy architecture, simple normalization can reproduce some paradox-style diversity "
            "signatures, but Muon's orthogonalized update still gives clearly different optimization quality."
        )
    else:
        critical_conclusion = (
            "On this toy architecture, the normalization baselines do not reproduce Muon's combination of "
            "diversity signature and loss behavior."
        )

    tests = {
        "t1": {
            "name": "Normalized SGD has lower F/W ratio than SGD",
            "passed": t1,
            "summary": f"Norm F/W={norm_ratio:.6f} vs SGD F/W={sgd_ratio:.6f}",
            "details": {
                "sgd_ratio": float(sgd_ratio),
                "norm_ratio": float(norm_ratio),
                "muon_ratio": float(muon_ratio),
            },
        },
        "t2": {
            "name": "Normalized SGD has roughly Muon-comparable mid/final mean loss",
            "passed": t2,
            "summary": (
                f"half-step ratio={half_ratio:.3f}, final-loss ratio={final_ratio:.3f} "
                f"(threshold > 0.33 at both checkpoints)"
            ),
            "details": {
                "half_index": int(half_idx),
                "muon_half_loss": float(muon_half),
                "norm_half_loss": float(norm_half),
                "half_ratio": float(half_ratio),
                "muon_final_loss": float(muon_final),
                "norm_final_loss": float(norm_final),
                "final_ratio": float(final_ratio),
            },
        },
        "t3": {
            "name": "Muon achieves lower final mean loss than normalized SGD",
            "passed": t3,
            "summary": f"Muon={muon_final:.6e}, Norm={norm_final:.6e}",
            "details": {
                "muon_final_loss": float(muon_final),
                "norm_final_loss": float(norm_final),
            },
        },
        "t4": {
            "name": "Muon has the lowest F/W ratio among the normalization-style methods",
            "passed": t4,
            "summary": (
                f"best F/W={OPTIMIZER_LABELS[best_ratio_method]}, "
                f"best ad hoc strength={OPTIMIZER_LABELS[best_strength_method]}"
            ),
            "details": {
                "ratios": {method: float(value) for method, value in ratios.items()},
                "strengths": {method: float(value) for method, value in strengths.items()},
                "best_ratio_method": best_ratio_method,
                "best_strength_method": best_strength_method,
            },
        },
    }

    return {
        "tests": tests,
        "ratios": ratios,
        "strengths": strengths,
        "best_ratio_method": best_ratio_method,
        "best_strength_method": best_strength_method,
        "critical_comparison": {
            "norm_matches_paradox": norm_matches_paradox,
            "norm_matches_loss": norm_matches_loss,
            "spec_matches_paradox": spec_matches_paradox,
            "spec_matches_loss": spec_matches_loss,
            "any_matches_both": bool(any_matches_both),
            "any_matches_paradox_only": bool(any_matches_paradox_only),
            "conclusion": critical_conclusion,
        },
        "pass_count": int(sum(1 for test in tests.values() if test["passed"])),
        "test_count": int(len(tests)),
    }


# =============================================================================
# Tabular helpers for the notebook and console summaries
# =============================================================================

def build_lr_summary_rows(results_bundle):
    rows = []
    for net_type in ARCHITECTURES:
        for method in OPTIMIZER_NAMES:
            sweep = results_bundle["lr_sweeps"][net_type][method]
            rows.append(
                {
                    "architecture": net_type,
                    "optimizer": OPTIMIZER_LABELS[method],
                    "selected_lr": float(sweep["best_lr"]),
                    "best_200_step_loss": float(sweep["best_loss"]),
                    "hit_grid_boundary": bool(sweep["hit_grid_boundary"]),
                    "boundary_side": sweep["boundary_side"] or "interior",
                    "num_candidates": int(len(sweep["candidates"])),
                }
            )
    return rows


def build_optimizer_summary_rows(results_bundle, net_type):
    rows = []
    for method in OPTIMIZER_NAMES:
        result = results_bundle["architectures"][net_type][method]
        rows.append(
            {
                "optimizer": OPTIMIZER_LABELS[method],
                "selected_lr": float(result["lr"]),
                "final_loss_mean": float(result["loss_mean"]),
                "final_loss_std": float(result["loss_std"]),
                "weight_diversity_mean": float(result["weight_diversity_mean"]),
                "func_diversity_mean": float(result["func_diversity_mean"]),
                "function_over_weight_ratio": float(result["function_over_weight_ratio"]),
                "condition_geom_mean": float(result["condition_geom_mean"]),
                "condition_geom_std": float(result["condition_geom_std"]),
                "stable_run_fraction": float(result["stable_run_fraction"]),
                "num_diverged_runs": int(result["num_diverged_runs"]),
            }
        )
    return rows


def build_paradox_metric_rows(results_bundle, net_type):
    heuristic = results_bundle["heuristic_results"][net_type]
    rows = []
    for method in OPTIMIZER_NAMES:
        result = results_bundle["architectures"][net_type][method]
        rows.append(
            {
                "optimizer": OPTIMIZER_LABELS[method],
                "weight_diversity_mean": float(result["weight_diversity_mean"]),
                "weight_diversity_std": float(result["weight_diversity_std"]),
                "func_diversity_mean": float(result["func_diversity_mean"]),
                "func_diversity_std": float(result["func_diversity_std"]),
                "function_over_weight_ratio": float(result["function_over_weight_ratio"]),
                "paradox_strength": float(heuristic["strengths"].get(method, np.nan)),
                "final_loss_std": float(result["loss_std"]),
            }
        )
    return rows


def build_heuristic_rule_rows(results_bundle, net_type):
    heuristic = results_bundle["heuristic_results"][net_type]
    rows = []
    for test_key in ("t1", "t2", "t3", "t4"):
        test = heuristic["tests"][test_key]
        rows.append(
            {
                "rule": test_key.upper(),
                "description": test["name"],
                "passed": bool(test["passed"]),
                "summary": test["summary"],
            }
        )
    return rows


def build_overall_summary(results_bundle):
    total_passes = sum(results_bundle["heuristic_results"][net_type]["pass_count"] for net_type in ARCHITECTURES)
    total_tests = sum(results_bundle["heuristic_results"][net_type]["test_count"] for net_type in ARCHITECTURES)

    both_match_both = all(
        results_bundle["heuristic_results"][net_type]["critical_comparison"]["any_matches_both"]
        for net_type in ARCHITECTURES
    )
    any_match_both = any(
        results_bundle["heuristic_results"][net_type]["critical_comparison"]["any_matches_both"]
        for net_type in ARCHITECTURES
    )
    any_paradox_only = any(
        results_bundle["heuristic_results"][net_type]["critical_comparison"]["any_matches_paradox_only"]
        for net_type in ARCHITECTURES
    )

    if both_match_both:
        interpretation = (
            "Within this toy setup, scale-normalized baselines can match Muon on both architectures. "
            "That still would not justify a universal claim about the polar factor outside this experiment."
        )
    elif any_match_both or any_paradox_only:
        interpretation = (
            "Evidence is mixed and architecture-dependent: some normalization baselines can reproduce parts of "
            "the diversity signature, but Muon still differs materially in optimization quality on at least one architecture."
        )
    else:
        interpretation = (
            "Within this toy setup, the normalization baselines do not reproduce Muon's combined behavior. "
            "This remains limited evidence about this specific experiment, not a universal theorem."
        )

    return {
        "total_passes": int(total_passes),
        "total_tests": int(total_tests),
        "interpretation": interpretation,
    }


# =============================================================================
# Plotting helpers
# =============================================================================

def _require_matplotlib():
    import matplotlib.pyplot as plt

    return plt


def _annotate_best_bar(ax, values, best_index, mode="min"):
    if best_index is None or best_index >= len(values):
        return
    best_value = values[best_index]
    if not np.isfinite(best_value):
        return
    if ax.get_yscale() == "log":
        y = best_value * 1.25
    else:
        spread = max(values) - min(values) if values else 0.0
        y = best_value + 0.05 * (spread if spread > 0 else max(abs(best_value), 1.0))
    label = "best" if mode == "min" else "best"
    ax.text(best_index, y, label, ha="center", va="bottom", fontsize=9)


def planned_figure_paths(output_dir):
    output_dir = resolve_output_dir(output_dir)
    return {
        "linear": str(output_dir / "h3_normalized_sgd_vs_muon_linear.png"),
        "relu": str(output_dir / "h3_normalized_sgd_vs_muon_relu.png"),
        "combined_summary": str(output_dir / "h3_combined_summary.png"),
    }


def plot_architecture_overview(results_bundle, net_type, save_path=None, show=False):
    plt = _require_matplotlib()

    arch_results = results_bundle["architectures"][net_type]
    heuristics = results_bundle["heuristic_results"][net_type]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax_loss, ax_div, ax_ratio, ax_strength = axes.flatten()

    for method in OPTIMIZER_NAMES:
        result = arch_results[method]
        color = OPTIMIZER_COLORS[method]
        mean_curve = result["loss_curve_mean"]
        std_curve = result["loss_curve_std"]
        x = np.arange(len(mean_curve))
        valid = np.isfinite(mean_curve)
        ax_loss.plot(x[valid], mean_curve[valid], label=OPTIMIZER_LABELS[method], color=color, linewidth=2)
        lower = np.clip(mean_curve - std_curve, 1e-18, None)
        upper = mean_curve + std_curve
        ax_loss.fill_between(x[valid], lower[valid], upper[valid], color=color, alpha=0.15)
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("Training step")
    ax_loss.set_ylabel("Mean loss across runs")
    ax_loss.set_title("Mean loss curve ± 1 std across 20 runs")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(fontsize=9)

    idx = np.arange(len(OPTIMIZER_NAMES))
    width = 0.38
    weight_vals = [arch_results[method]["weight_diversity_mean"] for method in OPTIMIZER_NAMES]
    func_vals = [arch_results[method]["func_diversity_mean"] for method in OPTIMIZER_NAMES]
    colors = [OPTIMIZER_COLORS[method] for method in OPTIMIZER_NAMES]
    ax_div.bar(idx - width / 2, weight_vals, width=width, color=colors, alpha=0.85, label="Weight diversity")
    ax_div.bar(idx + width / 2, func_vals, width=width, color=colors, alpha=0.35, label="Function diversity")
    ax_div.set_xticks(idx)
    ax_div.set_xticklabels([OPTIMIZER_LABELS[method] for method in OPTIMIZER_NAMES], rotation=25, ha="right")
    ax_div.set_yscale("log")
    ax_div.set_ylabel("Descriptive scale (log)")
    ax_div.set_title("Weight vs function diversity")
    ax_div.grid(True, axis="y", alpha=0.3)
    ax_div.legend(fontsize=9)

    ratio_vals = [arch_results[method]["function_over_weight_ratio"] for method in OPTIMIZER_NAMES]
    ratio_best_method = heuristics["best_ratio_method"]
    ratio_best_idx = OPTIMIZER_NAMES.index(ratio_best_method)
    ax_ratio.bar(idx, ratio_vals, color=colors, alpha=0.9)
    ax_ratio.set_xticks(idx)
    ax_ratio.set_xticklabels([OPTIMIZER_LABELS[method] for method in OPTIMIZER_NAMES], rotation=25, ha="right")
    ax_ratio.set_yscale("log")
    ax_ratio.set_ylabel("Function / weight diversity")
    ax_ratio.set_title("Lower F/W ratio = stronger paradox by this descriptive ratio")
    ax_ratio.grid(True, axis="y", alpha=0.3)
    _annotate_best_bar(ax_ratio, ratio_vals, ratio_best_idx, mode="min")

    strength_vals = [heuristics["strengths"].get(method, np.nan) for method in OPTIMIZER_NAMES]
    strength_best_method = heuristics["best_strength_method"]
    strength_best_idx = OPTIMIZER_NAMES.index(strength_best_method)
    ax_strength.bar(idx, strength_vals, color=colors, alpha=0.9)
    ax_strength.set_xticks(idx)
    ax_strength.set_xticklabels([OPTIMIZER_LABELS[method] for method in OPTIMIZER_NAMES], rotation=25, ha="right")
    ax_strength.set_yscale("log")
    ax_strength.set_ylabel("Weight / (function × loss std)")
    ax_strength.set_title("Ad hoc paradox-strength heuristic (higher = stronger)")
    ax_strength.grid(True, axis="y", alpha=0.3)
    _annotate_best_bar(ax_strength, strength_vals, strength_best_idx, mode="max")

    fig.suptitle(
        f"{EXPERIMENT_TITLE}: {net_type.upper()} architecture overview",
        fontsize=16,
    )
    fig.text(
        0.5,
        0.01,
        "Caveat: diversity metrics are descriptive summaries from 20 runs; the pairwise distances are not independent samples.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_combined_summary(results_bundle, save_path=None, show=False):
    plt = _require_matplotlib()

    fig, axes = plt.subplots(len(ARCHITECTURES), 3, figsize=(18, 10), squeeze=False)
    metric_specs = [
        ("loss_mean", "Final loss mean", "log"),
        ("function_over_weight_ratio", "F/W ratio", "log"),
        ("condition_geom_mean", "Geom. mean condition number", "log"),
    ]

    for row_idx, net_type in enumerate(ARCHITECTURES):
        arch_results = results_bundle["architectures"][net_type]
        colors = [OPTIMIZER_COLORS[method] for method in OPTIMIZER_NAMES]
        labels = [OPTIMIZER_LABELS[method] for method in OPTIMIZER_NAMES]
        idx = np.arange(len(OPTIMIZER_NAMES))

        for col_idx, (metric_key, title, yscale) in enumerate(metric_specs):
            ax = axes[row_idx, col_idx]
            values = [arch_results[method][metric_key] for method in OPTIMIZER_NAMES]
            ax.bar(idx, values, color=colors, alpha=0.9)
            if metric_key == "loss_mean":
                errors = [arch_results[method]["loss_std"] for method in OPTIMIZER_NAMES]
                ax.errorbar(idx, values, yerr=errors, fmt="none", ecolor="black", elinewidth=1, capsize=3)
            ax.set_xticks(idx)
            ax.set_xticklabels(labels, rotation=25, ha="right")
            if yscale == "log":
                ax.set_yscale("log")
            ax.set_title(f"{net_type.upper()}: {title}")
            ax.grid(True, axis="y", alpha=0.3)
            if col_idx == 0:
                ax.set_ylabel("Value")

    fig.suptitle(f"{EXPERIMENT_TITLE}: cross-architecture summary", fontsize=16)
    fig.text(
        0.5,
        0.01,
        "Loss bars include ±1 std across runs. Conditioning is the geometric mean over per-layer condition numbers within each run, then averaged across runs.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def save_all_figures(results_bundle, output_dir, verbose=True):
    figure_paths = planned_figure_paths(output_dir)
    try:
        for net_type in ARCHITECTURES:
            fig = plot_architecture_overview(results_bundle, net_type, save_path=figure_paths[net_type], show=False)
            fig.canvas.draw_idle()
            fig.clf()
        summary_fig = plot_combined_summary(
            results_bundle,
            save_path=figure_paths["combined_summary"],
            show=False,
        )
        summary_fig.canvas.draw_idle()
        summary_fig.clf()
    except ImportError as exc:
        if verbose:
            print(f"\n[warning] matplotlib unavailable; skipping figure generation: {exc}")
        return {"matplotlib_unavailable": str(exc), **figure_paths}

    if verbose:
        print("\nSaved figures:")
        for key, path in figure_paths.items():
            print(f"  - {key}: {path}")
    return figure_paths


# =============================================================================
# Execution and reporting
# =============================================================================

def resolve_output_dir(output_dir=None):
    resolved = DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def get_config(output_dir=None):
    resolved_output_dir = resolve_output_dir(output_dir)
    return {
        "experiment_id": EXPERIMENT_ID,
        "title": EXPERIMENT_TITLE,
        "scope_note": SCOPE_NOTE,
        "dim": DIM,
        "num_layers": NUM_LAYERS,
        "num_steps": NUM_STEPS,
        "batch_size": BATCH_SIZE,
        "momentum": MOMENTUM,
        "ns_iters": NS_ITERS,
        "num_independent_runs": NUM_INDEPENDENT_RUNS,
        "num_test_inputs": NUM_TEST_INPUTS,
        "lr_sweep_steps": LR_SWEEP_STEPS,
        "data_seed": DATA_SEED,
        "run_seed_start": RUN_SEED_START,
        "run_seeds": list(range(RUN_SEED_START, RUN_SEED_START + NUM_INDEPENDENT_RUNS)),
        "architectures": list(ARCHITECTURES),
        "optimizers": list(OPTIMIZER_NAMES),
        "lr_candidate_grids": {key: list(values) for key, values in LR_CANDIDATE_GRIDS.items()},
        "script_path": str(SCRIPT_DIR / "run_experiment.py"),
        "default_output_dir": str(DEFAULT_OUTPUT_DIR),
        "resolved_output_dir": str(resolved_output_dir),
    }


def _get_git_commit():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=SCRIPT_DIR,
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
        )
    except Exception:
        return None


def _print_header(output_dir):
    print("=" * 110)
    print(EXPERIMENT_TITLE)
    print("=" * 110)
    print("Question: does removing update scale alone reproduce Muon-like diversity signatures and loss behavior?")
    print(f"Scope: {SCOPE_NOTE}")
    print("Reporting note: heuristic comparison rules and descriptive spread, not formal hypothesis tests.")
    print(f"Output directory: {output_dir}")
    print("=" * 110)


def _print_architecture_summary(results_bundle, net_type):
    print(f"\n{'#' * 110}")
    print(f"ARCHITECTURE: {net_type.upper()}")
    print(f"{'#' * 110}")

    print("\nLearning-rate sweep (200-step final loss from one fixed initialization):")
    for method in OPTIMIZER_NAMES:
        sweep = results_bundle["lr_sweeps"][net_type][method]
        boundary_note = ""
        if sweep["hit_grid_boundary"]:
            boundary_note = f"  [grid-boundary warning: best is {sweep['boundary_side']} candidate]"
        print(
            f"  {OPTIMIZER_LABELS[method]:<30} best_lr={sweep['best_lr']:.5f}  "
            f"best_200_step_loss={sweep['best_loss']:.6e}{boundary_note}"
        )

    print("\nConvergence basin summary (20 runs, 500 steps):")
    print(
        f"  {'Optimizer':<30} | {'Final loss mean':>14} | {'Loss std':>12} | {'F/W ratio':>10} | {'Cond geom':>10} | {'Stable':>8}"
    )
    print(f"  {'-' * 102}")
    for row in build_optimizer_summary_rows(results_bundle, net_type):
        print(
            f"  {row['optimizer']:<30} | {row['final_loss_mean']:>14.6e} | {row['final_loss_std']:>12.6e} | "
            f"{row['function_over_weight_ratio']:>10.6f} | {row['condition_geom_mean']:>10.2f} | "
            f"{row['stable_run_fraction']:>7.2%}"
        )

    heuristics = results_bundle["heuristic_results"][net_type]
    print("\nHeuristic comparison rules:")
    for test_key in ("t1", "t2", "t3", "t4"):
        test = heuristics["tests"][test_key]
        print(f"  {test_key.upper()}: {'PASS' if test['passed'] else 'FAIL'}  {test['name']}")
        print(f"      {test['summary']}")
    print("\nCurrent interpretation:")
    print(f"  {heuristics['critical_comparison']['conclusion']}")


def _print_overall_summary(results_bundle):
    overall = results_bundle["overall_summary"]
    print(f"\n{'=' * 110}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 110}")
    print(
        f"Heuristic rule passes: {overall['total_passes']}/{overall['total_tests']} across both architectures."
    )
    print(overall["interpretation"])
    print("Caveat: this experiment does not directly test statistical significance, gauge coordinates, or universal necessity of the polar factor.")


def run_full_experiment(output_dir=None, make_plots=True, verbose=True):
    output_dir = resolve_output_dir(output_dir)
    config = get_config(output_dir=output_dir)
    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "title": EXPERIMENT_TITLE,
        "scope_note": SCOPE_NOTE,
        "script_path": str(SCRIPT_DIR / "run_experiment.py"),
        "output_dir": str(output_dir),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "git_commit": _get_git_commit(),
        "data_seed": DATA_SEED,
        "run_seeds": list(range(RUN_SEED_START, RUN_SEED_START + NUM_INDEPENDENT_RUNS)),
    }

    if verbose:
        _print_header(output_dir)

    experiment_start = time.time()
    architectures = {}
    lr_sweeps = {}
    timings = {net_type: {"lr_sweep_seconds": 0.0, "basin_seconds": {}, "total_seconds": 0.0} for net_type in ARCHITECTURES}

    for net_type in ARCHITECTURES:
        arch_start = time.time()
        lr_start = time.time()
        lr_sweeps[net_type] = {}
        selected_lrs = {}
        for method in OPTIMIZER_NAMES:
            sweep = find_best_lr(method, net_type, num_steps=LR_SWEEP_STEPS)
            lr_sweeps[net_type][method] = sweep
            selected_lrs[method] = sweep["best_lr"]
        timings[net_type]["lr_sweep_seconds"] = time.time() - lr_start

        architectures[net_type] = {}
        for method in OPTIMIZER_NAMES:
            t0 = time.time()
            result = measure_convergence_basin(method, selected_lrs[method], net_type)
            result["lr_sweep"] = lr_sweeps[net_type][method]
            architectures[net_type][method] = result
            timings[net_type]["basin_seconds"][method] = time.time() - t0

        timings[net_type]["total_seconds"] = time.time() - arch_start

    heuristic_results = {
        net_type: compute_heuristic_tests(architectures[net_type]) for net_type in ARCHITECTURES
    }

    results_bundle = {
        "metadata": metadata,
        "config": config,
        "lr_sweeps": lr_sweeps,
        "selected_lrs": {
            net_type: {method: architectures[net_type][method]["lr"] for method in OPTIMIZER_NAMES}
            for net_type in ARCHITECTURES
        },
        "architectures": architectures,
        "heuristic_results": heuristic_results,
        "timings": timings,
        "figure_paths": planned_figure_paths(output_dir),
    }
    results_bundle["overall_summary"] = build_overall_summary(results_bundle)
    results_bundle["metadata"]["runtime_seconds"] = time.time() - experiment_start

    if verbose:
        for net_type in ARCHITECTURES:
            _print_architecture_summary(results_bundle, net_type)

    if make_plots:
        results_bundle["figure_paths"] = save_all_figures(results_bundle, output_dir, verbose=verbose)

    if verbose:
        _print_overall_summary(results_bundle)

    return results_bundle


run_experiment = run_full_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run H3 normalized-SGD-vs-Muon toy basin experiment.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for saved figures. Defaults to the experiment directory.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip figure generation and only compute/print results.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console reporting and only run the computation.",
    )
    args = parser.parse_args(argv)

    run_full_experiment(
        output_dir=args.output_dir,
        make_plots=not args.no_plots,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
