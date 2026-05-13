#!/usr/bin/env python3
"""
MUON TOY BENCHMARK: larger weight-space divergence with lower function-space divergence proxies
===============================================================================================

This script preserves the original Muon-vs-SGD toy benchmark while making it
import-safe and notebook-friendly.

Scope and calibration:
  - Two small synthetic tasks are compared: a 4-layer deep linear network and a
    4-layer ReLU network.
  - Face 1 measures finite-time divergence proxies in weight space, function
    space, and loss space after small initialization perturbations.
  - Face 2 measures independent-run diversity after a fixed training budget.
  - Function-space quantities are evaluated on a fixed test batch `X_test` and
    therefore serve as toy proxy metrics rather than proofs of contraction.
  - Lower function/weight divergence ratios are consistent with more
    function-preserving weight variation, but they do not by themselves prove a
    gauge decomposition or establish mechanism.
"""

from __future__ import annotations

import argparse
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
NETWORK_TYPES = ("linear", "relu")
DEFAULT_SGD_LR_CANDIDATES = [0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.001]

DEFAULT_CONFIG = {
    "DIM": 32,
    "NUM_LAYERS": 4,
    "NUM_STEPS": 200,
    "FACE2_STEPS": 500,
    "BATCH_SIZE": 64,
    "LR_MUON": 0.005,
    "MOMENTUM": 0.9,
    "NS_ITERS": 5,
    "EPSILON": 0.001,
    "NUM_PERTURBATIONS": 20,
    "NUM_TEST_INPUTS": 50,
    "NUM_INDEPENDENT_RUNS": 20,
    "TARGET_SCALE": 0.5,
    "DATA_SCALE": 0.3,
    "SGD_LR_CANDIDATES": DEFAULT_SGD_LR_CANDIDATES,
    "LR_SEARCH_STEPS": 80,
    "LR_SEARCH_BLOWUP_FACTOR": 50.0,
    "LOSS_BLOWUP_THRESHOLD": 1e10,
}

SMOKE_TEST_CONFIG = {
    "NUM_STEPS": 8,
    "FACE2_STEPS": 12,
    "BATCH_SIZE": 16,
    "NUM_PERTURBATIONS": 2,
    "NUM_TEST_INPUTS": 12,
    "NUM_INDEPENDENT_RUNS": 3,
}

DEFAULT_SEEDS = {
    "global_seed": 42,
    "data_seed": 42,
    "base_init_seed": 42,
    "perturb_seed_base": 100,
    "run_seed_base": 1000,
}

CURRENT_CONFIG = {}
CURRENT_SEEDS = {}

DIM = DEFAULT_CONFIG["DIM"]
NUM_LAYERS = DEFAULT_CONFIG["NUM_LAYERS"]
NUM_STEPS = DEFAULT_CONFIG["NUM_STEPS"]
FACE2_STEPS = DEFAULT_CONFIG["FACE2_STEPS"]
BATCH_SIZE = DEFAULT_CONFIG["BATCH_SIZE"]
LR_MUON = DEFAULT_CONFIG["LR_MUON"]
MOMENTUM = DEFAULT_CONFIG["MOMENTUM"]
NS_ITERS = DEFAULT_CONFIG["NS_ITERS"]
EPSILON = DEFAULT_CONFIG["EPSILON"]
NUM_PERTURBATIONS = DEFAULT_CONFIG["NUM_PERTURBATIONS"]
NUM_TEST_INPUTS = DEFAULT_CONFIG["NUM_TEST_INPUTS"]
NUM_INDEPENDENT_RUNS = DEFAULT_CONFIG["NUM_INDEPENDENT_RUNS"]
TARGET_SCALE = DEFAULT_CONFIG["TARGET_SCALE"]
DATA_SCALE = DEFAULT_CONFIG["DATA_SCALE"]

W_target = None
X_data = None
X_test = None


def _float(value):
    return float(value) if np.isfinite(value) else float(value)


def _safe_ratio(numerator, denominator):
    return numerator / denominator if denominator > 1e-15 else np.nan


def _merge_config(config_overrides=None):
    config = deepcopy(DEFAULT_CONFIG)
    if config_overrides:
        for key, value in config_overrides.items():
            config[key] = value
    config["SGD_LR_CANDIDATES"] = list(config["SGD_LR_CANDIDATES"])
    return config


def _merge_seeds(seed_overrides=None):
    seeds = deepcopy(DEFAULT_SEEDS)
    if seed_overrides:
        for key, value in seed_overrides.items():
            seeds[key] = value
    return seeds


def _apply_context(config, seeds):
    global CURRENT_CONFIG, CURRENT_SEEDS
    global DIM, NUM_LAYERS, NUM_STEPS, FACE2_STEPS, BATCH_SIZE
    global LR_MUON, MOMENTUM, NS_ITERS, EPSILON, NUM_PERTURBATIONS
    global NUM_TEST_INPUTS, NUM_INDEPENDENT_RUNS, TARGET_SCALE, DATA_SCALE
    global W_target, X_data, X_test

    CURRENT_CONFIG = deepcopy(config)
    CURRENT_SEEDS = dict(seeds)

    DIM = config["DIM"]
    NUM_LAYERS = config["NUM_LAYERS"]
    NUM_STEPS = config["NUM_STEPS"]
    FACE2_STEPS = config["FACE2_STEPS"]
    BATCH_SIZE = config["BATCH_SIZE"]
    LR_MUON = config["LR_MUON"]
    MOMENTUM = config["MOMENTUM"]
    NS_ITERS = config["NS_ITERS"]
    EPSILON = config["EPSILON"]
    NUM_PERTURBATIONS = config["NUM_PERTURBATIONS"]
    NUM_TEST_INPUTS = config["NUM_TEST_INPUTS"]
    NUM_INDEPENDENT_RUNS = config["NUM_INDEPENDENT_RUNS"]
    TARGET_SCALE = config["TARGET_SCALE"]
    DATA_SCALE = config["DATA_SCALE"]

    np.random.seed(seeds["global_seed"])
    rng = np.random.RandomState(seeds["data_seed"])
    W_target = rng.randn(DIM, DIM) * TARGET_SCALE
    X_data = rng.randn(DIM, BATCH_SIZE) * DATA_SCALE
    X_test = rng.randn(DIM, NUM_TEST_INPUTS) * DATA_SCALE


def prepare_experiment(config_overrides=None, seed_overrides=None):
    """Merge config/seeds, regenerate fixed toy data, and update module globals."""
    config = _merge_config(config_overrides)
    seeds = _merge_seeds(seed_overrides)
    _apply_context(config, seeds)
    return deepcopy(config), deepcopy(seeds)


# =============================================================================
# NETWORK DEFINITIONS
# =============================================================================


def init_weights(num_layers, seed=42):
    """Initialize layers near identity for stability."""
    rng = np.random.RandomState(seed)
    weights = []
    for _ in range(num_layers):
        W = np.eye(DIM) + rng.randn(DIM, DIM) * 0.1
        weights.append(W.copy())
    return weights


# ---- DEEP LINEAR NET ----


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


# ---- RELU NET ----


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
# MUON CORE: NEWTON-SCHULZ ORTHOGONALIZATION
# =============================================================================


def newton_schulz_orthogonalize(G, num_iters=None):
    num_iters = NS_ITERS if num_iters is None else num_iters
    norm = np.linalg.norm(G, ord="fro")
    if norm < 1e-12:
        return G
    X = G / norm
    for _ in range(num_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


# =============================================================================
# OPTIMIZER STEP FUNCTIONS
# =============================================================================


def sgd_step(weights, velocities, grads, lr):
    for i in range(len(weights)):
        velocities[i] = MOMENTUM * velocities[i] + grads[i]
        weights[i] = weights[i] - lr * velocities[i]
    return weights, velocities


def muon_step(weights, velocities, grads, lr):
    for i in range(len(weights)):
        ortho_grad = newton_schulz_orthogonalize(grads[i])
        velocities[i] = MOMENTUM * velocities[i] + ortho_grad
        weights[i] = weights[i] - lr * velocities[i]
    return weights, velocities


# =============================================================================
# LEARNING RATE FINDER
# =============================================================================


def find_stable_lr_sgd(net_type, verbose=True):
    compute_loss_fn = compute_loss_linear if net_type == "linear" else compute_loss_relu
    compute_grad_fn = compute_gradients_linear if net_type == "linear" else compute_gradients_relu

    trials = []
    chosen_lr = CURRENT_CONFIG["SGD_LR_CANDIDATES"][-1]

    for lr in CURRENT_CONFIG["SGD_LR_CANDIDATES"]:
        weights = init_weights(NUM_LAYERS, seed=CURRENT_SEEDS["base_init_seed"])
        velocities = [np.zeros((DIM, DIM)) for _ in range(NUM_LAYERS)]
        initial_loss = compute_loss_fn(weights, X_data, W_target)
        stable = True
        final_loss = initial_loss
        steps_completed = 0

        for step in range(CURRENT_CONFIG["LR_SEARCH_STEPS"]):
            grads = compute_grad_fn(weights, X_data, W_target)
            weights, velocities = sgd_step(weights, velocities, grads, lr)
            final_loss = compute_loss_fn(weights, X_data, W_target)
            steps_completed = step + 1
            if np.isnan(final_loss) or final_loss > initial_loss * CURRENT_CONFIG["LR_SEARCH_BLOWUP_FACTOR"]:
                stable = False
                break

        trial = {
            "lr": float(lr),
            "stable": bool(stable),
            "initial_loss": float(initial_loss),
            "final_loss": float(final_loss),
            "steps_completed": int(steps_completed),
            "criterion": (
                f"stable if no NaN and loss <= {CURRENT_CONFIG['LR_SEARCH_BLOWUP_FACTOR']}x initial "
                f"for {CURRENT_CONFIG['LR_SEARCH_STEPS']} steps"
            ),
        }
        trials.append(trial)

        if stable:
            chosen_lr = float(lr)
            break

    info = {
        "net_type": net_type,
        "candidates": [float(v) for v in CURRENT_CONFIG["SGD_LR_CANDIDATES"]],
        "trials": trials,
        "chosen_lr": float(chosen_lr),
        "criterion": trials[0]["criterion"] if trials else "",
    }

    if verbose:
        print(f"  SGD lr selection for {net_type.upper()}: chosen {chosen_lr}")

    return float(chosen_lr), info


# =============================================================================
# TRAJECTORY ENGINE
# =============================================================================


def run_trajectory(
    weights_init,
    optimizer,
    lr,
    num_steps,
    net_type,
    record_weight_snapshots=True,
    record_function_outputs=True,
):
    """
    Run one optimizer trajectory from `weights_init`.

    Returns a dictionary containing losses, final state, and optionally the full
    weight/function trajectories.
    """
    compute_loss_fn = compute_loss_linear if net_type == "linear" else compute_loss_relu
    compute_grad_fn = compute_gradients_linear if net_type == "linear" else compute_gradients_relu
    forward_fn = forward_linear if net_type == "linear" else forward_relu

    weights = [w.copy() for w in weights_init]
    velocities = [np.zeros_like(w) for w in weights]

    weight_snapshots = [[w.copy() for w in weights]] if record_weight_snapshots else None
    function_outputs = [forward_fn(weights, X_test).copy()] if record_function_outputs else None
    losses = [compute_loss_fn(weights, X_data, W_target)]

    stopped_early = False
    stop_reason = "completed"
    steps_executed = 0

    for step in range(num_steps):
        grads = compute_grad_fn(weights, X_data, W_target)
        if optimizer == "sgd":
            weights, velocities = sgd_step(weights, velocities, grads, lr)
        elif optimizer == "muon":
            weights, velocities = muon_step(weights, velocities, grads, lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        current_loss = compute_loss_fn(weights, X_data, W_target)
        current_output = forward_fn(weights, X_test).copy() if record_function_outputs else None

        if record_weight_snapshots:
            weight_snapshots.append([w.copy() for w in weights])
        if record_function_outputs:
            function_outputs.append(current_output)
        losses.append(current_loss)
        steps_executed = step + 1

        if np.isnan(current_loss) or current_loss > CURRENT_CONFIG["LOSS_BLOWUP_THRESHOLD"]:
            stopped_early = True
            stop_reason = "nan_or_blowup_loss"
            remaining = num_steps - step - 1
            if record_weight_snapshots:
                for _ in range(remaining):
                    weight_snapshots.append([w.copy() for w in weights])
            if record_function_outputs:
                final_output = function_outputs[-1].copy()
                for _ in range(remaining):
                    function_outputs.append(final_output.copy())
            losses.extend([current_loss] * remaining)
            break

    final_output = function_outputs[-1].copy() if record_function_outputs else forward_fn(weights, X_test).copy()

    return {
        "optimizer": optimizer,
        "net_type": net_type,
        "lr": float(lr),
        "num_steps_requested": int(num_steps),
        "steps_executed": int(steps_executed),
        "stopped_early": bool(stopped_early),
        "stop_reason": stop_reason,
        "losses": np.asarray(losses, dtype=float),
        "weight_snapshots": weight_snapshots,
        "function_outputs": function_outputs,
        "final_weights": [w.copy() for w in weights],
        "final_output": final_output,
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
    }


# =============================================================================
# DIVERGENCE MEASUREMENT
# =============================================================================


def compute_weight_divergence(snap_a, snap_b):
    T = min(len(snap_a), len(snap_b))
    distances = np.zeros(T)
    for t in range(T):
        d_sq = 0.0
        for i in range(len(snap_a[t])):
            d_sq += np.linalg.norm(snap_a[t][i] - snap_b[t][i], "fro") ** 2
        distances[t] = np.sqrt(d_sq)
    return distances


def compute_function_divergence(func_a, func_b):
    T = min(len(func_a), len(func_b))
    x_norm = np.linalg.norm(X_test, "fro")
    distances = np.zeros(T)
    for t in range(T):
        distances[t] = np.linalg.norm(func_a[t] - func_b[t], "fro") / x_norm
    return distances


def compute_loss_divergence(loss_a, loss_b):
    T = min(len(loss_a), len(loss_b))
    return np.abs(loss_a[:T] - loss_b[:T])


def compute_lyapunov(d_series, N):
    d0 = d_series[0]
    dN = d_series[min(N, len(d_series) - 1)]
    if d0 > 1e-15 and dN > 1e-15:
        return (1.0 / N) * np.log(dN / d0)
    if dN < 1e-15:
        return -np.inf
    return np.nan


# =============================================================================
# FACE 1: PERTURBATION DIVERGENCE ANALYSIS
# =============================================================================


def measure_perturbation_lyapunov(net_type, lr_sgd, lr_muon, num_pert, base_init_seed=None, seed_base=None, verbose=True):
    """
    Measure finite-time divergence proxies from small initialization perturbations.

    Tracked quantities:
      - weight-space divergence
      - function-space divergence on fixed X_test
      - loss-space divergence
    """
    base_init_seed = CURRENT_SEEDS["base_init_seed"] if base_init_seed is None else base_init_seed
    seed_base = CURRENT_SEEDS["perturb_seed_base"] if seed_base is None else seed_base

    if verbose:
        print(f"\n  [FACE 1] Perturbation divergence analysis for {net_type.upper()} net")

    weights_base = init_weights(NUM_LAYERS, seed=base_init_seed)

    sgd_base = run_trajectory(weights_base, "sgd", lr_sgd, NUM_STEPS, net_type)
    muon_base = run_trajectory(weights_base, "muon", lr_muon, NUM_STEPS, net_type)

    if verbose:
        print(f"    SGD  final loss: {sgd_base['final_loss']:.6e}")
        print(f"    Muon final loss: {muon_base['final_loss']:.6e}")

    aggregate = {
        "sgd": {"lyap_w": [], "lyap_f": [], "lyap_l": [], "d_w": [], "d_f": [], "d_l": []},
        "muon": {"lyap_w": [], "lyap_f": [], "lyap_l": [], "d_w": [], "d_f": [], "d_l": []},
    }
    perturbations = []

    for p in range(num_pert):
        rng = np.random.RandomState(seed_base + p)
        weights_pert = []
        for layer_idx in range(NUM_LAYERS):
            delta_W = rng.randn(DIM, DIM)
            delta_W = delta_W / np.linalg.norm(delta_W, "fro")
            weights_pert.append(weights_base[layer_idx] + EPSILON * delta_W)

        perturbation_result = {
            "perturbation_index": int(p),
            "seed": int(seed_base + p),
            "epsilon": float(EPSILON),
        }

        for opt, lr, base_run in [
            ("sgd", lr_sgd, sgd_base),
            ("muon", lr_muon, muon_base),
        ]:
            pert_run = run_trajectory(weights_pert, opt, lr, NUM_STEPS, net_type)

            d_w = compute_weight_divergence(base_run["weight_snapshots"], pert_run["weight_snapshots"])
            d_f = compute_function_divergence(base_run["function_outputs"], pert_run["function_outputs"])
            d_l = compute_loss_divergence(base_run["losses"], pert_run["losses"])

            lyap_w = compute_lyapunov(d_w, NUM_STEPS)
            lyap_f = compute_lyapunov(d_f, NUM_STEPS)
            lyap_l = compute_lyapunov(d_l, NUM_STEPS)

            aggregate[opt]["lyap_w"].append(lyap_w)
            aggregate[opt]["lyap_f"].append(lyap_f)
            aggregate[opt]["lyap_l"].append(lyap_l)
            aggregate[opt]["d_w"].append(d_w)
            aggregate[opt]["d_f"].append(d_f)
            aggregate[opt]["d_l"].append(d_l)

            perturbation_result[opt] = {
                "loss_trajectory": pert_run["losses"],
                "final_loss": float(pert_run["final_loss"]),
                "steps_executed": int(pert_run["steps_executed"]),
                "stopped_early": bool(pert_run["stopped_early"]),
                "lyap_w": float(lyap_w),
                "lyap_f": float(lyap_f),
                "lyap_l": float(lyap_l),
                "d_w": d_w,
                "d_f": d_f,
                "d_l": d_l,
            }

        perturbations.append(perturbation_result)

        if verbose and (p + 1) % 5 == 0:
            print(f"    Completed {p + 1}/{num_pert} perturbations", flush=True)

    summary = {"time_axis": np.arange(NUM_STEPS + 1)}
    for opt in ["sgd", "muon"]:
        for metric in ["lyap_w", "lyap_f", "lyap_l"]:
            arr = np.asarray(aggregate[opt][metric], dtype=float)
            valid = arr[np.isfinite(arr)]
            summary[f"{opt}_{metric}_all"] = arr
            summary[f"{opt}_{metric}_mean"] = float(np.mean(valid)) if len(valid) > 0 else np.nan
            summary[f"{opt}_{metric}_std"] = float(np.std(valid)) if len(valid) > 0 else np.nan
        for metric in ["d_w", "d_f", "d_l"]:
            summary[f"{opt}_{metric}_all"] = aggregate[opt][metric]
            summary[f"{opt}_{metric}_mean_traj"] = np.mean(aggregate[opt][metric], axis=0)

    for opt in ["sgd", "muon"]:
        ratios_over_time = []
        for p in range(num_pert):
            d_w = aggregate[opt]["d_w"][p]
            d_f = aggregate[opt]["d_f"][p]
            ratio = np.zeros_like(d_w)
            for t in range(len(d_w)):
                ratio[t] = d_f[t] / d_w[t] if d_w[t] > 1e-15 else np.nan
            ratios_over_time.append(ratio)
        summary[f"{opt}_ratio_f_w_all"] = ratios_over_time
        final_ratios = [r[-1] for r in ratios_over_time if np.isfinite(r[-1])]
        summary[f"{opt}_ratio_f_w_final_mean"] = float(np.mean(final_ratios)) if final_ratios else np.nan
        summary[f"{opt}_ratio_f_w_final_std"] = float(np.std(final_ratios)) if final_ratios else np.nan
        summary[f"{opt}_ratio_f_w_mean_traj"] = np.nanmean(np.asarray(ratios_over_time), axis=0)

    return {
        "metadata": {
            "net_type": net_type,
            "num_perturbations": int(num_pert),
            "num_steps": int(NUM_STEPS),
            "epsilon": float(EPSILON),
            "base_init_seed": int(base_init_seed),
            "perturb_seed_base": int(seed_base),
            "function_metric_note": "Function-space distances are evaluated on fixed X_test.",
        },
        "base": {
            "sgd": {
                "loss_trajectory": sgd_base["losses"],
                "final_loss": float(sgd_base["final_loss"]),
                "steps_executed": int(sgd_base["steps_executed"]),
                "stopped_early": bool(sgd_base["stopped_early"]),
            },
            "muon": {
                "loss_trajectory": muon_base["losses"],
                "final_loss": float(muon_base["final_loss"]),
                "steps_executed": int(muon_base["steps_executed"]),
                "stopped_early": bool(muon_base["stopped_early"]),
            },
        },
        "perturbations": perturbations,
        "summary": summary,
    }


# =============================================================================
# FACE 2: INDEPENDENT-RUN DIVERSITY AFTER FIXED TRAINING
# =============================================================================


def measure_convergence_basin(net_type, lr_sgd, lr_muon, num_runs, num_steps=None, run_seed_base=None, verbose=True):
    """
    Run many independent initializations for a fixed number of steps and measure:
      - pairwise weight diversity
      - pairwise function diversity on fixed X_test
      - spread of final losses

    This is a fixed-budget comparison, not a convergence proof.
    """
    num_steps = FACE2_STEPS if num_steps is None else num_steps
    run_seed_base = CURRENT_SEEDS["run_seed_base"] if run_seed_base is None else run_seed_base

    if verbose:
        print(f"\n  [FACE 2] Independent-run diversity for {net_type.upper()} net ({num_runs} runs, {num_steps} steps)")

    results = {"metadata": {"net_type": net_type, "num_runs": int(num_runs), "num_steps": int(num_steps), "run_seed_base": int(run_seed_base)}}

    for opt_name, lr in [("sgd", lr_sgd), ("muon", lr_muon)]:
        final_weights_list = []
        final_functions = []
        final_losses = []
        run_records = []

        for run_idx in range(num_runs):
            init_seed = run_seed_base + run_idx
            weights_init = init_weights(NUM_LAYERS, seed=init_seed)
            run_data = run_trajectory(
                weights_init,
                opt_name,
                lr,
                num_steps,
                net_type,
                record_weight_snapshots=False,
                record_function_outputs=False,
            )

            final_weights = [w.copy() for w in run_data["final_weights"]]
            final_output = run_data["final_output"].copy()
            final_loss = float(run_data["final_loss"])

            final_weights_list.append(final_weights)
            final_functions.append(final_output)
            final_losses.append(final_loss)
            run_records.append(
                {
                    "run_idx": int(run_idx),
                    "init_seed": int(init_seed),
                    "loss_trajectory": run_data["losses"],
                    "final_loss": final_loss,
                    "steps_executed": int(run_data["steps_executed"]),
                    "stopped_early": bool(run_data["stopped_early"]),
                    "stop_reason": run_data["stop_reason"],
                }
            )

        n = len(final_weights_list)
        weight_dists = []
        func_dists = []
        x_norm = np.linalg.norm(X_test, "fro")
        for i in range(n):
            for j in range(i + 1, n):
                d_w = 0.0
                for k in range(NUM_LAYERS):
                    d_w += np.linalg.norm(final_weights_list[i][k] - final_weights_list[j][k], "fro") ** 2
                weight_dists.append(np.sqrt(d_w))
                d_f = np.linalg.norm(final_functions[i] - final_functions[j], "fro") / x_norm
                func_dists.append(d_f)

        weight_dists = np.asarray(weight_dists, dtype=float)
        func_dists = np.asarray(func_dists, dtype=float)
        final_losses = np.asarray(final_losses, dtype=float)

        results[opt_name] = {
            "runs": run_records,
            "weight_diversity_mean": float(np.mean(weight_dists)) if len(weight_dists) else np.nan,
            "weight_diversity_std": float(np.std(weight_dists)) if len(weight_dists) else np.nan,
            "func_diversity_mean": float(np.mean(func_dists)) if len(func_dists) else np.nan,
            "func_diversity_std": float(np.std(func_dists)) if len(func_dists) else np.nan,
            "loss_mean": float(np.mean(final_losses)) if len(final_losses) else np.nan,
            "loss_std": float(np.std(final_losses)) if len(final_losses) else np.nan,
            "losses": final_losses,
            "weight_dists": weight_dists,
            "func_dists": func_dists,
            "pairwise_count": int(len(weight_dists)),
        }

        if verbose:
            print(
                f"    {opt_name.upper()}: loss={np.mean(final_losses):.6e} +/- {np.std(final_losses):.6e}, "
                f"d_weight={np.mean(weight_dists):.4f}, d_func={np.mean(func_dists):.6f}"
            )

    return results


# =============================================================================
# HEURISTIC SUMMARY
# =============================================================================


def evaluate_heuristics(results):
    total_pass = 0
    total_tests = 0
    tests_by_net = {}

    for net_type in NETWORK_TYPES:
        face1 = results["networks"][net_type]["face1"]["summary"]
        face2 = results["networks"][net_type]["face2"]

        lw_s = face1["sgd_lyap_w_mean"]
        lw_m = face1["muon_lyap_w_mean"]
        lf_s = face1["sgd_lyap_f_mean"]
        lf_m = face1["muon_lyap_f_mean"]
        ll_s = face1["sgd_lyap_l_mean"]
        ll_m = face1["muon_lyap_l_mean"]
        rf_s = face1["sgd_ratio_f_w_final_mean"]
        rf_m = face1["muon_ratio_f_w_final_mean"]

        wd_s = face2["sgd"]["weight_diversity_mean"]
        wd_m = face2["muon"]["weight_diversity_mean"]
        fd_s = face2["sgd"]["func_diversity_mean"]
        fd_m = face2["muon"]["func_diversity_mean"]
        ls_s = face2["sgd"]["loss_std"]
        ls_m = face2["muon"]["loss_std"]
        basin_ratio_s = _safe_ratio(fd_s, wd_s)
        basin_ratio_m = _safe_ratio(fd_m, wd_m)

        t1 = bool(lw_m > lw_s)
        t2 = bool(rf_m < rf_s)
        t3 = bool(wd_m > wd_s)
        t4 = bool(fd_m < fd_s)
        t5 = bool(basin_ratio_m < basin_ratio_s)
        t6_lyap = bool(ll_m < ll_s)
        t6_basin = bool(ls_m < ls_s)
        t6 = bool(t6_lyap or t6_basin)

        tests = {
            "T1": {
                "pass": t1,
                "description": "Muon has larger finite-time weight-space Lyapunov proxy than SGD.",
                "metric_name": "lambda_weight",
                "sgd_value": float(lw_s),
                "muon_value": float(lw_m),
            },
            "T2": {
                "pass": t2,
                "description": "Muon has a lower final function/weight divergence ratio proxy than SGD.",
                "metric_name": "final_d_function_over_d_weight",
                "sgd_value": float(rf_s),
                "muon_value": float(rf_m),
            },
            "T3": {
                "pass": t3,
                "description": "After fixed training, Muon ends with larger pairwise weight diversity than SGD.",
                "metric_name": "weight_diversity_mean",
                "sgd_value": float(wd_s),
                "muon_value": float(wd_m),
            },
            "T4": {
                "pass": t4,
                "description": "After fixed training, Muon ends with lower pairwise function diversity than SGD.",
                "metric_name": "function_diversity_mean",
                "sgd_value": float(fd_s),
                "muon_value": float(fd_m),
            },
            "T5": {
                "pass": t5,
                "description": "Muon has a lower function-diversity/weight-diversity ratio proxy than SGD.",
                "metric_name": "function_diversity_over_weight_diversity",
                "sgd_value": float(basin_ratio_s),
                "muon_value": float(basin_ratio_m),
            },
            "T6": {
                "pass": t6,
                "description": "Muon shows lower loss instability on at least one current proxy (loss Lyapunov or cross-run loss std).",
                "metric_name": "loss_stability_proxy",
                "sgd_value": float(ls_s),
                "muon_value": float(ls_m),
                "alternate_metric_name": "lambda_loss",
                "sgd_alternate_value": float(ll_s),
                "muon_alternate_value": float(ll_m),
                "loss_std_subtest_pass": t6_basin,
                "lyap_subtest_pass": t6_lyap,
            },
        }

        net_pass = sum(1 for info in tests.values() if info["pass"])
        total_pass += net_pass
        total_tests += len(tests)

        separation = {
            "available": False,
            "note": "Heuristic only: pairwise distances are not independent samples.",
        }
        if len(face2["sgd"]["func_dists"]) > 5 and len(face2["muon"]["func_dists"]) > 5:
            fd_muon = face2["muon"]["func_dists"]
            fd_sgd = face2["sgd"]["func_dists"]
            n1, n2 = len(fd_sgd), len(fd_muon)
            m1, m2 = np.mean(fd_sgd), np.mean(fd_muon)
            v1, v2 = np.var(fd_sgd, ddof=1), np.var(fd_muon, ddof=1)
            se = np.sqrt(v1 / n1 + v2 / n2)
            t_stat = (m1 - m2) / se if se > 1e-15 else np.nan
            separation = {
                "available": True,
                "t_stat": float(t_stat),
                "sgd_mean": float(m1),
                "muon_mean": float(m2),
                "n_sgd_pairs": int(n1),
                "n_muon_pairs": int(n2),
                "note": "Heuristic only: pairwise distances share runs and are not independent samples.",
            }

        tests_by_net[net_type] = {
            "tests": tests,
            "passes": int(net_pass),
            "total": int(len(tests)),
            "heuristic_function_diversity_separation": separation,
        }

    if total_pass >= 10:
        support_label = "strong toy-level support"
        detail = "Across both network types, the proxy metrics are mostly aligned with the benchmark pattern."
    elif total_pass >= 7:
        support_label = "toy-level support"
        detail = "The proxy metrics support the benchmark pattern on a majority of the current checks."
    elif total_pass >= 5:
        support_label = "partial toy-level support"
        detail = "Some parts of the benchmark pattern appear, but the evidence is mixed."
    else:
        support_label = "limited support"
        detail = "The current proxy metrics do not consistently support the benchmark pattern."

    calibrated_conclusion = (
        f"This run provides {support_label} for the specific toy pattern tested here: Muon can exhibit larger "
        f"weight-space divergence while mapping less of that divergence into function-space proxies than SGD. "
        f"These are finite-time synthetic diagnostics on fixed test inputs; they do not prove global contraction, "
        f"resolve a paradox, or establish a gauge-fixing mechanism."
    )

    return {
        "tests_by_net": tests_by_net,
        "total_pass": int(total_pass),
        "total_tests": int(total_tests),
        "support_label": support_label,
        "detail": detail,
        "calibrated_conclusion": calibrated_conclusion,
    }


# =============================================================================
# SUMMARY TABLE HELPERS
# =============================================================================


def build_summary_tables(results):
    lr_rows = []
    face1_rows = []
    face2_rows = []
    heuristic_overview_rows = []
    heuristic_test_rows = []

    for net_type in NETWORK_TYPES:
        network = results["networks"][net_type]
        face1 = network["face1"]["summary"]
        face2 = network["face2"]
        heuristics = results["heuristics"]["tests_by_net"][net_type]

        lr_rows.append(
            {
                "net_type": net_type,
                "chosen_lr_sgd": float(network["lr_sgd"]),
                "lr_muon": float(network["lr_muon"]),
                "candidates_tried": int(len(results["lr_selection"][net_type]["trials"])),
                "criterion": results["lr_selection"][net_type]["criterion"],
            }
        )

        face1_rows.append(
            {
                "net_type": net_type,
                "lr_sgd": float(network["lr_sgd"]),
                "lr_muon": float(network["lr_muon"]),
                "sgd_lambda_weight": float(face1["sgd_lyap_w_mean"]),
                "muon_lambda_weight": float(face1["muon_lyap_w_mean"]),
                "sgd_lambda_function": float(face1["sgd_lyap_f_mean"]),
                "muon_lambda_function": float(face1["muon_lyap_f_mean"]),
                "sgd_lambda_loss": float(face1["sgd_lyap_l_mean"]),
                "muon_lambda_loss": float(face1["muon_lyap_l_mean"]),
                "sgd_final_func_over_weight_ratio": float(face1["sgd_ratio_f_w_final_mean"]),
                "muon_final_func_over_weight_ratio": float(face1["muon_ratio_f_w_final_mean"]),
            }
        )

        wd_s = face2["sgd"]["weight_diversity_mean"]
        wd_m = face2["muon"]["weight_diversity_mean"]
        fd_s = face2["sgd"]["func_diversity_mean"]
        fd_m = face2["muon"]["func_diversity_mean"]

        face2_rows.append(
            {
                "net_type": net_type,
                "num_runs": int(face2["metadata"]["num_runs"]),
                "num_steps": int(face2["metadata"]["num_steps"]),
                "sgd_weight_diversity_mean": float(wd_s),
                "muon_weight_diversity_mean": float(wd_m),
                "sgd_func_diversity_mean": float(fd_s),
                "muon_func_diversity_mean": float(fd_m),
                "sgd_loss_mean": float(face2["sgd"]["loss_mean"]),
                "muon_loss_mean": float(face2["muon"]["loss_mean"]),
                "sgd_loss_std": float(face2["sgd"]["loss_std"]),
                "muon_loss_std": float(face2["muon"]["loss_std"]),
                "sgd_func_over_weight_ratio": float(_safe_ratio(fd_s, wd_s)),
                "muon_func_over_weight_ratio": float(_safe_ratio(fd_m, wd_m)),
            }
        )

        heuristic_overview_rows.append(
            {
                "net_type": net_type,
                "passes": int(heuristics["passes"]),
                "total": int(heuristics["total"]),
                **{test_name: bool(test_info["pass"]) for test_name, test_info in heuristics["tests"].items()},
            }
        )

        for test_name, test_info in heuristics["tests"].items():
            heuristic_test_rows.append(
                {
                    "net_type": net_type,
                    "test": test_name,
                    "pass": bool(test_info["pass"]),
                    "metric_name": test_info["metric_name"],
                    "sgd_value": float(test_info["sgd_value"]),
                    "muon_value": float(test_info["muon_value"]),
                    "description": test_info["description"],
                }
            )

    return {
        "lr_selection": lr_rows,
        "face1": face1_rows,
        "face2": face2_rows,
        "heuristic_overview": heuristic_overview_rows,
        "heuristic_tests": heuristic_test_rows,
    }


# =============================================================================
# PLOTS
# =============================================================================


def generate_plots(results, output_dir=None, verbose=True):
    output_dir = SCRIPT_DIR if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = {}

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        if verbose:
            print("  WARNING: matplotlib not available; skipping plots.")
        return {
            "output_dir": str(output_dir),
            "plot_paths": plot_paths,
            "matplotlib_available": False,
        }

    for net_type in NETWORK_TYPES:
        face1 = results["networks"][net_type]["face1"]["summary"]
        face2 = results["networks"][net_type]["face2"]
        lr_sgd = results["networks"][net_type]["lr_sgd"]

        t_axis = face1["time_axis"]
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            f"Muon toy benchmark ({net_type.upper()} net): larger weight divergence, lower function-space proxy divergence\n"
            f"{NUM_LAYERS}-layer, dim={DIM}, lr_sgd={lr_sgd}, lr_muon={LR_MUON}",
            fontsize=14,
            fontweight="bold",
        )

        ax = axes[0, 0]
        ax.set_title("SGD: weight vs function divergence proxies")
        for p in range(len(face1["sgd_d_w_all"])):
            dw = face1["sgd_d_w_all"][p]
            df = face1["sgd_d_f_all"][p]
            ax.semilogy(t_axis[: len(dw)], dw, "b-", alpha=0.12, linewidth=0.5)
            ax.semilogy(t_axis[: len(df)], df, "r-", alpha=0.12, linewidth=0.5)
        ax.semilogy(
            t_axis[: len(face1["sgd_d_w_mean_traj"])],
            face1["sgd_d_w_mean_traj"],
            "b-",
            linewidth=2.5,
            label=f"d_weight (mean λ={face1['sgd_lyap_w_mean']:+.4f})",
        )
        ax.semilogy(
            t_axis[: len(face1["sgd_d_f_mean_traj"])],
            face1["sgd_d_f_mean_traj"],
            "r-",
            linewidth=2.5,
            label=f"d_function (mean λ={face1['sgd_lyap_f_mean']:+.4f})",
        )
        ax.set_xlabel("Step")
        ax.set_ylabel("Divergence proxy")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.set_title("Muon: weight vs function divergence proxies")
        for p in range(len(face1["muon_d_w_all"])):
            dw = face1["muon_d_w_all"][p]
            df = face1["muon_d_f_all"][p]
            ax.semilogy(t_axis[: len(dw)], dw, "b-", alpha=0.12, linewidth=0.5)
            ax.semilogy(t_axis[: len(df)], df, "r-", alpha=0.12, linewidth=0.5)
        ax.semilogy(
            t_axis[: len(face1["muon_d_w_mean_traj"])],
            face1["muon_d_w_mean_traj"],
            "b-",
            linewidth=2.5,
            label=f"d_weight (mean λ={face1['muon_lyap_w_mean']:+.4f})",
        )
        ax.semilogy(
            t_axis[: len(face1["muon_d_f_mean_traj"])],
            face1["muon_d_f_mean_traj"],
            "r-",
            linewidth=2.5,
            label=f"d_function (mean λ={face1['muon_lyap_f_mean']:+.4f})",
        )
        ax.set_xlabel("Step")
        ax.set_ylabel("Divergence proxy")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[0, 2]
        ax.set_title("Function/weight divergence ratio proxy over time")
        ax.plot(
            t_axis[: len(face1["sgd_ratio_f_w_mean_traj"])],
            face1["sgd_ratio_f_w_mean_traj"],
            "b-",
            linewidth=2.5,
            label=f"SGD (final={face1['sgd_ratio_f_w_final_mean']:.4f})",
        )
        ax.plot(
            t_axis[: len(face1["muon_ratio_f_w_mean_traj"])],
            face1["muon_ratio_f_w_mean_traj"],
            "r-",
            linewidth=2.5,
            label=f"Muon (final={face1['muon_ratio_f_w_final_mean']:.4f})",
        )
        ax.set_xlabel("Step")
        ax.set_ylabel("d_function / d_weight")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.set_title("Independent-run diversity after fixed training")
        categories = ["Weight\nDiversity", "Function\nDiversity"]
        sgd_vals = [face2["sgd"]["weight_diversity_mean"], face2["sgd"]["func_diversity_mean"]]
        muon_vals = [face2["muon"]["weight_diversity_mean"], face2["muon"]["func_diversity_mean"]]
        sgd_err = [face2["sgd"]["weight_diversity_std"], face2["sgd"]["func_diversity_std"]]
        muon_err = [face2["muon"]["weight_diversity_std"], face2["muon"]["func_diversity_std"]]
        x = np.arange(len(categories))
        width = 0.35
        b1 = ax.bar(x - width / 2, sgd_vals, width, yerr=sgd_err, label="SGD", color="#4477AA", edgecolor="black", capsize=4)
        b2 = ax.bar(x + width / 2, muon_vals, width, yerr=muon_err, label="Muon", color="#CC3311", edgecolor="black", capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylabel("Pairwise distance (mean ± std)")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        for bars in [b1, b2]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, h, f"{h:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax = axes[1, 1]
        ax.set_title("Function-diversity / weight-diversity ratio proxy")
        sgd_ratio_basin = _safe_ratio(face2["sgd"]["func_diversity_mean"], face2["sgd"]["weight_diversity_mean"])
        muon_ratio_basin = _safe_ratio(face2["muon"]["func_diversity_mean"], face2["muon"]["weight_diversity_mean"])
        bars = ax.bar(["SGD", "Muon"], [sgd_ratio_basin, muon_ratio_basin], color=["#4477AA", "#CC3311"], edgecolor="black", width=0.5)
        for bar, val in zip(bars, [sgd_ratio_basin, muon_ratio_basin]):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_ylabel("Function diversity / weight diversity")
        ax.grid(True, alpha=0.3, axis="y")

        ax = axes[1, 2]
        ax.set_title("Finite-time Lyapunov proxy summary")
        categories = ["Weight\nSpace", "Function\nSpace", "Loss\nSpace"]
        sgd_lyaps = [face1["sgd_lyap_w_mean"], face1["sgd_lyap_f_mean"], face1["sgd_lyap_l_mean"]]
        muon_lyaps = [face1["muon_lyap_w_mean"], face1["muon_lyap_f_mean"], face1["muon_lyap_l_mean"]]
        sgd_lyap_err = [face1["sgd_lyap_w_std"], face1["sgd_lyap_f_std"], face1["sgd_lyap_l_std"]]
        muon_lyap_err = [face1["muon_lyap_w_std"], face1["muon_lyap_f_std"], face1["muon_lyap_l_std"]]
        x = np.arange(len(categories))
        ax.bar(x - width / 2, sgd_lyaps, width, yerr=sgd_lyap_err, label="SGD", color="#4477AA", edgecolor="black", capsize=4)
        ax.bar(x + width / 2, muon_lyaps, width, yerr=muon_lyap_err, label="Muon", color="#CC3311", edgecolor="black", capsize=4)
        ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylabel("Finite-time Lyapunov proxy")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plot_path = output_dir / f"muon_paradox_{net_type}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plot_paths[net_type] = str(plot_path)
        if verbose:
            print(f"  Plot saved: {plot_path}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "Muon toy benchmark summary: function-diversity / weight-diversity ratio proxy\n"
        "(Lower means less output variation per unit weight variation in this toy setup)",
        fontsize=14,
        fontweight="bold",
    )

    for idx, net_type in enumerate(NETWORK_TYPES):
        face2 = results["networks"][net_type]["face2"]
        ax = axes[idx]
        ax.set_title(f"{net_type.upper()} net")

        sgd_r = _safe_ratio(face2["sgd"]["func_diversity_mean"], face2["sgd"]["weight_diversity_mean"])
        muon_r = _safe_ratio(face2["muon"]["func_diversity_mean"], face2["muon"]["weight_diversity_mean"])

        bars = ax.bar(["SGD", "Muon"], [sgd_r, muon_r], color=["#4477AA", "#CC3311"], edgecolor="black", width=0.5)
        for bar, val in zip(bars, [sgd_r, muon_r]):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{val:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax.set_ylabel("Function diversity / weight diversity")
        ax.grid(True, alpha=0.3, axis="y")
        ax.text(
            0.5,
            0.85,
            f"Weight div: SGD={face2['sgd']['weight_diversity_mean']:.3f}, Muon={face2['muon']['weight_diversity_mean']:.3f}\n"
            f"Func div:   SGD={face2['sgd']['func_diversity_mean']:.5f}, Muon={face2['muon']['func_diversity_mean']:.5f}",
            transform=ax.transAxes,
            fontsize=9,
            ha="center",
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    combined_path = output_dir / "muon_paradox_combined.png"
    plt.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plot_paths["combined"] = str(combined_path)
    if verbose:
        print(f"  Combined plot saved: {combined_path}")

    return {
        "output_dir": str(output_dir),
        "plot_paths": plot_paths,
        "matplotlib_available": True,
    }


# =============================================================================
# MAIN EXPERIMENT DRIVER
# =============================================================================


def run_experiment(config_overrides=None, seed_overrides=None, output_dir=None, make_plots=True, verbose=True):
    """Run the full toy benchmark and return structured results for scripts or notebooks."""
    started = time.time()
    config, seeds = prepare_experiment(config_overrides=config_overrides, seed_overrides=seed_overrides)

    results = {
        "identity": {
            "benchmark": "MUON_PARADOX_chaos_weights_order_functions",
            "script_path": str(SCRIPT_DIR / "run_experiment.py"),
            "scope": "toy benchmark",
            "claim_scope": "finite-time proxy evidence only",
        },
        "config": deepcopy(config),
        "seeds": deepcopy(seeds),
        "lr_selection": {},
        "networks": {},
    }

    if verbose:
        print("=" * 100)
        print("MUON TOY BENCHMARK: larger weight-space divergence with lower function-space divergence proxies")
        print("=" * 100)
        print("Scope: finite-time toy evidence only; not a proof of contraction or mechanism.")
        print(f"Setup: {NUM_LAYERS}-layer nets (dim={DIM}), quadratic loss")
        print(f"Face 1: eps={EPSILON}, {NUM_PERTURBATIONS} perturbations, {NUM_STEPS} steps")
        print(f"Face 2: {NUM_INDEPENDENT_RUNS} independent inits, {FACE2_STEPS} fixed training steps")
        print(f"LR_Muon={LR_MUON}, Momentum={MOMENTUM}, NS_iters={NS_ITERS}")
        print("=" * 100)

    for net_type in NETWORK_TYPES:
        if verbose:
            print(f"\n{'#' * 80}")
            print(f"  NETWORK TYPE: {net_type.upper()}")
            print(f"{'#' * 80}")

        lr_sgd, lr_info = find_stable_lr_sgd(net_type, verbose=verbose)
        face1 = measure_perturbation_lyapunov(net_type, lr_sgd, LR_MUON, NUM_PERTURBATIONS, verbose=verbose)
        face2 = measure_convergence_basin(net_type, lr_sgd, LR_MUON, NUM_INDEPENDENT_RUNS, num_steps=FACE2_STEPS, verbose=verbose)

        results["lr_selection"][net_type] = lr_info
        results["networks"][net_type] = {
            "lr_sgd": float(lr_sgd),
            "lr_muon": float(LR_MUON),
            "face1": face1,
            "face2": face2,
        }

    results["heuristics"] = evaluate_heuristics(results)
    results["summary_tables"] = build_summary_tables(results)
    results["runtime_seconds"] = float(time.time() - started)
    results["artifacts"] = generate_plots(results, output_dir=output_dir, verbose=verbose) if make_plots else {
        "output_dir": str(SCRIPT_DIR if output_dir is None else Path(output_dir)),
        "plot_paths": {},
        "matplotlib_available": None,
    }

    return results


# =============================================================================
# TEXT REPORTING
# =============================================================================


def print_summary(results):
    config = results["config"]
    seeds = results["seeds"]
    heuristics = results["heuristics"]

    print(f"\n\n{'=' * 100}")
    print("SUMMARY REPORT")
    print("=" * 100)
    print(f"Runtime: {results['runtime_seconds']:.2f} s")
    print(f"Output directory: {results['artifacts']['output_dir']}")
    print(f"Seeds: global={seeds['global_seed']}, data={seeds['data_seed']}, base_init={seeds['base_init_seed']}, perturb_base={seeds['perturb_seed_base']}, run_base={seeds['run_seed_base']}")
    print(f"Config: steps_face1={config['NUM_STEPS']}, steps_face2={config['FACE2_STEPS']}, perturbations={config['NUM_PERTURBATIONS']}, runs={config['NUM_INDEPENDENT_RUNS']}")

    print(f"\n{'=' * 100}")
    print("LR SELECTION")
    print("=" * 100)
    for row in results["summary_tables"]["lr_selection"]:
        print(f"  {row['net_type'].upper()}: SGD chosen lr={row['chosen_lr_sgd']}, Muon lr={row['lr_muon']}, candidates tried={row['candidates_tried']}")
        print(f"      criterion: {row['criterion']}")

    print(f"\n{'=' * 100}")
    print("FACE 1: PERTURBATION DIVERGENCE SUMMARY")
    print("=" * 100)
    for row in results["summary_tables"]["face1"]:
        print(f"\n  {row['net_type'].upper()} NET  (lr_sgd={row['lr_sgd']}, lr_muon={row['lr_muon']})")
        print(f"    lambda_weight:   SGD={row['sgd_lambda_weight']:+.6f} | Muon={row['muon_lambda_weight']:+.6f}")
        print(f"    lambda_function: SGD={row['sgd_lambda_function']:+.6f} | Muon={row['muon_lambda_function']:+.6f}")
        print(f"    lambda_loss:     SGD={row['sgd_lambda_loss']:+.6f} | Muon={row['muon_lambda_loss']:+.6f}")
        print(f"    final d_func/d_weight ratio: SGD={row['sgd_final_func_over_weight_ratio']:.6f} | Muon={row['muon_final_func_over_weight_ratio']:.6f}")

    print(f"\n{'=' * 100}")
    print("FACE 2: INDEPENDENT-RUN DIVERSITY AFTER FIXED TRAINING")
    print("=" * 100)
    for row in results["summary_tables"]["face2"]:
        print(f"\n  {row['net_type'].upper()} NET  ({row['num_runs']} runs, {row['num_steps']} steps)")
        print(f"    weight diversity: SGD={row['sgd_weight_diversity_mean']:.6f} | Muon={row['muon_weight_diversity_mean']:.6f}")
        print(f"    function diversity: SGD={row['sgd_func_diversity_mean']:.6f} | Muon={row['muon_func_diversity_mean']:.6f}")
        print(f"    loss mean: SGD={row['sgd_loss_mean']:.6e} | Muon={row['muon_loss_mean']:.6e}")
        print(f"    loss std:  SGD={row['sgd_loss_std']:.6e} | Muon={row['muon_loss_std']:.6e}")
        print(f"    function/weight ratio: SGD={row['sgd_func_over_weight_ratio']:.6f} | Muon={row['muon_func_over_weight_ratio']:.6f}")

    print(f"\n{'=' * 100}")
    print("HEURISTIC CHECKS")
    print("=" * 100)
    for net_type in NETWORK_TYPES:
        net_heur = heuristics["tests_by_net"][net_type]
        print(f"\n  {net_type.upper()} NET: {net_heur['passes']}/{net_heur['total']} checks passed")
        for test_name, test_info in net_heur["tests"].items():
            print(
                f"    {test_name}: {'PASS' if test_info['pass'] else 'FAIL'} | "
                f"SGD={test_info['sgd_value']:.6g} | Muon={test_info['muon_value']:.6g} | {test_info['description']}"
            )
            if test_name == "T6":
                print(
                    f"         alternate lambda_loss proxy: SGD={test_info['sgd_alternate_value']:+.6f} | "
                    f"Muon={test_info['muon_alternate_value']:+.6f}"
                )
        sep = net_heur["heuristic_function_diversity_separation"]
        if sep["available"]:
            print(
                f"    heuristic function-diversity separation t-stat={sep['t_stat']:.4f} "
                f"(SGD mean={sep['sgd_mean']:.6f}, Muon mean={sep['muon_mean']:.6f})"
            )
            print(f"         note: {sep['note']}")

    print(f"\n{'=' * 100}")
    print("CALIBRATED CONCLUSION")
    print("=" * 100)
    print(f"Support level: {heuristics['support_label']}")
    print(heuristics["detail"])
    print(heuristics["calibrated_conclusion"])
    print(f"Total heuristic passes: {heuristics['total_pass']}/{heuristics['total_tests']}")

    if results["artifacts"]["plot_paths"]:
        print("Generated plots:")
        for name, path in results["artifacts"]["plot_paths"].items():
            print(f"  {name}: {path}")


# =============================================================================
# CLI ENTRYPOINT
# =============================================================================


def main(argv=None):
    parser = argparse.ArgumentParser(description="Muon toy benchmark: import-safe experiment driver.")
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR), help="Directory for saved plot artifacts.")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    parser.add_argument("--quick-smoke", action="store_true", help="Run a small smoke configuration instead of the full default benchmark.")
    parser.add_argument("--quiet", action="store_true", help="Reduce progress logging during the run.")
    args = parser.parse_args(argv)

    config_overrides = deepcopy(SMOKE_TEST_CONFIG) if args.quick_smoke else None
    results = run_experiment(
        config_overrides=config_overrides,
        output_dir=args.output_dir,
        make_plots=not args.no_plots,
        verbose=not args.quiet,
    )
    print_summary(results)
    return results


prepare_experiment()


if __name__ == "__main__":
    main()
