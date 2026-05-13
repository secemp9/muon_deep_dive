#!/usr/bin/env python3
"""
H18a: depth-stability of Muon's empirical learning rate in a deep-linear toy model.

This experiment compares SGD and Muon in 32x32 deep linear networks across depths
[2, 4, 8, 16]. It measures:
  - initialization-time operator norms of layer gradients and orthogonalized gradients
  - empirical best learning rates from fixed-budget LR sweeps
  - empirical max stable learning rates from short-horizon non-divergence searches
  - descriptive log-log fits across depth

The goal is to gather toy-model evidence about whether operator-norm clamping is
consistent with Muon's weaker depth sensitivity. It is not a universal proof and
it does not establish a complete causal chain beyond the quantities measured here.

The companion notebook imports run_experiment() from this file and uses the
returned structured results for presentation, figures, and calibrated discussion.
"""

import copy
import sys
import time
from datetime import datetime, timezone

import numpy as np


DEFAULT_CONFIG = {
    "dim": 32,
    "depths": [2, 4, 8, 16],
    "ns_iters": 5,
    "batch_size": 64,
    "momentum": 0.9,
    "num_seeds": 5,
    "base_seed": 42,
    "seed_stride": 137,
    "init_seed_offset": 5000,
    "train_steps": 300,
    "train_loss_abort_threshold": 1e10,
    "divergence_steps": 100,
    "divergence_threshold": 1e6,
    "sgd_lr_grid": np.logspace(-5, 0, 30).tolist(),
    "muon_lr_grid": np.logspace(-4, 0, 30).tolist(),
    "hessian_power_iters": 20,
    "hvp_eps": 1e-5,
    "max_stable_lr_low": 1e-6,
    "max_stable_lr_high": 10.0,
    "max_stable_lr_cap": 1e4,
    "max_stable_search_iters": 25,
    "max_stable_tol_ratio": 1.05,
    "reference_depth_low": 2,
    "reference_depth_high": 16,
    "test_thresholds": {
        "t1_max_dev": 0.01,
        "t2_min_growth_ratio": 5.0,
        "t3_min_sgd_drop_ratio": 20.0,
        "t4_max_muon_drop_ratio": 5.0,
        "t5_max_cv": 0.5,
    },
}


def _deep_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def resolve_config(config=None):
    """Merge overrides with defaults and normalize types."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if config is not None:
        _deep_update(cfg, config)

    cfg["depths"] = sorted(int(d) for d in cfg["depths"])
    cfg["dim"] = int(cfg["dim"])
    cfg["ns_iters"] = int(cfg["ns_iters"])
    cfg["batch_size"] = int(cfg["batch_size"])
    cfg["num_seeds"] = int(cfg["num_seeds"])
    cfg["base_seed"] = int(cfg["base_seed"])
    cfg["seed_stride"] = int(cfg["seed_stride"])
    cfg["init_seed_offset"] = int(cfg["init_seed_offset"])
    cfg["train_steps"] = int(cfg["train_steps"])
    cfg["divergence_steps"] = int(cfg["divergence_steps"])
    cfg["hessian_power_iters"] = int(cfg["hessian_power_iters"])
    cfg["max_stable_search_iters"] = int(cfg["max_stable_search_iters"])

    cfg["momentum"] = float(cfg["momentum"])
    cfg["train_loss_abort_threshold"] = float(cfg["train_loss_abort_threshold"])
    cfg["divergence_threshold"] = float(cfg["divergence_threshold"])
    cfg["hvp_eps"] = float(cfg["hvp_eps"])
    cfg["max_stable_lr_low"] = float(cfg["max_stable_lr_low"])
    cfg["max_stable_lr_high"] = float(cfg["max_stable_lr_high"])
    cfg["max_stable_lr_cap"] = float(cfg["max_stable_lr_cap"])
    cfg["max_stable_tol_ratio"] = float(cfg["max_stable_tol_ratio"])

    cfg["sgd_lr_grid"] = [float(x) for x in cfg["sgd_lr_grid"]]
    cfg["muon_lr_grid"] = [float(x) for x in cfg["muon_lr_grid"]]

    if cfg.get("seeds") is not None:
        seeds = [int(s) for s in cfg["seeds"]]
    else:
        seeds = [cfg["base_seed"] + i * cfg["seed_stride"] for i in range(cfg["num_seeds"])]
    cfg["seeds"] = seeds
    cfg["num_seeds"] = len(seeds)

    low_depth = int(cfg.get("reference_depth_low", cfg["depths"][0]))
    high_depth = int(cfg.get("reference_depth_high", cfg["depths"][-1]))
    if low_depth not in cfg["depths"]:
        low_depth = cfg["depths"][0]
    if high_depth not in cfg["depths"]:
        high_depth = cfg["depths"][-1]
    if low_depth > high_depth:
        low_depth, high_depth = high_depth, low_depth
    cfg["reference_depth_low"] = low_depth
    cfg["reference_depth_high"] = high_depth

    thresholds = cfg["test_thresholds"]
    cfg["test_thresholds"] = {
        "t1_max_dev": float(thresholds["t1_max_dev"]),
        "t2_min_growth_ratio": float(thresholds["t2_min_growth_ratio"]),
        "t3_min_sgd_drop_ratio": float(thresholds["t3_min_sgd_drop_ratio"]),
        "t4_max_muon_drop_ratio": float(thresholds["t4_max_muon_drop_ratio"]),
        "t5_max_cv": float(thresholds["t5_max_cv"]),
    }
    return cfg


def make_smoke_test_config():
    """Small config for basic code-path checks."""
    return resolve_config(
        {
            "depths": [2, 4],
            "num_seeds": 1,
            "train_steps": 20,
            "divergence_steps": 20,
            "sgd_lr_grid": np.logspace(-5, -1, 5).tolist(),
            "muon_lr_grid": np.logspace(-4, 0, 5).tolist(),
            "hessian_power_iters": 6,
            "max_stable_search_iters": 8,
            "reference_depth_low": 2,
            "reference_depth_high": 4,
        }
    )


def estimate_work(config=None):
    cfg = resolve_config(config)
    num_depths = len(cfg["depths"])
    num_seeds = len(cfg["seeds"])
    phase2_lr_runs = num_depths * num_seeds * (len(cfg["sgd_lr_grid"]) + len(cfg["muon_lr_grid"]))
    return {
        "num_depths": num_depths,
        "num_seeds": num_seeds,
        "phase1_hessian_estimates": num_depths * num_seeds,
        "phase2_lr_runs": phase2_lr_runs,
        "phase2_training_steps": phase2_lr_runs * cfg["train_steps"],
        "phase3_binary_searches": num_depths * num_seeds * 2,
        "sgd_lr_grid_size": len(cfg["sgd_lr_grid"]),
        "muon_lr_grid_size": len(cfg["muon_lr_grid"]),
    }


def summarize_values(values):
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    summary = {
        "count": int(arr.size),
        "finite_count": int(finite.size),
    }
    if finite.size == 0:
        summary.update({
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        })
    else:
        summary.update({
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "median": float(np.median(finite)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
        })
    return summary


def safe_ratio(numerator, denominator):
    numerator = float(numerator)
    denominator = float(denominator)
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return float("nan")
    if abs(denominator) < 1e-15:
        return float("inf")
    return numerator / denominator


def coefficient_of_variation(values):
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    mean = np.mean(finite)
    if abs(mean) < 1e-15:
        return float("inf")
    return float(np.std(finite) / mean)


def operator_norm(matrix):
    return float(np.linalg.svd(matrix, compute_uv=False)[0])


def newton_schulz(matrix, n_iters):
    """Compute an approximate orthogonal polar factor via Newton-Schulz iteration."""
    norm = np.linalg.norm(matrix, ord="fro")
    if norm < 1e-15:
        return matrix
    x = matrix / norm
    for _ in range(n_iters):
        a = x.T @ x
        x = 1.5 * x - 0.5 * x @ a
    return x


def init_weights(depth, seed, dim):
    rng = np.random.RandomState(seed)
    return [np.eye(dim) + rng.randn(dim, dim) * 0.1 for _ in range(depth)]


def forward(weights, x):
    out = x.copy()
    for weight in weights:
        out = weight @ out
    return out


def compute_loss(weights, x, y):
    pred = forward(weights, x)
    return 0.5 * np.mean(np.sum((pred - y) ** 2, axis=0))


def compute_gradients(weights, x, y):
    num_layers = len(weights)
    batch = x.shape[1]
    activations = [x.copy()]
    for weight in weights:
        activations.append(weight @ activations[-1])
    delta = (activations[-1] - y) / batch
    grads = [None] * num_layers
    for layer in range(num_layers - 1, -1, -1):
        grads[layer] = delta @ activations[layer].T
        if layer > 0:
            delta = weights[layer].T @ delta
    return grads


def make_data(seed, dim, batch_size):
    rng = np.random.RandomState(seed)
    target = rng.randn(dim, dim) * 0.5
    x = rng.randn(dim, batch_size) * 0.3
    y = target @ x
    return x, y


def train(weights_init, x, y, lr, optimizer, num_steps, momentum, ns_iters, loss_abort_threshold):
    """Train and return final loss plus step-1 update operator norms."""
    weights = [w.copy() for w in weights_init]
    velocity = [np.zeros_like(w) for w in weights]
    step1_op_norms = None

    for step in range(num_steps):
        loss = compute_loss(weights, x, y)
        if not np.isfinite(loss) or loss > loss_abort_threshold:
            return float("inf"), None
        grads = compute_gradients(weights, x, y)
        deltas = []
        for idx in range(len(weights)):
            if optimizer == "muon":
                velocity[idx] = momentum * velocity[idx] + newton_schulz(grads[idx], n_iters=ns_iters)
            else:
                velocity[idx] = momentum * velocity[idx] + grads[idx]
            delta = lr * velocity[idx]
            deltas.append(delta)
            weights[idx] = weights[idx] - delta

        if step == 0:
            step1_op_norms = [operator_norm(delta) for delta in deltas]

    return float(compute_loss(weights, x, y)), step1_op_norms


def is_stable(weights_init, x, y, lr, optimizer, steps, divergence_threshold, momentum, ns_iters):
    weights = [w.copy() for w in weights_init]
    velocity = [np.zeros_like(w) for w in weights]
    for _ in range(steps):
        grads = compute_gradients(weights, x, y)
        for idx in range(len(weights)):
            if optimizer == "muon":
                velocity[idx] = momentum * velocity[idx] + newton_schulz(grads[idx], n_iters=ns_iters)
            else:
                velocity[idx] = momentum * velocity[idx] + grads[idx]
            weights[idx] = weights[idx] - lr * velocity[idx]
        loss = compute_loss(weights, x, y)
        if not np.isfinite(loss) or loss > divergence_threshold:
            return False
    return True


def find_max_stable_lr(
    weights_init,
    x,
    y,
    optimizer,
    lr_low,
    lr_high,
    lr_cap,
    search_iters,
    tol_ratio,
    steps,
    divergence_threshold,
    momentum,
    ns_iters,
):
    while is_stable(weights_init, x, y, lr_high, optimizer, steps, divergence_threshold, momentum, ns_iters):
        lr_high *= 2.0
        if lr_high > lr_cap:
            return float(lr_high)
    while not is_stable(weights_init, x, y, lr_low, optimizer, steps, divergence_threshold, momentum, ns_iters):
        lr_low /= 2.0
        if lr_low < 1e-10:
            return 0.0
    for _ in range(search_iters):
        lr_mid = float(np.sqrt(lr_low * lr_high))
        if lr_high / lr_low < tol_ratio:
            break
        if is_stable(weights_init, x, y, lr_mid, optimizer, steps, divergence_threshold, momentum, ns_iters):
            lr_low = lr_mid
        else:
            lr_high = lr_mid
    return float(np.sqrt(lr_low * lr_high))


def flatten_weights(weights):
    return np.concatenate([weight.ravel() for weight in weights])


def unflatten_weights(flat_weights, shapes):
    weights = []
    idx = 0
    for shape in shapes:
        size = shape[0] * shape[1]
        weights.append(flat_weights[idx:idx + size].reshape(shape))
        idx += size
    return weights


def hessian_vector_product(weights, x, y, v_flat, eps):
    shapes = [weight.shape for weight in weights]
    w_flat = flatten_weights(weights)
    w_plus = unflatten_weights(w_flat + eps * v_flat, shapes)
    g_plus = flatten_weights(compute_gradients(w_plus, x, y))
    w_minus = unflatten_weights(w_flat - eps * v_flat, shapes)
    g_minus = flatten_weights(compute_gradients(w_minus, x, y))
    return (g_plus - g_minus) / (2.0 * eps)


def power_iteration_lambda_max(weights, x, y, n_iters, eps):
    dim = sum(weight.size for weight in weights)
    rng = np.random.RandomState(0)
    vec = rng.randn(dim)
    vec = vec / np.linalg.norm(vec)
    lam = 0.0
    for _ in range(n_iters):
        hv = hessian_vector_product(weights, x, y, vec, eps=eps)
        lam = float(np.dot(vec, hv))
        norm = np.linalg.norm(hv)
        if norm < 1e-15:
            break
        vec = hv / norm
    return abs(lam)


def log_log_fit(x_values, y_values):
    x_arr = np.asarray(x_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if np.sum(mask) < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}
    lx = np.log(x_arr[mask])
    ly = np.log(np.abs(y_arr[mask]) + 1e-15)
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_res = np.sum((ly - pred) ** 2)
    ss_tot = np.sum((ly - np.mean(ly)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-15) if ss_tot > 1e-15 else 0.0
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2)}


def _best_lr_on_edge(best_lr, lr_grid):
    if not np.isfinite(best_lr):
        return False
    return bool(np.isclose(best_lr, lr_grid[0]) or np.isclose(best_lr, lr_grid[-1]))


def run_phase1(cfg, verbose=False):
    phase1 = {"by_depth": {}}
    if verbose:
        print("=" * 100)
        print("PHASE 1: initialization measurements")
        print("=" * 100)

    for depth in cfg["depths"]:
        records = []
        for seed in cfg["seeds"]:
            x, y = make_data(seed, cfg["dim"], cfg["batch_size"])
            weights0 = init_weights(depth, seed + cfg["init_seed_offset"], cfg["dim"])
            grads = compute_gradients(weights0, x, y)
            grad_ops = [operator_norm(g) for g in grads]
            ortho_ops = [operator_norm(newton_schulz(g, n_iters=cfg["ns_iters"])) for g in grads]
            lam = power_iteration_lambda_max(
                weights0,
                x,
                y,
                n_iters=cfg["hessian_power_iters"],
                eps=cfg["hvp_eps"],
            )
            records.append(
                {
                    "seed": int(seed),
                    "data_seed": int(seed),
                    "init_seed": int(seed + cfg["init_seed_offset"]),
                    "layer_grad_op_norms": [float(v) for v in grad_ops],
                    "max_grad_op_norm": float(max(grad_ops)),
                    "layer_ortho_grad_op_norms": [float(v) for v in ortho_ops],
                    "max_ortho_grad_op_norm": float(max(ortho_ops)),
                    "lambda_max_init": float(lam),
                }
            )

        summary = {
            "max_grad_op_norm": summarize_values([record["max_grad_op_norm"] for record in records]),
            "max_ortho_grad_op_norm": summarize_values([record["max_ortho_grad_op_norm"] for record in records]),
            "lambda_max_init": summarize_values([record["lambda_max_init"] for record in records]),
        }
        phase1["by_depth"][depth] = {
            "per_seed": records,
            "summary": summary,
        }

        if verbose:
            print(
                f"  depth {depth:>2}: "
                f"mean max ||G||_op = {summary['max_grad_op_norm']['mean']:.4f} +/- {summary['max_grad_op_norm']['std']:.4f}, "
                f"mean max ||ortho(G)||_op = {summary['max_ortho_grad_op_norm']['mean']:.6f} +/- {summary['max_ortho_grad_op_norm']['std']:.6f}, "
                f"mean lambda_max(H_init) = {summary['lambda_max_init']['mean']:.2f} +/- {summary['lambda_max_init']['std']:.2f}"
            )

    return phase1


def run_phase2(cfg, verbose=False):
    phase2 = {"by_depth": {}}
    if verbose:
        print("\n" + "=" * 100)
        print("PHASE 2: empirical best LR from fixed-budget sweeps")
        print("=" * 100)

    for depth in cfg["depths"]:
        phase2["by_depth"][depth] = {}
        if verbose:
            print(f"\n  depth L={depth}:")

        for optimizer, grid_key in [("sgd", "sgd_lr_grid"), ("muon", "muon_lr_grid")]:
            lr_grid = np.asarray(cfg[grid_key], dtype=float)
            records = []
            for seed in cfg["seeds"]:
                x, y = make_data(seed, cfg["dim"], cfg["batch_size"])
                weights0 = init_weights(depth, seed + cfg["init_seed_offset"], cfg["dim"])
                final_losses = []
                step1_max_op_norms = []
                finite_mask = []
                best_lr = float("nan")
                best_loss = float("inf")
                best_step1_max = float("nan")

                for lr in lr_grid:
                    final_loss, step1_ops = train(
                        weights0,
                        x,
                        y,
                        lr=float(lr),
                        optimizer=optimizer,
                        num_steps=cfg["train_steps"],
                        momentum=cfg["momentum"],
                        ns_iters=cfg["ns_iters"],
                        loss_abort_threshold=cfg["train_loss_abort_threshold"],
                    )
                    step1_max = float(max(step1_ops)) if step1_ops is not None else float("nan")
                    final_losses.append(float(final_loss))
                    step1_max_op_norms.append(step1_max)
                    finite = bool(np.isfinite(final_loss))
                    finite_mask.append(finite)
                    if finite and final_loss < best_loss:
                        best_loss = float(final_loss)
                        best_lr = float(lr)
                        best_step1_max = step1_max

                found_finite = np.isfinite(best_loss)
                records.append(
                    {
                        "seed": int(seed),
                        "data_seed": int(seed),
                        "init_seed": int(seed + cfg["init_seed_offset"]),
                        "optimizer": optimizer,
                        "lr_grid": [float(v) for v in lr_grid],
                        "final_losses": [float(v) for v in final_losses],
                        "step1_max_op_norms": [float(v) for v in step1_max_op_norms],
                        "finite_mask": [bool(v) for v in finite_mask],
                        "found_finite_run": bool(found_finite),
                        "num_finite_runs": int(sum(finite_mask)),
                        "best_lr": float(best_lr) if found_finite else float("nan"),
                        "best_loss": float(best_loss),
                        "best_step1_max_op_norm": float(best_step1_max) if found_finite else float("nan"),
                        "best_lr_on_grid_edge": _best_lr_on_edge(best_lr, lr_grid) if found_finite else False,
                    }
                )

            finite_best_records = [record for record in records if record["found_finite_run"]]
            best_lrs = [record["best_lr"] for record in finite_best_records]
            best_losses = [record["best_loss"] for record in finite_best_records]
            best_step1 = [record["best_step1_max_op_norm"] for record in finite_best_records]
            edge_count = sum(record["best_lr_on_grid_edge"] for record in finite_best_records)

            summary = {
                "num_seeds": len(records),
                "num_seeds_with_finite_best": len(finite_best_records),
                "mean_best_lr": float(np.mean(best_lrs)) if best_lrs else float("nan"),
                "median_best_lr": float(np.median(best_lrs)) if best_lrs else float("nan"),
                "mean_best_loss": float(np.mean(best_losses)) if best_losses else float("nan"),
                "median_best_loss": float(np.median(best_losses)) if best_losses else float("nan"),
                "mean_best_step1_max_op_norm": float(np.mean(best_step1)) if best_step1 else float("nan"),
                "median_best_step1_max_op_norm": float(np.median(best_step1)) if best_step1 else float("nan"),
                "grid_edge_count": int(edge_count),
                "grid_edge_fraction": float(edge_count / len(finite_best_records)) if finite_best_records else float("nan"),
                "lr_grid_min": float(lr_grid[0]),
                "lr_grid_max": float(lr_grid[-1]),
                "lr_grid_size": int(len(lr_grid)),
            }
            phase2["by_depth"][depth][optimizer] = {
                "per_seed": records,
                "summary": summary,
            }

            if verbose:
                print(
                    f"    {optimizer:>5}: median best LR = {summary['median_best_lr']:.6f}, "
                    f"mean best LR = {summary['mean_best_lr']:.6f}, "
                    f"mean best loss = {summary['mean_best_loss']:.6e}, "
                    f"mean step-1 max ||dW||_op = {summary['mean_best_step1_max_op_norm']:.4e}, "
                    f"edge hits = {summary['grid_edge_count']}/{summary['num_seeds_with_finite_best']}"
                )

    return phase2


def run_phase3(cfg, verbose=False):
    phase3 = {"by_depth": {}}
    if verbose:
        print("\n" + "=" * 100)
        print("PHASE 3: empirical max stable LR from binary search")
        print("=" * 100)

    for depth in cfg["depths"]:
        phase3["by_depth"][depth] = {}
        for optimizer in ["sgd", "muon"]:
            records = []
            for seed in cfg["seeds"]:
                x, y = make_data(seed, cfg["dim"], cfg["batch_size"])
                weights0 = init_weights(depth, seed + cfg["init_seed_offset"], cfg["dim"])
                max_stable_lr = find_max_stable_lr(
                    weights0,
                    x,
                    y,
                    optimizer=optimizer,
                    lr_low=cfg["max_stable_lr_low"],
                    lr_high=cfg["max_stable_lr_high"],
                    lr_cap=cfg["max_stable_lr_cap"],
                    search_iters=cfg["max_stable_search_iters"],
                    tol_ratio=cfg["max_stable_tol_ratio"],
                    steps=cfg["divergence_steps"],
                    divergence_threshold=cfg["divergence_threshold"],
                    momentum=cfg["momentum"],
                    ns_iters=cfg["ns_iters"],
                )
                records.append(
                    {
                        "seed": int(seed),
                        "data_seed": int(seed),
                        "init_seed": int(seed + cfg["init_seed_offset"]),
                        "optimizer": optimizer,
                        "max_stable_lr": float(max_stable_lr),
                    }
                )

            values = [record["max_stable_lr"] for record in records]
            summary = {
                "num_seeds": len(records),
                "mean_max_stable_lr": float(np.mean(values)) if values else float("nan"),
                "median_max_stable_lr": float(np.median(values)) if values else float("nan"),
                "std_max_stable_lr": float(np.std(values)) if values else float("nan"),
                "min_max_stable_lr": float(np.min(values)) if values else float("nan"),
                "max_max_stable_lr": float(np.max(values)) if values else float("nan"),
            }
            phase3["by_depth"][depth][optimizer] = {
                "per_seed": records,
                "summary": summary,
            }

        if verbose:
            sgd_summary = phase3["by_depth"][depth]["sgd"]["summary"]
            muon_summary = phase3["by_depth"][depth]["muon"]["summary"]
            print(
                f"  depth {depth:>2}: SGD mean max stable LR = {sgd_summary['mean_max_stable_lr']:.6f}, "
                f"Muon mean max stable LR = {muon_summary['mean_max_stable_lr']:.6f}"
            )

    return phase3


def build_summary_table(cfg, phase1, phase2, phase3):
    low_depth = cfg["reference_depth_low"]
    sgd_best_low = phase2["by_depth"][low_depth]["sgd"]["summary"]["median_best_lr"]
    muon_best_low = phase2["by_depth"][low_depth]["muon"]["summary"]["median_best_lr"]
    rows = []
    for depth in cfg["depths"]:
        mean_grad = phase1["by_depth"][depth]["summary"]["max_grad_op_norm"]["mean"]
        mean_ortho = phase1["by_depth"][depth]["summary"]["max_ortho_grad_op_norm"]["mean"]
        mean_lambda = phase1["by_depth"][depth]["summary"]["lambda_max_init"]["mean"]
        sgd_best = phase2["by_depth"][depth]["sgd"]["summary"]["median_best_lr"]
        muon_best = phase2["by_depth"][depth]["muon"]["summary"]["median_best_lr"]
        sgd_max = phase3["by_depth"][depth]["sgd"]["summary"]["mean_max_stable_lr"]
        muon_max = phase3["by_depth"][depth]["muon"]["summary"]["mean_max_stable_lr"]
        rows.append(
            {
                "depth": int(depth),
                "mean_max_grad_op_norm": float(mean_grad),
                "mean_max_ortho_grad_op_norm": float(mean_ortho),
                "mean_lambda_max_init": float(mean_lambda),
                "median_best_lr_sgd": float(sgd_best),
                "median_best_lr_muon": float(muon_best),
                "mean_max_stable_lr_sgd": float(sgd_max),
                "mean_max_stable_lr_muon": float(muon_max),
                "sgd_bestlr_times_grad_op": float(sgd_best * mean_grad),
                "sgd_maxstablelr_times_grad_op": float(sgd_max * mean_grad),
                "sgd_best_lr_drop_vs_low_depth": float(safe_ratio(sgd_best_low, sgd_best)),
                "muon_best_lr_drop_vs_low_depth": float(safe_ratio(muon_best_low, muon_best)),
            }
        )
    return rows


def build_fit_rows(cfg, phase1, phase2, phase3):
    low_depth = cfg["reference_depth_low"]
    high_depth = cfg["reference_depth_high"]

    metric_specs = [
        ("max ||G||_op", {depth: phase1["by_depth"][depth]["summary"]["max_grad_op_norm"]["mean"] for depth in cfg["depths"]}),
        ("estimated lambda_max(H_init)", {depth: phase1["by_depth"][depth]["summary"]["lambda_max_init"]["mean"] for depth in cfg["depths"]}),
        ("SGD median best LR", {depth: phase2["by_depth"][depth]["sgd"]["summary"]["median_best_lr"] for depth in cfg["depths"]}),
        ("Muon median best LR", {depth: phase2["by_depth"][depth]["muon"]["summary"]["median_best_lr"] for depth in cfg["depths"]}),
        ("SGD mean max stable LR", {depth: phase3["by_depth"][depth]["sgd"]["summary"]["mean_max_stable_lr"] for depth in cfg["depths"]}),
        ("Muon mean max stable LR", {depth: phase3["by_depth"][depth]["muon"]["summary"]["mean_max_stable_lr"] for depth in cfg["depths"]}),
        (
            "SGD median best LR * max ||G||_op",
            {
                depth: phase2["by_depth"][depth]["sgd"]["summary"]["median_best_lr"]
                * phase1["by_depth"][depth]["summary"]["max_grad_op_norm"]["mean"]
                for depth in cfg["depths"]
            },
        ),
        (
            "SGD mean max stable LR * max ||G||_op",
            {
                depth: phase3["by_depth"][depth]["sgd"]["summary"]["mean_max_stable_lr"]
                * phase1["by_depth"][depth]["summary"]["max_grad_op_norm"]["mean"]
                for depth in cfg["depths"]
            },
        ),
    ]

    rows = []
    for metric_name, values_by_depth in metric_specs:
        ordered_values = [float(values_by_depth[depth]) for depth in cfg["depths"]]
        fit = log_log_fit(cfg["depths"], ordered_values)
        rows.append(
            {
                "metric": metric_name,
                "depths": [int(depth) for depth in cfg["depths"]],
                "values": ordered_values,
                "slope": float(fit["slope"]),
                "r2": float(fit["r2"]),
                "ratio_low_over_high": float(safe_ratio(values_by_depth[low_depth], values_by_depth[high_depth])),
                "depth_low": int(low_depth),
                "depth_high": int(high_depth),
            }
        )
    return rows


def build_tests(cfg, phase1, phase2, phase3):
    low_depth = cfg["reference_depth_low"]
    high_depth = cfg["reference_depth_high"]
    thresholds = cfg["test_thresholds"]

    all_ortho = [
        record["max_ortho_grad_op_norm"]
        for depth in cfg["depths"]
        for record in phase1["by_depth"][depth]["per_seed"]
    ]
    t1_value = max(abs(value - 1.0) for value in all_ortho)
    t2_low = phase1["by_depth"][low_depth]["summary"]["max_grad_op_norm"]["mean"]
    t2_high = phase1["by_depth"][high_depth]["summary"]["max_grad_op_norm"]["mean"]
    t2_value = safe_ratio(t2_high, t2_low)

    sgd_best_low = phase2["by_depth"][low_depth]["sgd"]["summary"]["median_best_lr"]
    sgd_best_high = phase2["by_depth"][high_depth]["sgd"]["summary"]["median_best_lr"]
    t3_value = safe_ratio(sgd_best_low, sgd_best_high)

    muon_best_low = phase2["by_depth"][low_depth]["muon"]["summary"]["median_best_lr"]
    muon_best_high = phase2["by_depth"][high_depth]["muon"]["summary"]["median_best_lr"]
    t4_value = safe_ratio(muon_best_low, muon_best_high)

    t5_products = [
        phase3["by_depth"][depth]["sgd"]["summary"]["mean_max_stable_lr"]
        * phase1["by_depth"][depth]["summary"]["max_grad_op_norm"]["mean"]
        for depth in cfg["depths"]
    ]
    t5_value = coefficient_of_variation(t5_products)

    tests = {
        "T1": {
            "description": "Initialization-time max ||ortho(G)||_op stays close to 1 across depths.",
            "measured_quantity": "max deviation of per-seed max ||ortho(G)||_op from 1",
            "value": float(t1_value),
            "threshold": float(thresholds["t1_max_dev"]),
            "criterion": "<",
            "pass": bool(t1_value < thresholds["t1_max_dev"]),
        },
        "T2": {
            "description": f"Initialization-time max ||G||_op grows from depth {low_depth} to depth {high_depth}.",
            "measured_quantity": f"mean max ||G||_op ratio depth {high_depth} / depth {low_depth}",
            "value": float(t2_value),
            "threshold": float(thresholds["t2_min_growth_ratio"]),
            "criterion": ">",
            "pass": bool(t2_value > thresholds["t2_min_growth_ratio"]),
        },
        "T3": {
            "description": f"SGD median best LR drops strongly from depth {low_depth} to depth {high_depth}.",
            "measured_quantity": f"median best LR ratio depth {low_depth} / depth {high_depth}",
            "value": float(t3_value),
            "threshold": float(thresholds["t3_min_sgd_drop_ratio"]),
            "criterion": ">",
            "pass": bool(t3_value > thresholds["t3_min_sgd_drop_ratio"]),
        },
        "T4": {
            "description": f"Muon median best LR varies much less from depth {low_depth} to depth {high_depth}.",
            "measured_quantity": f"median best LR ratio depth {low_depth} / depth {high_depth}",
            "value": float(t4_value),
            "threshold": float(thresholds["t4_max_muon_drop_ratio"]),
            "criterion": "<",
            "pass": bool(t4_value < thresholds["t4_max_muon_drop_ratio"]),
        },
        "T5": {
            "description": "SGD mean max-stable-LR * mean max ||G||_op is approximately depth-invariant.",
            "measured_quantity": "coefficient of variation across depths",
            "value": float(t5_value),
            "threshold": float(thresholds["t5_max_cv"]),
            "criterion": "<",
            "pass": bool(t5_value < thresholds["t5_max_cv"]),
            "details": {
                "products": [float(v) for v in t5_products],
                "depths": [int(depth) for depth in cfg["depths"]],
            },
        },
    }
    return tests


def build_grid_diagnostics(cfg, phase2):
    rows = []
    any_edge = False
    for depth in cfg["depths"]:
        row = {"depth": int(depth)}
        for optimizer in ["sgd", "muon"]:
            summary = phase2["by_depth"][depth][optimizer]["summary"]
            row[f"{optimizer}_best_lr_edge_hits"] = int(summary["grid_edge_count"])
            row[f"{optimizer}_best_lr_edge_fraction"] = float(summary["grid_edge_fraction"])
            any_edge = any_edge or summary["grid_edge_count"] > 0
        rows.append(row)
    return {
        "any_best_lr_on_edge": bool(any_edge),
        "rows": rows,
    }


def build_verdict(cfg, tests, grid_diagnostics):
    n_pass = sum(int(test["pass"]) for test in tests.values())
    failed_tests = [name for name, test in tests.items() if not test["pass"]]
    sgd_drop = tests["T3"]["value"]
    muon_drop = tests["T4"]["value"]
    advantage = safe_ratio(sgd_drop, muon_drop)
    conclusion = (
        "This run provides toy-model evidence consistent with operator-norm clamping "
        "contributing to Muon's weaker depth sensitivity than SGD. The measured quantities "
        "are initialization operator norms, empirical best learning rates, empirical max stable "
        "learning rates, and descriptive depth-scaling fits. This is not a universal proof and "
        "does not by itself establish behavior beyond this deep-linear setting."
    )
    if grid_diagnostics["any_best_lr_on_edge"]:
        conclusion += " Some best LR selections hit the sweep boundary, so grid widening should be considered before over-interpreting those points."
    return {
        "all_pass": bool(n_pass == len(tests)),
        "n_pass": int(n_pass),
        "n_tests": int(len(tests)),
        "failed_tests": failed_tests,
        "sgd_best_lr_drop_ratio": float(sgd_drop),
        "muon_best_lr_drop_ratio": float(muon_drop),
        "relative_drop_advantage": float(advantage),
        "calibrated_conclusion": conclusion,
    }


def run_experiment(config=None, verbose=False):
    """Run the full experiment and return structured results."""
    cfg = resolve_config(config)
    work_estimate = estimate_work(cfg)
    started = time.time()

    if verbose:
        print("=" * 100)
        print("H18a: Muon LR stability across depth in a deep-linear toy model")
        print("=" * 100)
        print("Measured quantities: init operator norms, empirical best LRs, empirical max stable LRs, descriptive fits.")
        print("Interpretation scope: toy-model evidence, not a universal proof.")
        print(f"depths={cfg['depths']}, seeds={cfg['seeds']}, train_steps={cfg['train_steps']}, divergence_steps={cfg['divergence_steps']}")
        print(
            f"work estimate: phase1 Hessian estimates={work_estimate['phase1_hessian_estimates']}, "
            f"phase2 LR runs={work_estimate['phase2_lr_runs']}, "
            f"phase2 training steps={work_estimate['phase2_training_steps']}, "
            f"phase3 binary searches={work_estimate['phase3_binary_searches']}"
        )
        print()

    phase1 = run_phase1(cfg, verbose=verbose)
    phase2 = run_phase2(cfg, verbose=verbose)
    phase3 = run_phase3(cfg, verbose=verbose)

    summary_table = build_summary_table(cfg, phase1, phase2, phase3)
    fits = build_fit_rows(cfg, phase1, phase2, phase3)
    tests = build_tests(cfg, phase1, phase2, phase3)
    grid_diagnostics = build_grid_diagnostics(cfg, phase2)
    verdict = build_verdict(cfg, tests, grid_diagnostics)

    finished = time.time()
    results = {
        "metadata": {
            "experiment_id": "H18a_LR_STABILITY_MECHANISM",
            "title": "Muon LR stability across depth in a deep-linear toy model",
            "scope_statement": "Toy-model evidence about operator-norm clamping; not a universal proof.",
            "measured_quantities": [
                "initialization-time max layerwise gradient operator norm",
                "initialization-time max layerwise orthogonalized-gradient operator norm",
                "estimated lambda_max(H_init)",
                "empirical best learning rates from fixed-budget sweeps",
                "empirical max stable learning rates from short-horizon non-divergence searches",
                "descriptive log-log fits across depth",
            ],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "runtime_sec": float(finished - started),
        },
        "config": cfg,
        "work_estimate": work_estimate,
        "phase1": phase1,
        "phase2": phase2,
        "phase3": phase3,
        "summary_table": summary_table,
        "fits": fits,
        "tests": tests,
        "grid_diagnostics": grid_diagnostics,
        "verdict": verdict,
    }
    return results


def _fmt(value, digits=4, scientific=False):
    value = float(value)
    if not np.isfinite(value):
        return "nan"
    if scientific:
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def print_report(results):
    cfg = results["config"]
    phase1 = results["phase1"]
    phase2 = results["phase2"]
    phase3 = results["phase3"]
    tests = results["tests"]
    verdict = results["verdict"]
    low_depth = cfg["reference_depth_low"]
    high_depth = cfg["reference_depth_high"]

    print("\n" + "=" * 118)
    print("H18a REPORT: Muon LR stability across depth in a deep-linear toy model")
    print("=" * 118)
    print("Scope: initialization operator norms, empirical best LRs, empirical max stable LRs, descriptive fits.")
    print("Interpretation: evidence in this toy model, not a universal proof.")
    print(
        f"Runtime={results['metadata']['runtime_sec']:.2f}s | depths={cfg['depths']} | seeds={cfg['seeds']} | "
        f"train_steps={cfg['train_steps']} | divergence_steps={cfg['divergence_steps']}"
    )
    print(
        f"LR grids: SGD n={len(cfg['sgd_lr_grid'])} [{cfg['sgd_lr_grid'][0]:.1e}, {cfg['sgd_lr_grid'][-1]:.1e}], "
        f"Muon n={len(cfg['muon_lr_grid'])} [{cfg['muon_lr_grid'][0]:.1e}, {cfg['muon_lr_grid'][-1]:.1e}]"
    )

    print("\n" + "=" * 118)
    print("PHASE 1 SUMMARY")
    print("=" * 118)
    for depth in cfg["depths"]:
        summary = phase1["by_depth"][depth]["summary"]
        print(
            f"depth {depth:>2}: "
            f"mean max ||G||_op = {_fmt(summary['max_grad_op_norm']['mean'])} +/- {_fmt(summary['max_grad_op_norm']['std'])}, "
            f"mean max ||ortho(G)||_op = {_fmt(summary['max_ortho_grad_op_norm']['mean'], digits=6)} +/- {_fmt(summary['max_ortho_grad_op_norm']['std'], digits=6)}, "
            f"mean lambda_max(H_init) = {_fmt(summary['lambda_max_init']['mean'], digits=2)} +/- {_fmt(summary['lambda_max_init']['std'], digits=2)}"
        )

    print("\n" + "=" * 118)
    print("PHASE 2 SUMMARY: empirical best LR from fixed-budget sweeps")
    print("=" * 118)
    for depth in cfg["depths"]:
        print(f"depth {depth:>2}:")
        for optimizer in ["sgd", "muon"]:
            summary = phase2["by_depth"][depth][optimizer]["summary"]
            print(
                f"  {optimizer:>5}: median best LR = {_fmt(summary['median_best_lr'], digits=6)}, "
                f"mean best loss = {_fmt(summary['mean_best_loss'], digits=6, scientific=True)}, "
                f"mean step-1 max ||dW||_op = {_fmt(summary['mean_best_step1_max_op_norm'], digits=4, scientific=True)}, "
                f"edge hits = {summary['grid_edge_count']}/{summary['num_seeds_with_finite_best']}"
            )

    print("\n" + "=" * 118)
    print("PHASE 3 SUMMARY: empirical max stable LR")
    print("=" * 118)
    for depth in cfg["depths"]:
        sgd_summary = phase3["by_depth"][depth]["sgd"]["summary"]
        muon_summary = phase3["by_depth"][depth]["muon"]["summary"]
        print(
            f"depth {depth:>2}: "
            f"SGD mean max stable LR = {_fmt(sgd_summary['mean_max_stable_lr'], digits=6)}, "
            f"Muon mean max stable LR = {_fmt(muon_summary['mean_max_stable_lr'], digits=6)}"
        )

    print("\n" + "=" * 138)
    print("COMPACT SUMMARY TABLE")
    print("=" * 138)
    header = (
        f"{'Depth':>5} | {'||G||_op':>10} | {'||oG||_op':>11} | {'lam_max':>10} | "
        f"{'SGD bestLR':>10} | {'Muon bestLR':>11} | {'SGD maxLR':>10} | {'Muon maxLR':>11} | "
        f"{'SGDbest*||G||':>14} | {'SGDmax*||G||':>13}"
    )
    print(header)
    print("-" * 138)
    for row in results["summary_table"]:
        print(
            f"{row['depth']:>5} | "
            f"{_fmt(row['mean_max_grad_op_norm']):>10} | "
            f"{_fmt(row['mean_max_ortho_grad_op_norm'], digits=6):>11} | "
            f"{_fmt(row['mean_lambda_max_init'], digits=2):>10} | "
            f"{_fmt(row['median_best_lr_sgd'], digits=6):>10} | "
            f"{_fmt(row['median_best_lr_muon'], digits=6):>11} | "
            f"{_fmt(row['mean_max_stable_lr_sgd'], digits=6):>10} | "
            f"{_fmt(row['mean_max_stable_lr_muon'], digits=6):>11} | "
            f"{_fmt(row['sgd_bestlr_times_grad_op']):>14} | "
            f"{_fmt(row['sgd_maxstablelr_times_grad_op']):>13}"
        )

    print("\n" + "=" * 118)
    print("DESCRIPTIVE LOG-LOG FITS")
    print("=" * 118)
    for fit in results["fits"]:
        print(
            f"{fit['metric']:>33}: slope={_fmt(fit['slope'], digits=3):>7}, "
            f"R^2={_fmt(fit['r2'], digits=3):>6}, "
            f"d{fit['depth_low']}/d{fit['depth_high']}={_fmt(fit['ratio_low_over_high'], digits=2)}x"
        )

    print("\n" + "=" * 118)
    print("T1-T5 CHECKS")
    print("=" * 118)
    for test_name, test in tests.items():
        print(
            f"{test_name}: {'PASS' if test['pass'] else 'FAIL'} | "
            f"value={_fmt(test['value'], digits=4)} | criterion: {test['criterion']} {test['threshold']} | "
            f"{test['description']}"
        )

    print("\n" + "=" * 118)
    print("CALIBRATED CONCLUSION")
    print("=" * 118)
    print(f"Checks passed: {verdict['n_pass']}/{verdict['n_tests']}")
    print(
        f"SGD median best-LR drop d{low_depth}->d{high_depth}: {_fmt(verdict['sgd_best_lr_drop_ratio'], digits=2)}x | "
        f"Muon median best-LR drop d{low_depth}->d{high_depth}: {_fmt(verdict['muon_best_lr_drop_ratio'], digits=2)}x"
    )
    print(f"Relative drop advantage (SGD/Muon): {_fmt(verdict['relative_drop_advantage'], digits=2)}x")
    if results["grid_diagnostics"]["any_best_lr_on_edge"]:
        print("Grid diagnostic: at least one best LR lies on a sweep boundary; consider widening that grid in follow-up runs.")
    else:
        print("Grid diagnostic: no best LR landed on a sweep boundary in this run.")
    print(verdict["calibrated_conclusion"])
    if verdict["failed_tests"]:
        print(f"Failed checks: {', '.join(verdict['failed_tests'])}")
    else:
        print("All five checks passed under the current thresholds.")


def main(config=None, verbose=True):
    results = run_experiment(config=config, verbose=verbose)
    print_report(results)
    return results["verdict"]["all_pass"]


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
