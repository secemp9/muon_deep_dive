#!/usr/bin/env python3
"""
Experiment 3.2: Democracy Budget -- Partial Equalization
=========================================================

Toy final-loss recovery probe for the 48-parameter deep-linear benchmark used
in the surrounding Tier-1 mechanism tests.

Question:
    If full Hessian-basis equalization can recover or exceed Muon's advantage
    over SGD, how much of that effect survives when only the spectral extremes
    are equalized?

Protocol:
    - 3-layer 4x4 deep linear network (48 parameters total)
    - ill-conditioned target with singular values [100, 10, 1, 0.1]
    - optimizers: SGD, Muon, full democratic equalization, partial democratic
      equalization
    - partial sweep over k in {1, 2, 3, 5, 10, 15, 24}; SGD is the conceptual
      no-equalization baseline, but k=0 is not implemented in the partial sweep
    - 500 optimization steps; Hessian recomputed every 50 steps
    - learning rate chosen independently for each method on each seed
    - headline metric: final-loss recovery relative to Muon's gain over SGD

This script is import-safe and exposes:
    - prepare_config(...)
    - make_seed_list(...)
    - run_experiment(...)
    - summarize_results(...)
    - print_report(...)
    - main()
"""

import copy
import time

import numpy as np


DEFAULT_CONFIG = {
    "dim": 4,
    "n_layers": 3,
    "n_steps": 500,
    "hessian_recompute_every": 50,
    "n_seeds": 5,
    "seed_base": 42,
    "seed_stride": 7,
    "k_values": [1, 2, 3, 5, 10, 15, 24],
    "lr_candidates": [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
    "target_singular_values": [100.0, 10.0, 1.0, 0.1],
    "init_scale": 0.3,
    "hessian_eps": 1e-5,
    "divergence_threshold": 1e8,
}


def prepare_config(config_overrides=None):
    """Return a validated configuration dictionary for the toy experiment."""
    config = copy.deepcopy(DEFAULT_CONFIG)
    if config_overrides:
        config.update(config_overrides)

    config["dim"] = int(config["dim"])
    config["n_layers"] = int(config["n_layers"])
    config["n_steps"] = int(config["n_steps"])
    config["hessian_recompute_every"] = int(config["hessian_recompute_every"])
    config["n_seeds"] = int(config["n_seeds"])
    config["seed_base"] = int(config["seed_base"])
    config["seed_stride"] = int(config["seed_stride"])
    config["k_values"] = [int(k) for k in config["k_values"]]
    config["lr_candidates"] = [float(lr) for lr in config["lr_candidates"]]
    config["target_singular_values"] = [float(s) for s in config["target_singular_values"]]
    config["init_scale"] = float(config["init_scale"])
    config["hessian_eps"] = float(config["hessian_eps"])
    config["divergence_threshold"] = float(config["divergence_threshold"])
    config["n_params"] = config["n_layers"] * config["dim"] * config["dim"]

    if config["dim"] <= 0 or config["n_layers"] <= 0:
        raise ValueError("dim and n_layers must be positive.")
    if config["n_steps"] <= 0:
        raise ValueError("n_steps must be positive.")
    if config["hessian_recompute_every"] <= 0:
        raise ValueError("hessian_recompute_every must be positive.")
    if config["n_seeds"] <= 0:
        raise ValueError("n_seeds must be positive.")
    if len(config["target_singular_values"]) != config["dim"]:
        raise ValueError("target_singular_values must have length equal to dim.")
    if any(k <= 0 for k in config["k_values"]):
        raise ValueError(
            "All k_values must be positive. SGD is the conceptual k=0 baseline, "
            "but k=0 is not implemented in the partial-equalization sweep."
        )
    if not config["lr_candidates"]:
        raise ValueError("lr_candidates must be non-empty.")

    return config


def make_seed_list(config):
    """Return the deterministic seed list for the experiment."""
    return [config["seed_base"] + idx * config["seed_stride"] for idx in range(config["n_seeds"])]


# =============================================================================
# NETWORK / LOSS / GRAD / HESSIAN
# =============================================================================


def unpack(theta, dim, n_layers):
    weights = []
    idx = 0
    block = dim * dim
    for _ in range(n_layers):
        weights.append(theta[idx:idx + block].reshape(dim, dim))
        idx += block
    return weights


def forward(weights):
    out = weights[0]
    for weight in weights[1:]:
        out = weight @ out
    return out


def loss_fn(theta, target, dim, n_layers):
    weights = unpack(theta, dim, n_layers)
    diff = forward(weights) - target
    return 0.5 * np.sum(diff ** 2)


def grad_fn(theta, target, dim, n_layers):
    weights = unpack(theta, dim, n_layers)
    product = forward(weights)
    residual = product - target
    grads = []

    for layer_idx in range(n_layers):
        left = np.eye(dim)
        for j in range(layer_idx + 1, n_layers):
            left = weights[j] @ left

        right = np.eye(dim)
        for j in range(0, layer_idx):
            right = weights[j] @ right

        d_w = left.T @ residual @ right.T
        grads.append(d_w.ravel())

    return np.concatenate(grads)


def grad_matrices(theta, target, dim, n_layers):
    grad_vec = grad_fn(theta, target, dim, n_layers)
    block = dim * dim
    return [grad_vec[layer_idx * block:(layer_idx + 1) * block].reshape(dim, dim) for layer_idx in range(n_layers)]


def hessian_fn(theta, target, dim, n_layers, eps=1e-5):
    n = len(theta)
    hessian = np.zeros((n, n))
    for i in range(n):
        theta_p = theta.copy()
        theta_m = theta.copy()
        theta_p[i] += eps
        theta_m[i] -= eps
        g_p = grad_fn(theta_p, target, dim, n_layers)
        g_m = grad_fn(theta_m, target, dim, n_layers)
        hessian[:, i] = (g_p - g_m) / (2 * eps)
    return 0.5 * (hessian + hessian.T)


# =============================================================================
# MUON DIRECTION
# =============================================================================


def polar_factor_svd(matrix):
    u, _, vt = np.linalg.svd(matrix, full_matrices=True)
    return u @ vt


def muon_direction(theta, target, dim, n_layers):
    grad_blocks = grad_matrices(theta, target, dim, n_layers)
    return np.concatenate([polar_factor_svd(block).ravel() for block in grad_blocks])


# =============================================================================
# DEMOCRATIC DIRECTIONS
# =============================================================================


def partial_democratic_direction(grad_vec, eigvecs, eigvals=None, k=None):
    """
    Equalize only the bottom-k and top-k Hessian eigenvector components.

    The eigenpairs are assumed to come from ``np.linalg.eigh``, so the columns
    of ``eigvecs`` are ordered from smallest to largest eigenvalue.

    Notes
    -----
    - k=0 is intentionally not supported here. In this experiment, SGD plays the
      conceptual no-equalization baseline.
    - If 2*k >= n, all components are equalized, matching the full democratic
      direction before norm matching.
    """
    if k is None:
        raise ValueError("PartialDemocratic requires an explicit k value.")
    if k <= 0:
        raise ValueError("k must be positive; the partial sweep does not implement k=0.")

    n = len(grad_vec)
    if eigvecs.shape != (n, n):
        raise ValueError("eigvecs must be square with the same dimension as grad_vec.")
    if eigvals is not None and len(eigvals) != n:
        raise ValueError("eigvals must have the same dimension as grad_vec.")

    projs = eigvecs.T @ grad_vec

    if 2 * k >= n:
        selected = list(range(n))
    else:
        selected = list(range(k)) + list(range(n - k, n))

    selected_magnitudes = np.abs(projs[selected])
    mean_magnitude = np.mean(selected_magnitudes)

    equalized_projs = projs.copy()
    for idx in selected:
        equalized_projs[idx] = np.sign(projs[idx]) * mean_magnitude

    return eigvecs @ equalized_projs


def full_democratic_direction(grad_vec, eigvecs):
    """Equalize all Hessian-basis projection magnitudes."""
    projs = eigvecs.T @ grad_vec
    equalized_projs = np.sign(projs) * np.mean(np.abs(projs))
    return eigvecs @ equalized_projs


# =============================================================================
# TRAINING ENGINE
# =============================================================================


def _match_gradient_norm(direction, grad_vec):
    direction_norm = np.linalg.norm(direction)
    grad_norm = np.linalg.norm(grad_vec)
    if direction_norm > 1e-12:
        return direction * (grad_norm / direction_norm)
    return direction


def run_method(method, lr, theta0, target, config, k=None):
    """Run one method and return its loss trajectory (recorded pre-update)."""
    theta = theta0.copy()
    eigvecs = None
    eigvals = None
    losses = []

    for step in range(config["n_steps"]):
        loss_value = loss_fn(theta, target, config["dim"], config["n_layers"])
        losses.append(float(loss_value))

        if np.isnan(loss_value) or loss_value > config["divergence_threshold"]:
            losses.extend([config["divergence_threshold"]] * (config["n_steps"] - step - 1))
            break

        grad_vec = grad_fn(theta, target, config["dim"], config["n_layers"])

        if step % config["hessian_recompute_every"] == 0:
            hessian = hessian_fn(
                theta,
                target,
                config["dim"],
                config["n_layers"],
                eps=config["hessian_eps"],
            )
            eigvals, eigvecs = np.linalg.eigh(hessian)

        if method == "SGD":
            direction = grad_vec
        elif method == "Muon":
            direction = muon_direction(theta, target, config["dim"], config["n_layers"])
        elif method == "DemocraticSGD_full":
            direction = full_democratic_direction(grad_vec, eigvecs)
            direction = _match_gradient_norm(direction, grad_vec)
        elif method == "PartialDemocratic":
            direction = partial_democratic_direction(grad_vec, eigvecs, eigvals=eigvals, k=k)
            direction = _match_gradient_norm(direction, grad_vec)
        else:
            raise ValueError(f"Unknown method: {method}")

        theta -= lr * direction

    return np.asarray(losses, dtype=float)


def find_best_lr(method, theta0, target, config, k=None):
    """Grid search over the configured learning-rate candidates."""
    best_loss = np.inf
    best_lr = config["lr_candidates"][0]

    for lr in config["lr_candidates"]:
        losses = run_method(method, lr, theta0, target, config, k=k)
        final_loss = float(losses[-1])
        if final_loss < best_loss:
            best_loss = final_loss
            best_lr = float(lr)

    return best_lr


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================


def _compute_recovery(sgd_final, method_final, muon_final):
    gap = sgd_final - muon_final
    if gap > 1e-12:
        return float((sgd_final - method_final) / gap * 100.0)
    return 0.0


def _make_target_and_init(seed, config):
    rng = np.random.RandomState(seed)
    dim = config["dim"]

    u_t, _ = np.linalg.qr(rng.randn(dim, dim))
    v_t, _ = np.linalg.qr(rng.randn(dim, dim))
    sigma_t = np.asarray(config["target_singular_values"], dtype=float)
    target = u_t @ np.diag(sigma_t) @ v_t
    theta0 = config["init_scale"] * rng.randn(config["n_params"])

    return target, theta0


def _print_header(config):
    print("=" * 80)
    print("Experiment 3.2: Democracy Budget -- Partial Equalization")
    print("=" * 80)
    print(
        f"Network: {config['n_layers']}-layer deep linear {config['dim']}x{config['dim']} "
        f"({config['n_params']} params)"
    )
    print(
        f"Steps: {config['n_steps']}, Hessian recompute every "
        f"{config['hessian_recompute_every']}"
    )
    print(f"Seeds: {config['n_seeds']} -> {make_seed_list(config)}")
    print(f"k values: {config['k_values']}")
    print("Metric: final-loss recovery relative to Muon's gain over SGD")
    print("Note: SGD is the conceptual k=0 baseline; the implemented sweep starts at k=1.")
    print()


def run_experiment(config_overrides=None, verbose=True, store_trajectories=True):
    """
    Execute the toy democracy-budget experiment and return structured results.

    Parameters
    ----------
    config_overrides : dict or None
        Optional overrides for ``DEFAULT_CONFIG``.
    verbose : bool
        If True, print per-seed progress messages.
    store_trajectories : bool
        If True, store best-run loss trajectories for notebook analysis.
    """
    config = prepare_config(config_overrides)
    seeds = make_seed_list(config)
    start_time = time.time()

    if verbose:
        _print_header(config)

    aggregate_arrays = {
        "SGD_final_losses": [],
        "Muon_final_losses": [],
        "DemocraticSGD_full_final_losses": [],
        "DemocraticSGD_full_recoveries": [],
        "PartialDemocratic_final_losses": {k: [] for k in config["k_values"]},
        "PartialDemocratic_recoveries": {k: [] for k in config["k_values"]},
    }
    seed_results = []

    for seed in seeds:
        target, theta0 = _make_target_and_init(seed, config)
        target_condition_number = float(np.linalg.cond(target))

        if verbose:
            print(f"--- Seed {seed} (target cond={target_condition_number:.0f}) ---")

        lr_sgd = find_best_lr("SGD", theta0, target, config)
        lr_muon = find_best_lr("Muon", theta0, target, config)
        lr_dem_full = find_best_lr("DemocraticSGD_full", theta0, target, config)
        lr_partial = {
            k: find_best_lr("PartialDemocratic", theta0, target, config, k=k)
            for k in config["k_values"]
        }

        if verbose:
            print(f"  LRs: SGD={lr_sgd}, Muon={lr_muon}, DemFull={lr_dem_full}")
            print(f"  Partial LRs: {lr_partial}")

        sgd_losses = run_method("SGD", lr_sgd, theta0, target, config)
        muon_losses = run_method("Muon", lr_muon, theta0, target, config)
        dem_full_losses = run_method("DemocraticSGD_full", lr_dem_full, theta0, target, config)

        sgd_final = float(sgd_losses[-1])
        muon_final = float(muon_losses[-1])
        dem_full_final = float(dem_full_losses[-1])
        dem_full_recovery = _compute_recovery(sgd_final, dem_full_final, muon_final)

        aggregate_arrays["SGD_final_losses"].append(sgd_final)
        aggregate_arrays["Muon_final_losses"].append(muon_final)
        aggregate_arrays["DemocraticSGD_full_final_losses"].append(dem_full_final)
        aggregate_arrays["DemocraticSGD_full_recoveries"].append(dem_full_recovery)

        partial_finals = {}
        partial_recoveries = {}
        partial_trajectories = {} if store_trajectories else None

        for k in config["k_values"]:
            partial_losses = run_method("PartialDemocratic", lr_partial[k], theta0, target, config, k=k)
            partial_final = float(partial_losses[-1])
            partial_recovery = _compute_recovery(sgd_final, partial_final, muon_final)

            aggregate_arrays["PartialDemocratic_final_losses"][k].append(partial_final)
            aggregate_arrays["PartialDemocratic_recoveries"][k].append(partial_recovery)

            partial_finals[k] = partial_final
            partial_recoveries[k] = partial_recovery
            if store_trajectories:
                partial_trajectories[k] = partial_losses.tolist()

        best_run_trajectories = None
        if store_trajectories:
            best_run_trajectories = {
                "SGD": sgd_losses.tolist(),
                "Muon": muon_losses.tolist(),
                "DemocraticSGD_full": dem_full_losses.tolist(),
                "PartialDemocratic": partial_trajectories,
            }

        seed_result = {
            "seed": seed,
            "target_condition_number": target_condition_number,
            "best_lrs": {
                "SGD": float(lr_sgd),
                "Muon": float(lr_muon),
                "DemocraticSGD_full": float(lr_dem_full),
                "PartialDemocratic": {k: float(lr_partial[k]) for k in config["k_values"]},
            },
            "final_losses": {
                "SGD": sgd_final,
                "Muon": muon_final,
                "DemocraticSGD_full": dem_full_final,
                "PartialDemocratic": partial_finals,
            },
            "recoveries": {
                "DemocraticSGD_full": dem_full_recovery,
                "PartialDemocratic": partial_recoveries,
            },
            "best_run_trajectories": best_run_trajectories,
        }
        seed_results.append(seed_result)

        if verbose:
            print(
                f"  SGD={sgd_final:.4f}, Muon={muon_final:.4f}, "
                f"DemFull={dem_full_final:.4f} (rec={dem_full_recovery:.1f}%)"
            )
            partial_text = ", ".join(
                [f"k={k}:{partial_recoveries[k]:.1f}%" for k in config["k_values"]]
            )
            print(f"  Partial: {partial_text}")
            print()

    run_time_seconds = float(time.time() - start_time)
    return {
        "experiment_name": "Experiment 3.2: Democracy Budget -- Partial Equalization",
        "scope": "toy final-loss recovery probe",
        "recovery_definition": "(L_SGD - L_method) / (L_SGD - L_Muon) * 100",
        "config": config,
        "seeds": seeds,
        "store_trajectories": bool(store_trajectories),
        "seed_results": seed_results,
        "aggregate_arrays": aggregate_arrays,
        "run_time_seconds": run_time_seconds,
    }


# =============================================================================
# SUMMARIES / REPORTING
# =============================================================================


def _stats(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "sem": np.nan}

    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if arr.size > 1:
        sem = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
    else:
        sem = 0.0
    return {"n": int(arr.size), "mean": mean, "std": std, "sem": sem}


def summarize_results(results):
    """Compute aggregate tables and operational T1-T4 summaries."""
    config = results["config"]
    arrays = results["aggregate_arrays"]
    n_seeds = len(results["seeds"])

    sgd_stats = _stats(arrays["SGD_final_losses"])
    muon_stats = _stats(arrays["Muon_final_losses"])
    dem_full_stats = _stats(arrays["DemocraticSGD_full_final_losses"])
    dem_full_recovery_stats = _stats(arrays["DemocraticSGD_full_recoveries"])

    method_summary = [
        {
            "method": "SGD",
            "label": "SGD",
            "partial_k": None,
            "n_equalized": 0,
            "pct_equalized": 0.0,
            "mean_final_loss": sgd_stats["mean"],
            "std_final_loss": sgd_stats["std"],
            "sem_final_loss": sgd_stats["sem"],
            "mean_recovery": None,
            "std_recovery": None,
            "sem_recovery": None,
            "note": "Conceptual k=0 baseline; not part of the implemented partial sweep.",
        },
        {
            "method": "Muon",
            "label": "Muon",
            "partial_k": None,
            "n_equalized": None,
            "pct_equalized": None,
            "mean_final_loss": muon_stats["mean"],
            "std_final_loss": muon_stats["std"],
            "sem_final_loss": muon_stats["sem"],
            "mean_recovery": 100.0,
            "std_recovery": 0.0,
            "sem_recovery": 0.0,
            "note": "Reference point for the recovery metric.",
        },
        {
            "method": "DemocraticSGD_full",
            "label": "Democratic full",
            "partial_k": config["n_params"] // 2,
            "n_equalized": config["n_params"],
            "pct_equalized": 100.0,
            "mean_final_loss": dem_full_stats["mean"],
            "std_final_loss": dem_full_stats["std"],
            "sem_final_loss": dem_full_stats["sem"],
            "mean_recovery": dem_full_recovery_stats["mean"],
            "std_recovery": dem_full_recovery_stats["std"],
            "sem_recovery": dem_full_recovery_stats["sem"],
            "note": "All Hessian-basis components equalized before norm matching.",
        },
    ]

    recovery_by_k = []
    mean_recoveries = []
    for k in config["k_values"]:
        n_equalized = min(2 * k, config["n_params"])
        pct_equalized = 100.0 * n_equalized / config["n_params"]
        partial_loss_stats = _stats(arrays["PartialDemocratic_final_losses"][k])
        partial_recovery_stats = _stats(arrays["PartialDemocratic_recoveries"][k])

        method_summary.append(
            {
                "method": "PartialDemocratic",
                "label": f"Partial k={k}",
                "partial_k": k,
                "n_equalized": n_equalized,
                "pct_equalized": pct_equalized,
                "mean_final_loss": partial_loss_stats["mean"],
                "std_final_loss": partial_loss_stats["std"],
                "sem_final_loss": partial_loss_stats["sem"],
                "mean_recovery": partial_recovery_stats["mean"],
                "std_recovery": partial_recovery_stats["std"],
                "sem_recovery": partial_recovery_stats["sem"],
                "note": "Equalizes only the bottom-k and top-k Hessian eigendirections.",
            }
        )

        recovery_by_k.append(
            {
                "k": k,
                "n_equalized": n_equalized,
                "pct_equalized": pct_equalized,
                "mean_recovery": partial_recovery_stats["mean"],
                "std_recovery": partial_recovery_stats["std"],
                "sem_recovery": partial_recovery_stats["sem"],
                "per_seed_recoveries": list(arrays["PartialDemocratic_recoveries"][k]),
                "mean_final_loss": partial_loss_stats["mean"],
                "std_final_loss": partial_loss_stats["std"],
            }
        )
        mean_recoveries.append(partial_recovery_stats["mean"])

    min_k_100 = next((row["k"] for row in recovery_by_k if row["mean_recovery"] > 100.0), None)
    best_k_by_mean_recovery = recovery_by_k[int(np.argmax(mean_recoveries))]["k"] if recovery_by_k else None
    representative_partial_k = min_k_100 if min_k_100 is not None else best_k_by_mean_recovery

    monotonic_transitions = 0
    if len(mean_recoveries) >= 2:
        monotonic_transitions = sum(
            1 for i in range(1, len(mean_recoveries)) if mean_recoveries[i] >= mean_recoveries[i - 1] - 5.0
        )
        t2_pass = monotonic_transitions >= len(mean_recoveries) - 2
    else:
        t2_pass = True

    if 1 in config["k_values"]:
        full_like_k = config["k_values"][-1]
        k1_mean = next(row["mean_recovery"] for row in recovery_by_k if row["k"] == 1)
        kall_mean = next(row["mean_recovery"] for row in recovery_by_k if row["k"] == full_like_k)
        t3_pass = k1_mean < kall_mean - 10.0
        t3_metric = f"k=1 mean={k1_mean:.1f}%, k={full_like_k} mean={kall_mean:.1f}%"
    else:
        t3_pass = None
        t3_metric = "k=1 not included in current k_values."

    hypothesis_tests = [
        {
            "test": "T1",
            "statement": "Full equalization mean recovery exceeds 100%.",
            "status": "PASS" if dem_full_recovery_stats["mean"] > 100.0 else "FAIL",
            "metric": f"Mean full-democratic recovery = {dem_full_recovery_stats['mean']:.1f}%",
            "caveat": f"Operational threshold on the mean only; n={n_seeds} seeds.",
        },
        {
            "test": "T2",
            "statement": "Mean recovery is mostly monotone with k (allowing small slack).",
            "status": "PASS" if t2_pass else "FAIL",
            "metric": f"Monotonic transitions = {monotonic_transitions}/{max(len(mean_recoveries) - 1, 0)}",
            "caveat": "This is a heuristic trend check, not a formal curvature/shape test.",
        },
        {
            "test": "T3",
            "statement": "k=1 is materially worse than the largest implemented k.",
            "status": "PASS" if t3_pass is True else ("FAIL" if t3_pass is False else "N/A"),
            "metric": t3_metric,
            "caveat": "This compares means only and does not isolate a mechanism.",
        },
        {
            "test": "T4",
            "statement": "Smallest implemented k whose mean recovery exceeds 100%.",
            "status": str(min_k_100) if min_k_100 is not None else "Not reached",
            "metric": (
                f"First k above 100% mean recovery = {min_k_100}" if min_k_100 is not None
                else f"Best mean recovery = {max(mean_recoveries):.1f}% at k={best_k_by_mean_recovery}"
            ),
            "caveat": "This is not a confidence-qualified threshold claim.",
        },
    ]

    per_seed_final_losses = []
    per_seed_best_lrs = []
    per_seed_recoveries = []
    for seed_result in results["seed_results"]:
        loss_row = {
            "seed": seed_result["seed"],
            "target_condition_number": seed_result["target_condition_number"],
            "SGD": seed_result["final_losses"]["SGD"],
            "Muon": seed_result["final_losses"]["Muon"],
            "Democratic_full": seed_result["final_losses"]["DemocraticSGD_full"],
        }
        lr_row = {
            "seed": seed_result["seed"],
            "SGD": seed_result["best_lrs"]["SGD"],
            "Muon": seed_result["best_lrs"]["Muon"],
            "Democratic_full": seed_result["best_lrs"]["DemocraticSGD_full"],
        }
        recovery_row = {
            "seed": seed_result["seed"],
            "Democratic_full_recovery": seed_result["recoveries"]["DemocraticSGD_full"],
        }

        for k in config["k_values"]:
            loss_row[f"Partial_k_{k}"] = seed_result["final_losses"]["PartialDemocratic"][k]
            lr_row[f"Partial_k_{k}"] = seed_result["best_lrs"]["PartialDemocratic"][k]
            recovery_row[f"Partial_k_{k}"] = seed_result["recoveries"]["PartialDemocratic"][k]

        per_seed_final_losses.append(loss_row)
        per_seed_best_lrs.append(lr_row)
        per_seed_recoveries.append(recovery_row)

    return {
        "config": config,
        "n_seeds": n_seeds,
        "run_time_seconds": results["run_time_seconds"],
        "aggregate": {
            "sgd_mean_final_loss": sgd_stats["mean"],
            "muon_mean_final_loss": muon_stats["mean"],
            "democratic_full_mean_final_loss": dem_full_stats["mean"],
            "democratic_full_mean_recovery": dem_full_recovery_stats["mean"],
        },
        "method_summary": method_summary,
        "recovery_by_k": recovery_by_k,
        "per_seed_final_losses": per_seed_final_losses,
        "per_seed_best_lrs": per_seed_best_lrs,
        "per_seed_recoveries": per_seed_recoveries,
        "hypothesis_tests": hypothesis_tests,
        "min_k_100_mean_recovery": min_k_100,
        "best_k_by_mean_recovery": best_k_by_mean_recovery,
        "representative_partial_k": representative_partial_k,
    }


def print_report(results, summary):
    """Print a CLI-friendly report that mirrors the notebook's core statistics."""
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS ACROSS ALL SEEDS")
    print("=" * 80)
    print(f"\n{'Method':<24} {'Mean Final Loss':>16} {'Std':>10} {'Mean Recovery':>15}")
    print("-" * 75)

    for row in summary["method_summary"]:
        mean_recovery = row["mean_recovery"]
        recovery_text = "(baseline)" if row["method"] == "SGD" else (
            "(reference)" if row["method"] == "Muon" else f"{mean_recovery:>14.1f}%"
        )
        print(
            f"{row['label']:<24} {row['mean_final_loss']:>16.6f} "
            f"{row['std_final_loss']:>10.6f} {recovery_text:>15}"
        )

    print(f"\n{'=' * 80}")
    print("RECOVERY % vs k (implemented partial sweep only)")
    print(f"{'=' * 80}")
    print(f"\n{'k':>4} {'Equalized':>10} {'% of N':>8} {'Recovery %':>12} {'Per-seed recoveries'}")
    print("-" * 70)
    for row in summary["recovery_by_k"]:
        per_seed = "  ".join([f"{value:.1f}" for value in row["per_seed_recoveries"]])
        print(
            f"{row['k']:>4} {row['n_equalized']:>10} {row['pct_equalized']:>7.0f}% "
            f"{row['mean_recovery']:>11.1f}% {per_seed:>30}"
        )

    print(f"\n{'=' * 80}")
    print("HYPOTHESIS TESTS")
    print(f"{'=' * 80}")
    for row in summary["hypothesis_tests"]:
        print(f"\n{row['test']}: {row['statement']}")
        print(f"    Result: {row['status']}")
        print(f"    Metric: {row['metric']}")
        print(f"    Caveat: {row['caveat']}")

    print(f"\n{'=' * 80}")
    print("FINAL VERDICT: DEMOCRACY BUDGET")
    print(f"{'=' * 80}")
    full_recovery = summary["aggregate"]["democratic_full_mean_recovery"]
    min_k_100 = summary["min_k_100_mean_recovery"]
    print(
        f"\n  QUESTION: How many Hessian eigenvectors need equalizing to match Muon?\n\n"
        f"  Full equalization mean recovery: {full_recovery:.1f}%\n"
        f"  Smallest implemented k with mean recovery > 100%: {min_k_100 if min_k_100 is not None else 'N/A'}\n"
    )
    print("  Recovery curve (text summary; notebook supplies the actual figures):")
    for row in summary["recovery_by_k"]:
        bar = '#' * max(int(max(row["mean_recovery"], 0.0) / 5), 0)
        marker = " <-- beats Muon in mean" if row["mean_recovery"] > 100.0 else ""
        print(
            f"    k={row['k']:>2} ({row['n_equalized']:>2} eigvecs): "
            f"{row['mean_recovery']:>6.1f}% {bar}{marker}"
        )

    pass_like = sum(test["status"] == "PASS" for test in summary["hypothesis_tests"][:3])
    print()
    if pass_like >= 2:
        print("  CONCLUSION: In this toy final-loss probe, the benefit of Hessian-basis")
        print("  equalization appears to emerge gradually as more spectral extremes are")
        print("  equalized. This does not by itself establish the mechanism.")
    else:
        print("  CONCLUSION: The current toy probe does not yet give a stable qualitative")
        print("  pattern under these operational checks.")
    print(f"\nRuntime: {results['run_time_seconds']:.2f} seconds")
    print("=" * 80)


def main(config_overrides=None):
    """CLI entrypoint that preserves the original experiment behavior."""
    results = run_experiment(config_overrides=config_overrides, verbose=True, store_trajectories=True)
    summary = summarize_results(results)
    print_report(results, summary)
    return results, summary


if __name__ == "__main__":
    main()
