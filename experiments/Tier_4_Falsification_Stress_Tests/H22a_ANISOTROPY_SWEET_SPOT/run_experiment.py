#!/usr/bin/env python3
"""
H22a: Toy target-spectrum anisotropy sweep for Muon in a 4-layer deep linear task
=================================================================================

This file preserves the original H22a toy study while making its scope and outputs
more honest and reusable.

What is actually controlled here:
  - The power-law exponent ``alpha`` of the target matrix spectrum.
  - This directly changes the target's singular-value decay, target effective rank,
    and target anisotropy.

What is only measured diagnostically here:
  - The effective rank of the *initial gradients* in the deep linear network.
  - In the current setup, that measured gradient-rank range is fairly narrow, so the
    study should not be interpreted as a clean direct sweep of gradient effective rank.

What the experiment does:
  - Build a 4-layer 32x32 deep linear regression problem from a target matrix with
    power-law singular values.
  - For each alpha, measure initial gradient effective rank on a few seeds.
  - Sweep learning rates separately for SGD and Muon.
  - Re-evaluate both optimizers on all evaluation seeds using the chosen learning rates.
  - Report raw per-seed losses, optimizer advantage, target diagnostics, and the
    current T1/T2 summary outcomes.

Important limitations:
  - This is a small-sample toy study (3 seeds for diagnostics/LR sweeps, 5 seeds for
    final evaluation by default).
  - Raw per-seed losses are preserved, but no formal confidence intervals are computed.
"""

from __future__ import annotations

import os
import time
from copy import deepcopy
from datetime import datetime, timezone

import numpy as np

SCRIPT_PATH = os.path.abspath(__file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)

DEFAULT_CONFIG = {
    "dim": 32,
    "num_layers": 4,
    "num_steps": 500,
    "momentum": 0.9,
    "ns_iters": 5,
    "num_seeds": 5,
    "diagnostic_num_seeds": 3,
    "lr_sweep_num_seeds": 3,
    "batch_size": 64,
    "input_scale": 0.3,
    "init_noise_scale": 0.1,
    "divergence_threshold": 1e10,
    "seed_base": 42,
    "seed_stride": 137,
    "data_seed_offset": 7000,
    "init_seed_offset": 5000,
    "alpha_values": [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0],
    "lr_sgd": np.logspace(-4, -1, 12).tolist(),
    "lr_muon": np.logspace(-4, -1, 12).tolist(),
    "t1_lower_frac": 0.2,
    "t1_upper_frac": 0.8,
}


def get_default_config():
    """Return a deep copy of the default experiment configuration."""
    return deepcopy(DEFAULT_CONFIG)


def resolve_config(config_overrides=None):
    """Merge optional config overrides into the default config."""
    config = get_default_config()
    if config_overrides:
        for key, value in config_overrides.items():
            config[key] = deepcopy(value)

    config["alpha_values"] = [float(x) for x in config["alpha_values"]]
    config["lr_sgd"] = [float(x) for x in config["lr_sgd"]]
    config["lr_muon"] = [float(x) for x in config["lr_muon"]]
    return config


def count_train_calls(config):
    """Estimate the number of train() calls for the current configuration."""
    n_alpha = len(config["alpha_values"])
    lr_sweep_calls = n_alpha * (
        len(config["lr_sgd"]) * config["lr_sweep_num_seeds"]
        + len(config["lr_muon"]) * config["lr_sweep_num_seeds"]
    )
    final_eval_calls = n_alpha * 2 * config["num_seeds"]
    return {
        "lr_sweep_train_calls": int(lr_sweep_calls),
        "final_eval_train_calls": int(final_eval_calls),
        "total_train_calls": int(lr_sweep_calls + final_eval_calls),
    }


def build_seed_schedule(config):
    """Construct the deterministic seed schedule used by the experiment."""
    seeds = [
        int(config["seed_base"] + i * config["seed_stride"])
        for i in range(config["num_seeds"])
    ]
    diagnostic_count = min(config["diagnostic_num_seeds"], len(seeds))
    lr_sweep_count = min(config["lr_sweep_num_seeds"], len(seeds))
    return {
        "all_seeds": seeds,
        "diagnostic_seeds": seeds[:diagnostic_count],
        "lr_sweep_seeds": seeds[:lr_sweep_count],
        "data_seed_offset": int(config["data_seed_offset"]),
        "init_seed_offset": int(config["init_seed_offset"]),
    }


def newton_schulz(M, n_iters):
    norm = np.linalg.norm(M, "fro")
    if norm < 1e-15:
        return M
    X = M / norm
    for _ in range(n_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


def effective_rank(M):
    """Compute effective rank = exp(entropy of normalized SV^2)."""
    sv = np.linalg.svd(M, compute_uv=False)
    sv2 = sv**2
    sv2 = sv2 / (np.sum(sv2) + 1e-30)
    sv2 = sv2[sv2 > 1e-30]
    entropy = -np.sum(sv2 * np.log(sv2))
    return float(np.exp(entropy))


def make_anisotropic_target(alpha, seed, dim):
    """Create a target matrix with power-law singular values sigma_i = i^(-alpha)."""
    rng = np.random.RandomState(seed)
    U, _ = np.linalg.qr(rng.randn(dim, dim))
    V, _ = np.linalg.qr(rng.randn(dim, dim))
    sigmas = np.array([(i + 1) ** (-alpha) for i in range(dim)], dtype=float)
    sigmas = sigmas / np.linalg.norm(sigmas) * np.sqrt(dim)
    return U @ np.diag(sigmas) @ V.T


def init_weights(seed, dim, num_layers, init_noise_scale):
    rng = np.random.RandomState(seed)
    return [np.eye(dim) + rng.randn(dim, dim) * init_noise_scale for _ in range(num_layers)]


def forward(weights, X):
    out = X.copy()
    for W in weights:
        out = W @ out
    return out


def compute_loss(weights, X, Y):
    pred = forward(weights, X)
    return float(0.5 * np.mean(np.sum((pred - Y) ** 2, axis=0)))


def compute_gradients(weights, X, Y):
    L = len(weights)
    N = X.shape[1]
    acts = [X.copy()]
    for W in weights:
        acts.append(W @ acts[-1])
    delta = (acts[-1] - Y) / N
    grads = [None] * L
    for l in range(L - 1, -1, -1):
        grads[l] = delta @ acts[l].T
        if l > 0:
            delta = weights[l].T @ delta
    return grads


def train(w0, X, Y, lr, opt, num_steps, momentum, ns_iters, divergence_threshold):
    weights = [W.copy() for W in w0]
    mom = [np.zeros_like(W) for W in weights]
    for _ in range(num_steps):
        if compute_loss(weights, X, Y) > divergence_threshold:
            return float("inf")
        grads = compute_gradients(weights, X, Y)
        for i in range(len(weights)):
            if opt == "muon":
                mom[i] = momentum * mom[i] + newton_schulz(grads[i], n_iters=ns_iters)
            else:
                mom[i] = momentum * mom[i] + grads[i]
            weights[i] -= lr * mom[i]
    return compute_loss(weights, X, Y)


def build_problem(alpha, seed, config):
    dim = config["dim"]
    target = make_anisotropic_target(alpha, seed, dim)
    rng = np.random.RandomState(seed + config["data_seed_offset"])
    X = rng.randn(dim, config["batch_size"]) * config["input_scale"]
    Y = target @ X
    w0 = init_weights(
        seed + config["init_seed_offset"],
        dim,
        config["num_layers"],
        config["init_noise_scale"],
    )
    return {
        "target": target,
        "X": X,
        "Y": Y,
        "w0": w0,
    }


def summarize_finite_values(values):
    finite = [float(v) for v in values if np.isfinite(v)]
    if finite:
        mean_value = float(np.mean(finite))
        std_value = float(np.std(finite))
    else:
        mean_value = float("inf")
        std_value = None
    return {
        "values": [float(v) for v in values],
        "finite_count": len(finite),
        "mean": mean_value,
        "std": std_value,
    }


def evaluate_init_gradient_ranks(problem_cache, diagnostic_seeds, dim):
    records = []
    for seed in diagnostic_seeds:
        grads = compute_gradients(
            problem_cache[seed]["w0"],
            problem_cache[seed]["X"],
            problem_cache[seed]["Y"],
        )
        layer_effective_ranks = [effective_rank(G) for G in grads]
        mean_layer_effective_rank = float(np.mean(layer_effective_ranks))
        records.append(
            {
                "seed": int(seed),
                "layer_effective_ranks": [float(v) for v in layer_effective_ranks],
                "mean_layer_effective_rank": mean_layer_effective_rank,
                "mean_layer_effective_rank_frac": mean_layer_effective_rank / dim,
            }
        )

    mean_values = [record["mean_layer_effective_rank"] for record in records]
    return {
        "per_seed": records,
        "mean": float(np.mean(mean_values)),
        "std": float(np.std(mean_values)) if records else None,
        "frac": float(np.mean(mean_values) / dim) if records else None,
    }


def sweep_learning_rates(problem_cache, sweep_seeds, grid, opt, config):
    sweep_rows = []
    best_lr = float(grid[-1])
    best_mean_loss = float("inf")

    for lr in grid:
        losses = []
        for seed in sweep_seeds:
            problem = problem_cache[seed]
            final_loss = train(
                problem["w0"],
                problem["X"],
                problem["Y"],
                float(lr),
                opt,
                config["num_steps"],
                config["momentum"],
                config["ns_iters"],
                config["divergence_threshold"],
            )
            losses.append(final_loss)

        summary = summarize_finite_values(losses)
        row = {
            "lr": float(lr),
            "losses": summary["values"],
            "finite_count": int(summary["finite_count"]),
            "mean_loss": float(summary["mean"]),
            "std_loss": summary["std"],
        }
        sweep_rows.append(row)

        if row["mean_loss"] < best_mean_loss:
            best_mean_loss = row["mean_loss"]
            best_lr = float(lr)

    return {
        "best_lr": best_lr,
        "best_mean_loss": best_mean_loss,
        "grid_results": sweep_rows,
    }


def evaluate_final_losses(problem_cache, seeds, best_lrs, config):
    per_seed_rows = []
    sgd_losses = []
    muon_losses = []

    for seed in seeds:
        problem = problem_cache[seed]
        sgd_loss = train(
            problem["w0"],
            problem["X"],
            problem["Y"],
            best_lrs["sgd"],
            "sgd",
            config["num_steps"],
            config["momentum"],
            config["ns_iters"],
            config["divergence_threshold"],
        )
        muon_loss = train(
            problem["w0"],
            problem["X"],
            problem["Y"],
            best_lrs["muon"],
            "muon",
            config["num_steps"],
            config["momentum"],
            config["ns_iters"],
            config["divergence_threshold"],
        )
        sgd_losses.append(sgd_loss)
        muon_losses.append(muon_loss)
        per_seed_rows.append(
            {
                "seed": int(seed),
                "sgd_final_loss": float(sgd_loss),
                "muon_final_loss": float(muon_loss),
                "sgd_over_muon": float(sgd_loss / max(muon_loss, 1e-30)) if np.isfinite(sgd_loss) else float("inf"),
            }
        )

    sgd_summary = summarize_finite_values(sgd_losses)
    muon_summary = summarize_finite_values(muon_losses)

    advantage_ratio = float(sgd_summary["mean"] / max(muon_summary["mean"], 1e-30))
    return {
        "per_seed": per_seed_rows,
        "sgd_losses": sgd_summary["values"],
        "muon_losses": muon_summary["values"],
        "sgd_mean": float(sgd_summary["mean"]),
        "sgd_std": sgd_summary["std"],
        "sgd_finite_count": int(sgd_summary["finite_count"]),
        "muon_mean": float(muon_summary["mean"]),
        "muon_std": muon_summary["std"],
        "muon_finite_count": int(muon_summary["finite_count"]),
        "advantage_ratio": advantage_ratio,
        "log10_advantage": float(np.log10(max(advantage_ratio, 1e-300))),
        "loss_gap": float(sgd_summary["mean"] - muon_summary["mean"]),
    }


def run_experiment(config_overrides=None, verbose=False):
    """Run the H22a toy experiment and return structured results."""
    config = resolve_config(config_overrides)
    seed_schedule = build_seed_schedule(config)
    train_call_counts = count_train_calls(config)
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()

    if verbose:
        print("=" * 100)
        print("H22a: toy target-spectrum anisotropy sweep for Muon")
        print("=" * 100)
        print(f"Alpha values: {config['alpha_values']}")
        print(
            f"Network: {config['num_layers']}-layer deep linear "
            f"{config['dim']}x{config['dim']}, {config['num_steps']} steps"
        )
        print(
            "Diagnostic caveat: alpha directly sweeps the target spectrum; initial "
            "gradient effective rank is only measured diagnostically."
        )
        print(
            f"Seed counts: diagnostic={len(seed_schedule['diagnostic_seeds'])}, "
            f"lr_sweep={len(seed_schedule['lr_sweep_seeds'])}, "
            f"final_eval={len(seed_schedule['all_seeds'])}"
        )
        print(
            f"Estimated train() calls: {train_call_counts['total_train_calls']} "
            f"({train_call_counts['lr_sweep_train_calls']} LR-sweep + "
            f"{train_call_counts['final_eval_train_calls']} final-eval)"
        )

    alpha_results = []

    for alpha in config["alpha_values"]:
        if verbose:
            print(f"\n  alpha={alpha}")

        problem_cache = {
            seed: build_problem(alpha, seed, config)
            for seed in seed_schedule["all_seeds"]
        }

        reference_target = problem_cache[seed_schedule["all_seeds"][0]]["target"]
        target_singular_values = np.linalg.svd(reference_target, compute_uv=False)
        target_eff_rank = effective_rank(reference_target)
        anisotropy = float(target_singular_values[0] / (target_singular_values[-1] + 1e-15))

        init_grad = evaluate_init_gradient_ranks(
            problem_cache,
            seed_schedule["diagnostic_seeds"],
            config["dim"],
        )

        lr_sweep = {
            "sgd": sweep_learning_rates(
                problem_cache,
                seed_schedule["lr_sweep_seeds"],
                config["lr_sgd"],
                "sgd",
                config,
            ),
            "muon": sweep_learning_rates(
                problem_cache,
                seed_schedule["lr_sweep_seeds"],
                config["lr_muon"],
                "muon",
                config,
            ),
        }
        best_lrs = {
            "sgd": float(lr_sweep["sgd"]["best_lr"]),
            "muon": float(lr_sweep["muon"]["best_lr"]),
        }

        final_eval = evaluate_final_losses(
            problem_cache,
            seed_schedule["all_seeds"],
            best_lrs,
            config,
        )

        alpha_record = {
            "alpha": float(alpha),
            "target_diagnostics": {
                "reference_seed": int(seed_schedule["all_seeds"][0]),
                "target_effective_rank": float(target_eff_rank),
                "target_effective_rank_frac": float(target_eff_rank / config["dim"]),
                "target_singular_values": [float(v) for v in target_singular_values],
                "sigma_max": float(target_singular_values[0]),
                "sigma_min": float(target_singular_values[-1]),
                "anisotropy": anisotropy,
            },
            "init_gradient_diagnostics": {
                "per_seed": init_grad["per_seed"],
                "mean_effective_rank": float(init_grad["mean"]),
                "std_effective_rank": init_grad["std"],
                "mean_effective_rank_frac": float(init_grad["frac"]),
            },
            "best_lrs": best_lrs,
            "lr_sweep": lr_sweep,
            "final_eval": final_eval,
        }
        alpha_results.append(alpha_record)

        if verbose:
            print(
                f"    Target eff rank: {target_eff_rank:.1f}/{config['dim']} "
                f"({100 * target_eff_rank / config['dim']:.0f}%), anisotropy={anisotropy:.1f}"
            )
            print(
                f"    Init grad eff rank: {init_grad['mean']:.1f}/{config['dim']} "
                f"({100 * init_grad['frac']:.0f}%)"
            )
            print(
                f"    Best LR: SGD={best_lrs['sgd']:.4g}, Muon={best_lrs['muon']:.4g}"
            )
            print(
                f"    Final mean loss: SGD={final_eval['sgd_mean']:.3e}, "
                f"Muon={final_eval['muon_mean']:.3e}, "
                f"advantage={final_eval['advantage_ratio']:.1f}x"
            )

    advantages = [record["final_eval"]["advantage_ratio"] for record in alpha_results]
    init_grad_fracs = [
        record["init_gradient_diagnostics"]["mean_effective_rank_frac"]
        for record in alpha_results
    ]
    target_rank_fracs = [
        record["target_diagnostics"]["target_effective_rank_frac"]
        for record in alpha_results
    ]

    peak_index = int(np.argmax(advantages))
    peak_record = alpha_results[peak_index]
    peak_init_grad_frac = peak_record["init_gradient_diagnostics"]["mean_effective_rank_frac"]
    t1_pass = bool(config["t1_lower_frac"] < peak_init_grad_frac < config["t1_upper_frac"])
    t2_pass = bool(
        0 < peak_index < len(alpha_results) - 1
        and advantages[peak_index] > advantages[0]
        and advantages[peak_index] > advantages[-1]
    )

    completed_at = datetime.now(timezone.utc).isoformat()
    elapsed_seconds = float(time.time() - t0)

    summary = {
        "peak_index": peak_index,
        "peak_alpha": float(peak_record["alpha"]),
        "peak_advantage_ratio": float(peak_record["final_eval"]["advantage_ratio"]),
        "peak_init_gradient_effective_rank_frac": float(peak_init_grad_frac),
        "peak_target_effective_rank_frac": float(
            peak_record["target_diagnostics"]["target_effective_rank_frac"]
        ),
        "measured_init_gradient_effective_rank_frac_range": [
            float(min(init_grad_fracs)),
            float(max(init_grad_fracs)),
        ],
        "target_effective_rank_frac_range": [
            float(min(target_rank_fracs)),
            float(max(target_rank_fracs)),
        ],
        "interpretation_note": (
            "The controlled sweep strongly changes target spectrum diagnostics, but the "
            "measured initial gradient effective-rank fraction remains much narrower. "
            "Interpret T1/T2 as toy diagnostics of this implementation, not as a clean "
            "direct test of a gradient-rank sweet spot."
        ),
    }

    tests = {
        "T1": {
            "description": "Peak advantage occurs at a measured initial gradient effective-rank fraction between 20% and 80%.",
            "definition": (
                f"{config['t1_lower_frac']:.1f} < peak_init_gradient_effective_rank_frac < "
                f"{config['t1_upper_frac']:.1f}"
            ),
            "pass": t1_pass,
            "value": float(peak_init_grad_frac),
        },
        "T2": {
            "description": "Advantage shows an interior inverted-U relative to the alpha extremes.",
            "definition": "Peak alpha is interior and its advantage exceeds both extreme-alpha conditions.",
            "pass": t2_pass,
            "value": float(peak_record["final_eval"]["advantage_ratio"]),
            "peak_is_interior": bool(0 < peak_index < len(alpha_results) - 1),
        },
    }

    return {
        "metadata": {
            "study_id": "H22a_ANISOTROPY_SWEET_SPOT",
            "scope": "toy_target_spectrum_sweep",
            "script_path": SCRIPT_PATH,
            "script_dir": SCRIPT_DIR,
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
        },
        "config": config,
        "seed_schedule": seed_schedule,
        "train_call_counts": train_call_counts,
        "alpha_results": alpha_results,
        "summary": summary,
        "tests": tests,
        "runtime": {
            "elapsed_seconds": elapsed_seconds,
        },
    }


def print_report(results):
    """Print a human-readable summary of structured results."""
    config = results["config"]
    alpha_results = results["alpha_results"]
    summary = results["summary"]
    tests = results["tests"]
    seed_schedule = results["seed_schedule"]

    print(f"\n\n{'=' * 112}")
    print("H22a summary: target-spectrum sweep, measured initial-gradient diagnostics, and Muon advantage")
    print(f"{'=' * 112}")
    print(
        "Caveat: alpha directly controls the target spectrum; the measured initial gradient "
        "effective-rank range is only a diagnostic and remains fairly narrow in this setup."
    )
    print(
        "Small-sample note: raw per-seed final losses are preserved, and this run uses "
        f"{len(seed_schedule['diagnostic_seeds'])} diagnostic seeds, "
        f"{len(seed_schedule['lr_sweep_seeds'])} LR-sweep seeds, and "
        f"{len(seed_schedule['all_seeds'])} final-evaluation seeds."
    )

    header = (
        f"\n  {'alpha':>6}  {'target%':>8}  {'init-grad%':>10}  {'anisotropy':>12}  "
        f"{'best lr sgd':>11}  {'best lr muon':>12}  {'sgd mean':>11}  {'muon mean':>11}  {'adv':>9}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for record in alpha_results:
        target_diag = record["target_diagnostics"]
        init_diag = record["init_gradient_diagnostics"]
        final_eval = record["final_eval"]
        print(
            f"  {record['alpha']:>6.1f}  "
            f"{100 * target_diag['target_effective_rank_frac']:>7.0f}%  "
            f"{100 * init_diag['mean_effective_rank_frac']:>9.0f}%  "
            f"{target_diag['anisotropy']:>12.1f}  "
            f"{record['best_lrs']['sgd']:>11.4g}  "
            f"{record['best_lrs']['muon']:>12.4g}  "
            f"{final_eval['sgd_mean']:>11.3e}  "
            f"{final_eval['muon_mean']:>11.3e}  "
            f"{final_eval['advantage_ratio']:>8.1f}x"
        )

    init_grad_range = summary["measured_init_gradient_effective_rank_frac_range"]
    target_range = summary["target_effective_rank_frac_range"]
    print(
        f"\n  Measured init-gradient effective-rank fraction range: "
        f"{100 * init_grad_range[0]:.0f}% - {100 * init_grad_range[1]:.0f}%"
    )
    print(
        f"  Target effective-rank fraction range: {100 * target_range[0]:.0f}% - "
        f"{100 * target_range[1]:.0f}%"
    )
    print(
        f"  Peak advantage: {summary['peak_advantage_ratio']:.1f}x at alpha={summary['peak_alpha']} "
        f"(measured init-grad eff-rank frac={100 * summary['peak_init_gradient_effective_rank_frac']:.0f}%, "
        f"target eff-rank frac={100 * summary['peak_target_effective_rank_frac']:.0f}%)"
    )
    print(
        f"\n  T1: {tests['T1']['description']} --> "
        f"{'PASS' if tests['T1']['pass'] else 'FAIL'} "
        f"({100 * tests['T1']['value']:.0f}%)"
    )
    if tests["T2"]["peak_is_interior"]:
        print(
            f"  T2: {tests['T2']['description']} --> "
            f"{'PASS' if tests['T2']['pass'] else 'FAIL'}"
        )
    else:
        print(
            f"  T2: {tests['T2']['description']} --> FAIL "
            f"(peak occurs at an edge alpha)"
        )

    print(f"\n  Interpretation note: {summary['interpretation_note']}")
    print(f"  Runtime: {results['runtime']['elapsed_seconds']:.2f} s")
    print(f"{'=' * 112}")


def main():
    results = run_experiment(verbose=True)
    print_report(results)


if __name__ == "__main__":
    main()
