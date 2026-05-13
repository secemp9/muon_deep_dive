#!/usr/bin/env python3
"""
H20a: Sampled Cosine-Advantage Proxy Analysis for Trajectory Compounding
========================================================================

This first-pass implementation keeps the original toy experiment intact as much
as possible while making the reporting more truthful and notebook-reusable.

What is measured here:
  - A 2-layer 4x4 deep linear regression problem.
  - Muon versus layerwise normalized SGD (NormSGD).
  - A finite-difference Newton reference computed only at explicit Hessian
    sample steps.
  - A sampled cumulative cosine advantage: the running sum of
      cos(d_muon, d_newton) - cos(d_normsgd, d_newton)
    evaluated only at those sampled steps.

Important limitation:
  The cumulative cosine quantity exposed here is a sampled proxy, not an
  every-step cumulative sum. The primary compounding fit therefore relates
  log(mean loss ratio) to mean sampled cumulative cosine at matched sampled
  states in this toy setting.

The module is import-safe and exposes run_experiment() for notebook use while
preserving CLI execution via main().
"""

from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

DIM = 4
N_LAYERS = 2
N_PARAMS = N_LAYERS * DIM * DIM  # 32
NUM_STEPS = 500
NUM_SEEDS = 5
BATCH_SIZE = 64
FD_EPS = 1e-5
NS_ITERS = 5
LR_TUNING_STEPS = 200
HESSIAN_SAMPLE_EVERY = 50
CHECKPOINTS = [50, 100, 200, 300, 400, 500]
HESSIAN_SAMPLE_STEPS = list(range(0, NUM_STEPS, HESSIAN_SAMPLE_EVERY))
LR_GRID_MUON = np.logspace(-4, -1, 15)
LR_GRID_NORMSGD = np.logspace(-3, 0, 15)


def pack(Ws: List[np.ndarray]) -> np.ndarray:
    return np.concatenate([W.ravel() for W in Ws])


def unpack(theta: np.ndarray) -> List[np.ndarray]:
    return [theta[i * DIM * DIM:(i + 1) * DIM * DIM].reshape(DIM, DIM) for i in range(N_LAYERS)]


def forward(Ws: List[np.ndarray], X: np.ndarray) -> np.ndarray:
    out = X.copy()
    for W in Ws:
        out = W @ out
    return out


def loss_fn(theta: np.ndarray, X: np.ndarray, Y: np.ndarray) -> float:
    Ws = unpack(theta)
    pred = forward(Ws, X)
    return 0.5 * np.mean(np.sum((pred - Y) ** 2, axis=0))


def grad_fn(theta: np.ndarray, X: np.ndarray, Y: np.ndarray):
    Ws = unpack(theta)
    N = X.shape[1]
    acts = [X.copy()]
    for W in Ws:
        acts.append(W @ acts[-1])
    delta = (acts[-1] - Y) / N
    grads = []
    for l in range(N_LAYERS - 1, -1, -1):
        grads.insert(0, delta @ acts[l].T)
        if l > 0:
            delta = Ws[l].T @ delta
    return pack(grads), grads


def hessian_fd(theta: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    n = len(theta)
    H = np.zeros((n, n))
    for i in range(n):
        tp = theta.copy()
        tp[i] += FD_EPS
        tm = theta.copy()
        tm[i] -= FD_EPS
        gp, _ = grad_fn(tp, X, Y)
        gm, _ = grad_fn(tm, X, Y)
        H[:, i] = (gp - gm) / (2 * FD_EPS)
    return 0.5 * (H + H.T)


def newton_schulz(M: np.ndarray, n_iters: int = NS_ITERS) -> np.ndarray:
    norm = np.linalg.norm(M, "fro")
    if norm < 1e-15:
        return M
    X = M / norm
    for _ in range(n_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-15 or nb < 1e-15:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def muon_step_vec(grads_list: List[np.ndarray]) -> np.ndarray:
    return pack([-newton_schulz(G) for G in grads_list])


def normsgd_step_vec(grads_list: List[np.ndarray]) -> np.ndarray:
    dirs = []
    for G in grads_list:
        nrm = np.linalg.norm(G, "fro")
        dirs.append(-G / max(nrm, 1e-15))
    return pack(dirs)


def _make_main_seed_list() -> List[int]:
    return [42 + i * 137 for i in range(NUM_SEEDS)]


def _to_float_list(values) -> List[float]:
    return [float(v) for v in values]


def _summarize_series(series_list: List[List[float]]) -> Dict[str, List[float]]:
    arr = np.asarray(series_list, dtype=float)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0, ddof=0)
    ci95 = 1.96 * std / np.sqrt(arr.shape[0])
    return {
        "mean": _to_float_list(mean),
        "std": _to_float_list(std),
        "ci95": _to_float_list(ci95),
    }


def _summarize_scalars(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=0))
    ci95 = float(1.96 * std / np.sqrt(arr.shape[0]))
    return {"mean": mean, "std": std, "ci95": ci95}


def _linear_fit(x_values, y_values) -> Dict[str, float]:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if ss_tot < 1e-30 else 1.0 - ss_res / ss_tot
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "point_count": int(len(x)),
        "x": _to_float_list(x),
        "y": _to_float_list(y),
    }


def _select_best_learning_rates(seeds: List[int], verbose: bool = True):
    tuning_seed = seeds[0]
    rng = np.random.RandomState(tuning_seed)
    X = rng.randn(DIM, BATCH_SIZE) * 0.3
    Y = rng.randn(DIM, BATCH_SIZE) * 0.3

    search_spec = {
        "muon": {"grid": LR_GRID_MUON, "step_fn": muon_step_vec},
        "normsgd": {"grid": LR_GRID_NORMSGD, "step_fn": normsgd_step_vec},
    }

    best_lr = {}
    sweep_results = {}

    if verbose:
        print("Learning-rate search on first seed")

    for opt_name, spec in search_spec.items():
        grid = spec["grid"]
        step_fn = spec["step_fn"]
        best = float(grid[0])
        best_loss = float("inf")
        trials = []

        for lr in grid:
            rng_init = np.random.RandomState(tuning_seed)
            theta = pack([np.eye(DIM) + rng_init.randn(DIM, DIM) * 0.1 for _ in range(N_LAYERS)])
            ok = True
            final_loss = float("inf")

            for _ in range(LR_TUNING_STEPS):
                _, grads_list = grad_fn(theta, X, Y)
                d = step_fn(grads_list)
                theta = theta + lr * d
                final_loss = loss_fn(theta, X, Y)
                if not np.isfinite(final_loss) or final_loss > 1e6:
                    ok = False
                    break

            trials.append(
                {
                    "lr": float(lr),
                    "ok": bool(ok),
                    "final_loss": float(final_loss) if np.isfinite(final_loss) else None,
                }
            )

            if ok and final_loss < best_loss:
                best_loss = float(final_loss)
                best = float(lr)

        best_lr[opt_name] = best
        sweep_results[opt_name] = {
            "grid": _to_float_list(grid),
            "selected_best_lr": float(best),
            "best_final_loss": float(best_loss),
            "trials": trials,
        }

        if verbose:
            print(f"  {opt_name:8s} best LR = {best:.12f} (warmup final loss {best_loss:.6e})")

    return best_lr, sweep_results


def _build_verdict(sampled_fit_slope: float, final_mean_checkpoint_ratio: float) -> Dict[str, str]:
    if sampled_fit_slope > 0 and final_mean_checkpoint_ratio > 1.10:
        status = "tentative_support"
        message = (
            "Matched sampled-state fit is positive and the end-of-run loss-ratio gap remains nontrivial. "
            "This is only tentative support because the experiment is still a sparse sampled proxy in a toy setting."
        )
    elif sampled_fit_slope > 0:
        status = "weak_or_transient_support"
        message = (
            "Matched sampled-state fit is positive, but the checkpoint loss ratio contracts toward 1x by the end of training. "
            "Any compounding signal here is weak or transient rather than decisive."
        )
    else:
        status = "unsupported_under_current_configuration"
        message = (
            "Sampled Newton-alignment advantage is present, but the matched sampled-state fit is non-positive and the mean "
            "checkpoint loss ratio returns to approximately 1x. Geometric compounding is not supported by this configuration."
        )
    return {"status": status, "message": message}


def run_experiment(verbose: bool = True) -> Dict[str, object]:
    start_time = time.time()
    seeds = _make_main_seed_list()

    if verbose:
        print("=" * 100)
        print("H20a: SAMPLED COSINE-ADVANTAGE PROXY ANALYSIS")
        print("=" * 100)
        print(f"Network: {N_LAYERS}-layer deep linear {DIM}x{DIM} ({N_PARAMS} params)")
        print(f"Training steps: {NUM_STEPS}")
        print(f"Checkpoint steps (1-based, post-update): {CHECKPOINTS}")
        print(f"Hessian sample steps (0-based, pre-update): {HESSIAN_SAMPLE_STEPS}")
        print("Cumulative cosine quantity: sampled proxy, not an every-step sum")
        print()

    best_lr, lr_search = _select_best_learning_rates(seeds, verbose=verbose)

    per_seed_results = []

    for seed_index, seed in enumerate(seeds):
        rng = np.random.RandomState(seed)
        X = rng.randn(DIM, BATCH_SIZE) * 0.3
        Y = rng.randn(DIM, BATCH_SIZE) * 0.3
        weights_init = [np.eye(DIM) + rng.randn(DIM, DIM) * 0.1 for _ in range(N_LAYERS)]

        theta_muon = pack([W.copy() for W in weights_init])
        theta_norm = pack([W.copy() for W in weights_init])

        sampled_steps = []
        sampled_cosine_advantage = []
        sampled_cosine_muon_vs_newton = []
        sampled_cosine_normsgd_vs_newton = []
        sampled_cumulative_cosine_pre_update = []
        sampled_cumulative_cosine_post_measurement = []
        sampled_loss_muon = []
        sampled_loss_normsgd = []
        sampled_loss_ratio = []
        cosine_advantage_sparse_by_step = []
        loss_muon_by_step = []
        loss_normsgd_by_step = []
        checkpoint_loss_muon = []
        checkpoint_loss_normsgd = []
        checkpoint_loss_ratio = []
        checkpoint_cumulative_sampled_cosine = []

        cumulative_sampled_cosine = 0.0

        for step in range(NUM_STEPS):
            compute_hessian = step in HESSIAN_SAMPLE_STEPS

            g_muon, grads_muon = grad_fn(theta_muon, X, Y)
            d_muon = muon_step_vec(grads_muon)

            _, grads_norm = grad_fn(theta_norm, X, Y)
            d_norm = normsgd_step_vec(grads_norm)

            cos_adv = 0.0
            if compute_hessian:
                loss_muon_current = loss_fn(theta_muon, X, Y)
                loss_norm_current = loss_fn(theta_norm, X, Y)
                ratio_current = loss_norm_current / max(loss_muon_current, 1e-30)

                H = hessian_fd(theta_muon, X, Y)
                H_pinv = np.linalg.pinv(H, rcond=1e-6)
                d_newton = -H_pinv @ g_muon

                cos_muon_newton = cosine(d_muon, d_newton)
                cos_norm_newton = cosine(d_norm, d_newton)
                cos_adv = cos_muon_newton - cos_norm_newton

                sampled_steps.append(int(step))
                sampled_loss_muon.append(float(loss_muon_current))
                sampled_loss_normsgd.append(float(loss_norm_current))
                sampled_loss_ratio.append(float(ratio_current))
                sampled_cumulative_cosine_pre_update.append(float(cumulative_sampled_cosine))
                sampled_cosine_muon_vs_newton.append(float(cos_muon_newton))
                sampled_cosine_normsgd_vs_newton.append(float(cos_norm_newton))
                sampled_cosine_advantage.append(float(cos_adv))

                cumulative_sampled_cosine += cos_adv
                sampled_cumulative_cosine_post_measurement.append(float(cumulative_sampled_cosine))

            cosine_advantage_sparse_by_step.append(float(cos_adv))

            theta_muon = theta_muon + best_lr["muon"] * d_muon
            theta_norm = theta_norm + best_lr["normsgd"] * d_norm

            loss_muon_after = loss_fn(theta_muon, X, Y)
            loss_norm_after = loss_fn(theta_norm, X, Y)
            loss_muon_by_step.append(float(loss_muon_after))
            loss_normsgd_by_step.append(float(loss_norm_after))

            if step + 1 in CHECKPOINTS:
                checkpoint_loss_muon.append(float(loss_muon_after))
                checkpoint_loss_normsgd.append(float(loss_norm_after))
                checkpoint_loss_ratio.append(float(loss_norm_after / max(loss_muon_after, 1e-30)))
                checkpoint_cumulative_sampled_cosine.append(float(cumulative_sampled_cosine))

        final_parameter_divergence = float(np.linalg.norm(theta_muon - theta_norm))

        per_seed_results.append(
            {
                "seed": int(seed),
                "sample_steps_0based_pre_update": sampled_steps,
                "sampled_cosine_advantage": sampled_cosine_advantage,
                "sampled_cosine_muon_vs_newton": sampled_cosine_muon_vs_newton,
                "sampled_cosine_normsgd_vs_newton": sampled_cosine_normsgd_vs_newton,
                "sampled_cumulative_cosine_advantage_pre_update": sampled_cumulative_cosine_pre_update,
                "sampled_cumulative_cosine_advantage_post_measurement": sampled_cumulative_cosine_post_measurement,
                "sampled_loss_muon": sampled_loss_muon,
                "sampled_loss_normsgd": sampled_loss_normsgd,
                "sampled_loss_ratio": sampled_loss_ratio,
                "cosine_advantage_sparse_by_step": cosine_advantage_sparse_by_step,
                "loss_muon_by_step": loss_muon_by_step,
                "loss_normsgd_by_step": loss_normsgd_by_step,
                "checkpoint_steps_1based_post_update": list(CHECKPOINTS),
                "checkpoint_loss_muon": checkpoint_loss_muon,
                "checkpoint_loss_normsgd": checkpoint_loss_normsgd,
                "checkpoint_loss_ratio": checkpoint_loss_ratio,
                "checkpoint_cumulative_sampled_cosine_advantage": checkpoint_cumulative_sampled_cosine,
                "final_parameter_divergence": final_parameter_divergence,
            }
        )

        if verbose:
            print(
                f"  Seed {seed_index + 1}/{len(seeds)} | seed={seed} | "
                f"final checkpoint ratio={checkpoint_loss_ratio[-1]:.6f}x | "
                f"final sampled cumulative cosine={checkpoint_cumulative_sampled_cosine[-1]:.6f}"
            )

    sampled_steps = per_seed_results[0]["sample_steps_0based_pre_update"]
    checkpoint_steps = per_seed_results[0]["checkpoint_steps_1based_post_update"]

    sampled_aggregates = {
        "steps_0based_pre_update": sampled_steps,
        "cosine_advantage": _summarize_series([r["sampled_cosine_advantage"] for r in per_seed_results]),
        "cosine_muon_vs_newton": _summarize_series([r["sampled_cosine_muon_vs_newton"] for r in per_seed_results]),
        "cosine_normsgd_vs_newton": _summarize_series([r["sampled_cosine_normsgd_vs_newton"] for r in per_seed_results]),
        "cumulative_sampled_cosine_advantage_pre_update": _summarize_series(
            [r["sampled_cumulative_cosine_advantage_pre_update"] for r in per_seed_results]
        ),
        "cumulative_sampled_cosine_advantage_post_measurement": _summarize_series(
            [r["sampled_cumulative_cosine_advantage_post_measurement"] for r in per_seed_results]
        ),
        "loss_muon": _summarize_series([r["sampled_loss_muon"] for r in per_seed_results]),
        "loss_normsgd": _summarize_series([r["sampled_loss_normsgd"] for r in per_seed_results]),
        "loss_ratio": _summarize_series([r["sampled_loss_ratio"] for r in per_seed_results]),
    }
    sampled_log_mean_loss_ratio = np.log(np.maximum(np.asarray(sampled_aggregates["loss_ratio"]["mean"]), 1e-30))
    sampled_aggregates["log_mean_loss_ratio"] = _to_float_list(sampled_log_mean_loss_ratio)

    checkpoint_aggregates = {
        "steps_1based_post_update": checkpoint_steps,
        "cumulative_sampled_cosine_advantage": _summarize_series(
            [r["checkpoint_cumulative_sampled_cosine_advantage"] for r in per_seed_results]
        ),
        "loss_muon": _summarize_series([r["checkpoint_loss_muon"] for r in per_seed_results]),
        "loss_normsgd": _summarize_series([r["checkpoint_loss_normsgd"] for r in per_seed_results]),
        "loss_ratio": _summarize_series([r["checkpoint_loss_ratio"] for r in per_seed_results]),
    }
    checkpoint_log_mean_loss_ratio = np.log(np.maximum(np.asarray(checkpoint_aggregates["loss_ratio"]["mean"]), 1e-30))
    checkpoint_aggregates["log_mean_loss_ratio"] = _to_float_list(checkpoint_log_mean_loss_ratio)

    sampled_fit = _linear_fit(
        sampled_aggregates["cumulative_sampled_cosine_advantage_pre_update"]["mean"],
        sampled_aggregates["log_mean_loss_ratio"],
    )
    sampled_fit["description"] = (
        "Fit of log(mean sampled-state loss ratio) versus mean sampled cumulative cosine. "
        "Sampled states are pre-update states at the explicit Hessian sample schedule."
    )

    checkpoint_time_fit = _linear_fit(
        checkpoint_aggregates["steps_1based_post_update"],
        checkpoint_aggregates["log_mean_loss_ratio"],
    )
    checkpoint_time_fit["description"] = (
        "Fit of log(mean checkpoint loss ratio) versus training step using the original post-update checkpoints."
    )

    final_mean_checkpoint_ratio = checkpoint_aggregates["loss_ratio"]["mean"][-1]
    verdict = _build_verdict(sampled_fit["slope"], final_mean_checkpoint_ratio)

    results = {
        "experiment_id": "H20a_COSINE_COMPOUNDING_TRAJECTORY",
        "title": "H20a: sampled cosine-advantage proxy analysis for trajectory compounding",
        "notes": [
            "Toy setting: 2-layer 4x4 deep linear regression with exact gradients and sparse finite-difference Hessians.",
            "The cumulative cosine quantity is a sampled proxy, not an every-step cumulative sum.",
            "Primary fit uses matched sampled states: pre-update loss ratios at explicit Hessian sample steps.",
            "Newton reference is computed at Muon's state; the NormSGD comparison therefore remains a cross-trajectory comparison.",
        ],
        "config": {
            "dim": DIM,
            "n_layers": N_LAYERS,
            "n_params": N_PARAMS,
            "num_steps": NUM_STEPS,
            "num_seeds": NUM_SEEDS,
            "batch_size": BATCH_SIZE,
            "fd_eps": FD_EPS,
            "ns_iters": NS_ITERS,
            "lr_tuning_steps": LR_TUNING_STEPS,
            "checkpoints_1based_post_update": list(CHECKPOINTS),
            "hessian_sample_every": HESSIAN_SAMPLE_EVERY,
            "hessian_sample_steps_0based_pre_update": list(HESSIAN_SAMPLE_STEPS),
            "sampled_cumulative_cosine_definition": (
                "Running sum of cosine advantages evaluated only at Hessian sample steps. "
                "For sampled-state fits, the cumulative value paired with a sampled loss ratio is the pre-update total available at that state."
            ),
        },
        "seeds": [int(s) for s in seeds],
        "lr_search": {
            "tuning_seed": int(seeds[0]),
            "muon": lr_search["muon"],
            "normsgd": lr_search["normsgd"],
        },
        "per_seed": per_seed_results,
        "aggregates": {
            "sample_states": sampled_aggregates,
            "checkpoints": checkpoint_aggregates,
            "final_parameter_divergence": _summarize_scalars(
                [r["final_parameter_divergence"] for r in per_seed_results]
            ),
        },
        "fits": {
            "sampled_log_mean_loss_ratio_vs_mean_cumulative_sampled_cosine": sampled_fit,
            "checkpoint_log_mean_loss_ratio_vs_step": checkpoint_time_fit,
        },
        "verdict": verdict,
        "runtime_seconds": float(time.time() - start_time),
    }

    if verbose:
        print(f"\n{'=' * 100}")
        print("SUMMARY TABLE: MATCHED SAMPLED STATES")
        print(f"{'=' * 100}")
        print(
            f"{'Sample step':>12}  {'Mean cos adv':>14}  {'Mean cumul cos':>16}  "
            f"{'Mean loss ratio':>16}  {'log(mean ratio)':>16}"
        )
        print("  " + "-" * 82)
        for step, cos_adv, cumul, ratio, log_ratio in zip(
            sampled_aggregates["steps_0based_pre_update"],
            sampled_aggregates["cosine_advantage"]["mean"],
            sampled_aggregates["cumulative_sampled_cosine_advantage_pre_update"]["mean"],
            sampled_aggregates["loss_ratio"]["mean"],
            sampled_aggregates["log_mean_loss_ratio"],
        ):
            print(
                f"{step:>12}  {cos_adv:>14.6f}  {cumul:>16.6f}  {ratio:>16.6f}  {log_ratio:>16.6f}"
            )

        print(f"\n{'=' * 100}")
        print("SUMMARY TABLE: ORIGINAL CHECKPOINT REPORTING")
        print(f"{'=' * 100}")
        print(
            f"{'Checkpoint':>12}  {'Mean cumul cos':>16}  {'Mean loss ratio':>16}  {'log(mean ratio)':>16}"
        )
        print("  " + "-" * 68)
        for step, cumul, ratio, log_ratio in zip(
            checkpoint_aggregates["steps_1based_post_update"],
            checkpoint_aggregates["cumulative_sampled_cosine_advantage"]["mean"],
            checkpoint_aggregates["loss_ratio"]["mean"],
            checkpoint_aggregates["log_mean_loss_ratio"],
        ):
            print(f"{step:>12}  {cumul:>16.6f}  {ratio:>16.6f}  {log_ratio:>16.6f}")

        print(f"\n{'=' * 100}")
        print("FITS")
        print(f"{'=' * 100}")
        print(
            "Matched sampled-state fit: "
            f"log(mean loss ratio) = {sampled_fit['slope']:.8f} * sampled_cumul_cos + {sampled_fit['intercept']:.8f} "
            f"(R^2={sampled_fit['r2']:.4f})"
        )
        print(
            "Checkpoint time fit:      "
            f"log(mean loss ratio) = {checkpoint_time_fit['slope']:.8f} * T + {checkpoint_time_fit['intercept']:.8f} "
            f"(R^2={checkpoint_time_fit['r2']:.4f})"
        )
        print(f"\nVerdict: {verdict['status']}")
        print(verdict["message"])
        print(f"Runtime: {results['runtime_seconds']:.2f}s")

    return results


def main() -> Dict[str, object]:
    return run_experiment(verbose=True)


if __name__ == "__main__":
    main()
