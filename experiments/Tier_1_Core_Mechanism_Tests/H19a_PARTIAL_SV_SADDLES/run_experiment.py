#!/usr/bin/env python3
"""
H19a: partial SV equalization — one-step local Hessian probe
===========================================================

Purpose
-------
Measure how a *single* partial-SV-equalized update changes local Hessian
geometry in a tiny deep linear network.

Important scope limits
----------------------
- This is a one-step local Hessian probe, not a trajectory study.
- `k=0` is a per-layer norm-matched raw-gradient control, not plain SGD.
- `k=1` is degenerate with `k=0` under the current top-k-to-mean construction.
- The default sweep is per-layer `k in [0, 1, 2, 3, 4]` for 4x4 matrices.

The module exposes `run_experiment()` for notebook reuse and keeps `main()` as
an executable CLI entrypoint.
"""

from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

DIM = 4
N_LAYERS = 2
N_PARAMS = N_LAYERS * DIM * DIM  # 32
N_SAMPLES = 64
DATA_SCALE = 0.3
INIT_SCALE = 0.1
WARMUP_STEPS = 50
NUM_SEEDS = 5
MOMENTUM = 0.9
LR_WARMUP = 0.01
LR_STEP = 0.01
FD_EPS = 1e-5
NS_ITERS = 5
NEG_EIG_THRESHOLD = -1e-8
NEAR_ZERO_THRESHOLD = 1e-6
K_PER_LAYER = [0, 1, 2, 3, 4]
DEFAULT_SEEDS = [42 + i * 137 for i in range(NUM_SEEDS)]


# =============================================================================
# NETWORK
# =============================================================================

def pack(Ws: List[np.ndarray]) -> np.ndarray:
    return np.concatenate([W.ravel() for W in Ws])


def unpack(theta: np.ndarray) -> List[np.ndarray]:
    Ws = []
    idx = 0
    for _ in range(N_LAYERS):
        Ws.append(theta[idx:idx + DIM * DIM].reshape(DIM, DIM))
        idx += DIM * DIM
    return Ws


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
    """Full Hessian via central finite differences (32x32 = tractable)."""
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


# =============================================================================
# PARTIAL SV EQUALIZATION
# =============================================================================

def partial_sv_equalize_layer(M: np.ndarray, k: int) -> np.ndarray:
    """
    Equalize the top-k singular values of a layer gradient matrix M (DIM x DIM).

    k=0:
        No SV equalization. Preserve singular-value ratios but rescale each layer
        to match the Frobenius norm of the full polar factor.
    k=DIM:
        Full polar factor (U V^T).
    k=1..DIM-1:
        Equalize top-k singular values to their mean, then norm-match.
    """
    U, sigma, Vt = np.linalg.svd(M, full_matrices=False)
    d = len(sigma)

    if k == 0:
        target_norm = np.sqrt(d)  # ||U V^T||_F = sqrt(d)
        current_norm = np.linalg.norm(sigma)
        if current_norm > 1e-15:
            sigma_scaled = sigma * (target_norm / current_norm)
            return U @ np.diag(sigma_scaled) @ Vt
        return M

    kk = min(k, d)
    if kk >= d:
        return U @ Vt

    sigma_new = sigma.copy()
    top_mean = np.mean(sigma[:kk])
    sigma_new[:kk] = top_mean

    target_norm = np.sqrt(d)
    current_norm = np.linalg.norm(sigma_new)
    if current_norm > 1e-15:
        sigma_new *= target_norm / current_norm

    return U @ np.diag(sigma_new) @ Vt


def newton_schulz_layer(M: np.ndarray, n_iters: int = NS_ITERS) -> np.ndarray:
    """Reference polar-factor approximation; not used in the default experiment."""
    norm = np.linalg.norm(M, "fro")
    if norm < 1e-15:
        return M
    X = M / norm
    for _ in range(n_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


# =============================================================================
# EXPERIMENT CORE
# =============================================================================

def k_label(k: int) -> str:
    if k == 0:
        return "norm-matched raw-gradient control"
    if k == 1:
        return "degenerate top-1 equalization (= k=0 here)"
    if k == DIM:
        return "full equalization / polar factor"
    return f"partial top-{k} equalization"


def eigen_metrics(eigs: np.ndarray) -> Dict[str, float]:
    return {
        "neg_count": int(np.sum(eigs < NEG_EIG_THRESHOLD)),
        "near_zero_count": int(np.sum(np.abs(eigs) < NEAR_ZERO_THRESHOLD)),
        "min_eig": float(eigs[0]),
        "max_eig": float(eigs[-1]),
    }


def mean_std(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}


def build_config(seeds: List[int]) -> Dict[str, object]:
    return {
        "dim": DIM,
        "n_layers": N_LAYERS,
        "n_params": N_PARAMS,
        "n_samples": N_SAMPLES,
        "data_scale": DATA_SCALE,
        "init_scale": INIT_SCALE,
        "warmup_steps": WARMUP_STEPS,
        "momentum": MOMENTUM,
        "lr_warmup": LR_WARMUP,
        "lr_step": LR_STEP,
        "fd_eps": FD_EPS,
        "ns_iters": NS_ITERS,
        "neg_eig_threshold": NEG_EIG_THRESHOLD,
        "near_zero_threshold": NEAR_ZERO_THRESHOLD,
        "k_per_layer": list(K_PER_LAYER),
        "seeds": [int(seed) for seed in seeds],
        "k0_control": "Per-layer norm-matched raw-gradient control; not plain SGD.",
        "k1_note": "Under the current construction, k=1 is degenerate with k=0.",
        "scope_note": "Single-step local Hessian probe in a tiny 2-layer deep linear toy model.",
        "hessian_evaluations_per_seed": 1 + len(K_PER_LAYER),
        "gradient_evaluations_per_hessian": 2 * N_PARAMS,
        "warmup_gradient_evaluations_per_seed": WARMUP_STEPS,
    }


def run_experiment(seeds: List[int] | None = None, verbose: bool = True) -> Dict[str, object]:
    seeds = list(DEFAULT_SEEDS if seeds is None else seeds)
    config = build_config(seeds)
    start_time = time.time()

    baseline_metrics: List[Dict[str, object]] = []
    per_k_metrics: List[Dict[str, object]] = []
    max_abs_step_diff_k1_vs_k0 = 0.0

    if verbose:
        print("=" * 100)
        print("H19a: partial SV equalization — one-step local Hessian probe")
        print("=" * 100)
        print(f"Network: {N_LAYERS}-layer deep linear {DIM}x{DIM} ({N_PARAMS} parameters)")
        print(f"Warmup: {WARMUP_STEPS} momentum-SGD steps at LR={LR_WARMUP}, then one test step at LR={LR_STEP}")
        print(f"k sweep (per layer): {K_PER_LAYER}")
        print(f"Seeds: {seeds}")
        print("Scope note: one-step local curvature measurement, not a trajectory-level test.")
        print("Control note: k=0 is a per-layer norm-matched raw-gradient control, not plain SGD.")
        print("Degeneracy note: k=1 is mathematically identical to k=0 here.")

    for seed_idx, seed in enumerate(seeds, start=1):
        rng = np.random.RandomState(seed)
        X = rng.randn(DIM, N_SAMPLES) * DATA_SCALE
        Y = rng.randn(DIM, N_SAMPLES) * DATA_SCALE
        weights = [np.eye(DIM) + rng.randn(DIM, DIM) * INIT_SCALE for _ in range(N_LAYERS)]
        theta = pack(weights)

        initial_loss = float(loss_fn(theta, X, Y))

        mom = np.zeros_like(theta)
        for _ in range(WARMUP_STEPS):
            g, _ = grad_fn(theta, X, Y)
            mom = MOMENTUM * mom + g
            theta = theta - LR_WARMUP * mom

        warmup_loss = float(loss_fn(theta, X, Y))
        H_before = hessian_fd(theta, X, Y)
        eigs_before = np.linalg.eigvalsh(H_before)
        before = eigen_metrics(eigs_before)

        baseline_record = {
            "seed": int(seed),
            "initial_loss": initial_loss,
            "warmup_loss": warmup_loss,
            "loss": warmup_loss,
            "neg_count": before["neg_count"],
            "near_zero_count": before["near_zero_count"],
            "min_eig": before["min_eig"],
            "max_eig": before["max_eig"],
            "theta_norm": float(np.linalg.norm(theta)),
        }
        baseline_metrics.append(baseline_record)

        raw_grad_vec, grads_list = grad_fn(theta, X, Y)
        raw_grad_norm = float(np.linalg.norm(raw_grad_vec))
        step_layers_cache = {}
        step_vec_cache = {}
        for k in K_PER_LAYER:
            step_layers = [partial_sv_equalize_layer(grad_layer, k) for grad_layer in grads_list]
            step_layers_cache[k] = step_layers
            step_vec_cache[k] = pack(step_layers)

        max_abs_step_diff_k1_vs_k0 = max(
            max_abs_step_diff_k1_vs_k0,
            float(np.max(np.abs(step_vec_cache[1] - step_vec_cache[0]))),
        )

        if verbose:
            print(f"\nSeed {seed_idx}/{len(seeds)} (seed={seed})")
            print(
                "  Warmup baseline: "
                f"neg={before['neg_count']}, near0={before['near_zero_count']}, "
                f"min={before['min_eig']:.4e}, loss={warmup_loss:.6e}"
            )

        for k in K_PER_LAYER:
            step_vec = step_vec_cache[k]
            step_norm = float(np.linalg.norm(step_vec))
            cos_with_raw_grad = float(
                np.dot(step_vec, raw_grad_vec) / (step_norm * raw_grad_norm + 1e-30)
            )

            theta_new = theta - LR_STEP * step_vec
            H_after = hessian_fd(theta_new, X, Y)
            eigs_after = np.linalg.eigvalsh(H_after)
            after = eigen_metrics(eigs_after)
            loss_after = float(loss_fn(theta_new, X, Y))

            record = {
                "seed": int(seed),
                "k": int(k),
                "k_label": k_label(k),
                "raw_grad_norm": raw_grad_norm,
                "step_norm": step_norm,
                "cos_with_raw_grad": cos_with_raw_grad,
                "neg_before": before["neg_count"],
                "neg_after": after["neg_count"],
                "delta_neg": int(after["neg_count"] - before["neg_count"]),
                "near_zero_before": before["near_zero_count"],
                "near_zero_after": after["near_zero_count"],
                "delta_near_zero": int(after["near_zero_count"] - before["near_zero_count"]),
                "min_eig_before": before["min_eig"],
                "min_eig_after": after["min_eig"],
                "delta_min": float(after["min_eig"] - before["min_eig"]),
                "loss_before": warmup_loss,
                "loss_after": loss_after,
                "delta_loss": float(loss_after - warmup_loss),
            }
            per_k_metrics.append(record)

            if verbose:
                print(
                    f"  k={k:<1} [{k_label(k)}]: "
                    f"neg={record['neg_after']:>2} (Δ{record['delta_neg']:+d}), "
                    f"min={record['min_eig_after']:.4e} (Δ{record['delta_min']:+.2e}), "
                    f"loss={record['loss_after']:.6e} (Δ{record['delta_loss']:+.2e}), "
                    f"cos={record['cos_with_raw_grad']:.4f}"
                )

    baseline_neg = [row["neg_count"] for row in baseline_metrics]
    baseline_near_zero = [row["near_zero_count"] for row in baseline_metrics]
    baseline_min = [row["min_eig"] for row in baseline_metrics]
    baseline_loss = [row["loss"] for row in baseline_metrics]

    aggregate_by_k = []
    mean_neg_after_by_k = {}
    for k in K_PER_LAYER:
        rows = [row for row in per_k_metrics if row["k"] == k]
        neg_after = [row["neg_after"] for row in rows]
        delta_neg = [row["delta_neg"] for row in rows]
        near_zero_after = [row["near_zero_after"] for row in rows]
        delta_near_zero = [row["delta_near_zero"] for row in rows]
        min_after = [row["min_eig_after"] for row in rows]
        delta_min = [row["delta_min"] for row in rows]
        loss_after = [row["loss_after"] for row in rows]
        delta_loss = [row["delta_loss"] for row in rows]
        step_norm = [row["step_norm"] for row in rows]
        cosines = [row["cos_with_raw_grad"] for row in rows]

        agg = {
            "k": int(k),
            "k_label": k_label(k),
            "n_seeds": len(rows),
            "mean_neg_before": float(np.mean(baseline_neg)),
            "mean_neg_after": float(np.mean(neg_after)),
            "std_neg_after": float(np.std(neg_after)),
            "mean_delta_neg": float(np.mean(delta_neg)),
            "std_delta_neg": float(np.std(delta_neg)),
            "mean_near_zero_before": float(np.mean(baseline_near_zero)),
            "mean_near_zero_after": float(np.mean(near_zero_after)),
            "std_near_zero_after": float(np.std(near_zero_after)),
            "mean_delta_near_zero": float(np.mean(delta_near_zero)),
            "std_delta_near_zero": float(np.std(delta_near_zero)),
            "mean_min_eig_before": float(np.mean(baseline_min)),
            "mean_min_eig_after": float(np.mean(min_after)),
            "std_min_eig_after": float(np.std(min_after)),
            "mean_delta_min": float(np.mean(delta_min)),
            "std_delta_min": float(np.std(delta_min)),
            "mean_loss_before": float(np.mean(baseline_loss)),
            "mean_loss_after": float(np.mean(loss_after)),
            "std_loss_after": float(np.std(loss_after)),
            "mean_delta_loss": float(np.mean(delta_loss)),
            "std_delta_loss": float(np.std(delta_loss)),
            "mean_step_norm": float(np.mean(step_norm)),
            "mean_cos_with_raw_grad": float(np.mean(cosines)),
        }
        aggregate_by_k.append(agg)
        mean_neg_after_by_k[k] = agg["mean_neg_after"]

    mean_neg_0 = mean_neg_after_by_k[0]
    mean_neg_full = mean_neg_after_by_k[DIM]
    endpoints_max = max(mean_neg_0, mean_neg_full)
    peak_row = max(aggregate_by_k, key=lambda row: row["mean_neg_after"])
    intermediate_ks = [k for k in K_PER_LAYER if 0 < k < DIM]
    intermediate_more_neg_confirmed = any(
        mean_neg_after_by_k[k] > endpoints_max + 0.5 for k in intermediate_ks
    )

    runtime_seconds = float(time.time() - start_time)
    total_hessian_evaluations = len(seeds) * (1 + len(K_PER_LAYER))
    total_hessian_gradient_evaluations = total_hessian_evaluations * (2 * N_PARAMS)

    analysis = {
        "mean_neg_before": float(np.mean(baseline_neg)),
        "mean_neg_after_k0": mean_neg_0,
        "mean_neg_after_full": mean_neg_full,
        "endpoint_max_mean_neg_after": float(endpoints_max),
        "peak_k": int(peak_row["k"]),
        "peak_mean_neg_after": float(peak_row["mean_neg_after"]),
        "intermediate_more_neg_confirmed": bool(intermediate_more_neg_confirmed),
        "max_abs_step_diff_k1_vs_k0": float(max_abs_step_diff_k1_vs_k0),
        "k1_matches_k0_within_tolerance": bool(max_abs_step_diff_k1_vs_k0 <= 1e-12),
        "runtime_seconds": runtime_seconds,
        "total_hessian_evaluations": int(total_hessian_evaluations),
        "total_hessian_gradient_evaluations": int(total_hessian_gradient_evaluations),
        "notes": [
            "This run measures immediate post-step local Hessian geometry only.",
            "k=0 is a norm-matched raw-gradient control, not plain SGD.",
            "k=1 is degenerate with k=0 under the current construction.",
        ],
    }

    results = {
        "experiment_id": "H19a_PARTIAL_SV_SADDLES",
        "title": "Partial SV equalization: one-step local Hessian probe",
        "config": config,
        "baseline_metrics": baseline_metrics,
        "per_k_metrics": per_k_metrics,
        "aggregate_by_k": aggregate_by_k,
        "analysis": analysis,
    }
    return results


# =============================================================================
# REPORTING
# =============================================================================

def print_results_report(results: Dict[str, object]) -> None:
    config = results["config"]
    baseline_metrics = results["baseline_metrics"]
    aggregate_by_k = results["aggregate_by_k"]
    analysis = results["analysis"]

    print(f"\n{'=' * 100}")
    print("AGGREGATE RESULTS: ONE-STEP HESSIAN PROBE")
    print(f"{'=' * 100}")
    print("Interpretation scope: local one-step curvature only; not a trajectory-level conclusion.")
    print()
    print(
        f"{'k':>3}  {'label':<43} {'mean neg':>9}  {'mean Δneg':>10}  "
        f"{'mean min eig':>13}  {'mean Δmin':>11}  {'mean loss':>12}  {'mean Δloss':>12}"
    )
    print("-" * 128)
    for row in aggregate_by_k:
        print(
            f"{row['k']:>3}  {row['k_label']:<43} "
            f"{row['mean_neg_after']:>9.1f}  {row['mean_delta_neg']:>10.1f}  "
            f"{row['mean_min_eig_after']:>13.4e}  {row['mean_delta_min']:>11.4e}  "
            f"{row['mean_loss_after']:>12.6e}  {row['mean_delta_loss']:>12.4e}"
        )

    print("\nBaseline at the warmup point (before the test step):")
    print(
        f"  mean neg eig count = {np.mean([row['neg_count'] for row in baseline_metrics]):.1f}\n"
        f"  mean near-zero count = {np.mean([row['near_zero_count'] for row in baseline_metrics]):.1f}\n"
        f"  mean min eigenvalue = {np.mean([row['min_eig'] for row in baseline_metrics]):.4e}\n"
        f"  mean loss = {np.mean([row['loss'] for row in baseline_metrics]):.6e}"
    )

    print("\nKey check: do intermediate k values show more negative-eigenvalue directions than endpoints?")
    print(f"  k=0 control mean neg count: {analysis['mean_neg_after_k0']:.1f}")
    print(f"  k={DIM} full equalization mean neg count: {analysis['mean_neg_after_full']:.1f}")
    print(f"  peak mean neg count occurs at k={analysis['peak_k']} with {analysis['peak_mean_neg_after']:.1f}")
    print(
        "  intermediate-k increase above endpoints: "
        f"{'CONFIRMED' if analysis['intermediate_more_neg_confirmed'] else 'NOT CONFIRMED'}"
    )

    if not analysis["intermediate_more_neg_confirmed"]:
        print("  Current default run does NOT show an intermediate-k increase in negative-eigenvalue count.")
        print("  The visible effect, if any, should be read from Δmin eigenvalue / Δloss instead.")

    print("\nControl / implementation notes:")
    print(f"  - {config['k0_control']}")
    print(f"  - {config['k1_note']}")
    print(
        f"  - max |step(k=1) - step(k=0)| across tested warmup gradients: "
        f"{analysis['max_abs_step_diff_k1_vs_k0']:.3e}"
    )
    print(
        f"  - runtime: {analysis['runtime_seconds']:.2f}s, "
        f"{analysis['total_hessian_evaluations']} Hessians, "
        f"{analysis['total_hessian_gradient_evaluations']} Hessian-side gradient evaluations"
    )


def main() -> None:
    results = run_experiment(verbose=True)
    print_results_report(results)


if __name__ == "__main__":
    main()
