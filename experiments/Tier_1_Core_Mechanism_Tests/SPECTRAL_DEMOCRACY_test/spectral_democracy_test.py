#!/usr/bin/env python3
"""
Experiment 2.11: Spectral democracy as a candidate mechanism for Muon's
advantage in a 3-layer 4x4 deep-linear toy model.

This script compares four update rules on the same ill-conditioned deep-linear
regression problem:
  (a) SGD (baseline)
  (b) Muon (per-layer SVD polar factor update)
  (c) DemocraticSGD (equalize gradient projection magnitudes in the current
      Hessian eigenbasis, then rescale to the gradient norm)
  (d) RandomDemocratic (same equalization idea in a random orthogonal basis)

The goal is descriptive and mechanistic within this toy setting: test whether a
curvature-basis "spectral democracy" proxy tracks Muon's advantage and whether
an oracle Hessian-basis control recovers part of that advantage. The T1/T2/T3
summaries are heuristic diagnostics, not formal statistical proof and not a
claim of generality beyond this 3-layer 4x4 deep-linear family.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Sequence

import numpy as np

# Fixed toy-model architecture (kept intentionally unchanged in this pass).
DIM = 4
N_LAYERS = 3
N_PARAMS = N_LAYERS * DIM * DIM
N_STEPS = 500
HESSIAN_RECOMPUTE_EVERY = 50
N_SEEDS = 5
TARGET_SINGULAR_VALUES = np.array([100.0, 10.0, 1.0, 0.1], dtype=float)
DEFAULT_LR_CANDIDATES = np.array([1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2], dtype=float)
METHODS = ("SGD", "Muon", "DemocraticSGD", "RandomDemocratic")


# ── configuration / metadata helpers ────────────────────────────────────────

def make_default_config() -> Dict[str, Any]:
    return {
        "experiment_name": "Experiment 2.11: Spectral Democracy Test",
        "scope_note": (
            "Toy-model evidence for spectral democracy as a candidate mechanism "
            "for Muon's advantage in a 3-layer 4x4 deep-linear network."
        ),
        "dim": DIM,
        "n_layers": N_LAYERS,
        "n_params": N_PARAMS,
        "n_steps": N_STEPS,
        "hessian_recompute_every": HESSIAN_RECOMPUTE_EVERY,
        "n_seeds": N_SEEDS,
        "seed_start": 42,
        "seed_stride": 7,
        "target_singular_values": TARGET_SINGULAR_VALUES.copy(),
        "lr_candidates": DEFAULT_LR_CANDIDATES.copy(),
        "random_basis_seed_policy": {
            "mode": "fixed",
            "base_seed": 999,
            "description": (
                "Preserve the current experiment's single fixed random-basis seed "
                "for all seeds and methods."
            ),
        },
        "representative_seed_index": 0,
        "loss_sample_stride": 50,
        "democracy_sample_stride": 100,
        "hypothesis_thresholds": {
            "T1_d_ratio_min": 3.0,
            "T2_recovery_min_percent": 60.0,
            "T3_fraction_min": 0.8,
        },
        "loss_trajectory_semantics": (
            "losses[0] is the initial loss before any update; losses[t+1] is the "
            "post-update loss after optimization step t. final_loss == losses[-1]."
        ),
        "democracy_trajectory_semantics": (
            "democracy_ratios[t] is the democracy proxy p10/p90 of absolute "
            "Hessian-eigenbasis projections for the update direction used at step t."
        ),
    }


def resolve_config(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = copy.deepcopy(make_default_config())
    if config is not None:
        for key, value in config.items():
            if isinstance(cfg.get(key), dict) and isinstance(value, dict):
                cfg[key].update(copy.deepcopy(value))
            else:
                cfg[key] = copy.deepcopy(value)

    cfg["target_singular_values"] = np.asarray(cfg["target_singular_values"], dtype=float)
    cfg["lr_candidates"] = np.asarray(cfg["lr_candidates"], dtype=float)

    if cfg["dim"] != DIM or cfg["n_layers"] != N_LAYERS or cfg["n_params"] != N_PARAMS:
        raise ValueError(
            "This first completion pass intentionally keeps the fixed 3-layer 4x4 "
            "deep-linear toy architecture unchanged."
        )
    return cfg


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {key: to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(value) for value in obj]
    return obj


def build_seed_list(config: Dict[str, Any]) -> List[int]:
    return [int(config["seed_start"] + idx * config["seed_stride"]) for idx in range(int(config["n_seeds"]))]


def resolve_random_basis_seed(seed: int, config: Dict[str, Any]) -> int:
    policy = config["random_basis_seed_policy"]
    mode = policy.get("mode", "fixed")
    base_seed = int(policy.get("base_seed", 999))
    if mode == "fixed":
        return base_seed
    if mode == "offset_per_seed":
        return base_seed + int(seed)
    raise ValueError(f"Unsupported random-basis seed policy mode: {mode!r}")


def estimate_workload(config: Dict[str, Any]) -> Dict[str, Any]:
    n_steps = int(config["n_steps"])
    hessian_every = int(config["hessian_recompute_every"])
    lr_count = len(config["lr_candidates"])
    n_methods = len(METHODS)
    n_seeds = int(config["n_seeds"])
    hessian_builds_per_run = len(range(0, n_steps, hessian_every))
    total_training_runs = n_seeds * n_methods * (lr_count + 1)
    total_hessian_builds = total_training_runs * hessian_builds_per_run
    gradient_evals_for_hessians = total_hessian_builds * (2 * N_PARAMS)
    gradient_evals_for_training = total_training_runs * n_steps
    approx_total_gradient_evals = gradient_evals_for_hessians + gradient_evals_for_training
    return {
        "hessian_builds_per_run": hessian_builds_per_run,
        "total_training_runs": total_training_runs,
        "total_hessian_builds": total_hessian_builds,
        "gradient_evaluations_for_hessians": gradient_evals_for_hessians,
        "gradient_evaluations_for_training": gradient_evals_for_training,
        "approx_total_gradient_evaluations": approx_total_gradient_evals,
        "notes": (
            "Approximate operation count for the current implementation; excludes "
            "the cost of eigendecompositions and per-layer SVDs."
        ),
    }


# ── model / optimization helpers ────────────────────────────────────────────

def pack(Ws: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate([W.ravel() for W in Ws])


def unpack(theta: np.ndarray) -> List[np.ndarray]:
    Ws = []
    idx = 0
    for _ in range(N_LAYERS):
        Ws.append(theta[idx:idx + DIM * DIM].reshape(DIM, DIM))
        idx += DIM * DIM
    return Ws


def forward(Ws: Sequence[np.ndarray]) -> np.ndarray:
    out = Ws[0]
    for W in Ws[1:]:
        out = W @ out
    return out


def loss_fn(theta: np.ndarray, target: np.ndarray) -> float:
    Ws = unpack(theta)
    diff = forward(Ws) - target
    return 0.5 * np.sum(diff ** 2)


def grad_fn(theta: np.ndarray, target: np.ndarray) -> np.ndarray:
    Ws = unpack(theta)
    prod = forward(Ws)
    residual = prod - target

    grads = []
    for k in range(N_LAYERS):
        left_factor = np.eye(DIM)
        for j in range(k + 1, N_LAYERS):
            left_factor = Ws[j] @ left_factor

        right_factor = np.eye(DIM)
        for j in range(0, k):
            right_factor = Ws[j] @ right_factor

        dWk = left_factor.T @ residual @ right_factor.T
        grads.append(dWk.ravel())
    return np.concatenate(grads)


def grad_matrices(theta: np.ndarray, target: np.ndarray) -> List[np.ndarray]:
    return grad_matrices_from_grad_vec(grad_fn(theta, target))


def grad_matrices_from_grad_vec(grad_vec: np.ndarray) -> List[np.ndarray]:
    mats = []
    for k in range(N_LAYERS):
        mats.append(grad_vec[k * DIM * DIM:(k + 1) * DIM * DIM].reshape(DIM, DIM))
    return mats


def hessian_fn(theta: np.ndarray, target: np.ndarray) -> np.ndarray:
    n = len(theta)
    H = np.zeros((n, n))
    eps = 1e-5
    for i in range(n):
        theta_p = theta.copy()
        theta_m = theta.copy()
        theta_p[i] += eps
        theta_m[i] -= eps
        g_p = grad_fn(theta_p, target)
        g_m = grad_fn(theta_m, target)
        H[:, i] = (g_p - g_m) / (2 * eps)
    H = 0.5 * (H + H.T)
    return H


def democracy_ratio(direction_vec: np.ndarray, eigvecs: np.ndarray) -> float:
    projs = np.abs(eigvecs.T @ direction_vec)
    if np.max(projs) < 1e-30:
        return 0.0
    p10 = np.percentile(projs, 10)
    p90 = np.percentile(projs, 90)
    if p90 < 1e-30:
        return 0.0
    return float(p10 / p90)


def polar_factor_svd(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M, full_matrices=True)
    return U @ Vt


def muon_direction(theta: np.ndarray, target: np.ndarray) -> np.ndarray:
    return muon_direction_from_grad_vec(grad_fn(theta, target))


def muon_direction_from_grad_vec(grad_vec: np.ndarray) -> np.ndarray:
    gmats = grad_matrices_from_grad_vec(grad_vec)
    polars = []
    for gm in gmats:
        polars.append(polar_factor_svd(gm).ravel())
    return np.concatenate(polars)


def democratic_direction(grad_vec: np.ndarray, eigvecs: np.ndarray) -> np.ndarray:
    projs = eigvecs.T @ grad_vec
    signs = np.sign(projs)
    magnitudes = np.abs(projs)
    mean_mag = np.mean(magnitudes)
    eq_projs = signs * mean_mag
    return eigvecs @ eq_projs


def random_democratic_direction(grad_vec: np.ndarray, rng_basis: np.ndarray) -> np.ndarray:
    projs = rng_basis.T @ grad_vec
    signs = np.sign(projs)
    magnitudes = np.abs(projs)
    mean_mag = np.mean(magnitudes)
    eq_projs = signs * mean_mag
    return rng_basis @ eq_projs


def random_orthogonal(n: int, rng: np.random.RandomState) -> np.ndarray:
    M = rng.randn(n, n)
    Q, R = np.linalg.qr(M)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    return Q


def rescale_to_match_norm(candidate_direction: np.ndarray, reference_direction: np.ndarray) -> np.ndarray:
    ref_norm = np.linalg.norm(reference_direction)
    cand_norm = np.linalg.norm(candidate_direction)
    if cand_norm > 1e-12:
        return candidate_direction * (ref_norm / cand_norm)
    return candidate_direction


def compute_update_direction(
    method: str,
    theta: np.ndarray,
    target: np.ndarray,
    grad_vec: np.ndarray,
    hessian_eigvecs: np.ndarray,
    rand_basis: np.ndarray,
) -> np.ndarray:
    if method == "SGD":
        return grad_vec
    if method == "Muon":
        return muon_direction_from_grad_vec(grad_vec)
    if method == "DemocraticSGD":
        return rescale_to_match_norm(democratic_direction(grad_vec, hessian_eigvecs), grad_vec)
    if method == "RandomDemocratic":
        return rescale_to_match_norm(random_democratic_direction(grad_vec, rand_basis), grad_vec)
    raise ValueError(f"Unknown method: {method}")


def safe_ratio(numer: float, denom: float, zero_value: float = 0.0) -> float:
    if abs(denom) <= 1e-12:
        return zero_value
    return float(numer / denom)


def compute_recovery(sgd_loss: float, muon_loss: float, alt_loss: float) -> float:
    gap = sgd_loss - muon_loss
    if gap > 1e-12:
        return float((sgd_loss - alt_loss) / gap * 100.0)
    return 0.0


def make_problem_for_seed(seed: int, config: Dict[str, Any]) -> Dict[str, Any]:
    rng_init = np.random.RandomState(seed)
    U_t, _ = np.linalg.qr(rng_init.randn(DIM, DIM))
    V_t, _ = np.linalg.qr(rng_init.randn(DIM, DIM))
    target = U_t @ np.diag(config["target_singular_values"]) @ V_t
    theta0 = 0.3 * rng_init.randn(N_PARAMS)
    return {
        "target": target,
        "theta0": theta0,
        "target_condition_number": float(np.linalg.cond(target)),
    }


def run_single_method(
    method: str,
    lr: float,
    theta0: np.ndarray,
    target: np.ndarray,
    *,
    n_steps: int = N_STEPS,
    hessian_recompute_every: int = HESSIAN_RECOMPUTE_EVERY,
    seed_rb: int = 999,
) -> float:
    """Run one optimizer / learning-rate candidate and return the terminal loss."""
    theta = theta0.copy()
    rng = np.random.RandomState(seed_rb)
    hessian_eigvecs = None
    rand_basis = None

    for step in range(n_steps):
        grad_vec = grad_fn(theta, target)

        if step % hessian_recompute_every == 0:
            H = hessian_fn(theta, target)
            _, hessian_eigvecs = np.linalg.eigh(H)
            rand_basis = random_orthogonal(len(theta), rng)

        direction = compute_update_direction(method, theta, target, grad_vec, hessian_eigvecs, rand_basis)
        theta -= lr * direction
        current_loss = loss_fn(theta, target)
        if np.isnan(current_loss) or current_loss > 1e8:
            return 1e8
    return float(loss_fn(theta, target))


def run_full(
    theta0: np.ndarray,
    target: np.ndarray,
    best_lrs: Dict[str, float],
    *,
    n_steps: int = N_STEPS,
    hessian_recompute_every: int = HESSIAN_RECOMPUTE_EVERY,
    seed_rb: int = 999,
) -> Dict[str, Dict[str, Any]]:
    """
    Full optimizer runs returning trajectories.

    Trajectory semantics:
      - losses has length n_steps + 1, with losses[0] the initial loss before any
        update and losses[-1] the terminal post-training loss.
      - democracy_ratios has length n_steps, one proxy measurement per update.
    """
    results: Dict[str, Dict[str, Any]] = {}
    for name in METHODS:
        rng = np.random.RandomState(seed_rb)
        theta = theta0.copy()
        lr = float(best_lrs[name])
        losses = [float(loss_fn(theta, target))]
        dem_ratios: List[float] = []
        hessian_eigvecs = None
        rand_basis = None
        diverged = False

        for step in range(n_steps):
            grad_vec = grad_fn(theta, target)

            if step % hessian_recompute_every == 0:
                H = hessian_fn(theta, target)
                _, hessian_eigvecs = np.linalg.eigh(H)
                rand_basis = random_orthogonal(len(theta), rng)

            direction = compute_update_direction(name, theta, target, grad_vec, hessian_eigvecs, rand_basis)
            dem_ratios.append(democracy_ratio(direction, hessian_eigvecs))
            theta -= lr * direction
            current_loss = float(loss_fn(theta, target))
            if np.isnan(current_loss) or current_loss > 1e8:
                current_loss = 1e8
                diverged = True
            losses.append(current_loss)
            if diverged:
                remaining_steps = n_steps - (step + 1)
                if remaining_steps > 0:
                    losses.extend([1e8] * remaining_steps)
                    dem_ratios.extend([np.nan] * remaining_steps)
                break

        loss_array = np.asarray(losses, dtype=float)
        democracy_array = np.asarray(dem_ratios, dtype=float)
        finite_democracy = democracy_array[np.isfinite(democracy_array)]
        mean_democracy = float(np.mean(finite_democracy)) if finite_democracy.size else np.nan
        results[name] = {
            "losses": loss_array,
            "loss_steps": np.arange(len(loss_array), dtype=int),
            "final_loss": float(loss_array[-1]),
            "democracy_ratios": democracy_array,
            "democracy_steps": np.arange(len(democracy_array), dtype=int),
            "mean_democracy": mean_democracy,
            "diverged": bool(diverged),
        }
    return results


# ── higher-level experiment assembly ────────────────────────────────────────

def run_lr_search(
    theta0: np.ndarray,
    target: np.ndarray,
    config: Dict[str, Any],
    *,
    seed_rb: int,
) -> Dict[str, Dict[str, Any]]:
    lr_candidates = [float(lr) for lr in config["lr_candidates"]]
    lr_search: Dict[str, Dict[str, Any]] = {}

    for method in METHODS:
        best_loss = 1e20
        best_lr = lr_candidates[0]
        candidate_final_losses = []
        for lr in lr_candidates:
            final_loss = run_single_method(
                method,
                lr,
                theta0,
                target,
                n_steps=int(config["n_steps"]),
                hessian_recompute_every=int(config["hessian_recompute_every"]),
                seed_rb=seed_rb,
            )
            candidate_final_losses.append(final_loss)
            if final_loss < best_loss:
                best_loss = final_loss
                best_lr = lr
        lr_search[method] = {
            "candidate_lrs": np.asarray(lr_candidates, dtype=float),
            "candidate_final_losses": np.asarray(candidate_final_losses, dtype=float),
            "best_lr": float(best_lr),
            "best_final_loss": float(best_loss),
        }
    return lr_search


def build_seed_result(seed: int, config: Dict[str, Any]) -> Dict[str, Any]:
    problem = make_problem_for_seed(seed, config)
    theta0 = problem["theta0"]
    target = problem["target"]
    seed_rb = resolve_random_basis_seed(seed, config)

    lr_search = run_lr_search(theta0, target, config, seed_rb=seed_rb)
    best_lrs = {method: lr_search[method]["best_lr"] for method in METHODS}
    method_runs = run_full(
        theta0,
        target,
        best_lrs,
        n_steps=int(config["n_steps"]),
        hessian_recompute_every=int(config["hessian_recompute_every"]),
        seed_rb=seed_rb,
    )

    final_losses = {method: float(method_runs[method]["final_loss"]) for method in METHODS}
    recoveries = {
        "DemocraticSGD": compute_recovery(final_losses["SGD"], final_losses["Muon"], final_losses["DemocraticSGD"]),
        "RandomDemocratic": compute_recovery(final_losses["SGD"], final_losses["Muon"], final_losses["RandomDemocratic"]),
    }
    d_ratio_muon_over_sgd = safe_ratio(method_runs["Muon"]["mean_democracy"], method_runs["SGD"]["mean_democracy"])

    return {
        "seed": int(seed),
        "random_basis_seed": int(seed_rb),
        "target_condition_number": float(problem["target_condition_number"]),
        "target_matrix": target.copy(),
        "theta0": theta0.copy(),
        "lr_search": lr_search,
        "best_lrs": best_lrs,
        "method_runs": method_runs,
        "final_losses": final_losses,
        "recoveries": recoveries,
        "d_ratio_muon_over_sgd": float(d_ratio_muon_over_sgd),
    }


def build_seed_summary_rows(seed_results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for seed_result in seed_results:
        row = {
            "seed": int(seed_result["seed"]),
            "target_condition_number": float(seed_result["target_condition_number"]),
            "SGD_final_loss": float(seed_result["final_losses"]["SGD"]),
            "Muon_final_loss": float(seed_result["final_losses"]["Muon"]),
            "DemocraticSGD_final_loss": float(seed_result["final_losses"]["DemocraticSGD"]),
            "RandomDemocratic_final_loss": float(seed_result["final_losses"]["RandomDemocratic"]),
            "Democratic_recovery_percent": float(seed_result["recoveries"]["DemocraticSGD"]),
            "Random_recovery_percent": float(seed_result["recoveries"]["RandomDemocratic"]),
            "Muon_over_SGD_democracy_ratio": float(seed_result["d_ratio_muon_over_sgd"]),
        }
        for method in METHODS:
            row[f"{method}_best_lr"] = float(seed_result["best_lrs"][method])
        rows.append(row)
    return rows


def build_aggregate_summary(seed_results: Sequence[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    final_losses = {
        method: np.asarray([seed_result["method_runs"][method]["final_loss"] for seed_result in seed_results], dtype=float)
        for method in METHODS
    }
    mean_democracies = {
        method: np.asarray([seed_result["method_runs"][method]["mean_democracy"] for seed_result in seed_results], dtype=float)
        for method in METHODS
    }
    recoveries_dem = np.asarray([seed_result["recoveries"]["DemocraticSGD"] for seed_result in seed_results], dtype=float)
    recoveries_rnd = np.asarray([seed_result["recoveries"]["RandomDemocratic"] for seed_result in seed_results], dtype=float)
    d_ratios = np.asarray([seed_result["d_ratio_muon_over_sgd"] for seed_result in seed_results], dtype=float)

    summary_table_rows = []
    for method in METHODS:
        fl = final_losses[method]
        dm = mean_democracies[method]
        summary_table_rows.append(
            {
                "method": method,
                "mean_final_loss": float(np.mean(fl)),
                "std_final_loss": float(np.std(fl)),
                "mean_democracy": float(np.nanmean(dm)),
                "std_democracy": float(np.nanstd(dm)),
            }
        )

    thresholds = config["hypothesis_thresholds"]
    mean_d_ratio = float(np.mean(d_ratios))
    mean_rec_dem = float(np.mean(recoveries_dem))
    mean_rec_rnd = float(np.mean(recoveries_rnd))
    t3_count = int(np.sum(recoveries_rnd < recoveries_dem))
    t3_fraction = float(t3_count / max(len(recoveries_dem), 1))

    tests = {
        "T1": {
            "description": "Mean Muon/SGD democracy-ratio multiplier exceeds heuristic 3x threshold.",
            "threshold": float(thresholds["T1_d_ratio_min"]),
            "value": mean_d_ratio,
            "per_seed_values": d_ratios,
            "passed": bool(mean_d_ratio >= thresholds["T1_d_ratio_min"]),
            "caveat": "Heuristic descriptive threshold; not a formal significance test.",
        },
        "T2": {
            "description": "Mean DemocraticSGD recovery exceeds heuristic 60% threshold.",
            "threshold": float(thresholds["T2_recovery_min_percent"]),
            "value": mean_rec_dem,
            "per_seed_values": recoveries_dem,
            "passed": bool(mean_rec_dem > thresholds["T2_recovery_min_percent"]),
            "caveat": "Recovery is defined relative to the SGD-to-Muon final-loss gap in this toy experiment.",
        },
        "T3": {
            "description": "RandomDemocratic underperforms Hessian-basis DemocraticSGD in at least 80% of seeds.",
            "threshold_fraction": float(thresholds["T3_fraction_min"]),
            "count": t3_count,
            "n_seeds": int(len(recoveries_dem)),
            "fraction": t3_fraction,
            "recoveries_random": recoveries_rnd,
            "recoveries_democratic": recoveries_dem,
            "passed": bool(t3_fraction >= thresholds["T3_fraction_min"]),
            "caveat": "The random-basis control currently uses a fixed basis-seed policy by default.",
        },
    }
    tests["overall_pass"] = bool(tests["T1"]["passed"] and tests["T2"]["passed"] and tests["T3"]["passed"])
    tests["note"] = (
        "T1/T2/T3 are heuristic summaries intended for descriptive mechanistic evidence "
        "within this toy model, not definitive or general proof."
    )

    return {
        "final_losses": final_losses,
        "mean_democracies": mean_democracies,
        "recoveries_dem": recoveries_dem,
        "recoveries_rnd": recoveries_rnd,
        "d_ratios_muon_over_sgd": d_ratios,
        "summary_table_rows": summary_table_rows,
        "tests": tests,
        "mean_recovery_dem": mean_rec_dem,
        "mean_recovery_rnd": mean_rec_rnd,
        "mean_d_ratio_muon_over_sgd": mean_d_ratio,
        "seed_summary_rows": build_seed_summary_rows(seed_results),
    }


def _sample_steps(total_steps_inclusive: int, stride: int) -> List[int]:
    steps = list(range(0, total_steps_inclusive + 1, max(1, stride)))
    if steps[-1] != total_steps_inclusive:
        steps.append(total_steps_inclusive)
    return steps


def _sample_update_steps(n_steps: int, stride: int) -> List[int]:
    if n_steps <= 0:
        return []
    steps = list(range(0, n_steps, max(1, stride)))
    if steps[-1] != n_steps - 1:
        steps.append(n_steps - 1)
    return steps


def compute_energy_distribution_summary(
    theta0: np.ndarray,
    target: np.ndarray,
    *,
    seed_rb: int,
) -> Dict[str, Any]:
    H0 = hessian_fn(theta0, target)
    evals0, evecs0 = np.linalg.eigh(H0)
    g0 = grad_fn(theta0, target)
    rng_rb = np.random.RandomState(seed_rb)
    rand_basis0 = random_orthogonal(len(theta0), rng_rb)

    directions = {
        "SGD": g0,
        "Muon": muon_direction_from_grad_vec(g0),
        "DemocraticSGD": rescale_to_match_norm(democratic_direction(g0, evecs0), g0),
        "RandomDemocratic": rescale_to_match_norm(random_democratic_direction(g0, rand_basis0), g0),
    }

    topk_summary: Dict[str, Dict[str, float]] = {}
    quartile_summary: Dict[str, Dict[str, float]] = {}
    quartile_labels = ["Q1_smallest_eigs", "Q2", "Q3", "Q4_largest_eigs"]
    eig_order = np.argsort(evals0)
    quartile_indices = np.array_split(eig_order, 4)

    for method, direction in directions.items():
        projs_sq = (evecs0.T @ direction) ** 2
        total = float(np.sum(projs_sq))
        if total <= 1e-30:
            topk_summary[method] = {"top1": 0.0, "top3": 0.0, "top10": 0.0}
            quartile_summary[method] = {label: 0.0 for label in quartile_labels}
            continue

        sorted_sq = np.sort(projs_sq)[::-1]
        topk_summary[method] = {
            "top1": float(sorted_sq[0] / total * 100.0),
            "top3": float(np.sum(sorted_sq[:3]) / total * 100.0),
            "top10": float(np.sum(sorted_sq[:10]) / total * 100.0),
        }
        quartile_summary[method] = {
            label: float(np.sum(projs_sq[idxs]) / total * 100.0)
            for label, idxs in zip(quartile_labels, quartile_indices)
        }

    nonzero_abs_evals = np.abs(evals0[np.abs(evals0) > 1e-6])
    spectrum_condition = float(np.max(np.abs(evals0)) / (np.min(nonzero_abs_evals) + 1e-12)) if nonzero_abs_evals.size else np.inf

    return {
        "seed_random_basis": int(seed_rb),
        "topk_percent": topk_summary,
        "quartile_energy_percent": quartile_summary,
        "hessian_spectrum": {
            "smallest5": evals0[:5],
            "largest5": evals0[-5:],
            "condition_estimate": spectrum_condition,
        },
        "initial_direction_democracy": {
            method: democracy_ratio(direction, evecs0)
            for method, direction in directions.items()
        },
    }


def build_representative_summary(seed_results: Sequence[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    rep_idx = int(config["representative_seed_index"])
    if not seed_results:
        raise ValueError("Need at least one seed result to build a representative summary.")
    if rep_idx < 0 or rep_idx >= len(seed_results):
        raise IndexError("representative_seed_index is out of range for the available seed results.")

    rep_seed = seed_results[rep_idx]
    n_steps = int(config["n_steps"])
    loss_steps = _sample_steps(n_steps, int(config["loss_sample_stride"]))
    democracy_steps = _sample_update_steps(n_steps, int(config["democracy_sample_stride"]))

    loss_table_rows = []
    for step in loss_steps:
        row = {"step": int(step)}
        for method in METHODS:
            row[method] = float(rep_seed["method_runs"][method]["losses"][step])
        loss_table_rows.append(row)

    democracy_table_rows = []
    for step in democracy_steps:
        row = {"step": int(step)}
        for method in METHODS:
            row[method] = float(rep_seed["method_runs"][method]["democracy_ratios"][step])
        democracy_table_rows.append(row)

    energy_distribution = compute_energy_distribution_summary(
        rep_seed["theta0"],
        rep_seed["target_matrix"],
        seed_rb=int(rep_seed["random_basis_seed"]),
    )

    return {
        "seed_index": rep_idx,
        "seed": int(rep_seed["seed"]),
        "loss_table_rows": loss_table_rows,
        "democracy_table_rows": democracy_table_rows,
        "energy_distribution": energy_distribution,
    }


def run_experiment(config: Dict[str, Any] | None = None, *, verbose: bool = True) -> Dict[str, Any]:
    cfg = resolve_config(config)
    workload = estimate_workload(cfg)
    seed_results = []
    seeds = build_seed_list(cfg)

    if verbose:
        print("=" * 78)
        print(cfg["experiment_name"])
        print("=" * 78)
        print(cfg["scope_note"])
        print(f"Network: {N_LAYERS}-layer deep linear {DIM}x{DIM} ({N_PARAMS} params)")
        print(f"Steps: {cfg['n_steps']}, Hessian recompute every {cfg['hessian_recompute_every']} steps")
        print(f"Seeds: {seeds}")
        print(f"Random-basis seed policy: {cfg['random_basis_seed_policy']}")
        print(
            f"Approx workload: {workload['total_training_runs']} training runs, "
            f"{workload['total_hessian_builds']} Hessian builds, "
            f"~{workload['approx_total_gradient_evaluations']} gradient evaluations"
        )
        print()

    for seed in seeds:
        seed_result = build_seed_result(seed, cfg)
        seed_results.append(seed_result)
        if verbose:
            print(f"--- Seed {seed_result['seed']} (target cond={seed_result['target_condition_number']:.0f}) ---")
            print(
                "  LRs: "
                f"SGD={seed_result['best_lrs']['SGD']}, "
                f"Muon={seed_result['best_lrs']['Muon']}, "
                f"Dem={seed_result['best_lrs']['DemocraticSGD']}, "
                f"Rnd={seed_result['best_lrs']['RandomDemocratic']}"
            )
            print(
                "  Final: "
                f"SGD={seed_result['final_losses']['SGD']:.4f} "
                f"Muon={seed_result['final_losses']['Muon']:.4f} "
                f"Dem={seed_result['final_losses']['DemocraticSGD']:.4f} "
                f"Rnd={seed_result['final_losses']['RandomDemocratic']:.4f}"
            )
            print(
                "  Proxy ratios: "
                f"D_Muon/D_SGD={seed_result['d_ratio_muon_over_sgd']:.2f}x  "
                f"RecDem={seed_result['recoveries']['DemocraticSGD']:.1f}%  "
                f"RecRnd={seed_result['recoveries']['RandomDemocratic']:.1f}%"
            )
            print()

    results = {
        "experiment_name": cfg["experiment_name"],
        "scope_note": cfg["scope_note"],
        "methods": list(METHODS),
        "config": cfg,
        "workload_estimate": workload,
        "seed_results": seed_results,
    }
    results["aggregate"] = build_aggregate_summary(seed_results, cfg)
    results["representative"] = build_representative_summary(seed_results, cfg)
    results["notes"] = {
        "loss_trajectory_semantics": cfg["loss_trajectory_semantics"],
        "democracy_trajectory_semantics": cfg["democracy_trajectory_semantics"],
        "claim_scope": cfg["scope_note"],
    }
    return results


def run_basic_consistency_checks(results: Dict[str, Any]) -> Dict[str, Any]:
    cfg = results["config"]
    n_steps = int(cfg["n_steps"])
    checked_runs = 0

    for seed_result in results["seed_results"]:
        for method in METHODS:
            run = seed_result["method_runs"][method]
            losses = np.asarray(run["losses"], dtype=float)
            democracy = np.asarray(run["democracy_ratios"], dtype=float)
            if len(losses) != n_steps + 1:
                raise AssertionError(f"{method} losses length mismatch for seed {seed_result['seed']}: {len(losses)} != {n_steps + 1}")
            if len(democracy) != n_steps:
                raise AssertionError(f"{method} democracy length mismatch for seed {seed_result['seed']}: {len(democracy)} != {n_steps}")
            if not np.isclose(float(run["final_loss"]), float(losses[-1]), rtol=1e-12, atol=1e-12):
                raise AssertionError(f"{method} final_loss does not match terminal loss trajectory for seed {seed_result['seed']}")
            checked_runs += 1

    rep = results["representative"]
    if not rep["loss_table_rows"]:
        raise AssertionError("Representative loss table is unexpectedly empty.")
    if not rep["democracy_table_rows"]:
        raise AssertionError("Representative democracy table is unexpectedly empty.")

    return {
        "checked_seed_method_runs": checked_runs,
        "n_steps": n_steps,
        "loss_semantics": results["notes"]["loss_trajectory_semantics"],
        "democracy_semantics": results["notes"]["democracy_trajectory_semantics"],
    }


# ── reporting ────────────────────────────────────────────────────────────────

def print_experiment_report(results: Dict[str, Any]) -> None:
    cfg = results["config"]
    aggregate = results["aggregate"]
    representative = results["representative"]

    print()
    print("=" * 78)
    print("AGGREGATE RESULTS ACROSS ALL SEEDS")
    print("=" * 78)
    print()
    print(f"{'Optimizer':<22} {'Mean Final Loss':>16} {'Std':>10} {'Mean Democracy':>16}")
    print("-" * 70)
    for row in aggregate["summary_table_rows"]:
        print(
            f"{row['method']:<22} "
            f"{row['mean_final_loss']:>16.6f} "
            f"{row['std_final_loss']:>10.6f} "
            f"{row['mean_democracy']:>16.6f}"
        )
    print()

    d_ratios = aggregate["d_ratios_muon_over_sgd"]
    print(f"D_Muon / D_SGD across seeds: {np.round(d_ratios, 2)}")
    print(f"  mean = {aggregate['mean_d_ratio_muon_over_sgd']:.2f}x, min = {np.min(d_ratios):.2f}x, max = {np.max(d_ratios):.2f}x")
    print()

    recoveries_dem = aggregate["recoveries_dem"]
    recoveries_rnd = aggregate["recoveries_rnd"]
    print(f"DemocraticSGD recovery across seeds: {np.round(recoveries_dem, 1)}")
    print(f"  mean = {aggregate['mean_recovery_dem']:.1f}%, min = {np.min(recoveries_dem):.1f}%")
    print()
    print(f"RandomDemocratic recovery across seeds: {np.round(recoveries_rnd, 1)}")
    print(f"  mean = {aggregate['mean_recovery_rnd']:.1f}%, min = {np.min(recoveries_rnd):.1f}%")
    print()

    print(f"Representative loss trajectory (seed {representative['seed']}):")
    header = f"{'Step':>6}  {'SGD':>14}  {'Muon':>14}  {'DemSGD':>14}  {'RndDem':>14}"
    print(header)
    for row in representative["loss_table_rows"]:
        print(
            f"{row['step']:>6}  "
            f"{row['SGD']:>14.6f}  "
            f"{row['Muon']:>14.6f}  "
            f"{row['DemocraticSGD']:>14.6f}  "
            f"{row['RandomDemocratic']:>14.6f}"
        )
    print()

    print(f"Representative democracy-ratio trajectory (seed {representative['seed']}):")
    header = f"{'Step':>6}  {'SGD':>10}  {'Muon':>10}  {'DemSGD':>10}  {'RndDem':>10}"
    print(header)
    for row in representative["democracy_table_rows"]:
        print(
            f"{row['step']:>6}  "
            f"{row['SGD']:>10.6f}  "
            f"{row['Muon']:>10.6f}  "
            f"{row['DemocraticSGD']:>10.6f}  "
            f"{row['RandomDemocratic']:>10.6f}"
        )
    print()

    tests = aggregate["tests"]
    print("=" * 78)
    print("HEURISTIC T1/T2/T3 SUMMARIES (DESCRIPTIVE, TOY-MODEL ONLY)")
    print("=" * 78)
    print(f"T1: mean D_Muon / D_SGD = {tests['T1']['value']:.2f}x vs threshold {tests['T1']['threshold']:.2f}x -> {'PASS' if tests['T1']['passed'] else 'FAIL'}")
    print(f"T2: mean DemocraticSGD recovery = {tests['T2']['value']:.1f}% vs threshold {tests['T2']['threshold']:.1f}% -> {'PASS' if tests['T2']['passed'] else 'FAIL'}")
    print(
        f"T3: RandomDemocratic recovery < DemocraticSGD recovery in "
        f"{tests['T3']['count']}/{tests['T3']['n_seeds']} seeds "
        f"({100.0 * tests['T3']['fraction']:.1f}% vs threshold {100.0 * tests['T3']['threshold_fraction']:.1f}%) "
        f"-> {'PASS' if tests['T3']['passed'] else 'FAIL'}"
    )
    print(f"Overall heuristic verdict: {'ALL THREE PASS' if tests['overall_pass'] else 'SOME TESTS FAIL'}")
    print(f"Note: {tests['note']}")
    print()

    print("Interpretation (calibrated to implemented evidence):")
    if tests["T2"]["value"] > 0.0:
        print(
            f"  • In this toy model, Hessian-basis equalization recovers {tests['T2']['value']:.1f}% "
            "of Muon's final-loss advantage on average, supporting spectral democracy as a candidate mechanism."
        )
    if tests["T3"]["passed"]:
        print(
            "  • The Hessian-specific control usually outperforms random-basis equalization, suggesting that "
            "curvature-basis structure matters here."
        )
    elif tests["T3"]["value"] if "value" in tests["T3"] else False:
        pass
    if tests["T1"]["value"] > 1.0:
        print(
            f"  • Muon is more democratic than SGD by the p10/p90 proxy (mean ratio {tests['T1']['value']:.2f}x), "
            "though this is only one summary statistic and not a direct uniformity proof."
        )
    print(
        "  • Limits: exact finite-difference Hessians, a single 3-layer 4x4 deep-linear family, 5 seeds, "
        "and a fixed random-basis seed policy by default."
    )
    print()

    energy = representative["energy_distribution"]
    print(f"Energy distribution in Hessian eigenbasis at init (representative seed {representative['seed']}):")
    for method in METHODS:
        topk = energy["topk_percent"][method]
        print(
            f"  {method:<18} top-1: {topk['top1']:.1f}%  top-3: {topk['top3']:.1f}%  top-10: {topk['top10']:.1f}%"
        )
    print()
    print("Initial Hessian spectrum (representative seed):")
    print(f"  5 smallest: {np.round(energy['hessian_spectrum']['smallest5'], 2)}")
    print(f"  5 largest:  {np.round(energy['hessian_spectrum']['largest5'], 2)}")
    print(f"  condition estimate: {energy['hessian_spectrum']['condition_estimate']:.0f}")
    print()
    print("=" * 78)


def main() -> None:
    results = run_experiment(verbose=True)
    run_basic_consistency_checks(results)
    print_experiment_report(results)


if __name__ == "__main__":
    main()
