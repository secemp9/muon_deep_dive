#!/usr/bin/env python3
"""
Experiment 2.6: Product kappa crossover vs depth
================================================

Toy-scope study of conditioning trajectories in deep linear networks.

What this script actually does:
- builds one fixed synthetic linear-regression problem;
- compares a deterministic full-batch gradient-descent baseline (historical
  key name ``sgd`` retained for continuity) against Muon updates;
- tracks ``log kappa_prod = sum_l log kappa(W_l)`` at every training step; and
- records the first ``80%-dominance crossover`` step, defined as the first
  tracked step from which Muon's log-product-condition-number is lower than the
  baseline's for at least 80% of the remaining tracked steps.

This is an exploratory single-seed nested-depth sweep, not a statistically
strong test of ``O(n / L^2)`` scaling. Any fit of crossover step versus depth
should be read as descriptive of this one run.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np


DEFAULT_CONFIG: Dict[str, Any] = {
    "experiment_name": "Experiment 2.6: Product Kappa Crossover vs Depth",
    "study_scope": "single-seed deterministic fixed-batch toy sweep",
    "width": 32,
    "depths": [4, 6, 8, 12, 16],
    "num_steps": 500,
    "lr_sgd": 0.005,
    "lr_muon": 0.01,
    "ns_iters": 5,
    "batch_size": 64,
    "input_dim": 32,
    "output_dim": 32,
    "seed": 42,
    "track_every": 1,
    "input_scale": 0.5,
    "target_sigma0": 10.0,
    "target_decay": 0.7,
    "crossover_fraction": 0.80,
    "baseline_label": "sgd",
    "baseline_description": (
        "deterministic full-batch gradient descent on one fixed synthetic batch; "
        "historical 'sgd' label retained for continuity"
    ),
    "muon_description": "gradient descent with Newton-Schulz-orthogonalized layer updates",
    "reported_loss_definition": "0.5 * ||Y_pred - Y_target||_F^2 / batch_size",
    "crossover_definition": (
        "first tracked step i such that log_kappa_muon[i] < log_kappa_sgd[i] "
        "for at least 80% of tracked steps from i through the end"
    ),
    "depth_initialization_note": (
        "depths reuse the same random seed, so deeper networks extend shallower "
        "initial prefixes rather than providing independent random draws"
    ),
}


# =============================================================================
# CONFIGURATION UTILITIES
# =============================================================================


def get_default_config() -> Dict[str, Any]:
    """Return a copy of the default experiment configuration."""
    config = dict(DEFAULT_CONFIG)
    config["depths"] = list(DEFAULT_CONFIG["depths"])
    return config


def resolve_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Merge caller overrides with defaults and validate first-pass assumptions."""
    resolved = get_default_config()
    if config:
        for key, value in config.items():
            resolved[key] = list(value) if key == "depths" else value

    if resolved["track_every"] <= 0:
        raise ValueError("track_every must be positive.")

    if not (
        resolved["width"] == resolved["input_dim"] == resolved["output_dim"]
    ):
        raise ValueError(
            "This first-pass implementation assumes square deep linear nets with "
            "width == input_dim == output_dim."
        )

    return resolved


# =============================================================================
# NETWORK UTILITIES
# =============================================================================


def init_weights(num_layers: int, width: int, seed: int) -> List[np.ndarray]:
    """Initialize deep linear net weights with Xavier scaling."""
    rng = np.random.RandomState(seed)
    weights = []
    std = np.sqrt(2.0 / (width + width))
    for _ in range(num_layers):
        weights.append((rng.randn(width, width) * std).copy())
    return weights



def forward_linear(weights: List[np.ndarray], X: np.ndarray) -> np.ndarray:
    """Forward pass through a deep linear net."""
    out = X.copy()
    for W in weights:
        out = W @ out
    return out



def compute_loss(weights: List[np.ndarray], X: np.ndarray, Y_target: np.ndarray) -> float:
    """Loss used for reporting and matched to the implemented gradient.

    The gradient code divides by batch size only, so the consistent scalar
    objective is 0.5 * ||Y_pred - Y_target||_F^2 / batch_size.
    """
    Y_pred = forward_linear(weights, X)
    diff = Y_pred - Y_target
    batch_size = X.shape[1]
    return float(0.5 * np.sum(diff ** 2) / batch_size)



def compute_gradients(
    weights: List[np.ndarray], X: np.ndarray, Y_target: np.ndarray
) -> List[np.ndarray]:
    """Backpropagation through a deep linear net."""
    num_layers = len(weights)
    batch_size = X.shape[1]

    activations = [X.copy()]
    out = X.copy()
    for W in weights:
        out = W @ out
        activations.append(out.copy())

    diff = activations[-1] - Y_target
    delta = diff / batch_size

    grads: List[np.ndarray] = [None] * num_layers  # type: ignore[assignment]
    for layer_idx in range(num_layers - 1, -1, -1):
        grads[layer_idx] = delta @ activations[layer_idx].T
        if layer_idx > 0:
            delta = weights[layer_idx].T @ delta

    return grads



def newton_schulz_orthogonalize(G: np.ndarray, num_iters: int = 5) -> np.ndarray:
    """Approximate the polar factor of G via Newton-Schulz iteration."""
    norm = np.linalg.norm(G, "fro")
    if norm < 1e-12:
        return G

    X = G / norm
    for _ in range(num_iters):
        A = X.T @ X
        X = (15.0 / 8.0) * X - (10.0 / 8.0) * X @ A + (3.0 / 8.0) * X @ A @ A
    return X



def log_product_condition_number(weights: List[np.ndarray]) -> float:
    """Compute sum_l log(kappa(W_l)), using logs to avoid overflow."""
    log_kappa = 0.0
    for W in weights:
        sv = np.linalg.svd(W, compute_uv=False)
        sigma_min = max(float(sv[-1]), 1e-12)
        log_kappa += float(np.log(float(sv[0])) - np.log(sigma_min))
    return float(log_kappa)


# =============================================================================
# TRAINING ROUTINES
# =============================================================================


def train_sgd_tracked(
    weights: List[np.ndarray],
    X: np.ndarray,
    Y: np.ndarray,
    num_steps: int,
    lr: float,
    track_every: int = 1,
) -> tuple[List[np.ndarray], List[float], List[float]]:
    """Train with the baseline update, tracking log-product-kappa and loss."""
    weights = [W.copy() for W in weights]
    kappa_history: List[float] = []
    loss_history: List[float] = []

    for step in range(num_steps):
        if step % track_every == 0:
            kappa_history.append(log_product_condition_number(weights))
            loss_history.append(compute_loss(weights, X, Y))

        grads = compute_gradients(weights, X, Y)
        for i in range(len(weights)):
            weights[i] -= lr * grads[i]

    kappa_history.append(log_product_condition_number(weights))
    loss_history.append(compute_loss(weights, X, Y))
    return weights, kappa_history, loss_history



def train_muon_tracked(
    weights: List[np.ndarray],
    X: np.ndarray,
    Y: np.ndarray,
    num_steps: int,
    lr: float,
    ns_iters: int = 5,
    track_every: int = 1,
) -> tuple[List[np.ndarray], List[float], List[float]]:
    """Train with Muon, tracking log-product-kappa and loss."""
    weights = [W.copy() for W in weights]
    kappa_history: List[float] = []
    loss_history: List[float] = []

    for step in range(num_steps):
        if step % track_every == 0:
            kappa_history.append(log_product_condition_number(weights))
            loss_history.append(compute_loss(weights, X, Y))

        grads = compute_gradients(weights, X, Y)
        for i in range(len(weights)):
            G_orth = newton_schulz_orthogonalize(grads[i], ns_iters)
            weights[i] -= lr * G_orth

    kappa_history.append(log_product_condition_number(weights))
    loss_history.append(compute_loss(weights, X, Y))
    return weights, kappa_history, loss_history



def find_80pct_dominance_crossover(
    kappa_sgd: List[float], kappa_muon: List[float], fraction: float = 0.80
) -> Optional[int]:
    """Find the first tracked step where Muon wins on >= fraction remaining steps."""
    n = len(kappa_sgd)
    for i in range(n):
        if kappa_muon[i] < kappa_sgd[i]:
            remaining = n - i
            if remaining <= 1:
                return i
            count_better = sum(
                1 for j in range(i, n) if kappa_muon[j] < kappa_sgd[j]
            )
            if count_better / remaining >= fraction:
                return i
    return None


# Backwards-compatible alias for earlier terminology.
find_sustained_crossover = find_80pct_dominance_crossover


# =============================================================================
# ANALYSIS UTILITIES
# =============================================================================


def build_problem(config: Dict[str, Any]) -> Dict[str, Any]:
    """Construct the fixed synthetic regression problem used by the experiment."""
    rng = np.random.RandomState(config["seed"])
    X = rng.randn(config["input_dim"], config["batch_size"]) * config["input_scale"]

    rank = min(config["output_dim"], config["input_dim"])
    U, _ = np.linalg.qr(rng.randn(config["output_dim"], config["output_dim"]))
    V, _ = np.linalg.qr(rng.randn(config["input_dim"], config["input_dim"]))
    sigma = np.array(
        [config["target_sigma0"] * (config["target_decay"] ** i) for i in range(rank)],
        dtype=float,
    )
    T = U @ np.diag(sigma) @ V
    Y = T @ X

    return {
        "X": X,
        "Y": Y,
        "sigma": sigma,
        "target_condition_number": float(sigma[0] / sigma[-1]),
        "x_fro_norm": float(np.linalg.norm(X, "fro")),
        "y_fro_norm": float(np.linalg.norm(Y, "fro")),
        "x_shape": tuple(X.shape),
        "y_shape": tuple(Y.shape),
    }



def make_tracked_steps(num_steps: int, track_every: int) -> List[int]:
    """Return the training steps corresponding to tracked states."""
    tracked_steps = list(range(0, num_steps, track_every))
    if not tracked_steps or tracked_steps[-1] != num_steps:
        tracked_steps.append(num_steps)
    return tracked_steps



def compute_initial_conditioning(
    depths: List[int], width: int, seed: int
) -> Dict[int, Dict[str, Any]]:
    """Summarize initial layer-wise and product conditioning at each depth."""
    diagnostics: Dict[int, Dict[str, Any]] = {}
    for depth in depths:
        weights = init_weights(depth, width, seed)
        layer_kappas: List[float] = []
        for W in weights:
            sv = np.linalg.svd(W, compute_uv=False)
            sigma_min = max(float(sv[-1]), 1e-12)
            layer_kappas.append(float(float(sv[0]) / sigma_min))

        log_prod_kappa = float(sum(np.log(layer_kappas)))
        diagnostics[depth] = {
            "per_layer_condition_numbers": layer_kappas,
            "min_per_layer_condition_number": float(min(layer_kappas)),
            "max_per_layer_condition_number": float(max(layer_kappas)),
            "log_product_condition_number": log_prod_kappa,
            "product_condition_number": float(np.exp(log_prod_kappa)),
        }
    return diagnostics



def fit_crossover_power_law(
    depth_results: Dict[int, Dict[str, Any]], depths: List[int]
) -> Dict[str, Any]:
    """Fit crossover_step = a / L^b on valid crossover points only."""
    valid_depths: List[int] = []
    valid_crossovers: List[int] = []
    excluded_depths: List[int] = []

    for depth in depths:
        cs = depth_results[depth]["crossover_step"]
        if cs is not None and cs > 0:
            valid_depths.append(depth)
            valid_crossovers.append(int(cs))
        else:
            excluded_depths.append(depth)

    fit_summary: Dict[str, Any] = {
        "model": "crossover_step = a / L^b fitted in log-log space on crossover_step > 0 points",
        "valid_depths": valid_depths,
        "valid_crossovers": valid_crossovers,
        "excluded_depths": excluded_depths,
        "n_valid": len(valid_depths),
        "a": None,
        "b": None,
        "r_squared": None,
        "fit_points": [],
        "note": (
            "Descriptive fit for a single nested-seed sweep; not independent multi-seed evidence."
        ),
    }

    if len(valid_depths) < 2:
        return fit_summary

    log_L = np.log(np.array(valid_depths, dtype=float))
    log_cs = np.log(np.array(valid_crossovers, dtype=float))
    A_mat = np.vstack([np.ones_like(log_L), -log_L]).T
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, log_cs, rcond=None)
    log_a = float(coeffs[0])
    b = float(coeffs[1])
    a = float(np.exp(log_a))

    predicted = a / np.array(valid_depths, dtype=float) ** b
    ss_res = float(np.sum((log_cs - (log_a - b * log_L)) ** 2))
    ss_tot = float(np.sum((log_cs - np.mean(log_cs)) ** 2))
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    fit_summary.update(
        {
            "a": a,
            "b": b,
            "r_squared": r_squared,
            "fit_points": [
                {
                    "depth": depth,
                    "crossover_step": crossover,
                    "predicted_crossover_step": float(pred),
                }
                for depth, crossover, pred in zip(valid_depths, valid_crossovers, predicted)
            ],
        }
    )
    return fit_summary



def evaluate_hypotheses(
    depth_results: Dict[int, Dict[str, Any]], depths: List[int], fit_summary: Dict[str, Any]
) -> Dict[str, Any]:
    """Compute the legacy test battery and a calibrated decision summary."""
    tests: List[Dict[str, Any]] = []

    crossover_count = sum(
        1 for depth in depths if depth_results[depth]["crossover_step"] is not None
    )
    test1_pass = crossover_count >= len(depths) * 0.5
    tests.append(
        {
            "id": "test1",
            "name": "80%-dominance crossover exists for at least half of the tested depths",
            "passed": bool(test1_pass),
            "details": {
                "crossover_count": int(crossover_count),
                "num_depths": int(len(depths)),
                "threshold_fraction": 0.5,
            },
        }
    )

    valid_crossovers = fit_summary["valid_crossovers"]
    if len(valid_crossovers) >= 2:
        non_increasing = sum(
            1
            for i in range(1, len(valid_crossovers))
            if valid_crossovers[i] <= valid_crossovers[i - 1]
        )
        monotonic_frac = non_increasing / (len(valid_crossovers) - 1)
        test2_pass = monotonic_frac >= 0.5
    else:
        non_increasing = 0
        monotonic_frac = 0.0
        test2_pass = False
    tests.append(
        {
            "id": "test2",
            "name": "Valid crossover steps are non-increasing with depth often enough",
            "passed": bool(test2_pass),
            "details": {
                "valid_depths": list(fit_summary["valid_depths"]),
                "valid_crossovers": list(valid_crossovers),
                "non_increasing_pairs": int(non_increasing),
                "total_pairs": max(int(len(valid_crossovers) - 1), 0),
                "monotonic_fraction": float(monotonic_frac),
            },
        }
    )

    if fit_summary["b"] is not None:
        test3_pass = fit_summary["b"] > 0.5
        test3_details = {
            "b": float(fit_summary["b"]),
            "r_squared": float(fit_summary["r_squared"]),
            "threshold_b": 0.5,
            "n_valid": int(fit_summary["n_valid"]),
        }
    else:
        test3_pass = False
        test3_details = {
            "b": None,
            "r_squared": None,
            "threshold_b": 0.5,
            "n_valid": int(fit_summary["n_valid"]),
        }
    tests.append(
        {
            "id": "test3",
            "name": "Descriptive power-law exponent exceeds 0.5",
            "passed": bool(test3_pass),
            "details": test3_details,
        }
    )

    transient_count = 0
    transient_rows: List[Dict[str, Any]] = []
    for depth in depths:
        diffs = depth_results[depth]["difference_sgd_minus_muon_trajectory"]
        any_better = any(diff > 0 for diff in diffs)
        if any_better:
            transient_count += 1
        transient_rows.append(
            {
                "depth": int(depth),
                "any_muon_better": bool(any_better),
                "best_advantage": float(depth_results[depth]["best_advantage"]),
                "best_advantage_step": int(depth_results[depth]["best_advantage_step"]),
            }
        )
    test4_pass = transient_count >= len(depths) * 0.8
    tests.append(
        {
            "id": "test4",
            "name": "Muon is transiently better than the baseline at most tested depths",
            "passed": bool(test4_pass),
            "details": {
                "transient_count": int(transient_count),
                "num_depths": int(len(depths)),
                "per_depth": transient_rows,
            },
        }
    )

    best_advs = [float(depth_results[depth]["best_advantage"]) for depth in depths]
    if len(best_advs) >= 2:
        growing_pairs = sum(
            1 for i in range(1, len(best_advs)) if best_advs[i] > best_advs[i - 1]
        )
        test5_pass = growing_pairs >= len(best_advs) - 2
    else:
        growing_pairs = 0
        test5_pass = False
    tests.append(
        {
            "id": "test5",
            "name": "Peak Muon advantage generally grows with depth",
            "passed": bool(test5_pass),
            "details": {
                "best_advantages": best_advs,
                "growing_pairs": int(growing_pairs),
                "total_pairs": max(int(len(best_advs) - 1), 0),
            },
        }
    )

    final_muon_wins = sum(1 for depth in depths if depth_results[depth]["final_win"] == "muon")
    legacy_core_pass = bool(test1_pass and test2_pass)

    if fit_summary["b"] is None:
        fit_fragment = "no crossover-depth fit was available"
    else:
        fit_fragment = (
            f"the descriptive fit on n_valid={fit_summary['n_valid']} points gives "
            f"b={fit_summary['b']:.3f} with R^2={fit_summary['r_squared']:.4f}"
        )

    statement = (
        f"80%-dominance crossovers occur at {crossover_count}/{len(depths)} depths, "
        f"Muon finishes with lower final log-product-kappa at {final_muon_wins}/{len(depths)} depths, and {fit_fragment}. "
        "This is mixed exploratory evidence from one nested-seed toy sweep, not strong support for O(n/L^2)."
    )

    return {
        "tests": tests,
        "decision_summary": {
            "legacy_core_pass": legacy_core_pass,
            "final_muon_wins": int(final_muon_wins),
            "num_depths": int(len(depths)),
            "label": "mixed exploratory evidence",
            "statement": statement,
            "single_seed_limitation": True,
            "nested_depth_seed_limitation": True,
        },
    }



def build_summary_rows(
    depths: List[int],
    depth_results: Dict[int, Dict[str, Any]],
    initial_conditioning: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Create per-depth summary rows for notebooks and lightweight reporting."""
    rows = []
    for depth in depths:
        row = {
            "depth": int(depth),
            "initial_log_product_kappa": float(
                initial_conditioning[depth]["log_product_condition_number"]
            ),
            "initial_min_layer_kappa": float(
                initial_conditioning[depth]["min_per_layer_condition_number"]
            ),
            "initial_max_layer_kappa": float(
                initial_conditioning[depth]["max_per_layer_condition_number"]
            ),
        }
        row.update(depth_results[depth])
        rows.append(row)
    return rows


# =============================================================================
# REPORTING UTILITIES
# =============================================================================


def format_crossover_step(crossover_step: Optional[int]) -> str:
    return "never" if crossover_step is None else str(int(crossover_step))



def print_section(title: str) -> None:
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)



def print_report(results: Dict[str, Any]) -> None:
    """Print a calibrated console report from structured results."""
    config = results["config"]
    data_summary = results["data_summary"]
    fit_summary = results["fit_summary"]
    decision_summary = results["decision_summary"]
    depth_results = results["depth_results"]
    depths = config["depths"]

    print_section(results["experiment_name"])
    print("Scope:")
    print(f"  - {config['study_scope']}")
    print(f"  - Baseline ('sgd' key): {config['baseline_description']}")
    print(f"  - Muon: {config['muon_description']}")
    print(f"  - Crossover metric: {config['crossover_definition']}")
    print(f"  - Depth note: {config['depth_initialization_note']}")
    print()
    print("Configuration:")
    print(
        f"  width={config['width']}, depths={depths}, steps={config['num_steps']}, "
        f"lr_sgd={config['lr_sgd']}, lr_muon={config['lr_muon']}, ns_iters={config['ns_iters']}"
    )
    print(
        f"  target_condition_number={data_summary['target_condition_number']:.1f}, "
        f"X_shape={data_summary['x_shape']}, Y_shape={data_summary['y_shape']}"
    )
    print(
        f"  loss definition: {config['reported_loss_definition']}"
    )
    print(f"  runtime_seconds={results['runtime_seconds']:.3f}")

    print_section("INITIAL CONDITIONING SUMMARY")
    print(
        f"{'Depth':>6} {'Init log_k_prod':>16} {'Min layer kappa':>18} {'Max layer kappa':>18}"
    )
    print("-" * 66)
    for depth in depths:
        init_row = results["initial_conditioning"][depth]
        print(
            f"{depth:>6} {init_row['log_product_condition_number']:>16.2f} "
            f"{init_row['min_per_layer_condition_number']:>18.2f} "
            f"{init_row['max_per_layer_condition_number']:>18.2f}"
        )

    print_section("LOG PRODUCT KAPPA TRAJECTORIES (sampled every 50 steps)")
    sample_steps = list(range(0, config["num_steps"] + 1, 50))
    if config["num_steps"] not in sample_steps:
        sample_steps.append(config["num_steps"])
    for depth in depths:
        r = depth_results[depth]
        print(f"\nDepth {depth}:")
        print(f"{'Step':>6} {'log_k_sgd':>12} {'log_k_muon':>12} {'Muon<sgd?':>10}")
        print("-" * 46)
        for step in sample_steps:
            if step < len(r["kappa_sgd"]):
                ks = r["kappa_sgd"][step]
                km = r["kappa_muon"][step]
                better = "YES" if km < ks else "no"
                print(f"{step:>6} {ks:>12.2f} {km:>12.2f} {better:>10}")

    print_section("PER-DEPTH SUMMARY")
    print(
        f"{'Depth':>6} {'Crossover':>12} {'Final log_k_sgd':>16} {'Final log_k_muon':>18} "
        f"{'Diff (sgd-muon)':>18} {'Final win':>12}"
    )
    print("-" * 92)
    for depth in depths:
        r = depth_results[depth]
        print(
            f"{depth:>6} {format_crossover_step(r['crossover_step']):>12} "
            f"{r['final_kappa_sgd']:>16.2f} {r['final_kappa_muon']:>18.2f} "
            f"{r['final_difference_sgd_minus_muon']:>18.2f} {r['final_win']:>12}"
        )

    print_section("CROSSOVER STEP SCALING ANALYSIS")
    print(
        f"Valid fit points (crossover_step > 0): n_valid={fit_summary['n_valid']} / {len(depths)}"
    )
    if fit_summary["fit_points"]:
        for row in fit_summary["fit_points"]:
            print(
                f"  L={row['depth']:>2}: actual={row['crossover_step']:>4}, "
                f"predicted={row['predicted_crossover_step']:.1f}"
            )
        print(
            f"\nDescriptive fit: crossover_step = {fit_summary['a']:.1f} / L^{fit_summary['b']:.2f}"
        )
        print(f"  exponent b = {fit_summary['b']:.3f}")
        print(f"  R^2 = {fit_summary['r_squared']:.4f}")
        print(f"  note: {fit_summary['note']}")
    else:
        print("  Not enough valid crossover points for a fit.")

    print_section("HYPOTHESIS TESTS")
    for test in results["hypothesis_tests"]:
        print(f"{test['id'].upper()}: {test['name']}")
        print(f"  status: {'PASS' if test['passed'] else 'FAIL'}")
        for key, value in test["details"].items():
            print(f"  {key}: {value}")
        print()

    print_section("CALIBRATED INTERPRETATION")
    print(decision_summary["statement"])
    print(
        f"Legacy core-pass flag (tests 1 & 2 only): {decision_summary['legacy_core_pass']}"
    )


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================


def run_experiment(
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the depth sweep and return structured results for reuse."""
    resolved = resolve_config(config)
    problem = build_problem(resolved)
    tracked_steps = make_tracked_steps(resolved["num_steps"], resolved["track_every"])
    initial_conditioning = compute_initial_conditioning(
        resolved["depths"], resolved["width"], resolved["seed"]
    )

    if verbose:
        print(f"Running {resolved['experiment_name']} ...")

    start_time = time.perf_counter()
    depth_results: Dict[int, Dict[str, Any]] = {}

    for depth in resolved["depths"]:
        if verbose:
            print(f"  depth={depth:>2} ...", end=" ", flush=True)

        weights_init = init_weights(depth, resolved["width"], seed=resolved["seed"])

        _, kappa_sgd, loss_sgd = train_sgd_tracked(
            weights_init,
            problem["X"],
            problem["Y"],
            resolved["num_steps"],
            resolved["lr_sgd"],
            resolved["track_every"],
        )
        _, kappa_muon, loss_muon = train_muon_tracked(
            weights_init,
            problem["X"],
            problem["Y"],
            resolved["num_steps"],
            resolved["lr_muon"],
            resolved["ns_iters"],
            resolved["track_every"],
        )

        if len(kappa_sgd) != len(tracked_steps) or len(kappa_muon) != len(tracked_steps):
            raise RuntimeError("Tracked trajectory lengths do not match the expected step axis.")

        crossover_index = find_80pct_dominance_crossover(
            kappa_sgd, kappa_muon, fraction=resolved["crossover_fraction"]
        )
        crossover_step = (
            None if crossover_index is None else int(tracked_steps[crossover_index])
        )

        diff_traj = [float(ks - km) for ks, km in zip(kappa_sgd, kappa_muon)]
        best_advantage_index = int(np.argmax(diff_traj))
        best_advantage_step = int(tracked_steps[best_advantage_index])
        best_advantage = float(diff_traj[best_advantage_index])
        final_diff = float(kappa_sgd[-1] - kappa_muon[-1])
        final_loss_diff = float(loss_sgd[-1] - loss_muon[-1])

        if final_diff > 0:
            final_win = "muon"
        elif final_diff < 0:
            final_win = "sgd"
        else:
            final_win = "tie"

        depth_results[depth] = {
            "depth": int(depth),
            "trajectory_steps": list(tracked_steps),
            "kappa_sgd": [float(v) for v in kappa_sgd],
            "kappa_muon": [float(v) for v in kappa_muon],
            "loss_sgd": [float(v) for v in loss_sgd],
            "loss_muon": [float(v) for v in loss_muon],
            "difference_sgd_minus_muon_trajectory": diff_traj,
            "crossover_step": None if crossover_step is None else int(crossover_step),
            "crossover_definition": resolved["crossover_definition"],
            "final_kappa_sgd": float(kappa_sgd[-1]),
            "final_kappa_muon": float(kappa_muon[-1]),
            "final_difference_sgd_minus_muon": final_diff,
            "final_loss_sgd": float(loss_sgd[-1]),
            "final_loss_muon": float(loss_muon[-1]),
            "final_loss_difference_sgd_minus_muon": final_loss_diff,
            "final_loss_ratio_muon_over_sgd": (
                float(loss_muon[-1] / loss_sgd[-1]) if loss_sgd[-1] > 0 else None
            ),
            "final_win": final_win,
            "any_muon_better": bool(any(diff > 0 for diff in diff_traj)),
            "best_advantage": best_advantage,
            "best_advantage_step": best_advantage_step,
            "trajectory_length": int(len(kappa_sgd)),
        }

        if verbose:
            print(
                f"crossover={format_crossover_step(crossover_step):>5}, "
                f"final diff (sgd-muon)={final_diff:+.2f}, final win={final_win}"
            )

    runtime_seconds = float(time.perf_counter() - start_time)

    fit_summary = fit_crossover_power_law(depth_results, resolved["depths"])
    hypothesis_bundle = evaluate_hypotheses(
        depth_results, resolved["depths"], fit_summary
    )

    results = {
        "experiment_name": resolved["experiment_name"],
        "config": resolved,
        "data_summary": {
            "target_condition_number": problem["target_condition_number"],
            "target_singular_values": [float(v) for v in problem["sigma"]],
            "x_shape": problem["x_shape"],
            "y_shape": problem["y_shape"],
            "x_fro_norm": problem["x_fro_norm"],
            "y_fro_norm": problem["y_fro_norm"],
        },
        "tracked_steps": tracked_steps,
        "initial_conditioning": initial_conditioning,
        "depth_results": depth_results,
        "summary_rows": build_summary_rows(
            resolved["depths"], depth_results, initial_conditioning
        ),
        "fit_summary": fit_summary,
        "hypothesis_tests": hypothesis_bundle["tests"],
        "decision_summary": hypothesis_bundle["decision_summary"],
        "runtime_seconds": runtime_seconds,
    }

    if verbose:
        print_report(results)

    return results



def main() -> Dict[str, Any]:
    """CLI entrypoint."""
    return run_experiment(verbose=True)


if __name__ == "__main__":
    main()
