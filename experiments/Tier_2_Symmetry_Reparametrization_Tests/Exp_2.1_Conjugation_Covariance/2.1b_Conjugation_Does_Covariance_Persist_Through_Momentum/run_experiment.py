#!/usr/bin/env python3
"""
Exp 2.1b: Does Conjugation Covariance Persist Through Momentum?
================================================================

Finite-scope numerical study of Muon+momentum under bilateral orthogonal
conjugation. The default configuration preserves the original first-pass setup:
50 Muon-style updates, momentum beta=0.9, 20 seeds, and matrix sizes 4x4 and 8x8.

This file compares three single-matrix scenarios under W -> R W S^T:
  A. Random conjugated gradients (positive control)
  B. Fixed-frame single-layer linear MSE gradients (negative control)
  C. Equivariant Frobenius-loss gradients (positive control)

Primary metric:
  final relative covariance error
    ||W'_T - R W_T S^T||_F / ||W_T||_F

Secondary diagnostics:
  sampled single-trial trajectories over the 50 updates.

Interpretation is intentionally calibrated:
  - Positive-control success here is sampled numerical evidence at the tested
    float64 configuration, not a universal proof claim by itself.
  - The failure in case B is attributed to the fixed-frame data-driven gradient
    map not being conjugation-equivariant under this setup; this file does not
    isolate momentum as the sole causal variable.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class ExperimentConfig:
    n_steps: int = 50
    lr: float = 0.02
    momentum_beta: float = 0.9
    ns_iters: int = 5
    n_trials: int = 20
    base_seed: int = 42
    matrix_sizes: Tuple[Tuple[int, int], ...] = ((4, 4), (8, 8))
    n_samples: int = 50
    trial_seed_stride: int = 17
    trajectory_size: Tuple[int, int] = (8, 8)
    trajectory_seed: int = 42
    random_trajectory_seed_offset: int = 777
    trajectory_checkpoints: Tuple[int, ...] = (0, 1, 2, 5, 10, 20, 30, 40, 49)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["matrix_sizes"] = [list(size) for size in self.matrix_sizes]
        data["trajectory_size"] = list(self.trajectory_size)
        data["trajectory_checkpoints"] = list(self.trajectory_checkpoints)
        return data


DEFAULT_CONFIG = ExperimentConfig()


EXPERIMENT_SPECS = {
    "A_random_conjugated_gradients": {
        "label": "A. Random conjugated gradients",
        "short_label": "A (random G)",
        "description": (
            "Positive control: both paths see the same random gradient sequence, "
            "with path B using G'_t = R G_t S^T."
        ),
        "gradient_map_equivariant": True,
    },
    "B_fixed_frame_data_driven": {
        "label": "B. Fixed-frame data-driven gradients",
        "short_label": "B (data-driven)",
        "description": (
            "Negative control: a single linear layer is trained against fixed X and Y, "
            "so the induced gradient map does not commute with W -> R W S^T."
        ),
        "gradient_map_equivariant": False,
    },
    "C_equivariant_frobenius_loss": {
        "label": "C. Equivariant Frobenius-loss gradients",
        "short_label": "C (equiv loss)",
        "description": (
            "Positive control: L(W) = 0.5 ||W||_F^2 gives G(W) = W, which is conjugation-equivariant."
        ),
        "gradient_map_equivariant": True,
    },
}

EXPERIMENT_ORDER = (
    "A_random_conjugated_gradients",
    "B_fixed_frame_data_driven",
    "C_equivariant_frobenius_loss",
)

HEURISTIC_CHECK_ORDER = ("H1", "H2", "H3", "H4")


def validate_config(config: ExperimentConfig) -> None:
    if config.n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if config.n_trials <= 0:
        raise ValueError("n_trials must be positive")
    if config.ns_iters <= 0:
        raise ValueError("ns_iters must be positive")
    if not config.matrix_sizes:
        raise ValueError("matrix_sizes must be non-empty")
    if max(config.trajectory_checkpoints) >= config.n_steps:
        raise ValueError("trajectory checkpoints must be < n_steps")


def size_key(m: int, n: int) -> str:
    return f"{m}x{n}"


def newton_schulz(M: np.ndarray, n_iters: int) -> np.ndarray:
    norm = np.linalg.norm(M, ord="fro")
    if norm < 1e-15:
        return M.copy()
    X = M / norm
    for _ in range(n_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


def random_orthogonal(n: int, rng: np.random.RandomState) -> np.ndarray:
    A = rng.randn(n, n)
    Q, R = np.linalg.qr(A)
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    D = np.diag(signs)
    return Q @ D


def relative_covariance_error(
    W_a: np.ndarray,
    W_b: np.ndarray,
    R: np.ndarray,
    S: np.ndarray,
) -> float:
    expected = R @ W_a @ S.T
    denom = max(np.linalg.norm(W_a), 1e-30)
    return float(np.linalg.norm(W_b - expected) / denom)


def summarize_errors(errors: List[float]) -> Dict[str, float]:
    values = np.asarray(errors, dtype=float)
    n = len(values)
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 1 else 0.0
    mean = float(np.mean(values))
    return {
        "n": int(n),
        "mean": mean,
        "std": std,
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "ci95_low": mean - 1.96 * sem,
        "ci95_high": mean + 1.96 * sem,
    }


def sample_trace(trace: List[float], checkpoints: Tuple[int, ...], value_name: str) -> List[Dict[str, float]]:
    return [{"step": int(step), value_name: float(trace[step])} for step in checkpoints]


def linear_mse_grad(W: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    n_samples = X.shape[1]
    pred = W @ X
    return (pred - Y) @ X.T / n_samples


def linear_mse_loss(W: np.ndarray, X: np.ndarray, Y: np.ndarray) -> float:
    n_samples = X.shape[1]
    residual = W @ X - Y
    return 0.5 * float(np.sum(residual**2) / n_samples)


def frobenius_half_squared_loss(W: np.ndarray) -> float:
    return 0.5 * float(np.sum(W**2))


def run_random_gradient_test(
    m: int,
    n: int,
    rng: np.random.RandomState,
    *,
    config: ExperimentConfig = DEFAULT_CONFIG,
    return_trace: bool = False,
) -> Dict[str, Any]:
    """
    Positive control: path B receives the conjugated gradient sequence G'_t = R G_t S^T.
    When the update rule is implemented consistently, covariance should remain at
    machine precision in this float64 experiment.
    """
    W0 = rng.randn(m, n)
    R = random_orthogonal(m, rng)
    S = random_orthogonal(n, rng)
    gradients = [rng.randn(m, n) for _ in range(config.n_steps)]

    W_a = W0.copy()
    W_b = R @ W0 @ S.T
    mom_a = np.zeros((m, n))
    mom_b = np.zeros((m, n))

    error_trace: List[float] = []
    for step in range(config.n_steps):
        G = gradients[step]
        G_conj = R @ G @ S.T
        mom_a = config.momentum_beta * mom_a + (1.0 - config.momentum_beta) * G
        mom_b = config.momentum_beta * mom_b + (1.0 - config.momentum_beta) * G_conj
        W_a = W_a - config.lr * newton_schulz(mom_a, config.ns_iters)
        W_b = W_b - config.lr * newton_schulz(mom_b, config.ns_iters)
        if return_trace:
            error_trace.append(relative_covariance_error(W_a, W_b, R, S))

    result: Dict[str, Any] = {
        "final_error": relative_covariance_error(W_a, W_b, R, S),
        "metadata": {
            "initial_weight_norm": float(np.linalg.norm(W0)),
            "det_R": float(np.linalg.det(R)),
            "det_S": float(np.linalg.det(S)),
            "orthogonality_error_R": float(np.linalg.norm(R.T @ R - np.eye(m))),
            "orthogonality_error_S": float(np.linalg.norm(S.T @ S - np.eye(n))),
        },
    }
    if return_trace:
        result.update(
            {
                "step_indices": list(range(config.n_steps)),
                "error_trace": [float(x) for x in error_trace],
                "sampled_errors": sample_trace(error_trace, config.trajectory_checkpoints, "error"),
            }
        )
    return result


def run_data_driven_test(
    m: int,
    n: int,
    rng: np.random.RandomState,
    *,
    config: ExperimentConfig = DEFAULT_CONFIG,
    return_trace: bool = False,
) -> Dict[str, Any]:
    """
    Negative control: a single linear layer y = W x is trained against fixed-frame
    Gaussian X and Y. Under this setup, the induced gradient map does not satisfy
    G(R W S^T) = R G(W) S^T, so covariance is expected to break.
    """
    W0 = rng.randn(m, n)
    R = random_orthogonal(m, rng)
    S = random_orthogonal(n, rng)
    X = rng.randn(n, config.n_samples) * 0.3
    Y = rng.randn(m, config.n_samples) * 0.3

    W_a = W0.copy()
    W_b = R @ W0 @ S.T
    mom_a = np.zeros((m, n))
    mom_b = np.zeros((m, n))

    initial_loss_a = linear_mse_loss(W_a, X, Y)
    initial_loss_b = linear_mse_loss(W_b, X, Y)

    error_trace: List[float] = []
    loss_trace_a: List[float] = []
    loss_trace_b: List[float] = []

    for step in range(config.n_steps):
        G_a = linear_mse_grad(W_a, X, Y)
        G_b = linear_mse_grad(W_b, X, Y)
        mom_a = config.momentum_beta * mom_a + (1.0 - config.momentum_beta) * G_a
        mom_b = config.momentum_beta * mom_b + (1.0 - config.momentum_beta) * G_b
        W_a = W_a - config.lr * newton_schulz(mom_a, config.ns_iters)
        W_b = W_b - config.lr * newton_schulz(mom_b, config.ns_iters)
        if return_trace:
            error_trace.append(relative_covariance_error(W_a, W_b, R, S))
            loss_trace_a.append(linear_mse_loss(W_a, X, Y))
            loss_trace_b.append(linear_mse_loss(W_b, X, Y))

    final_loss_a = linear_mse_loss(W_a, X, Y)
    final_loss_b = linear_mse_loss(W_b, X, Y)

    result = {
        "final_error": relative_covariance_error(W_a, W_b, R, S),
        "metadata": {
            "initial_weight_norm": float(np.linalg.norm(W0)),
            "input_norm": float(np.linalg.norm(X)),
            "target_norm": float(np.linalg.norm(Y)),
            "initial_loss_path_a": initial_loss_a,
            "initial_loss_path_b": initial_loss_b,
            "initial_loss_gap": abs(initial_loss_b - initial_loss_a),
            "final_loss_path_a": final_loss_a,
            "final_loss_path_b": final_loss_b,
            "final_loss_gap": abs(final_loss_b - final_loss_a),
        },
    }
    if return_trace:
        result.update(
            {
                "step_indices": list(range(config.n_steps)),
                "error_trace": [float(x) for x in error_trace],
                "sampled_errors": sample_trace(error_trace, config.trajectory_checkpoints, "error"),
                "loss_trace_a": [float(x) for x in loss_trace_a],
                "loss_trace_b": [float(x) for x in loss_trace_b],
            }
        )
    return result


def run_equivariant_loss_test(
    m: int,
    n: int,
    rng: np.random.RandomState,
    *,
    config: ExperimentConfig = DEFAULT_CONFIG,
    return_trace: bool = False,
) -> Dict[str, Any]:
    """
    Positive control: L(W) = 0.5 ||W||_F^2 gives G(W) = W, so the gradient map is
    conjugation-equivariant and covariance should persist at machine precision in
    this float64 experiment.
    """
    W0 = rng.randn(m, n)
    R = random_orthogonal(m, rng)
    S = random_orthogonal(n, rng)

    W_a = W0.copy()
    W_b = R @ W0 @ S.T
    mom_a = np.zeros((m, n))
    mom_b = np.zeros((m, n))

    error_trace: List[float] = []
    loss_trace_a: List[float] = []
    loss_trace_b: List[float] = []

    for step in range(config.n_steps):
        G_a = W_a
        G_b = W_b
        mom_a = config.momentum_beta * mom_a + (1.0 - config.momentum_beta) * G_a
        mom_b = config.momentum_beta * mom_b + (1.0 - config.momentum_beta) * G_b
        W_a = W_a - config.lr * newton_schulz(mom_a, config.ns_iters)
        W_b = W_b - config.lr * newton_schulz(mom_b, config.ns_iters)
        if return_trace:
            error_trace.append(relative_covariance_error(W_a, W_b, R, S))
            loss_trace_a.append(frobenius_half_squared_loss(W_a))
            loss_trace_b.append(frobenius_half_squared_loss(W_b))

    result = {
        "final_error": relative_covariance_error(W_a, W_b, R, S),
        "metadata": {
            "initial_weight_norm": float(np.linalg.norm(W0)),
            "initial_loss_path_a": frobenius_half_squared_loss(W0),
            "initial_loss_path_b": frobenius_half_squared_loss(R @ W0 @ S.T),
            "initial_singular_values": [float(x) for x in np.linalg.svd(W0, compute_uv=False)],
            "final_loss_path_a": frobenius_half_squared_loss(W_a),
            "final_loss_path_b": frobenius_half_squared_loss(W_b),
        },
    }
    if return_trace:
        result.update(
            {
                "step_indices": list(range(config.n_steps)),
                "error_trace": [float(x) for x in error_trace],
                "sampled_errors": sample_trace(error_trace, config.trajectory_checkpoints, "error"),
                "loss_trace_a": [float(x) for x in loss_trace_a],
                "loss_trace_b": [float(x) for x in loss_trace_b],
            }
        )
    return result


def _trial_seed(config: ExperimentConfig, trial_index: int) -> int:
    return config.base_seed + trial_index * config.trial_seed_stride


def _run_suite_for_experiment(
    experiment_key: str,
    runner,
    config: ExperimentConfig,
) -> Dict[str, Any]:
    spec = EXPERIMENT_SPECS[experiment_key]
    size_results: Dict[str, Any] = {}

    for m, n in config.matrix_sizes:
        errors: List[float] = []
        trial_seeds: List[int] = []
        for trial in range(config.n_trials):
            seed = _trial_seed(config, trial)
            trial_seeds.append(seed)
            rng = np.random.RandomState(seed)
            trial_result = runner(m, n, rng, config=config, return_trace=False)
            errors.append(float(trial_result["final_error"]))

        size_results[size_key(m, n)] = {
            "shape": [m, n],
            "trial_seeds": trial_seeds,
            "raw_trial_errors": [float(x) for x in errors],
            "summary": summarize_errors(errors),
        }

    return {
        "label": spec["label"],
        "short_label": spec["short_label"],
        "description": spec["description"],
        "gradient_map_equivariant": spec["gradient_map_equivariant"],
        "sizes": size_results,
    }


def collect_single_trial_diagnostics(config: ExperimentConfig) -> Dict[str, Any]:
    m, n = config.trajectory_size
    data_trace = run_data_driven_test(
        m,
        n,
        np.random.RandomState(config.trajectory_seed),
        config=config,
        return_trace=True,
    )
    equiv_trace = run_equivariant_loss_test(
        m,
        n,
        np.random.RandomState(config.trajectory_seed),
        config=config,
        return_trace=True,
    )
    random_trace = run_random_gradient_test(
        m,
        n,
        np.random.RandomState(config.trajectory_seed + config.random_trajectory_seed_offset),
        config=config,
        return_trace=True,
    )

    checkpoints = list(config.trajectory_checkpoints)
    paired_rows = []
    for step in checkpoints:
        paired_rows.append(
            {
                "step": int(step),
                "data_driven_error": float(data_trace["error_trace"][step]),
                "equivariant_loss_error": float(equiv_trace["error_trace"][step]),
            }
        )

    random_rows = []
    for step in checkpoints:
        random_rows.append({"step": int(step), "random_gradient_error": float(random_trace["error_trace"][step])})

    return {
        "data_vs_equivariant": {
            "size": [m, n],
            "seed": int(config.trajectory_seed),
            "step_indices": data_trace["step_indices"],
            "checkpoints": checkpoints,
            "data_driven_error_trace": data_trace["error_trace"],
            "equivariant_loss_error_trace": equiv_trace["error_trace"],
            "data_driven_loss_trace_path_a": data_trace["loss_trace_a"],
            "data_driven_loss_trace_path_b": data_trace["loss_trace_b"],
            "sampled_rows": paired_rows,
            "metadata": {
                "initial_loss_gap": data_trace["metadata"]["initial_loss_gap"],
                "final_loss_gap": data_trace["metadata"]["final_loss_gap"],
                "initial_equivariant_loss_path_a": equiv_trace["metadata"]["initial_loss_path_a"],
                "initial_equivariant_loss_path_b": equiv_trace["metadata"]["initial_loss_path_b"],
            },
        },
        "random_gradient": {
            "size": [m, n],
            "seed": int(config.trajectory_seed + config.random_trajectory_seed_offset),
            "step_indices": random_trace["step_indices"],
            "checkpoints": checkpoints,
            "error_trace": random_trace["error_trace"],
            "sampled_rows": random_rows,
            "metadata": random_trace["metadata"],
        },
    }


def compute_heuristic_checks(experiment_results: Dict[str, Any], config: ExperimentConfig) -> Dict[str, Any]:
    def collect(experiment_key: str) -> np.ndarray:
        values: List[float] = []
        for size_result in experiment_results[experiment_key]["sizes"].values():
            values.extend(size_result["raw_trial_errors"])
        return np.asarray(values, dtype=float)

    all_A = collect("A_random_conjugated_gradients")
    all_B = collect("B_fixed_frame_data_driven")
    all_C = collect("C_equivariant_frobenius_loss")

    mean_A = float(np.mean(all_A))
    mean_B = float(np.mean(all_B))
    mean_C = float(np.mean(all_C))
    ratio_B_over_A = mean_B / max(mean_A, 1e-30)
    ratio_B_over_C = mean_B / max(mean_C, 1e-30)

    checks = {
        "H1": {
            "statement": (
                "Positive control A stays at machine precision at the tested final endpoints"
            ),
            "criterion": f"max(final error over all sizes/trials) < 1e-12 after {config.n_steps} steps",
            "observed": {"max_error": float(np.max(all_A))},
            "pass": bool(np.max(all_A) < 1e-12),
        },
        "H2": {
            "statement": "Negative control B shows macroscopic covariance failure",
            "criterion": "mean(final error over all sizes/trials) > 1e-2",
            "observed": {"mean_error": mean_B},
            "pass": bool(mean_B > 1e-2),
        },
        "H3": {
            "statement": (
                "Positive control C stays at machine precision at the tested final endpoints"
            ),
            "criterion": f"max(final error over all sizes/trials) < 1e-12 after {config.n_steps} steps",
            "observed": {"max_error": float(np.max(all_C))},
            "pass": bool(np.max(all_C) < 1e-12),
        },
        "H4": {
            "statement": "Negative control B is orders of magnitude worse than the positive controls",
            "criterion": "heuristic only: mean(B) / mean(A) > 1e6; mean(B) / mean(C) is reported descriptively",
            "observed": {
                "ratio_B_over_A": float(ratio_B_over_A),
                "ratio_B_over_C": float(ratio_B_over_C),
            },
            "pass": bool(ratio_B_over_A > 1e6),
        },
        "aggregate_means": {
            "A_random_conjugated_gradients": mean_A,
            "B_fixed_frame_data_driven": mean_B,
            "C_equivariant_frobenius_loss": mean_C,
        },
        "aggregate_maxima": {
            "A_random_conjugated_gradients": float(np.max(all_A)),
            "B_fixed_frame_data_driven": float(np.max(all_B)),
            "C_equivariant_frobenius_loss": float(np.max(all_C)),
        },
    }
    checks["checks_total"] = 4
    checks["checks_passed"] = int(sum(bool(checks[key]["pass"]) for key in ("H1", "H2", "H3", "H4")))
    return checks


def build_comparison_rows(experiment_results: Dict[str, Any], config: ExperimentConfig) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for m, n in config.matrix_sizes:
        key = size_key(m, n)
        rows.append(
            {
                "size": key,
                "A_mean_error": experiment_results["A_random_conjugated_gradients"]["sizes"][key]["summary"]["mean"],
                "B_mean_error": experiment_results["B_fixed_frame_data_driven"]["sizes"][key]["summary"]["mean"],
                "C_mean_error": experiment_results["C_equivariant_frobenius_loss"]["sizes"][key]["summary"]["mean"],
            }
        )
    return rows


def run_experiment(config: ExperimentConfig = DEFAULT_CONFIG) -> Dict[str, Any]:
    validate_config(config)
    start = time.perf_counter()

    runners = {
        "A_random_conjugated_gradients": run_random_gradient_test,
        "B_fixed_frame_data_driven": run_data_driven_test,
        "C_equivariant_frobenius_loss": run_equivariant_loss_test,
    }
    experiment_results = {
        experiment_key: _run_suite_for_experiment(experiment_key, runners[experiment_key], config)
        for experiment_key in EXPERIMENT_ORDER
    }

    diagnostics = collect_single_trial_diagnostics(config)
    heuristic_checks = compute_heuristic_checks(experiment_results, config)
    comparison_rows = build_comparison_rows(experiment_results, config)
    runtime_seconds = time.perf_counter() - start

    return {
        "title": "Exp 2.1b: Does Conjugation Covariance Persist Through Momentum?",
        "scope_note": (
            "Finite-scope float64 study at the default configuration only; the negative-control "
            "failure in case B is attributed here to the non-equivariant fixed-frame gradient map, "
            "not to momentum in isolation."
        ),
        "config": config.to_dict(),
        "experiment_results": experiment_results,
        "comparison_rows": comparison_rows,
        "single_trial_diagnostics": diagnostics,
        "heuristic_checks": heuristic_checks,
        "runtime_seconds": float(runtime_seconds),
    }


def _print_experiment_block(experiment_key: str, results: Dict[str, Any], config: Dict[str, Any]) -> None:
    block = results["experiment_results"][experiment_key]
    print(f"\n\n{'=' * 90}")
    print(block["label"])
    print(f"{'=' * 90}")
    print(block["description"])

    for size_label, size_result in block["sizes"].items():
        summary = size_result["summary"]
        print(f"\n  Size {size_label} ({config['n_steps']} steps, {config['n_trials']} trials):")
        print(f"    Mean rel error: {summary['mean']:.2e}")
        print(f"    Std rel error:  {summary['std']:.2e}")
        print(f"    Median error:   {summary['median']:.2e}")
        print(f"    95% CI mean:    [{summary['ci95_low']:.2e}, {summary['ci95_high']:.2e}]")
        print(f"    Max rel error:  {summary['max']:.2e}")
        print(f"    Min rel error:  {summary['min']:.2e}")


def print_cli_report(results: Dict[str, Any]) -> None:
    config = results["config"]
    checks = results["heuristic_checks"]

    print("=" * 90)
    print(results["title"].upper())
    print("=" * 90)
    print("Finite-scope numerical report of Muon+momentum covariance under W -> R W S^T.")
    print(results["scope_note"])
    print(f"Steps: {config['n_steps']}, Momentum beta: {config['momentum_beta']}, LR: {config['lr']}")
    print(f"Trials: {config['n_trials']} per size per sub-experiment")
    print(f"Sizes: {config['matrix_sizes']}")
    print(f"Newton-Schulz iterations: {config['ns_iters']}")
    print()
    print("SUB-EXPERIMENTS:")
    print("  A: Random conjugated gradients -- positive control")
    print("  B: Fixed-frame data-driven gradients -- negative control")
    print("  C: Equivariant Frobenius-loss gradients -- positive control")

    for experiment_key in EXPERIMENT_ORDER:
        _print_experiment_block(experiment_key, results, config)

    print(f"\n\n{'=' * 90}")
    print(f"COMPARISON TABLE: Mean Relative Error After {config['n_steps']} Steps")
    print(f"{'=' * 90}")
    print(f"\n{'Size':>6}  {'A (random G)':>14}  {'B (data-driven)':>16}  {'C (equiv loss)':>15}")
    print("-" * 59)
    for row in results["comparison_rows"]:
        print(
            f"{row['size']:>6}  {row['A_mean_error']:>14.2e}  "
            f"{row['B_mean_error']:>16.2e}  {row['C_mean_error']:>15.2e}"
        )

    print(f"\n\n{'=' * 90}")
    trajectory = results["single_trial_diagnostics"]["data_vs_equivariant"]
    size = trajectory["size"]
    print(f"SAMPLED STEP-BY-STEP DRIFT ANALYSIS (single trial, {size[0]}x{size[1]})")
    print(f"{'=' * 90}")
    print(f"Seed: {trajectory['seed']}")
    print(f"\n{'Step':>6}  {'Data-driven err':>16}  {'Equiv loss err':>15}")
    print("-" * 44)
    for row in trajectory["sampled_rows"]:
        print(
            f"{row['step']:>6}  {row['data_driven_error']:>16.2e}  "
            f"{row['equivariant_loss_error']:>15.2e}"
        )

    random_traj = results["single_trial_diagnostics"]["random_gradient"]
    print(f"\nRandom-gradient sampled trajectory ({size[0]}x{size[1]}, seed={random_traj['seed']}):")
    print(f"\n{'Step':>6}  {'Rel error':>12}")
    print("-" * 22)
    for row in random_traj["sampled_rows"]:
        print(f"{row['step']:>6}  {row['random_gradient_error']:>12.2e}")

    print(f"\n\n{'=' * 90}")
    print("HEURISTIC THRESHOLD CHECKS (NOT FORMAL HYPOTHESIS TESTS)")
    print(f"{'=' * 90}")
    for key in HEURISTIC_CHECK_ORDER:
        item = checks[key]
        print(f"\n{key}: {item['statement']}")
        print(f"    Criterion: {item['criterion']}")
        observed_parts = ", ".join(f"{name}={value:.2e}" for name, value in item["observed"].items())
        print(f"    Observed:  {observed_parts}")
        print(f"    --> {'PASS' if item['pass'] else 'FAIL'}")

    total_pass = checks["checks_passed"]
    total_checks = checks["checks_total"]
    print(f"\n\n{'=' * 90}")
    print(f"FINAL VERDICT: {results['title']} ({config['n_steps']} steps)")
    print(f"{'=' * 90}")
    print(
        f"\n  A mean error: {checks['aggregate_means']['A_random_conjugated_gradients']:.2e}"
        f" | max: {checks['aggregate_maxima']['A_random_conjugated_gradients']:.2e}"
    )
    print(
        f"  B mean error: {checks['aggregate_means']['B_fixed_frame_data_driven']:.2e}"
        f" | max: {checks['aggregate_maxima']['B_fixed_frame_data_driven']:.2e}"
    )
    print(
        f"  C mean error: {checks['aggregate_means']['C_equivariant_frobenius_loss']:.2e}"
        f" | max: {checks['aggregate_maxima']['C_equivariant_frobenius_loss']:.2e}"
    )
    print(f"  Heuristic checks passed: {total_pass}/{total_checks}")
    print(f"  Runtime: {results['runtime_seconds']:.2f} s")
    print()
    print("  CALIBRATED CONCLUSION:")
    print("  - At the tested float64 configuration, positive controls A and C remain at")
    print("    machine precision at the sampled final endpoints and single-trial trajectory.")
    print("  - The fixed-frame data-driven single-layer MSE case B develops macroscopic")
    print("    covariance error under the same optimizer settings.")
    print("  - In this first pass, that failure is attributed to the non-equivariant")
    print("    fixed-frame gradient map under W -> R W S^T, not to momentum alone.")
    print("  - This report does not establish unconditional behavior across all momentum")
    print("    values, losses, matrix sizes, or network architectures.")


def main() -> None:
    results = run_experiment()
    print_cli_report(results)


if __name__ == "__main__":
    main()
