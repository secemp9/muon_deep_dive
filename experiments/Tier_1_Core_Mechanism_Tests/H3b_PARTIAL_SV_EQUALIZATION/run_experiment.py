#!/usr/bin/env python3
"""
H3b: Partial singular-value equalization in a toy deep-linear setting.

Toy scope:
- This experiment compares final loss after per-method learning-rate tuning.
- The intervention acts on the momentum buffer spectrum before each update.
- The results are useful as final-loss evidence in this toy setting only.

Not measured here:
- per-step singular-value rescue over training,
- update effective rank / subspace collapse,
- any direct directional-quality metric.

Implemented spectral transform:
For a momentum matrix M = U diag(sigma) V^T, the code flattens the first k
entries of the input singular-value spectrum to their mean, then rescales the
remaining tail proportionally so the transformed matrix has Frobenius norm
sqrt(d), matching the full polar-factor norm. k=d recovers the polar factor
U V^T exactly.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = SCRIPT_DIR / "run_experiment.py"
NOTEBOOK_PATH = SCRIPT_DIR / "run_experiment.ipynb"
DEFAULT_PLOT_PATH = SCRIPT_DIR / "h3b_partial_sv_equalization.png"

DIM = 32
NUM_LAYERS = 4
NUM_STEPS = 500
MOMENTUM = 0.9
NUM_SEEDS = 5
BATCH_SIZE = 64

K_VALUES = [1, 2, 4, 8, 16, 32]
LR_CANDIDATES = [0.1, 0.07, 0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.002, 0.001]


# =============================================================================
# CONFIG / METADATA
# =============================================================================


def default_seeds() -> list[int]:
    return [42 + i * 137 for i in range(NUM_SEEDS)]


def get_experiment_config() -> dict[str, Any]:
    seeds = default_seeds()
    selection_seeds = seeds[:3]
    expected_training_runs = (
        len(K_VALUES) * len(LR_CANDIDATES) * len(selection_seeds)  # partial-SV LR sweep
        + len(LR_CANDIDATES) * len(selection_seeds)                # SGD LR sweep
        + len(K_VALUES) * len(seeds)                               # partial-SV evaluation
        + len(seeds)                                               # SGD evaluation
        + len(seeds)                                               # explicit full-polar reference check
    )
    return {
        "experiment_id": "H3b_PARTIAL_SV_EQUALIZATION",
        "title": "H3b: Partial singular-value equalization",
        "scope": "Toy deep-linear final-loss benchmark with per-method LR tuning.",
        "measured_outcome": "Final loss after training; no direct per-step spectral dynamics are measured.",
        "not_measured": [
            "Per-step singular-value rescue over training",
            "Update effective rank / subspace collapse",
            "Any direct directional-quality or weak-direction-rescue metric",
        ],
        "counterpart_script": str(SCRIPT_PATH),
        "counterpart_notebook": str(NOTEBOOK_PATH),
        "default_plot_path": str(DEFAULT_PLOT_PATH),
        "cli_reproduction_command": "python experiments/Tier_1_Core_Mechanism_Tests/H3b_PARTIAL_SV_EQUALIZATION/run_experiment.py",
        "dim": DIM,
        "num_layers": NUM_LAYERS,
        "num_steps": NUM_STEPS,
        "momentum": MOMENTUM,
        "num_seeds": NUM_SEEDS,
        "batch_size": BATCH_SIZE,
        "k_values": list(K_VALUES),
        "lr_candidates": list(LR_CANDIDATES),
        "seeds": seeds,
        "selection_seeds": selection_seeds,
        "full_polar_label": f"k={DIM} full polar / Muon-style reference",
        "expected_training_runs": expected_training_runs,
        "target_step_frobenius_norm": float(np.sqrt(DIM)),
    }


# =============================================================================
# NETWORK
# =============================================================================


def init_weights(seed: int) -> list[np.ndarray]:
    rng = np.random.RandomState(seed)
    return [np.eye(DIM) + rng.randn(DIM, DIM) * 0.1 for _ in range(NUM_LAYERS)]



def forward(weights: list[np.ndarray], X: np.ndarray) -> np.ndarray:
    out = X.copy()
    for W in weights:
        out = W @ out
    return out



def compute_loss(weights: list[np.ndarray], X: np.ndarray, Y: np.ndarray) -> float:
    pred = forward(weights, X)
    return float(0.5 * np.mean(np.sum((pred - Y) ** 2, axis=0)))



def compute_gradients(weights: list[np.ndarray], X: np.ndarray, Y: np.ndarray) -> list[np.ndarray]:
    num_layers = len(weights)
    batch_n = X.shape[1]
    activations = [X.copy()]
    for W in weights:
        activations.append(W @ activations[-1])
    delta = (activations[-1] - Y) / batch_n
    grads: list[np.ndarray] = [None] * num_layers  # type: ignore[assignment]
    for layer_idx in range(num_layers - 1, -1, -1):
        grads[layer_idx] = delta @ activations[layer_idx].T
        if layer_idx > 0:
            delta = weights[layer_idx].T @ delta
    return grads


# =============================================================================
# SPECTRAL TRANSFORMS
# =============================================================================


def polar_factor(M: np.ndarray) -> np.ndarray:
    """Explicit polar-factor reference used for T3 sanity checking."""
    if np.linalg.norm(M) < 1e-15:
        return np.zeros_like(M)
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    return U @ Vt



def transform_spectrum(sigma: np.ndarray, k: int, target_norm: float | None = None) -> np.ndarray:
    """
    Transform a singular-value vector according to the implemented intervention.

    Given a descending spectrum sigma:
      1. Flatten the first k entries of sigma to their mean.
      2. Rescale the remaining entries proportionally so the full transformed
         spectrum has Frobenius norm sqrt(d) unless target_norm is overridden.

    Important precision note:
    this function equalizes the first k entries of the *input* spectrum before
    reconstruction. After reconstruction and re-sorting, the largest k singular
    values of the resulting matrix are not separately re-equalized again.
    """
    sigma = np.array(sigma, dtype=float, copy=True)
    d = len(sigma)
    if target_norm is None:
        target_norm = float(np.sqrt(d))
    if np.linalg.norm(sigma) < 1e-15:
        return sigma

    sigma_new = sigma.copy()
    kk = int(min(max(k, 0), d))

    if kk > 0:
        sigma_new[:kk] = float(np.mean(sigma[:kk]))

    if kk >= d:
        current_norm = np.linalg.norm(sigma_new)
        if current_norm > 1e-15:
            sigma_new *= target_norm / current_norm
        return sigma_new

    top_energy = float(np.sum(sigma_new[:kk] ** 2))
    remaining_energy_budget = float(target_norm ** 2 - top_energy)

    if remaining_energy_budget > 1e-15:
        rest_current_energy = float(np.sum(sigma[kk:] ** 2))
        if rest_current_energy > 1e-15:
            scale = np.sqrt(remaining_energy_budget / rest_current_energy)
            sigma_new[kk:] = sigma[kk:] * scale
        else:
            sigma_new[kk:] = np.sqrt(remaining_energy_budget / max(d - kk, 1))
    else:
        sigma_new[kk:] = 0.0
        current_norm = np.linalg.norm(sigma_new)
        if current_norm > 1e-15:
            sigma_new *= target_norm / current_norm

    return sigma_new



def partial_sv_equalize(M: np.ndarray, k: int) -> np.ndarray:
    """
    Apply the implemented partial singular-value equalization to matrix M.

    Boundary cases:
    - k = 0: no flattening; the entire spectrum is only norm-matched to sqrt(d)
    - k = d: exact full polar factor U V^T
    """
    U, sigma, Vt = np.linalg.svd(M, full_matrices=False)
    if np.linalg.norm(sigma) < 1e-15:
        return np.zeros_like(M)
    sigma_new = transform_spectrum(sigma, k, target_norm=float(np.sqrt(len(sigma))))
    return U @ np.diag(sigma_new) @ Vt


# =============================================================================
# TRAINING
# =============================================================================


def train_partial_sv(weights_init: list[np.ndarray], X: np.ndarray, Y: np.ndarray, lr: float, k: int) -> float:
    """Train with partial SV equalization applied to each layer momentum buffer."""
    weights = [W.copy() for W in weights_init]
    momentum_buffers = [np.zeros((DIM, DIM)) for _ in range(NUM_LAYERS)]

    for _step in range(NUM_STEPS):
        loss = compute_loss(weights, X, Y)
        if not np.isfinite(loss) or loss > 1e10:
            return float("inf")
        grads = compute_gradients(weights, X, Y)
        for layer_idx in range(NUM_LAYERS):
            momentum_buffers[layer_idx] = MOMENTUM * momentum_buffers[layer_idx] + grads[layer_idx]
            step_dir = partial_sv_equalize(momentum_buffers[layer_idx], k)
            weights[layer_idx] = weights[layer_idx] - lr * step_dir
    return compute_loss(weights, X, Y)



def train_sgd(weights_init: list[np.ndarray], X: np.ndarray, Y: np.ndarray, lr: float) -> float:
    """Plain SGD with momentum baseline."""
    weights = [W.copy() for W in weights_init]
    momentum_buffers = [np.zeros((DIM, DIM)) for _ in range(NUM_LAYERS)]

    for _step in range(NUM_STEPS):
        loss = compute_loss(weights, X, Y)
        if not np.isfinite(loss) or loss > 1e10:
            return float("inf")
        grads = compute_gradients(weights, X, Y)
        for layer_idx in range(NUM_LAYERS):
            momentum_buffers[layer_idx] = MOMENTUM * momentum_buffers[layer_idx] + grads[layer_idx]
            weights[layer_idx] = weights[layer_idx] - lr * momentum_buffers[layer_idx]
    return compute_loss(weights, X, Y)



def train_full_polar_reference(weights_init: list[np.ndarray], X: np.ndarray, Y: np.ndarray, lr: float) -> float:
    """Separate explicit full-polar reference used for the non-tautological T3 check."""
    weights = [W.copy() for W in weights_init]
    momentum_buffers = [np.zeros((DIM, DIM)) for _ in range(NUM_LAYERS)]

    for _step in range(NUM_STEPS):
        loss = compute_loss(weights, X, Y)
        if not np.isfinite(loss) or loss > 1e10:
            return float("inf")
        grads = compute_gradients(weights, X, Y)
        for layer_idx in range(NUM_LAYERS):
            momentum_buffers[layer_idx] = MOMENTUM * momentum_buffers[layer_idx] + grads[layer_idx]
            step_dir = polar_factor(momentum_buffers[layer_idx])
            weights[layer_idx] = weights[layer_idx] - lr * step_dir
    return compute_loss(weights, X, Y)



def make_data(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    W_target = rng.randn(DIM, DIM) * 0.5
    X = rng.randn(DIM, BATCH_SIZE) * 0.3
    Y = W_target @ X
    return X, Y


# =============================================================================
# EVALUATION HELPERS
# =============================================================================


def _safe_float_list(values: list[float]) -> list[float]:
    return [float(v) for v in values]



def sweep_lr(train_fn: Callable[[list[np.ndarray], np.ndarray, np.ndarray, float], float], seeds: list[int], candidates: list[float]) -> dict[str, Any]:
    """Sweep LR candidates using the first three seeds for selection."""
    records = []
    best_lr = float(candidates[-1])
    best_mean_loss = float("inf")

    for lr in candidates:
        losses = []
        for seed in seeds[:3]:
            X, Y = make_data(seed)
            weights = init_weights(seed + 5000)
            final_loss = float(train_fn(weights, X, Y, lr))
            losses.append(final_loss)
        finite = [loss for loss in losses if np.isfinite(loss)]
        mean_loss = float(np.mean(finite)) if finite else float("inf")
        record = {
            "lr": float(lr),
            "selection_losses": _safe_float_list(losses),
            "selection_mean_loss": mean_loss,
            "n_finite": len(finite),
            "n_diverged": len(losses) - len(finite),
        }
        records.append(record)
        if mean_loss < best_mean_loss:
            best_mean_loss = mean_loss
            best_lr = float(lr)

    return {
        "best_lr": best_lr,
        "best_selection_mean_loss": best_mean_loss,
        "records": records,
    }



def evaluate_optimizer(train_fn: Callable[[list[np.ndarray], np.ndarray, np.ndarray, float], float], seeds: list[int], lr: float) -> dict[str, Any]:
    losses = []
    for seed in seeds:
        X, Y = make_data(seed)
        weights = init_weights(seed + 5000)
        final_loss = float(train_fn(weights, X, Y, lr))
        losses.append(final_loss)
    finite = [loss for loss in losses if np.isfinite(loss)]
    mean_loss = float(np.mean(finite)) if finite else float("inf")
    std_loss = float(np.std(finite)) if len(finite) > 1 else 0.0
    return {
        "lr": float(lr),
        "final_losses": _safe_float_list(losses),
        "mean_loss": mean_loss,
        "std_loss": std_loss,
        "n_finite": len(finite),
        "n_diverged": len(losses) - len(finite),
    }



def compute_gap_closed_pct(loss_value: float, sgd_loss: float, full_polar_loss: float) -> float:
    total_gap = sgd_loss - full_polar_loss
    if not (np.isfinite(loss_value) and np.isfinite(sgd_loss) and np.isfinite(full_polar_loss)):
        return float("nan")
    if abs(total_gap) < 1e-30:
        return float("nan")
    return float(100.0 * (sgd_loss - loss_value) / total_gap)



def build_summary_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    partial_eval = results["evaluation"]["partial"]
    sgd_eval = results["evaluation"]["sgd"]
    derived_by_k = results["derived_metrics"]["by_k"]
    rows = [{
        "label": "SGD",
        "method": "sgd",
        "k": None,
        "equalized_fraction_pct": 0.0,
        "best_lr": sgd_eval["lr"],
        "mean_loss": sgd_eval["mean_loss"],
        "std_loss": sgd_eval["std_loss"],
        "gap_closed_pct": 0.0,
        "marginal_gap_closed_pct": 0.0,
        "vs_sgd": 1.0,
        "vs_full_polar": sgd_eval["mean_loss"] / max(results["derived_metrics"]["full_polar_mean_loss"], 1e-30),
        "n_finite": sgd_eval["n_finite"],
        "n_diverged": sgd_eval["n_diverged"],
    }]

    for k in K_VALUES:
        eval_row = partial_eval[k]
        derived = derived_by_k[k]
        rows.append({
            "label": f"k={k}",
            "method": "partial_sv_equalization",
            "k": int(k),
            "equalized_fraction_pct": 100.0 * k / DIM,
            "best_lr": eval_row["lr"],
            "mean_loss": eval_row["mean_loss"],
            "std_loss": eval_row["std_loss"],
            "gap_closed_pct": derived["gap_closed_pct"],
            "marginal_gap_closed_pct": derived["marginal_gap_closed_pct"],
            "vs_sgd": derived["vs_sgd"],
            "vs_full_polar": derived["vs_full_polar"],
            "n_finite": eval_row["n_finite"],
            "n_diverged": eval_row["n_diverged"],
        })
    return rows



def _full_polar_matrix_check(num_matrices: int = 3, rng_seed: int = 20260511) -> dict[str, Any]:
    rng = np.random.RandomState(rng_seed)
    fro_errors = []
    rel_errors = []
    for _ in range(num_matrices):
        M = rng.randn(DIM, DIM)
        approx = partial_sv_equalize(M, DIM)
        ref = polar_factor(M)
        abs_err = float(np.linalg.norm(approx - ref))
        rel_err = abs_err / max(float(np.linalg.norm(ref)), 1e-30)
        fro_errors.append(abs_err)
        rel_errors.append(float(rel_err))
    return {
        "rng_seed": rng_seed,
        "num_matrices": num_matrices,
        "frobenius_errors": fro_errors,
        "relative_errors": rel_errors,
        "max_frobenius_error": max(fro_errors),
        "max_relative_error": max(rel_errors),
    }


# =============================================================================
# REPORTING / PLOTTING
# =============================================================================


def plot_results(results: dict[str, Any], save_path: str | Path | None = None, show: bool = False):
    import matplotlib.pyplot as plt

    partial_eval = results["evaluation"]["partial"]
    derived_by_k = results["derived_metrics"]["by_k"]
    sgd_eval = results["evaluation"]["sgd"]
    full_polar_loss = results["derived_metrics"]["full_polar_mean_loss"]

    k_vals = list(K_VALUES)
    loss_vals = [partial_eval[k]["mean_loss"] for k in k_vals]
    std_vals = [partial_eval[k]["std_loss"] for k in k_vals]
    gaps = [derived_by_k[k]["gap_closed_pct"] for k in k_vals]
    marginals = [derived_by_k[k]["marginal_gap_closed_pct"] for k in k_vals]

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    fig.suptitle(
        "H3b: Partial SV Equalization (toy final-loss benchmark)\n"
        f"{NUM_LAYERS}-layer deep linear {DIM}x{DIM}, {NUM_STEPS} steps, {NUM_SEEDS} seeds",
        fontsize=13,
        fontweight="bold",
    )

    # (a) mean final loss vs k
    ax = axes[0]
    ax.errorbar(k_vals, loss_vals, yerr=std_vals, marker="o", linewidth=2, capsize=4, color="#CC3311", label="Partial SV equalization")
    ax.axhline(y=sgd_eval["mean_loss"], color="#4477AA", linestyle="--", linewidth=1.5, label=f"SGD ({sgd_eval['mean_loss']:.2e})")
    ax.axhline(y=full_polar_loss, color="#228B22", linestyle="--", linewidth=1.5, label=f"k={DIM} full polar ({full_polar_loss:.2e})")
    ax.set_xlabel("k (number of input-spectrum entries flattened)")
    ax.set_ylabel("Final loss")
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_xticks(k_vals)
    ax.set_xticklabels([str(k) for k in k_vals])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Final mean loss vs k")

    # (b) percent of SGD->full-polar gap closed
    ax = axes[1]
    ax.plot(k_vals, gaps, marker="s", linewidth=2, color="#9933CC")
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.4)
    ax.axhline(y=50, color="gray", linestyle=":", alpha=0.6, label="50% threshold")
    ax.axhline(y=80, color="gray", linestyle="--", alpha=0.6, label="80% threshold")
    ax.axhline(y=100, color="#228B22", linestyle="-", alpha=0.3, label="100% (full polar)")
    finite_gaps = [g for g in gaps if np.isfinite(g)]
    if finite_gaps:
        low = min(finite_gaps + [0.0])
        high = max(finite_gaps + [100.0])
        pad = max(5.0, 0.08 * (high - low if high > low else 1.0))
        ax.set_ylim(low - pad, high + pad)
    ax.set_xlabel("k (number of input-spectrum entries flattened)")
    ax.set_ylabel("% of SGD→full-polar gap closed")
    ax.set_xscale("log", base=2)
    ax.set_xticks(k_vals)
    ax.set_xticklabels([str(k) for k in k_vals])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Gap closure vs k")

    # (c) marginal gap closure
    ax = axes[2]
    colors = ["#FF8800" if value >= 0 else "#BB5566" for value in marginals]
    ax.bar(range(len(k_vals)), marginals, color=colors, edgecolor="black")
    ax.axhline(y=0, color="black", linewidth=1, alpha=0.5)
    ax.set_xticks(range(len(k_vals)))
    ax.set_xticklabels([f"k={k}" for k in k_vals], rotation=30)
    ax.set_ylabel("Marginal % gap closed")
    ax.set_title("Marginal gain from increasing k")
    ax.grid(True, alpha=0.3, axis="y")
    if marginals:
        y_low = min(marginals + [0.0])
        y_high = max(marginals + [0.0])
        pad = max(3.0, 0.08 * (y_high - y_low if y_high > y_low else 1.0))
        ax.set_ylim(y_low - pad, y_high + pad)
        for idx, value in enumerate(marginals):
            text_y = value + (1.0 if value >= 0 else -1.0)
            va = "bottom" if value >= 0 else "top"
            ax.text(idx, text_y, f"{value:.1f}%", ha="center", va=va, fontsize=9)

    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes



def print_results_report(results: dict[str, Any]) -> None:
    config = results["config"]
    partial_eval = results["evaluation"]["partial"]
    sgd_eval = results["evaluation"]["sgd"]
    derived = results["derived_metrics"]
    tests = results["tests"]

    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)
    print("Toy scope reminder: this file measures final loss after LR tuning only.")
    print("It does not directly measure per-step singular-value rescue or update-rank dynamics.")
    print()
    print(f"SGD mean loss:          {sgd_eval['mean_loss']:.6e}")
    print(f"Full-polar mean loss:   {derived['full_polar_mean_loss']:.6e}")
    print(f"SGD→full-polar gap:     {derived['total_gap']:.6e}")
    print()
    print(f"{'label':>8}  {'mean loss':>14}  {'std':>14}  {'best lr':>8}  {'vs SGD':>10}  {'vs full polar':>15}  {'gap closed':>12}")
    print("  " + "-" * 98)
    print(
        f"{'SGD':>8}  {sgd_eval['mean_loss']:>14.6e}  {sgd_eval['std_loss']:>14.6e}  {sgd_eval['lr']:>8.4f}  "
        f"{1.0:>10.2f}x  {sgd_eval['mean_loss']/max(derived['full_polar_mean_loss'], 1e-30):>15.2f}x  {0.0:>11.1f}%"
    )
    for k in K_VALUES:
        row = partial_eval[k]
        metric_row = derived["by_k"][k]
        marker = "  <-- full polar" if k == DIM else ""
        print(
            f"{f'k={k}':>8}  {row['mean_loss']:>14.6e}  {row['std_loss']:>14.6e}  {row['lr']:>8.4f}  "
            f"{metric_row['vs_sgd']:>10.2f}x  {metric_row['vs_full_polar']:>15.2f}x  {metric_row['gap_closed_pct']:>11.1f}%{marker}"
        )

    print("\n" + "=" * 100)
    print("HYPOTHESIS TESTS")
    print("=" * 100)
    print(f"T1: {tests['T1']['description']}")
    print(f"    gap closed by k=1: {tests['T1']['gap_closed_pct']:.1f}% -> {'PASS' if tests['T1']['passed'] else 'FAIL'}")
    print()
    print(f"T2: {tests['T2']['description']}")
    if tests['T2']['knee_k'] is None:
        print("    no k crossed the 80% threshold -> FAIL")
    else:
        print(f"    knee_k={tests['T2']['knee_k']} with gap_closed={tests['T2']['knee_gap_closed_pct']:.1f}% -> {'PASS' if tests['T2']['passed'] else 'FAIL'}")
    print()
    print(f"T3: {tests['T3']['description']}")
    print(
        "    matrix-level reference check: "
        f"max_matrix_frob_error={tests['T3']['matrix_check']['max_frobenius_error']:.3e} -> "
        f"{'PASS' if tests['T3']['passed'] else 'FAIL'}"
    )
    print(
        "    training-loss diagnostic at same LR: "
        f"max_abs_loss_diff={tests['T3']['max_abs_loss_diff']:.3e}"
    )

    print("\n" + "=" * 100)
    print("CALIBRATED CONCLUSION")
    print("=" * 100)
    if tests["T1"]["passed"]:
        print(
            "In this toy final-loss benchmark, k=1 recovered more than half of the "
            "SGD→full-polar gap. That is consistent with a top-heavy mechanism, but it is "
            "still not a direct measurement of singular-value rescue."
        )
    else:
        print(
            "In this toy final-loss benchmark, k=1 did not recover more than half of the "
            "SGD→full-polar gap. Partial equalization at very small k was not sufficient."
        )

    if tests["T2"]["passed"]:
        print(
            f"A knee appeared at k={tests['T2']['knee_k']}, so a relatively small number of "
            "flattened spectrum entries captured most of the benchmarked final-loss gain."
        )
    else:
        print(
            "No early knee was detected under the current default setup, so the final-loss "
            "benefit did not concentrate cleanly at small k."
        )

    print(
        f"Runtime: {results['runtime_sec']:.1f}s across an expected {config['expected_training_runs']} training runs."
    )


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================


def run_experiment(verbose: bool = False, save_plot: bool = False, plot_path: str | Path | None = None) -> dict[str, Any]:
    config = get_experiment_config()
    seeds = list(config["seeds"])
    selection_seeds = list(config["selection_seeds"])

    if verbose:
        print("=" * 100)
        print("H3b: PARTIAL SV EQUALIZATION")
        print("=" * 100)
        print("Toy scope: final loss after LR tuning in a deep-linear benchmark.")
        print("Mechanistic claims should be read cautiously unless direct spectrum-dynamics metrics are added.")
        print(f"Counterpart notebook: {config['counterpart_notebook']}")
        print(f"CLI reproduction: {config['cli_reproduction_command']}")
        print(f"k values: {config['k_values']} (k={DIM} is the full-polar / Muon-style reference)")
        print(
            f"Network: {NUM_LAYERS}-layer deep linear, {DIM}x{DIM}, {NUM_STEPS} steps, "
            f"{NUM_SEEDS} seeds, batch size {BATCH_SIZE}"
        )
        print(f"Seeds: {seeds}")
        print(f"Expected training runs: {config['expected_training_runs']}")
        print()

    start_time = time.time()

    # Phase 1: LR sweep
    if verbose:
        print("Phase 1: LR sweep per k (selection seeds only)")
    lr_sweep_partial: dict[int, dict[str, Any]] = {}
    best_lrs_partial: dict[int, float] = {}
    for k in K_VALUES:
        train_fn = lambda weights, X, Y, lr, _k=k: train_partial_sv(weights, X, Y, lr, _k)
        sweep_data = sweep_lr(train_fn, seeds, LR_CANDIDATES)
        lr_sweep_partial[k] = sweep_data
        best_lrs_partial[k] = float(sweep_data["best_lr"])
        if verbose:
            print(f"  k={k:>3}: best_lr={best_lrs_partial[k]:.4f}  selection_mean={sweep_data['best_selection_mean_loss']:.6e}")

    lr_sweep_sgd = sweep_lr(train_sgd, seeds, LR_CANDIDATES)
    sgd_best_lr = float(lr_sweep_sgd["best_lr"])
    if verbose:
        print(f"  SGD:   best_lr={sgd_best_lr:.4f}  selection_mean={lr_sweep_sgd['best_selection_mean_loss']:.6e}")

    # Phase 2: full evaluation
    if verbose:
        print("\nPhase 2: full evaluation on all seeds")
    evaluation_partial: dict[int, dict[str, Any]] = {}
    for k in K_VALUES:
        train_fn = lambda weights, X, Y, lr, _k=k: train_partial_sv(weights, X, Y, lr, _k)
        eval_data = evaluate_optimizer(train_fn, seeds, best_lrs_partial[k])
        evaluation_partial[k] = eval_data
        if verbose:
            status = "" if eval_data["n_diverged"] == 0 else f"  [diverged seeds: {eval_data['n_diverged']}]"
            print(
                f"  k={k:>3}: loss={eval_data['mean_loss']:.6e} +/- {eval_data['std_loss']:.6e} "
                f"(lr={eval_data['lr']:.4f}){status}"
            )

    evaluation_sgd = evaluate_optimizer(train_sgd, seeds, sgd_best_lr)
    if verbose:
        status = "" if evaluation_sgd["n_diverged"] == 0 else f"  [diverged seeds: {evaluation_sgd['n_diverged']}]"
        print(
            f"  SGD:   loss={evaluation_sgd['mean_loss']:.6e} +/- {evaluation_sgd['std_loss']:.6e} "
            f"(lr={evaluation_sgd['lr']:.4f}){status}"
        )

    # Explicit full-polar reference check for T3
    full_polar_reference_eval = evaluate_optimizer(train_full_polar_reference, seeds, best_lrs_partial[DIM])

    sgd_mean = float(evaluation_sgd["mean_loss"])
    full_polar_mean = float(evaluation_partial[DIM]["mean_loss"])
    total_gap = float(sgd_mean - full_polar_mean)

    derived_by_k: dict[int, dict[str, Any]] = {}
    previous_gap = 0.0
    for k in K_VALUES:
        loss_value = float(evaluation_partial[k]["mean_loss"])
        gap_closed_pct = compute_gap_closed_pct(loss_value, sgd_mean, full_polar_mean)
        marginal_gap = gap_closed_pct - previous_gap if np.isfinite(gap_closed_pct) else float("nan")
        derived_by_k[k] = {
            "gap_closed_pct": gap_closed_pct,
            "marginal_gap_closed_pct": float(marginal_gap),
            "vs_sgd": float(sgd_mean / max(loss_value, 1e-30)),
            "vs_full_polar": float(loss_value / max(full_polar_mean, 1e-30)),
        }
        previous_gap = gap_closed_pct if np.isfinite(gap_closed_pct) else previous_gap

    gap_k1 = float(derived_by_k[1]["gap_closed_pct"])
    t1_passed = bool(np.isfinite(gap_k1) and gap_k1 > 50.0)

    knee_k = None
    knee_gap_closed = float("nan")
    for k in K_VALUES:
        gap_k = derived_by_k[k]["gap_closed_pct"]
        if np.isfinite(gap_k) and gap_k > 80.0:
            knee_k = int(k)
            knee_gap_closed = float(gap_k)
            break
    t2_passed = bool(knee_k is not None and knee_k < DIM // 2)

    loss_diffs = [
        abs(a - b)
        for a, b in zip(evaluation_partial[DIM]["final_losses"], full_polar_reference_eval["final_losses"])
        if np.isfinite(a) and np.isfinite(b)
    ]
    matrix_check = _full_polar_matrix_check()
    max_abs_loss_diff = max(loss_diffs) if loss_diffs else float("inf")
    mean_abs_loss_diff = float(np.mean(loss_diffs)) if loss_diffs else float("inf")
    # Primary correctness criterion: the k=d transform matches an explicit
    # polar-factor reference at the matrix level on dense test matrices.
    # The full-run loss comparison is retained as a diagnostic because repeated
    # SVD calls inside training can accumulate small numerical differences.
    t3_passed = bool(matrix_check["max_frobenius_error"] <= 1e-12)

    runtime_sec = float(time.time() - start_time)

    results = {
        "config": config,
        "seeds": seeds,
        "selection_seeds": selection_seeds,
        "lr_sweep": {
            "partial": lr_sweep_partial,
            "sgd": lr_sweep_sgd,
        },
        "best_lrs": {
            "partial": best_lrs_partial,
            "sgd": sgd_best_lr,
        },
        "evaluation": {
            "partial": evaluation_partial,
            "sgd": evaluation_sgd,
            "full_polar_reference": full_polar_reference_eval,
        },
        "derived_metrics": {
            "sgd_mean_loss": sgd_mean,
            "full_polar_mean_loss": full_polar_mean,
            "total_gap": total_gap,
            "by_k": derived_by_k,
        },
        "tests": {
            "T1": {
                "description": "Does k=1 recover more than 50% of the SGD→full-polar final-loss gap?",
                "gap_closed_pct": gap_k1,
                "threshold_pct": 50.0,
                "passed": t1_passed,
            },
            "T2": {
                "description": f"Is there an early knee: >80% gap closure at k < {DIM // 2}?",
                "knee_k": knee_k,
                "knee_gap_closed_pct": knee_gap_closed,
                "threshold_pct": 80.0,
                "passed": t2_passed,
            },
            "T3": {
                "description": f"Does k={DIM} implement the explicit full polar factor U V^T?",
                "reference_lr": best_lrs_partial[DIM],
                "k_dim_losses": evaluation_partial[DIM]["final_losses"],
                "reference_losses": full_polar_reference_eval["final_losses"],
                "max_abs_loss_diff": float(max_abs_loss_diff),
                "mean_abs_loss_diff": float(mean_abs_loss_diff),
                "matrix_check": matrix_check,
                "diagnostic_note": "Matrix-level equality is the primary correctness check; training-loss differences are retained as a numerical diagnostic only.",
                "passed": t3_passed,
            },
        },
        "runtime_sec": runtime_sec,
        "summary_rows": None,
        "artifacts": {
            "default_plot_path": str(DEFAULT_PLOT_PATH),
            "plot_saved": False,
            "plot_path": None,
        },
    }
    results["summary_rows"] = build_summary_rows(results)

    if save_plot:
        import matplotlib

        matplotlib.use("Agg")
        actual_plot_path = Path(plot_path) if plot_path is not None else DEFAULT_PLOT_PATH
        fig, _axes = plot_results(results, save_path=actual_plot_path, show=False)
        import matplotlib.pyplot as plt

        plt.close(fig)
        results["artifacts"]["plot_saved"] = True
        results["artifacts"]["plot_path"] = str(actual_plot_path)
        if verbose:
            print(f"\nPlot saved: {actual_plot_path}")

    if verbose:
        print_results_report(results)

    return results



def main() -> dict[str, Any]:
    return run_experiment(verbose=True, save_plot=True)


if __name__ == "__main__":
    main()
