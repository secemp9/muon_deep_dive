#!/usr/bin/env python3
"""
Experiment 2.8: Phantom-rank toy study with noise injected before Muon orthogonalization.

This module preserves the original deep-linear-network toy setup while making the
experiment import-safe, reproducible across Python processes, and reusable from a
notebook.

What is measured here:
  - loss trajectories
  - mean effective rank of the raw per-layer gradients
  - mean effective rank of the pre-Newton-Schulz noisy gradients actually fed into
    the orthogonalization step during tracked updates

What is not measured here:
  - hard numerical rank
  - post-orthogonalization update span / full parameter-space coverage
  - multi-seed statistical stability

The effective-rank quantity is an entropy-based surrogate, not a hard-rank test.
"""

from __future__ import annotations

import sys
import time
from typing import Dict, List, Optional

import numpy as np

# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

WIDTH = 32
DEPTH = 4
NUM_STEPS = 1000
LR_MUON = 0.02
NS_ITERS = 5
BATCH_SIZE = 64
INPUT_DIM = 32
OUTPUT_DIM = 32
SEED = 42
TRACK_EVERY = 10

NOISE_LEVELS = [0.0, 0.001, 0.01, 0.05, 0.10]  # fraction of ||G||_F
NOISE_LABELS = ["0%", "0.1%", "1%", "5%", "10%"]


# =============================================================================
# CONFIG / PROBLEM HELPERS
# =============================================================================


def get_default_config() -> Dict[str, object]:
    """Return a copy of the default experiment configuration."""
    return {
        "width": WIDTH,
        "depth": DEPTH,
        "num_steps": NUM_STEPS,
        "lr_muon": LR_MUON,
        "ns_iters": NS_ITERS,
        "batch_size": BATCH_SIZE,
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "seed": SEED,
        "track_every": TRACK_EVERY,
        "noise_levels": list(NOISE_LEVELS),
        "noise_labels": list(NOISE_LABELS),
    }


def prepare_config(config_overrides: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Merge user overrides into the default configuration and validate it."""
    config = get_default_config()
    if config_overrides:
        config.update(config_overrides)

    noise_levels = list(config["noise_levels"])
    noise_labels = list(config["noise_labels"])
    if len(noise_levels) != len(noise_labels):
        raise ValueError("noise_levels and noise_labels must have the same length")
    if int(config["track_every"]) <= 0:
        raise ValueError("track_every must be positive")
    if int(config["num_steps"]) <= 0:
        raise ValueError("num_steps must be positive")

    config["noise_levels"] = noise_levels
    config["noise_labels"] = noise_labels
    config["half_rank_threshold"] = float(config["width"]) / 2.0
    return config


def build_problem(config: Dict[str, object]) -> Dict[str, np.ndarray]:
    """Construct the fixed deep-linear regression problem used by the study."""
    rng = np.random.RandomState(int(config["seed"]))
    input_dim = int(config["input_dim"])
    output_dim = int(config["output_dim"])
    batch_size = int(config["batch_size"])

    # Fixed Gaussian inputs.
    X = rng.randn(input_dim, batch_size) * 0.5

    # Ill-conditioned target operator.
    U, _ = np.linalg.qr(rng.randn(output_dim, output_dim))
    V, _ = np.linalg.qr(rng.randn(input_dim, input_dim))
    sigma = np.array([10.0 * (0.5 ** i) for i in range(min(output_dim, input_dim))])
    T = U @ np.diag(sigma) @ V
    Y = T @ X

    return {
        "X": X,
        "Y": Y,
        "T": T,
        "sigma": sigma,
        "target_condition_number": float(sigma[0] / sigma[-1]),
        "input_fro_norm": float(np.linalg.norm(X, "fro")),
        "target_fro_norm": float(np.linalg.norm(T, "fro")),
        "target_output_fro_norm": float(np.linalg.norm(Y, "fro")),
    }


# =============================================================================
# NETWORK UTILITIES
# =============================================================================


def init_weights(num_layers: int, width: int, seed: int) -> List[np.ndarray]:
    """Initialize deep linear net weights with Xavier-style variance scaling."""
    rng = np.random.RandomState(seed)
    weights = []
    for _ in range(num_layers):
        std = np.sqrt(2.0 / (width + width))
        weights.append((rng.randn(width, width) * std).copy())
    return weights


def forward_linear(weights: List[np.ndarray], X: np.ndarray) -> np.ndarray:
    """Forward pass through a deep linear net (no activation)."""
    out = X.copy()
    for W in weights:
        out = W @ out
    return out


def compute_loss(weights: List[np.ndarray], X: np.ndarray, Y_target: np.ndarray) -> float:
    """Mean-squared-error loss."""
    Y_pred = forward_linear(weights, X)
    diff = Y_pred - Y_target
    return float(0.5 * np.mean(diff ** 2))


def compute_gradients(weights: List[np.ndarray], X: np.ndarray, Y_target: np.ndarray) -> List[np.ndarray]:
    """Exact backpropagation through the deep linear network."""
    num_layers = len(weights)
    batch_size = X.shape[1]

    activations = [X.copy()]
    out = X.copy()
    for W in weights:
        out = W @ out
        activations.append(out.copy())

    Y_pred = activations[-1]
    diff = Y_pred - Y_target
    delta = diff / batch_size

    grads = [None] * num_layers
    for layer in range(num_layers - 1, -1, -1):
        grads[layer] = delta @ activations[layer].T
        if layer > 0:
            delta = weights[layer].T @ delta

    return grads


def newton_schulz_orthogonalize(G: np.ndarray, num_iters: int = 5) -> np.ndarray:
    """Newton-Schulz iteration used by Muon to approximate the polar factor."""
    norm = np.linalg.norm(G, "fro")
    if norm < 1e-12:
        return G
    X = G / norm

    for _ in range(num_iters):
        A = X.T @ X
        X = (15.0 / 8.0) * X - (10.0 / 8.0) * X @ A + (3.0 / 8.0) * X @ A @ A

    return X


def effective_rank(M: np.ndarray) -> float:
    """Entropy-based effective rank surrogate in [0, min(m, n)]."""
    sv = np.linalg.svd(M, compute_uv=False)
    sv = sv[sv > 1e-12]
    if len(sv) == 0:
        return 0.0
    p = sv / np.sum(sv)
    H = -np.sum(p * np.log(p))
    return float(np.exp(H))


# =============================================================================
# TRAINING ROUTINES
# =============================================================================


def deterministic_noise_seed(base_seed: int, noise_frac: float) -> int:
    """Stable mapping from noise level to RNG seed.

    This replaces the old hash-based mapping so that repeated runs in separate
    Python processes produce identical noisy trajectories.
    """
    if noise_frac <= 0:
        return int(base_seed)
    noise_key = int(round(float(noise_frac) * 1_000_000))
    return int(base_seed + 137 + noise_key)


def add_scaled_noise(G: np.ndarray, rng: np.random.RandomState, noise_frac: float) -> np.ndarray:
    """Inject isotropic Gaussian noise with Frobenius norm noise_frac * ||G||_F."""
    if noise_frac <= 0:
        return G.copy()

    gnorm = np.linalg.norm(G, "fro")
    if gnorm < 1e-12:
        return G.copy()

    noise = rng.randn(*G.shape)
    noise_norm = np.linalg.norm(noise, "fro")
    if noise_norm < 1e-12:
        return G.copy()

    scaled_noise = noise * (noise_frac * gnorm / noise_norm)
    return G + scaled_noise


def first_step_below_threshold(values: List[float], steps: List[int], threshold: float) -> Optional[int]:
    """Return the first tracked step with value < threshold, else None."""
    for step, value in zip(steps, values):
        if value < threshold:
            return int(step)
    return None


def train_muon_with_noise(
    weights: List[np.ndarray],
    X: np.ndarray,
    Y: np.ndarray,
    num_steps: int,
    lr: float,
    noise_frac: float,
    ns_iters: int = 5,
    seed: int = 42,
    track_every: int = 10,
    noise_seed: Optional[int] = None,
) -> Dict[str, object]:
    """Train with Muon and optional pre-orthogonalization noise injection.

    Returned metrics are intentionally explicit:
      - raw_grad_mean_erank_history: mean erank of the raw layer gradients
      - pre_ns_noisy_grad_mean_erank_history: mean erank of the gradients after
        noise injection and before Newton-Schulz, tracked only on actual update
        steps (so no synthetic final-point value is appended)
    """
    if noise_seed is None:
        noise_seed = deterministic_noise_seed(seed, noise_frac)

    rng = np.random.RandomState(int(noise_seed))
    weights_local = [W.copy() for W in weights]

    tracked_steps_pre_update: List[int] = []
    loss_history_pre_update: List[float] = []
    raw_grad_mean_erank_history_pre_update: List[float] = []
    pre_ns_noisy_grad_mean_erank_history: List[float] = []

    for step in range(num_steps):
        track_now = (step % track_every == 0)
        if track_now:
            tracked_steps_pre_update.append(step)
            loss_history_pre_update.append(compute_loss(weights_local, X, Y))

        grads = compute_gradients(weights_local, X, Y)
        noisy_grads: List[np.ndarray] = []
        if track_now:
            raw_layer_eranks = []
            noisy_layer_eranks = []

        for G in grads:
            if track_now:
                raw_layer_eranks.append(effective_rank(G))
            G_pre_ns = add_scaled_noise(G, rng, noise_frac)
            if track_now:
                noisy_layer_eranks.append(effective_rank(G_pre_ns))
            noisy_grads.append(G_pre_ns)

        if track_now:
            raw_grad_mean_erank_history_pre_update.append(float(np.mean(raw_layer_eranks)))
            pre_ns_noisy_grad_mean_erank_history.append(float(np.mean(noisy_layer_eranks)))

        for layer_index, G_pre_ns in enumerate(noisy_grads):
            G_orth = newton_schulz_orthogonalize(G_pre_ns, ns_iters)
            weights_local[layer_index] -= lr * G_orth

    final_loss = compute_loss(weights_local, X, Y)
    final_raw_grads = compute_gradients(weights_local, X, Y)
    final_raw_layer_eranks = [effective_rank(g) for g in final_raw_grads]
    final_raw_grad_mean_erank = float(np.mean(final_raw_layer_eranks))

    tracked_steps_with_final = tracked_steps_pre_update + [num_steps]
    loss_history = loss_history_pre_update + [final_loss]
    raw_grad_mean_erank_history = raw_grad_mean_erank_history_pre_update + [final_raw_grad_mean_erank]

    if len(loss_history) != len(tracked_steps_with_final):
        raise RuntimeError("loss history length mismatch")
    if len(raw_grad_mean_erank_history) != len(tracked_steps_with_final):
        raise RuntimeError("raw erank history length mismatch")
    if len(pre_ns_noisy_grad_mean_erank_history) != len(tracked_steps_pre_update):
        raise RuntimeError("pre-NS noisy erank history length mismatch")

    numeric_arrays = [loss_history, raw_grad_mean_erank_history, pre_ns_noisy_grad_mean_erank_history]
    if not all(np.all(np.isfinite(arr)) for arr in numeric_arrays if len(arr) > 0):
        raise FloatingPointError("Encountered non-finite metrics during training")

    return {
        "weights_final": weights_local,
        "noise_frac": float(noise_frac),
        "noise_seed": int(noise_seed),
        "track_every": int(track_every),
        "loss_steps": tracked_steps_with_final,
        "loss_history": loss_history,
        "raw_grad_mean_erank_steps": tracked_steps_with_final,
        "raw_grad_mean_erank_history": raw_grad_mean_erank_history,
        "pre_ns_noisy_grad_mean_erank_steps": tracked_steps_pre_update,
        "pre_ns_noisy_grad_mean_erank_history": pre_ns_noisy_grad_mean_erank_history,
        "initial_loss": float(loss_history[0]),
        "final_loss": float(final_loss),
        "initial_raw_grad_mean_erank": float(raw_grad_mean_erank_history[0]),
        "final_raw_grad_mean_erank": float(final_raw_grad_mean_erank),
        "time_avg_raw_grad_mean_erank": float(np.mean(raw_grad_mean_erank_history)),
        "initial_pre_ns_noisy_grad_mean_erank": float(pre_ns_noisy_grad_mean_erank_history[0]),
        "time_avg_pre_ns_noisy_grad_mean_erank": float(np.mean(pre_ns_noisy_grad_mean_erank_history)),
    }


def train_plain_muon(
    weights: List[np.ndarray],
    X: np.ndarray,
    Y: np.ndarray,
    num_steps: int,
    lr: float,
    ns_iters: int = 5,
    track_every: int = 10,
    seed: int = 42,
) -> Dict[str, object]:
    """Plain Muon convenience wrapper (equivalent to noise_frac=0)."""
    return train_muon_with_noise(
        weights,
        X,
        Y,
        num_steps,
        lr,
        0.0,
        ns_iters=ns_iters,
        seed=seed,
        track_every=track_every,
        noise_seed=deterministic_noise_seed(seed, 0.0),
    )


# =============================================================================
# ANALYSIS / REPORTING
# =============================================================================


def build_summary_rows(config: Dict[str, object], per_noise: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    """Create summary rows for text output and notebook tables."""
    half_threshold = float(config["half_rank_threshold"])
    rows = []
    for noise_label, noise_frac in zip(config["noise_labels"], config["noise_levels"]):
        run = per_noise[noise_label]
        first_below = first_step_below_threshold(
            run["raw_grad_mean_erank_history"],
            run["raw_grad_mean_erank_steps"],
            half_threshold,
        )
        rows.append(
            {
                "noise_label": noise_label,
                "noise_frac": float(noise_frac),
                "noise_seed": int(run["noise_seed"]),
                "final_loss": float(run["final_loss"]),
                "initial_loss": float(run["initial_loss"]),
                "initial_raw_grad_mean_erank": float(run["initial_raw_grad_mean_erank"]),
                "time_avg_raw_grad_mean_erank": float(run["time_avg_raw_grad_mean_erank"]),
                "final_raw_grad_mean_erank": float(run["final_raw_grad_mean_erank"]),
                "initial_pre_ns_noisy_grad_mean_erank": float(run["initial_pre_ns_noisy_grad_mean_erank"]),
                "time_avg_pre_ns_noisy_grad_mean_erank": float(run["time_avg_pre_ns_noisy_grad_mean_erank"]),
                "first_raw_grad_erank_below_half_step": first_below,
                "starts_below_half": bool(run["initial_raw_grad_mean_erank"] < half_threshold),
            }
        )
    return rows


def build_verdict(config: Dict[str, object], per_noise: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    """Build a conservative verdict that matches the measured metrics."""
    half_threshold = float(config["half_rank_threshold"])
    baseline = per_noise["0%"]
    one_pct = per_noise.get("1%")

    baseline_initial = float(baseline["initial_raw_grad_mean_erank"])
    baseline_final = float(baseline["final_raw_grad_mean_erank"])
    baseline_steps = baseline["raw_grad_mean_erank_steps"]
    baseline_history = baseline["raw_grad_mean_erank_history"]
    baseline_first_below = first_step_below_threshold(baseline_history, baseline_steps, half_threshold)
    baseline_initial_below = baseline_initial < half_threshold

    baseline_loss = float(baseline["final_loss"])
    best_noise_label = min(per_noise, key=lambda label: per_noise[label]["final_loss"])
    best_final_loss = float(per_noise[best_noise_label]["final_loss"])
    any_noise_improves = any(
        float(run["final_loss"]) < baseline_loss
        for label, run in per_noise.items()
        if label != "0%"
    )

    verdict = {
        "heuristic_half_rank_threshold": half_threshold,
        "baseline_initial_raw_grad_mean_erank": baseline_initial,
        "baseline_final_raw_grad_mean_erank": baseline_final,
        "baseline_initial_below_half": baseline_initial_below,
        "baseline_first_below_half_step": baseline_first_below,
        "baseline_ever_below_half": baseline_first_below is not None,
        "baseline_is_already_low_erank_at_step0": baseline_initial_below,
        "best_noise_label": best_noise_label,
        "best_final_loss": best_final_loss,
        "any_noise_improves_final_loss": any_noise_improves,
        "support_level": "mixed or null",
        "interpretation": "No verdict computed yet.",
    }

    if one_pct is not None:
        loss_1pct = float(one_pct["final_loss"])
        improvement_abs = baseline_loss - loss_1pct
        improvement_pct = 100.0 * improvement_abs / baseline_loss if baseline_loss > 1e-12 else 0.0
        verdict.update(
            {
                "one_percent_final_loss": loss_1pct,
                "one_percent_improves_final_loss": loss_1pct < baseline_loss,
                "one_percent_improvement_abs": improvement_abs,
                "one_percent_improvement_pct": improvement_pct,
                "one_percent_raises_time_avg_raw_grad_mean_erank": float(one_pct["time_avg_raw_grad_mean_erank"])
                > float(baseline["time_avg_raw_grad_mean_erank"]),
                "one_percent_raises_time_avg_pre_ns_noisy_grad_mean_erank": float(one_pct["time_avg_pre_ns_noisy_grad_mean_erank"])
                > float(baseline["time_avg_pre_ns_noisy_grad_mean_erank"]),
            }
        )
    else:
        verdict.update(
            {
                "one_percent_final_loss": None,
                "one_percent_improves_final_loss": None,
                "one_percent_improvement_abs": None,
                "one_percent_improvement_pct": None,
                "one_percent_raises_time_avg_raw_grad_mean_erank": None,
                "one_percent_raises_time_avg_pre_ns_noisy_grad_mean_erank": None,
            }
        )

    if baseline_initial_below and any_noise_improves:
        verdict["support_level"] = "supportive but limited"
        verdict["interpretation"] = (
            "In this single-seed toy run, the baseline already starts in a low-erank "
            "regime under the n/2 heuristic, and at least one noisy run reaches lower "
            "final loss. This is supportive of the intervention, but it does not show "
            "hard-rank recovery or full parameter-space restoration."
        )
    elif baseline_initial_below and not any_noise_improves:
        verdict["support_level"] = "low-erank without loss benefit"
        verdict["interpretation"] = (
            "The baseline begins below the n/2 heuristic threshold, but the tested "
            "noise levels do not improve final loss in this run. The intervention is "
            "not supported here."
        )
    else:
        verdict["support_level"] = "no baseline low-erank trigger"
        verdict["interpretation"] = (
            "The baseline does not begin below the n/2 heuristic threshold, so this "
            "run does not clearly instantiate the intended low-erank regime."
        )

    return verdict


def format_experiment_report(results: Dict[str, object]) -> str:
    """Format a human-readable experiment report for CLI/script execution."""
    config = results["config"]
    diagnostics = results["data_diagnostics"]
    summary_rows = results["summary_rows"]
    verdict = results["verdict"]

    lines = []
    lines.append("=" * 96)
    lines.append("Experiment 2.8: Phantom-rank toy study -- noise injection before Muon orthogonalization")
    lines.append("=" * 96)
    lines.append("Scope: single-seed deep-linear pilot. Measured metrics are loss and effective-rank")
    lines.append("surrogates of raw/pre-NS gradients; this does NOT directly measure hard rank or")
    lines.append("full parameter-space restoration.")
    lines.append("")
    lines.append(
        f"Config: depth={config['depth']}, width={config['width']}, steps={config['num_steps']}, "
        f"lr={config['lr_muon']}, ns_iters={config['ns_iters']}, batch={config['batch_size']}, "
        f"seed={config['seed']}, track_every={config['track_every']}"
    )
    lines.append(
        f"Target condition number: {diagnostics['target_condition_number']:.0f}; "
        f"half-rank heuristic n/2 = {config['half_rank_threshold']:.1f}"
    )
    lines.append(f"Runtime: {results['runtime_sec']:.2f} s")
    lines.append("")
    lines.append("Summary metrics")
    lines.append("-" * 96)
    lines.append(
        f"{'Noise':<8} {'Seed':>8} {'Final loss':>12} {'Init raw erank':>16} "
        f"{'Avg raw erank':>16} {'Final raw erank':>16} {'Avg pre-NS erank':>18}"
    )
    for row in summary_rows:
        lines.append(
            f"{row['noise_label']:<8} {row['noise_seed']:>8d} {row['final_loss']:>12.6f} "
            f"{row['initial_raw_grad_mean_erank']:>16.2f} {row['time_avg_raw_grad_mean_erank']:>16.2f} "
            f"{row['final_raw_grad_mean_erank']:>16.2f} {row['time_avg_pre_ns_noisy_grad_mean_erank']:>18.2f}"
        )
    lines.append("")
    lines.append("Interpretation checks")
    lines.append("-" * 96)
    lines.append(
        f"Baseline initial raw-grad mean erank: {verdict['baseline_initial_raw_grad_mean_erank']:.2f} "
        f"vs n/2={verdict['heuristic_half_rank_threshold']:.1f}"
    )
    if verdict["baseline_initial_below_half"]:
        lines.append("Baseline status: already below the heuristic threshold at step 0 (no observed collapse from near-full rank).")
    else:
        first_below = verdict["baseline_first_below_half_step"]
        if first_below is None:
            lines.append("Baseline status: never drops below the heuristic threshold in the tracked history.")
        else:
            lines.append(f"Baseline status: first drops below the heuristic threshold at tracked step {first_below}.")

    if verdict["one_percent_improves_final_loss"] is not None:
        lines.append(
            f"1% noise final loss: {verdict['one_percent_final_loss']:.6f} "
            f"({'improves' if verdict['one_percent_improves_final_loss'] else 'does not improve'} vs baseline; "
            f"delta={verdict['one_percent_improvement_abs']:.6f}, "
            f"{verdict['one_percent_improvement_pct']:.2f}%)."
        )
        lines.append(
            f"1% noise raises time-avg raw-grad erank: {verdict['one_percent_raises_time_avg_raw_grad_mean_erank']}"
        )
        lines.append(
            f"1% noise raises time-avg pre-NS noisy-grad erank: {verdict['one_percent_raises_time_avg_pre_ns_noisy_grad_mean_erank']}"
        )

    lines.append(f"Best final loss in sweep: {verdict['best_noise_label']} (loss={verdict['best_final_loss']:.6f})")
    lines.append("")
    lines.append(f"Overall reading: {verdict['support_level']}")
    lines.append(verdict["interpretation"])
    lines.append("=" * 96)
    return "\n".join(lines)


# =============================================================================
# MAIN EXPERIMENT API
# =============================================================================


def run_experiment(
    config_overrides: Optional[Dict[str, object]] = None,
    emit_report: bool = False,
) -> Dict[str, object]:
    """Run the default phantom-rank toy study and return structured results."""
    start_time = time.perf_counter()
    config = prepare_config(config_overrides)
    problem = build_problem(config)
    weights_init = init_weights(int(config["depth"]), int(config["width"]), int(config["seed"]))

    per_noise: Dict[str, Dict[str, object]] = {}
    for noise_label, noise_frac in zip(config["noise_labels"], config["noise_levels"]):
        noise_seed = deterministic_noise_seed(int(config["seed"]), float(noise_frac))
        run = train_muon_with_noise(
            weights=weights_init,
            X=problem["X"],
            Y=problem["Y"],
            num_steps=int(config["num_steps"]),
            lr=float(config["lr_muon"]),
            noise_frac=float(noise_frac),
            ns_iters=int(config["ns_iters"]),
            seed=int(config["seed"]),
            track_every=int(config["track_every"]),
            noise_seed=noise_seed,
        )
        run["noise_label"] = noise_label
        per_noise[noise_label] = run

    summary_rows = build_summary_rows(config, per_noise)
    verdict = build_verdict(config, per_noise)
    runtime_sec = float(time.perf_counter() - start_time)

    results = {
        "config": config,
        "data_diagnostics": {
            "target_condition_number": problem["target_condition_number"],
            "target_singular_values": problem["sigma"].tolist(),
            "input_fro_norm": problem["input_fro_norm"],
            "target_fro_norm": problem["target_fro_norm"],
            "target_output_fro_norm": problem["target_output_fro_norm"],
        },
        "runtime_sec": runtime_sec,
        "per_noise": per_noise,
        "summary_rows": summary_rows,
        "verdict": verdict,
    }

    if emit_report:
        print(format_experiment_report(results))

    return results


def main() -> int:
    """CLI entrypoint preserving normal script behavior."""
    results = run_experiment(emit_report=False)
    print(format_experiment_report(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
