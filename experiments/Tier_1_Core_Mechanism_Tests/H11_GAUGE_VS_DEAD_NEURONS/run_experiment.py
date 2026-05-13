#!/usr/bin/env python3
"""
H11: Gauge fraction vs batch-dead neuron fraction (toy probe)
=================================================================

This experiment trains a 6-layer width-32 ReLU MLP on synthetic regression data.
It sweeps a constant bias initialization value to change how many hidden units are
batch-dead on the sampled training inputs X. At the final checkpoint it measures:

  1) hidden-layer batch-dead fraction
  2) per-layer gauge fraction of the weight gradient via Stiefel normal projection

Important scope / limitations
-----------------------------
- "Dead" means dead on the sampled batch X, not globally over input space.
- The primary dead-vs-gauge analysis uses hidden layers only.
- The last layer has no ReLU and is reported separately as a control.
- If ||G||_F^2 < GRAD_NORM_THRESHOLD, gauge fraction is reported as NaN.
- This script does not run Muon and does not measure Muon benefit directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

EXPERIMENT_NAME = "H11_GAUGE_VS_DEAD_NEURONS"
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent

DIM = 32
NUM_LAYERS = 6
NUM_SAMPLES = 100
NUM_STEPS = 100
LR = 0.003
MOMENTUM = 0.9
NUM_SEEDS = 5
BASE_SEED = 42
SEED_STRIDE = 31
BIAS_INITS = [+2.0, +1.0, 0.0, -1.0, -2.0, -3.0, -5.0]
GRAD_NORM_THRESHOLD = 1e-10
LOSS_BLOWUP_THRESHOLD = 1e10

LAYER_NAMES = [f"L{i+1}" for i in range(NUM_LAYERS)]
HIDDEN_LAYER_NAMES = LAYER_NAMES[:-1]
LAST_LAYER_NAME = LAYER_NAMES[-1]
DEFAULT_DEAD_GAUGE_BINS = [(0, 10), (10, 30), (30, 50), (50, 70), (70, 90), (90, 100)]


def get_default_config() -> Dict[str, Any]:
    """Return a fresh default config dict."""
    return {
        "dim": DIM,
        "num_layers": NUM_LAYERS,
        "num_samples": NUM_SAMPLES,
        "num_steps": NUM_STEPS,
        "lr": LR,
        "momentum": MOMENTUM,
        "num_seeds": NUM_SEEDS,
        "base_seed": BASE_SEED,
        "seed_stride": SEED_STRIDE,
        "bias_inits": list(BIAS_INITS),
        "grad_norm_threshold": GRAD_NORM_THRESHOLD,
        "loss_blowup_threshold": LOSS_BLOWUP_THRESHOLD,
        "input_scale": 0.3,
        "target_scale": 0.3,
        "dead_gauge_bins": list(DEFAULT_DEAD_GAUGE_BINS),
    }


def prepare_config(config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Merge user overrides into the default config."""
    config = get_default_config()
    if config_overrides:
        config.update(config_overrides)
    config["bias_inits"] = list(config["bias_inits"])
    config["dead_gauge_bins"] = [tuple(x) for x in config["dead_gauge_bins"]]
    return config


def build_seed_list(config: Dict[str, Any]) -> List[int]:
    return [config["base_seed"] + idx * config["seed_stride"] for idx in range(config["num_seeds"])]


# =============================================================================
# RELU MLP WITH BIASES
# =============================================================================


def init_weights(rng: np.random.RandomState, bias_init_val: float, dim: int = DIM,
                 num_layers: int = NUM_LAYERS) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Initialize an MLP with He-initialized weights and constant biases."""
    weights: List[np.ndarray] = []
    biases: List[np.ndarray] = []
    for _ in range(num_layers):
        std = np.sqrt(2.0 / dim)
        W = rng.randn(dim, dim) * std
        b = np.full(dim, bias_init_val)
        weights.append(W.copy())
        biases.append(b.copy())
    return weights, biases


def forward(weights: List[np.ndarray], biases: List[np.ndarray], X: np.ndarray):
    """Forward pass with ReLU on all but the last layer."""
    out = X.copy()
    pre_acts = []
    post_acts = [X.copy()]
    relu_masks = []

    for idx in range(len(weights)):
        pre = weights[idx] @ out + biases[idx][:, None]
        pre_acts.append(pre.copy())
        if idx < len(weights) - 1:
            mask = (pre > 0).astype(float)
            relu_masks.append(mask)
            out = pre * mask
        else:
            relu_masks.append(np.ones_like(pre))
            out = pre
        post_acts.append(out.copy())

    return out, pre_acts, post_acts, relu_masks


def compute_loss(weights: List[np.ndarray], biases: List[np.ndarray], X: np.ndarray,
                 Y: np.ndarray) -> float:
    pred, _, _, _ = forward(weights, biases, X)
    diff = pred - Y
    return float(0.5 * np.mean(np.sum(diff ** 2, axis=0)))


def compute_gradients(weights: List[np.ndarray], biases: List[np.ndarray], X: np.ndarray,
                      Y: np.ndarray):
    """Backprop through biased ReLU MLP. Returns weight grads and bias grads."""
    num_layers = len(weights)
    N = X.shape[1]

    pred, pre_acts, post_acts, relu_masks = forward(weights, biases, X)
    delta = (pred - Y) / N

    w_grads = [None] * num_layers
    b_grads = [None] * num_layers

    for l in range(num_layers - 1, -1, -1):
        w_grads[l] = delta @ post_acts[l].T
        b_grads[l] = np.sum(delta, axis=1)
        if l > 0:
            delta = weights[l].T @ delta
            delta = delta * relu_masks[l - 1]

    return w_grads, b_grads


# =============================================================================
# GAUGE DECOMPOSITION
# =============================================================================


def compute_polar_factor(W: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(W, full_matrices=True)
    return U @ Vt


def gauge_decomposition(G: np.ndarray, W: np.ndarray,
                        grad_norm_threshold: float = GRAD_NORM_THRESHOLD) -> float:
    """
    Decompose gradient G into tangent and normal components at Q = polar(W).

    Returns NaN if the squared Frobenius norm of G is below threshold or non-finite.
    """
    G_norm_sq = float(np.sum(G ** 2))
    if (not np.isfinite(G_norm_sq)) or G_norm_sq < grad_norm_threshold:
        return float("nan")

    if not np.all(np.isfinite(W)) or not np.all(np.isfinite(G)):
        return float("nan")

    Q = compute_polar_factor(W)
    QtG = Q.T @ G
    sym_QtG = 0.5 * (QtG + QtG.T)
    G_normal = Q @ sym_QtG
    G_normal_norm_sq = float(np.sum(G_normal ** 2))
    return G_normal_norm_sq / G_norm_sq


# =============================================================================
# DEAD-NEURON MEASUREMENT
# =============================================================================


def measure_dead_fraction(weights: List[np.ndarray], biases: List[np.ndarray],
                          X: np.ndarray) -> list[float]:
    """
    Measure hidden-layer batch-dead fraction on the sampled inputs X.

    A hidden neuron is counted as dead if its pre-activation is <= 0 for every
    sample in X, equivalently its ReLU output is zero on the whole sampled batch.
    """
    _, pre_acts, _, _ = forward(weights, biases, X)
    dead_fractions = []
    for l in range(len(weights) - 1):
        activations = pre_acts[l]
        alive_mask = np.any(activations > 0, axis=1)
        dead_frac = 1.0 - np.mean(alive_mask)
        dead_fractions.append(float(dead_frac))
    return dead_fractions


# =============================================================================
# TRAINING LOOP
# =============================================================================


def train_and_measure(weights_init: List[np.ndarray], biases_init: List[np.ndarray],
                      X: np.ndarray, Y: np.ndarray, n_steps: int = NUM_STEPS,
                      lr: float = LR, momentum: float = MOMENTUM,
                      grad_norm_threshold: float = GRAD_NORM_THRESHOLD,
                      loss_blowup_threshold: float = LOSS_BLOWUP_THRESHOLD) -> Dict[str, Any]:
    """
    Train with full-batch momentum SGD and measure at the final checkpoint.

    Returned fields include hidden batch-dead fractions, per-layer gauge fractions,
    final loss, and early-stop diagnostics.
    """
    weights = [w.copy() for w in weights_init]
    biases = [b.copy() for b in biases_init]
    w_vel = [np.zeros_like(w) for w in weights]
    b_vel = [np.zeros_like(b) for b in biases]

    stopped_early = False
    break_reason = None
    steps_completed = 0
    last_pre_update_loss = None

    for step in range(n_steps):
        loss = compute_loss(weights, biases, X, Y)
        last_pre_update_loss = float(loss)
        if np.isnan(loss):
            stopped_early = True
            break_reason = "loss_nan"
            break
        if loss > loss_blowup_threshold:
            stopped_early = True
            break_reason = "loss_blowup"
            break

        w_grads, b_grads = compute_gradients(weights, biases, X, Y)

        for i in range(len(weights)):
            w_vel[i] = momentum * w_vel[i] + w_grads[i]
            b_vel[i] = momentum * b_vel[i] + b_grads[i]
            weights[i] = weights[i] - lr * w_vel[i]
            biases[i] = biases[i] - lr * b_vel[i]

        steps_completed = step + 1

    dead_fracs = measure_dead_fraction(weights, biases, X)
    w_grads, _ = compute_gradients(weights, biases, X, Y)
    gauge_fracs = [
        gauge_decomposition(G=w_grads[l], W=weights[l], grad_norm_threshold=grad_norm_threshold)
        for l in range(len(weights))
    ]
    final_loss = compute_loss(weights, biases, X, Y)

    return {
        "dead_fractions": np.array(dead_fracs, dtype=float),
        "gauge_fractions": np.array(gauge_fracs, dtype=float),
        "final_loss": float(final_loss),
        "steps_completed": int(steps_completed),
        "stopped_early": bool(stopped_early),
        "break_reason": break_reason,
        "last_pre_update_loss": None if last_pre_update_loss is None else float(last_pre_update_loss),
    }


# =============================================================================
# EXPERIMENT EXECUTION
# =============================================================================


def _format_percent(value: float) -> str:
    if value is None or np.isnan(value):
        return "NaN"
    return f"{value:.1f}%"


def _safe_nanmean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return float("nan")
    return float(np.mean(valid))


def _safe_nanstd(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return float("nan")
    return float(np.std(valid))


def _to_float_list(values: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(values, dtype=float).tolist()]


def run_experiment(config_overrides: Optional[Dict[str, Any]] = None,
                   verbose: bool = False) -> Dict[str, Any]:
    """Run the H11 toy probe and return structured raw results."""
    config = prepare_config(config_overrides)
    seeds = build_seed_list(config)
    layer_names = [f"L{i+1}" for i in range(config["num_layers"])]
    hidden_layer_names = layer_names[:-1]

    if verbose:
        print("=" * 90)
        print("H11: GAUGE FRACTION vs BATCH-DEAD HIDDEN FRACTION (toy probe)")
        print("=" * 90)
        print(f"Architecture: {config['num_layers']}-layer ReLU MLP, width={config['dim']}")
        print(f"Bias init values: {config['bias_inits']}")
        print(f"Steps: {config['num_steps']}, Seeds: {len(seeds)}")
        print("Primary summary uses hidden layers only; last layer reported separately as control.")
        print("NaN gauge = squared gradient Frobenius norm below threshold or non-finite.")
        print("=" * 90)

    condition_results = []
    results_by_bias = {}
    run_records = []

    for bias_val in config["bias_inits"]:
        dead_all = []
        gauge_all = []
        loss_all = []
        steps_all = []
        stopped_all = []
        break_reasons = []
        last_pre_update_losses = []
        per_seed_records = []

        for run_seed in seeds:
            rng = np.random.RandomState(run_seed)
            X = rng.randn(config["dim"], config["num_samples"]) * config["input_scale"]
            Y = rng.randn(config["dim"], config["num_samples"]) * config["target_scale"]

            weights_init, biases_init = init_weights(
                rng=rng,
                bias_init_val=bias_val,
                dim=config["dim"],
                num_layers=config["num_layers"],
            )
            run_result = train_and_measure(
                weights_init=weights_init,
                biases_init=biases_init,
                X=X,
                Y=Y,
                n_steps=config["num_steps"],
                lr=config["lr"],
                momentum=config["momentum"],
                grad_norm_threshold=config["grad_norm_threshold"],
                loss_blowup_threshold=config["loss_blowup_threshold"],
            )

            dead_fracs = run_result["dead_fractions"]
            gauge_fracs = run_result["gauge_fractions"]
            hidden_gauge_fracs = gauge_fracs[:-1]
            last_layer_gauge = gauge_fracs[-1]
            seed_mean_hidden_dead = float(np.mean(dead_fracs))
            seed_mean_hidden_gauge = _safe_nanmean(hidden_gauge_fracs)

            seed_record = {
                "bias_init": float(bias_val),
                "seed": int(run_seed),
                "hidden_dead_fractions": _to_float_list(dead_fracs),
                "hidden_gauge_fractions": _to_float_list(hidden_gauge_fracs),
                "all_layer_gauge_fractions": _to_float_list(gauge_fracs),
                "last_layer_gauge_fraction": float(last_layer_gauge),
                "mean_hidden_dead_fraction": float(seed_mean_hidden_dead),
                "mean_hidden_gauge_fraction": float(seed_mean_hidden_gauge),
                "final_loss": float(run_result["final_loss"]),
                "steps_completed": int(run_result["steps_completed"]),
                "stopped_early": bool(run_result["stopped_early"]),
                "break_reason": run_result["break_reason"],
                "last_pre_update_loss": run_result["last_pre_update_loss"],
                "valid_hidden_gauge_count": int(np.sum(~np.isnan(hidden_gauge_fracs))),
                "nan_hidden_gauge_count": int(np.sum(np.isnan(hidden_gauge_fracs))),
                "valid_total_gauge_count": int(np.sum(~np.isnan(gauge_fracs))),
                "nan_total_gauge_count": int(np.sum(np.isnan(gauge_fracs))),
            }

            dead_all.append(dead_fracs)
            gauge_all.append(gauge_fracs)
            loss_all.append(run_result["final_loss"])
            steps_all.append(run_result["steps_completed"])
            stopped_all.append(run_result["stopped_early"])
            break_reasons.append(run_result["break_reason"])
            last_pre_update_losses.append(run_result["last_pre_update_loss"])
            per_seed_records.append(seed_record)
            run_records.append(seed_record)

        dead_arr = np.array(dead_all, dtype=float)
        gauge_arr = np.array(gauge_all, dtype=float)
        hidden_gauge_arr = gauge_arr[:, :-1]
        last_layer_arr = gauge_arr[:, -1]

        condition_result = {
            "bias_init": float(bias_val),
            "seeds": list(seeds),
            "dead": dead_arr,
            "gauge": gauge_arr,
            "hidden_gauge": hidden_gauge_arr,
            "last_layer_gauge": last_layer_arr,
            "loss": np.array(loss_all, dtype=float),
            "steps_completed": np.array(steps_all, dtype=int),
            "stopped_early": np.array(stopped_all, dtype=bool),
            "break_reasons": list(break_reasons),
            "last_pre_update_loss": np.array([
                np.nan if value is None else value for value in last_pre_update_losses
            ], dtype=float),
            "per_seed": per_seed_records,
        }
        condition_results.append(condition_result)
        results_by_bias[float(bias_val)] = condition_result

        if verbose:
            hidden_dead_mean = float(np.mean(dead_arr) * 100)
            hidden_gauge_mean = _safe_nanmean(hidden_gauge_arr) * 100
            hidden_nan_count = int(np.sum(np.isnan(hidden_gauge_arr)))
            print(
                f"  bias_init={bias_val:+.0f}: "
                f"hidden_dead={hidden_dead_mean:.1f}%, "
                f"hidden_gauge={_format_percent(hidden_gauge_mean)}, "
                f"hidden_NaNs={hidden_nan_count}, "
                f"L6={_format_percent(_safe_nanmean(last_layer_arr) * 100)}, "
                f"loss={np.mean(loss_all):.4f}, "
                f"early_stops={int(np.sum(stopped_all))}/{len(stopped_all)}"
            )

    return {
        "experiment_name": EXPERIMENT_NAME,
        "script_path": str(SCRIPT_PATH),
        "script_dir": str(SCRIPT_DIR),
        "config": config,
        "seeds": seeds,
        "layer_names": layer_names,
        "hidden_layer_names": hidden_layer_names,
        "last_layer_name": layer_names[-1],
        "condition_results": condition_results,
        "per_bias_results": condition_results,
        "conditions": condition_results,
        "results_by_bias": results_by_bias,
        "run_records": run_records,
    }


# =============================================================================
# ANALYSIS
# =============================================================================


def _compute_condition_level_correlation(dead_arr: np.ndarray, gauge_arr: np.ndarray) -> Dict[str, Any]:
    valid_mask = ~np.isnan(gauge_arr)
    dead_valid = dead_arr[valid_mask]
    gauge_valid = gauge_arr[valid_mask]

    result = {
        "valid_condition_count": int(len(dead_valid)),
        "total_condition_count": int(len(dead_arr)),
        "dead_valid_percent": _to_float_list(dead_valid),
        "gauge_valid_percent": _to_float_list(gauge_valid),
        "pearson_r": float("nan"),
        "slope": float("nan"),
        "status": "NO_VALID_CONDITIONS",
        "descriptive_only": True,
    }

    if len(dead_valid) < 2:
        result["status"] = "INSUFFICIENT_POINTS"
        return result

    if np.std(dead_valid) <= 1e-10 or np.std(gauge_valid) <= 1e-10:
        result["status"] = "INSUFFICIENT_VARIANCE"
        return result

    result["pearson_r"] = float(np.corrcoef(dead_valid, gauge_valid)[0, 1])
    result["slope"] = float(np.polyfit(dead_valid, gauge_valid, 1)[0])
    if len(dead_valid) < 3:
        result["status"] = "DESCRIPTIVE_ONLY"
        result["descriptive_only"] = True
    else:
        result["status"] = "OK"
        result["descriptive_only"] = False
    return result


def _build_bin_summary(all_dead_points: np.ndarray, all_gauge_points: np.ndarray,
                       bins: List[tuple[int, int]]) -> List[Dict[str, Any]]:
    bin_summary = []
    for lo, hi in bins:
        if hi >= 100:
            mask = (all_dead_points >= lo) & (all_dead_points <= hi)
        else:
            mask = (all_dead_points >= lo) & (all_dead_points < hi)
        n = int(np.sum(mask))
        if n > 0:
            mean_gauge = float(np.mean(all_gauge_points[mask]))
            std_gauge = float(np.std(all_gauge_points[mask]))
        else:
            mean_gauge = float("nan")
            std_gauge = float("nan")
        bin_summary.append({
            "dead_bin": (int(lo), int(hi)),
            "dead_bin_label": f"{lo}-{hi}%",
            "n_points": n,
            "mean_gauge_percent": mean_gauge,
            "std_gauge_percent": std_gauge,
        })
    return bin_summary


def analyze_results(experiment_results: Dict[str, Any]) -> Dict[str, Any]:
    """Compute summary tables, pooled analyses, and H1-H4 outcomes."""
    config = experiment_results["config"]
    condition_results = experiment_results["condition_results"]
    num_hidden_layers = config["num_layers"] - 1

    condition_summaries = []
    per_layer_dead_table = []
    per_layer_gauge_table = []
    pooled_hidden_points = []

    for condition in condition_results:
        bias_val = float(condition["bias_init"])
        dead = np.asarray(condition["dead"], dtype=float)
        gauge = np.asarray(condition["gauge"], dtype=float)
        hidden_gauge = np.asarray(condition["hidden_gauge"], dtype=float)
        last_layer_gauge = np.asarray(condition["last_layer_gauge"], dtype=float)
        loss = np.asarray(condition["loss"], dtype=float)
        steps_completed = np.asarray(condition["steps_completed"], dtype=int)
        stopped_early = np.asarray(condition["stopped_early"], dtype=bool)
        per_seed = condition["per_seed"]

        mean_hidden_dead_per_seed = np.mean(dead, axis=1) * 100
        mean_hidden_gauge_per_seed = np.array([
            _safe_nanmean(hidden_gauge[seed_idx]) * 100 for seed_idx in range(hidden_gauge.shape[0])
        ], dtype=float)
        valid_hidden = hidden_gauge[~np.isnan(hidden_gauge)]
        valid_last = last_layer_gauge[~np.isnan(last_layer_gauge)]

        summary = {
            "bias_init": bias_val,
            "mean_hidden_dead_percent": float(np.mean(dead) * 100),
            "seed_mean_hidden_dead_percent": _to_float_list(mean_hidden_dead_per_seed),
            "mean_hidden_gauge_percent": _safe_nanmean(hidden_gauge) * 100,
            "seed_mean_hidden_gauge_percent": _to_float_list(mean_hidden_gauge_per_seed),
            "valid_hidden_gauge_count": int(np.sum(~np.isnan(hidden_gauge))),
            "nan_hidden_gauge_count": int(np.sum(np.isnan(hidden_gauge))),
            "mean_last_layer_gauge_percent": _safe_nanmean(last_layer_gauge) * 100,
            "seed_last_layer_gauge_percent": _to_float_list(last_layer_gauge * 100),
            "valid_last_layer_count": int(np.sum(~np.isnan(last_layer_gauge))),
            "nan_last_layer_count": int(np.sum(np.isnan(last_layer_gauge))),
            "mean_loss": float(np.mean(loss)),
            "std_loss": float(np.std(loss)),
            "mean_steps_completed": float(np.mean(steps_completed)),
            "num_stopped_early": int(np.sum(stopped_early)),
            "stopped_early_seeds": [record["seed"] for record in per_seed if record["stopped_early"]],
        }
        condition_summaries.append(summary)

        per_layer_dead_table.append({
            "bias_init": bias_val,
            "layer_dead_percent": _to_float_list(np.mean(dead, axis=0) * 100),
            "mean_hidden_dead_percent": float(np.mean(dead) * 100),
        })

        per_layer_gauge_means = []
        for layer_idx in range(config["num_layers"]):
            layer_values = gauge[:, layer_idx]
            per_layer_gauge_means.append(_safe_nanmean(layer_values) * 100)
        per_layer_gauge_table.append({
            "bias_init": bias_val,
            "layer_gauge_percent": per_layer_gauge_means,
            "mean_hidden_gauge_percent": _safe_nanmean(hidden_gauge) * 100,
            "mean_last_layer_gauge_percent": _safe_nanmean(last_layer_gauge) * 100,
        })

        for seed_idx, seed in enumerate(condition["seeds"]):
            for layer_idx in range(num_hidden_layers):
                gauge_value = hidden_gauge[seed_idx, layer_idx]
                if np.isnan(gauge_value):
                    continue
                pooled_hidden_points.append({
                    "bias_init": bias_val,
                    "seed": int(seed),
                    "layer_index": int(layer_idx),
                    "layer_name": experiment_results["hidden_layer_names"][layer_idx],
                    "dead_percent": float(dead[seed_idx, layer_idx] * 100),
                    "gauge_percent": float(gauge_value * 100),
                })

    summary_dead = np.array([item["mean_hidden_dead_percent"] for item in condition_summaries], dtype=float)
    summary_gauge = np.array([item["mean_hidden_gauge_percent"] for item in condition_summaries], dtype=float)
    condition_level = _compute_condition_level_correlation(summary_dead, summary_gauge)

    all_dead_points = np.array([item["dead_percent"] for item in pooled_hidden_points], dtype=float)
    all_gauge_points = np.array([item["gauge_percent"] for item in pooled_hidden_points], dtype=float)

    pooled_summary = {
        "n_points": int(len(all_dead_points)),
        "pearson_r": float("nan"),
        "status": "INSUFFICIENT_POINTS",
        "dead_points_percent": _to_float_list(all_dead_points),
        "gauge_points_percent": _to_float_list(all_gauge_points),
        "points": pooled_hidden_points,
        "bin_summary": _build_bin_summary(all_dead_points, all_gauge_points, config["dead_gauge_bins"]),
    }
    if len(all_dead_points) >= 2 and np.std(all_dead_points) > 1e-10 and np.std(all_gauge_points) > 1e-10:
        pooled_summary["pearson_r"] = float(np.corrcoef(all_dead_points, all_gauge_points)[0, 1])
        pooled_summary["status"] = "OK"
    elif len(all_dead_points) >= 2:
        pooled_summary["status"] = "INSUFFICIENT_VARIANCE"

    last_layer_control = []
    for summary in condition_summaries:
        last_layer_control.append({
            "bias_init": summary["bias_init"],
            "mean_last_layer_gauge_percent": summary["mean_last_layer_gauge_percent"],
            "valid_last_layer_count": summary["valid_last_layer_count"],
            "nan_last_layer_count": summary["nan_last_layer_count"],
        })

    valid_hidden_gauges = [item["mean_hidden_gauge_percent"] for item in condition_summaries
                           if not np.isnan(item["mean_hidden_gauge_percent"])]

    h1_result = len(valid_hidden_gauges) > 0 and all(g > 30.0 for g in valid_hidden_gauges)
    h2_result = None
    h2_spread = float("nan")
    if len(valid_hidden_gauges) >= 3:
        gauge_valid_arr = np.array(valid_hidden_gauges, dtype=float)
        h2_spread = float(gauge_valid_arr.max() - gauge_valid_arr.min())
        h2_result = bool(h2_spread < 15.0)

    fully_dead_biases = [
        float(condition["bias_init"])
        for condition in condition_results
        if float(np.mean(condition["dead"])) > 0.95
    ]
    h3_result = None
    if fully_dead_biases:
        all_nan_for_dead = True
        for condition in condition_results:
            if float(condition["bias_init"]) not in fully_dead_biases:
                continue
            hidden_gauge = np.asarray(condition["hidden_gauge"], dtype=float)
            if np.any(~np.isnan(hidden_gauge)):
                all_nan_for_dead = False
        h3_result = bool(all_nan_for_dead)

    h4_result = None
    if pooled_summary["n_points"] >= 5 and np.isfinite(pooled_summary["pearson_r"]):
        h4_result = bool(abs(pooled_summary["pearson_r"]) < 0.5)

    test_records = [
        {
            "name": "H1",
            "question": "Is hidden-layer gauge substantial where gradients are non-trivial?",
            "result": h1_result,
            "status": "PASS" if bool(h1_result) else "FAIL",
            "details": {
                "valid_hidden_gauge_condition_means": valid_hidden_gauges,
                "criterion": "> 30% for all valid hidden-layer condition means",
            },
        },
        {
            "name": "H2",
            "question": "Is hidden-layer gauge approximately constant across valid conditions?",
            "result": h2_result,
            "status": "SKIPPED" if h2_result is None else ("PASS" if bool(h2_result) else "FAIL"),
            "details": {
                "spread_percent_points": h2_spread,
                "valid_condition_count": len(valid_hidden_gauges),
                "criterion": "spread < 15 percentage points; require at least 3 valid conditions",
            },
        },
        {
            "name": "H3",
            "question": "Do near-fully-dead hidden conditions yield NaN hidden-layer gauge?",
            "result": h3_result,
            "status": "SKIPPED" if h3_result is None else ("PASS" if bool(h3_result) else "FAIL"),
            "details": {
                "fully_dead_biases": fully_dead_biases,
                "criterion": "all hidden-layer gauges NaN when mean hidden dead fraction > 95%",
            },
        },
        {
            "name": "H4",
            "question": "Is pooled hidden dead-vs-gauge correlation weak (|r| < 0.5)?",
            "result": h4_result,
            "status": "SKIPPED" if h4_result is None else ("PASS" if bool(h4_result) else "FAIL"),
            "details": {
                "pearson_r": pooled_summary["pearson_r"],
                "n_points": pooled_summary["n_points"],
                "criterion": "|r| < 0.5 with at least 5 valid hidden-layer points",
            },
        },
    ]

    total_pass = sum(bool(test["result"]) for test in test_records if test["result"] is not None)
    total_tests = sum(1 for test in test_records if test["result"] is not None)

    calibrated_conclusion = [
        "This is a toy probe of batch-dead hidden units versus gauge fraction in the raw gradient.",
        "The primary summaries use hidden layers only; the last layer is a separate control.",
        "NaN gauge values indicate hidden layers whose gradients are too small (or non-finite) for a reliable decomposition.",
        "Condition-level conclusions are descriptive only when fewer than 3 bias conditions have valid hidden-layer gauge summaries.",
        "This experiment does not run Muon and does not directly measure Muon benefit.",
    ]

    return {
        "condition_summaries": condition_summaries,
        "summary_dead_percent": _to_float_list(summary_dead),
        "summary_hidden_gauge_percent": _to_float_list(summary_gauge),
        "per_layer_dead_table": per_layer_dead_table,
        "per_layer_gauge_table": per_layer_gauge_table,
        "condition_level": condition_level,
        "pooled_hidden_summary": pooled_summary,
        "last_layer_control": last_layer_control,
        "tests": {
            "records": test_records,
            "total_pass": int(total_pass),
            "total_tests": int(total_tests),
        },
        "calibrated_conclusion": calibrated_conclusion,
    }


# =============================================================================
# REPORTING
# =============================================================================


def _print_condition_summary_table(analysis: Dict[str, Any]) -> None:
    print(f"\n\n{'=' * 110}")
    print("SUMMARY TABLE (primary analysis uses hidden layers only)")
    print(f"{'=' * 110}")
    header = (
        f"\n{'Bias':>6}  {'Hidden dead %':>13}  {'Hidden gauge %':>15}  "
        f"{'Hidden NaNs':>11}  {'L6 gauge %':>11}  {'Loss':>12}  {'Early stops':>11}"
    )
    print(header)
    print("-" * 92)
    for row in analysis["condition_summaries"]:
        print(
            f"{row['bias_init']:>+6.0f}  "
            f"{row['mean_hidden_dead_percent']:>13.1f}  "
            f"{_format_percent(row['mean_hidden_gauge_percent']):>15}  "
            f"{row['nan_hidden_gauge_count']:>11}  "
            f"{_format_percent(row['mean_last_layer_gauge_percent']):>11}  "
            f"{row['mean_loss']:>12.4f}  "
            f"{row['num_stopped_early']:>11}"
        )


def _print_per_layer_tables(experiment_results: Dict[str, Any], analysis: Dict[str, Any]) -> None:
    print(f"\n\n{'=' * 110}")
    print("PER-LAYER HIDDEN DEAD FRACTION (%, mean over seeds)")
    print(f"{'=' * 110}")
    print(f"\n{'Bias':>6}", end="")
    for name in experiment_results["hidden_layer_names"]:
        print(f"  {name:>8}", end="")
    print(f"  {'Mean':>8}")
    print("-" * (8 + 10 * len(experiment_results["hidden_layer_names"]) + 10))
    for row in analysis["per_layer_dead_table"]:
        print(f"{row['bias_init']:>+6.0f}", end="")
        for value in row["layer_dead_percent"]:
            print(f"  {value:>8.1f}", end="")
        print(f"  {row['mean_hidden_dead_percent']:>8.1f}")

    print(f"\n\n{'=' * 110}")
    print("PER-LAYER GAUGE FRACTION (%, mean over seeds, NaN = unreliable / zero-gradient regime)")
    print(f"{'=' * 110}")
    print(f"\n{'Bias':>6}", end="")
    for name in experiment_results["layer_names"]:
        print(f"  {name:>8}", end="")
    print(f"  {'Hidden mean':>12}  {'L6':>8}")
    print("-" * (8 + 10 * len(experiment_results["layer_names"]) + 24))
    for row in analysis["per_layer_gauge_table"]:
        print(f"{row['bias_init']:>+6.0f}", end="")
        for value in row["layer_gauge_percent"]:
            print(f"  {_format_percent(value):>8}", end="")
        print(
            f"  {_format_percent(row['mean_hidden_gauge_percent']):>12}"
            f"  {_format_percent(row['mean_last_layer_gauge_percent']):>8}"
        )


def _print_condition_level_analysis(analysis: Dict[str, Any]) -> None:
    level = analysis["condition_level"]
    print(f"\n\n{'=' * 110}")
    print("CONDITION-LEVEL DEAD-vs-GAUGE ANALYSIS (hidden layers only)")
    print(f"{'=' * 110}")
    print(f"\nValid conditions: {level['valid_condition_count']}/{level['total_condition_count']}")
    if np.isfinite(level["pearson_r"]):
        print(f"Pearson r: {level['pearson_r']:.4f}")
        print(f"Slope:     {level['slope']:.4f} gauge-pp per dead-pp")
    else:
        print("Pearson r: NaN")
        print("Slope:     NaN")
    print(f"Status:    {level['status']}")
    if level["status"] == "DESCRIPTIVE_ONLY":
        print("Note: fewer than 3 valid conditions; treat this only as descriptive.")


def _print_pooled_hidden_analysis(analysis: Dict[str, Any]) -> None:
    pooled = analysis["pooled_hidden_summary"]
    print(f"\n\n{'=' * 110}")
    print("POOLED HIDDEN-LAYER DEAD-vs-GAUGE ANALYSIS")
    print(f"{'=' * 110}")
    print(f"\nValid hidden-layer points: {pooled['n_points']}")
    if np.isfinite(pooled["pearson_r"]):
        print(f"Pearson r: {pooled['pearson_r']:.4f}")
    else:
        print(f"Pearson r: NaN ({pooled['status']})")

    print(f"\n{'Dead % bin':>12}  {'N':>4}  {'Mean gauge %':>13}  {'Std gauge %':>12}")
    print(f"  {'-' * 50}")
    for row in pooled["bin_summary"]:
        if row["n_points"] > 0:
            print(
                f"{row['dead_bin_label']:>12}  {row['n_points']:>4}  "
                f"{row['mean_gauge_percent']:>13.1f}  {row['std_gauge_percent']:>12.1f}"
            )
        else:
            print(f"{row['dead_bin_label']:>12}  {0:>4}  {'---':>13}  {'---':>12}")


def _print_last_layer_control(analysis: Dict[str, Any], last_layer_name: str) -> None:
    print(f"\n\n{'=' * 110}")
    print(f"LAST-LAYER CONTROL ({last_layer_name}; no ReLU after it)")
    print(f"{'=' * 110}")
    print(f"\n{'Bias':>6}  {'Gauge %':>10}  {'Valid seeds':>12}  {'NaN seeds':>10}")
    print("-" * 48)
    for row in analysis["last_layer_control"]:
        print(
            f"{row['bias_init']:>+6.0f}  "
            f"{_format_percent(row['mean_last_layer_gauge_percent']):>10}  "
            f"{row['valid_last_layer_count']:>12}  "
            f"{row['nan_last_layer_count']:>10}"
        )


def _print_tests(analysis: Dict[str, Any]) -> None:
    print(f"\n\n{'=' * 110}")
    print("HYPOTHESIS CHECKS")
    print(f"{'=' * 110}")
    for test in analysis["tests"]["records"]:
        print(f"\n{test['name']}: {test['question']}")
        print(f"    --> {test['status']}")
        details = test["details"]
        if test["name"] == "H1":
            values = details["valid_hidden_gauge_condition_means"]
            print(f"    Valid hidden-layer condition means: {[f'{v:.1f}%' for v in values]}")
        elif test["name"] == "H2":
            spread = details["spread_percent_points"]
            if np.isfinite(spread):
                print(f"    Spread: {spread:.1f}pp")
            print(f"    Valid conditions: {details['valid_condition_count']}")
        elif test["name"] == "H3":
            print(f"    Fully dead biases: {details['fully_dead_biases']}")
        elif test["name"] == "H4":
            pearson_r = details["pearson_r"]
            if np.isfinite(pearson_r):
                print(f"    r = {pearson_r:.4f}, N = {details['n_points']}")
            else:
                print(f"    r = NaN, N = {details['n_points']}")

    print(
        f"\nTests passed: {analysis['tests']['total_pass']}/"
        f"{analysis['tests']['total_tests']}"
    )


def _print_conclusion(analysis: Dict[str, Any]) -> None:
    print(f"\n\n{'=' * 110}")
    print("CALIBRATED CONCLUSION")
    print(f"{'=' * 110}")
    for bullet in analysis["calibrated_conclusion"]:
        print(f"- {bullet}")


def print_report(experiment_results: Dict[str, Any], analysis: Dict[str, Any]) -> None:
    """Print the full report corresponding to the current analysis."""
    _print_condition_summary_table(analysis)
    _print_per_layer_tables(experiment_results, analysis)
    _print_condition_level_analysis(analysis)
    _print_pooled_hidden_analysis(analysis)
    _print_last_layer_control(analysis, experiment_results["last_layer_name"])
    _print_tests(analysis)
    _print_conclusion(analysis)


def main() -> None:
    experiment_results = run_experiment(verbose=True)
    analysis = analyze_results(experiment_results)
    print_report(experiment_results, analysis)


if __name__ == "__main__":
    main()
