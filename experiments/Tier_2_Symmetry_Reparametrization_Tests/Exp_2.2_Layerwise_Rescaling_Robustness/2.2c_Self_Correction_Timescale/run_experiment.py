#!/usr/bin/env python3
"""
2.2c: Toy study of initial momentum-imbalance compression under layerwise rescaling
===================================================================================

This file preserves the original 2.2c toy setup: a 4-layer 32x32 deep linear
regression problem trained with a Muon-like momentum + Newton-Schulz update under
scalar gauge rescaling `c`.

What is actually measured here:
  - momentum imbalance ratio at each recorded step:
      max_l ||m_l||_F / min_l ||m_l||_F
  - a 50%-compression "half-life": the first recorded step where that ratio is
    at most half of its first recorded value
  - training onset: the first recorded pre-update loss index where the loss is
    < 0.99 * its initial pre-update value

Important caveats:
  - the 50%-compression metric is an operational proxy for early imbalance
    compression, not a full self-correction / equilibration timescale
  - loss_history[0] is recorded before the first optimizer update, while
    imbalance_history[0] is recorded after the first optimizer update, so onset
    and half-life live on slightly different operational clocks
  - this toy measurement alone does not estimate an RG fixed-point gap or prove
    broad scale-invariant asymptotics
"""

import os
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CONFIG = {
    "dim": 32,
    "n_layers": 4,
    "num_steps": 500,
    "lr": 0.02,
    "momentum_beta": 0.9,
    "ns_iters": 5,
    "num_seeds": 5,
    "batch_size": 128,
    "c_values": [1, 10, 100, 1000, 10000],
    "data_seed_base": 42,
    "data_seed_stride": 137,
    "weight_seed_base": 1000,
    "loss_abort_threshold": 1e10,
    "onset_drop_fraction": 0.99,
}


def get_default_config():
    config = dict(DEFAULT_CONFIG)
    config["c_values"] = list(DEFAULT_CONFIG["c_values"])
    return config


def compute_ratio(values):
    if not values:
        return float("nan")
    values = [float(v) for v in values]
    return float(max(values) / max(min(values), 1e-30))


def finite_values(values):
    return [float(v) for v in values if np.isfinite(v)]


def safe_mean(values):
    vals = finite_values(values)
    return float(np.mean(vals)) if vals else float("nan")


def safe_std(values):
    vals = finite_values(values)
    return float(np.std(vals)) if vals else float("nan")


def newton_schulz(matrix, n_iters):
    norm = np.linalg.norm(matrix, ord="fro")
    if norm < 1e-15:
        return matrix
    x = matrix / norm
    for _ in range(n_iters):
        gram = x.T @ x
        x = 1.5 * x - 0.5 * x @ gram
    return x


def init_weights(rng, c, dim, n_layers):
    weights = []
    for _ in range(n_layers):
        weight = rng.randn(dim, dim) / np.sqrt(dim)
        weights.append(weight)
    weights[0] = weights[0] * c
    weights[-1] = weights[-1] / c
    return weights


def forward(weights, x):
    out = x
    for weight in weights:
        out = weight @ out
    return out


def compute_gradients(weights, x, y):
    n_layers = len(weights)
    batch = x.shape[1]
    activations = [x]
    for weight in weights:
        activations.append(weight @ activations[-1])
    delta = 2.0 * (activations[-1] - y) / batch
    grads = [None] * n_layers
    for layer_idx in range(n_layers - 1, -1, -1):
        grads[layer_idx] = delta @ activations[layer_idx].T
        if layer_idx > 0:
            delta = weights[layer_idx].T @ delta
    return grads


def compute_loss(weights, x, y):
    pred = forward(weights, x)
    return float(np.mean((pred - y) ** 2))


def train_and_track(weights_init, x, y, config):
    weights = [weight.copy() for weight in weights_init]
    mom = [np.zeros_like(weight) for weight in weights]
    imbalance_history = []
    loss_history = []
    stop_reason = "completed_num_steps"

    for step in range(config["num_steps"]):
        loss = compute_loss(weights, x, y)
        loss_history.append(float(loss))
        if (not np.isfinite(loss)) or loss > config["loss_abort_threshold"]:
            stop_reason = "non_finite_or_exploded_loss"
            break

        grads = compute_gradients(weights, x, y)
        for layer_idx in range(len(weights)):
            mom[layer_idx] = (
                config["momentum_beta"] * mom[layer_idx]
                + (1.0 - config["momentum_beta"]) * grads[layer_idx]
            )
            ortho_mom = newton_schulz(mom[layer_idx], config["ns_iters"])
            weights[layer_idx] = weights[layer_idx] - config["lr"] * ortho_mom

        mom_norms = [np.linalg.norm(m, ord="fro") for m in mom]
        imbalance_history.append(compute_ratio(mom_norms))

    return {
        "imbalance_history": imbalance_history,
        "loss_history": loss_history,
        "num_recorded_loss_steps": len(loss_history),
        "num_recorded_imbalance_steps": len(imbalance_history),
        "stopped_early": stop_reason != "completed_num_steps",
        "stop_reason": stop_reason,
    }


def find_half_life(imbalance_history):
    """First recorded index where imbalance <= 0.5 * first recorded imbalance."""
    if not imbalance_history:
        return float("nan")
    initial = imbalance_history[0]
    target = initial * 0.5
    for idx, value in enumerate(imbalance_history):
        if value <= target:
            return float(idx)
    return float("nan")


def find_training_onset(loss_history, onset_drop_fraction):
    """First recorded pre-update index where loss < onset_drop_fraction * initial."""
    if not loss_history:
        return float("nan")
    initial = loss_history[0]
    for idx in range(1, len(loss_history)):
        if loss_history[idx] < onset_drop_fraction * initial:
            return float(idx)
    return float("nan")


def build_seed_schedule(config):
    schedule = []
    for seed_idx in range(config["num_seeds"]):
        schedule.append(
            {
                "seed_idx": seed_idx,
                "data_seed": config["data_seed_base"] + seed_idx * config["data_seed_stride"],
                "weight_seed": config["weight_seed_base"] + seed_idx,
            }
        )
    return schedule


def summarize_seed_results(c_value, seed_results):
    half_lives = [seed_result["half_life"] for seed_result in seed_results]
    onsets = [seed_result["training_onset"] for seed_result in seed_results]
    init_grad_imbalances = [seed_result["initial_gradient_imbalance"] for seed_result in seed_results]
    first_mom_imbalances = [seed_result["first_recorded_momentum_imbalance"] for seed_result in seed_results]
    final_mom_imbalances = [seed_result["final_momentum_imbalance"] for seed_result in seed_results]
    initial_losses = [seed_result["initial_loss"] for seed_result in seed_results]
    final_losses = [seed_result["final_loss"] for seed_result in seed_results]
    raw_index_flags = [
        seed_result["raw_index_onset_before_half_life"]
        for seed_result in seed_results
        if seed_result["raw_index_onset_before_half_life"] is not None
    ]

    return {
        "c": c_value,
        "num_seeds": len(seed_results),
        "finite_half_life_count": len(finite_values(half_lives)),
        "finite_onset_count": len(finite_values(onsets)),
        "mean_half_life": safe_mean(half_lives),
        "std_half_life": safe_std(half_lives),
        "mean_training_onset": safe_mean(onsets),
        "std_training_onset": safe_std(onsets),
        "mean_initial_gradient_imbalance": safe_mean(init_grad_imbalances),
        "std_initial_gradient_imbalance": safe_std(init_grad_imbalances),
        "mean_first_recorded_momentum_imbalance": safe_mean(first_mom_imbalances),
        "std_first_recorded_momentum_imbalance": safe_std(first_mom_imbalances),
        "mean_final_momentum_imbalance": safe_mean(final_mom_imbalances),
        "std_final_momentum_imbalance": safe_std(final_mom_imbalances),
        "mean_initial_loss": safe_mean(initial_losses),
        "mean_final_loss": safe_mean(final_losses),
        "mean_recorded_loss_steps": safe_mean(
            [seed_result["num_recorded_loss_steps"] for seed_result in seed_results]
        ),
        "mean_recorded_imbalance_steps": safe_mean(
            [seed_result["num_recorded_imbalance_steps"] for seed_result in seed_results]
        ),
        "any_early_stop": any(seed_result["stopped_early"] for seed_result in seed_results),
        "raw_index_onset_before_half_life_count": int(sum(raw_index_flags)),
        "raw_index_comparison_count": len(raw_index_flags),
    }


def compute_hypothesis_tests(aggregate_summary, c_values):
    by_c = {summary["c"]: summary for summary in aggregate_summary}

    log_cs = []
    half_lives = []
    for c_value in c_values:
        mean_half_life = by_c[c_value]["mean_half_life"]
        if c_value > 1 and np.isfinite(mean_half_life):
            log_cs.append(float(np.log10(c_value)))
            half_lives.append(float(mean_half_life))

    slope = float("nan")
    intercept = float("nan")
    r_squared = float("nan")
    if len(log_cs) >= 2:
        slope, intercept = np.polyfit(log_cs, half_lives, 1)
        preds = [slope * log_c + intercept for log_c in log_cs]
        ss_res = float(np.sum((np.array(half_lives) - np.array(preds)) ** 2))
        ss_tot = float(np.sum((np.array(half_lives) - np.mean(half_lives)) ** 2))
        r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0

    t1_legacy_pass = bool(np.isfinite(slope) and slope < 50.0)
    t1_monotone_support = bool(np.isfinite(slope) and 0.0 <= slope < 50.0)

    t2_control_cs = [c_value for c_value in c_values if c_value <= 1000]
    t2_per_c = []
    t2_finite_flags = []
    t2_strict_flags = []
    for c_value in t2_control_cs:
        mean_half_life = by_c[c_value]["mean_half_life"]
        finite = bool(np.isfinite(mean_half_life))
        under_50 = bool(finite and mean_half_life < 50.0)
        t2_per_c.append(
            {
                "c": c_value,
                "mean_half_life": mean_half_life,
                "finite": finite,
                "under_50": under_50,
            }
        )
        if finite:
            t2_finite_flags.append(under_50)
        t2_strict_flags.append(under_50)

    t2_legacy_finite_only_pass = bool(all(t2_finite_flags)) if t2_finite_flags else False
    t2_strict_all_reported_pass = bool(all(t2_strict_flags)) if t2_strict_flags else False
    t2_nonfinite_cs = [entry["c"] for entry in t2_per_c if not entry["finite"]]

    t3_per_c = []
    t3_valid_flags = []
    for c_value in c_values:
        mean_half_life = by_c[c_value]["mean_half_life"]
        mean_onset = by_c[c_value]["mean_training_onset"]
        comparable = bool(np.isfinite(mean_half_life) and np.isfinite(mean_onset))
        raw_flag = bool(mean_onset < mean_half_life) if comparable else None
        t3_per_c.append(
            {
                "c": c_value,
                "mean_training_onset": mean_onset,
                "mean_half_life": mean_half_life,
                "raw_index_onset_before_half_life": raw_flag,
                "comparable": comparable,
            }
        )
        if comparable:
            t3_valid_flags.append(raw_flag)

    t3_legacy_any_pass = bool(any(t3_valid_flags)) if t3_valid_flags else False
    t3_all_pass = bool(all(t3_valid_flags)) if t3_valid_flags else False
    t3_support_fraction = (
        float(sum(t3_valid_flags) / len(t3_valid_flags)) if t3_valid_flags else float("nan")
    )

    return {
        "T1": {
            "name": "T1",
            "question": "Does mean 50%-compression time stay within a loose <50 steps/decade bound when regressed on log10(c)?",
            "fit_c_values": [c_value for c_value in c_values if c_value > 1],
            "num_fit_points": len(log_cs),
            "slope_steps_per_decade": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_squared),
            "legacy_bound_pass": t1_legacy_pass,
            "monotone_log_growth_support": t1_monotone_support,
            "caveat": (
                "A negative slope can satisfy the loose legacy bound while failing to support the"
                " narrative that each decade in c adds extra correction steps."
            ),
        },
        "T2": {
            "name": "T2",
            "question": "Are the finite mean half-lives below 50 steps for the c <= 1000 control range?",
            "reported_c_values": t2_control_cs,
            "per_c": t2_per_c,
            "legacy_finite_only_pass": t2_legacy_finite_only_pass,
            "strict_all_reported_pass": t2_strict_all_reported_pass,
            "nonfinite_c_values": t2_nonfinite_cs,
            "caveat": (
                "c=1 can legitimately return no finite half-life because the baseline may start near"
                " balanced and never compress by another factor of two."
            ),
        },
        "T3": {
            "name": "T3",
            "question": "Does loss onset precede 50%-compression under the current raw index convention?",
            "per_c": t3_per_c,
            "legacy_any_pass": t3_legacy_any_pass,
            "all_comparable_c_pass": t3_all_pass,
            "support_fraction": t3_support_fraction,
            "num_comparable_c_values": len(t3_valid_flags),
            "caveat": (
                "This comparison is descriptive only because loss and imbalance histories are recorded"
                " on slightly different operational clocks."
            ),
        },
    }


def get_provenance_metadata(collect_histories):
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "cwd_at_invocation": os.getcwd(),
        "collect_histories": bool(collect_histories),
    }


def run_experiment(config_overrides=None, collect_histories=True):
    config = get_default_config()
    if config_overrides:
        for key, value in config_overrides.items():
            if key == "c_values":
                config[key] = list(value)
            else:
                config[key] = value

    start_time = time.perf_counter()
    seed_schedule = build_seed_schedule(config)
    per_c = {}

    for c_value in config["c_values"]:
        seed_results = []
        for seed_info in seed_schedule:
            rng_data = np.random.RandomState(seed_info["data_seed"])
            x = rng_data.randn(config["dim"], config["batch_size"]) * 0.3
            y = rng_data.randn(config["dim"], config["batch_size"]) * 0.3

            rng_weights = np.random.RandomState(seed_info["weight_seed"])
            weights_init = init_weights(rng_weights, c_value, config["dim"], config["n_layers"])

            initial_weight_norms = [float(np.linalg.norm(weight, ord="fro")) for weight in weights_init]
            initial_loss = compute_loss(weights_init, x, y)
            initial_grads = compute_gradients(weights_init, x, y)
            initial_grad_norms = [float(np.linalg.norm(grad, ord="fro")) for grad in initial_grads]
            initial_grad_imbalance = compute_ratio(initial_grad_norms)

            tracked = train_and_track(weights_init, x, y, config)
            imbalance_history = tracked["imbalance_history"]
            loss_history = tracked["loss_history"]
            half_life = find_half_life(imbalance_history)
            training_onset = find_training_onset(loss_history, config["onset_drop_fraction"])
            raw_index_flag = (
                bool(training_onset < half_life)
                if np.isfinite(training_onset) and np.isfinite(half_life)
                else None
            )

            seed_result = {
                "seed_idx": seed_info["seed_idx"],
                "data_seed": seed_info["data_seed"],
                "weight_seed": seed_info["weight_seed"],
                "c": c_value,
                "initial_weight_norms": initial_weight_norms,
                "initial_gradient_norms": initial_grad_norms,
                "initial_gradient_imbalance": initial_grad_imbalance,
                "initial_loss": float(initial_loss),
                "first_recorded_momentum_imbalance": (
                    float(imbalance_history[0]) if imbalance_history else float("nan")
                ),
                "final_momentum_imbalance": (
                    float(imbalance_history[-1]) if imbalance_history else float("nan")
                ),
                "final_loss": float(loss_history[-1]) if loss_history else float("nan"),
                "half_life": float(half_life),
                "training_onset": float(training_onset),
                "raw_index_onset_before_half_life": raw_index_flag,
                "num_recorded_loss_steps": tracked["num_recorded_loss_steps"],
                "num_recorded_imbalance_steps": tracked["num_recorded_imbalance_steps"],
                "stopped_early": tracked["stopped_early"],
                "stop_reason": tracked["stop_reason"],
                "loss_history": loss_history if collect_histories else None,
                "imbalance_history": imbalance_history if collect_histories else None,
            }
            seed_results.append(seed_result)

        summary = summarize_seed_results(c_value, seed_results)
        per_c[c_value] = {
            "c": c_value,
            "seed_results": seed_results,
            "summary": summary,
        }

    aggregate_summary = [per_c[c_value]["summary"] for c_value in config["c_values"]]
    hypothesis_tests = compute_hypothesis_tests(aggregate_summary, config["c_values"])
    runtime_seconds = time.perf_counter() - start_time

    return {
        "experiment_id": "2.2c",
        "title": "Toy study of initial 50%-compression of momentum imbalance under layerwise rescaling",
        "script_path": os.path.abspath(__file__),
        "script_dir": SCRIPT_DIR,
        "provenance": get_provenance_metadata(collect_histories),
        "config": config,
        "seed_schedule": seed_schedule,
        "metric_notes": {
            "toy_scope": (
                "4-layer 32x32 deep-linear regression with synthetic Gaussian data and a Muon-like"
                " update; this is a mechanistic toy study, not a realistic benchmark."
            ),
            "half_life": (
                "The reported half-life is the first recorded step where post-update momentum"
                " imbalance drops below half of its first recorded value. It measures early 50%"
                " compression, not full rebalancing."
            ),
            "training_onset": (
                "Training onset is the first recorded pre-update loss index where loss < 0.99 *"
                " initial loss."
            ),
            "clock_mismatch": (
                "loss_history[0] is pre-update, whereas imbalance_history[0] is post-first-update;"
                " onset-versus-half-life comparisons are therefore descriptive only."
            ),
            "theory_scope": (
                "These diagnostics alone do not estimate an RG spectral gap, fixed-point stability,"
                " or asymptotic scale-invariant behavior."
            ),
        },
        "runtime_seconds": float(runtime_seconds),
        "per_c": per_c,
        "aggregate_summary": aggregate_summary,
        "hypothesis_tests": hypothesis_tests,
    }


def format_float(value, fmt=".3g"):
    return format(value, fmt) if np.isfinite(value) else "nan"


def print_summary(results):
    config = results["config"]
    aggregate_summary = results["aggregate_summary"]
    hypothesis_tests = results["hypothesis_tests"]
    provenance = results.get("provenance", {})

    print("=" * 120)
    print("2.2c: TOY STUDY OF INITIAL MOMENTUM-IMBALANCE COMPRESSION UNDER LAYERWISE RESCALING")
    print("=" * 120)
    print(results["title"])
    print(f"Script: {results['script_path']}")
    print()
    print("Configuration")
    print("-" * 120)
    print(
        f"c values={config['c_values']} | layers={config['n_layers']} | dim={config['dim']} | "
        f"steps={config['num_steps']} | seeds={config['num_seeds']} | batch={config['batch_size']}"
    )
    print(
        f"lr={config['lr']} | momentum_beta={config['momentum_beta']} | ns_iters={config['ns_iters']} | "
        f"runtime={results['runtime_seconds']:.2f}s"
    )
    print()
    print("Execution provenance")
    print("-" * 120)
    for key in [
        "generated_at_utc",
        "python_executable",
        "python_version",
        "platform",
        "numpy_version",
        "cwd_at_invocation",
        "collect_histories",
    ]:
        if key in provenance:
            print(f"{key}: {provenance[key]}")

    print()
    print("Metric caveats")
    print("-" * 120)
    for key, note in results["metric_notes"].items():
        print(f"{key}: {note}")

    print()
    print("Aggregate summary by c")
    print("-" * 120)
    header = (
        f"{'c':>8} {'grad imb':>12} {'1st mom imb':>12} {'final mom imb':>14} "
        f"{'HL mean':>8} {'HL std':>8} {'Onset':>8} {'Onset std':>10} {'raw onset<HL':>14}"
    )
    print(header)
    print("-" * len(header))
    for summary in aggregate_summary:
        comparison = f"{summary['raw_index_onset_before_half_life_count']}/{summary['raw_index_comparison_count']}"
        print(
            f"{summary['c']:>8} "
            f"{format_float(summary['mean_initial_gradient_imbalance'], '.2e'):>12} "
            f"{format_float(summary['mean_first_recorded_momentum_imbalance'], '.2e'):>12} "
            f"{format_float(summary['mean_final_momentum_imbalance'], '.2e'):>14} "
            f"{format_float(summary['mean_half_life'], '.1f'):>8} "
            f"{format_float(summary['std_half_life'], '.1f'):>8} "
            f"{format_float(summary['mean_training_onset'], '.1f'):>8} "
            f"{format_float(summary['std_training_onset'], '.1f'):>10} "
            f"{comparison:>14}"
        )

    t1 = hypothesis_tests["T1"]
    t2 = hypothesis_tests["T2"]
    t3 = hypothesis_tests["T3"]

    print()
    print("Current T1/T2/T3 outputs (with caveats)")
    print("-" * 120)
    print("T1:")
    print(f"  question: {t1['question']}")
    print(
        f"  slope={format_float(t1['slope_steps_per_decade'], '.2f')} steps/decade | "
        f"R^2={format_float(t1['r_squared'], '.3f')} | legacy bound pass={t1['legacy_bound_pass']} | "
        f"monotone support={t1['monotone_log_growth_support']}"
    )
    print(f"  caveat: {t1['caveat']}")

    print("T2:")
    print(f"  question: {t2['question']}")
    for entry in t2["per_c"]:
        print(
            f"    c={entry['c']}: mean_half_life={format_float(entry['mean_half_life'], '.1f')} | "
            f"finite={entry['finite']} | under_50={entry['under_50']}"
        )
    print(
        f"  legacy finite-only pass={t2['legacy_finite_only_pass']} | "
        f"strict all-reported pass={t2['strict_all_reported_pass']} | "
        f"nonfinite c values={t2['nonfinite_c_values']}"
    )
    print(f"  caveat: {t2['caveat']}")

    print("T3:")
    print(f"  question: {t3['question']}")
    for entry in t3["per_c"]:
        print(
            f"    c={entry['c']}: onset={format_float(entry['mean_training_onset'], '.1f')} | "
            f"half_life={format_float(entry['mean_half_life'], '.1f')} | "
            f"raw onset < half-life={entry['raw_index_onset_before_half_life']}"
        )
    print(
        f"  legacy any-pass={t3['legacy_any_pass']} | all comparable pass={t3['all_comparable_c_pass']} | "
        f"support_fraction={format_float(t3['support_fraction'], '.2f')} over {t3['num_comparable_c_values']} comparable c values"
    )
    print(f"  caveat: {t3['caveat']}")

    print()
    print("Calibrated readout")
    print("-" * 120)
    print("- The toy study supports very rapid initial 50% compression of momentum imbalance for large c.")
    print("- It does not, by this metric alone, establish full rebalancing or a complete self-correction timescale.")
    print("- Large c can still leave substantial final imbalance and delayed loss reduction despite fast initial compression.")


def main():
    results = run_experiment()
    print_summary(results)


if __name__ == "__main__":
    main()
