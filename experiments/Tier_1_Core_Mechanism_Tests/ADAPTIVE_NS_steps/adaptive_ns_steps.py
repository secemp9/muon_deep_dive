#!/usr/bin/env python3
"""
Experiment 2.18: Adaptive NS steps -- toy single-seed schedule comparison.

Script counterpart to `run_experiment.ipynb`.

This first completion pass keeps the original toy experiment and default settings,
but makes the outputs more honest and reusable:
- tracked effective-rank histories are for the first layer's gradient only
- counted NS matmuls are an NS-only proxy, not full training compute
- conclusions are framed as single-seed architecture-specific observations
- the experiment can be imported and reused via `run_experiment()`
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

SCHEDULE_ORDER = [
    "(a) Fixed k=1",
    "(b) Fixed k=5",
    "(c) Fixed k=10",
    "(d) Adaptive-linear",
    "(e) Adaptive-erank",
]

PLOT_STYLES = {
    "colors": {
        "(a) Fixed k=1": "#1f77b4",
        "(b) Fixed k=5": "#ff7f0e",
        "(c) Fixed k=10": "#2ca02c",
        "(d) Adaptive-linear": "#d62728",
        "(e) Adaptive-erank": "#9467bd",
    },
    "linestyles": {
        "(a) Fixed k=1": "-",
        "(b) Fixed k=5": "-",
        "(c) Fixed k=10": "-",
        "(d) Adaptive-linear": "--",
        "(e) Adaptive-erank": ":",
    },
}


def get_default_config():
    """Return the default toy experiment configuration."""
    return {
        "experiment_id": "Experiment 2.18",
        "title": "Adaptive NS steps -- k(t) decreasing over training",
        "question": (
            "In this toy single-seed setting, can lower or adaptive Newton-Schulz "
            "iteration counts preserve or improve final loss while reducing counted "
            "NS-only matmul usage relative to fixed k=5?"
        ),
        "counterparts": {
            "script": "adaptive_ns_steps.py",
            "notebook": "run_experiment.ipynb",
        },
        "n_steps": 500,
        "tracked_layer_index": 0,
        "hypothesis_thresholds": {
            "t1_loss_improvement_pct": 3.0,
            "t2_proxy_improvement_pct": 10.0,
        },
        "deep_linear": {
            "display_name": "Deep Linear Network",
            "short_name": "Deep Linear",
            "depth": 4,
            "width": 32,
            "input_dim": 32,
            "output_dim": 32,
            "lr": 0.02,
            "seed": 42,
        },
        "relu": {
            "display_name": "ReLU Network",
            "short_name": "ReLU",
            "depth": 4,
            "width": 32,
            "input_dim": 32,
            "output_dim": 32,
            "lr": 0.01,
            "seed": 42,
            "batch_size": 64,
        },
        "caveats": {
            "single_seed": (
                "This is still a single-seed toy comparison. It is useful for "
                "mechanistic inspection, not for strong statistical claims."
            ),
            "tracked_erank": (
                "The stored erank traces are first-layer gradient effective-rank "
                "histories only. They are not whole-network or all-layer rank "
                "measurements."
            ),
            "counted_ns_matmuls": (
                "The counted NS matmuls only cover the quintic Newton-Schulz update "
                "matmuls (4 per NS iteration per layer). They exclude forward/backward "
                "passes, effective-rank evaluation, Python overhead, plotting, and "
                "wall-clock/runtime differences."
            ),
            "scope": (
                "Deep-linear and ReLU observations below are architecture-specific toy "
                "results, not universal gauge-fixing or rank-decay conclusions."
            ),
        },
    }


# ============================================================================
# Core functions
# ============================================================================

def newton_schulz_quintic(G, num_iters=5):
    """
    Muon's quintic Newton-Schulz iteration.

    Each iteration uses 4 matrix multiplications in this counted proxy:
      1) X^T @ X
      2) X @ (X^T @ X)
      3) (X^T @ X)^2
      4) X @ (X^T @ X)^2
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (np.linalg.norm(G, "fro") + 1e-30)

    for _ in range(num_iters):
        XtX = X.T @ X
        X_XtX = X @ XtX
        XtX2 = XtX @ XtX
        X_XtX2 = X @ XtX2
        X = a * X + b * X_XtX + c * X_XtX2

    return X


def effective_rank(M):
    """Effective rank via Shannon entropy of normalized singular values."""
    s = np.linalg.svd(M, compute_uv=False)
    s = s[s > 1e-30]
    if len(s) == 0:
        return 1.0
    p = s / s.sum()
    H = -np.sum(p * np.log(p + 1e-30))
    return float(np.exp(H))


def relu(x):
    return np.maximum(0, x)


def relu_deriv(x):
    return (x > 0).astype(float)


# ============================================================================
# Schedule definitions
# ============================================================================

def schedule_fixed(k_val):
    """Return a schedule function that always returns k_val."""

    def sched(t, T, G=None, n=None):
        return k_val

    sched.__name__ = f"Fixed k={k_val}"
    return sched


def schedule_adaptive_linear(t, T, G=None, n=None):
    """k(t) = max(1, round(5 - 4*t/T)) -- starts at 5, linearly to 1."""
    return max(1, round(5 - 4 * t / T))


schedule_adaptive_linear.__name__ = "Adaptive-linear"


def schedule_adaptive_erank(t, T, G=None, n=None):
    """k(t) = max(1, round(5 * erank(G)/n)) -- lower k for lower-rank gradients."""
    if G is None:
        return 5
    er = effective_rank(G)
    return max(1, round(5 * er / n))


schedule_adaptive_erank.__name__ = "Adaptive-erank"


def get_schedule_suite():
    """Ordered schedule registry used by both script and notebook."""
    return {
        "(a) Fixed k=1": schedule_fixed(1),
        "(b) Fixed k=5": schedule_fixed(5),
        "(c) Fixed k=10": schedule_fixed(10),
        "(d) Adaptive-linear": schedule_adaptive_linear,
        "(e) Adaptive-erank": schedule_adaptive_erank,
    }


# ============================================================================
# Helpers
# ============================================================================

def make_checkpoint_steps(n_steps):
    """Representative checkpoint steps for compact k-summary tables."""
    candidates = [0, n_steps // 4, n_steps // 2, (3 * n_steps) // 4, n_steps - 1]
    return sorted(set(int(c) for c in candidates if 0 <= c < n_steps))


def summarize_history_edges(history):
    """Summaries for start/mid/end of a trajectory, robust to short runs."""
    T = len(history)
    window = max(1, min(20, T))
    start = float(np.mean(history[:window]))

    mid_start = max(0, T // 2 - window // 2)
    mid_end = min(T, mid_start + window)
    mid = float(np.mean(history[mid_start:mid_end]))

    end = float(np.mean(history[-window:]))
    return {
        "start_mean": start,
        "mid_mean": mid,
        "end_mean": end,
        "decreasing": bool(end < start),
    }


def build_layer_k_checkpoint_summary(run_result):
    """Compact checkpoint table for first-layer and all-layer k summaries."""
    checkpoints = []
    for step in make_checkpoint_steps(len(run_result["loss_curve"])):
        checkpoints.append(
            {
                "step": int(step),
                "first_layer_k": int(run_result["first_layer_k_history"][step]),
                "layer_k_min": int(run_result["all_layer_k_min_history"][step]),
                "layer_k_mean": float(run_result["all_layer_k_mean_history"][step]),
                "layer_k_max": int(run_result["all_layer_k_max_history"][step]),
            }
        )
    return checkpoints


def build_summary_table(results_by_schedule):
    """Compact per-schedule summary rows."""
    ref = results_by_schedule["(b) Fixed k=5"]
    ref_loss = ref["final_loss"]
    ref_matmuls = ref["counted_ns_matmuls"]

    rows = []
    for schedule_name in SCHEDULE_ORDER:
        run_result = results_by_schedule[schedule_name]
        erank_summary = summarize_history_edges(run_result["first_layer_grad_erank_history"])
        row = {
            "schedule": schedule_name,
            "final_loss": float(run_result["final_loss"]),
            "counted_ns_matmuls": int(run_result["counted_ns_matmuls"]),
            "loss_x_counted_ns_matmuls": float(
                run_result["final_loss"] * run_result["counted_ns_matmuls"]
            ),
            "vs_k5_loss_pct": float(
                (run_result["final_loss"] - ref_loss) / ref_loss * 100.0
            ),
            "vs_k5_counted_ns_matmuls_pct": float(
                (run_result["counted_ns_matmuls"] - ref_matmuls) / ref_matmuls * 100.0
            ),
            "first_layer_erank_start": erank_summary["start_mean"],
            "first_layer_erank_mid": erank_summary["mid_mean"],
            "first_layer_erank_end": erank_summary["end_mean"],
            "first_layer_erank_decreasing": bool(erank_summary["decreasing"]),
            "first_layer_k_min": int(min(run_result["first_layer_k_history"])),
            "first_layer_k_mean": float(np.mean(run_result["first_layer_k_history"])),
            "first_layer_k_max": int(max(run_result["first_layer_k_history"])),
        }
        rows.append(row)
    return rows


def evaluate_hypotheses(results_by_schedule, thresholds):
    """Single-seed T1/T2/T3/T4 checks with explicit caveats."""
    ref = results_by_schedule["(b) Fixed k=5"]
    ref_loss = ref["final_loss"]
    ref_matmuls = ref["counted_ns_matmuls"]
    ref_proxy = ref_loss * ref_matmuls

    checks = []

    adaptive_linear = results_by_schedule["(d) Adaptive-linear"]
    t1_improvement = (ref_loss - adaptive_linear["final_loss"]) / ref_loss * 100.0
    checks.append(
        {
            "test_id": "T1",
            "subject": "(d) Adaptive-linear",
            "description": "Adaptive-linear final loss improves on fixed k=5 by more than 3%.",
            "caveat": "Single-seed final-loss comparison only.",
            "passed": bool(t1_improvement > thresholds["t1_loss_improvement_pct"]),
            "metric_name": "loss_improvement_pct_vs_k5",
            "metric_value": float(t1_improvement),
            "threshold": float(thresholds["t1_loss_improvement_pct"]),
            "details": {
                "adaptive_final_loss": float(adaptive_linear["final_loss"]),
                "reference_final_loss": float(ref_loss),
            },
        }
    )

    for schedule_name in ["(d) Adaptive-linear", "(e) Adaptive-erank"]:
        run_result = results_by_schedule[schedule_name]
        proxy = run_result["final_loss"] * run_result["counted_ns_matmuls"]
        proxy_improvement = (ref_proxy - proxy) / ref_proxy * 100.0
        checks.append(
            {
                "test_id": "T2",
                "subject": schedule_name,
                "description": (
                    "Loss × counted-NS-matmul proxy improves on fixed k=5 by more than 10%."
                ),
                "caveat": (
                    "This is a toy proxy score, not a full compute or wall-clock measurement."
                ),
                "passed": bool(proxy_improvement > thresholds["t2_proxy_improvement_pct"]),
                "metric_name": "proxy_improvement_pct_vs_k5",
                "metric_value": float(proxy_improvement),
                "threshold": float(thresholds["t2_proxy_improvement_pct"]),
                "details": {
                    "adaptive_proxy": float(proxy),
                    "reference_proxy": float(ref_proxy),
                },
            }
        )

    erank_summary = summarize_history_edges(ref["first_layer_grad_erank_history"])
    checks.append(
        {
            "test_id": "T3",
            "subject": "(b) Fixed k=5",
            "description": (
                "Tracked first-layer gradient effective rank decreases over training."
            ),
            "caveat": (
                "This only uses the first-layer gradient trace under the k=5 reference run."
            ),
            "passed": bool(erank_summary["decreasing"]),
            "metric_name": "first_layer_erank_end_minus_start",
            "metric_value": float(
                erank_summary["end_mean"] - erank_summary["start_mean"]
            ),
            "threshold": 0.0,
            "details": {
                "start_mean": float(erank_summary["start_mean"]),
                "end_mean": float(erank_summary["end_mean"]),
            },
        }
    )

    for schedule_name in ["(d) Adaptive-linear", "(e) Adaptive-erank"]:
        run_result = results_by_schedule[schedule_name]
        saving = (ref_matmuls - run_result["counted_ns_matmuls"]) / ref_matmuls * 100.0
        checks.append(
            {
                "test_id": "T4",
                "subject": schedule_name,
                "description": "Counted NS matmuls are lower than fixed k=5.",
                "caveat": (
                    "Counted NS matmuls exclude forward/backward pass and auxiliary overhead."
                ),
                "passed": bool(run_result["counted_ns_matmuls"] < ref_matmuls),
                "metric_name": "counted_ns_matmul_saving_pct_vs_k5",
                "metric_value": float(saving),
                "threshold": 0.0,
                "details": {
                    "adaptive_counted_ns_matmuls": int(run_result["counted_ns_matmuls"]),
                    "reference_counted_ns_matmuls": int(ref_matmuls),
                },
            }
        )

    return checks


def print_header(config):
    print("=" * 100)
    print(f"  {config['experiment_id']}: {config['title']}")
    print("=" * 100)
    print(f"  Counterpart notebook: {config['counterparts']['notebook']}")
    print(f"  Question: {config['question']}")
    print(f"  Steps: {config['n_steps']}")
    print(f"  Tracked layer index for erank/k histories: {config['tracked_layer_index']}")
    print()
    print("  Caveats:")
    print(f"    - {config['caveats']['single_seed']}")
    print(f"    - {config['caveats']['tracked_erank']}")
    print(f"    - {config['caveats']['counted_ns_matmuls']}")
    print(f"    - {config['caveats']['scope']}")
    print()


def print_summary_table(architecture_payload):
    print(f"{'=' * 100}")
    print(f"  Results: {architecture_payload['display_name']}")
    print(f"{'=' * 100}")
    print(
        f"  {'Schedule':<25s} {'Final loss':>12s} {'Counted NS matmuls':>20s} "
        f"{'Loss x counted NS':>18s} {'vs k=5 loss':>14s} {'vs k=5 counted':>16s}"
    )
    print(
        f"  {'-' * 25} {'-' * 12} {'-' * 20} {'-' * 18} {'-' * 14} {'-' * 16}"
    )

    for row in architecture_payload["summary_table"]:
        print(
            f"  {row['schedule']:<25s} {row['final_loss']:12.6e} "
            f"{row['counted_ns_matmuls']:20d} {row['loss_x_counted_ns_matmuls']:18.4e} "
            f"{row['vs_k5_loss_pct']:+13.1f}% {row['vs_k5_counted_ns_matmuls_pct']:+15.1f}%"
        )

    print()
    print(
        "  Note: counted NS matmuls only count quintic Newton-Schulz update matmuls, "
        "not full training compute."
    )
    print()


def print_erank_analysis(architecture_payload):
    print(
        f"  Tracked first-layer gradient effective-rank analysis "
        f"({architecture_payload['short_name']}):"
    )
    print(
        f"  {'Schedule':<25s} {'start':>10s} {'mid':>10s} {'end':>10s} {'decreasing?':>12s}"
    )
    print(f"  {'-' * 25} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 12}")

    for row in architecture_payload["summary_table"]:
        print(
            f"  {row['schedule']:<25s} {row['first_layer_erank_start']:10.2f} "
            f"{row['first_layer_erank_mid']:10.2f} {row['first_layer_erank_end']:10.2f} "
            f"{'YES' if row['first_layer_erank_decreasing'] else 'NO':>12s}"
        )

    print()


def print_hypothesis_checks(architecture_payload):
    print(f"  --- {architecture_payload['short_name']} ---")
    for check in architecture_payload["hypothesis_checks"]:
        passed_label = "PASS" if check["passed"] else "FAIL"
        print(
            f"  {check['test_id']}: {check['subject']} :: {passed_label} :: "
            f"{check['description']}"
        )
        print(
            f"      {check['metric_name']} = {check['metric_value']:+.4f} "
            f"(threshold {check['threshold']:+.4f})"
        )
        print(f"      Caveat: {check['caveat']}")
    print()


# ============================================================================
# Training: Deep Linear Network
# ============================================================================

def train_deep_linear(
    depth=4,
    width=32,
    input_dim=32,
    output_dim=32,
    n_steps=500,
    lr=0.02,
    schedule_fn=None,
    seed=42,
    tracked_layer_index=0,
):
    """Train the deep linear toy problem and return structured histories."""
    if schedule_fn is None:
        schedule_fn = schedule_fixed(5)

    rng = np.random.RandomState(seed)
    target = rng.randn(output_dim, input_dim) * 0.5

    dims = [input_dim] + [width] * (depth - 1) + [output_dim]
    weights = []
    for i in range(depth):
        fan_in, fan_out = dims[i], dims[i + 1]
        W = rng.randn(fan_out, fan_in) * np.sqrt(2.0 / (fan_in + fan_out))
        weights.append(W)

    loss_curve = []
    first_layer_grad_erank_history = []
    first_layer_k_history = []
    all_layer_k_min_history = []
    all_layer_k_mean_history = []
    all_layer_k_max_history = []
    counted_ns_matmuls = 0

    for step in range(n_steps):
        product = np.eye(input_dim)
        for W in weights:
            product = W @ product

        diff = product - target
        loss = 0.5 * np.linalg.norm(diff, "fro") ** 2
        loss_curve.append(float(loss))

        grads = []
        for k_layer in range(depth):
            left = np.eye(weights[k_layer].shape[0])
            for j in range(k_layer + 1, depth):
                left = weights[j] @ left

            right = np.eye(input_dim)
            for j in range(k_layer):
                right = weights[j] @ right

            grad = left.T @ diff @ right.T
            grads.append(grad)

        tracked_grad = grads[tracked_layer_index]
        first_layer_grad_erank_history.append(float(effective_rank(tracked_grad)))

        layer_k_values = []
        for k_layer, G in enumerate(grads):
            k_ns = int(schedule_fn(step, n_steps, G=G, n=min(G.shape)))
            layer_k_values.append(k_ns)
            G_orth = newton_schulz_quintic(G, num_iters=k_ns)
            counted_ns_matmuls += k_ns * 4
            weights[k_layer] -= lr * G_orth

        first_layer_k_history.append(int(layer_k_values[tracked_layer_index]))
        all_layer_k_min_history.append(int(min(layer_k_values)))
        all_layer_k_mean_history.append(float(np.mean(layer_k_values)))
        all_layer_k_max_history.append(int(max(layer_k_values)))

    return {
        "loss_curve": loss_curve,
        "final_loss": float(loss_curve[-1]),
        "first_layer_grad_erank_history": first_layer_grad_erank_history,
        "first_layer_k_history": first_layer_k_history,
        "all_layer_k_min_history": all_layer_k_min_history,
        "all_layer_k_mean_history": all_layer_k_mean_history,
        "all_layer_k_max_history": all_layer_k_max_history,
        "counted_ns_matmuls": int(counted_ns_matmuls),
        "tracked_layer_index": int(tracked_layer_index),
    }


# ============================================================================
# Training: ReLU Network
# ============================================================================

def train_relu_net(
    depth=4,
    width=32,
    input_dim=32,
    output_dim=32,
    n_steps=500,
    lr=0.01,
    schedule_fn=None,
    seed=42,
    batch_size=64,
    tracked_layer_index=0,
):
    """Train the ReLU toy problem and return structured histories."""
    if schedule_fn is None:
        schedule_fn = schedule_fixed(5)

    rng = np.random.RandomState(seed)
    X_data = rng.randn(batch_size, input_dim)
    Y_data = rng.randn(batch_size, output_dim) * 0.3

    dims = [input_dim] + [width] * (depth - 1) + [output_dim]
    weights = []
    for i in range(depth):
        fan_in, fan_out = dims[i], dims[i + 1]
        W = rng.randn(fan_out, fan_in) * np.sqrt(2.0 / fan_in)
        weights.append(W)

    loss_curve = []
    first_layer_grad_erank_history = []
    first_layer_k_history = []
    all_layer_k_min_history = []
    all_layer_k_mean_history = []
    all_layer_k_max_history = []
    counted_ns_matmuls = 0

    for step in range(n_steps):
        activations = [X_data.T]
        pre_activations = []
        for i in range(depth):
            z = weights[i] @ activations[-1]
            pre_activations.append(z)
            if i < depth - 1:
                a = relu(z)
            else:
                a = z
            activations.append(a)

        output = activations[-1].T
        diff = output - Y_data
        loss = 0.5 * np.mean(np.sum(diff**2, axis=1))
        loss_curve.append(float(loss))

        delta = diff.T / batch_size
        grads = [None] * depth
        for i in range(depth - 1, -1, -1):
            grads[i] = delta @ activations[i].T
            if i > 0:
                delta = weights[i].T @ delta
                delta = delta * relu_deriv(pre_activations[i - 1])

        tracked_grad = grads[tracked_layer_index]
        first_layer_grad_erank_history.append(float(effective_rank(tracked_grad)))

        layer_k_values = []
        for k_layer, G in enumerate(grads):
            k_ns = int(schedule_fn(step, n_steps, G=G, n=min(G.shape)))
            layer_k_values.append(k_ns)
            G_orth = newton_schulz_quintic(G, num_iters=k_ns)
            counted_ns_matmuls += k_ns * 4
            weights[k_layer] -= lr * G_orth

        first_layer_k_history.append(int(layer_k_values[tracked_layer_index]))
        all_layer_k_min_history.append(int(min(layer_k_values)))
        all_layer_k_mean_history.append(float(np.mean(layer_k_values)))
        all_layer_k_max_history.append(int(max(layer_k_values)))

    return {
        "loss_curve": loss_curve,
        "final_loss": float(loss_curve[-1]),
        "first_layer_grad_erank_history": first_layer_grad_erank_history,
        "first_layer_k_history": first_layer_k_history,
        "all_layer_k_min_history": all_layer_k_min_history,
        "all_layer_k_mean_history": all_layer_k_mean_history,
        "all_layer_k_max_history": all_layer_k_max_history,
        "counted_ns_matmuls": int(counted_ns_matmuls),
        "tracked_layer_index": int(tracked_layer_index),
    }


# ============================================================================
# Experiment assembly
# ============================================================================

def run_schedule_suite(
    train_fn,
    architecture_key,
    architecture_config,
    n_steps,
    tracked_layer_index,
    hypothesis_thresholds,
):
    """Run all schedules for one architecture and build structured outputs."""
    schedules = get_schedule_suite()
    results_by_schedule = {}

    for schedule_name in SCHEDULE_ORDER:
        schedule_fn = schedules[schedule_name]
        train_kwargs = copy.deepcopy(architecture_config)
        display_name = train_kwargs.pop("display_name")
        short_name = train_kwargs.pop("short_name")

        run_result = train_fn(
            n_steps=n_steps,
            schedule_fn=schedule_fn,
            tracked_layer_index=tracked_layer_index,
            **train_kwargs,
        )
        run_result["architecture_key"] = architecture_key
        run_result["architecture_display_name"] = display_name
        run_result["architecture_short_name"] = short_name
        run_result["schedule_name"] = schedule_name
        run_result["seed"] = int(architecture_config["seed"])
        run_result["n_steps"] = int(n_steps)
        run_result["layer_k_checkpoint_summary"] = build_layer_k_checkpoint_summary(
            run_result
        )
        results_by_schedule[schedule_name] = run_result

    summary_table = build_summary_table(results_by_schedule)
    hypothesis_checks = evaluate_hypotheses(
        results_by_schedule,
        thresholds=hypothesis_thresholds,
    )

    return {
        "architecture_key": architecture_key,
        "display_name": architecture_config["display_name"],
        "short_name": architecture_config["short_name"],
        "seed": int(architecture_config["seed"]),
        "n_steps": int(n_steps),
        "tracked_layer_index": int(tracked_layer_index),
        "results_by_schedule": results_by_schedule,
        "summary_table": summary_table,
        "hypothesis_checks": hypothesis_checks,
    }


def run_experiment(config=None, verbose=False):
    """Run the full toy experiment and return notebook-friendly structured results."""
    base_config = get_default_config()
    if config is not None:
        merged = copy.deepcopy(base_config)
        for key, value in config.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key].update(value)
            else:
                merged[key] = value
        config = merged
    else:
        config = base_config

    start_time = time.perf_counter()

    if verbose:
        print_header(config)

    deep_linear_payload = run_schedule_suite(
        train_deep_linear,
        architecture_key="deep_linear",
        architecture_config=config["deep_linear"],
        n_steps=config["n_steps"],
        tracked_layer_index=config["tracked_layer_index"],
        hypothesis_thresholds=config["hypothesis_thresholds"],
    )

    if verbose:
        print_summary_table(deep_linear_payload)
        print_erank_analysis(deep_linear_payload)

    relu_payload = run_schedule_suite(
        train_relu_net,
        architecture_key="relu",
        architecture_config=config["relu"],
        n_steps=config["n_steps"],
        tracked_layer_index=config["tracked_layer_index"],
        hypothesis_thresholds=config["hypothesis_thresholds"],
    )

    if verbose:
        print_summary_table(relu_payload)
        print_erank_analysis(relu_payload)
        print("=" * 100)
        print("  HYPOTHESIS CHECKS (single-seed toy diagnostics)")
        print("=" * 100)
        print_hypothesis_checks(deep_linear_payload)
        print_hypothesis_checks(relu_payload)

    elapsed_seconds = time.perf_counter() - start_time

    results = {
        "identity": {
            "experiment_id": config["experiment_id"],
            "title": config["title"],
            "script": config["counterparts"]["script"],
            "notebook": config["counterparts"]["notebook"],
            "question": config["question"],
        },
        "caveats": copy.deepcopy(config["caveats"]),
        "config": copy.deepcopy(config),
        "plot_styles": copy.deepcopy(PLOT_STYLES),
        "runtime": {
            "elapsed_seconds": float(elapsed_seconds),
        },
        "architectures": {
            "deep_linear": deep_linear_payload,
            "relu": relu_payload,
        },
    }

    if verbose:
        print(f"  Runtime: {elapsed_seconds:.2f} seconds")
        print()

    return results


# ============================================================================
# Plotting helpers used by both script and notebook
# ============================================================================

def plot_training_dynamics(results, save_path=None):
    """3x2 figure: loss, tracked first-layer erank, tracked first-layer k(t)."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 18), sharex="col")
    fig.suptitle(
        "Exp 2.18: Adaptive NS Steps (toy single-seed schedule comparison)",
        fontsize=14,
        fontweight="bold",
    )

    colors = results["plot_styles"]["colors"]
    linestyles = results["plot_styles"]["linestyles"]

    for col, architecture_key in enumerate(["deep_linear", "relu"]):
        architecture_payload = results["architectures"][architecture_key]
        schedule_results = architecture_payload["results_by_schedule"]
        short_name = architecture_payload["short_name"]

        ax = axes[0, col]
        for schedule_name in SCHEDULE_ORDER:
            run_result = schedule_results[schedule_name]
            ax.semilogy(
                run_result["loss_curve"],
                color=colors[schedule_name],
                linestyle=linestyles[schedule_name],
                linewidth=1.8,
                label=schedule_name,
                alpha=0.9,
            )
        ax.set_title(f"{short_name}: loss trajectory")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

        ax = axes[1, col]
        for schedule_name in SCHEDULE_ORDER:
            run_result = schedule_results[schedule_name]
            ax.plot(
                run_result["first_layer_grad_erank_history"],
                color=colors[schedule_name],
                linestyle=linestyles[schedule_name],
                linewidth=1.5,
                label=schedule_name,
                alpha=0.85,
            )
        ax.set_title(f"{short_name}: tracked first-layer gradient erank")
        ax.set_ylabel("First-layer grad erank")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

        ax = axes[2, col]
        for schedule_name in SCHEDULE_ORDER:
            run_result = schedule_results[schedule_name]
            ax.plot(
                run_result["first_layer_k_history"],
                color=colors[schedule_name],
                linestyle=linestyles[schedule_name],
                linewidth=1.5,
                label=schedule_name,
                alpha=0.85,
            )
        ax.set_title(f"{short_name}: tracked first-layer k(t)")
        ax.set_xlabel("Training step")
        ax.set_ylabel("First-layer k")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.text(
        0.5,
        0.01,
        "Caveat: erank and k(t) traces above track the first layer only. "
        "Use the adaptive-erank checkpoint table for layerwise min/mean/max k. "
        "Counted NS matmuls are an NS-only proxy, not full training compute.",
        ha="center",
        fontsize=9,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, axes


def plot_pareto_summary(results, save_path=None):
    """1x2 figure: final loss vs counted NS-only matmul proxy."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Exp 2.18: final loss vs counted NS matmuls (toy proxy)",
        fontsize=13,
        fontweight="bold",
    )

    colors = results["plot_styles"]["colors"]

    for col, architecture_key in enumerate(["deep_linear", "relu"]):
        architecture_payload = results["architectures"][architecture_key]
        schedule_results = architecture_payload["results_by_schedule"]
        ax = axes[col]

        for schedule_name in SCHEDULE_ORDER:
            run_result = schedule_results[schedule_name]
            ax.scatter(
                run_result["counted_ns_matmuls"],
                run_result["final_loss"],
                c=colors[schedule_name],
                s=100,
                zorder=5,
                edgecolors="black",
            )
            ax.annotate(
                schedule_name.split(")")[0] + ")",
                (run_result["counted_ns_matmuls"], run_result["final_loss"]),
                textcoords="offset points",
                xytext=(8, 5),
                fontsize=8,
            )

        ax.set_xlabel("Counted NS matmuls (NS-only proxy)")
        ax.set_ylabel("Final loss")
        ax.set_yscale("log")
        ax.set_title(f"{architecture_payload['short_name']}: Pareto-style view")
        ax.grid(True, alpha=0.3)

    fig.text(
        0.5,
        0.01,
        "Proxy caveat: counted NS matmuls exclude forward/backward passes, effective-rank "
        "evaluation, Python overhead, and wall-clock runtime.",
        ha="center",
        fontsize=9,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, axes


def save_default_figures(results, plot_dir=SCRIPT_DIR):
    """Save the standard figures used by the script counterpart."""
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    training_path = plot_dir / "adaptive_ns_steps.png"
    pareto_path = plot_dir / "adaptive_ns_pareto.png"

    fig1, _ = plot_training_dynamics(results, save_path=training_path)
    plt.close(fig1)

    fig2, _ = plot_pareto_summary(results, save_path=pareto_path)
    plt.close(fig2)

    return {
        "training_dynamics": str(training_path),
        "pareto_summary": str(pareto_path),
    }


# ============================================================================
# Main entrypoint
# ============================================================================

def main():
    results = run_experiment(verbose=True)
    artifacts = save_default_figures(results, plot_dir=SCRIPT_DIR)

    print("  Saved figures:")
    print(f"    - {artifacts['training_dynamics']}")
    print(f"    - {artifacts['pareto_summary']}")
    print()
    print("=" * 100)
    print("  EXPERIMENT COMPLETE")
    print("=" * 100)

    return results


if __name__ == "__main__":
    main()
