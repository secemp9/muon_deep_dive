#!/usr/bin/env python3
"""
Toy diagnostic: gradient gauge-fraction tracking in a ReLU MLP.

This companion script for ``KILL_EXPERIMENT_gauge_fraction_relu`` measures the
Stiefel-normal fraction of the raw gradient ``G`` relative to ``Q(W)=polar(W)``
along training trajectories in a small full-batch random-regression setup.

Measured here
-------------
- ``gauge_fraction = ||Q sym(Q^T G)||_F^2 / ||G||_F^2``
- the same diagnostic along SGD and Muon training trajectories

Not directly established here
-----------------------------
- that this gauge fraction is exactly the component Muon removes in practice
- that Muon's practical update is identical to the tangent projection
- broad claims beyond this toy NumPy experiment

Primary analysis: hidden layers (all layers except the final linear output
layer). Control analysis: the final output layer.
"""

from __future__ import annotations

import time
from textwrap import dedent

import numpy as np


DEFAULT_CONFIG = {
    "seed": 42,
    "seed_stride": 31,
    "input_dim": 32,
    "hidden_dim": 32,
    "output_dim": 32,
    "num_layers": 6,
    "num_samples": 100,
    "num_steps": 500,
    "lr_sgd": 0.003,
    "lr_muon": 0.005,
    "momentum": 0.9,
    "ns_iters": 5,
    "num_seeds": 3,
    "data_scale": 0.3,
}

CONDITION_ORDER = ("relu_sgd", "relu_muon", "linear_sgd", "linear_muon")
CONDITION_LABELS = {
    "relu_sgd": "ReLU + SGD",
    "relu_muon": "ReLU + Muon",
    "linear_sgd": "Linear + SGD",
    "linear_muon": "Linear + Muon",
}
CONDITION_COLORS = {
    "relu_sgd": "#1f77b4",
    "relu_muon": "#ff7f0e",
    "linear_sgd": "#2ca02c",
    "linear_muon": "#d62728",
}
CONDITION_SPECS = {
    "relu_sgd": {"net_type": "relu", "optimizer": "sgd", "lr_key": "lr_sgd"},
    "relu_muon": {"net_type": "relu", "optimizer": "muon", "lr_key": "lr_muon"},
    "linear_sgd": {"net_type": "linear", "optimizer": "sgd", "lr_key": "lr_sgd"},
    "linear_muon": {"net_type": "linear", "optimizer": "muon", "lr_key": "lr_muon"},
}


def get_default_config():
    """Return a copy of the default experiment configuration."""
    return dict(DEFAULT_CONFIG)


def validate_config(config):
    """Validate the first-pass toy setup constraints."""
    if config["num_layers"] < 2:
        raise ValueError("num_layers must be at least 2 for hidden-vs-last analysis.")
    if not (
        config["input_dim"] == config["hidden_dim"] == config["output_dim"]
    ):
        raise ValueError(
            "This first-pass implementation preserves the original square setup: "
            "input_dim, hidden_dim, and output_dim must match."
        )
    if config["num_steps"] <= 0 or config["num_seeds"] <= 0:
        raise ValueError("num_steps and num_seeds must be positive.")


# =============================================================================
# NETWORKS AND LOSSES
# =============================================================================

def init_weights(rng, num_layers, hidden_dim):
    """Initialize a square MLP with He-scaled Gaussian weights."""
    weights = []
    std = np.sqrt(2.0 / hidden_dim)
    for _ in range(num_layers):
        weights.append((rng.randn(hidden_dim, hidden_dim) * std).copy())
    return weights


def forward_relu(weights, X):
    """Forward pass with ReLU on all but the last layer."""
    out = X.copy()
    pre_acts = []
    post_acts = [X.copy()]
    for idx, W in enumerate(weights):
        pre = W @ out
        pre_acts.append(pre.copy())
        if idx < len(weights) - 1:
            out = np.maximum(0, pre)
        else:
            out = pre
        post_acts.append(out.copy())
    return out, pre_acts, post_acts


def compute_loss(weights, X, Y):
    pred, _, _ = forward_relu(weights, X)
    diff = pred - Y
    return 0.5 * np.mean(np.sum(diff ** 2, axis=0))


def compute_gradients(weights, X, Y):
    """Backprop through the ReLU MLP."""
    num_layers = len(weights)
    N = X.shape[1]

    pred, pre_acts, post_acts = forward_relu(weights, X)
    delta = (pred - Y) / N

    grads = [None] * num_layers
    for layer in range(num_layers - 1, -1, -1):
        grads[layer] = delta @ post_acts[layer].T
        if layer > 0:
            delta = weights[layer].T @ delta
            delta = delta * (pre_acts[layer - 1] > 0).astype(float)

    return grads


def forward_linear(weights, X):
    out = X.copy()
    for W in weights:
        out = W @ out
    return out


def compute_loss_linear(weights, X, Y):
    pred = forward_linear(weights, X)
    diff = pred - Y
    return 0.5 * np.mean(np.sum(diff ** 2, axis=0))


def compute_gradients_linear(weights, X, Y):
    num_layers = len(weights)
    N = X.shape[1]
    activations = [X.copy()]
    out = X.copy()
    for W in weights:
        out = W @ out
        activations.append(out.copy())

    delta = (activations[-1] - Y) / N
    grads = [None] * num_layers
    for layer in range(num_layers - 1, -1, -1):
        grads[layer] = delta @ activations[layer].T
        if layer > 0:
            delta = weights[layer].T @ delta
    return grads


# =============================================================================
# GAUGE DECOMPOSITION AND MUON STEP
# =============================================================================

def compute_polar_factor(W):
    """Compute the orthogonal polar factor Q = U V^T from the SVD of W."""
    U, _, Vt = np.linalg.svd(W, full_matrices=True)
    return U @ Vt


def gauge_decomposition(G, W):
    """Decompose G into normal and tangent parts at Q(W)."""
    Q = compute_polar_factor(W)
    QtG = Q.T @ G
    sym_QtG = 0.5 * (QtG + QtG.T)
    G_normal = Q @ sym_QtG
    G_tangent = G - G_normal

    G_norm_sq = np.sum(G ** 2)
    G_normal_norm_sq = np.sum(G_normal ** 2)
    if G_norm_sq < 1e-30:
        return 0.0, G_normal, G_tangent

    gauge_fraction = G_normal_norm_sq / G_norm_sq
    return gauge_fraction, G_normal, G_tangent


def newton_schulz_ortho(M, n_iters):
    """Orthogonalize M with a fixed number of Newton-Schulz iterations."""
    norm = np.linalg.norm(M, ord="fro")
    if norm < 1e-12:
        return M
    X = M / norm
    for _ in range(n_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


# =============================================================================
# TRAINING
# =============================================================================

def train_with_gauge_tracking(
    net_type,
    weights_init,
    X,
    Y,
    lr,
    optimizer,
    n_steps,
    momentum,
    ns_iters,
):
    """Train and track gauge fractions with the original toy update rules."""
    weights = [w.copy() for w in weights_init]
    velocities = [np.zeros_like(w) for w in weights]
    num_layers = len(weights)

    if net_type == "relu":
        loss_fn = compute_loss
        grad_fn = compute_gradients
    else:
        loss_fn = compute_loss_linear
        grad_fn = compute_gradients_linear

    losses = []
    gauge_fractions = []

    for step in range(n_steps):
        loss = loss_fn(weights, X, Y)
        losses.append(loss)

        if np.isnan(loss) or loss > 1e10:
            losses.extend([1e10] * (n_steps - step - 1))
            gauge_fractions.extend([[0.0] * num_layers] * (n_steps - step - 1))
            break

        grads = grad_fn(weights, X, Y)

        step_fractions = []
        for layer in range(num_layers):
            gf, _, _ = gauge_decomposition(grads[layer], weights[layer])
            step_fractions.append(gf)
        gauge_fractions.append(step_fractions)

        if optimizer == "sgd":
            for layer in range(num_layers):
                velocities[layer] = momentum * velocities[layer] + grads[layer]
                weights[layer] = weights[layer] - lr * velocities[layer]
        elif optimizer == "muon":
            for layer in range(num_layers):
                ortho_grad = newton_schulz_ortho(grads[layer], n_iters=ns_iters)
                velocities[layer] = momentum * velocities[layer] + ortho_grad
                weights[layer] = weights[layer] - lr * velocities[layer]
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

    return np.array(losses), np.array(gauge_fractions)


# =============================================================================
# SUMMARIES
# =============================================================================

def _seed_list(config):
    return [config["seed"] + idx * config["seed_stride"] for idx in range(config["num_seeds"])]


def summarize_condition(losses, gauge_fractions):
    """Build descriptive summaries for one condition."""
    num_seeds, num_steps, num_layers = gauge_fractions.shape
    ddof = 1 if num_seeds > 1 else 0

    hidden_exists = num_layers > 1
    hidden_layers = gauge_fractions[:, :, :-1] if hidden_exists else None

    per_seed_all_layers_pct = gauge_fractions.mean(axis=(1, 2)) * 100.0
    per_seed_hidden_layers_pct = (
        hidden_layers.mean(axis=(1, 2)) * 100.0 if hidden_exists else np.full(num_seeds, np.nan)
    )
    per_seed_last_layer_pct = gauge_fractions[:, :, -1].mean(axis=1) * 100.0

    per_step_all_layers_pct_by_seed = gauge_fractions.mean(axis=2) * 100.0
    per_step_last_layer_pct_by_seed = gauge_fractions[:, :, -1] * 100.0

    if hidden_exists:
        per_step_hidden_layers_pct_by_seed = hidden_layers.mean(axis=2) * 100.0
        per_layer_hidden_mean_pct = hidden_layers.mean(axis=(0, 1)) * 100.0
    else:
        per_step_hidden_layers_pct_by_seed = np.full((num_seeds, num_steps), np.nan)
        per_layer_hidden_mean_pct = np.array([], dtype=float)

    early_stop = min(50, num_steps)
    late_start = max(0, num_steps - 100)

    return {
        "per_seed_all_layers_pct": per_seed_all_layers_pct,
        "per_seed_hidden_layers_pct": per_seed_hidden_layers_pct,
        "per_seed_last_layer_pct": per_seed_last_layer_pct,
        "all_layers_mean_pct": float(per_seed_all_layers_pct.mean()),
        "all_layers_sd_pct": float(per_seed_all_layers_pct.std(ddof=ddof)),
        "hidden_layers_mean_pct": float(np.nanmean(per_seed_hidden_layers_pct)),
        "hidden_layers_sd_pct": float(np.nanstd(per_seed_hidden_layers_pct, ddof=ddof)),
        "last_layer_mean_pct": float(per_seed_last_layer_pct.mean()),
        "last_layer_sd_pct": float(per_seed_last_layer_pct.std(ddof=ddof)),
        "per_layer_mean_pct": gauge_fractions.mean(axis=(0, 1)) * 100.0,
        "per_layer_sd_across_seeds_pct": gauge_fractions.mean(axis=1).std(axis=0, ddof=ddof) * 100.0,
        "per_layer_hidden_mean_pct": per_layer_hidden_mean_pct,
        "per_step_mean_all_layers_pct": per_step_all_layers_pct_by_seed.mean(axis=0),
        "per_step_sd_all_layers_pct": per_step_all_layers_pct_by_seed.std(axis=0, ddof=ddof),
        "per_step_mean_hidden_layers_pct": np.nanmean(per_step_hidden_layers_pct_by_seed, axis=0),
        "per_step_sd_hidden_layers_pct": np.nanstd(per_step_hidden_layers_pct_by_seed, axis=0, ddof=ddof),
        "per_step_mean_last_layer_pct": per_step_last_layer_pct_by_seed.mean(axis=0),
        "per_step_sd_last_layer_pct": per_step_last_layer_pct_by_seed.std(axis=0, ddof=ddof),
        "loss_per_step_mean": losses.mean(axis=0),
        "loss_per_step_sd": losses.std(axis=0, ddof=ddof),
        "initial_loss_mean": float(losses[:, 0].mean()),
        "initial_loss_sd": float(losses[:, 0].std(ddof=ddof)),
        "final_loss_mean": float(losses[:, -1].mean()),
        "final_loss_sd": float(losses[:, -1].std(ddof=ddof)),
        "time_windows_pct": {
            "early_window": [0, early_stop],
            "late_window": [late_start, num_steps],
            "early_all_layers_pct": float(gauge_fractions[:, :early_stop, :].mean() * 100.0),
            "late_all_layers_pct": float(gauge_fractions[:, late_start:, :].mean() * 100.0),
            "early_hidden_layers_pct": float(hidden_layers[:, :early_stop, :].mean() * 100.0) if hidden_exists else float("nan"),
            "late_hidden_layers_pct": float(hidden_layers[:, late_start:, :].mean() * 100.0) if hidden_exists else float("nan"),
            "early_last_layer_pct": float(gauge_fractions[:, :early_stop, -1].mean() * 100.0),
            "late_last_layer_pct": float(gauge_fractions[:, late_start:, -1].mean() * 100.0),
        },
    }


def compute_legacy_rule_checks(results):
    """Preserve the original T1-T6 rules as descriptive continuity checks."""
    summaries = results["summaries"]

    relu_sgd_all = summaries["relu_sgd"]["all_layers_mean_pct"]
    linear_sgd_all = summaries["linear_sgd"]["all_layers_mean_pct"]
    relu_per_layer = summaries["relu_sgd"]["per_layer_mean_pct"]
    late_relu_all = summaries["relu_sgd"]["time_windows_pct"]["late_all_layers_pct"]
    nontrivial_count = int(np.sum(relu_per_layer > 10.0))

    return {
        "T1": {
            "description": "Linear net gauge fraction lies in the legacy 30-70% reference band.",
            "scope": "all_layers",
            "metric_value_pct": float(linear_sgd_all),
            "target": "30 < linear_sgd_overall < 70",
            "pass": bool(30.0 < linear_sgd_all < 70.0),
        },
        "T2": {
            "description": "ReLU gauge fraction exceeds 5% in the legacy all-layer summary.",
            "scope": "all_layers",
            "metric_value_pct": float(relu_sgd_all),
            "target": "relu_sgd_overall > 5",
            "pass": bool(relu_sgd_all > 5.0),
        },
        "T3": {
            "description": "ReLU gauge fraction exceeds 15% in the legacy all-layer summary.",
            "scope": "all_layers",
            "metric_value_pct": float(relu_sgd_all),
            "target": "relu_sgd_overall > 15",
            "pass": bool(relu_sgd_all > 15.0),
        },
        "T4": {
            "description": "ReLU gauge fraction is lower than the linear reference in the all-layer summary.",
            "scope": "all_layers",
            "metric_value_pct": float(relu_sgd_all),
            "reference_value_pct": float(linear_sgd_all),
            "target": "relu_sgd_overall < linear_sgd_overall",
            "pass": bool(relu_sgd_all < linear_sgd_all),
        },
        "T5": {
            "description": "At least 4/6 layers have mean ReLU gauge fraction above 10%.",
            "scope": "per_layer_all_layers",
            "metric_values_pct": relu_per_layer.tolist(),
            "count_above_threshold": nontrivial_count,
            "threshold_pct": 10.0,
            "target": "count >= 4",
            "pass": bool(nontrivial_count >= 4),
        },
        "T6": {
            "description": "Late-training ReLU gauge fraction remains above 5% in the all-layer summary.",
            "scope": "late_all_layers",
            "metric_value_pct": float(late_relu_all),
            "target": "late_relu_all > 5",
            "pass": bool(late_relu_all > 5.0),
        },
    }


# =============================================================================
# MAIN API
# =============================================================================

def run_experiment(config=None, verbose=True):
    """Run the toy experiment and return structured raw results and summaries."""
    merged_config = get_default_config()
    if config is not None:
        merged_config.update(config)
    validate_config(merged_config)

    seeds = _seed_list(merged_config)
    snapshot_steps = [
        step for step in [0, 10, 25, 50, 100, 200, 300, 400, 499]
        if step < merged_config["num_steps"]
    ]

    raw_losses = {name: [] for name in CONDITION_ORDER}
    raw_gauge_fractions = {name: [] for name in CONDITION_ORDER}

    start_time = time.time()

    if verbose:
        print("=" * 88)
        print("KILL_EXPERIMENT_gauge_fraction_relu :: toy gradient gauge-fraction diagnostic")
        print("=" * 88)
        print("Scope: small full-batch NumPy random-regression study; not a broad falsification test.")
        print("Measured quantity: gauge_fraction = ||Q sym(Q^T G)||_F^2 / ||G||_F^2 on the raw gradient.")
        print("Primary analysis: hidden layers. Control analysis: the final output layer.")
        print()

    for seed_index, run_seed in enumerate(seeds, start=1):
        rng = np.random.RandomState(run_seed)

        if verbose:
            print(f"--- Seed {run_seed} ({seed_index}/{len(seeds)}) ---")

        X = rng.randn(merged_config["input_dim"], merged_config["num_samples"]) * merged_config["data_scale"]
        Y = rng.randn(merged_config["output_dim"], merged_config["num_samples"]) * merged_config["data_scale"]
        weights_init = init_weights(rng, merged_config["num_layers"], merged_config["hidden_dim"])

        seed_results = {}
        for condition_name in CONDITION_ORDER:
            spec = CONDITION_SPECS[condition_name]
            if verbose:
                print(f"  Training {CONDITION_LABELS[condition_name]}...", flush=True)
            losses, gauge_fractions = train_with_gauge_tracking(
                spec["net_type"],
                weights_init,
                X,
                Y,
                merged_config[spec["lr_key"]],
                spec["optimizer"],
                merged_config["num_steps"],
                merged_config["momentum"],
                merged_config["ns_iters"],
            )
            raw_losses[condition_name].append(losses)
            raw_gauge_fractions[condition_name].append(gauge_fractions)
            seed_results[condition_name] = gauge_fractions

        if verbose:
            def seed_mean(condition_name, layer_selector):
                data = seed_results[condition_name][:, layer_selector]
                return float(np.mean(data) * 100.0)

            hidden_selector = slice(0, merged_config["num_layers"] - 1)
            print("  Mean gauge fraction across this seed (%):")
            print(
                "    Hidden layers (primary): "
                f"ReLU+SGD {seed_mean('relu_sgd', hidden_selector):5.1f} | "
                f"ReLU+Muon {seed_mean('relu_muon', hidden_selector):5.1f} | "
                f"Linear+SGD {seed_mean('linear_sgd', hidden_selector):5.1f} | "
                f"Linear+Muon {seed_mean('linear_muon', hidden_selector):5.1f}"
            )
            print(
                "    Last layer (control):    "
                f"ReLU+SGD {seed_mean('relu_sgd', -1):5.1f} | "
                f"ReLU+Muon {seed_mean('relu_muon', -1):5.1f} | "
                f"Linear+SGD {seed_mean('linear_sgd', -1):5.1f} | "
                f"Linear+Muon {seed_mean('linear_muon', -1):5.1f}"
            )
            print()

    raw_losses = {name: np.array(values) for name, values in raw_losses.items()}
    raw_gauge_fractions = {name: np.array(values) for name, values in raw_gauge_fractions.items()}

    summaries = {
        name: summarize_condition(raw_losses[name], raw_gauge_fractions[name])
        for name in CONDITION_ORDER
    }

    results = {
        "experiment_id": "KILL_EXPERIMENT_gauge_fraction_relu",
        "title": "Toy diagnostic: gradient gauge fraction in a ReLU MLP",
        "scope_note": (
            "Toy full-batch NumPy random-regression experiment. Useful as a narrow "
            "mechanistic diagnostic, not as a broad falsification of Muon."
        ),
        "measured_quantity": "gauge_fraction = ||Q sym(Q^T G)||_F^2 / ||G||_F^2 with Q = polar(W)",
        "not_measured": [
            "Whether Muon's practical update is exactly the tangent projection G_tangent.",
            "Behavior outside this one-width, one-depth, random-regression toy setup.",
            "Formal statistical significance; the current summaries are descriptive over n=3 seeds.",
        ],
        "config": merged_config,
        "seeds": seeds,
        "condition_order": list(CONDITION_ORDER),
        "condition_labels": dict(CONDITION_LABELS),
        "condition_colors": dict(CONDITION_COLORS),
        "hidden_layer_indices": list(range(merged_config["num_layers"] - 1)),
        "last_layer_index": merged_config["num_layers"] - 1,
        "snapshot_steps": snapshot_steps,
        "losses": raw_losses,
        "gauge_fractions": raw_gauge_fractions,
        "summaries": summaries,
        "runtime_seconds": float(time.time() - start_time),
    }
    results["legacy_rule_checks"] = compute_legacy_rule_checks(results)
    results["legacy_rule_pass_count"] = int(
        sum(check["pass"] for check in results["legacy_rule_checks"].values())
    )
    results["legacy_rule_total"] = len(results["legacy_rule_checks"])
    return results


def print_report(results):
    """Print a calibrated text report from a results dictionary."""
    config = results["config"]
    summaries = results["summaries"]
    rule_checks = results["legacy_rule_checks"]

    print("\n" + "=" * 88)
    print("AGGREGATE REPORT")
    print("=" * 88)
    print(results["scope_note"])
    print(f"Runtime: {results['runtime_seconds']:.2f} s")
    print(
        "Config: "
        f"layers={config['num_layers']}, width={config['hidden_dim']}, samples={config['num_samples']}, "
        f"steps={config['num_steps']}, seeds={config['num_seeds']}"
    )
    print(f"Seeds: {results['seeds']}")
    print("Measured quantity: " + results["measured_quantity"])
    print("Not directly measured:")
    for note in results["not_measured"]:
        print(f"  - {note}")

    print("\nAggregate gauge-fraction summary (% of gradient energy)")
    print("-" * 88)
    header = (
        f"{'Condition':<18}"
        f"{'All layers':>18}"
        f"{'Hidden layers':>18}"
        f"{'Last layer':>18}"
    )
    print(header)
    print("-" * 88)
    for condition_name in results["condition_order"]:
        summary = summaries[condition_name]
        print(
            f"{results['condition_labels'][condition_name]:<18}"
            f"{summary['all_layers_mean_pct']:>8.2f} ± {summary['all_layers_sd_pct']:<6.2f}"
            f"{summary['hidden_layers_mean_pct']:>8.2f} ± {summary['hidden_layers_sd_pct']:<6.2f}"
            f"{summary['last_layer_mean_pct']:>8.2f} ± {summary['last_layer_sd_pct']:<6.2f}"
        )

    print("\nPer-layer mean gauge fraction (%; averaged over seeds and steps)")
    print("-" * 88)
    layer_header = f"{'Layer':>6}"
    for condition_name in results["condition_order"]:
        layer_header += f"{results['condition_labels'][condition_name]:>18}"
    print(layer_header)
    print("-" * 88)
    for layer in range(config["num_layers"]):
        row = f"{layer + 1:>6}"
        for condition_name in results["condition_order"]:
            row += f"{summaries[condition_name]['per_layer_mean_pct'][layer]:>17.2f}%"
        if layer == config["num_layers"] - 1:
            row += "   <- final output layer (control)"
        print(row)

    print("\nEarly-vs-late gauge fraction (%; hidden layers primary, last layer control)")
    print("-" * 88)
    print(
        f"{'Condition':<18}"
        f"{'Hidden early':>14}"
        f"{'Hidden late':>14}"
        f"{'Last early':>14}"
        f"{'Last late':>14}"
    )
    print("-" * 88)
    for condition_name in results["condition_order"]:
        tw = summaries[condition_name]["time_windows_pct"]
        print(
            f"{results['condition_labels'][condition_name]:<18}"
            f"{tw['early_hidden_layers_pct']:>13.2f}%"
            f"{tw['late_hidden_layers_pct']:>13.2f}%"
            f"{tw['early_last_layer_pct']:>13.2f}%"
            f"{tw['late_last_layer_pct']:>13.2f}%"
        )

    print("\nLegacy T1-T6 rule-based checks (kept for continuity; these are not formal hypothesis tests)")
    print("-" * 88)
    for rule_name in ["T1", "T2", "T3", "T4", "T5", "T6"]:
        rule = rule_checks[rule_name]
        status = "PASS" if rule["pass"] else "FAIL"
        print(f"{rule_name}: {status} | {rule['description']}")
        if rule_name == "T4":
            print(
                f"    ReLU all-layer mean = {rule['metric_value_pct']:.2f}%, "
                f"Linear all-layer mean = {rule['reference_value_pct']:.2f}%"
            )
        elif rule_name == "T5":
            values = ", ".join(f"{value:.2f}%" for value in rule["metric_values_pct"])
            print(
                f"    Per-layer means = [{values}] | count above {rule['threshold_pct']:.1f}% = "
                f"{rule['count_above_threshold']}"
            )
        else:
            print(f"    Metric value = {rule['metric_value_pct']:.2f}%")

    relu_hidden = summaries["relu_sgd"]["hidden_layers_mean_pct"]
    linear_hidden = summaries["linear_sgd"]["hidden_layers_mean_pct"]
    relu_last = summaries["relu_sgd"]["last_layer_mean_pct"]

    if relu_hidden > 15.0:
        hidden_reading = "substantial within this toy setup"
    elif relu_hidden > 5.0:
        hidden_reading = "present but modest within this toy setup"
    else:
        hidden_reading = "small within this toy setup"

    print("\n" + "=" * 88)
    print("CALIBRATED CONCLUSION")
    print("=" * 88)
    print(
        dedent(
            f"""
            Primary hidden-layer headline (ReLU + SGD): {relu_hidden:.2f}%
            Hidden-layer linear reference (Linear + SGD): {linear_hidden:.2f}%
            Last-layer control (ReLU + SGD): {relu_last:.2f}%

            Reading: the hidden-layer Stiefel-normal fraction of the raw gradient is
            {hidden_reading}. This keeps the toy gauge-fraction signal alive in this
            nonlinear NumPy model, but it does not by itself establish Muon's full
            mechanism, nor does it generalize beyond the present width-32, depth-6,
            random-regression setting.

            Legacy rule checks passed: {results['legacy_rule_pass_count']}/{results['legacy_rule_total']}.
            """
        ).strip()
    )
    print("=" * 88)


def main():
    results = run_experiment(verbose=True)
    print_report(results)


if __name__ == "__main__":
    main()
