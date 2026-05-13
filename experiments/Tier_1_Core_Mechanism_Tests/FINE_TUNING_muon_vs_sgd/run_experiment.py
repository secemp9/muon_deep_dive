#!/usr/bin/env python3
"""
Experiment 2.5: Fine-tuning vs scratch in a deep-linear toy model
=================================================================

Question:
  In this toy setup, does Muon behave differently from SGD when adapting a
  pre-trained checkpoint versus training from scratch after a target shift?

Protocol:
  Phase 1 -- Pre-training (shared):
      Train a 4-layer deep linear net (32x32) from random init with SGD for
      500 steps on the original target. Save checkpoint.

  Phase 2 -- Fine-tuning (from checkpoint):
      Modify 20% of the target matrix -> W_target_modified.
      From the checkpoint, fine-tune 200 steps with:
        (a) SGD   lr=0.01
        (b) Muon  lr=0.005

  Phase 3 -- From-scratch comparison:
      Train from random init for 700 steps on W_target_modified with both
      optimizers.

Measured quantities:
  - Pre-training loss curves
  - Fine-tuning loss curves (SGD vs Muon)
  - From-scratch loss curves (SGD vs Muon)
  - Fine-tuning checkpoint parameter distance ||theta_t - theta_ckpt||_2

Important scope limitations:
  - This script does NOT measure Lyapunov exponents.
  - It does NOT directly diagnose chaos.
  - It does NOT measure Hessian geometry.
  - Checkpoint distance is a total parameter-space metric, not a direct
    gauge-direction or function-space decomposition.
"""

from __future__ import annotations

import time
import numpy as np


DEFAULT_CONFIG = {
    "seed": 42,
    "seed_stride": 137,
    "dim": 32,
    "num_layers": 4,
    "batch_size": 64,
    "input_scale": 0.3,
    "target_scale": 0.5,
    "init_scale": 0.1,
    "pretrain_steps": 500,
    "pretrain_lr": 0.01,
    "finetune_steps": 200,
    "sgd_finetune_lr": 0.01,
    "muon_finetune_lr": 0.005,
    "scratch_steps": 700,
    "sgd_scratch_lr": 0.01,
    "muon_scratch_lr": 0.005,
    "momentum": 0.9,
    "ns_iters": 5,
    "modify_frac": 0.20,
    "num_runs": 5,
    "threshold_fraction": 0.50,
}


def get_default_config():
    """Return a copy of the default experiment configuration."""
    return dict(DEFAULT_CONFIG)


def build_run_seeds(config):
    return [config["seed"] + idx * config["seed_stride"] for idx in range(config["num_runs"])]


def make_base_problem(config):
    """Construct the fixed data/target pair used across runs."""
    base_rng = np.random.RandomState(config["seed"])
    dim = config["dim"]
    batch_size = config["batch_size"]

    x_data = base_rng.randn(dim, batch_size) * config["input_scale"]
    w_target_original = base_rng.randn(dim, dim) * config["target_scale"]

    return {
        "X_data": x_data,
        "W_target_original": w_target_original,
        "summary": {
            "x_shape": x_data.shape,
            "target_shape": w_target_original.shape,
            "input_scale": config["input_scale"],
            "target_scale": config["target_scale"],
        },
    }


# =============================================================================
# TARGET MATRICES
# =============================================================================


def make_modified_target(w_original, frac, rng, target_scale):
    """Change `frac` of entries to new random values."""
    w_mod = w_original.copy()
    n_entries = w_mod.size
    n_change = int(frac * n_entries)
    indices = rng.choice(n_entries, size=n_change, replace=False)
    flat = w_mod.ravel()
    flat[indices] = rng.randn(n_change) * target_scale
    return w_mod


# =============================================================================
# NETWORK HELPERS
# =============================================================================


def init_weights(num_layers, dim, rng, init_scale):
    """Initialize layers near identity for stability."""
    weights = []
    for _ in range(num_layers):
        w = np.eye(dim) + rng.randn(dim, dim) * init_scale
        weights.append(w.copy())
    return weights


def forward_linear(weights, x):
    """Forward pass through deep linear net."""
    out = x.copy()
    for w in weights:
        out = w @ out
    return out


def compute_loss(weights, x, y_target):
    """Quadratic loss: 0.5 * ||f(X) - Y||^2 / N."""
    y_pred = forward_linear(weights, x)
    diff = y_pred - y_target
    return 0.5 * np.mean(np.sum(diff**2, axis=0))


def compute_gradients(weights, x, y_target):
    """Backprop through deep linear net for quadratic loss."""
    num_layers = len(weights)
    n = x.shape[1]

    activations = [x.copy()]
    for w in weights:
        activations.append(w @ activations[-1])

    delta = (activations[-1] - y_target) / n

    grads = [None] * num_layers
    for layer_idx in range(num_layers - 1, -1, -1):
        grads[layer_idx] = delta @ activations[layer_idx].T
        if layer_idx > 0:
            delta = weights[layer_idx].T @ delta

    return grads


def flatten_weights(weights):
    return np.concatenate([w.ravel() for w in weights])


def checkpoint_distance(weights, checkpoint_weights):
    """Euclidean distance over all layer parameters relative to the checkpoint."""
    flat_w = flatten_weights(weights)
    flat_c = flatten_weights(checkpoint_weights)
    return np.linalg.norm(flat_w - flat_c)


def copy_weights(weights):
    return [w.copy() for w in weights]


# =============================================================================
# OPTIMIZERS
# =============================================================================


def newton_schulz_ortho(matrix, n_iters=5):
    """Approximate the orthogonal polar factor via Newton-Schulz iteration."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = matrix / (np.linalg.norm(matrix, ord="fro") + 1e-7)
    if x.shape[0] > x.shape[1]:
        x = x.T
        transposed = True
    else:
        transposed = False

    identity = np.eye(x.shape[0])
    for _ in range(n_iters):
        gram = x @ x.T
        x = (a * identity + b * gram + c * gram @ gram) @ x

    if transposed:
        x = x.T
    return x


class SGDOptimizer:
    def __init__(self, weights, lr, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = [np.zeros_like(w) for w in weights]

    def step(self, weights, grads):
        for idx in range(len(weights)):
            self.velocity[idx] = self.momentum * self.velocity[idx] + grads[idx]
            weights[idx] -= self.lr * self.velocity[idx]
        return weights


class MuonOptimizer:
    def __init__(self, weights, lr, momentum=0.9, ns_iters=5):
        self.lr = lr
        self.momentum = momentum
        self.ns_iters = ns_iters
        self.velocity = [np.zeros_like(w) for w in weights]

    def step(self, weights, grads):
        for idx in range(len(weights)):
            self.velocity[idx] = self.momentum * self.velocity[idx] + grads[idx]
            ortho_update = newton_schulz_ortho(self.velocity[idx], self.ns_iters)
            weights[idx] -= self.lr * ortho_update
        return weights


# =============================================================================
# TRAINING LOOP
# =============================================================================


def train(weights, optimizer, w_target, x, n_steps, checkpoint_weights=None):
    """Train and record loss + checkpoint distance at each step."""
    losses = []
    distances = []
    y_target = w_target @ x

    for _ in range(n_steps):
        loss = compute_loss(weights, x, y_target)
        losses.append(loss)

        if checkpoint_weights is not None:
            distances.append(checkpoint_distance(weights, checkpoint_weights))

        grads = compute_gradients(weights, x, y_target)
        weights = optimizer.step(weights, grads)

    final_loss = compute_loss(weights, x, y_target)
    losses.append(final_loss)
    if checkpoint_weights is not None:
        distances.append(checkpoint_distance(weights, checkpoint_weights))

    losses = np.asarray(losses, dtype=float)
    distances = np.asarray(distances, dtype=float) if distances else None
    return weights, losses, distances


# =============================================================================
# ANALYSIS HELPERS
# =============================================================================


def steps_to_threshold(losses, frac=0.5):
    threshold = losses[0] * frac
    for idx, loss in enumerate(losses):
        if loss <= threshold:
            return idx
    return len(losses)


def validate_curve(name, curve, expected_length):
    if curve.shape != (expected_length,):
        raise ValueError(f"{name} has shape {curve.shape}, expected {(expected_length,)}")
    if not np.all(np.isfinite(curve)):
        raise ValueError(f"{name} contains non-finite values")


def validate_run_result(run_result, config):
    validate_curve("pretrain_losses", run_result["pretrain_losses"], config["pretrain_steps"] + 1)
    validate_curve("ft_sgd_losses", run_result["ft_sgd_losses"], config["finetune_steps"] + 1)
    validate_curve("ft_muon_losses", run_result["ft_muon_losses"], config["finetune_steps"] + 1)
    validate_curve("ft_sgd_dists", run_result["ft_sgd_dists"], config["finetune_steps"] + 1)
    validate_curve("ft_muon_dists", run_result["ft_muon_dists"], config["finetune_steps"] + 1)
    validate_curve("scratch_sgd_losses", run_result["scratch_sgd_losses"], config["scratch_steps"] + 1)
    validate_curve("scratch_muon_losses", run_result["scratch_muon_losses"], config["scratch_steps"] + 1)


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================


def run_single_experiment(run_seed, config, base_problem):
    """Run one full experiment with a given seed."""
    rng = np.random.RandomState(run_seed)
    x_data = base_problem["X_data"]
    w_target_original = base_problem["W_target_original"]

    w_target_mod = make_modified_target(
        w_target_original,
        config["modify_frac"],
        rng,
        config["target_scale"],
    )

    weights_init = init_weights(
        config["num_layers"],
        config["dim"],
        rng,
        config["init_scale"],
    )
    pretrain_opt = SGDOptimizer(
        copy_weights(weights_init),
        lr=config["pretrain_lr"],
        momentum=config["momentum"],
    )
    weights_pretrained, pretrain_losses, _ = train(
        copy_weights(weights_init),
        pretrain_opt,
        w_target_original,
        x_data,
        config["pretrain_steps"],
    )
    checkpoint = copy_weights(weights_pretrained)

    ft_sgd_opt = SGDOptimizer(
        copy_weights(checkpoint),
        lr=config["sgd_finetune_lr"],
        momentum=config["momentum"],
    )
    _, ft_sgd_losses, ft_sgd_dists = train(
        copy_weights(checkpoint),
        ft_sgd_opt,
        w_target_mod,
        x_data,
        config["finetune_steps"],
        checkpoint_weights=checkpoint,
    )

    ft_muon_opt = MuonOptimizer(
        copy_weights(checkpoint),
        lr=config["muon_finetune_lr"],
        momentum=config["momentum"],
        ns_iters=config["ns_iters"],
    )
    _, ft_muon_losses, ft_muon_dists = train(
        copy_weights(checkpoint),
        ft_muon_opt,
        w_target_mod,
        x_data,
        config["finetune_steps"],
        checkpoint_weights=checkpoint,
    )

    scratch_init = init_weights(
        config["num_layers"],
        config["dim"],
        rng,
        config["init_scale"],
    )
    scratch_sgd_opt = SGDOptimizer(
        copy_weights(scratch_init),
        lr=config["sgd_scratch_lr"],
        momentum=config["momentum"],
    )
    _, scratch_sgd_losses, _ = train(
        copy_weights(scratch_init),
        scratch_sgd_opt,
        w_target_mod,
        x_data,
        config["scratch_steps"],
    )

    scratch_muon_opt = MuonOptimizer(
        copy_weights(scratch_init),
        lr=config["muon_scratch_lr"],
        momentum=config["momentum"],
        ns_iters=config["ns_iters"],
    )
    _, scratch_muon_losses, _ = train(
        copy_weights(scratch_init),
        scratch_muon_opt,
        w_target_mod,
        x_data,
        config["scratch_steps"],
    )

    threshold_fraction = config["threshold_fraction"]
    run_result = {
        "run_seed": run_seed,
        "target_shift_frobenius": float(np.linalg.norm(w_target_mod - w_target_original)),
        "pretrain_losses": pretrain_losses,
        "ft_sgd_losses": ft_sgd_losses,
        "ft_muon_losses": ft_muon_losses,
        "ft_sgd_dists": ft_sgd_dists,
        "ft_muon_dists": ft_muon_dists,
        "scratch_sgd_losses": scratch_sgd_losses,
        "scratch_muon_losses": scratch_muon_losses,
        "pretrain_final_loss": float(pretrain_losses[-1]),
        "ft_sgd_final_loss": float(ft_sgd_losses[-1]),
        "ft_muon_final_loss": float(ft_muon_losses[-1]),
        "ft_sgd_final_checkpoint_distance": float(ft_sgd_dists[-1]),
        "ft_muon_final_checkpoint_distance": float(ft_muon_dists[-1]),
        "scratch_sgd_final_loss": float(scratch_sgd_losses[-1]),
        "scratch_muon_final_loss": float(scratch_muon_losses[-1]),
        "threshold_steps": {
            "ft_sgd_half_loss": int(steps_to_threshold(ft_sgd_losses, threshold_fraction)),
            "ft_muon_half_loss": int(steps_to_threshold(ft_muon_losses, threshold_fraction)),
            "scratch_sgd_half_loss": int(steps_to_threshold(scratch_sgd_losses, threshold_fraction)),
            "scratch_muon_half_loss": int(steps_to_threshold(scratch_muon_losses, threshold_fraction)),
        },
    }
    validate_run_result(run_result, config)
    return run_result


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    sd = np.std(values, ddof=1) if values.size > 1 else 0.0
    return {
        "mean": float(np.mean(values)),
        "sd": float(sd),
        "n": int(values.size),
        "values": values,
    }


def stack_curve_stats(curves):
    stacked = np.stack(curves, axis=0)
    sd = np.std(stacked, axis=0, ddof=1) if stacked.shape[0] > 1 else np.zeros_like(stacked[0])
    return {
        "mean": np.mean(stacked, axis=0),
        "sd": sd,
        "n": int(stacked.shape[0]),
        "all_curves": stacked,
    }


def aggregate_results(run_results, config):
    curve_stats = {
        "pretrain_losses": stack_curve_stats([r["pretrain_losses"] for r in run_results]),
        "ft_sgd_losses": stack_curve_stats([r["ft_sgd_losses"] for r in run_results]),
        "ft_muon_losses": stack_curve_stats([r["ft_muon_losses"] for r in run_results]),
        "ft_sgd_dists": stack_curve_stats([r["ft_sgd_dists"] for r in run_results]),
        "ft_muon_dists": stack_curve_stats([r["ft_muon_dists"] for r in run_results]),
        "scratch_sgd_losses": stack_curve_stats([r["scratch_sgd_losses"] for r in run_results]),
        "scratch_muon_losses": stack_curve_stats([r["scratch_muon_losses"] for r in run_results]),
    }

    final_metrics = {
        "pretrain_final_loss": mean_sd([r["pretrain_final_loss"] for r in run_results]),
        "ft_sgd_final_loss": mean_sd([r["ft_sgd_final_loss"] for r in run_results]),
        "ft_muon_final_loss": mean_sd([r["ft_muon_final_loss"] for r in run_results]),
        "ft_sgd_final_checkpoint_distance": mean_sd(
            [r["ft_sgd_final_checkpoint_distance"] for r in run_results]
        ),
        "ft_muon_final_checkpoint_distance": mean_sd(
            [r["ft_muon_final_checkpoint_distance"] for r in run_results]
        ),
        "scratch_sgd_final_loss": mean_sd([r["scratch_sgd_final_loss"] for r in run_results]),
        "scratch_muon_final_loss": mean_sd([r["scratch_muon_final_loss"] for r in run_results]),
        "target_shift_frobenius": mean_sd([r["target_shift_frobenius"] for r in run_results]),
    }

    threshold_steps = {
        "ft_sgd_half_loss": mean_sd([r["threshold_steps"]["ft_sgd_half_loss"] for r in run_results]),
        "ft_muon_half_loss": mean_sd([r["threshold_steps"]["ft_muon_half_loss"] for r in run_results]),
        "scratch_sgd_half_loss": mean_sd(
            [r["threshold_steps"]["scratch_sgd_half_loss"] for r in run_results]
        ),
        "scratch_muon_half_loss": mean_sd(
            [r["threshold_steps"]["scratch_muon_half_loss"] for r in run_results]
        ),
    }

    paired_differences = {
        "finetune_final_loss_muon_minus_sgd": mean_sd(
            [r["ft_muon_final_loss"] - r["ft_sgd_final_loss"] for r in run_results]
        ),
        "finetune_final_checkpoint_distance_muon_minus_sgd": mean_sd(
            [
                r["ft_muon_final_checkpoint_distance"]
                - r["ft_sgd_final_checkpoint_distance"]
                for r in run_results
            ]
        ),
        "scratch_final_loss_muon_minus_sgd": mean_sd(
            [r["scratch_muon_final_loss"] - r["scratch_sgd_final_loss"] for r in run_results]
        ),
    }

    early_vs_late = {}
    for label, key in (("sgd", "ft_sgd_losses"), ("muon", "ft_muon_losses")):
        early_drops = [r[key][0] - r[key][50] for r in run_results]
        late_drops = [r[key][150] - r[key][200] for r in run_results]
        early_vs_late[label] = {
            "early_drop_0_to_50": mean_sd(early_drops),
            "late_drop_150_to_200": mean_sd(late_drops),
        }

    h1_confirmed = (
        final_metrics["ft_muon_final_checkpoint_distance"]["mean"]
        > final_metrics["ft_sgd_final_checkpoint_distance"]["mean"]
    )
    h2_confirmed = (
        final_metrics["ft_muon_final_loss"]["mean"]
        > final_metrics["ft_sgd_final_loss"]["mean"]
    )
    h3_confirmed = final_metrics["scratch_muon_final_loss"]["mean"] < final_metrics["scratch_sgd_final_loss"]["mean"]

    hypothesis_tests = {
        "H1": {
            "question": "By the end of fine-tuning, does Muon end farther from the checkpoint in parameter space than SGD?",
            "measured_quantity": "Final checkpoint parameter distance over concatenated layer weights",
            "lhs_name": "Muon final checkpoint distance",
            "lhs_mean": final_metrics["ft_muon_final_checkpoint_distance"]["mean"],
            "rhs_name": "SGD final checkpoint distance",
            "rhs_mean": final_metrics["ft_sgd_final_checkpoint_distance"]["mean"],
            "confirmed": bool(h1_confirmed),
            "verdict": "CONFIRMED" if h1_confirmed else "REJECTED",
        },
        "H2": {
            "question": "Does Muon finish fine-tuning with a higher final loss than SGD?",
            "measured_quantity": "Final fine-tuning loss at the 200-step budget",
            "lhs_name": "Muon final fine-tune loss",
            "lhs_mean": final_metrics["ft_muon_final_loss"]["mean"],
            "rhs_name": "SGD final fine-tune loss",
            "rhs_mean": final_metrics["ft_sgd_final_loss"]["mean"],
            "confirmed": bool(h2_confirmed),
            "verdict": "CONFIRMED" if h2_confirmed else "REJECTED",
        },
        "H3": {
            "question": "Does Muon finish from-scratch training with a lower final loss than SGD?",
            "measured_quantity": "Final from-scratch loss at the 700-step budget",
            "lhs_name": "Muon final scratch loss",
            "lhs_mean": final_metrics["scratch_muon_final_loss"]["mean"],
            "rhs_name": "SGD final scratch loss",
            "rhs_mean": final_metrics["scratch_sgd_final_loss"]["mean"],
            "confirmed": bool(h3_confirmed),
            "verdict": "CONFIRMED" if h3_confirmed else "REJECTED",
        },
    }

    if threshold_steps["ft_muon_half_loss"]["mean"] > threshold_steps["ft_sgd_half_loss"]["mean"]:
        early_speed_clause = "Muon is slower than SGD on the fine-tuning half-loss threshold"
    elif threshold_steps["ft_muon_half_loss"]["mean"] < threshold_steps["ft_sgd_half_loss"]["mean"]:
        early_speed_clause = "Muon is faster than SGD on the fine-tuning half-loss threshold"
    else:
        early_speed_clause = "Muon and SGD are tied on the fine-tuning half-loss threshold"

    fine_tune_endpoint_clause = (
        "Muon finishes fine-tuning with a higher final loss than SGD"
        if h2_confirmed
        else "Muon finishes fine-tuning with a lower final loss than SGD"
    )
    scratch_clause = (
        "Muon finishes from-scratch training with a lower final loss than SGD"
        if h3_confirmed
        else "Muon does not finish from-scratch training with a lower final loss than SGD"
    )
    distance_clause = (
        "Muon ends farther from the checkpoint in parameter space by the fine-tuning endpoint"
        if h1_confirmed
        else "Muon does not end farther from the checkpoint in parameter space by the fine-tuning endpoint"
    )

    overall_conclusion = (
        f"{early_speed_clause}; {distance_clause}; {fine_tune_endpoint_clause}; and {scratch_clause}. "
        "This toy experiment therefore distinguishes slower early adaptation from final-loss performance, "
        "and checkpoint parameter distance should not be interpreted as a direct gauge-only metric."
    )

    return {
        "curve_stats": curve_stats,
        "final_metrics": final_metrics,
        "threshold_steps": threshold_steps,
        "paired_differences": paired_differences,
        "early_vs_late_finetune_loss_drop": early_vs_late,
        "hypothesis_tests": hypothesis_tests,
        "overall_conclusion": overall_conclusion,
    }


def run_experiment(config_overrides=None, verbose=False):
    """Run the full experiment and return structured results."""
    config = get_default_config()
    if config_overrides:
        config.update(config_overrides)

    base_problem = make_base_problem(config)
    run_seeds = build_run_seeds(config)

    start_time = time.perf_counter()
    run_results = []
    for run_idx, run_seed in enumerate(run_seeds, start=1):
        if verbose:
            print(f"  Run {run_idx}/{config['num_runs']} (seed={run_seed})...")
        run_results.append(run_single_experiment(run_seed, config, base_problem))
    runtime_seconds = time.perf_counter() - start_time

    aggregates = aggregate_results(run_results, config)

    return {
        "config": config,
        "run_seeds": run_seeds,
        "runtime_seconds": runtime_seconds,
        "base_problem_summary": base_problem["summary"],
        "per_run_results": run_results,
        "aggregates": {
            "curve_stats": aggregates["curve_stats"],
            "final_metrics": aggregates["final_metrics"],
            "threshold_steps": aggregates["threshold_steps"],
            "paired_differences": aggregates["paired_differences"],
            "early_vs_late_finetune_loss_drop": aggregates["early_vs_late_finetune_loss_drop"],
        },
        "hypothesis_tests": aggregates["hypothesis_tests"],
        "overall_conclusion": aggregates["overall_conclusion"],
        "limitations": [
            "No Lyapunov exponent is measured.",
            "No direct chaos diagnostic is measured.",
            "No Hessian or local-curvature quantity is measured.",
            "Checkpoint distance is a parameter-space proxy, not a direct gauge-direction metric.",
        ],
    }


def format_mean_sd(summary, precision=6):
    return f"{summary['mean']:.{precision}f} +/- {summary['sd']:.{precision}f}"


def print_curve_snapshot_table(results):
    curve_stats = results["aggregates"]["curve_stats"]

    ft_sgd_curve = curve_stats["ft_sgd_losses"]["mean"]
    ft_muon_curve = curve_stats["ft_muon_losses"]["mean"]
    ft_sgd_dist_curve = curve_stats["ft_sgd_dists"]["mean"]
    ft_muon_dist_curve = curve_stats["ft_muon_dists"]["mean"]
    sc_sgd_curve = curve_stats["scratch_sgd_losses"]["mean"]
    sc_muon_curve = curve_stats["scratch_muon_losses"]["mean"]

    print()
    print("-" * 80)
    print("LOSS CURVE SNAPSHOTS (mean over runs)")
    print("-" * 80)
    print()
    print("Fine-tuning from checkpoint:")
    print(f"  {'Step':>6}  {'SGD loss':>12}  {'Muon loss':>12}  {'SGD dist':>12}  {'Muon dist':>12}")
    for step_idx in [0, 10, 25, 50, 100, 150, 200]:
        print(
            f"  {step_idx:>6}  {ft_sgd_curve[step_idx]:>12.6f}  {ft_muon_curve[step_idx]:>12.6f}  "
            f"{ft_sgd_dist_curve[step_idx]:>12.4f}  {ft_muon_dist_curve[step_idx]:>12.4f}"
        )

    print()
    print("From scratch:")
    print(f"  {'Step':>6}  {'SGD loss':>12}  {'Muon loss':>12}")
    for step_idx in [0, 50, 100, 200, 350, 500, 700]:
        print(f"  {step_idx:>6}  {sc_sgd_curve[step_idx]:>12.6f}  {sc_muon_curve[step_idx]:>12.6f}")


def print_results_report(results):
    config = results["config"]
    final_metrics = results["aggregates"]["final_metrics"]
    threshold_steps = results["aggregates"]["threshold_steps"]
    paired_differences = results["aggregates"]["paired_differences"]
    early_vs_late = results["aggregates"]["early_vs_late_finetune_loss_drop"]
    hypothesis_tests = results["hypothesis_tests"]

    print("=" * 80)
    print("Experiment 2.5: Fine-tuning vs scratch in a deep-linear toy model")
    print("=" * 80)
    print()
    print("Question:")
    print("  Does Muon adapt differently from SGD when starting from a pre-trained")
    print("  checkpoint versus training from scratch after a target perturbation?")
    print()
    print("Scope note:")
    print("  This run measures losses and checkpoint parameter distance only.")
    print("  It does not directly measure Lyapunov exponents, chaos, Hessians,")
    print("  or gauge-only motion.")
    print()
    print("Configuration:")
    print(
        f"  dim={config['dim']}, layers={config['num_layers']}, batch={config['batch_size']}, "
        f"pretrain_steps={config['pretrain_steps']}, finetune_steps={config['finetune_steps']}, "
        f"scratch_steps={config['scratch_steps']}, num_runs={config['num_runs']}"
    )
    print(f"  run_seeds={results['run_seeds']}")
    print(f"  runtime={results['runtime_seconds']:.2f}s")

    print_curve_snapshot_table(results)

    print()
    print("=" * 80)
    print("FINAL METRICS (mean +/- sample sd)")
    print("=" * 80)
    print(f"Pre-training final loss (SGD):      {format_mean_sd(final_metrics['pretrain_final_loss'])}")
    print(f"Fine-tune final loss (SGD):         {format_mean_sd(final_metrics['ft_sgd_final_loss'])}")
    print(f"Fine-tune final loss (Muon):        {format_mean_sd(final_metrics['ft_muon_final_loss'])}")
    print(
        f"Fine-tune final ckpt dist (SGD):    {format_mean_sd(final_metrics['ft_sgd_final_checkpoint_distance'], precision=4)}"
    )
    print(
        f"Fine-tune final ckpt dist (Muon):   {format_mean_sd(final_metrics['ft_muon_final_checkpoint_distance'], precision=4)}"
    )
    print(f"Scratch final loss (SGD):           {format_mean_sd(final_metrics['scratch_sgd_final_loss'])}")
    print(f"Scratch final loss (Muon):          {format_mean_sd(final_metrics['scratch_muon_final_loss'])}")
    print()
    print("Threshold summary: steps to reach 50% of initial loss")
    print(f"  Fine-tune SGD:      {format_mean_sd(threshold_steps['ft_sgd_half_loss'], precision=1)}")
    print(f"  Fine-tune Muon:     {format_mean_sd(threshold_steps['ft_muon_half_loss'], precision=1)}")
    print(f"  Scratch SGD:        {format_mean_sd(threshold_steps['scratch_sgd_half_loss'], precision=1)}")
    print(f"  Scratch Muon:       {format_mean_sd(threshold_steps['scratch_muon_half_loss'], precision=1)}")
    print()
    print("Paired differences (Muon - SGD; mean +/- sample sd)")
    print(
        f"  Fine-tune final loss:             {format_mean_sd(paired_differences['finetune_final_loss_muon_minus_sgd'])}"
    )
    print(
        "  Fine-tune final ckpt distance:    "
        f"{format_mean_sd(paired_differences['finetune_final_checkpoint_distance_muon_minus_sgd'], precision=4)}"
    )
    print(
        f"  Scratch final loss:               {format_mean_sd(paired_differences['scratch_final_loss_muon_minus_sgd'])}"
    )
    print()
    print("Fine-tuning loss-drop summary (mean +/- sample sd)")
    print(
        "  SGD  early 0->50:  "
        f"{format_mean_sd(early_vs_late['sgd']['early_drop_0_to_50'])}"
        f" | late 150->200: {format_mean_sd(early_vs_late['sgd']['late_drop_150_to_200'])}"
    )
    print(
        "  Muon early 0->50:  "
        f"{format_mean_sd(early_vs_late['muon']['early_drop_0_to_50'])}"
        f" | late 150->200: {format_mean_sd(early_vs_late['muon']['late_drop_150_to_200'])}"
    )

    print()
    print("=" * 80)
    print("HYPOTHESIS CHECKS")
    print("=" * 80)
    for key in ["H1", "H2", "H3"]:
        test = hypothesis_tests[key]
        print(f"[{key}] {test['question']}")
        print(f"     measured quantity: {test['measured_quantity']}")
        print(f"     {test['lhs_name']} = {test['lhs_mean']:.6f}")
        print(f"     {test['rhs_name']} = {test['rhs_mean']:.6f}")
        print(f"     --> {test['verdict']}")
        print()

    print("Calibrated overall conclusion:")
    print(f"  {results['overall_conclusion']}")
    print()
    print("Caveats:")
    for limitation in results["limitations"]:
        print(f"  - {limitation}")
    print()
    print("Experiment complete.")


def main():
    results = run_experiment(verbose=True)
    print_results_report(results)
    return results


if __name__ == "__main__":
    main()
