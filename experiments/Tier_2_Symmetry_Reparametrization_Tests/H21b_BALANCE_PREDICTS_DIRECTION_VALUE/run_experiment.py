#!/usr/bin/env python3
"""
H21b: Prescribed Layer Imbalance vs Muon's Residual Advantage over NormSGD
===========================================================================

This file implements a small deep linear toy experiment testing a hypothesis,
not asserting a conclusion: as prescribed layer imbalance increases, Muon's
advantage over NormSGD may shrink.

Current implementation scope:
  - 4-layer deep linear network with 32x32 weights
  - prescribed imbalance c applied as W0 *= c and W_{L-1} /= c
  - compare Muon against NormSGD only
  - choose one global learning rate per optimizer by a small selection sweep
  - evaluate the chosen LR across multiple seeds
  - primary headline metric (kept for parity with prior versions):
        residual_advantage = mean(final_loss_normsgd) / mean(final_loss_muon)

The script now exposes structured raw sweep/evaluation data so that the paired
notebook can present the experiment more seriously without re-implementing the
core loop.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
NOTEBOOK_PATH = SCRIPT_PATH.with_suffix(".ipynb")
EXPERIMENT_ID = "H21b_BALANCE_HYPOTHESIS_TOY_MUON_VS_NORMSGD"

DIM = 32
N_LAYERS = 4
NUM_STEPS = 300
MOMENTUM = 0.9
NS_ITERS = 5
NUM_SEEDS = 5
NUM_SELECTION_SEEDS = 3
BATCH_SIZE = 64
DIVERGENCE_THRESHOLD = 1e10

C_VALUES = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0]
LR_MUON = np.logspace(-4, -1, 10).tolist()
LR_NORM = np.logspace(-3, 0, 10).tolist()

DEFAULT_CONFIG = {
    "dim": DIM,
    "n_layers": N_LAYERS,
    "num_steps": NUM_STEPS,
    "momentum": MOMENTUM,
    "ns_iters": NS_ITERS,
    "num_seeds": NUM_SEEDS,
    "num_selection_seeds": NUM_SELECTION_SEEDS,
    "batch_size": BATCH_SIZE,
    "divergence_threshold": DIVERGENCE_THRESHOLD,
    "c_values": C_VALUES,
    "lr_muon": LR_MUON,
    "lr_normsgd": LR_NORM,
    "seed_base": 42,
    "seed_stride": 137,
    "init_seed_offset": 5000,
}


def get_default_config():
    """Return a deep copy of the default experiment configuration."""
    return deepcopy(DEFAULT_CONFIG)


def normalize_config(config=None):
    """Merge optional overrides into the default config and normalize types."""
    cfg = get_default_config()
    if config:
        cfg.update(config)

    int_keys = [
        "dim",
        "n_layers",
        "num_steps",
        "ns_iters",
        "num_seeds",
        "num_selection_seeds",
        "batch_size",
        "seed_base",
        "seed_stride",
        "init_seed_offset",
    ]
    float_keys = ["momentum", "divergence_threshold"]

    for key in int_keys:
        cfg[key] = int(cfg[key])
    for key in float_keys:
        cfg[key] = float(cfg[key])

    cfg["c_values"] = [float(x) for x in cfg["c_values"]]
    cfg["lr_muon"] = [float(x) for x in cfg["lr_muon"]]
    cfg["lr_normsgd"] = [float(x) for x in cfg["lr_normsgd"]]

    if cfg["num_selection_seeds"] <= 0:
        raise ValueError("num_selection_seeds must be positive")
    if cfg["num_selection_seeds"] > cfg["num_seeds"]:
        raise ValueError("num_selection_seeds cannot exceed num_seeds")
    if cfg["n_layers"] <= 0:
        raise ValueError("n_layers must be positive")
    if cfg["dim"] <= 0:
        raise ValueError("dim must be positive")

    return cfg


def get_default_seeds(num_seeds=None, seed_base=None, seed_stride=None):
    """Deterministic seed list used across selection and evaluation."""
    num_seeds = NUM_SEEDS if num_seeds is None else int(num_seeds)
    seed_base = DEFAULT_CONFIG["seed_base"] if seed_base is None else int(seed_base)
    seed_stride = DEFAULT_CONFIG["seed_stride"] if seed_stride is None else int(seed_stride)
    return [seed_base + i * seed_stride for i in range(num_seeds)]


def estimate_train_calls(config=None):
    """Estimated number of train() calls for one full run under the given config."""
    cfg = normalize_config(config)
    calls_per_c = (
        cfg["num_selection_seeds"] * (len(cfg["lr_muon"]) + len(cfg["lr_normsgd"]))
        + 2 * cfg["num_seeds"]
    )
    return len(cfg["c_values"]) * calls_per_c


def newton_schulz(M, n_iters=NS_ITERS):
    norm = np.linalg.norm(M, "fro")
    if norm < 1e-15:
        return M
    X = M / norm
    for _ in range(n_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


def init_weights(seed, c, config=None):
    cfg = normalize_config(config)
    rng = np.random.RandomState(seed)
    dim = cfg["dim"]
    n_layers = cfg["n_layers"]
    weights = [np.eye(dim) + rng.randn(dim, dim) * 0.1 for _ in range(n_layers)]
    weights[0] = weights[0] * c
    weights[-1] = weights[-1] / c
    return weights


def forward(weights, X):
    out = X.copy()
    for W in weights:
        out = W @ out
    return out


def compute_loss(weights, X, Y):
    pred = forward(weights, X)
    return 0.5 * np.mean(np.sum((pred - Y) ** 2, axis=0))


def compute_gradients(weights, X, Y):
    L = len(weights)
    N = X.shape[1]
    acts = [X.copy()]
    for W in weights:
        acts.append(W @ acts[-1])
    delta = (acts[-1] - Y) / N
    grads = [None] * L
    for l in range(L - 1, -1, -1):
        grads[l] = delta @ acts[l].T
        if l > 0:
            delta = weights[l].T @ delta
    return grads


def make_data(seed, config=None):
    cfg = normalize_config(config)
    rng = np.random.RandomState(seed)
    dim = cfg["dim"]
    batch_size = cfg["batch_size"]
    W_target = rng.randn(dim, dim) * 0.5
    X = rng.randn(dim, batch_size) * 0.3
    Y = W_target @ X
    return X, Y


def train(w0, X, Y, lr, opt, config=None):
    cfg = normalize_config(config)
    weights = [W.copy() for W in w0]
    mom = [np.zeros_like(W) for W in weights]

    for _ in range(cfg["num_steps"]):
        if compute_loss(weights, X, Y) > cfg["divergence_threshold"]:
            return float("inf")

        grads = compute_gradients(weights, X, Y)
        for i in range(cfg["n_layers"]):
            if opt == "muon":
                mom[i] = cfg["momentum"] * mom[i] + newton_schulz(grads[i], n_iters=cfg["ns_iters"])
            elif opt == "normsgd":
                nrm = np.linalg.norm(grads[i], "fro")
                mom[i] = cfg["momentum"] * mom[i] + grads[i] / max(nrm, 1e-15)
            else:
                raise ValueError(f"Unknown optimizer: {opt}")
            weights[i] -= lr * mom[i]

    return compute_loss(weights, X, Y)


def finite_mean(values):
    finite_values = [float(v) for v in values if np.isfinite(v)]
    if not finite_values:
        return float("inf")
    return float(np.mean(finite_values))


def pearson_correlation(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def classify_directional_advantage(residual_advantage):
    if residual_advantage > 2.0:
        return "YES"
    if residual_advantage > 1.2:
        return "MARGINAL"
    return "NO"


def sweep_learning_rates(c, opt, grid, selection_seeds, config=None):
    cfg = normalize_config(config)
    records = []
    best_lr = float(grid[-1])
    best_mean_loss = float("inf")

    for lr in grid:
        losses = []
        for seed in selection_seeds:
            init_seed = seed + cfg["init_seed_offset"]
            loss = train(init_weights(init_seed, c, cfg), *make_data(seed, cfg), lr, opt, cfg)
            losses.append(float(loss))

        mean_loss = finite_mean(losses)
        record = {
            "lr": float(lr),
            "selection_losses": losses,
            "selection_mean_loss": mean_loss,
            "num_finite": int(sum(np.isfinite(loss) for loss in losses)),
        }
        records.append(record)

        if mean_loss < best_mean_loss:
            best_mean_loss = mean_loss
            best_lr = float(lr)

    return {
        "optimizer": opt,
        "records": records,
        "best_lr": best_lr,
        "best_selection_mean_loss": best_mean_loss,
    }


def evaluate_optimizer(c, opt, lr, seeds, config=None):
    cfg = normalize_config(config)
    records = []
    for seed in seeds:
        init_seed = seed + cfg["init_seed_offset"]
        loss = train(init_weights(init_seed, c, cfg), *make_data(seed, cfg), lr, opt, cfg)
        records.append({
            "seed": int(seed),
            "init_seed": int(init_seed),
            "loss": float(loss),
        })
    return records


def collect_initial_norm_diagnostics(c, seeds, config=None):
    cfg = normalize_config(config)
    per_seed = []
    ratios = []

    for seed in seeds:
        init_seed = seed + cfg["init_seed_offset"]
        weights = init_weights(init_seed, c, cfg)
        layer_norms = [float(np.linalg.norm(W, "fro")) for W in weights]
        ratio = float(max(layer_norms) / max(min(layer_norms), 1e-30))
        ratios.append(ratio)
        per_seed.append({
            "seed": int(seed),
            "init_seed": int(init_seed),
            "layer_norms": layer_norms,
            "ratio": ratio,
        })

    return {
        "per_seed": per_seed,
        "mean": finite_mean(ratios),
        "min": float(np.min(ratios)),
        "max": float(np.max(ratios)),
    }


def evaluate_c_value(c, seeds, config=None, verbose=False):
    cfg = normalize_config(config)
    selection_seeds = seeds[: cfg["num_selection_seeds"]]

    muon_sweep = sweep_learning_rates(c, "muon", cfg["lr_muon"], selection_seeds, cfg)
    norm_sweep = sweep_learning_rates(c, "normsgd", cfg["lr_normsgd"], selection_seeds, cfg)

    muon_eval = evaluate_optimizer(c, "muon", muon_sweep["best_lr"], seeds, cfg)
    norm_eval = evaluate_optimizer(c, "normsgd", norm_sweep["best_lr"], seeds, cfg)

    muon_losses = [record["loss"] for record in muon_eval]
    norm_losses = [record["loss"] for record in norm_eval]
    muon_mean = finite_mean(muon_losses)
    norm_mean = finite_mean(norm_losses)
    residual_advantage = float(norm_mean / max(muon_mean, 1e-30))
    norm_diagnostics = collect_initial_norm_diagnostics(c, seeds, cfg)

    result = {
        "c": float(c),
        "selection_seeds": [int(seed) for seed in selection_seeds],
        "best_lrs": {
            "muon": muon_sweep["best_lr"],
            "normsgd": norm_sweep["best_lr"],
        },
        "lr_sweeps": {
            "muon": muon_sweep["records"],
            "normsgd": norm_sweep["records"],
        },
        "evaluation_losses": {
            "muon": muon_eval,
            "normsgd": norm_eval,
        },
        "evaluation_finite_counts": {
            "muon": int(sum(np.isfinite(loss) for loss in muon_losses)),
            "normsgd": int(sum(np.isfinite(loss) for loss in norm_losses)),
        },
        "mean_losses": {
            "muon": muon_mean,
            "normsgd": norm_mean,
        },
        "residual_advantage": residual_advantage,
        "actual_initial_norm_ratio": norm_diagnostics,
        "directional_advantage_label": classify_directional_advantage(residual_advantage),
    }

    if verbose:
        print(f"\n  c={c:.1f}")
        print(
            f"    best LR: Muon={result['best_lrs']['muon']:.3e}, "
            f"NormSGD={result['best_lrs']['normsgd']:.3e}"
        )
        print(
            f"    mean final loss: Muon={muon_mean:.4e}, NormSGD={norm_mean:.4e}, "
            f"residual advantage={residual_advantage:.2f}x"
        )
        print(
            f"    actual initial ||W||_F ratio across seeds: "
            f"mean={norm_diagnostics['mean']:.2f}, "
            f"range=[{norm_diagnostics['min']:.2f}, {norm_diagnostics['max']:.2f}]"
        )

    return result


def summarize_results(per_c_results):
    c_values = np.array([entry["c"] for entry in per_c_results], dtype=float)
    advs = np.array([entry["residual_advantage"] for entry in per_c_results], dtype=float)

    c_thresh = None
    for entry in per_c_results:
        if entry["residual_advantage"] < 2.0:
            c_thresh = float(entry["c"])
            break

    log_cs = np.log(c_values)
    log_advs = np.log(np.clip(advs, 1e-10, None))
    pearson_r = pearson_correlation(log_cs, log_advs)
    t1 = None if not np.isfinite(pearson_r) else bool(pearson_r < -0.5)

    results_by_c = {entry["c"]: entry for entry in per_c_results}
    if 1.0 in results_by_c and 100.0 in results_by_c:
        t2 = bool(results_by_c[1.0]["residual_advantage"] > 3 * results_by_c[100.0]["residual_advantage"])
    else:
        t2 = None

    if t1 is False or t2 is False:
        supports_hypothesis = False
    elif t1 is None or t2 is None:
        supports_hypothesis = None
    else:
        supports_hypothesis = True

    return {
        "c_threshold_lt_2x": c_thresh,
        "pearson_r_log_c_vs_log_residual_advantage": pearson_r,
        "t1_negative_correlation_pass": t1,
        "t2_c1_adv_gt_3x_c100_pass": t2,
        "supports_hypothesis": supports_hypothesis,
        "c_values": c_values.tolist(),
        "residual_advantages": advs.tolist(),
    }


def run_experiment(config=None, seeds=None, verbose=True):
    """
    Execute the full toy experiment and return structured results.

    Parameters
    ----------
    config : dict or None
        Optional overrides for the default configuration.
    seeds : list[int] or None
        Optional explicit seed list. If omitted, deterministic default seeds are used.
    verbose : bool
        If True, print a human-readable run log and summary.
    """
    cfg = normalize_config(config)
    if seeds is None:
        seeds = get_default_seeds(
            num_seeds=cfg["num_seeds"],
            seed_base=cfg["seed_base"],
            seed_stride=cfg["seed_stride"],
        )
    else:
        seeds = [int(seed) for seed in seeds]
        if len(seeds) != cfg["num_seeds"]:
            raise ValueError("Explicit seeds length must match config['num_seeds']")

    if verbose:
        print("=" * 100)
        print("H21b: prescribed layer imbalance vs Muon's residual advantage over NormSGD")
        print("=" * 100)
        print(
            "Question: in this toy deep linear setup, does Muon's residual advantage over "
            "NormSGD shrink as prescribed imbalance c grows?"
        )
        print(
            "Primary metric (kept for parity with prior versions): "
            "mean(final_loss_normsgd) / mean(final_loss_muon)"
        )
        print(
            "Comparator note: this pair implements Muon vs NormSGD only; "
            "it does not include an oracle per-layer-LR SGD control."
        )
        print(f"c values: {cfg['c_values']}")
        print(f"seeds: {seeds}")
        print(f"estimated train() calls: {estimate_train_calls(cfg)}")

    per_c_results = []
    for c in cfg["c_values"]:
        per_c_results.append(evaluate_c_value(c, seeds, cfg, verbose=verbose))

    summary = summarize_results(per_c_results)
    results_by_c = {entry["c"]: entry for entry in per_c_results}

    results = {
        "experiment_id": EXPERIMENT_ID,
        "question": (
            "Does prescribed layer imbalance reduce Muon's residual advantage over "
            "NormSGD in this toy deep linear setup?"
        ),
        "headline_metric": "mean(final_loss_normsgd) / mean(final_loss_muon)",
        "script_path": str(SCRIPT_PATH),
        "counterpart_notebook": str(NOTEBOOK_PATH),
        "config": cfg,
        "seeds": seeds,
        "estimated_train_calls": estimate_train_calls(cfg),
        "per_c": per_c_results,
        "results_by_c": results_by_c,
        "summary": summary,
    }

    if verbose:
        print(f"\n\n{'=' * 100}")
        print("RESULTS: RESIDUAL ADVANTAGE VS PRESCRIBED IMBALANCE")
        print(f"{'=' * 100}")
        print(f"\n  {'c':>6}  {'mean ||W|| ratio':>16}  {'Residual adv':>14}  {'Direction matters?':>20}")
        print("  " + "-" * 64)
        for entry in per_c_results:
            print(
                f"  {entry['c']:>6.1f}  "
                f"{entry['actual_initial_norm_ratio']['mean']:>16.2f}  "
                f"{entry['residual_advantage']:>14.2f}x  "
                f"{entry['directional_advantage_label']:>20}"
            )

        threshold_label = (
            f"{summary['c_threshold_lt_2x']:.1f}"
            if summary["c_threshold_lt_2x"] is not None
            else f">{cfg['c_values'][-1]:.1f}"
        )
        print(f"\n  First grid c with residual advantage < 2x: {threshold_label}")

        pearson_r = summary["pearson_r_log_c_vs_log_residual_advantage"]
        if np.isfinite(pearson_r):
            print(f"  Pearson corr(log(c), log(residual_advantage)): r = {pearson_r:.3f}")
        else:
            print("  Pearson corr(log(c), log(residual_advantage)): undefined")

        t1 = summary["t1_negative_correlation_pass"]
        t2 = summary["t2_c1_adv_gt_3x_c100_pass"]
        print(
            f"\n  T1: negative correlation (r < -0.5)?   --> "
            f"{('PASS' if t1 else 'FAIL') if t1 is not None else 'N/A'}"
            + (f"  (r={pearson_r:.3f})" if np.isfinite(pearson_r) else "")
        )
        print(
            f"  T2: adv(c=1) > 3 x adv(c=100)?        --> "
            f"{('PASS' if t2 else 'FAIL') if t2 is not None else 'N/A'}"
        )

        support = summary["supports_hypothesis"]
        if support is True:
            conclusion_line = "This run supports the stated toy hypothesis."
        elif support is False:
            conclusion_line = "This run does not support the stated toy hypothesis."
        else:
            conclusion_line = "This run is inconclusive for the stated toy hypothesis."

        print(f"\n  Conclusion: {conclusion_line}")
        print(f"\n{'=' * 100}")
        print("EXPERIMENT COMPLETE")
        print(f"{'=' * 100}")

    return results


def main():
    """CLI entrypoint preserving the default experiment behavior."""
    return run_experiment(verbose=True)


if __name__ == "__main__":
    main()
