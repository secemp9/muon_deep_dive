#!/usr/bin/env python3
"""
H18b: Dense LR-sweep robustness check for the Muon-vs-SGD advantage
===================================================================

This experiment keeps the toy deep-linear setup from the earlier H18 series and
asks a narrower question: does the optimizer gap persist when each optimizer
gets a denser 50-point log-spaced learning-rate sweep? This is a robustness
check at one LR-grid resolution, not a direct multi-resolution test of
learning-rate independence.

Protocol:
  For depths L in {2, 4, 8, 16}:
    - SGD: 50 LRs log-spaced in [1e-6, 0.5]
    - Muon: 50 LRs log-spaced in [1e-5, 0.2]
    - Selection phase: 3 seeds per LR, choose the LR with the lowest median
      finite final loss
    - Evaluation phase: 5 seeds at the chosen LR, report the mean of finite
      final losses and expose finite counts explicitly

Heuristic summary tests:
  - T1: all depthwise advantages lie in [5x, 100x]
  - T2: CV of log(advantage) across depths < 0.3
  - T3: |Spearman(depth, advantage)| < 0.8

Important caveats:
  - Only one dense LR grid per optimizer is tested here.
  - T3 is a heuristic threshold, not a formal significance test.
  - The primary loss metric preserves historical parity by averaging only
    finite evaluation losses; the returned results make divergence visible
    through finite-count fields.
"""

import time
import numpy as np

EXPERIMENT_ID = "H18b_30X_FACTOR_LR_INDEPENDENCE"
EXPERIMENT_TITLE = "H18b: Dense LR-sweep robustness check for the Muon-vs-SGD advantage"
EXPERIMENT_SCOPE = (
    "Single-grid dense LR-sweep robustness check for a toy deep-linear Muon-vs-SGD "
    "comparison. This is not a direct multi-grid LR-independence test."
)
PRIMARY_METRIC_CAVEAT = (
    "Primary evaluation means are computed over finite final losses only; "
    "finite-count fields must be checked before over-interpreting the advantage ratio."
)

DIM = 32
DEPTHS = [2, 4, 8, 16]
NUM_STEPS = 300
MOMENTUM = 0.9
NS_ITERS = 5
NUM_SEEDS = 5
BATCH_SIZE = 64
SEED_START = 42
SEED_STRIDE = 137
SELECTION_SEED_COUNT = 3
DIVERGENCE_THRESHOLD = 1e10
ADVANTAGE_RANGE = (5.0, 100.0)
CV_THRESHOLD = 0.3
SPEARMAN_ABS_THRESHOLD = 0.8

# Dense 50-point log-spaced grids used in this first-pass robustness check.
SGD_LR_GRID = np.logspace(-6, np.log10(0.5), 50)
MUON_LR_GRID = np.logspace(-5, np.log10(0.2), 50)


def get_default_config():
    return {
        "dim": DIM,
        "depths": list(DEPTHS),
        "num_steps": NUM_STEPS,
        "momentum": MOMENTUM,
        "ns_iters": NS_ITERS,
        "num_seeds": NUM_SEEDS,
        "batch_size": BATCH_SIZE,
        "seed_start": SEED_START,
        "seed_stride": SEED_STRIDE,
        "selection_seed_count": SELECTION_SEED_COUNT,
        "divergence_threshold": DIVERGENCE_THRESHOLD,
        "sgd_lr_grid": [float(x) for x in SGD_LR_GRID],
        "muon_lr_grid": [float(x) for x in MUON_LR_GRID],
        "advantage_range": [float(ADVANTAGE_RANGE[0]), float(ADVANTAGE_RANGE[1])],
        "cv_threshold": CV_THRESHOLD,
        "spearman_abs_threshold": SPEARMAN_ABS_THRESHOLD,
    }


def resolve_config(config=None):
    resolved = get_default_config()
    if config:
        resolved.update(config)

    resolved["dim"] = int(resolved["dim"])
    resolved["depths"] = [int(d) for d in resolved["depths"]]
    resolved["num_steps"] = int(resolved["num_steps"])
    resolved["momentum"] = float(resolved["momentum"])
    resolved["ns_iters"] = int(resolved["ns_iters"])
    resolved["num_seeds"] = int(resolved["num_seeds"])
    resolved["batch_size"] = int(resolved["batch_size"])
    resolved["seed_start"] = int(resolved["seed_start"])
    resolved["seed_stride"] = int(resolved["seed_stride"])
    resolved["selection_seed_count"] = int(resolved["selection_seed_count"])
    resolved["divergence_threshold"] = float(resolved["divergence_threshold"])
    resolved["sgd_lr_grid"] = [float(x) for x in resolved["sgd_lr_grid"]]
    resolved["muon_lr_grid"] = [float(x) for x in resolved["muon_lr_grid"]]
    resolved["advantage_range"] = [float(x) for x in resolved["advantage_range"]]
    resolved["cv_threshold"] = float(resolved["cv_threshold"])
    resolved["spearman_abs_threshold"] = float(resolved["spearman_abs_threshold"])

    if resolved["selection_seed_count"] > resolved["num_seeds"]:
        raise ValueError("selection_seed_count cannot exceed num_seeds")

    return resolved


def make_seed_list(config):
    return [config["seed_start"] + i * config["seed_stride"] for i in range(config["num_seeds"])]


def newton_schulz(M, n_iters=NS_ITERS):
    norm = np.linalg.norm(M, ord='fro')
    if norm < 1e-15:
        return M
    X = M / norm
    for _ in range(n_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


def init_weights(depth, seed, dim=DIM):
    rng = np.random.RandomState(seed)
    return [np.eye(dim) + rng.randn(dim, dim) * 0.1 for _ in range(depth)]


def forward(weights, X):
    out = X.copy()
    for W in weights:
        out = W @ out
    return out


def compute_loss(weights, X, Y):
    pred = forward(weights, X)
    return 0.5 * np.mean(np.sum((pred - Y)**2, axis=0))


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


def make_data(seed, dim=DIM, batch_size=BATCH_SIZE):
    rng = np.random.RandomState(seed)
    W_target = rng.randn(dim, dim) * 0.5
    X = rng.randn(dim, batch_size) * 0.3
    Y = W_target @ X
    return X, Y


def train(
    weights_init,
    X,
    Y,
    lr,
    optimizer,
    num_steps=NUM_STEPS,
    momentum=MOMENTUM,
    ns_iters=NS_ITERS,
    divergence_threshold=DIVERGENCE_THRESHOLD,
):
    weights = [W.copy() for W in weights_init]
    mom = [np.zeros_like(W) for W in weights]
    for _ in range(num_steps):
        loss = compute_loss(weights, X, Y)
        if not np.isfinite(loss) or loss > divergence_threshold:
            return float('inf')
        grads = compute_gradients(weights, X, Y)
        for i in range(len(weights)):
            if optimizer == 'muon':
                mom[i] = momentum * mom[i] + newton_schulz(grads[i], n_iters=ns_iters)
            else:
                mom[i] = momentum * mom[i] + grads[i]
            weights[i] = weights[i] - lr * mom[i]
    return compute_loss(weights, X, Y)


def spearman_rank(x, y):
    n = len(x)
    if n < 3:
        return float('nan')
    rx = np.argsort(np.argsort(x)).astype(float) + 1
    ry = np.argsort(np.argsort(y)).astype(float) + 1
    return np.corrcoef(rx, ry)[0, 1]


def _format_float(value, precision=6, suffix=""):
    if value is None or not np.isfinite(value):
        return f"inf{suffix}" if value == float('inf') else f"nan{suffix}"
    return f"{value:.{precision}e}{suffix}" if abs(value) >= 1e3 or (0 < abs(value) < 1e-2) else f"{value:.{precision}f}{suffix}"


def _format_ratio(value):
    if not np.isfinite(value):
        return "inf"
    return f"{value:.2f}x"


def summarize_results(depth_results, config):
    depths = [int(record["depth"]) for record in depth_results]
    advantages = [float(record["advantage_sgd_over_muon"]) for record in depth_results]

    unstable_settings = []
    for depth_record in depth_results:
        depth = int(depth_record["depth"])
        for optimizer, opt_result in depth_record["optimizers"].items():
            finite_count = int(opt_result["evaluation_finite_count"])
            if finite_count < config["num_seeds"]:
                unstable_settings.append({
                    "depth": depth,
                    "optimizer": optimizer,
                    "best_lr": float(opt_result["best_lr"]),
                    "evaluation_finite_count": finite_count,
                    "evaluation_total_count": int(config["num_seeds"]),
                })

    log_advantages = [
        float(np.log(a)) if np.isfinite(a) and a > 0 else float('nan')
        for a in advantages
    ]
    all_log_finite = all(np.isfinite(v) for v in log_advantages)
    all_adv_finite = all(np.isfinite(a) for a in advantages)

    if all_adv_finite:
        advantage_array = np.array(advantages, dtype=float)
        mean_advantage = float(np.mean(advantage_array))
        std_advantage = float(np.std(advantage_array))
        min_advantage = float(np.min(advantage_array))
        max_advantage = float(np.max(advantage_array))
    else:
        mean_advantage = float('nan')
        std_advantage = float('nan')
        min_advantage = float('nan')
        max_advantage = float('nan')

    if all_log_finite:
        log_array = np.array(log_advantages, dtype=float)
        cv_log_advantage = float(np.std(log_array) / (np.abs(np.mean(log_array)) + 1e-15))
        geometric_mean_advantage = float(np.exp(np.mean(log_array)))
    else:
        cv_log_advantage = float('nan')
        geometric_mean_advantage = float('nan')

    if all_adv_finite:
        rho = float(spearman_rank(np.array(depths, dtype=float), np.array(advantages, dtype=float)))
    else:
        rho = float('nan')

    low, high = config["advantage_range"]
    pass_flags = {
        "T1_advantage_range": bool(all(np.isfinite(a) and low < a < high for a in advantages)),
        "T2_cv_log_advantage": bool(np.isfinite(cv_log_advantage) and cv_log_advantage < config["cv_threshold"]),
        "T3_no_monotone_depth_trend": bool(np.isfinite(rho) and abs(rho) < config["spearman_abs_threshold"]),
    }

    return {
        "depth_advantages": [
            {"depth": depth, "advantage_sgd_over_muon": advantage}
            for depth, advantage in zip(depths, advantages)
        ],
        "advantages_by_depth": {depth: advantage for depth, advantage in zip(depths, advantages)},
        "log_advantages": log_advantages,
        "mean_advantage": mean_advantage,
        "std_advantage": std_advantage,
        "min_advantage": min_advantage,
        "max_advantage": max_advantage,
        "geometric_mean_advantage": geometric_mean_advantage,
        "cv_log_advantage": cv_log_advantage,
        "spearman_depth_advantage": rho,
        "pass_flags": pass_flags,
        "overall_pass": bool(all(pass_flags.values())),
        "all_best_lr_evaluations_fully_finite": len(unstable_settings) == 0,
        "unstable_settings": unstable_settings,
        "heuristic_thresholds": {
            "advantage_range": [low, high],
            "cv_threshold": float(config["cv_threshold"]),
            "spearman_abs_threshold": float(config["spearman_abs_threshold"]),
        },
    }


def _print_run_header(config, seeds, selection_seeds):
    print("=" * 100)
    print(EXPERIMENT_TITLE)
    print("=" * 100)
    print(EXPERIMENT_SCOPE)
    print(f"Depths: {config['depths']}")
    print(
        f"Dense LR grids: SGD={len(config['sgd_lr_grid'])} candidates in "
        f"[{min(config['sgd_lr_grid']):.1e}, {max(config['sgd_lr_grid']):.1e}], "
        f"Muon={len(config['muon_lr_grid'])} candidates in "
        f"[{min(config['muon_lr_grid']):.1e}, {max(config['muon_lr_grid']):.1e}]"
    )
    print(f"Seeds: {seeds}")
    print(f"Selection seeds ({len(selection_seeds)}): {selection_seeds}")
    print(f"Evaluation seeds ({len(seeds)}): {seeds}")
    print(f"Primary metric caveat: {PRIMARY_METRIC_CAVEAT}")
    print()


def _print_summary_report(results):
    config = results["config"]
    summary = results["summary"]

    print(f"\n{'=' * 100}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 100}")
    print(
        f"\n{'Depth':>6}  {'SGD best LR':>12}  {'SGD eval':>14}  {'SGD finite':>10}  "
        f"{'Muon best LR':>12}  {'Muon eval':>14}  {'Muon finite':>11}  {'Advantage':>10}"
    )
    print("  " + "-" * 102)

    for depth_record in results["depth_results"]:
        depth = depth_record["depth"]
        sgd = depth_record["optimizers"]["sgd"]
        muon = depth_record["optimizers"]["muon"]
        adv = depth_record["advantage_sgd_over_muon"]
        print(
            f"  {depth:>6}  {sgd['best_lr']:>12.2e}  {sgd['evaluation_mean_loss']:>14.4e}  "
            f"{sgd['evaluation_finite_count']:>2}/{config['num_seeds']:<7}  {muon['best_lr']:>12.2e}  "
            f"{muon['evaluation_mean_loss']:>14.4e}  {muon['evaluation_finite_count']:>2}/{config['num_seeds']:<8}  {adv:>9.2f}x"
        )

    print(f"\nRuntime: {results['runtime_seconds']:.2f}s")
    print(f"Mean advantage: {_format_ratio(summary['mean_advantage'])}")
    print(f"Std advantage:  {_format_ratio(summary['std_advantage'])}")
    print(f"Geometric mean advantage: {_format_ratio(summary['geometric_mean_advantage'])}")
    print(f"CV of log(advantage): {_format_float(summary['cv_log_advantage'], precision=3)}")
    print(
        "Spearman(depth, advantage): "
        f"{_format_float(summary['spearman_depth_advantage'], precision=3)} "
        f"(heuristic |rho| < {config['spearman_abs_threshold']})"
    )

    print("\nHeuristic pass flags:")
    print(
        f"  T1: all advantages in [{config['advantage_range'][0]:.0f}x, {config['advantage_range'][1]:.0f}x]? "
        f"--> {'PASS' if summary['pass_flags']['T1_advantage_range'] else 'FAIL'}"
    )
    print(
        f"  T2: CV of log(advantage) < {config['cv_threshold']:.3f}? "
        f"--> {'PASS' if summary['pass_flags']['T2_cv_log_advantage'] else 'FAIL'}"
    )
    print(
        f"  T3: |Spearman rho| < {config['spearman_abs_threshold']:.3f}? "
        f"--> {'PASS' if summary['pass_flags']['T3_no_monotone_depth_trend'] else 'FAIL'}"
    )

    if summary["unstable_settings"]:
        print("\nInstability caveat at chosen best LR(s):")
        for entry in summary["unstable_settings"]:
            print(
                f"  depth={entry['depth']}, optimizer={entry['optimizer']}, best_lr={entry['best_lr']:.2e}, "
                f"finite_eval={entry['evaluation_finite_count']}/{entry['evaluation_total_count']}"
            )
    else:
        print("\nAll chosen best LRs evaluated cleanly on all seeds.")

    print(f"\nOverall heuristic verdict: {'PASS' if summary['overall_pass'] else 'FAIL'}")
    print(f"{'=' * 100}")


def run_experiment(config=None, verbose=True):
    config = resolve_config(config)
    seeds = make_seed_list(config)
    selection_seeds = seeds[:config["selection_seed_count"]]

    if verbose:
        _print_run_header(config, seeds, selection_seeds)

    start_time = time.time()
    depth_results = []

    for depth in config["depths"]:
        if verbose:
            print(f"\nDepth L={depth}")
            print("-" * 100)

        depth_record = {
            "depth": int(depth),
            "optimizers": {},
        }
        best_losses = {}

        for optimizer, lr_grid_key in (("sgd", "sgd_lr_grid"), ("muon", "muon_lr_grid")):
            lr_grid = [float(x) for x in config[lr_grid_key]]
            best_lr = lr_grid[0]
            best_selection_median = float('inf')
            lr_sweep = []

            for lr in lr_grid:
                selection_losses = []
                for seed in selection_seeds:
                    X, Y = make_data(seed, dim=config["dim"], batch_size=config["batch_size"])
                    weights = init_weights(depth, seed + 5000, dim=config["dim"])
                    final_loss = train(
                        weights,
                        X,
                        Y,
                        lr,
                        optimizer,
                        num_steps=config["num_steps"],
                        momentum=config["momentum"],
                        ns_iters=config["ns_iters"],
                        divergence_threshold=config["divergence_threshold"],
                    )
                    selection_losses.append(float(final_loss))

                finite_selection_losses = [loss for loss in selection_losses if np.isfinite(loss)]
                selection_median = float(np.median(finite_selection_losses)) if finite_selection_losses else float('inf')
                lr_record = {
                    "lr": float(lr),
                    "selection_seed_losses": selection_losses,
                    "selection_finite_count": int(len(finite_selection_losses)),
                    "selection_median_finite_loss": selection_median,
                }
                lr_sweep.append(lr_record)

                if selection_median < best_selection_median:
                    best_selection_median = selection_median
                    best_lr = float(lr)

            evaluation_seed_losses = []
            for seed in seeds:
                X, Y = make_data(seed, dim=config["dim"], batch_size=config["batch_size"])
                weights = init_weights(depth, seed + 5000, dim=config["dim"])
                final_loss = train(
                    weights,
                    X,
                    Y,
                    best_lr,
                    optimizer,
                    num_steps=config["num_steps"],
                    momentum=config["momentum"],
                    ns_iters=config["ns_iters"],
                    divergence_threshold=config["divergence_threshold"],
                )
                evaluation_seed_losses.append(float(final_loss))

            finite_eval_losses = [loss for loss in evaluation_seed_losses if np.isfinite(loss)]
            evaluation_mean = float(np.mean(finite_eval_losses)) if finite_eval_losses else float('inf')
            evaluation_std = float(np.std(finite_eval_losses)) if finite_eval_losses else float('inf')
            evaluation_median = float(np.median(finite_eval_losses)) if finite_eval_losses else float('inf')
            evaluation_min = float(np.min(finite_eval_losses)) if finite_eval_losses else float('inf')
            evaluation_max = float(np.max(finite_eval_losses)) if finite_eval_losses else float('inf')

            optimizer_record = {
                "optimizer": optimizer,
                "depth": int(depth),
                "lr_grid": lr_grid,
                "lr_grid_size": int(len(lr_grid)),
                "lr_sweep": lr_sweep,
                "best_lr": float(best_lr),
                "best_selection_median_loss": float(best_selection_median),
                "evaluation_seed_losses": evaluation_seed_losses,
                "evaluation_finite_count": int(len(finite_eval_losses)),
                "evaluation_mean_loss": evaluation_mean,
                "evaluation_std_loss": evaluation_std,
                "evaluation_median_loss": evaluation_median,
                "evaluation_min_loss": evaluation_min,
                "evaluation_max_loss": evaluation_max,
                "evaluation_all_finite": bool(len(finite_eval_losses) == len(seeds)),
            }
            depth_record["optimizers"][optimizer] = optimizer_record
            best_losses[optimizer] = evaluation_mean

            if verbose:
                finite_lr_count = sum(record["selection_finite_count"] > 0 for record in lr_sweep)
                print(
                    f"  {optimizer.upper():>5}: best_lr={best_lr:.6e}, "
                    f"selection_median={best_selection_median:.6e}, eval_mean={evaluation_mean:.6e}, "
                    f"eval_std={evaluation_std:.6e}, eval_finite={len(finite_eval_losses)}/{len(seeds)}, "
                    f"finite_LRs={finite_lr_count}/{len(lr_grid)}"
                )
                if len(finite_eval_losses) < len(seeds):
                    print("         caveat: evaluation mean uses only finite losses at the chosen best LR.")

        advantage = float(best_losses["sgd"] / max(best_losses["muon"], 1e-30))
        depth_record["advantage_sgd_over_muon"] = advantage
        depth_results.append(depth_record)

        if verbose:
            print(f"  Advantage (SGD mean loss / Muon mean loss): {_format_ratio(advantage)}")

    runtime_seconds = float(time.time() - start_time)
    summary = summarize_results(depth_results, config)
    results = {
        "experiment_id": EXPERIMENT_ID,
        "title": EXPERIMENT_TITLE,
        "scope": EXPERIMENT_SCOPE,
        "primary_metric_caveat": PRIMARY_METRIC_CAVEAT,
        "config": config,
        "seeds": seeds,
        "selection_seeds": selection_seeds,
        "evaluation_seeds": list(seeds),
        "depth_results": depth_results,
        "summary": summary,
        "runtime_seconds": runtime_seconds,
    }

    if verbose:
        _print_summary_report(results)

    return results


def main():
    return run_experiment(verbose=True)


if __name__ == '__main__':
    main()
