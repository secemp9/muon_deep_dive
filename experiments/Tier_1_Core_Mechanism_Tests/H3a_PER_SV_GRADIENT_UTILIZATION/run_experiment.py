#!/usr/bin/env python3
"""
H3a: Per-SV Gradient Utilization -- Muon vs Normalized SGD
==========================================================

This file preserves the original toy experiment setup:
- a 4-layer 32x32 deep linear network
- 500 training steps
- 5 random seeds
- methods: momentum SGD, momentum of orthogonalized gradients ("muon"),
  and momentum SGD with post-momentum Frobenius normalization ("norm_sgd")

What is actually measured
-------------------------
At each layer and step, the current gradient G is decomposed as
    G = U diag(sigma) V^T.
For the optimizer's actual update matrix Delta_W used at that step, we compute
absolute singular-direction utilization coefficients
    utilization_i = |u_i^T Delta_W v_i|.
After normalizing these coefficients to a probability distribution p_i, we report:
- Shannon entropy H(p)
- the top-1 absolute-utilization fraction p_0

Important scope notes
---------------------
- The main experiment studies momentum-based updates projected onto the CURRENT
  gradient singular-vector basis.
- This is not the same as a single-gradient / no-momentum calculation.
- The reported top-1 fraction is not squared update energy.
- A separate theory sanity check is included to demonstrate the simpler
  no-momentum expectation on one fixed gradient.
"""

from __future__ import annotations

from time import perf_counter
import numpy as np

DIM = 32
NUM_LAYERS = 4
NUM_STEPS = 500
MOMENTUM = 0.9
NS_ITERS = 5
NUM_SEEDS = 5
BATCH_SIZE = 64
LR = 0.01
METHODS = ("sgd", "muon", "norm_sgd")
METHOD_LABELS = {
    "sgd": "SGD + momentum",
    "muon": "Orthogonalized gradient + momentum",
    "norm_sgd": "Momentum then Frobenius-normalize",
}


def get_default_config():
    """Return the default experiment configuration as a plain dict."""
    return {
        "dim": DIM,
        "num_layers": NUM_LAYERS,
        "num_steps": NUM_STEPS,
        "momentum": MOMENTUM,
        "ns_iters": NS_ITERS,
        "num_seeds": NUM_SEEDS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "methods": list(METHODS),
        "method_labels": dict(METHOD_LABELS),
        "h_max_bits": float(np.log2(DIM)),
    }


def get_default_seeds(num_seeds=NUM_SEEDS):
    """Return the deterministic seed list used by the original script."""
    return [42 + i * 137 for i in range(num_seeds)]


def newton_schulz(M, n_iters=NS_ITERS):
    """Approximate the polar factor of M via Newton-Schulz iteration."""
    norm = np.linalg.norm(M, ord="fro")
    if norm < 1e-15:
        return M
    X = M / norm
    for _ in range(n_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


def init_weights(seed, dim=DIM, num_layers=NUM_LAYERS):
    """Initialize near-identity square weight matrices."""
    rng = np.random.RandomState(seed)
    return [np.eye(dim) + rng.randn(dim, dim) * 0.1 for _ in range(num_layers)]


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


def compute_utilization(update, U, V):
    """
    Compute absolute utilization coefficients |u_i^T update v_i|.

    These are absolute projection magnitudes in the gradient's singular-vector
    basis, not squared energy contributions.
    """
    k = min(U.shape[1], V.shape[1])
    util = np.zeros(k)
    for i in range(k):
        util[i] = abs(np.dot(U[:, i], update @ V[:, i]))
    return util


def entropy(probs):
    """Shannon entropy in bits for a probability distribution."""
    p = probs[probs > 1e-30]
    return -np.sum(p * np.log2(p))


def _distribution_stats(utilization):
    util_sum = float(np.sum(utilization))
    if util_sum <= 1e-30:
        return None
    probs = utilization / util_sum
    return {
        "distribution": probs.tolist(),
        "entropy": float(entropy(probs)),
        "top1_abs_utilization_fraction": float(probs[0]),
        "top3_abs_utilization_fraction": float(np.sum(np.sort(probs)[-3:])),
    }


def _single_gradient_projection_stats(gradient, update):
    U_g, sigma_g, Vt_g = np.linalg.svd(gradient, full_matrices=False)
    util = compute_utilization(update, U_g, Vt_g.T)
    stats = _distribution_stats(util)
    if stats is None:
        raise ValueError("Degenerate utilization distribution encountered.")
    stats.update(
        {
            "gradient_rank": int(np.sum(sigma_g > 1e-12)),
            "gradient_condition_number": float(sigma_g[0] / max(sigma_g[-1], 1e-30)),
            "utilization": util.tolist(),
        }
    )
    return stats


def run_theory_sanity_check(seed=1234, dim=DIM, ns_iters=NS_ITERS):
    """
    Demonstrate the simpler no-momentum expectation on one fixed gradient.

    This helper intentionally separates the clean single-gradient story from the
    actual momentum-based training experiment.
    """
    rng = np.random.RandomState(seed)
    gradient = rng.randn(dim, dim)
    gradient_norm = np.linalg.norm(gradient, ord="fro")
    normalized_gradient = gradient / max(gradient_norm, 1e-12)
    orthogonalized_gradient = newton_schulz(gradient, n_iters=ns_iters)

    raw_stats = _single_gradient_projection_stats(gradient, gradient)
    normalized_stats = _single_gradient_projection_stats(gradient, normalized_gradient)
    orthogonalized_stats = _single_gradient_projection_stats(gradient, orthogonalized_gradient)

    raw_distribution = np.array(raw_stats["distribution"])
    normalized_distribution = np.array(normalized_stats["distribution"])
    orthogonalized_distribution = np.array(orthogonalized_stats["distribution"])
    uniform_distribution = np.ones(dim) / dim

    return {
        "seed": int(seed),
        "dim": int(dim),
        "ns_iters": int(ns_iters),
        "gradient_fro_norm": float(gradient_norm),
        "raw_sgd": raw_stats,
        "normalized_raw_gradient": normalized_stats,
        "orthogonalized_gradient": orthogonalized_stats,
        "checks": {
            "raw_vs_normalized_l1_difference": float(
                np.sum(np.abs(raw_distribution - normalized_distribution))
            ),
            "orthogonalized_vs_uniform_max_abs_deviation": float(
                np.max(np.abs(orthogonalized_distribution - uniform_distribution))
            ),
            "expected_uniform_entropy_bits": float(np.log2(dim)),
        },
    }


def _compute_update(method, grad, momentum_buffer, momentum=MOMENTUM, ns_iters=NS_ITERS):
    if method == "sgd":
        momentum_buffer = momentum * momentum_buffer + grad
        update = momentum_buffer
    elif method == "muon":
        ortho_grad = newton_schulz(grad, n_iters=ns_iters)
        momentum_buffer = momentum * momentum_buffer + ortho_grad
        update = momentum_buffer
    elif method == "norm_sgd":
        momentum_buffer = momentum * momentum_buffer + grad
        buf_norm = np.linalg.norm(momentum_buffer, ord="fro")
        update = momentum_buffer / max(buf_norm, 1e-12)
    else:
        raise ValueError(f"Unknown method: {method}")
    return momentum_buffer, update


def _mean_std(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None, None
    return float(np.mean(arr)), float(np.std(arr))


def _group_per_seed_summaries(per_seed_summaries, methods):
    grouped = {method: [] for method in methods}
    for row in per_seed_summaries:
        grouped[row["method"]].append(row)
    for method in methods:
        grouped[method].sort(key=lambda row: row["seed"])
    return grouped


def _build_per_step_summary(per_step_records, methods):
    summary = {method: [] for method in methods}
    grouped = {method: {} for method in methods}
    for row in per_step_records:
        grouped[row["method"]].setdefault(row["step"], []).append(row)

    for method in methods:
        for step in sorted(grouped[method]):
            rows = grouped[method][step]
            entropy_vals = [row["mean_entropy"] for row in rows]
            top1_vals = [row["mean_top1_abs_utilization_fraction"] for row in rows]
            loss_vals = [row["loss"] for row in rows]
            entropy_mean, entropy_std = _mean_std(entropy_vals)
            top1_mean, top1_std = _mean_std(top1_vals)
            loss_mean, loss_std = _mean_std(loss_vals)
            summary[method].append(
                {
                    "step": int(step),
                    "mean_entropy": entropy_mean,
                    "std_entropy": entropy_std,
                    "mean_top1_abs_utilization_fraction": top1_mean,
                    "std_top1_abs_utilization_fraction": top1_std,
                    "mean_loss": loss_mean,
                    "std_loss": loss_std,
                    "num_seeds": int(len(rows)),
                }
            )
    return summary


def _build_pooled_summary(pooled_records, methods, h_max):
    summary = {}
    for method in methods:
        rows = [row for row in pooled_records if row["method"] == method]
        entropies = np.array([row["entropy"] for row in rows], dtype=float)
        top1 = np.array([row["top1_abs_utilization_fraction"] for row in rows], dtype=float)
        summary[method] = {
            "mean_entropy": float(np.mean(entropies)),
            "std_entropy": float(np.std(entropies)),
            "mean_entropy_over_hmax": float(np.mean(entropies) / h_max),
            "mean_top1_abs_utilization_fraction": float(np.mean(top1)),
            "num_measurements": int(len(rows)),
        }
    return summary


def _build_tests(pooled_summary, h_max):
    muon_h = pooled_summary["muon"]["mean_entropy"]
    norm_h = pooled_summary["norm_sgd"]["mean_entropy"]
    sgd_top1 = pooled_summary["sgd"]["mean_top1_abs_utilization_fraction"]

    return {
        "T1": {
            "description": "Legacy threshold: pooled mean Muon entropy > pooled mean normalized-SGD entropy + 0.5 bits",
            "passed": bool(muon_h > norm_h + 0.5),
            "actual_muon_mean_entropy": float(muon_h),
            "actual_norm_sgd_mean_entropy": float(norm_h),
            "actual_gap_bits": float(muon_h - norm_h),
            "required_gap_bits": 0.5,
        },
        "T2": {
            "description": "Legacy threshold: pooled mean Muon entropy > 0.9 * log2(dim)",
            "passed": bool(muon_h > 0.9 * h_max),
            "actual_muon_mean_entropy": float(muon_h),
            "threshold_entropy_bits": float(0.9 * h_max),
            "actual_fraction_of_hmax": float(muon_h / h_max),
        },
        "T3": {
            "description": "Legacy threshold: pooled mean SGD top-1 absolute-utilization fraction > 0.5",
            "passed": bool(sgd_top1 > 0.5),
            "actual_sgd_mean_top1_abs_utilization_fraction": float(sgd_top1),
            "threshold_top1_abs_utilization_fraction": 0.5,
        },
    }


def run_experiment(config=None, seeds=None, snapshot_steps=(0, 49, 99, 249, 499), include_theory_sanity=True):
    """
    Run the main momentum-based training experiment and return structured results.

    Returned fields include config, seeds, pooled per-layer records, per-step
    summaries, per-seed summaries, legacy threshold outcomes, snapshot records,
    and runtime.
    """
    cfg = get_default_config()
    if config is not None:
        cfg.update(config)

    dim = int(cfg["dim"])
    num_layers = int(cfg["num_layers"])
    num_steps = int(cfg["num_steps"])
    momentum = float(cfg["momentum"])
    ns_iters = int(cfg["ns_iters"])
    num_seeds = int(cfg["num_seeds"])
    batch_size = int(cfg["batch_size"])
    lr = float(cfg["lr"])
    methods = list(cfg["methods"])
    method_labels = dict(cfg.get("method_labels", METHOD_LABELS))
    h_max = float(np.log2(dim))
    cfg["h_max_bits"] = h_max
    cfg["method_labels"] = method_labels

    if seeds is None:
        seeds = get_default_seeds(num_seeds=num_seeds)
    else:
        seeds = list(seeds)

    pooled_records = []
    per_step_records = []
    per_seed_summaries = []
    snapshot_records = []
    failures = []
    snapshot_step_set = set(snapshot_steps)

    started = perf_counter()

    for seed in seeds:
        rng = np.random.RandomState(seed)
        X = rng.randn(dim, batch_size) * 0.3
        Y = rng.randn(dim, batch_size) * 0.3

        for method in methods:
            weights = init_weights(seed + 5000, dim=dim, num_layers=num_layers)
            momentum_buffers = [np.zeros((dim, dim)) for _ in range(num_layers)]
            seed_method_entropies = []
            seed_method_top1 = []
            completed_steps = 0

            for step in range(num_steps):
                loss = compute_loss(weights, X, Y)
                if not np.isfinite(loss) or loss > 1e10:
                    failures.append(
                        {
                            "seed": int(seed),
                            "method": method,
                            "step": int(step),
                            "loss": float(loss),
                            "reason": "non-finite-or-too-large-loss",
                        }
                    )
                    break

                grads = compute_gradients(weights, X, Y)
                step_entropies = []
                step_top1 = []
                completed_steps = step + 1

                for layer_idx in range(num_layers):
                    grad = grads[layer_idx]
                    U_g, sigma_g, Vt_g = np.linalg.svd(grad, full_matrices=False)
                    momentum_buffers[layer_idx], update = _compute_update(
                        method,
                        grad,
                        momentum_buffers[layer_idx],
                        momentum=momentum,
                        ns_iters=ns_iters,
                    )

                    util = compute_utilization(update, U_g, Vt_g.T)
                    stats = _distribution_stats(util)
                    if stats is not None:
                        record = {
                            "method": method,
                            "seed": int(seed),
                            "step": int(step),
                            "layer": int(layer_idx),
                            "loss": float(loss),
                            "entropy": stats["entropy"],
                            "top1_abs_utilization_fraction": stats["top1_abs_utilization_fraction"],
                            "top3_abs_utilization_fraction": stats["top3_abs_utilization_fraction"],
                            "grad_fro_norm": float(np.linalg.norm(grad, ord="fro")),
                            "update_fro_norm": float(np.linalg.norm(update, ord="fro")),
                            "grad_condition_number": float(sigma_g[0] / max(sigma_g[-1], 1e-30)),
                        }
                        pooled_records.append(record)
                        step_entropies.append(record["entropy"])
                        step_top1.append(record["top1_abs_utilization_fraction"])
                        seed_method_entropies.append(record["entropy"])
                        seed_method_top1.append(record["top1_abs_utilization_fraction"])

                        if layer_idx == 0 and step in snapshot_step_set:
                            snapshot_records.append(
                                {
                                    **record,
                                    "snapshot": True,
                                }
                            )

                    weights[layer_idx] = weights[layer_idx] - lr * update

                if step_entropies:
                    per_step_records.append(
                        {
                            "method": method,
                            "seed": int(seed),
                            "step": int(step),
                            "loss": float(loss),
                            "mean_entropy": float(np.mean(step_entropies)),
                            "mean_top1_abs_utilization_fraction": float(np.mean(step_top1)),
                            "num_layers_counted": int(len(step_entropies)),
                        }
                    )

            if seed_method_entropies:
                per_seed_summaries.append(
                    {
                        "method": method,
                        "seed": int(seed),
                        "mean_entropy": float(np.mean(seed_method_entropies)),
                        "std_entropy": float(np.std(seed_method_entropies)),
                        "mean_entropy_over_hmax": float(np.mean(seed_method_entropies) / h_max),
                        "mean_top1_abs_utilization_fraction": float(np.mean(seed_method_top1)),
                        "num_measurements": int(len(seed_method_entropies)),
                        "completed_steps": int(completed_steps),
                    }
                )

    runtime_seconds = perf_counter() - started
    per_step_summary_by_method = _build_per_step_summary(per_step_records, methods)
    per_seed_summary_by_method = _group_per_seed_summaries(per_seed_summaries, methods)
    pooled_summary = _build_pooled_summary(pooled_records, methods, h_max)
    tests = _build_tests(pooled_summary, h_max)

    results = {
        "config": cfg,
        "seeds": [int(seed) for seed in seeds],
        "methods": methods,
        "method_labels": method_labels,
        "notes": {
            "metric": "Entropy and top-1 fraction are computed from normalized absolute utilization coefficients |u_i^T Delta_W v_i|.",
            "scope": "Main experiment uses momentum-based updates projected onto the current gradient singular-vector basis.",
            "not_energy": "The reported top-1 fraction is not squared update energy.",
            "thresholds": "T1/T2/T3 are retained as legacy threshold checks from the original narrative; they are reported descriptively, not treated as pre-validated theory.",
        },
        "pooled_records": pooled_records,
        "per_step_records": per_step_records,
        "per_step_summary_by_method": per_step_summary_by_method,
        "per_seed_summaries": per_seed_summaries,
        "per_seed_summary_by_method": per_seed_summary_by_method,
        "snapshot_records": snapshot_records,
        "pooled_summary": pooled_summary,
        "tests": tests,
        "failures": failures,
        "runtime_seconds": float(runtime_seconds),
    }

    if include_theory_sanity:
        results["theory_sanity_check"] = run_theory_sanity_check(
            seed=1234,
            dim=dim,
            ns_iters=ns_iters,
        )

    return results


def print_report(results):
    """Pretty-print a concise report from run_experiment() results."""
    cfg = results["config"]
    methods = results["methods"]
    h_max = cfg["h_max_bits"]

    print("=" * 100)
    print("H3a: PER-SV ABSOLUTE-UTILIZATION UNDER MOMENTUM UPDATES")
    print("=" * 100)
    print(
        f"Network: {cfg['num_layers']}-layer, {cfg['dim']}x{cfg['dim']}, "
        f"{cfg['num_steps']} steps, {cfg['num_seeds']} seeds, batch={cfg['batch_size']}, lr={cfg['lr']}"
    )
    print(f"Seeds: {results['seeds']}")
    print(f"Runtime: {results['runtime_seconds']:.2f} s")
    print()
    print("Metric note:")
    print("  utilization_i = |u_i^T Delta_W v_i| in the CURRENT gradient singular basis")
    print("  H(p) and top-1 fraction are computed from normalized absolute utilization coefficients")
    print("  These top-1 fractions are not squared update energy shares.")

    theory = results.get("theory_sanity_check")
    if theory is not None:
        print(f"\n{'=' * 100}")
        print("SINGLE-GRADIENT / NO-MOMENTUM THEORY SANITY CHECK")
        print(f"{'=' * 100}")
        print(f"  seed={theory['seed']}, dim={theory['dim']}, ns_iters={theory['ns_iters']}")
        print(
            "  Raw SGD vs normalized raw gradient L1 difference in normalized utilization: "
            f"{theory['checks']['raw_vs_normalized_l1_difference']:.3e}"
        )
        print(
            "  Orthogonalized gradient max abs deviation from uniform: "
            f"{theory['checks']['orthogonalized_vs_uniform_max_abs_deviation']:.3e}"
        )
        print(
            f"  Entropies (bits): raw={theory['raw_sgd']['entropy']:.3f}, "
            f"normalized={theory['normalized_raw_gradient']['entropy']:.3f}, "
            f"orthogonalized={theory['orthogonalized_gradient']['entropy']:.3f}, "
            f"uniform_max={theory['checks']['expected_uniform_entropy_bits']:.3f}"
        )

    print(f"\n{'=' * 100}")
    print("POOLED MAIN-EXPERIMENT SUMMARY")
    print(f"{'=' * 100}")
    print(f"  {'Method':<18}{'Mean H':>10}{'H/H_max':>10}{'Mean top-1':>14}{'Std H':>10}{'N':>8}")
    print("  " + "-" * 70)
    for method in methods:
        row = results["pooled_summary"][method]
        print(
            f"  {method:<18}{row['mean_entropy']:>10.3f}{row['mean_entropy_over_hmax']:>10.3f}"
            f"{row['mean_top1_abs_utilization_fraction']:>14.4f}{row['std_entropy']:>10.3f}{row['num_measurements']:>8d}"
        )
    ordering = sorted(
        methods,
        key=lambda name: results["pooled_summary"][name]["mean_entropy"],
        reverse=True,
    )
    print(f"\n  Entropy ordering observed: {' > '.join(ordering)}")
    print(f"  H_max = log2({cfg['dim']}) = {h_max:.3f} bits")

    print(f"\n{'=' * 100}")
    print("PER-SEED MEAN ENTROPY")
    print(f"{'=' * 100}")
    for method in methods:
        values = [row["mean_entropy"] for row in results["per_seed_summary_by_method"][method]]
        formatted = ", ".join(f"{value:.3f}" for value in values)
        print(f"  {method:<18}[{formatted}]")

    print(f"\n{'=' * 100}")
    print("LEGACY THRESHOLD CHECKS (DESCRIPTIVE ONLY)")
    print(f"{'=' * 100}")
    for key in ("T1", "T2", "T3"):
        row = results["tests"][key]
        print(f"  {key}: {row['description']}")
        print(f"      --> {'PASS' if row['passed'] else 'FAIL'}")
        if key == "T1":
            print(
                f"      actual_gap_bits = {row['actual_gap_bits']:.3f} "
                f"(required > {row['required_gap_bits']:.3f})"
            )
        elif key == "T2":
            print(
                f"      muon_mean_entropy = {row['actual_muon_mean_entropy']:.3f}, "
                f"threshold = {row['threshold_entropy_bits']:.3f}, "
                f"fraction_of_hmax = {row['actual_fraction_of_hmax']:.3f}"
            )
        elif key == "T3":
            print(
                "      sgd_mean_top1_abs_utilization_fraction = "
                f"{row['actual_sgd_mean_top1_abs_utilization_fraction']:.4f} "
                f"(threshold > {row['threshold_top1_abs_utilization_fraction']:.4f})"
            )

    if results["failures"]:
        print(f"\nFailures encountered: {len(results['failures'])}")
        for row in results["failures"][:5]:
            print(f"  {row}")
    else:
        print("\nNo divergence failures encountered.")


def main():
    results = run_experiment()
    print_report(results)


if __name__ == "__main__":
    main()
