#!/usr/bin/env python3
"""
2.1b-i: Equivariance Error vs Depth
===================================

Toy study: for a deep linear network trained with Muon, measure how the final
*target-layer* equivariance error changes with the position of the layer that was
bilaterally conjugated at initialization.

Implemented metric for a chosen target layer l:
    err(l) = ||W_B[l] - R W_A[l] S^T|| / ||W_A[l]||
where
    Path A: train from the original initialization,
    Path B: conjugate only layer l at initialization and retrain on the same data.

What this script DOES measure:
- final target-layer mismatch after 100 Muon steps,
- as a function of target-layer index and network depth,
- for the exact toy setup used in the paired notebook.

What this script DOES NOT by itself establish:
- a causal mechanism for any observed depth effect,
- a full cross-layer propagation profile,
- a proof of correlation length or transport of perturbations through all layers.
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

DIM = 32
DEPTHS = [4, 8, 16]
NUM_STEPS = 100
LR = 0.01
MOMENTUM = 0.9
NS_ITERS = 5
NUM_SEEDS = 10
BATCH_SIZE = 64
DATA_SEED_BASE = 42
DATA_SEED_STRIDE = 137
WEIGHT_SEED_BASE = 5000


def get_default_config() -> Dict[str, Any]:
    return {
        "dim": DIM,
        "depths": list(DEPTHS),
        "num_steps": NUM_STEPS,
        "lr": LR,
        "momentum": MOMENTUM,
        "ns_iters": NS_ITERS,
        "num_seeds": NUM_SEEDS,
        "batch_size": BATCH_SIZE,
        "data_seed_base": DATA_SEED_BASE,
        "data_seed_stride": DATA_SEED_STRIDE,
        "weight_seed_base": WEIGHT_SEED_BASE,
        "data_scale": 0.3,
        "weight_init_scale": 0.1,
    }


def ci95_half_width(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        return 0.0
    return float(1.96 * np.std(arr) / np.sqrt(arr.size))


def safe_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size != y_arr.size or x_arr.size <= 1:
        return 0.0
    if np.std(x_arr) < 1e-15 or np.std(y_arr) < 1e-15:
        return 0.0
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def get_center_layer_indices(depth: int) -> List[int]:
    if depth <= 0:
        raise ValueError("depth must be positive")
    if depth % 2 == 1:
        return [depth // 2]
    return [depth // 2 - 1, depth // 2]


def newton_schulz(M: np.ndarray, n_iters: int = NS_ITERS) -> np.ndarray:
    norm = np.linalg.norm(M, ord="fro")
    if norm < 1e-15:
        return M
    X = M / norm
    for _ in range(n_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


def random_orthogonal(n: int, rng: np.random.RandomState) -> np.ndarray:
    A = rng.randn(n, n)
    Q, R = np.linalg.qr(A)
    D = np.diag(np.sign(np.diag(R)))
    return Q @ D


def init_weights(depth: int, seed: int, dim: int = DIM) -> List[np.ndarray]:
    rng = np.random.RandomState(seed)
    return [np.eye(dim) + rng.randn(dim, dim) * 0.1 for _ in range(depth)]


def compute_loss_and_grads(
    weights: Sequence[np.ndarray],
    X: np.ndarray,
    Y: np.ndarray,
) -> tuple[float, List[np.ndarray]]:
    L = len(weights)
    N = X.shape[1]
    acts = [X.copy()]
    for W in weights:
        acts.append(W @ acts[-1])
    delta = (acts[-1] - Y) / N
    loss = 0.5 * np.mean(np.sum((acts[-1] - Y) ** 2, axis=0))
    grads: List[np.ndarray] = [None] * L  # type: ignore[assignment]
    for l in range(L - 1, -1, -1):
        grads[l] = delta @ acts[l].T
        if l > 0:
            delta = weights[l].T @ delta
    return float(loss), grads


def train_muon(
    weights_init: Sequence[np.ndarray],
    X: np.ndarray,
    Y: np.ndarray,
    n_steps: int,
    *,
    lr: float = LR,
    momentum: float = MOMENTUM,
    ns_iters: int = NS_ITERS,
) -> List[np.ndarray]:
    weights = [W.copy() for W in weights_init]
    mom = [np.zeros_like(W) for W in weights]
    for _ in range(n_steps):
        loss, grads = compute_loss_and_grads(weights, X, Y)
        if not np.isfinite(loss) or loss > 1e10:
            break
        for i in range(len(weights)):
            mom[i] = momentum * mom[i] + newton_schulz(grads[i], n_iters=ns_iters)
            weights[i] = weights[i] - lr * mom[i]
    return weights


def summarize_depth_result(
    depth: int,
    layer_errors: np.ndarray,
    pathA_final_loss_by_seed: np.ndarray,
    pathB_final_loss_by_seed_and_layer: np.ndarray,
) -> Dict[str, Any]:
    dist_from_edge = np.array([min(l, depth - 1 - l) for l in range(depth)], dtype=int)
    mean_errors = np.mean(layer_errors, axis=0)
    std_errors = np.std(layer_errors, axis=0)
    min_errors = np.min(layer_errors, axis=0)
    max_errors = np.max(layer_errors, axis=0)

    center_layers = get_center_layer_indices(depth)
    edge_mean_by_seed = np.mean(layer_errors[:, [0, depth - 1]], axis=1)
    center_mean_by_seed = np.mean(layer_errors[:, center_layers], axis=1)
    edge_minus_center_by_seed = edge_mean_by_seed - center_mean_by_seed
    edge_over_center_by_seed = edge_mean_by_seed / np.maximum(center_mean_by_seed, 1e-30)

    per_seed_correlations = np.array(
        [safe_correlation(dist_from_edge, layer_errors[seed_idx]) for seed_idx in range(layer_errors.shape[0])],
        dtype=float,
    )
    mean_profile_correlation = safe_correlation(dist_from_edge, mean_errors)

    layer_summary = []
    for l in range(depth):
        layer_summary.append(
            {
                "layer": l,
                "dist_from_edge": int(dist_from_edge[l]),
                "mean_error": float(mean_errors[l]),
                "std_error": float(std_errors[l]),
                "min_error": float(min_errors[l]),
                "max_error": float(max_errors[l]),
            }
        )

    min_error_layer = int(np.argmin(mean_errors))
    max_error_layer = int(np.argmax(mean_errors))
    edge_mean_error = float(np.mean(edge_mean_by_seed))
    center_mean_error = float(np.mean(center_mean_by_seed))
    pathB_over_pathA_loss_ratio_by_seed = np.mean(pathB_final_loss_by_seed_and_layer, axis=1) / np.maximum(
        pathA_final_loss_by_seed,
        1e-30,
    )

    return {
        "depth": depth,
        "dist_from_edge": dist_from_edge,
        "center_layers": center_layers,
        "layer_errors": layer_errors,
        "mean_errors": mean_errors,
        "std_errors": std_errors,
        "min_errors": min_errors,
        "max_errors": max_errors,
        "layer_summary": layer_summary,
        "min_error_layer": min_error_layer,
        "max_error_layer": max_error_layer,
        "mean_profile_correlation": float(mean_profile_correlation),
        "per_seed_correlations": per_seed_correlations,
        "per_seed_correlation_mean": float(np.mean(per_seed_correlations)),
        "per_seed_correlation_ci95": ci95_half_width(per_seed_correlations),
        "edge_mean_by_seed": edge_mean_by_seed,
        "center_mean_by_seed": center_mean_by_seed,
        "edge_minus_center_by_seed": edge_minus_center_by_seed,
        "edge_over_center_by_seed": edge_over_center_by_seed,
        "edge_mean_error": edge_mean_error,
        "center_mean_error": center_mean_error,
        "edge_minus_center_mean": float(np.mean(edge_minus_center_by_seed)),
        "edge_minus_center_ci95": ci95_half_width(edge_minus_center_by_seed),
        "pathA_final_loss_by_seed": pathA_final_loss_by_seed,
        "pathB_final_loss_by_seed_and_layer": pathB_final_loss_by_seed_and_layer,
        "pathB_over_pathA_loss_ratio_by_seed": pathB_over_pathA_loss_ratio_by_seed,
        "pathA_mean_final_loss": float(np.mean(pathA_final_loss_by_seed)),
        "pathB_mean_final_loss": float(np.mean(pathB_final_loss_by_seed_and_layer)),
        "pathB_over_pathA_loss_ratio_mean": float(np.mean(pathB_over_pathA_loss_ratio_by_seed)),
        "pathB_over_pathA_loss_ratio_ci95": ci95_half_width(pathB_over_pathA_loss_ratio_by_seed),
        "tests": {
            "legacy_T1_edge_layers_smallest": bool(min_error_layer in {0, depth - 1}),
            "legacy_T2_middle_peak": bool((0 < max_error_layer < depth - 1) and (mean_profile_correlation > 0.0)),
            "observed_edge_higher_than_center": bool(edge_mean_error > center_mean_error),
            "observed_negative_distance_correlation": bool(mean_profile_correlation < 0.0),
        },
    }


def summarize_overall(depth_results: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    overall_rows = []
    for depth in sorted(depth_results):
        r = depth_results[depth]
        overall_rows.append(
            {
                "depth": depth,
                "mean_error": float(np.mean(r["mean_errors"])),
                "min_error_layer": int(r["min_error_layer"]),
                "max_error_layer": int(r["max_error_layer"]),
                "edge_mean_error": float(r["edge_mean_error"]),
                "center_mean_error": float(r["center_mean_error"]),
                "edge_minus_center_mean": float(r["edge_minus_center_mean"]),
                "mean_profile_correlation": float(r["mean_profile_correlation"]),
                "per_seed_correlation_mean": float(r["per_seed_correlation_mean"]),
                "pathA_mean_final_loss": float(r["pathA_mean_final_loss"]),
                "pathB_mean_final_loss": float(r["pathB_mean_final_loss"]),
                "legacy_T1_edge_layers_smallest": bool(r["tests"]["legacy_T1_edge_layers_smallest"]),
                "legacy_T2_middle_peak": bool(r["tests"]["legacy_T2_middle_peak"]),
                "observed_edge_higher_than_center": bool(r["tests"]["observed_edge_higher_than_center"]),
            }
        )
    return overall_rows


def print_depth_report(depth_result: Dict[str, Any]) -> None:
    depth = depth_result["depth"]
    center_layers = ", ".join(str(l) for l in depth_result["center_layers"])

    print(f"\n{'=' * 80}")
    print(f"  DEPTH = {depth}")
    print(f"{'=' * 80}")
    print(
        f"\n  {'Layer':>6}  {'Mean Error':>12}  {'Std':>10}  {'Min':>10}  {'Max':>10}  {'Dist':>6}"
    )
    print("  " + "-" * 66)
    for row in depth_result["layer_summary"]:
        print(
            f"  {row['layer']:>6}  {row['mean_error']:>12.4e}  {row['std_error']:>10.4e}  "
            f"{row['min_error']:>10.4e}  {row['max_error']:>10.4e}  {row['dist_from_edge']:>6}"
        )

    print(f"\n  Mean-profile corr(dist_from_edge, error) = {depth_result['mean_profile_correlation']:.3f}")
    print(
        f"  Per-seed corr mean ± 95% CI            = "
        f"{depth_result['per_seed_correlation_mean']:.3f} ± {depth_result['per_seed_correlation_ci95']:.3f}"
    )
    print(f"  Edge mean error                        = {depth_result['edge_mean_error']:.4e}")
    print(
        f"  Center mean error (layers {center_layers})       = "
        f"{depth_result['center_mean_error']:.4e}"
    )
    print(
        f"  Edge - center (mean ± 95% CI)          = "
        f"{depth_result['edge_minus_center_mean']:.4e} ± {depth_result['edge_minus_center_ci95']:.4e}"
    )
    print(f"  Mean Path A final loss                 = {depth_result['pathA_mean_final_loss']:.4e}")
    print(f"  Mean Path B final loss                 = {depth_result['pathB_mean_final_loss']:.4e}")
    print(
        f"  Legacy T1 (edge layers smallest?)      = "
        f"{'SUPPORTED' if depth_result['tests']['legacy_T1_edge_layers_smallest'] else 'NOT SUPPORTED'}"
    )
    print(
        f"  Legacy T2 (middle-peaked profile?)     = "
        f"{'SUPPORTED' if depth_result['tests']['legacy_T2_middle_peak'] else 'NOT SUPPORTED'}"
    )
    print(
        "  Metric caveat                          = final target-layer self-error only; "
        "not a full cross-layer propagation measurement"
    )


def print_overall_report(results: Dict[str, Any]) -> None:
    print(f"\n{'=' * 100}")
    print("CROSS-DEPTH SUMMARY")
    print(f"{'=' * 100}")
    print(
        f"\n{'Depth':>6}  {'Mean Err':>10}  {'Min L':>6}  {'Max L':>6}  {'Edge Err':>10}  "
        f"{'Center Err':>11}  {'Edge-Center':>12}  {'Corr(mean)':>11}"
    )
    print("-" * 96)
    for row in results["overall_summary"]:
        print(
            f"{row['depth']:>6}  {row['mean_error']:>10.4e}  {row['min_error_layer']:>6}  "
            f"{row['max_error_layer']:>6}  {row['edge_mean_error']:>10.4e}  "
            f"{row['center_mean_error']:>11.4e}  {row['edge_minus_center_mean']:>12.4e}  "
            f"{row['mean_profile_correlation']:>11.3f}"
        )

    print("\nInterpretation guardrail:")
    print("  These summaries describe the implemented target-layer mismatch metric only.")
    print("  They do not by themselves prove a specific mechanism for depth effects.")
    print(f"\nRuntime: {results['runtime_seconds']:.2f} seconds")
    print(f"\n{'=' * 100}")
    print("EXPERIMENT COMPLETE")
    print(f"{'=' * 100}")


def run_experiment(
    *,
    dim: int = DIM,
    depths: Iterable[int] = DEPTHS,
    num_steps: int = NUM_STEPS,
    lr: float = LR,
    momentum: float = MOMENTUM,
    ns_iters: int = NS_ITERS,
    num_seeds: int = NUM_SEEDS,
    batch_size: int = BATCH_SIZE,
    data_seed_base: int = DATA_SEED_BASE,
    data_seed_stride: int = DATA_SEED_STRIDE,
    weight_seed_base: int = WEIGHT_SEED_BASE,
    verbose: bool = True,
) -> Dict[str, Any]:
    depths = list(depths)
    config = {
        "dim": int(dim),
        "depths": [int(depth) for depth in depths],
        "num_steps": int(num_steps),
        "lr": float(lr),
        "momentum": float(momentum),
        "ns_iters": int(ns_iters),
        "num_seeds": int(num_seeds),
        "batch_size": int(batch_size),
        "data_seed_base": int(data_seed_base),
        "data_seed_stride": int(data_seed_stride),
        "weight_seed_base": int(weight_seed_base),
        "data_scale": 0.3,
        "weight_init_scale": 0.1,
    }
    seed_schedule = [
        {
            "seed_index": int(seed),
            "data_seed": int(data_seed_base + seed * data_seed_stride),
            "weight_seed": int(weight_seed_base + seed),
            "conjugators": "sampled from the same data RNG after drawing X and Y",
        }
        for seed in range(num_seeds)
    ]

    if verbose:
        print("=" * 100)
        print("2.1b-i: EQUIVARIANCE ERROR vs DEPTH")
        print("=" * 100)
        print(
            f"Depths: {config['depths']}, DIM={dim}, Steps={num_steps}, Seeds={num_seeds}, "
            f"Batch={batch_size}"
        )
        print("Metric: err(l) = ||W_B[l] - R W_A[l] S^T|| / ||W_A[l]||")
        print("Caveat: this is target-layer self-error only, not a full propagation profile.")

    t0 = time.perf_counter()
    depth_results: Dict[int, Dict[str, Any]] = {}

    for depth in depths:
        layer_errors = np.zeros((num_seeds, depth), dtype=float)
        pathA_final_loss_by_seed = np.zeros(num_seeds, dtype=float)
        pathB_final_loss_by_seed_and_layer = np.zeros((num_seeds, depth), dtype=float)

        for seed in range(num_seeds):
            rng = np.random.RandomState(data_seed_base + seed * data_seed_stride)
            X = rng.randn(dim, batch_size) * config["data_scale"]
            Y = rng.randn(dim, batch_size) * config["data_scale"]
            R = random_orthogonal(dim, rng)
            S = random_orthogonal(dim, rng)

            weights_init = init_weights(depth, weight_seed_base + seed, dim=dim)

            weights_A = train_muon(
                weights_init,
                X,
                Y,
                num_steps,
                lr=lr,
                momentum=momentum,
                ns_iters=ns_iters,
            )
            loss_A, _ = compute_loss_and_grads(weights_A, X, Y)
            pathA_final_loss_by_seed[seed] = loss_A

            for target_layer in range(depth):
                weights_conj = [W.copy() for W in weights_init]
                weights_conj[target_layer] = R @ weights_conj[target_layer] @ S.T

                weights_B = train_muon(
                    weights_conj,
                    X,
                    Y,
                    num_steps,
                    lr=lr,
                    momentum=momentum,
                    ns_iters=ns_iters,
                )
                loss_B, _ = compute_loss_and_grads(weights_B, X, Y)
                pathB_final_loss_by_seed_and_layer[seed, target_layer] = loss_B

                expected = R @ weights_A[target_layer] @ S.T
                actual = weights_B[target_layer]
                layer_errors[seed, target_layer] = (
                    np.linalg.norm(actual - expected)
                    / max(np.linalg.norm(weights_A[target_layer]), 1e-30)
                )

        depth_result = summarize_depth_result(
            depth,
            layer_errors,
            pathA_final_loss_by_seed,
            pathB_final_loss_by_seed_and_layer,
        )
        depth_results[int(depth)] = depth_result

        if verbose:
            print_depth_report(depth_result)

    runtime_seconds = time.perf_counter() - t0
    results = {
        "experiment": "2.1b-i_Equivariance_Error_vs_Depth",
        "metric_name": "final_target_layer_equivariance_error",
        "metric_formula": "||W_B[l] - R W_A[l] S^T|| / ||W_A[l]||",
        "scope": (
            "Deep linear toy study with single-layer bilateral conjugation at initialization; "
            "reports only the final mismatch at the same target layer after Muon training."
        ),
        "limitations": [
            "Does not measure non-target-layer errors.",
            "Does not measure per-step trajectories.",
            "Does not by itself identify a causal mechanism for the depth profile.",
        ],
        "config": config,
        "seed_schedule": seed_schedule,
        "depth_results": depth_results,
        "overall_summary": summarize_overall(depth_results),
        "runtime_seconds": float(runtime_seconds),
    }

    if verbose:
        print_overall_report(results)

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=DIM)
    parser.add_argument("--depths", nargs="+", type=int, default=list(DEPTHS))
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--momentum", type=float, default=MOMENTUM)
    parser.add_argument("--ns-iters", type=int, default=NS_ITERS)
    parser.add_argument("--num-seeds", type=int, default=NUM_SEEDS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--data-seed-base", type=int, default=DATA_SEED_BASE)
    parser.add_argument("--data-seed-stride", type=int, default=DATA_SEED_STRIDE)
    parser.add_argument("--weight-seed-base", type=int, default=WEIGHT_SEED_BASE)
    parser.add_argument("--quiet", action="store_true", help="suppress the human-readable report")
    return parser


def main(argv: Sequence[str] | None = None) -> Dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    return run_experiment(
        dim=args.dim,
        depths=args.depths,
        num_steps=args.num_steps,
        lr=args.lr,
        momentum=args.momentum,
        ns_iters=args.ns_iters,
        num_seeds=args.num_seeds,
        batch_size=args.batch_size,
        data_seed_base=args.data_seed_base,
        data_seed_stride=args.data_seed_stride,
        weight_seed_base=args.weight_seed_base,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
