#!/usr/bin/env python3
"""
H19a: One-step local Hessian probe for partial SV equalization
==============================================================

This experiment does NOT establish that partial SV equalization creates
spurious saddles in general. It is a small local probe on a toy 3-layer
4x4 deep linear network where the full 48x48 Hessian is tractable.

Protocol
--------
1. Warm up each seed for 50 momentum-SGD steps.
2. Compute the Hessian at the warmup point.
3. For each per-layer equalization depth k in {0, 1, 2, 3, 4}, take one
   transformed step.
4. Recompute the Hessian at the new point and compare local curvature
   metrics.

Primary metrics
---------------
- baseline / after negative-eigenvalue counts
- baseline / after minimum eigenvalues
- delta_neg = after_neg - baseline_neg
- delta_min = after_min - baseline_min

Interpretation caveats
----------------------
- This is a one-step local Hessian probe, not a trajectory study.
- Under the current implementation, k=1 is NOT identical to k=0:
  it preserves singular directions and within-layer singular-value ratios,
  but rescales each layer gradient to Frobenius norm sqrt(d). It is thus a
  norm-normalized raw-gradient variant, not a new partially equalized
  geometry.
- For nonzero gradients, all k>0 layer updates are Frobenius-normalized to
  sqrt(d), so positive-k comparisons are step-norm matched with each other,
  but not with k=0.
"""

import numpy as np

DIM = 4
N_LAYERS = 3
N_PARAMS = N_LAYERS * DIM * DIM  # 48
N_SAMPLES = 64
DATA_STD = 0.3
INIT_STD = 0.1
WARMUP_STEPS = 50
NUM_SEEDS = 5
SEED_BASE = 42
SEED_STRIDE = 137
MOMENTUM = 0.9
LR = 0.01
FD_EPS = 1e-5
NEG_EIG_THRESHOLD = -1e-8
K_PER_LAYER = [0, 1, 2, 3, 4]


def pack(Ws):
    return np.concatenate([W.ravel() for W in Ws])


def unpack(theta):
    Ws = []
    idx = 0
    for _ in range(N_LAYERS):
        Ws.append(theta[idx:idx + DIM * DIM].reshape(DIM, DIM))
        idx += DIM * DIM
    return Ws


def forward(Ws, X):
    out = X.copy()
    for W in Ws:
        out = W @ out
    return out


def loss_fn(theta, X, Y):
    Ws = unpack(theta)
    pred = forward(Ws, X)
    return 0.5 * np.mean(np.sum((pred - Y) ** 2, axis=0))


def grad_fn(theta, X, Y):
    Ws = unpack(theta)
    n = X.shape[1]
    acts = [X.copy()]
    for W in Ws:
        acts.append(W @ acts[-1])
    delta = (acts[-1] - Y) / n
    grads = []
    for layer_idx in range(N_LAYERS - 1, -1, -1):
        grads.insert(0, delta @ acts[layer_idx].T)
        if layer_idx > 0:
            delta = Ws[layer_idx].T @ delta
    return pack(grads), grads


def hessian_fd(theta, X, Y, fd_eps=FD_EPS, return_stats=False):
    n = len(theta)
    H_raw = np.zeros((n, n))
    for i in range(n):
        tp = theta.copy()
        tp[i] += fd_eps
        tm = theta.copy()
        tm[i] -= fd_eps
        gp, _ = grad_fn(tp, X, Y)
        gm, _ = grad_fn(tm, X, Y)
        H_raw[:, i] = (gp - gm) / (2 * fd_eps)

    raw_norm = np.linalg.norm(H_raw)
    symmetry_residual = 0.0 if raw_norm < 1e-15 else np.linalg.norm(H_raw - H_raw.T) / raw_norm
    H = 0.5 * (H_raw + H_raw.T)

    if return_stats:
        return H, {
            "fd_eps": float(fd_eps),
            "symmetry_residual": float(symmetry_residual),
        }
    return H


def partial_sv_equalize_layer(M, k):
    """Equalize top-k singular values of one layer gradient matrix.

    k=0 returns the raw gradient.
    k=1 preserves the singular values exactly, then rescales the whole layer
    to Frobenius norm sqrt(d), so it differs from k=0 mostly by magnitude.
    k=d returns the polar factor U @ V^T.
    For 1 < k < d, the top-k singular values are equalized to their mean,
    and the whole layer is then normalized to Frobenius norm sqrt(d).
    """
    if k == 0:
        return M

    U, sigma, Vt = np.linalg.svd(M, full_matrices=False)
    d = len(sigma)
    kk = min(k, d)
    sigma_new = sigma.copy()

    if kk >= d:
        return U @ Vt

    top_mean = np.mean(sigma[:kk])
    sigma_new[:kk] = top_mean

    target_norm = np.sqrt(d)
    current_norm = np.linalg.norm(sigma_new)
    if current_norm > 1e-15:
        sigma_new *= target_norm / current_norm

    return U @ np.diag(sigma_new) @ Vt


def newton_schulz_layer(M, n_iters=5):
    """Unused in H19a; retained only as a future polar-factor approximation."""
    norm = np.linalg.norm(M, "fro")
    if norm < 1e-15:
        return M
    X = M / norm
    for _ in range(n_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


def get_default_seeds(num_seeds=NUM_SEEDS, seed_base=SEED_BASE, seed_stride=SEED_STRIDE):
    return [seed_base + i * seed_stride for i in range(num_seeds)]


def summarize_hessian(H, symmetry_residual, neg_eig_threshold=NEG_EIG_THRESHOLD):
    eigs = np.linalg.eigvalsh(H)
    return {
        "neg_count": int(np.sum(eigs < neg_eig_threshold)),
        "min_eig": float(eigs[0]),
        "max_eig": float(eigs[-1]),
        "trace": float(np.trace(H)),
        "symmetry_residual": float(symmetry_residual),
    }


def describe_k_behavior(k, dim=DIM):
    if k == 0:
        return "raw gradient (SGD-like baseline; no singular-value modification)"
    if k == 1:
        return (
            "same singular directions and within-layer singular-value ratios as the raw gradient, "
            "followed by layerwise Frobenius normalization"
        )
    if k >= dim:
        return "full equalization / polar factor (all singular values mapped to 1)"
    return (
        f"top-{k} singular values equalized to their mean, then layerwise Frobenius-normalized; "
        f"bottom-{dim - k} singular values keep their relative anisotropy"
    )


def build_seed_k_records(seed_results):
    rows = []
    for seed_result in seed_results:
        baseline = seed_result["baseline"]
        for k_result in seed_result["k_results"]:
            row = {
                "seed_index": int(seed_result["seed_index"]),
                "seed": int(seed_result["seed"]),
                "init_loss": float(seed_result["init_loss"]),
                "warmup_loss": float(seed_result["warmup_loss"]),
                "warmup_grad_norm": float(seed_result["warmup_grad_norm"]),
                "baseline_neg": int(baseline["neg_count"]),
                "baseline_min": float(baseline["min_eig"]),
                "baseline_max": float(baseline["max_eig"]),
                "baseline_trace": float(baseline["trace"]),
                "baseline_symmetry_residual": float(baseline["symmetry_residual"]),
            }
            row.update(k_result)
            rows.append(row)
    return rows


def aggregate_by_k(seed_k_records, k_values=K_PER_LAYER):
    def mean_of(rows, key):
        return float(np.mean([row[key] for row in rows]))

    def std_of(rows, key):
        return float(np.std([row[key] for row in rows]))

    aggregate_rows = []
    for k in k_values:
        rows = [row for row in seed_k_records if row["k"] == k]
        aggregate_rows.append({
            "k": int(k),
            "k_behavior": rows[0]["k_behavior"],
            "num_seeds": int(len(rows)),
            "mean_baseline_neg": mean_of(rows, "baseline_neg"),
            "std_baseline_neg": std_of(rows, "baseline_neg"),
            "mean_after_neg": mean_of(rows, "after_neg"),
            "std_after_neg": std_of(rows, "after_neg"),
            "mean_delta_neg": mean_of(rows, "delta_neg"),
            "std_delta_neg": std_of(rows, "delta_neg"),
            "mean_baseline_min": mean_of(rows, "baseline_min"),
            "std_baseline_min": std_of(rows, "baseline_min"),
            "mean_after_min": mean_of(rows, "after_min"),
            "std_after_min": std_of(rows, "after_min"),
            "mean_delta_min": mean_of(rows, "delta_min"),
            "std_delta_min": std_of(rows, "delta_min"),
            "mean_step_norm": mean_of(rows, "step_norm"),
            "std_step_norm": std_of(rows, "step_norm"),
            "mean_new_loss": mean_of(rows, "new_loss"),
            "std_new_loss": std_of(rows, "new_loss"),
            "mean_after_max": mean_of(rows, "after_max"),
            "mean_after_trace": mean_of(rows, "after_trace"),
            "mean_after_symmetry_residual": mean_of(rows, "after_symmetry_residual"),
        })
    return aggregate_rows


def evaluate_headline_tests(seed_k_records, aggregate_rows):
    mean_after_neg = {row["k"]: row["mean_after_neg"] for row in aggregate_rows}
    mean_delta_neg = {row["k"]: row["mean_delta_neg"] for row in aggregate_rows}
    intermediate_ks = [k for k in K_PER_LAYER if k not in (0, DIM)]

    positive_k_step_norms = [row["step_norm"] for row in seed_k_records if row["k"] > 0]
    positive_k_step_norms_match = (
        max(positive_k_step_norms) - min(positive_k_step_norms) < 1e-12
        if positive_k_step_norms
        else True
    )

    return {
        "intermediate_k_increases_mean_after_neg": bool(any(
            mean_after_neg[k] > max(mean_after_neg[0], mean_after_neg[DIM]) + 0.5
            for k in intermediate_ks
        )),
        "intermediate_k_increases_mean_delta_neg": bool(any(
            mean_delta_neg[k] > max(mean_delta_neg[0], mean_delta_neg[DIM]) + 0.5
            for k in intermediate_ks
        )),
        "all_delta_neg_zero": bool(all(row["delta_neg"] == 0 for row in seed_k_records)),
        "positive_k_step_norms_match": bool(positive_k_step_norms_match),
        "k1_is_identical_to_k0": False,
    }


def run_experiment(seeds=None, verbose=False):
    if seeds is None:
        seeds = get_default_seeds()
    seeds = [int(seed) for seed in seeds]

    config = {
        "dim": int(DIM),
        "n_layers": int(N_LAYERS),
        "n_params": int(N_PARAMS),
        "n_samples": int(N_SAMPLES),
        "data_std": float(DATA_STD),
        "init_std": float(INIT_STD),
        "warmup_steps": int(WARMUP_STEPS),
        "num_seeds": int(len(seeds)),
        "momentum": float(MOMENTUM),
        "lr": float(LR),
        "fd_eps": float(FD_EPS),
        "neg_eig_threshold": float(NEG_EIG_THRESHOLD),
        "k_per_layer": list(K_PER_LAYER),
        "hessians_per_seed": int(1 + len(K_PER_LAYER)),
        "grad_evals_per_hessian": int(2 * N_PARAMS),
        "hessian_grad_evals_per_seed": int((1 + len(K_PER_LAYER)) * 2 * N_PARAMS),
        "minimum_total_grad_evals_per_seed": int(WARMUP_STEPS + 1 + len(K_PER_LAYER) + (1 + len(K_PER_LAYER)) * 2 * N_PARAMS),
        "scope": "one-step local Hessian probe after warmup, not a trajectory study",
    }

    seed_results = []

    if verbose:
        print("=" * 100)
        print("H19a: ONE-STEP LOCAL HESSIAN PROBE FOR PARTIAL SV EQUALIZATION")
        print("=" * 100)
        print(f"Network: {N_LAYERS}-layer deep linear {DIM}x{DIM} ({N_PARAMS} params)")
        print(f"k per layer: {K_PER_LAYER}")
        print(f"Seeds: {seeds}")
        print(f"Warmup: {WARMUP_STEPS} momentum-SGD steps, then baseline Hessian + one transformed step")
        print(f"Negative-eigenvalue threshold: {NEG_EIG_THRESHOLD}")
        print("Scope: one-step local Hessian probe only; not a trajectory or convergence study.")
        print("k=1 note: same singular directions as k=0 per layer, but Frobenius-normalized to sqrt(d).")
        print()

    for seed_index, seed in enumerate(seeds):
        rng = np.random.RandomState(seed)
        X = rng.randn(DIM, N_SAMPLES) * DATA_STD
        Y = rng.randn(DIM, N_SAMPLES) * DATA_STD
        weights = [np.eye(DIM) + rng.randn(DIM, DIM) * INIT_STD for _ in range(N_LAYERS)]
        theta = pack(weights)

        init_loss = loss_fn(theta, X, Y)
        mom = np.zeros_like(theta)
        for _ in range(WARMUP_STEPS):
            g, _ = grad_fn(theta, X, Y)
            mom = MOMENTUM * mom + g
            theta = theta - LR * mom
        warmup_loss = loss_fn(theta, X, Y)

        warmup_grad_vec, warmup_grad_layers = grad_fn(theta, X, Y)
        H_before, before_stats = hessian_fd(theta, X, Y, fd_eps=FD_EPS, return_stats=True)
        baseline = summarize_hessian(H_before, before_stats["symmetry_residual"], NEG_EIG_THRESHOLD)

        if verbose:
            print(f"Seed {seed_index + 1}/{len(seeds)} (seed={seed})")
            print(
                f"  baseline: neg={baseline['neg_count']}, min={baseline['min_eig']:.4e}, "
                f"max={baseline['max_eig']:.4e}, trace={baseline['trace']:.4e}, "
                f"sym_resid={baseline['symmetry_residual']:.2e}"
            )

        k_results = []
        for k in K_PER_LAYER:
            step_layers = [partial_sv_equalize_layer(layer_grad, k) for layer_grad in warmup_grad_layers]
            step_vec = pack(step_layers)
            update_vec = LR * step_vec
            theta_new = theta - update_vec
            new_loss = loss_fn(theta_new, X, Y)

            H_after, after_stats = hessian_fd(theta_new, X, Y, fd_eps=FD_EPS, return_stats=True)
            after = summarize_hessian(H_after, after_stats["symmetry_residual"], NEG_EIG_THRESHOLD)

            k_result = {
                "k": int(k),
                "k_behavior": describe_k_behavior(k),
                "after_neg": int(after["neg_count"]),
                "after_min": float(after["min_eig"]),
                "after_max": float(after["max_eig"]),
                "after_trace": float(after["trace"]),
                "after_symmetry_residual": float(after["symmetry_residual"]),
                "delta_neg": int(after["neg_count"] - baseline["neg_count"]),
                "delta_min": float(after["min_eig"] - baseline["min_eig"]),
                "delta_max": float(after["max_eig"] - baseline["max_eig"]),
                "delta_trace": float(after["trace"] - baseline["trace"]),
                "step_vec_norm": float(np.linalg.norm(step_vec)),
                "step_norm": float(np.linalg.norm(update_vec)),
                "new_loss": float(new_loss),
            }
            k_results.append(k_result)

            if verbose:
                print(
                    f"    k={k}: after_neg={k_result['after_neg']}, delta_neg={k_result['delta_neg']:+d}, "
                    f"after_min={k_result['after_min']:.4e}, delta_min={k_result['delta_min']:+.4e}, "
                    f"step_norm={k_result['step_norm']:.6f}, new_loss={k_result['new_loss']:.6f}"
                )

        seed_results.append({
            "seed_index": int(seed_index),
            "seed": int(seed),
            "x_norm": float(np.linalg.norm(X)),
            "y_norm": float(np.linalg.norm(Y)),
            "init_loss": float(init_loss),
            "warmup_loss": float(warmup_loss),
            "warmup_grad_norm": float(np.linalg.norm(warmup_grad_vec)),
            "warmup_grad_layer_fro_norms": [float(np.linalg.norm(layer_grad, "fro")) for layer_grad in warmup_grad_layers],
            "baseline": baseline,
            "k_results": k_results,
        })

        if verbose:
            print()

    seed_k_records = build_seed_k_records(seed_results)
    aggregate_rows = aggregate_by_k(seed_k_records, K_PER_LAYER)
    tests = evaluate_headline_tests(seed_k_records, aggregate_rows)

    return {
        "experiment_id": "H19a_PARTIAL_SV_SPURIOUS_SADDLES",
        "title": "H19a: one-step local Hessian probe for partial SV equalization",
        "question": "Do intermediate per-layer equalization depths increase local negative Hessian curvature after one transformed step?",
        "config": config,
        "seeds": seeds,
        "seed_results": seed_results,
        "seed_k_records": seed_k_records,
        "aggregate_by_k": aggregate_rows,
        "tests": tests,
        "notes": {
            "scope": "one-step local Hessian probe on a toy deep linear network; not evidence about full trajectories or general training behavior by itself",
            "k1_behavior": (
                "k=1 is distinct from k=0: it preserves each layer's singular directions and within-layer singular-value ratios, "
                "but rescales the whole layer update to Frobenius norm sqrt(d)."
            ),
            "positive_k_step_norm_note": (
                "For nonzero gradients, every k>0 layer update is normalized to Frobenius norm sqrt(d), so k=1,2,3,4 have matched packed step norms by construction. "
                "The k=0 comparison is therefore not step-norm matched."
            ),
            "unused_helper": "newton_schulz_layer is retained for possible future comparison but is not used in this experiment.",
        },
    }


def print_results_summary(results):
    aggregate_rows = results["aggregate_by_k"]
    tests = results["tests"]

    print("=" * 100)
    print("SUMMARY: NEGATIVE HESSIAN CURVATURE AFTER ONE TRANSFORMED STEP")
    print("=" * 100)
    print(
        f"{'k':>3}  {'mean baseline neg':>18}  {'mean after neg':>14}  {'mean delta neg':>14}  "
        f"{'mean after min':>15}  {'mean delta min':>15}  {'mean step norm':>15}"
    )
    print("-" * 108)
    for row in aggregate_rows:
        print(
            f"{row['k']:>3}  {row['mean_baseline_neg']:>18.1f}  {row['mean_after_neg']:>14.1f}  {row['mean_delta_neg']:>14.1f}  "
            f"{row['mean_after_min']:>15.4e}  {row['mean_delta_min']:>15.4e}  {row['mean_step_norm']:>15.6f}"
        )

    print()
    print(f"k=1 behavior: {results['notes']['k1_behavior']}")
    print(f"Step-norm note: {results['notes']['positive_k_step_norm_note']}")
    print(
        "Intermediate k creates MORE saddle directions (mean after-neg count): "
        f"{'CONFIRMED' if tests['intermediate_k_increases_mean_after_neg'] else 'NOT CONFIRMED'}"
    )
    print(
        "Intermediate k increases mean delta_neg relative to both extremes: "
        f"{'CONFIRMED' if tests['intermediate_k_increases_mean_delta_neg'] else 'NOT CONFIRMED'}"
    )
    if tests["all_delta_neg_zero"]:
        print("Observed current default result: delta_neg = 0 for every seed and every k.")
    print("Interpretation: under current defaults, this probe does not show an intermediate-k increase in negative-eigenvalue count.")
    print("=" * 100)


def main():
    results = run_experiment(verbose=True)
    print_results_summary(results)


if __name__ == "__main__":
    main()
