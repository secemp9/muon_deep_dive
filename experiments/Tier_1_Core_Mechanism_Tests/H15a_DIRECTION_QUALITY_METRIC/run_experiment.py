#!/usr/bin/env python3
"""
H15a: Direction quality in a toy deep-linear setting
====================================================

This experiment measures how closely several optimizer step directions align
with a local pseudoinverse-Newton direction in a tiny 2-layer deep linear
network (32 scalar parameters total). The setup is intentionally small so that a
full 32x32 Hessian can be estimated by central finite differences.

What is measured
----------------
For each optimizer, on each seed:
- choose a best-in-grid learning rate by lowest 200-step training loss on a
  fixed logarithmic LR sweep;
- train on the same random data from the same initialization;
- at steps 10, 20, ..., 200 compute a finite-difference Hessian, the local
  pseudoinverse-Newton direction d_N = -pinv(H) g, and the optimizer step
  direction d_opt;
- record cos(d_opt, d_N) and cos(d_opt, -g).

What this script does *not* establish
-------------------------------------
This is a toy directional-alignment probe, not a general proof of Muon's
mechanism. The Newton direction is approximate (finite-difference Hessian plus
pseudoinverse regularization), learning rates are only best within the scanned
grid, and the model is a tiny deep-linear network rather than a realistic
nonlinear training setting.
"""

from pathlib import Path
import time

import numpy as np

# ── Network config ──────────────────────────────────────────────────────────
DIM = 4
NUM_LAYERS = 2
TOTAL_PARAMS = NUM_LAYERS * DIM * DIM  # 32
NS_ITERS = 5
FD_EPS = 1e-5
NEWTON_PINV_RCOND = 1e-6
NUM_SEEDS = 5
BATCH_SIZE = 64
MEASURE_STEPS = list(range(10, 201, 10))  # 10,20,...,200

# LR sweep grids
LR_GRID_SGD = np.logspace(np.log10(0.001), np.log10(0.1), 12)
LR_GRID_MUON = np.logspace(np.log10(0.0001), np.log10(0.01), 12)
LR_GRID_NORMED = np.logspace(np.log10(0.001), np.log10(0.1), 12)
LR_GRID_ADAM = np.logspace(np.log10(0.001), np.log10(0.1), 12)

ADAM_EPS = 1e-8


def newton_schulz(M, n_iters=NS_ITERS):
    """Approximate the polar factor of M with Newton-Schulz iterations."""
    norm = np.linalg.norm(M, ord='fro')
    if norm < 1e-15:
        return M
    X = M / norm
    for _ in range(n_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X



def init_weights(rng):
    """Two 4x4 layers initialized as identity plus small Gaussian noise."""
    return [np.eye(DIM) + rng.randn(DIM, DIM) * 0.1 for _ in range(NUM_LAYERS)]



def pack(weights):
    """Flatten a list of weight matrices into one vector."""
    return np.concatenate([W.ravel() for W in weights])



def unpack(vec):
    """Reshape a flat vector back into the list-of-matrices parameterization."""
    ws = []
    offset = 0
    for _ in range(NUM_LAYERS):
        ws.append(vec[offset:offset + DIM * DIM].reshape(DIM, DIM))
        offset += DIM * DIM
    return ws



def forward(weights, X):
    out = X.copy()
    for W in weights:
        out = W @ out
    return out



def loss_fn(weights, X, Y):
    pred = forward(weights, X)
    return 0.5 * np.mean(np.sum((pred - Y) ** 2, axis=0))



def compute_gradients(weights, X, Y):
    """Backprop gradients for the two-layer deep linear network."""
    N = X.shape[1]
    acts = [X.copy()]
    for W in weights:
        acts.append(W @ acts[-1])
    delta = (acts[-1] - Y) / N
    grads = [None] * NUM_LAYERS
    for l in range(NUM_LAYERS - 1, -1, -1):
        grads[l] = delta @ acts[l].T
        if l > 0:
            delta = weights[l].T @ delta
    return grads



def grad_vec(weights, X, Y):
    """Return the flattened gradient vector."""
    return pack(compute_gradients(weights, X, Y))



def full_hessian_fd(weights, X, Y):
    """Compute the full Hessian via central finite differences on the gradient."""
    w0 = pack(weights)
    n = len(w0)
    H = np.zeros((n, n))
    for i in range(n):
        w_plus = w0.copy()
        w_plus[i] += FD_EPS
        w_minus = w0.copy()
        w_minus[i] -= FD_EPS
        g_plus = grad_vec(unpack(w_plus), X, Y)
        g_minus = grad_vec(unpack(w_minus), X, Y)
        H[:, i] = (g_plus - g_minus) / (2.0 * FD_EPS)
    return 0.5 * (H + H.T)



def step_sgd(grads):
    """SGD direction = -gradient."""
    return pack([-G for G in grads])



def step_muon(grads):
    """Muon direction = -newton_schulz(G) for each layer."""
    return pack([-newton_schulz(G) for G in grads])



def step_normed_sgd(grads):
    """Normalized SGD direction = -G / ||G||_F per layer."""
    dirs = []
    for G in grads:
        nrm = np.linalg.norm(G, "fro")
        dirs.append(-G / max(nrm, 1e-15))
    return pack(dirs)



def step_adam_like(grads):
    """Adam-like direction = -G / sqrt(G^2 + eps) element-wise."""
    dirs = []
    for G in grads:
        dirs.append(-G / np.sqrt(G ** 2 + ADAM_EPS))
    return pack(dirs)


OPTIMIZERS = {
    "SGD": (step_sgd, LR_GRID_SGD),
    "Muon_k5": (step_muon, LR_GRID_MUON),
    "NormSGD": (step_normed_sgd, LR_GRID_NORMED),
    "AdamLike": (step_adam_like, LR_GRID_ADAM),
}



def make_seeds(num_seeds=NUM_SEEDS):
    return [42 + i * 137 for i in range(num_seeds)]



def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-15 or nb < 1e-15:
        return np.nan
    return np.dot(a, b) / (na * nb)



def summarize_values(values):
    arr = np.asarray(values, dtype=float)
    valid = arr[np.isfinite(arr)]
    n = int(valid.size)
    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "sem": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    std = float(np.std(valid))
    return {
        "n": n,
        "mean": float(np.mean(valid)),
        "std": std,
        "sem": float(std / np.sqrt(n)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
    }



def classify_evidence(pooled_delta, paired_summary):
    if not np.isfinite(pooled_delta):
        return "unavailable"
    if pooled_delta <= 0:
        return "not supported"

    n = int(paired_summary.get("n", 0))
    pos = int(paired_summary.get("positive_count", 0))
    neg = int(paired_summary.get("negative_count", 0))

    if n == 0:
        return "positive pooled mean only"
    if neg == 0:
        return "supported in this toy setting"
    if pos > neg:
        return "weak / mixed"
    return "mixed / inconclusive"



def find_best_grid_lr(step_fn, lr_grid, weights_init, X, Y, warmup_steps):
    """Select the best learning rate within the provided grid."""
    best_lr = float(lr_grid[0])
    best_loss = np.inf
    any_finite = False

    for lr in lr_grid:
        lr = float(lr)
        ws = [W.copy() for W in weights_init]
        diverged = False
        for _ in range(warmup_steps):
            grads = compute_gradients(ws, X, Y)
            d = step_fn(grads)
            d_layers = unpack(d)
            for i in range(NUM_LAYERS):
                ws[i] = ws[i] + lr * d_layers[i]
            lo = loss_fn(ws, X, Y)
            if not np.isfinite(lo) or lo > 1e6:
                diverged = True
                break
        if not diverged:
            lo = loss_fn(ws, X, Y)
            if np.isfinite(lo):
                any_finite = True
                if lo < best_loss:
                    best_loss = float(lo)
                    best_lr = lr

    grid_min = float(np.min(lr_grid))
    grid_max = float(np.max(lr_grid))
    return {
        "best_grid_lr": best_lr,
        "best_grid_loss": float(best_loss),
        "grid_min": grid_min,
        "grid_max": grid_max,
        "grid_size": int(len(lr_grid)),
        "hit_grid_min": bool(np.isclose(best_lr, grid_min)),
        "hit_grid_max": bool(np.isclose(best_lr, grid_max)),
        "any_finite_run": bool(any_finite),
    }



def train_and_collect(step_fn, lr, weights_init, X, Y, max_step):
    """Train with a fixed optimizer and capture states at MEASURE_STEPS."""
    ws = [W.copy() for W in weights_init]
    snapshots = {}
    diverged = False
    final_loss = float(loss_fn(ws, X, Y))

    for t in range(max_step + 1):
        grads = compute_gradients(ws, X, Y)
        if t in MEASURE_STEPS:
            snapshots[t] = ([W.copy() for W in ws], [G.copy() for G in grads])
        d = step_fn(grads)
        d_layers = unpack(d)
        for i in range(NUM_LAYERS):
            ws[i] = ws[i] + lr * d_layers[i]
        final_loss = float(loss_fn(ws, X, Y))
        if not np.isfinite(final_loss) or final_loss > 1e6:
            diverged = True
            break

    meta = {
        "diverged": bool(diverged),
        "num_snapshots": int(len(snapshots)),
        "final_loss": final_loss,
    }
    return snapshots, meta



def measure_direction_record(seed, seed_index, step, optimizer_name, step_fn, best_grid_lr, weights, grads, X, Y):
    """Measure local direction-quality quantities at one optimizer state."""
    H = full_hessian_fd(weights, X, Y)
    g = pack(grads)
    d_newton = -np.linalg.pinv(H, rcond=NEWTON_PINV_RCOND) @ g
    d_opt = step_fn(grads)
    return {
        "seed": int(seed),
        "seed_index": int(seed_index),
        "optimizer": optimizer_name,
        "step": int(step),
        "best_grid_lr": float(best_grid_lr),
        "cos_newton": float(cosine(d_opt, d_newton)),
        "cos_grad": float(cosine(d_opt, -g)),
        "loss": float(loss_fn(weights, X, Y)),
        "grad_norm": float(np.linalg.norm(g)),
        "optimizer_direction_norm": float(np.linalg.norm(d_opt)),
        "newton_direction_norm": float(np.linalg.norm(d_newton)),
    }



def build_summary(results):
    records = results["records"]
    lr_records = results["best_grid_lr_records"]
    seeds = results["seeds"]
    optimizer_names = results["config"]["optimizer_names"]

    summary_by_optimizer = {}
    for name in optimizer_names:
        opt_records = [r for r in records if r["optimizer"] == name]
        opt_lr_records = [r for r in lr_records if r["optimizer"] == name]
        cos_newton_stats = summarize_values([r["cos_newton"] for r in opt_records])
        cos_grad_stats = summarize_values([r["cos_grad"] for r in opt_records])
        lr_stats = summarize_values([r["best_grid_lr"] for r in opt_lr_records])
        loss_stats = summarize_values([r["best_grid_loss"] for r in opt_lr_records])
        summary_by_optimizer[name] = {
            "optimizer": name,
            "n_measurements": int(len(opt_records)),
            "n_seeds_with_measurements": int(len({r["seed"] for r in opt_records})),
            "mean_cos_newton": cos_newton_stats["mean"],
            "std_cos_newton": cos_newton_stats["std"],
            "sem_cos_newton": cos_newton_stats["sem"],
            "mean_cos_grad": cos_grad_stats["mean"],
            "std_cos_grad": cos_grad_stats["std"],
            "sem_cos_grad": cos_grad_stats["sem"],
            "mean_best_grid_lr": lr_stats["mean"],
            "std_best_grid_lr": lr_stats["std"],
            "mean_best_grid_loss": loss_stats["mean"],
            "hit_grid_min_count": int(sum(r["hit_grid_min"] for r in opt_lr_records)),
            "hit_grid_max_count": int(sum(r["hit_grid_max"] for r in opt_lr_records)),
            "lr_grid_min": opt_lr_records[0]["grid_min"] if opt_lr_records else float("nan"),
            "lr_grid_max": opt_lr_records[0]["grid_max"] if opt_lr_records else float("nan"),
        }

    per_step_summary = []
    for name in optimizer_names:
        for step in MEASURE_STEPS:
            step_records = [r for r in records if r["optimizer"] == name and r["step"] == step]
            cos_newton_stats = summarize_values([r["cos_newton"] for r in step_records])
            cos_grad_stats = summarize_values([r["cos_grad"] for r in step_records])
            loss_stats = summarize_values([r["loss"] for r in step_records])
            grad_norm_stats = summarize_values([r["grad_norm"] for r in step_records])
            per_step_summary.append({
                "optimizer": name,
                "step": int(step),
                "n": cos_newton_stats["n"],
                "mean_cos_newton": cos_newton_stats["mean"],
                "std_cos_newton": cos_newton_stats["std"],
                "sem_cos_newton": cos_newton_stats["sem"],
                "mean_cos_grad": cos_grad_stats["mean"],
                "std_cos_grad": cos_grad_stats["std"],
                "sem_cos_grad": cos_grad_stats["sem"],
                "mean_loss": loss_stats["mean"],
                "mean_grad_norm": grad_norm_stats["mean"],
            })

    seed_level_summary = []
    for name in optimizer_names:
        for seed in seeds:
            seed_records = [r for r in records if r["optimizer"] == name and r["seed"] == seed]
            if not seed_records:
                continue
            cos_newton_stats = summarize_values([r["cos_newton"] for r in seed_records])
            cos_grad_stats = summarize_values([r["cos_grad"] for r in seed_records])
            seed_level_summary.append({
                "optimizer": name,
                "seed": int(seed),
                "n_steps": int(len(seed_records)),
                "mean_cos_newton": cos_newton_stats["mean"],
                "std_cos_newton": cos_newton_stats["std"],
                "mean_cos_grad": cos_grad_stats["mean"],
                "std_cos_grad": cos_grad_stats["std"],
            })

    seed_lookup = {(row["optimizer"], row["seed"]): row["mean_cos_newton"] for row in seed_level_summary}
    paired_differences = {}
    for baseline in ["SGD", "NormSGD", "AdamLike"]:
        diff_records = []
        for seed in seeds:
            muon_val = seed_lookup.get(("Muon_k5", seed))
            base_val = seed_lookup.get((baseline, seed))
            if muon_val is None or base_val is None:
                continue
            diff_records.append({
                "seed": int(seed),
                "baseline": baseline,
                "muon_mean_cos_newton": float(muon_val),
                "baseline_mean_cos_newton": float(base_val),
                "muon_minus_baseline": float(muon_val - base_val),
            })
        diff_values = [r["muon_minus_baseline"] for r in diff_records]
        stats = summarize_values(diff_values)
        stats.update({
            "positive_count": int(sum(v > 0 for v in diff_values if np.isfinite(v))),
            "negative_count": int(sum(v < 0 for v in diff_values if np.isfinite(v))),
            "zero_count": int(sum(v == 0 for v in diff_values if np.isfinite(v))),
        })
        paired_differences[baseline] = {
            "records": diff_records,
            "summary": stats,
        }

    muon_summary = summary_by_optimizer["Muon_k5"]
    hypothesis_tests = {}
    for test_name, baseline in [("T1", "SGD"), ("T2", "NormSGD"), ("T3", "AdamLike")]:
        base_summary = summary_by_optimizer[baseline]
        pooled_delta = muon_summary["mean_cos_newton"] - base_summary["mean_cos_newton"]
        paired_summary = paired_differences[baseline]["summary"]
        hypothesis_tests[test_name] = {
            "test": test_name,
            "description": f"Muon_k5 mean cos(step, Newton) > {baseline} mean cos(step, Newton)",
            "muon_optimizer": "Muon_k5",
            "baseline": baseline,
            "pooled_mean_muon": muon_summary["mean_cos_newton"],
            "pooled_mean_baseline": base_summary["mean_cos_newton"],
            "pooled_delta": float(pooled_delta),
            "pooled_pass": bool(pooled_delta > 0),
            "seed_level_mean_delta": paired_summary["mean"],
            "seed_level_std_delta": paired_summary["std"],
            "seed_level_sem_delta": paired_summary["sem"],
            "seed_level_positive_count": int(paired_summary["positive_count"]),
            "seed_level_negative_count": int(paired_summary["negative_count"]),
            "seed_level_zero_count": int(paired_summary["zero_count"]),
            "seed_level_n": int(paired_summary["n"]),
            "evidence_label": classify_evidence(pooled_delta, paired_summary),
        }

    gradient_deviation_analysis = {
        "per_optimizer_mean_cos_grad": {name: summary_by_optimizer[name]["mean_cos_grad"] for name in optimizer_names},
        "muon_vs_sgd": {
            "muon_mean_cos_grad": muon_summary["mean_cos_grad"],
            "sgd_mean_cos_grad": summary_by_optimizer["SGD"]["mean_cos_grad"],
            "muon_deviates_more_than_sgd": bool(muon_summary["mean_cos_grad"] < summary_by_optimizer["SGD"]["mean_cos_grad"]),
            "muon_more_newton_aligned_than_sgd": bool(muon_summary["mean_cos_newton"] > summary_by_optimizer["SGD"]["mean_cos_newton"]),
        },
    }
    gradient_deviation_analysis["muon_vs_sgd"]["joint_pass"] = bool(
        gradient_deviation_analysis["muon_vs_sgd"]["muon_deviates_more_than_sgd"]
        and gradient_deviation_analysis["muon_vs_sgd"]["muon_more_newton_aligned_than_sgd"]
    )

    boundary_flags = {
        name: {
            "hit_grid_min_count": summary_by_optimizer[name]["hit_grid_min_count"],
            "hit_grid_max_count": summary_by_optimizer[name]["hit_grid_max_count"],
            "n_selections": int(sum(r["optimizer"] == name for r in lr_records)),
        }
        for name in optimizer_names
    }

    return {
        "summary_by_optimizer": summary_by_optimizer,
        "per_step_summary": per_step_summary,
        "seed_level_summary": seed_level_summary,
        "paired_differences": paired_differences,
        "hypothesis_tests": hypothesis_tests,
        "gradient_deviation_analysis": gradient_deviation_analysis,
        "boundary_flags": boundary_flags,
    }



def print_report(results):
    config = results["config"]
    optimizer_names = config["optimizer_names"]

    print("\n" + "=" * 100)
    print("RESULTS: DIRECTION QUALITY IN THE TOY DEEP-LINEAR SETTING")
    print("=" * 100)
    print("Learning-rate selection is best-in-grid, not a continuous optimum.")
    print("Newton directions use a finite-difference Hessian and pseudoinverse; interpret them as local approximations.")

    print(
        f"\n  {'Optimizer':>10} | {'Mean cos(step,Newton)':>22} | {'Std':>8} | {'Mean cos(step,-grad)':>20} | {'Std':>8} | {'Mean best-grid LR':>18} | {'Grid hits min/max':>17}"
    )
    print("  " + "-" * 132)
    for name in optimizer_names:
        row = results["summary_by_optimizer"][name]
        hit_str = f"{row['hit_grid_min_count']}/{row['hit_grid_max_count']}"
        print(
            f"  {name:>10} | {row['mean_cos_newton']:>22.6f} | {row['std_cos_newton']:>8.6f} | "
            f"{row['mean_cos_grad']:>20.6f} | {row['std_cos_grad']:>8.6f} | "
            f"{row['mean_best_grid_lr']:>18.6f} | {hit_str:>17}"
        )

    print("\n" + "=" * 100)
    print("BEST-IN-GRID LR SELECTIONS BY SEED")
    print("=" * 100)
    for record in results["best_grid_lr_records"]:
        boundary_labels = []
        if record["hit_grid_min"]:
            boundary_labels.append("MIN")
        if record["hit_grid_max"]:
            boundary_labels.append("MAX")
        boundary_text = f" [{' + '.join(boundary_labels)}]" if boundary_labels else ""
        print(
            f"  seed={record['seed']:>3}  {record['optimizer']:>10}: "
            f"best-grid LR = {record['best_grid_lr']:.6f}  "
            f"(loss after {config['lr_selection_horizon_steps']} steps: {record['best_grid_loss']:.6f}){boundary_text}"
        )

    print("\n" + "=" * 100)
    print("PER-STEP MEAN cos(step, Newton) [mean over available seeds]")
    print("=" * 100)
    header = f"{'Step':>5} | " + " | ".join([f"{name:>10}" for name in optimizer_names])
    print(header)
    print("-" * len(header))
    per_step_lookup = {(row["optimizer"], row["step"]): row for row in results["per_step_summary"]}
    for step in config["measure_steps"]:
        vals = []
        for name in optimizer_names:
            row = per_step_lookup[(name, step)]
            vals.append(f"{row['mean_cos_newton']:>10.4f}")
        print(f"{step:>5} | " + " | ".join(vals))

    print("\n" + "=" * 100)
    print("PER-STEP MEAN cos(step, -grad) [mean over available seeds]")
    print("=" * 100)
    print(header)
    print("-" * len(header))
    for step in config["measure_steps"]:
        vals = []
        for name in optimizer_names:
            row = per_step_lookup[(name, step)]
            vals.append(f"{row['mean_cos_grad']:>10.4f}")
        print(f"{step:>5} | " + " | ".join(vals))

    print("\n" + "=" * 100)
    print("HYPOTHESIS TESTS (TOY-SETTING, BEST-IN-GRID-LR COMPARISONS)")
    print("=" * 100)
    for test_name in ["T1", "T2", "T3"]:
        row = results["hypothesis_tests"][test_name]
        print(f"\n  {test_name}: {row['description']}")
        print(
            f"      pooled means: Muon_k5 = {row['pooled_mean_muon']:.6f}   "
            f"{row['baseline']} = {row['pooled_mean_baseline']:.6f}   delta = {row['pooled_delta']:+.6f}"
        )
        print(
            f"      seed-level mean deltas: {row['seed_level_mean_delta']:+.6f} +/- {row['seed_level_std_delta']:.6f} "
            f"(SEM {row['seed_level_sem_delta']:.6f}); positives = {row['seed_level_positive_count']}/{row['seed_level_n']}, "
            f"negatives = {row['seed_level_negative_count']}"
        )
        print(f"      pooled pass = {row['pooled_pass']}   |   evidence label = {row['evidence_label']}")

    print("\n" + "=" * 100)
    print("GRADIENT-DEVIATION CHECK")
    print("=" * 100)
    gd = results["gradient_deviation_analysis"]["muon_vs_sgd"]
    print(
        f"  Muon_k5 mean cos(step, -grad) = {gd['muon_mean_cos_grad']:.6f}   "
        f"SGD mean cos(step, -grad) = {gd['sgd_mean_cos_grad']:.6f}"
    )
    print(f"  Muon deviates more from steepest descent than SGD? {gd['muon_deviates_more_than_sgd']}")
    print(f"  Muon is more Newton-aligned than SGD? {gd['muon_more_newton_aligned_than_sgd']}")
    print(f"  Joint check (deviates from -grad yet moves toward Newton relative to SGD): {gd['joint_pass']}")

    print("\n" + "=" * 100)
    print("TOY-SETTING CONCLUSION")
    print("=" * 100)
    t1 = results["hypothesis_tests"]["T1"]["evidence_label"]
    t2 = results["hypothesis_tests"]["T2"]["evidence_label"]
    t3 = results["hypothesis_tests"]["T3"]["evidence_label"]
    print(f"  - Muon_k5 vs SGD: {t1}.")
    print(f"  - Muon_k5 vs NormSGD: {t2}.")
    print(f"  - Muon_k5 vs AdamLike: {t3}.")
    print("  - Interpretation: this toy experiment is consistent with a direction-quality story, especially relative")
    print("    to plain SGD, but it is not a general proof of mechanism and the NormSGD comparison should be treated")
    print("    cautiously when seed-level signs are mixed or the LR sweep hits grid boundaries.")
    print(f"\n  Runtime: {results['runtime_sec']:.2f}s")
    print("=" * 100)



def run_experiment(verbose=True):
    """Run the default H15a toy experiment and return structured results."""
    t0 = time.time()
    seeds = make_seeds(NUM_SEEDS)
    config = {
        "dim": DIM,
        "num_layers": NUM_LAYERS,
        "total_params": TOTAL_PARAMS,
        "ns_iters": NS_ITERS,
        "fd_eps": FD_EPS,
        "newton_pinv_rcond": NEWTON_PINV_RCOND,
        "num_seeds": NUM_SEEDS,
        "batch_size": BATCH_SIZE,
        "measure_steps": list(MEASURE_STEPS),
        "optimizer_names": list(OPTIMIZERS.keys()),
        "lr_selection_label": "best_grid_lr",
        "lr_selection_horizon_steps": int(max(MEASURE_STEPS)),
        "lr_selection_metric": "lowest final training loss after the full training horizon on the same data/init",
        "lr_grids": {name: [float(x) for x in grid] for name, (_, grid) in OPTIMIZERS.items()},
    }

    if verbose:
        print("=" * 100)
        print("H15a: Direction quality in a toy deep-linear setting")
        print("=" * 100)
        print(f"Network : {NUM_LAYERS}-layer deep linear, {DIM}x{DIM} => {TOTAL_PARAMS} params")
        print(f"Hessian : full {TOTAL_PARAMS}x{TOTAL_PARAMS} by central finite differences (FD_EPS={FD_EPS})")
        print(f"Seeds   : {NUM_SEEDS}  |  Measure steps: {MEASURE_STEPS[0]}..{MEASURE_STEPS[-1]} ({len(MEASURE_STEPS)} points)")
        print(f"Optimizers: {config['optimizer_names']}")
        print("LRs are reported as best-in-grid values, not continuous optima.")

    measurement_records = []
    best_grid_lr_records = []
    training_run_records = []

    for seed_index, seed in enumerate(seeds):
        rng = np.random.RandomState(seed)
        X = rng.randn(DIM, BATCH_SIZE)
        Y = rng.randn(DIM, BATCH_SIZE)
        weights_init = init_weights(rng)

        if verbose:
            print(f"\n  Seed {seed_index + 1}/{NUM_SEEDS} (seed={seed})")

        best_grid_lr_by_optimizer = {}
        for name, (step_fn, lr_grid) in OPTIMIZERS.items():
            lr_info = find_best_grid_lr(
                step_fn=step_fn,
                lr_grid=lr_grid,
                weights_init=weights_init,
                X=X,
                Y=Y,
                warmup_steps=max(MEASURE_STEPS),
            )
            best_grid_lr_by_optimizer[name] = lr_info["best_grid_lr"]
            best_grid_lr_records.append({
                "seed": int(seed),
                "seed_index": int(seed_index),
                "optimizer": name,
                **lr_info,
            })
            if verbose:
                boundary_labels = []
                if lr_info["hit_grid_min"]:
                    boundary_labels.append("MIN")
                if lr_info["hit_grid_max"]:
                    boundary_labels.append("MAX")
                boundary_text = f" [{' + '.join(boundary_labels)}]" if boundary_labels else ""
                print(
                    f"    {name:>10}: best-grid LR = {lr_info['best_grid_lr']:.6f}  "
                    f"(loss after {max(MEASURE_STEPS)} steps: {lr_info['best_grid_loss']:.6f}){boundary_text}"
                )

        snapshots_by_opt = {}
        for name, (step_fn, _) in OPTIMIZERS.items():
            snapshots, train_meta = train_and_collect(
                step_fn=step_fn,
                lr=best_grid_lr_by_optimizer[name],
                weights_init=weights_init,
                X=X,
                Y=Y,
                max_step=max(MEASURE_STEPS),
            )
            snapshots_by_opt[name] = snapshots
            training_run_records.append({
                "seed": int(seed),
                "seed_index": int(seed_index),
                "optimizer": name,
                "best_grid_lr": float(best_grid_lr_by_optimizer[name]),
                **train_meta,
            })

        for step in MEASURE_STEPS:
            for name, (step_fn, _) in OPTIMIZERS.items():
                if step not in snapshots_by_opt[name]:
                    continue
                weights, grads = snapshots_by_opt[name][step]
                measurement_records.append(
                    measure_direction_record(
                        seed=seed,
                        seed_index=seed_index,
                        step=step,
                        optimizer_name=name,
                        step_fn=step_fn,
                        best_grid_lr=best_grid_lr_by_optimizer[name],
                        weights=weights,
                        grads=grads,
                        X=X,
                        Y=Y,
                    )
                )

    elapsed = time.time() - t0
    results = {
        "script_path": str(Path(__file__).resolve()),
        "config": config,
        "seeds": [int(seed) for seed in seeds],
        "best_grid_lr_records": best_grid_lr_records,
        "training_run_records": training_run_records,
        "records": measurement_records,
        "runtime_sec": float(elapsed),
        "notes": [
            "best_grid_lr means best within the scanned LR grid, not a continuous optimum",
            "Newton directions are approximate because the Hessian is finite-difference estimated and pseudoinverted",
            "the model is a tiny deep-linear network; results should be interpreted as toy-setting evidence only",
        ],
    }
    results.update(build_summary(results))

    if verbose:
        print_report(results)

    return results



def main():
    return run_experiment(verbose=True)


if __name__ == "__main__":
    main()
