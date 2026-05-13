#!/usr/bin/env python3
"""
H3b: Partial Equalization -- How much of Muon-like final-loss improvement
survives when only the top-k singular values are equalized?

Toy-scope framing
-----------------
This is a deep-linear, final-loss study. It probes whether partial singular-value
(SV) equalization recovers Muon-like optimization behavior in this toy setting.
It does NOT directly measure causal "singular-value rescue" during training:
there is no per-step spectral logging or direct mechanism attribution here.

Protocol
--------
For a gradient matrix G = U diag(sigma) V^T, define a partial polar factor:
  - top-k singular values are set to 1
  - the remaining tail singular values keep their relative ratios, but are
    rescaled as sigma_i / ||sigma_tail|| * sqrt(d - k)
This preserves the tail structure while matching the tail Frobenius energy to
what full equalization would give.

We sweep k in {1, 2, 4, 8, 16, 32}, where k = dim recovers the full polar
factor U V^T. We compare against a normalized-SGD baseline that accumulates raw
momentum and then Frobenius-normalizes the momentum before each step.

Key heuristic tests
-------------------
T1: Does k=1 capture >50% of the NormSGD-to-Muon final-loss gap?
T2: Is there a small-k knee (>80% gap captured with k < dim/2)?
T3: Does k=dim numerically match an explicit polar-factor / Muon reference?

Default setup: 4-layer, 32x32, 500 steps, 10 seeds, LR swept per method.
"""

import os
import time
import numpy as np


SCRIPT_PATH = os.path.abspath(__file__)
NOTEBOOK_PATH = os.path.splitext(SCRIPT_PATH)[0] + ".ipynb"


DEFAULT_CONFIG = {
    "dim": 32,
    "num_layers": 4,
    "num_steps": 500,
    "momentum": 0.9,
    "num_seeds": 10,
    "batch_size": 64,
    "k_values": [1, 2, 4, 8, 16, 32],
    "lr_candidates": [0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.002, 0.001],
    "seed_base": 42,
    "seed_stride": 137,
    "weight_seed_offset": 5000,
    "lr_tuning_seeds": 3,
    "divergence_threshold": 1e10,
    "zero_tol": 1e-15,
    "t3_matrix_samples": 5,
    "t3_matrix_seed": 9000,
    "t3_matrix_rtol": 1e-10,
    "t3_training_rtol": 1e-10,
    "t3_training_atol": 1e-12,
    "muon_like_loss_rtol": 0.05,
}

# Backward-friendly module constants for the default setup.
DIM = DEFAULT_CONFIG["dim"]
NUM_LAYERS = DEFAULT_CONFIG["num_layers"]
NUM_STEPS = DEFAULT_CONFIG["num_steps"]
MOMENTUM = DEFAULT_CONFIG["momentum"]
NUM_SEEDS = DEFAULT_CONFIG["num_seeds"]
BATCH_SIZE = DEFAULT_CONFIG["batch_size"]
K_VALUES = list(DEFAULT_CONFIG["k_values"])
LR_CANDIDATES = list(DEFAULT_CONFIG["lr_candidates"])


def resolve_config(config_overrides=None):
    cfg = dict(DEFAULT_CONFIG)
    cfg["k_values"] = list(DEFAULT_CONFIG["k_values"])
    cfg["lr_candidates"] = list(DEFAULT_CONFIG["lr_candidates"])

    if config_overrides:
        for key, value in config_overrides.items():
            if key in {"k_values", "lr_candidates"}:
                cfg[key] = list(value)
            else:
                cfg[key] = value

    cfg["k_values"] = [int(k) for k in cfg["k_values"]]
    cfg["lr_candidates"] = [float(lr) for lr in cfg["lr_candidates"]]

    if cfg["dim"] not in cfg["k_values"]:
        raise ValueError("config must include k=dim so the full-Muon endpoint exists")
    if cfg["lr_tuning_seeds"] > cfg["num_seeds"]:
        raise ValueError("lr_tuning_seeds cannot exceed num_seeds")
    if any(k < 0 or k > cfg["dim"] for k in cfg["k_values"]):
        raise ValueError("all k values must satisfy 0 <= k <= dim")

    return cfg


def make_seed_schedule(config=None):
    cfg = resolve_config(config)
    return [cfg["seed_base"] + i * cfg["seed_stride"] for i in range(cfg["num_seeds"])]


def finite_mean(losses):
    finite = [float(x) for x in losses if np.isfinite(x)]
    return float(np.mean(finite)) if finite else float("inf")


def summarize_losses(losses):
    finite = np.array([float(x) for x in losses if np.isfinite(x)], dtype=float)
    summary = {
        "n_total": int(len(losses)),
        "n_finite": int(finite.size),
        "mean": float("inf"),
        "std": float("nan"),
        "sem": float("nan"),
        "min": float("inf"),
        "max": float("inf"),
    }
    if finite.size == 0:
        return summary

    summary["mean"] = float(np.mean(finite))
    summary["min"] = float(np.min(finite))
    summary["max"] = float(np.max(finite))
    if finite.size == 1:
        summary["std"] = 0.0
        summary["sem"] = 0.0
    else:
        std = float(np.std(finite, ddof=1))
        summary["std"] = std
        summary["sem"] = float(std / np.sqrt(finite.size))
    return summary


def safe_ratio(numerator, denominator):
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) < 1e-30:
        return float("nan")
    return float(numerator / denominator)


def compute_gap_captured(norm_loss, muon_loss, candidate_loss):
    if not (np.isfinite(norm_loss) and np.isfinite(muon_loss) and np.isfinite(candidate_loss)):
        return float("nan")
    total_gap = norm_loss - muon_loss
    if abs(total_gap) < 1e-30:
        return float("nan")
    return float(100.0 * (norm_loss - candidate_loss) / total_gap)


def partial_polar(G, k):
    """
    Equalize the top-k singular values and tail-normalize the rest.

    Notes:
      - k=dim gives the full polar factor U V^T.
      - k=0 gives a per-layer Frobenius-normalized gradient before momentum.
        That is NOT the same optimizer as the NormSGD baseline implemented here,
        which normalizes accumulated momentum after raw-gradient accumulation.
    """
    U, sigma, Vt = np.linalg.svd(G, full_matrices=False)
    d = len(sigma)
    k_eff = min(max(int(k), 0), d)
    if np.linalg.norm(sigma) < DEFAULT_CONFIG["zero_tol"]:
        return G

    sigma_new = sigma.copy()
    sigma_new[:k_eff] = 1.0
    if k_eff < d:
        remaining = sigma[k_eff:]
        r_norm = np.linalg.norm(remaining)
        if r_norm > DEFAULT_CONFIG["zero_tol"]:
            sigma_new[k_eff:] = remaining / r_norm * np.sqrt(d - k_eff)

    return U @ np.diag(sigma_new) @ Vt


def polar_factor(G):
    """Return the orthogonal polar factor U V^T of G."""
    U, sigma, Vt = np.linalg.svd(G, full_matrices=False)
    if np.linalg.norm(sigma) < DEFAULT_CONFIG["zero_tol"]:
        return G
    return U @ Vt


def init_weights(seed, config=None):
    cfg = resolve_config(config)
    rng = np.random.RandomState(seed)
    return [np.eye(cfg["dim"]) + rng.randn(cfg["dim"], cfg["dim"]) * 0.1 for _ in range(cfg["num_layers"])]


def forward(weights, X):
    out = X.copy()
    for W in weights:
        out = W @ out
    return out


def compute_loss(weights, X, Y):
    pred = forward(weights, X)
    return float(0.5 * np.mean(np.sum((pred - Y) ** 2, axis=0)))


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


def train(weights_init, X, Y, lr, k, config=None):
    """Train with partial polar updates; k=dim is the full polar-factor endpoint."""
    cfg = resolve_config(config)
    weights = [W.copy() for W in weights_init]
    mom = [np.zeros_like(W) for W in weights]
    for _ in range(cfg["num_steps"]):
        loss = compute_loss(weights, X, Y)
        if not np.isfinite(loss) or loss > cfg["divergence_threshold"]:
            return float("inf")
        grads = compute_gradients(weights, X, Y)
        for i, grad in enumerate(grads):
            pp = partial_polar(grad, k)
            mom[i] = cfg["momentum"] * mom[i] + pp
            weights[i] = weights[i] - lr * mom[i]
    return compute_loss(weights, X, Y)


def train_muon(weights_init, X, Y, lr, config=None):
    """Explicit Muon reference: apply the full polar factor to each raw gradient."""
    cfg = resolve_config(config)
    weights = [W.copy() for W in weights_init]
    mom = [np.zeros_like(W) for W in weights]
    for _ in range(cfg["num_steps"]):
        loss = compute_loss(weights, X, Y)
        if not np.isfinite(loss) or loss > cfg["divergence_threshold"]:
            return float("inf")
        grads = compute_gradients(weights, X, Y)
        for i, grad in enumerate(grads):
            mu = polar_factor(grad)
            mom[i] = cfg["momentum"] * mom[i] + mu
            weights[i] = weights[i] - lr * mom[i]
    return compute_loss(weights, X, Y)


def train_norm_sgd(weights_init, X, Y, lr, config=None):
    """Normalized SGD baseline: normalize accumulated momentum, not the raw gradient."""
    cfg = resolve_config(config)
    weights = [W.copy() for W in weights_init]
    mom = [np.zeros_like(W) for W in weights]
    for _ in range(cfg["num_steps"]):
        loss = compute_loss(weights, X, Y)
        if not np.isfinite(loss) or loss > cfg["divergence_threshold"]:
            return float("inf")
        grads = compute_gradients(weights, X, Y)
        for i, grad in enumerate(grads):
            mom[i] = cfg["momentum"] * mom[i] + grad
            v_norm = np.linalg.norm(mom[i], ord="fro")
            step_dir = mom[i] / max(v_norm, 1e-12)
            weights[i] = weights[i] - lr * step_dir
    return compute_loss(weights, X, Y)


def make_data(seed, config=None):
    cfg = resolve_config(config)
    rng = np.random.RandomState(seed)
    X = rng.randn(cfg["dim"], cfg["batch_size"]) * 0.3
    Y = rng.randn(cfg["dim"], cfg["batch_size"]) * 0.3
    return X, Y


def evaluate_partial_for_seed(seed, lr, k, cfg):
    X, Y = make_data(seed, cfg)
    weights = init_weights(seed + cfg["weight_seed_offset"], cfg)
    return train(weights, X, Y, lr, k, cfg)


def evaluate_muon_for_seed(seed, lr, cfg):
    X, Y = make_data(seed, cfg)
    weights = init_weights(seed + cfg["weight_seed_offset"], cfg)
    return train_muon(weights, X, Y, lr, cfg)


def evaluate_norm_for_seed(seed, lr, cfg):
    X, Y = make_data(seed, cfg)
    weights = init_weights(seed + cfg["weight_seed_offset"], cfg)
    return train_norm_sgd(weights, X, Y, lr, cfg)


def compare_loss_vectors(losses_a, losses_b, rtol, atol):
    if len(losses_a) != len(losses_b):
        raise ValueError("loss vectors must have the same length")

    finite_a = []
    finite_b = []
    per_seed = []
    inf_pattern_match = True
    for idx, (a, b) in enumerate(zip(losses_a, losses_b)):
        a_finite = np.isfinite(a)
        b_finite = np.isfinite(b)
        if a_finite and b_finite:
            abs_diff = abs(a - b)
            rel_diff = abs_diff / max(abs(b), 1e-30)
            finite_a.append(a)
            finite_b.append(b)
        else:
            abs_diff = float("nan")
            rel_diff = float("nan")
            if a_finite != b_finite:
                inf_pattern_match = False
        per_seed.append(
            {
                "seed_index": int(idx),
                "a": float(a),
                "b": float(b),
                "abs_diff": float(abs_diff),
                "rel_diff": float(rel_diff),
            }
        )

    if finite_a:
        max_abs_diff = float(np.max([row["abs_diff"] for row in per_seed if np.isfinite(row["abs_diff"])]))
        max_rel_diff = float(np.max([row["rel_diff"] for row in per_seed if np.isfinite(row["rel_diff"])]))
        allclose = bool(np.allclose(np.array(finite_a), np.array(finite_b), rtol=rtol, atol=atol)) and inf_pattern_match
    else:
        max_abs_diff = float("nan")
        max_rel_diff = float("nan")
        allclose = bool(inf_pattern_match)

    return {
        "allclose": allclose,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "inf_pattern_match": bool(inf_pattern_match),
        "per_seed": per_seed,
        "rtol": float(rtol),
        "atol": float(atol),
    }


def run_experiment(config_overrides=None, verbose=True):
    cfg = resolve_config(config_overrides)
    seeds = make_seed_schedule(cfg)
    tuning_seeds = seeds[: cfg["lr_tuning_seeds"]]
    start_time = time.time()

    if verbose:
        print("=" * 100)
        print("H3b: PARTIAL EQUALIZATION -- Top-k SVs Capture Muon-like Final-Loss Advantage?")
        print("=" * 100)
        print(f"Script: {SCRIPT_PATH}")
        print(f"Notebook counterpart: {NOTEBOOK_PATH}")
        print(f"k values: {cfg['k_values']} (k={cfg['dim']} is the full polar-factor endpoint)")
        print(
            f"Network: {cfg['num_layers']}-layer, {cfg['dim']}x{cfg['dim']}, "
            f"{cfg['num_steps']} steps, batch_size={cfg['batch_size']}, {cfg['num_seeds']} seeds"
        )
        print(f"LR tuning seeds: {tuning_seeds}")
        print(f"Evaluation seeds: {seeds}")
        print("Note: this is a final-loss probe, not a direct mechanistic measurement.")
        print()

    phase1_partial = {}
    best_lrs = {}

    if verbose:
        print("Phase 1: LR sweep per k...")
    for k in cfg["k_values"]:
        lr_records = []
        best_lr = cfg["lr_candidates"][-1]
        best_loss = float("inf")
        for lr in cfg["lr_candidates"]:
            losses = [evaluate_partial_for_seed(seed, lr, k, cfg) for seed in tuning_seeds]
            summary = summarize_losses(losses)
            record = {
                "lr": float(lr),
                "seed_losses": [float(x) for x in losses],
                "summary": summary,
            }
            lr_records.append(record)
            if summary["mean"] < best_loss:
                best_loss = summary["mean"]
                best_lr = lr
        best_lrs[k] = float(best_lr)
        phase1_partial[k] = {
            "best_lr": float(best_lr),
            "best_mean_loss": float(best_loss),
            "lr_records": lr_records,
        }
        if verbose:
            landscape = "  ".join(
                f"{record['lr']:.3f}:{record['summary']['mean']:.2e}"
                if np.isfinite(record["summary"]["mean"])
                else f"{record['lr']:.3f}:diverged"
                for record in lr_records
            )
            print(f"  k={k:>3}: best_lr={best_lr:.4f}, best_mean_loss={best_loss:.6e}")
            print(f"         LR landscape: {landscape}")

    norm_lr_records = []
    best_norm_lr = cfg["lr_candidates"][-1]
    best_norm_loss = float("inf")
    if verbose:
        print("  Sweeping LR for NormSGD baseline...")
    for lr in cfg["lr_candidates"]:
        losses = [evaluate_norm_for_seed(seed, lr, cfg) for seed in tuning_seeds]
        summary = summarize_losses(losses)
        record = {
            "lr": float(lr),
            "seed_losses": [float(x) for x in losses],
            "summary": summary,
        }
        norm_lr_records.append(record)
        if summary["mean"] < best_norm_loss:
            best_norm_loss = summary["mean"]
            best_norm_lr = lr
    phase1_norm = {
        "best_lr": float(best_norm_lr),
        "best_mean_loss": float(best_norm_loss),
        "lr_records": norm_lr_records,
    }
    if verbose:
        print(f"  NormSGD: best_lr={best_norm_lr:.4f}, best_mean_loss={best_norm_loss:.6e}")

    if verbose:
        print("\nPhase 2: full evaluation on all seeds...")
    phase2_partial = {}
    partial_mean_losses = {}
    for k in cfg["k_values"]:
        losses = [evaluate_partial_for_seed(seed, best_lrs[k], k, cfg) for seed in seeds]
        summary = summarize_losses(losses)
        phase2_partial[k] = {
            "lr": float(best_lrs[k]),
            "seed_losses": [float(x) for x in losses],
            "summary": summary,
        }
        partial_mean_losses[k] = summary["mean"]
        if verbose:
            print(
                f"  k={k:>3}: mean={summary['mean']:.6e}, std={summary['std']:.6e}, "
                f"sem={summary['sem']:.6e}, finite={summary['n_finite']}/{summary['n_total']}"
            )

    norm_losses = [evaluate_norm_for_seed(seed, best_norm_lr, cfg) for seed in seeds]
    norm_summary = summarize_losses(norm_losses)
    phase2_norm = {
        "lr": float(best_norm_lr),
        "seed_losses": [float(x) for x in norm_losses],
        "summary": norm_summary,
    }
    if verbose:
        print(
            f"  NormSGD: mean={norm_summary['mean']:.6e}, std={norm_summary['std']:.6e}, "
            f"sem={norm_summary['sem']:.6e}, finite={norm_summary['n_finite']}/{norm_summary['n_total']}"
        )

    muon_reference_lr = float(best_lrs[cfg["dim"]])
    muon_reference_losses = [evaluate_muon_for_seed(seed, muon_reference_lr, cfg) for seed in seeds]
    muon_reference_summary = summarize_losses(muon_reference_losses)
    phase2_muon_reference = {
        "lr": muon_reference_lr,
        "seed_losses": [float(x) for x in muon_reference_losses],
        "summary": muon_reference_summary,
    }
    if verbose:
        print(
            f"  Muon reference @ lr={muon_reference_lr:.4f}: mean={muon_reference_summary['mean']:.6e}, "
            f"std={muon_reference_summary['std']:.6e}, sem={muon_reference_summary['sem']:.6e}, "
            f"finite={muon_reference_summary['n_finite']}/{muon_reference_summary['n_total']}"
        )

    matrix_rel_errors = []
    matrix_abs_errors = []
    matrix_trials = []
    for i in range(cfg["t3_matrix_samples"]):
        rng = np.random.RandomState(cfg["t3_matrix_seed"] + i)
        G = rng.randn(cfg["dim"], cfg["dim"])
        partial_full = partial_polar(G, cfg["dim"])
        explicit_full = polar_factor(G)
        abs_err = np.linalg.norm(partial_full - explicit_full, ord="fro")
        rel_err = abs_err / max(np.linalg.norm(explicit_full, ord="fro"), 1e-30)
        matrix_abs_errors.append(float(abs_err))
        matrix_rel_errors.append(float(rel_err))
        matrix_trials.append(
            {
                "sample_index": int(i),
                "matrix_seed": int(cfg["t3_matrix_seed"] + i),
                "abs_error_fro": float(abs_err),
                "rel_error_fro": float(rel_err),
            }
        )
    matrix_check = {
        "samples": int(cfg["t3_matrix_samples"]),
        "rtol": float(cfg["t3_matrix_rtol"]),
        "max_abs_error_fro": float(max(matrix_abs_errors) if matrix_abs_errors else float("nan")),
        "max_rel_error_fro": float(max(matrix_rel_errors) if matrix_rel_errors else float("nan")),
        "passed": bool(max(matrix_rel_errors) <= cfg["t3_matrix_rtol"] if matrix_rel_errors else False),
        "trials": matrix_trials,
    }

    training_equivalence = compare_loss_vectors(
        phase2_partial[cfg["dim"]]["seed_losses"],
        muon_reference_losses,
        rtol=cfg["t3_training_rtol"],
        atol=cfg["t3_training_atol"],
    )

    muon_loss = partial_mean_losses[cfg["dim"]]
    norm_loss = norm_summary["mean"]
    total_gap = norm_loss - muon_loss if np.isfinite(norm_loss) and np.isfinite(muon_loss) else float("nan")

    gap_captured_pct_by_k = {}
    vs_norm_ratio_by_k = {}
    vs_muon_ratio_by_k = {}
    summary_rows = []
    for k in cfg["k_values"]:
        mean_loss = partial_mean_losses[k]
        gap_captured = compute_gap_captured(norm_loss, muon_loss, mean_loss)
        gap_captured_pct_by_k[k] = gap_captured
        vs_norm_ratio_by_k[k] = safe_ratio(norm_loss, mean_loss)
        vs_muon_ratio_by_k[k] = safe_ratio(mean_loss, muon_loss)
        summary_rows.append(
            {
                "method": "partial_equalization",
                "k": int(k),
                "best_lr": float(best_lrs[k]),
                "mean_loss": float(mean_loss),
                "std_loss": float(phase2_partial[k]["summary"]["std"]),
                "sem_loss": float(phase2_partial[k]["summary"]["sem"]),
                "n_finite": int(phase2_partial[k]["summary"]["n_finite"]),
                "gap_captured_pct": float(gap_captured),
                "vs_normsgd": float(vs_norm_ratio_by_k[k]),
                "vs_muon": float(vs_muon_ratio_by_k[k]),
            }
        )

    summary_rows.append(
        {
            "method": "norm_sgd",
            "k": None,
            "best_lr": float(best_norm_lr),
            "mean_loss": float(norm_summary["mean"]),
            "std_loss": float(norm_summary["std"]),
            "sem_loss": float(norm_summary["sem"]),
            "n_finite": int(norm_summary["n_finite"]),
            "gap_captured_pct": 0.0,
            "vs_normsgd": 1.0,
            "vs_muon": float(safe_ratio(norm_summary["mean"], muon_loss)),
        }
    )
    summary_rows.append(
        {
            "method": "muon_reference",
            "k": int(cfg["dim"]),
            "best_lr": float(muon_reference_lr),
            "mean_loss": float(muon_reference_summary["mean"]),
            "std_loss": float(muon_reference_summary["std"]),
            "sem_loss": float(muon_reference_summary["sem"]),
            "n_finite": int(muon_reference_summary["n_finite"]),
            "gap_captured_pct": float(compute_gap_captured(norm_loss, muon_loss, muon_reference_summary["mean"])),
            "vs_normsgd": float(safe_ratio(norm_loss, muon_reference_summary["mean"])),
            "vs_muon": float(safe_ratio(muon_reference_summary["mean"], muon_loss)),
        }
    )

    positive_gap = np.isfinite(total_gap) and total_gap > 0
    gap_k1 = gap_captured_pct_by_k[1] if 1 in gap_captured_pct_by_k else float("nan")
    t1_pass = bool(positive_gap and np.isfinite(gap_k1) and gap_k1 > 50.0)

    knee_k = None
    knee_gap = float("nan")
    if positive_gap:
        for k in cfg["k_values"]:
            pct = gap_captured_pct_by_k[k]
            if np.isfinite(pct) and pct > 80.0:
                knee_k = int(k)
                knee_gap = float(pct)
                break
    t2_pass = bool(knee_k is not None and knee_k < cfg["dim"] // 2)

    muon_like_threshold = (1.0 + cfg["muon_like_loss_rtol"]) * muon_loss if np.isfinite(muon_loss) else float("nan")
    smallest_k_within_muon = None
    if np.isfinite(muon_like_threshold):
        for k in cfg["k_values"]:
            if np.isfinite(partial_mean_losses[k]) and partial_mean_losses[k] <= muon_like_threshold:
                smallest_k_within_muon = int(k)
                break

    t3_pass = bool(matrix_check["passed"] and training_equivalence["allclose"])

    tests = {
        "T1": {
            "question": "Does k=1 capture >50% of the NormSGD-to-Muon mean-loss gap?",
            "gap_captured_pct": float(gap_k1),
            "threshold_pct": 50.0,
            "gap_interpretable": bool(positive_gap),
            "passed": t1_pass,
        },
        "T2": {
            "question": f"Is there a small-k knee with >80% gap captured at k < {cfg['dim'] // 2}?",
            "threshold_pct": 80.0,
            "knee_k": knee_k,
            "knee_gap_captured_pct": float(knee_gap),
            "gap_interpretable": bool(positive_gap),
            "passed": t2_pass,
        },
        "T3": {
            "question": "Does k=dim numerically match an explicit polar-factor / Muon reference?",
            "matrix_equivalence": matrix_check,
            "training_equivalence": training_equivalence,
            "passed": t3_pass,
        },
    }

    runtime_sec = float(time.time() - start_time)

    results = {
        "metadata": {
            "experiment": "H3b_PARTIAL_EQUALIZATION",
            "script_path": SCRIPT_PATH,
            "notebook_path": NOTEBOOK_PATH,
            "description": "Toy deep-linear final-loss probe of top-k singular-value equalization.",
        },
        "config": {
            **cfg,
            "k_values": list(cfg["k_values"]),
            "lr_candidates": list(cfg["lr_candidates"]),
        },
        "seeds": {
            "all": list(seeds),
            "lr_tuning": list(tuning_seeds),
            "weight_seed_offset": int(cfg["weight_seed_offset"]),
        },
        "phase1": {
            "partial": phase1_partial,
            "norm_sgd": phase1_norm,
        },
        "phase2": {
            "partial": phase2_partial,
            "norm_sgd": phase2_norm,
            "muon_reference": phase2_muon_reference,
        },
        "derived_metrics": {
            "muon_loss_mean": float(muon_loss),
            "normsgd_loss_mean": float(norm_loss),
            "total_gap": float(total_gap),
            "gap_captured_pct_by_k": gap_captured_pct_by_k,
            "vs_normsgd_ratio_by_k": vs_norm_ratio_by_k,
            "vs_muon_ratio_by_k": vs_muon_ratio_by_k,
            "summary_rows": summary_rows,
            "muon_like_loss_rtol": float(cfg["muon_like_loss_rtol"]),
            "muon_like_loss_threshold": float(muon_like_threshold),
            "smallest_k_within_muon_like_mean_loss": smallest_k_within_muon,
        },
        "tests": tests,
        "runtime_sec": runtime_sec,
    }

    if verbose:
        print(f"\n{'=' * 100}")
        print("RESULTS: Mean final loss vs number of equalized singular values")
        print(f"{'=' * 100}")
        print(f"  {'method':>18}  {'k':>5}  {'best_lr':>8}  {'mean_loss':>14}  {'std':>12}  {'% gap':>8}")
        print("  " + "-" * 82)
        print(
            f"  {'NormSGD':>18}  {'-':>5}  {best_norm_lr:>8.4f}  {norm_summary['mean']:>14.6e}  "
            f"{norm_summary['std']:>12.6e}  {0.0:>7.1f}%"
        )
        for k in cfg["k_values"]:
            row = phase2_partial[k]
            gap_pct = gap_captured_pct_by_k[k]
            marker = " <-- k=dim partial path" if k == cfg["dim"] else ""
            print(
                f"  {'partial_equalization':>18}  {k:>5}  {row['lr']:>8.4f}  {row['summary']['mean']:>14.6e}  "
                f"{row['summary']['std']:>12.6e}  {gap_pct:>7.1f}%{marker}"
            )
        print(
            f"  {'Muon reference':>18}  {cfg['dim']:>5}  {muon_reference_lr:>8.4f}  {muon_reference_summary['mean']:>14.6e}  "
            f"{muon_reference_summary['std']:>12.6e}  {compute_gap_captured(norm_loss, muon_loss, muon_reference_summary['mean']):>7.1f}%"
        )

        print(f"\n{'=' * 100}")
        print("HEURISTIC TESTS")
        print(f"{'=' * 100}")
        print(f"  T1: {tests['T1']['question']}")
        print(
            f"      gap captured by k=1 = {tests['T1']['gap_captured_pct']:.1f}% | "
            f"interpretable={tests['T1']['gap_interpretable']} | {'PASS' if tests['T1']['passed'] else 'FAIL'}"
        )
        print(f"  T2: {tests['T2']['question']}")
        if knee_k is None:
            print(f"      no k crossed 80% gap captured | interpretable={tests['T2']['gap_interpretable']} | FAIL")
        else:
            print(
                f"      first k over 80% gap = {knee_k} ({knee_gap:.1f}%) | "
                f"interpretable={tests['T2']['gap_interpretable']} | {'PASS' if tests['T2']['passed'] else 'FAIL'}"
            )
        print(f"  T3: {tests['T3']['question']}")
        print(
            f"      matrix max rel err = {matrix_check['max_rel_error_fro']:.3e} "
            f"(tol {matrix_check['rtol']:.1e}) | passed={matrix_check['passed']}"
        )
        print(
            f"      training max rel diff = {training_equivalence['max_rel_diff']:.3e} "
            f"(rtol {training_equivalence['rtol']:.1e}, atol {training_equivalence['atol']:.1e}) | "
            f"passed={training_equivalence['allclose']}"
        )
        print(f"      overall T3 = {'PASS' if t3_pass else 'FAIL'}")

        if smallest_k_within_muon is None:
            print(
                f"\nNo tested k reached mean loss within {100 * cfg['muon_like_loss_rtol']:.1f}% of the k=dim mean-loss endpoint."
            )
        else:
            print(
                f"\nSmallest tested k within {100 * cfg['muon_like_loss_rtol']:.1f}% of the k=dim mean loss: "
                f"k={smallest_k_within_muon}."
            )
        print(f"Runtime: {runtime_sec:.2f} seconds")

    return results


def main():
    return run_experiment(verbose=True)


if __name__ == "__main__":
    main()
