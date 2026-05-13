#!/usr/bin/env python3
"""
Experiment 2.13: Activation Curvature Toy Sweep
===============================================

This script runs a single-seed, final-loss-only toy comparison across six
activations for four update-rule families:
  (a) SGD
  (b) Muon-style Newton-Schulz orthogonalized gradients
  (c) SGD + static weight orthogonality penalty
  (d) SGD + partial orthogonal-gradient blend (alpha sweep)

Current scope and interpretation:
  - one synthetic dataset, one seed, one initialization recipe
  - 500 steps per run
  - the current verdict is driven by partial-blend recovery, not by the
    static weight-penalty control
  - the static weight penalty is still reported for comparison

Important implementation note:
  - train_partial_ortho() uses LR_SGD together with a norm-matched
    orthogonalized gradient. Even at alpha=1.0, it would not be numerically
    identical to train_muon(), which uses LR_MUON and an unscaled orthogonal
    update. In the current default sweep, alpha stops at 0.9.

The curvature proxy is mean |f''(x)| over x ~ N(0, 1). This is a convenient
proxy for this toy study, not a direct measurement of empirical
preactivation statistics during training.
"""

import time
import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

WIDTH = 32
DEPTH = 4
NUM_STEPS = 500
LR_SGD = 0.01
LR_MUON = 0.02
ORTHO_LAMBDA = 0.003
NS_ITERS = 5
BATCH_SIZE = 64
INPUT_DIM = 32
OUTPUT_DIM = 32
SEED = 42
X_SCALE = 0.5
Y_SCALE = 0.3
ALPHA_SWEEP = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)


# =============================================================================
# ACTIVATION FUNCTIONS
# =============================================================================

def act_linear(x):
    return x.copy()


def act_relu(x):
    return np.maximum(0, x)


def act_leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)


def act_gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def act_tanh(x):
    return np.tanh(x)


def act_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))



def dact_linear(x):
    return np.ones_like(x)


def dact_relu(x):
    return (x > 0).astype(float)


def dact_leaky_relu(x, alpha=0.1):
    return np.where(x > 0, 1.0, alpha)


def dact_gelu(x):
    eps = 1e-5
    return (act_gelu(x + eps) - act_gelu(x - eps)) / (2 * eps)


def dact_tanh(x):
    return 1.0 - np.tanh(x) ** 2


def dact_sigmoid(x):
    s = act_sigmoid(x)
    return s * (1.0 - s)


ACTIVATIONS = {
    "Linear": (act_linear, dact_linear),
    "ReLU": (act_relu, dact_relu),
    "LeakyReLU(0.1)": (act_leaky_relu, dact_leaky_relu),
    "GELU": (act_gelu, dact_gelu),
    "Tanh": (act_tanh, dact_tanh),
    "Sigmoid": (act_sigmoid, dact_sigmoid),
}


# =============================================================================
# COMPUTE MEAN |f''(x)| NUMERICALLY
# =============================================================================

def estimate_mean_second_derivative(act_fn, n_samples=10000, seed=42):
    """Numerically estimate mean |f''(x)| over x ~ N(0, 1)."""
    rng = np.random.RandomState(seed)
    x = rng.randn(n_samples)
    h = 1e-4
    fpp = (act_fn(x + h) - 2.0 * act_fn(x) + act_fn(x - h)) / (h * h)
    return float(np.mean(np.abs(fpp)))


# =============================================================================
# NETWORK UTILITIES
# =============================================================================

def init_weights(num_layers, width, seed):
    rng = np.random.RandomState(seed)
    weights = []
    for _ in range(num_layers):
        std = np.sqrt(2.0 / (width + width))
        W = rng.randn(width, width) * std
        weights.append(W.copy())
    return weights



def forward(weights, X, act_fn):
    activations = [X.copy()]
    pre_activations = []
    out = X.copy()
    for W in weights:
        z = W @ out
        pre_activations.append(z)
        out = act_fn(z)
        activations.append(out)
    return activations, pre_activations



def compute_loss(weights, X, Y_target, act_fn):
    activations, _ = forward(weights, X, act_fn)
    Y_pred = activations[-1]
    diff = Y_pred - Y_target
    return 0.5 * np.mean(diff ** 2)



def compute_gradients(weights, X, Y_target, act_fn, dact_fn):
    num_layers = len(weights)
    batch_size = X.shape[1]
    activations, pre_activations = forward(weights, X, act_fn)
    Y_pred = activations[-1]
    diff = Y_pred - Y_target
    delta = diff / batch_size

    grads = [None] * num_layers
    for l in range(num_layers - 1, -1, -1):
        act_deriv = dact_fn(pre_activations[l])
        delta_z = delta * act_deriv
        grads[l] = delta_z @ activations[l].T
        if l > 0:
            delta = weights[l].T @ delta_z
    return grads



def ortho_penalty_gradient(W):
    """Gradient of ||W^T W - I||_F^2."""
    WtW = W.T @ W
    I = np.eye(W.shape[0])
    return 4.0 * W @ (WtW - I)



def newton_schulz_orthogonalize(G, num_iters=5):
    """Approximate an orthogonalized version of G via fixed Newton-Schulz steps."""
    norm = np.linalg.norm(G, "fro")
    if norm < 1e-12:
        return G
    X = G / norm
    for _ in range(num_iters):
        A = X.T @ X
        X = (15.0 / 8.0) * X - (10.0 / 8.0) * X @ A + (3.0 / 8.0) * X @ A @ A
    return X


# =============================================================================
# TRAINING ROUTINES
# =============================================================================

def safe_loss(weights, X, Y, act_fn):
    loss = compute_loss(weights, X, Y, act_fn)
    if np.isnan(loss) or np.isinf(loss):
        return 1e10
    return float(loss)



def train_sgd(weights, X, Y, num_steps, lr, act_fn, dact_fn):
    weights = [W.copy() for W in weights]
    for step in range(num_steps):
        grads = compute_gradients(weights, X, Y, act_fn, dact_fn)
        for i in range(len(weights)):
            weights[i] -= lr * grads[i]
        if step % 50 == 0 and safe_loss(weights, X, Y, act_fn) > 1e8:
            return weights, 1e10
    return weights, safe_loss(weights, X, Y, act_fn)



def train_muon(weights, X, Y, num_steps, lr, act_fn, dact_fn, ns_iters=5):
    weights = [W.copy() for W in weights]
    for step in range(num_steps):
        grads = compute_gradients(weights, X, Y, act_fn, dact_fn)
        for i in range(len(weights)):
            G_orth = newton_schulz_orthogonalize(grads[i], ns_iters)
            weights[i] -= lr * G_orth
        if step % 50 == 0 and safe_loss(weights, X, Y, act_fn) > 1e8:
            return weights, 1e10
    return weights, safe_loss(weights, X, Y, act_fn)



def train_sgd_ortho_penalty(weights, X, Y, num_steps, lr, lam, act_fn, dact_fn):
    """SGD with a static weight orthogonality penalty."""
    weights = [W.copy() for W in weights]
    for step in range(num_steps):
        grads = compute_gradients(weights, X, Y, act_fn, dact_fn)
        for i in range(len(weights)):
            pen_grad = ortho_penalty_gradient(weights[i])
            weights[i] -= lr * (grads[i] + lam * pen_grad)
        if step % 50 == 0 and safe_loss(weights, X, Y, act_fn) > 1e8:
            return weights, 1e10
    return weights, safe_loss(weights, X, Y, act_fn)



def train_partial_ortho(weights, X, Y, num_steps, lr, alpha, act_fn, dact_fn, ns_iters=5):
    """Blend raw and orthogonalized gradients.

    The update is:
        blended = (1 - alpha) * G + alpha * G_orth_scaled

    where G_orth_scaled is the Newton-Schulz orthogonalized gradient rescaled to
    match ||G||_F before blending. This makes alpha=0 equivalent to SGD, but even
    alpha=1 would still not be numerically identical to train_muon(), because this
    routine uses LR_SGD and norm matching while train_muon() uses LR_MUON and the
    unscaled orthogonalized update.
    """
    weights = [W.copy() for W in weights]
    for step in range(num_steps):
        grads = compute_gradients(weights, X, Y, act_fn, dact_fn)
        for i in range(len(weights)):
            G = grads[i]
            G_orth = newton_schulz_orthogonalize(G, ns_iters)
            gn = np.linalg.norm(G, "fro")
            on = np.linalg.norm(G_orth, "fro")
            if on > 1e-12:
                G_orth_scaled = G_orth * (gn / on)
            else:
                G_orth_scaled = G_orth
            blended = (1.0 - alpha) * G + alpha * G_orth_scaled
            weights[i] -= lr * blended
        if step % 50 == 0 and safe_loss(weights, X, Y, act_fn) > 1e8:
            return weights, 1e10
    return weights, safe_loss(weights, X, Y, act_fn)


# =============================================================================
# ANALYSIS HELPERS
# =============================================================================

def build_config():
    return {
        "width": WIDTH,
        "depth": DEPTH,
        "num_steps": NUM_STEPS,
        "lr_sgd": LR_SGD,
        "lr_muon": LR_MUON,
        "ortho_lambda": ORTHO_LAMBDA,
        "ns_iters": NS_ITERS,
        "batch_size": BATCH_SIZE,
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "seed": SEED,
        "x_scale": X_SCALE,
        "y_scale": Y_SCALE,
        "alpha_sweep": [float(alpha) for alpha in ALPHA_SWEEP],
        "activation_count": len(ACTIVATIONS),
        "optimizer_variants_reported": [
            "SGD",
            "Muon",
            "SGD + static weight orthogonality penalty",
            "Best partial orthogonal-gradient blend",
        ],
        "training_runs_per_activation": 3 + len(ALPHA_SWEEP),
        "total_training_runs": len(ACTIVATIONS) * (3 + len(ALPHA_SWEEP)),
        "scope": "single-seed final-loss-only toy study",
        "primary_analysis_metric": "recovery_partial",
        "control_metric": "recovery_penalty",
        "curvature_proxy": "mean_abs_second_derivative_over_standard_normal",
        "partial_blend_note": (
            "Partial blend uses LR_SGD and a norm-matched orthogonalized update; "
            "it is not identical to Muon even if alpha were 1.0."
        ),
    }



def make_dataset(seed):
    rng = np.random.RandomState(seed)
    X = rng.randn(INPUT_DIM, BATCH_SIZE) * X_SCALE
    Y = rng.randn(OUTPUT_DIM, BATCH_SIZE) * Y_SCALE
    return X, Y



def summarize_data(X, Y):
    return {
        "X_shape": list(X.shape),
        "Y_shape": list(Y.shape),
        "X_mean": float(X.mean()),
        "X_std": float(X.std()),
        "X_min": float(X.min()),
        "X_max": float(X.max()),
        "Y_mean": float(Y.mean()),
        "Y_std": float(Y.std()),
        "Y_min": float(Y.min()),
        "Y_max": float(Y.max()),
    }



def average_ranks(values, atol=1e-12):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)

    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and abs(values[order[j + 1]] - values[order[i]]) <= atol:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks



def spearman_rank_correlation(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rx = average_ranks(x)
    ry = average_ranks(y)
    rx_centered = rx - rx.mean()
    ry_centered = ry - ry.mean()
    denom = np.linalg.norm(rx_centered) * np.linalg.norm(ry_centered)
    if denom < 1e-12:
        return 0.0
    return float((rx_centered @ ry_centered) / denom)



def concordance_analysis(sorted_names, per_activation):
    n_pairs = 0
    n_correct = 0
    inversions = []

    for i in range(len(sorted_names)):
        for j in range(i + 1, len(sorted_names)):
            name_i = sorted_names[i]
            name_j = sorted_names[j]
            curv_i = per_activation[name_i]["curvature"]
            curv_j = per_activation[name_j]["curvature"]
            rec_i = per_activation[name_i]["recovery_partial"]
            rec_j = per_activation[name_j]["recovery_partial"]

            if abs(curv_i - curv_j) < 1e-12:
                continue

            n_pairs += 1
            if rec_i >= rec_j - 1e-12:
                n_correct += 1
            else:
                inversions.append(
                    {
                        "lower_curvature_activation": name_i,
                        "higher_curvature_activation": name_j,
                        "lower_curvature": float(curv_i),
                        "higher_curvature": float(curv_j),
                        "lower_curvature_recovery_partial": float(rec_i),
                        "higher_curvature_recovery_partial": float(rec_j),
                    }
                )

    fraction = (n_correct / n_pairs) if n_pairs > 0 else 0.0
    return {
        "n_pairs": int(n_pairs),
        "n_concordant": int(n_correct),
        "fraction": float(fraction),
        "inversions": inversions,
    }



def build_tests(per_activation, sorted_names, spearman_partial, concordance_fraction):
    activation_names = list(ACTIVATIONS.keys())
    n_acts = len(activation_names)

    muon_wins = sum(1 for name in activation_names if per_activation[name]["loss_muon"] < per_activation[name]["loss_sgd"])
    positive_rec = sum(1 for name in activation_names if per_activation[name]["recovery_partial"] > 10.0)

    zero_curv = [name for name in sorted_names if per_activation[name]["curvature"] < 0.01]
    nonzero_curv = [name for name in sorted_names if per_activation[name]["curvature"] >= 0.01]

    mean_zero_rec = None
    mean_nz_rec = None
    test5_pass = False
    test5_metric = "not enough groups"
    if zero_curv and nonzero_curv:
        mean_zero_rec = float(np.mean([per_activation[name]["recovery_partial"] for name in zero_curv]))
        mean_nz_rec = float(np.mean([per_activation[name]["recovery_partial"] for name in nonzero_curv]))
        test5_pass = mean_zero_rec > mean_nz_rec
        test5_metric = f"zero={mean_zero_rec:.1f}%, nonzero={mean_nz_rec:.1f}%"

    tests = [
        {
            "id": "test1_muon_beats_sgd_majority",
            "description": "Muon beats SGD for at least 50% of activations",
            "analysis_target": "loss_muon_vs_loss_sgd",
            "metric_display": f"{muon_wins}/{n_acts}",
            "passed": bool(muon_wins >= n_acts * 0.5),
        },
        {
            "id": "test2_partial_recovery_present",
            "description": "Partial-blend recovery exceeds 10% for at least 3 activations",
            "analysis_target": "recovery_partial",
            "metric_display": f"{positive_rec}/{n_acts}",
            "passed": bool(positive_rec >= 3),
        },
        {
            "id": "test3_negative_spearman_partial",
            "description": "Spearman rho between curvature and partial recovery is below -0.3",
            "analysis_target": "recovery_partial",
            "metric_display": f"rho={spearman_partial:.4f}",
            "passed": bool(spearman_partial < -0.3),
        },
        {
            "id": "test4_partial_concordance",
            "description": "Pairwise concordance for decreasing partial recovery is at least 60%",
            "analysis_target": "recovery_partial",
            "metric_display": f"{concordance_fraction:.0%}",
            "passed": bool(concordance_fraction >= 0.60),
        },
        {
            "id": "test5_zero_vs_nonzero_curvature_partial",
            "description": "Zero-curvature activations have higher mean partial recovery than nonzero-curvature activations",
            "analysis_target": "recovery_partial",
            "metric_display": test5_metric,
            "passed": bool(test5_pass),
        },
    ]

    core_pass = (tests[2]["passed"] or tests[3]["passed"]) and tests[0]["passed"]
    direction_pass = spearman_partial < 0.0

    if core_pass:
        overall_verdict = "PASS"
        overall_message = (
            "Within this toy single-seed sweep, partial-blend recovery tends to decrease "
            "with activation curvature."
        )
    elif direction_pass and tests[0]["passed"]:
        overall_verdict = "PARTIAL PASS"
        overall_message = (
            "The direction of the partial-blend trend is negative, but the monotonic evidence "
            "is weaker than the nominal threshold."
        )
    else:
        overall_verdict = "FAIL"
        overall_message = (
            "This single-seed toy sweep does not show the expected negative relationship between "
            "curvature and partial-blend recovery."
        )

    group_summary = {
        "zero_curvature_activation_names": zero_curv,
        "nonzero_curvature_activation_names": nonzero_curv,
        "mean_zero_curvature_recovery_partial": mean_zero_rec,
        "mean_nonzero_curvature_recovery_partial": mean_nz_rec,
    }

    return tests, overall_verdict, overall_message, group_summary


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment(verbose=False):
    start_time = time.time()
    np.random.seed(SEED)

    config = build_config()
    X, Y = make_dataset(SEED)
    data_stats = summarize_data(X, Y)

    curvatures = {}
    for name, (act_fn, _) in ACTIVATIONS.items():
        curvatures[name] = estimate_mean_second_derivative(act_fn)

    per_activation = {}
    for name, (act_fn, dact_fn) in ACTIVATIONS.items():
        weights_init = init_weights(DEPTH, WIDTH, seed=SEED)

        _, loss_sgd = train_sgd(weights_init, X, Y, NUM_STEPS, LR_SGD, act_fn, dact_fn)
        _, loss_muon = train_muon(weights_init, X, Y, NUM_STEPS, LR_MUON, act_fn, dact_fn, NS_ITERS)
        _, loss_penalty = train_sgd_ortho_penalty(
            weights_init, X, Y, NUM_STEPS, LR_SGD, ORTHO_LAMBDA, act_fn, dact_fn
        )

        best_alpha = 0.0
        best_loss_partial = float(loss_sgd)
        alpha_records = []
        for alpha in ALPHA_SWEEP:
            _, loss_partial = train_partial_ortho(
                weights_init, X, Y, NUM_STEPS, LR_SGD, alpha, act_fn, dact_fn, NS_ITERS
            )
            loss_partial = float(loss_partial)
            alpha_records.append({"alpha": float(alpha), "loss": loss_partial})
            if loss_partial < best_loss_partial:
                best_loss_partial = loss_partial
                best_alpha = float(alpha)

        gap = float(loss_sgd - loss_muon)
        if gap > 1e-8 and loss_muon < loss_sgd:
            recovery_penalty = float((loss_sgd - loss_penalty) / gap * 100.0)
            recovery_partial = float((loss_sgd - best_loss_partial) / gap * 100.0)
        else:
            recovery_penalty = 0.0
            recovery_partial = 0.0

        per_activation[name] = {
            "curvature": float(curvatures[name]),
            "loss_sgd": float(loss_sgd),
            "loss_muon": float(loss_muon),
            "loss_penalty": float(loss_penalty),
            "loss_partial": float(best_loss_partial),
            "best_alpha": float(best_alpha),
            "recovery_penalty": float(recovery_penalty),
            "recovery_partial": float(recovery_partial),
            "sgd_muon_gap": gap,
            "muon_beats_sgd": bool(loss_muon < loss_sgd),
            "alpha_sweep": alpha_records,
        }

    sorted_activation_names = sorted(
        per_activation.keys(),
        key=lambda name: (per_activation[name]["curvature"], name),
    )

    curv_array = np.array([per_activation[name]["curvature"] for name in sorted_activation_names], dtype=float)
    partial_array = np.array([per_activation[name]["recovery_partial"] for name in sorted_activation_names], dtype=float)
    penalty_array = np.array([per_activation[name]["recovery_penalty"] for name in sorted_activation_names], dtype=float)

    spearman_partial = spearman_rank_correlation(curv_array, partial_array)
    spearman_penalty = spearman_rank_correlation(curv_array, penalty_array)
    concordance = concordance_analysis(sorted_activation_names, per_activation)

    tests, overall_verdict, overall_message, group_summary = build_tests(
        per_activation,
        sorted_activation_names,
        spearman_partial,
        concordance["fraction"],
    )

    runtime_sec = float(time.time() - start_time)
    mean_recovery_partial = float(np.mean([per_activation[name]["recovery_partial"] for name in sorted_activation_names]))
    mean_recovery_penalty = float(np.mean([per_activation[name]["recovery_penalty"] for name in sorted_activation_names]))
    muon_wins = sum(1 for name in ACTIVATIONS if per_activation[name]["muon_beats_sgd"])

    results = {
        "config": config,
        "activation_names": list(ACTIVATIONS.keys()),
        "sorted_activation_names": sorted_activation_names,
        "curvature_proxy_values": {name: float(curvatures[name]) for name in ACTIVATIONS},
        "data_stats": data_stats,
        "per_activation": per_activation,
        "concordance": concordance,
        "tests": tests,
        "summary": {
            "overall_verdict": overall_verdict,
            "overall_message": overall_message,
            "primary_analysis_metric": "recovery_partial",
            "control_metric": "recovery_penalty",
            "spearman_partial": float(spearman_partial),
            "spearman_penalty": float(spearman_penalty),
            "concordance_partial": float(concordance["fraction"]),
            "muon_wins": int(muon_wins),
            "activation_count": len(ACTIVATIONS),
            "mean_recovery_partial": mean_recovery_partial,
            "mean_recovery_penalty": mean_recovery_penalty,
            "runtime_sec": runtime_sec,
            "scope": config["scope"],
            "group_summary": group_summary,
        },
    }

    if verbose:
        print_experiment_report(results)
    return results



def print_experiment_report(results):
    config = results["config"]
    data_stats = results["data_stats"]
    per_activation = results["per_activation"]
    sorted_names = results["sorted_activation_names"]
    summary = results["summary"]
    concordance = results["concordance"]

    print("=" * 90)
    print("Experiment 2.13: Activation Curvature Toy Sweep")
    print("=" * 90)
    print("Scope: single-seed, final-loss-only toy study.")
    print("Primary analyzed metric: partial orthogonal-blend recovery.")
    print("Static weight orthogonality penalty is reported as a control, not the supported verdict target.")
    print()
    print(f"Config: depth={config['depth']}, width={config['width']}, steps={config['num_steps']}")
    print(
        f"  lr_sgd={config['lr_sgd']}, lr_muon={config['lr_muon']}, "
        f"ortho_lambda={config['ortho_lambda']}, ns_iters={config['ns_iters']}"
    )
    print(
        f"  activations={config['activation_count']}, alpha_sweep={config['alpha_sweep']}, "
        f"total training runs={config['total_training_runs']}"
    )
    print(f"  seed={config['seed']}, X scale={config['x_scale']}, Y scale={config['y_scale']}")
    print(f"  note: {config['partial_blend_note']}")
    print()
    print("Synthetic data summary")
    print(
        f"  X shape={tuple(data_stats['X_shape'])}, mean={data_stats['X_mean']:.4f}, "
        f"std={data_stats['X_std']:.4f}, range=[{data_stats['X_min']:.3f}, {data_stats['X_max']:.3f}]"
    )
    print(
        f"  Y shape={tuple(data_stats['Y_shape'])}, mean={data_stats['Y_mean']:.4f}, "
        f"std={data_stats['Y_std']:.4f}, range=[{data_stats['Y_min']:.3f}, {data_stats['Y_max']:.3f}]"
    )

    print()
    print("=" * 90)
    print("Results by activation (sorted by curvature)")
    print("=" * 90)
    print(
        f"{'Activation':<18} {'mean|fpp|':>9} {'SGD':>9} {'Muon':>9} {'Penalty':>9} "
        f"{'BestPart':>9} {'Best a':>7} {'Rec_pen%':>10} {'Rec_part%':>10}"
    )
    print("-" * 90)
    for name in sorted_names:
        r = per_activation[name]
        print(
            f"{name:<18} {r['curvature']:>9.4f} {r['loss_sgd']:>9.5f} {r['loss_muon']:>9.5f} "
            f"{r['loss_penalty']:>9.5f} {r['loss_partial']:>9.5f} {r['best_alpha']:>7.1f} "
            f"{r['recovery_penalty']:>10.1f} {r['recovery_partial']:>10.1f}"
        )

    print()
    print("Recovery vs curvature summary")
    print(f"  Spearman rho (curvature vs partial recovery): {summary['spearman_partial']:.4f}")
    print(f"  Spearman rho (curvature vs penalty recovery): {summary['spearman_penalty']:.4f}")
    print(
        f"  Concordance (lower curvature -> higher/equal partial recovery): "
        f"{concordance['n_concordant']}/{concordance['n_pairs']} ({summary['concordance_partial']:.0%})"
    )
    if concordance["inversions"]:
        print("  Inversions:")
        for inversion in concordance["inversions"]:
            print(
                "    "
                f"{inversion['lower_curvature_activation']} "
                f"(c={inversion['lower_curvature']:.4f}, r={inversion['lower_curvature_recovery_partial']:.1f}%) vs "
                f"{inversion['higher_curvature_activation']} "
                f"(c={inversion['higher_curvature']:.4f}, r={inversion['higher_curvature_recovery_partial']:.1f}%)"
            )

    group_summary = summary["group_summary"]
    if group_summary["mean_zero_curvature_recovery_partial"] is not None:
        print(
            f"  Zero-curvature mean partial recovery: {group_summary['mean_zero_curvature_recovery_partial']:.1f}%"
        )
        print(
            f"  Nonzero-curvature mean partial recovery: {group_summary['mean_nonzero_curvature_recovery_partial']:.1f}%"
        )
    print(f"  Mean penalty recovery across activations: {summary['mean_recovery_penalty']:.1f}%")
    print(f"  Mean partial recovery across activations: {summary['mean_recovery_partial']:.1f}%")

    print()
    print("=" * 90)
    print("Hypothesis tests")
    print("=" * 90)
    for test in results["tests"]:
        status = "PASS" if test["passed"] else "FAIL"
        print(f"  [{status}] {test['description']}  ({test['metric_display']})")

    print()
    print("=" * 90)
    print(f"OVERALL: {summary['overall_verdict']}")
    print(summary["overall_message"])
    print(
        "Interpretation: this run can support, at most, a partial-blend surrogate trend in this toy setting; "
        "it does not by itself establish reliable recovery by the static weight penalty."
    )
    print(f"Runtime: {summary['runtime_sec']:.2f} seconds")
    print("=" * 90)



def main():
    run_experiment(verbose=True)


if __name__ == "__main__":
    main()
