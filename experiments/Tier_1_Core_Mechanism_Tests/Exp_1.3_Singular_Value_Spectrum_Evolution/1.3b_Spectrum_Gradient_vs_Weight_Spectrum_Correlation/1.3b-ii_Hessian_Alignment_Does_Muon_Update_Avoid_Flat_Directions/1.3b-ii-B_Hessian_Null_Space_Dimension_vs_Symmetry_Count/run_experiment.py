#!/usr/bin/env python3
"""
Experiment 1.3b-ii-B -- Thresholded Near-Zero Hessian Counts vs Linear Gauge-Dimension Prediction

This first completion pass preserves the original toy finite-difference Hessian
experiment while narrowing the claims to what the code actually measures.

Primary observable
------------------
For each configuration, compute the full Hessian of the loss with respect to all
weights, diagonalize it, and count Hessian eigenvalues satisfying

    |lambda| < tau * max_abs_eigenvalue

for relative thresholds tau in {1e-2, 1e-3, 1e-4}.

Primary comparison
------------------
For selected deep linear networks at constructed exact minima, compare these
thresholded near-zero counts against the linear gauge-dimension prediction

    d^2 * (L - 1).

Controls and caveats
--------------------
- Randomly initialized linear networks are included as a control.
- ReLU runs are exploratory only: the code does not compute a nonlinear
  symmetry dimension and does not prove that every near-zero Hessian mode is a
  gauge direction.
- No Muon update analysis is performed here.
"""

from __future__ import annotations

import argparse
import numpy as np
import time

DEFAULT_NULL_THRESHOLDS = (1e-2, 1e-3, 1e-4)
DEFAULT_LINEAR_MIN_CONFIGS = ((4, 2), (4, 3), (4, 4), (6, 3))
DEFAULT_LINEAR_CONTROL_CONFIGS = ((4, 2), (4, 3))
DEFAULT_RELU_CONFIGS = ((4, 2), (4, 3))
DEFAULT_N_SAMPLES = 20
DEFAULT_MODEL_SEED = 42
DEFAULT_DATA_SEED_OFFSET = 100
DEFAULT_GRAD_EPS = 1e-6
DEFAULT_HESS_EPS = 1e-5

GROUP_TITLES = {
    "linear_exact_minimum": "Deep linear networks at constructed exact minima",
    "linear_random_control": "Deep linear networks at random initialization (control)",
    "relu_trained_exploratory": "ReLU networks trained toward low loss (exploratory)",
}


# ============================================================================
# Network forward pass and loss
# ============================================================================

def forward_linear(weights, x):
    """Forward pass: y = W_L ... W_2 W_1 x."""
    h = x.copy()
    for W in weights:
        h = W @ h
    return h



def forward_relu(weights, x):
    """Forward pass: y = W_L relu(... relu(W_1 x))."""
    h = x.copy()
    for i, W in enumerate(weights):
        h = W @ h
        if i < len(weights) - 1:
            h = np.maximum(0, h)
    return h



def loss_fn(weights, x, y_target, forward_fn):
    """MSE loss: 0.5 * ||forward(x) - y_target||^2, averaged over samples."""
    y_pred = forward_fn(weights, x)
    return 0.5 * np.mean(np.sum((y_pred - y_target) ** 2, axis=0))


# ============================================================================
# Flatten / unflatten parameters
# ============================================================================

def flatten_weights(weights):
    return np.concatenate([W.ravel() for W in weights])



def unflatten_weights(flat, shapes):
    weights = []
    idx = 0
    for shape in shapes:
        size = shape[0] * shape[1]
        weights.append(flat[idx:idx + size].reshape(shape))
        idx += size
    return weights


# ============================================================================
# Gradient via finite differences
# ============================================================================

def compute_gradient(weights, x, y_target, forward_fn, eps=DEFAULT_GRAD_EPS):
    shapes = [W.shape for W in weights]
    theta = flatten_weights(weights)
    n = len(theta)
    grad = np.zeros(n)
    for i in range(n):
        theta_p = theta.copy()
        theta_p[i] += eps
        fp = loss_fn(unflatten_weights(theta_p, shapes), x, y_target, forward_fn)
        theta_m = theta.copy()
        theta_m[i] -= eps
        fm = loss_fn(unflatten_weights(theta_m, shapes), x, y_target, forward_fn)
        grad[i] = (fp - fm) / (2 * eps)
    return grad


# ============================================================================
# Full Hessian via finite differences
# ============================================================================

def compute_hessian(weights, x, y_target, forward_fn, eps=DEFAULT_HESS_EPS):
    shapes = [W.shape for W in weights]
    theta = flatten_weights(weights)
    n = len(theta)
    H = np.zeros((n, n))

    f0 = loss_fn(weights, x, y_target, forward_fn)

    f_plus = np.zeros(n)
    f_minus = np.zeros(n)
    for i in range(n):
        theta_p = theta.copy()
        theta_p[i] += eps
        f_plus[i] = loss_fn(unflatten_weights(theta_p, shapes), x, y_target, forward_fn)
        theta_m = theta.copy()
        theta_m[i] -= eps
        f_minus[i] = loss_fn(unflatten_weights(theta_m, shapes), x, y_target, forward_fn)

    for i in range(n):
        H[i, i] = (f_plus[i] - 2 * f0 + f_minus[i]) / (eps ** 2)

    for i in range(n):
        for j in range(i + 1, n):
            theta_pp = theta.copy()
            theta_pp[i] += eps
            theta_pp[j] += eps
            f_pp = loss_fn(unflatten_weights(theta_pp, shapes), x, y_target, forward_fn)

            theta_pm = theta.copy()
            theta_pm[i] += eps
            theta_pm[j] -= eps
            f_pm = loss_fn(unflatten_weights(theta_pm, shapes), x, y_target, forward_fn)

            theta_mp = theta.copy()
            theta_mp[i] -= eps
            theta_mp[j] += eps
            f_mp = loss_fn(unflatten_weights(theta_mp, shapes), x, y_target, forward_fn)

            theta_mm = theta.copy()
            theta_mm[i] -= eps
            theta_mm[j] -= eps
            f_mm = loss_fn(unflatten_weights(theta_mm, shapes), x, y_target, forward_fn)

            H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps ** 2)
            H[j, i] = H[i, j]

    return H


# ============================================================================
# Construct exact minimum for deep linear network
# ============================================================================

def construct_linear_minimum(d, L, W_star, scramble=True, seed=99):
    """
    Construct an exact minimum: product W_L...W_1 = W_star.

    Strategy: Start with W_1 = W_star, W_k = I for k > 1.
    If scramble=True, apply random gauge transformations:
      W_l -> W_l G_l^{-1}, W_{l-1} -> G_l W_{l-1}
    This gives a non-trivial factorization that is still an exact minimum.
    """
    weights = [np.eye(d) for _ in range(L)]
    weights[0] = W_star.copy()

    if scramble and L > 1:
        rng = np.random.RandomState(seed)
        for l in range(1, L):
            G = rng.randn(d, d) * 0.5
            G += np.eye(d)
            G_inv = np.linalg.inv(G)
            weights[l] = weights[l] @ G_inv
            weights[l - 1] = G @ weights[l - 1]

    product = np.eye(d)
    for W in weights:
        product = W @ product
    recon_err = np.linalg.norm(product - W_star) / (np.linalg.norm(W_star) + 1e-15)

    return weights, recon_err



def train_to_minimum_gd(weights, x, y_target, forward_fn, lr=0.01,
                        n_steps=10000, tol=1e-12, verbose=False,
                        grad_eps=DEFAULT_GRAD_EPS):
    """Train network to a low-loss endpoint using finite-difference gradient descent."""
    shapes = [W.shape for W in weights]
    theta = flatten_weights(weights)
    stop_reason = "max_steps_reached"
    steps_completed = 0

    for step in range(n_steps):
        w = unflatten_weights(theta, shapes)
        current_loss = loss_fn(w, x, y_target, forward_fn)

        if current_loss < tol:
            stop_reason = "loss_below_tol"
            if verbose:
                print(f"    Converged at step {step}, loss={current_loss:.2e}")
            steps_completed = step
            break

        grad = compute_gradient(w, x, y_target, forward_fn, eps=grad_eps)
        grad_norm = np.linalg.norm(grad)

        if verbose and step % 1000 == 0:
            print(f"    Step {step}: loss={current_loss:.2e}, |grad|={grad_norm:.2e}")

        theta -= lr * grad
        steps_completed = step + 1
    else:
        step = n_steps - 1

    final_weights = unflatten_weights(theta, shapes)
    final_loss = loss_fn(final_weights, x, y_target, forward_fn)
    final_grad = compute_gradient(final_weights, x, y_target, forward_fn, eps=grad_eps)

    return final_weights, {
        "final_loss": float(final_loss),
        "final_grad_norm": float(np.linalg.norm(final_grad)),
        "stop_reason": stop_reason,
        "steps_completed": int(steps_completed),
        "last_step_index": int(step),
        "lr": float(lr),
        "tol": float(tol),
        "n_steps": int(n_steps),
        "grad_eps": float(grad_eps),
    }


# ============================================================================
# Experiment planning helpers
# ============================================================================

def dataset_seed_for_config(d, L, data_seed_offset=DEFAULT_DATA_SEED_OFFSET):
    return int(data_seed_offset + d * 10 + L)



def get_default_experiment_plan(relu_verbose=False):
    """Return the default experiment plan, matching the original configuration set."""
    return {
        "plan_name": "default",
        "linear_min_configs": [tuple(cfg) for cfg in DEFAULT_LINEAR_MIN_CONFIGS],
        "linear_control_configs": [tuple(cfg) for cfg in DEFAULT_LINEAR_CONTROL_CONFIGS],
        "relu_configs": [tuple(cfg) for cfg in DEFAULT_RELU_CONFIGS],
        "n_samples": int(DEFAULT_N_SAMPLES),
        "model_seed": int(DEFAULT_MODEL_SEED),
        "data_seed_offset": int(DEFAULT_DATA_SEED_OFFSET),
        "grad_eps": float(DEFAULT_GRAD_EPS),
        "hess_eps": float(DEFAULT_HESS_EPS),
        "null_thresholds": tuple(float(t) for t in DEFAULT_NULL_THRESHOLDS),
        "relu_train_kwargs": {
            "lr": 0.005,
            "n_steps": 10000,
            "tol": 1e-12,
            "verbose": bool(relu_verbose),
        },
    }



def get_smoke_experiment_plan(relu_verbose=False):
    """A reduced plan for quick verification of script and notebook code paths."""
    return {
        "plan_name": "smoke",
        "linear_min_configs": [(4, 2)],
        "linear_control_configs": [(4, 2)],
        "relu_configs": [],
        "n_samples": int(DEFAULT_N_SAMPLES),
        "model_seed": int(DEFAULT_MODEL_SEED),
        "data_seed_offset": int(DEFAULT_DATA_SEED_OFFSET),
        "grad_eps": float(DEFAULT_GRAD_EPS),
        "hess_eps": float(DEFAULT_HESS_EPS),
        "null_thresholds": tuple(float(t) for t in DEFAULT_NULL_THRESHOLDS),
        "relu_train_kwargs": {
            "lr": 0.005,
            "n_steps": 2000,
            "tol": 1e-12,
            "verbose": bool(relu_verbose),
        },
    }



def make_dataset(d, L, n_samples, data_seed):
    """Construct the synthetic linear target dataset used throughout the experiment."""
    rng = np.random.RandomState(data_seed)
    x = rng.randn(d, n_samples)
    W_target = rng.randn(d, d) * 0.5
    y_target = W_target @ x
    return {
        "x": x,
        "y_target": y_target,
        "W_target": W_target,
        "data_seed": int(data_seed),
        "n_samples": int(n_samples),
        "config": (int(d), int(L)),
    }



def build_experiment_tasks(plan):
    tasks = []
    model_seed = int(plan["model_seed"])
    data_seed_offset = int(plan["data_seed_offset"])

    for d, L in plan["linear_min_configs"]:
        tasks.append({
            "group": "linear_exact_minimum",
            "condition": "exact_minimum",
            "label": f"LINEAR@exact-min d={d},L={L}",
            "d": int(d),
            "L": int(L),
            "net_type": "linear",
            "forward_fn": forward_linear,
            "at_minimum": True,
            "data_seed": dataset_seed_for_config(d, L, data_seed_offset),
            "model_seed": model_seed,
        })

    for d, L in plan["linear_control_configs"]:
        tasks.append({
            "group": "linear_random_control",
            "condition": "random_init",
            "label": f"LINEAR@rand d={d},L={L}",
            "d": int(d),
            "L": int(L),
            "net_type": "linear",
            "forward_fn": forward_linear,
            "at_minimum": False,
            "data_seed": dataset_seed_for_config(d, L, data_seed_offset),
            "model_seed": model_seed,
        })

    for d, L in plan["relu_configs"]:
        tasks.append({
            "group": "relu_trained_exploratory",
            "condition": "trained_endpoint",
            "label": f"RELU@trained d={d},L={L}",
            "d": int(d),
            "L": int(L),
            "net_type": "relu",
            "forward_fn": forward_relu,
            "at_minimum": True,
            "data_seed": dataset_seed_for_config(d, L, data_seed_offset),
            "model_seed": model_seed,
        })

    return tasks


# ============================================================================
# Reporting helpers
# ============================================================================

def threshold_to_display(threshold):
    return f"{threshold * 100:g}%"



def threshold_to_key(threshold):
    return threshold_to_display(threshold).replace("%", "pct").replace(".", "p")



def format_float(value, precision=3):
    if value is None:
        return "n/a"
    return f"{value:.{precision}e}"



def select_results(results, group=None, net_type=None, condition=None):
    selected = list(results)
    if group is not None:
        selected = [r for r in selected if r["group"] == group]
    if net_type is not None:
        selected = [r for r in selected if r["net_type"] == net_type]
    if condition is not None:
        selected = [r for r in selected if r["condition"] == condition]
    return selected



def make_summary_rows(results):
    rows = []
    for r in results:
        row = {
            "group": r["group"],
            "label": r["label"],
            "d": r["d"],
            "L": r["L"],
            "net_type": r["net_type"],
            "condition": r["condition"],
            "total_params": r["total_params"],
            "predicted_gauge_dim": r["predicted_gauge_dim"],
            "loss_value": r["loss_value"],
            "grad_norm": r["grad_norm"],
            "reconstruction_error": r["reconstruction_error"],
            "n_negative": r["n_negative"],
            "hessian_asymmetry": r["hessian_asymmetry"],
            "max_abs_eig": r["max_abs_eig"],
            "elapsed_s": r["elapsed"],
        }
        for threshold in r["null_thresholds"]:
            key = threshold_to_key(threshold)
            row[f"null_count_{key}"] = r["null_counts"][threshold]
            row[f"null_ratio_{key}"] = r["null_ratios_to_prediction"][threshold]
            row[f"abs_eps_{key}"] = r["threshold_epsilons"][threshold]
        rows.append(row)
    return rows



def print_results_table(results, title=""):
    if not results:
        return

    thresholds = results[0]["null_thresholds"]
    if title:
        print(f"\n{'=' * 150}")
        print(title)
        print("=" * 150)

    header = [
        f"{'Label':<28}",
        f"{'Pred':>6}",
    ]
    for threshold in thresholds:
        header.append(f"{threshold_to_display(threshold):>8}")
    header.extend([
        f"{'|grad|':>11}",
        f"{'neg':>5}",
        f"{'asym':>11}",
        f"{'elapsed':>10}",
    ])
    print(" ".join(header))
    print("-" * 150)

    for r in results:
        row = [
            f"{r['label']:<28}",
            f"{r['predicted_gauge_dim']:>6}",
        ]
        for threshold in thresholds:
            row.append(f"{r['null_counts'][threshold]:>8}")
        row.extend([
            f"{r['grad_norm']:>11.3e}",
            f"{r['n_negative']:>5}",
            f"{r['hessian_asymmetry']:>11.3e}",
            f"{r['elapsed']:>10.2f}s",
        ])
        print(" ".join(row))



def print_eigenvalue_details(result):
    """Print eigenvalue diagnostics for a single result."""
    eigs = result["eigenvalues_abs_sorted"]
    max_eig = result["max_abs_eig"]
    pred = result["predicted_gauge_dim"]
    n = len(eigs)

    print(f"\n  --- {result['label']} ---")
    print(f"  Max |eigenvalue|: {max_eig:.8f}")
    print(f"  Min |eigenvalue|: {eigs[0]:.2e}")
    print(f"  Gradient norm:    {result['grad_norm']:.3e}")
    print(f"  Negative eigs:    {result['n_negative']}")
    print(f"  Hessian asymmetry:{result['hessian_asymmetry']:.3e}")

    if n <= 70:
        print("  Full spectrum (sorted by |value|):")
        for idx in range(n):
            marker = ""
            if idx == pred:
                marker = " <<<< predicted linear gauge boundary"
            ratio_to_max = eigs[idx] / max_eig if max_eig > 0 else 0.0
            print(f"    [{idx:3d}] {eigs[idx]:15.8e} ({ratio_to_max:.4e}){marker}")
    else:
        print("  Smallest 15:")
        for idx in range(min(15, n)):
            print(f"    [{idx:3d}] {eigs[idx]:15.8e}")
        if pred > 0 and pred < n:
            lo = max(0, pred - 5)
            hi = min(n, pred + 5)
            print(f"  Around predicted boundary (idx {pred}):")
            for idx in range(lo, hi):
                marker = " <<<< predicted" if idx == pred else ""
                print(f"    [{idx:3d}] {eigs[idx]:15.8e}{marker}")
        print("  Largest 5:")
        for idx in range(max(0, n - 5), n):
            print(f"    [{idx:3d}] {eigs[idx]:15.8e}")

    if 0 < pred < n and eigs[pred - 1] > 0:
        gap = eigs[pred] / eigs[pred - 1]
        print(f"  Boundary gap eig[{pred}]/eig[{pred - 1}] = {gap:.1f}x")



def print_interpretive_summary(bundle):
    """Print an evidence-tied summary without overclaiming."""
    results = bundle["results"]
    thresholds = bundle["plan"]["null_thresholds"]
    primary_threshold = thresholds[0]

    linear_min = select_results(results, group="linear_exact_minimum")
    linear_rand = select_results(results, group="linear_random_control")
    relu_runs = select_results(results, group="relu_trained_exploratory")

    print(f"\n{'=' * 100}")
    print("INTERPRETIVE SUMMARY")
    print("=" * 100)
    print(
        "Primary observable: thresholded counts of near-zero Hessian eigenvalues, "
        f"using relative thresholds {', '.join(threshold_to_display(t) for t in thresholds)}."
    )
    print(
        "Primary theoretical comparison: deep linear exact-minimum cases vs the "
        "linear gauge-dimension prediction d^2 (L - 1)."
    )

    if linear_min:
        print("\nLinear exact-minimum cases (primary test):")
        match_like = 0
        for r in linear_min:
            ratio = r["null_ratios_to_prediction"][primary_threshold]
            verdict = "consistent" if 0.8 <= ratio <= 1.2 else "threshold-sensitive"
            if verdict == "consistent":
                match_like += 1
            print(
                f"  {r['label']}: predicted={r['predicted_gauge_dim']}, "
                f"observed@{threshold_to_display(primary_threshold)}={r['null_counts'][primary_threshold]}, "
                f"ratio={ratio:.3f}, verdict={verdict}"
            )
        print(
            f"  -> {match_like}/{len(linear_min)} exact-minimum linear runs are within a "
            "0.8-1.2 observed/predicted ratio band at the loosest threshold."
        )

    if linear_rand:
        print("\nLinear random-initialization controls:")
        for r in linear_rand:
            ratio = r["null_ratios_to_prediction"][primary_threshold]
            print(
                f"  {r['label']}: predicted={r['predicted_gauge_dim']}, "
                f"observed@{threshold_to_display(primary_threshold)}={r['null_counts'][primary_threshold]}, "
                f"ratio={ratio:.3f}"
            )

    if relu_runs:
        print("\nExploratory ReLU-trained runs:")
        for rr in relu_runs:
            matched_linear = next(
                (
                    r for r in linear_min
                    if r["d"] == rr["d"] and r["L"] == rr["L"]
                ),
                None,
            )
            comparison = ""
            if matched_linear is not None:
                comparison = (
                    f", linear exact-minimum observed@{threshold_to_display(primary_threshold)}="
                    f"{matched_linear['null_counts'][primary_threshold]}"
                )
            print(
                f"  {rr['label']}: loss={rr['loss_value']:.3e}, |grad|={rr['grad_norm']:.3e}, "
                f"observed@{threshold_to_display(primary_threshold)}={rr['null_counts'][primary_threshold]}"
                f"{comparison}"
            )
        print("  -> These runs are exploratory only; no nonlinear symmetry dimension is computed.")

    print("\nEstablished in this script's actual computation:")
    print("  - The measured quantity is a thresholded near-zero Hessian count, not an exact symmetry count.")
    print("  - Constructed deep linear minima are the main test bed; random linear initializations are controls.")
    print("  - ReLU results, if any, should be interpreted cautiously and only as exploratory comparisons.")

    print("\nUntested in this experiment:")
    print("  - direct construction of gauge tangent vectors and overlap with the null eigenspace")
    print("  - finite-difference step-size robustness sweeps")
    print("  - Muon-vs-SGD update alignment with Hessian flat directions")



def print_bundle_report(bundle):
    results = bundle["results"]
    groups = bundle["groups"]

    print("=" * 100)
    print(bundle["title"])
    print("=" * 100)
    print(bundle["scope"])
    print(
        f"\nPlan '{bundle['plan']['plan_name']}' | n_samples={bundle['plan']['n_samples']} | "
        f"grad_eps={bundle['plan']['grad_eps']} | hess_eps={bundle['plan']['hess_eps']}"
    )
    print(
        "Thresholds: "
        + ", ".join(threshold_to_display(t) for t in bundle["plan"]["null_thresholds"])
    )

    for group_name in [
        "linear_exact_minimum",
        "linear_random_control",
        "relu_trained_exploratory",
    ]:
        group_results = groups[group_name]
        if not group_results:
            continue
        print_results_table(group_results, GROUP_TITLES[group_name])
        print("\n  EIGENVALUE DETAILS:")
        for result in group_results:
            print_eigenvalue_details(result)

    print_interpretive_summary(bundle)


# ============================================================================
# Single-run and full-run execution
# ============================================================================

def run_experiment(d, L, forward_fn, net_type, x, y_target, seed=42,
                   at_minimum=True, label="", group="", condition="",
                   grad_eps=DEFAULT_GRAD_EPS, hess_eps=DEFAULT_HESS_EPS,
                   thresholds=DEFAULT_NULL_THRESHOLDS,
                   relu_train_kwargs=None, data_seed=None,
                   print_progress=True):
    """Run Hessian near-zero-count analysis for a single configuration."""
    total_params = d * d * L
    predicted_gauge_dim = d * d * (L - 1)
    thresholds = tuple(float(t) for t in thresholds)
    relu_train_kwargs = dict(relu_train_kwargs or {})
    warnings = []

    loss_value = None
    grad_norm = None
    reconstruction_error = None
    xtx_condition_number = None
    training_info = None

    if print_progress:
        print(f"\n{'=' * 88}")
        print(
            f"{label} | group={group} | params={total_params} | "
            f"predicted linear gauge dim={predicted_gauge_dim}"
        )
        print(f"{'=' * 88}")

    if net_type == "linear" and at_minimum:
        xtx = x @ x.T
        xtx_condition_number = float(np.linalg.cond(xtx))
        if xtx_condition_number > 1e8:
            warnings.append(
                f"x @ x.T is ill-conditioned (cond={xtx_condition_number:.3e}); W_star recovery may be unstable."
            )
        W_star = y_target @ x.T @ np.linalg.inv(xtx)
        weights, reconstruction_error = construct_linear_minimum(
            d, L, W_star, scramble=True, seed=seed
        )
        loss_value = float(loss_fn(weights, x, y_target, forward_fn))
        grad_norm = float(
            np.linalg.norm(compute_gradient(weights, x, y_target, forward_fn, eps=grad_eps))
        )
        if print_progress:
            print(
                f"    Constructed exact minimum: recon_err={reconstruction_error:.2e}, "
                f"loss={loss_value:.2e}, |grad|={grad_norm:.2e}"
            )
    elif net_type == "relu" and at_minimum:
        rng = np.random.RandomState(seed)
        weights = [np.eye(d) * 0.5 + rng.randn(d, d) * 0.1 for _ in range(L)]
        if print_progress:
            print("    Training ReLU network toward low loss...", flush=True)
        weights, training_info = train_to_minimum_gd(
            weights,
            x,
            y_target,
            forward_fn,
            grad_eps=grad_eps,
            **relu_train_kwargs,
        )
        loss_value = float(training_info["final_loss"])
        grad_norm = float(training_info["final_grad_norm"])
        if print_progress:
            print(
                f"    Final trained endpoint: loss={loss_value:.2e}, |grad|={grad_norm:.2e}, "
                f"stop_reason={training_info['stop_reason']}"
            )
    else:
        rng = np.random.RandomState(seed)
        weights = [rng.randn(d, d) * 0.5 / np.sqrt(d) for _ in range(L)]
        loss_value = float(loss_fn(weights, x, y_target, forward_fn))
        grad_norm = float(
            np.linalg.norm(compute_gradient(weights, x, y_target, forward_fn, eps=grad_eps))
        )
        if print_progress:
            print(f"    Random init control: loss={loss_value:.2e}, |grad|={grad_norm:.2e}")

    if print_progress:
        print(f"    Computing Hessian ({total_params}x{total_params})...", end=" ", flush=True)
    t0 = time.time()
    H_raw = compute_hessian(weights, x, y_target, forward_fn, eps=hess_eps)
    elapsed = time.time() - t0
    if print_progress:
        print(f"done in {elapsed:.1f}s")

    hessian_asymmetry = float(
        np.linalg.norm(H_raw - H_raw.T) / (np.linalg.norm(H_raw) + 1e-15)
    )
    H = 0.5 * (H_raw + H_raw.T)
    eigenvalues = np.linalg.eigvalsh(H)
    eigenvalues_abs_sorted = np.sort(np.abs(eigenvalues))

    max_abs_eig = float(np.max(np.abs(eigenvalues)))
    if max_abs_eig == 0.0:
        max_abs_eig = 1.0

    n_negative = int(np.sum(eigenvalues < 0.0))
    null_counts = {}
    null_ratios_to_prediction = {}
    threshold_epsilons = {}
    for threshold in thresholds:
        abs_eps = float(threshold * max_abs_eig)
        count = int(np.sum(np.abs(eigenvalues) < abs_eps))
        null_counts[threshold] = count
        null_ratios_to_prediction[threshold] = (
            count / predicted_gauge_dim if predicted_gauge_dim > 0 else 0.0
        )
        threshold_epsilons[threshold] = abs_eps

    if at_minimum and grad_norm is not None and grad_norm > 1e-5:
        warnings.append(
            f"Gradient norm at claimed minimum/endpoint is not very small (|grad|={grad_norm:.3e})."
        )
    if reconstruction_error is not None and reconstruction_error > 1e-10:
        warnings.append(
            f"Linear factorization reconstruction error is not tiny (recon_err={reconstruction_error:.3e})."
        )
    if at_minimum and n_negative > 0:
        warnings.append(
            f"Hessian has {n_negative} negative eigenvalues at the analyzed endpoint."
        )
    if null_counts:
        spread = max(null_counts.values()) - min(null_counts.values())
        if predicted_gauge_dim > 0 and spread > max(2, int(0.25 * predicted_gauge_dim)):
            warnings.append(
                "Thresholded null count varies substantially across thresholds; interpretation is threshold-sensitive."
            )

    return {
        "d": int(d),
        "L": int(L),
        "net_type": net_type,
        "forward_name": forward_fn.__name__,
        "group": group,
        "condition": condition,
        "label": label,
        "at_minimum": bool(at_minimum),
        "data_seed": data_seed,
        "model_seed": int(seed),
        "n_samples": int(x.shape[1]),
        "total_params": int(total_params),
        "predicted_gauge_dim": int(predicted_gauge_dim),
        "loss_value": loss_value,
        "grad_norm": grad_norm,
        "reconstruction_error": reconstruction_error,
        "xtx_condition_number": xtx_condition_number,
        "training_info": training_info,
        "hessian_asymmetry": hessian_asymmetry,
        "eigenvalues": eigenvalues,
        "eigenvalues_abs_sorted": eigenvalues_abs_sorted,
        "min_eigenvalue": float(np.min(eigenvalues)),
        "max_eigenvalue": float(np.max(eigenvalues)),
        "max_abs_eig": max_abs_eig,
        "n_negative": n_negative,
        "null_thresholds": thresholds,
        "null_counts": null_counts,
        "null_ratios_to_prediction": null_ratios_to_prediction,
        "threshold_epsilons": threshold_epsilons,
        "elapsed": float(elapsed),
        "warnings": warnings,
    }



def run_full_experiment(plan=None, print_progress=True):
    """Run the full configured experiment and return a notebook-friendly result bundle."""
    if plan is None:
        plan = get_default_experiment_plan()

    plan = dict(plan)
    plan["linear_min_configs"] = [tuple(cfg) for cfg in plan.get("linear_min_configs", [])]
    plan["linear_control_configs"] = [tuple(cfg) for cfg in plan.get("linear_control_configs", [])]
    plan["relu_configs"] = [tuple(cfg) for cfg in plan.get("relu_configs", [])]
    plan["null_thresholds"] = tuple(float(t) for t in plan.get("null_thresholds", DEFAULT_NULL_THRESHOLDS))
    plan["relu_train_kwargs"] = dict(plan.get("relu_train_kwargs", {}))

    tasks = build_experiment_tasks(plan)
    results = []

    last_group = None
    for task in tasks:
        if print_progress and task["group"] != last_group:
            print("\n" + "#" * 100)
            print(f"# {GROUP_TITLES[task['group']]}")
            print("#" * 100)
            last_group = task["group"]

        dataset = make_dataset(
            task["d"], task["L"], plan["n_samples"], task["data_seed"]
        )
        result = run_experiment(
            task["d"],
            task["L"],
            task["forward_fn"],
            task["net_type"],
            dataset["x"],
            dataset["y_target"],
            seed=task["model_seed"],
            at_minimum=task["at_minimum"],
            label=task["label"],
            group=task["group"],
            condition=task["condition"],
            grad_eps=plan["grad_eps"],
            hess_eps=plan["hess_eps"],
            thresholds=plan["null_thresholds"],
            relu_train_kwargs=plan["relu_train_kwargs"],
            data_seed=task["data_seed"],
            print_progress=print_progress,
        )
        results.append(result)

    groups = {
        "linear_exact_minimum": select_results(results, group="linear_exact_minimum"),
        "linear_random_control": select_results(results, group="linear_random_control"),
        "relu_trained_exploratory": select_results(results, group="relu_trained_exploratory"),
    }

    return {
        "experiment_id": "1.3b-ii-B",
        "title": "Experiment 1.3b-ii-B: Thresholded Near-Zero Hessian Counts vs Linear Gauge-Dimension Prediction",
        "scope": (
            "Toy finite-difference Hessian study. Linear exact-minimum cases are the main comparison; "
            "random linear initializations are controls; ReLU runs are exploratory only."
        ),
        "plan": plan,
        "tasks": tasks,
        "results": results,
        "groups": groups,
        "summary_rows": make_summary_rows(results),
    }



def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run Experiment 1.3b-ii-B: compare thresholded near-zero Hessian counts "
            "to the linear gauge-dimension prediction in selected toy cases."
        )
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a reduced configuration set for quick verification.",
    )
    parser.add_argument(
        "--no-relu",
        action="store_true",
        help="Skip the exploratory ReLU runs.",
    )
    parser.add_argument(
        "--quiet-progress",
        action="store_true",
        help="Suppress per-configuration progress messages during execution.",
    )
    parser.add_argument(
        "--relu-verbose",
        action="store_true",
        help="Print intermediate ReLU training progress every 1000 steps.",
    )
    args = parser.parse_args(argv)

    plan = (
        get_smoke_experiment_plan(relu_verbose=args.relu_verbose)
        if args.smoke
        else get_default_experiment_plan(relu_verbose=args.relu_verbose)
    )
    if args.no_relu:
        plan["relu_configs"] = []

    bundle = run_full_experiment(plan=plan, print_progress=not args.quiet_progress)
    print_bundle_report(bundle)
    return bundle


if __name__ == "__main__":
    main()
