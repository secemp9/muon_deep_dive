#!/usr/bin/env python3
"""
Experiment 2.15: Toy tanh gradient-transport probe under orthogonality constraints
=================================================================================

This script runs a deterministic, single-seed, full-batch tanh regression toy
study across depths {2, 4, 6, 8}. It compares:
  - SGD
  - Muon-style orthogonalized gradient steps
  - SGD + soft orthogonality penalty (lambda=0.003)
  - SGD + strong orthogonality penalty (lambda=1.0)
  - SGD + hard orthogonal projection after every step

What is measured after 200 training steps:
  - per-layer gradient Frobenius norms
  - per-layer sigma_max(W)
  - per-layer mean |tanh'(z)|
  - fitted alpha from log-gradient-norm vs distance-from-output
  - proxy per-layer multiplier sigma_max(W) * mean|tanh'(z)|

Scope and caveats:
  - single random batch, single seed, no held-out data, no multiseed aggregation
  - measurements are taken after a fixed number of steps, not at verified convergence
  - sigma_max(W) * mean|tanh'(z)| is a heuristic diagnostic proxy, not a full
    Jacobian transport or exact product-of-gradients measurement
  - this script can probe whether orthogonality constraints reduce the proxy below 1,
    but it does not by itself establish severe exponential vanishing such as alpha < 0.7
"""

from pathlib import Path
import time

import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

WIDTH = 32
DEPTHS = [2, 4, 6, 8]
NUM_STEPS = 200
LR_SGD = 0.01
LR_MUON = 0.02
ORTHO_LAMBDA = 0.003
ORTHO_LAMBDA_STRONG = 1.0
NS_ITERS = 5
BATCH_SIZE = 64
INPUT_DIM = 32
OUTPUT_DIM = 32
SEED = 42

METHODS = [
    "SGD",
    "Muon",
    "SGD+OrthoPen(0.003)",
    "SGD+OrthoPen(1.0)",
    "SGD+HardOrtho",
]

PAIR_RELATIVE_DIR = Path(
    "experiments/Tier_1_Core_Mechanism_Tests/TANH_VANISHING_ortho_penalty"
)


# =============================================================================
# CONFIG HELPERS
# =============================================================================

def get_default_config():
    """Return the default experiment configuration."""
    return {
        "width": WIDTH,
        "depths": list(DEPTHS),
        "num_steps": NUM_STEPS,
        "lr_sgd": LR_SGD,
        "lr_muon": LR_MUON,
        "ortho_lambda": ORTHO_LAMBDA,
        "ortho_lambda_strong": ORTHO_LAMBDA_STRONG,
        "ns_iters": NS_ITERS,
        "batch_size": BATCH_SIZE,
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "seed": SEED,
        "methods": list(METHODS),
        "loss_definition": "0.5 * sum(diff**2) / batch_size",
    }


# =============================================================================
# NETWORK UTILITIES
# =============================================================================

def init_weights(num_layers, width, seed):
    """Initialize tanh net weights with Xavier-style Gaussian init."""
    rng = np.random.RandomState(seed)
    weights = []
    for _ in range(num_layers):
        fan_in = width
        fan_out = width
        std = np.sqrt(2.0 / (fan_in + fan_out))
        weights.append((rng.randn(width, width) * std).copy())
    return weights



def forward_tanh(weights, X):
    """Forward pass through tanh net. Returns activations and pre-activations."""
    activations = [X.copy()]
    pre_activations = []
    out = X.copy()
    for W in weights:
        z = W @ out
        pre_activations.append(z)
        out = np.tanh(z)
        activations.append(out)
    return activations, pre_activations



def compute_loss(weights, X, Y_target):
    """Sample-averaged half squared error.

    This scaling matches the manual backprop in ``compute_gradients``, which divides
    the output residual by batch size but not by output dimension.
    """
    activations, _ = forward_tanh(weights, X)
    diff = activations[-1] - Y_target
    batch_size = X.shape[1]
    return 0.5 * np.sum(diff ** 2) / batch_size



def compute_gradients(weights, X, Y_target):
    """Backprop through tanh net. Returns per-layer gradients."""
    num_layers = len(weights)
    batch_size = X.shape[1]

    activations, _ = forward_tanh(weights, X)
    diff = activations[-1] - Y_target
    delta = diff / batch_size

    grads = [None] * num_layers
    for layer in range(num_layers - 1, -1, -1):
        tanh_deriv = 1.0 - activations[layer + 1] ** 2
        delta_z = delta * tanh_deriv
        grads[layer] = delta_z @ activations[layer].T
        if layer > 0:
            delta = weights[layer].T @ delta_z

    return grads



def ortho_penalty_gradient(W):
    """Gradient of ||W^T W - I||_F^2 with respect to W."""
    WtW = W.T @ W
    I = np.eye(W.shape[0])
    return 4.0 * W @ (WtW - I)



def project_to_orthogonal(W):
    """Project W onto the nearest orthogonal matrix via SVD: U @ V^T."""
    U, _, Vt = np.linalg.svd(W, full_matrices=False)
    return U @ Vt



def newton_schulz_orthogonalize(G, num_iters=5):
    """Newton-Schulz iteration to find an orthogonalized version of G."""
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

def train_sgd(weights, X, Y, num_steps, lr):
    """Train with plain SGD."""
    weights = [W.copy() for W in weights]
    for _ in range(num_steps):
        grads = compute_gradients(weights, X, Y)
        for i in range(len(weights)):
            weights[i] -= lr * grads[i]
    return weights



def train_muon(weights, X, Y, num_steps, lr, ns_iters=5):
    """Train with Muon-style orthogonalized gradient steps."""
    weights = [W.copy() for W in weights]
    for _ in range(num_steps):
        grads = compute_gradients(weights, X, Y)
        for i in range(len(weights)):
            G_orth = newton_schulz_orthogonalize(grads[i], ns_iters)
            weights[i] -= lr * G_orth
    return weights



def train_sgd_ortho_penalty(weights, X, Y, num_steps, lr, lam):
    """Train with SGD plus an orthogonality penalty on weights."""
    weights = [W.copy() for W in weights]
    for _ in range(num_steps):
        grads = compute_gradients(weights, X, Y)
        for i in range(len(weights)):
            penalty_grad = ortho_penalty_gradient(weights[i])
            weights[i] -= lr * (grads[i] + lam * penalty_grad)
    return weights



def train_sgd_hard_ortho(weights, X, Y, num_steps, lr):
    """Train with SGD followed by hard orthogonal projection every step."""
    weights = [W.copy() for W in weights]
    for _ in range(num_steps):
        grads = compute_gradients(weights, X, Y)
        for i in range(len(weights)):
            weights[i] -= lr * grads[i]
            weights[i] = project_to_orthogonal(weights[i])
    return weights


# =============================================================================
# MEASUREMENT
# =============================================================================

def measure_at_step(weights, X, Y):
    """Compute per-layer gradient norms, sigma_max values, and loss."""
    grads = compute_gradients(weights, X, Y)
    loss = compute_loss(weights, X, Y)

    grad_norms = []
    sigma_maxes = []
    for W, G in zip(weights, grads):
        grad_norms.append(float(np.linalg.norm(G, "fro")))
        sigma_maxes.append(float(np.linalg.svd(W, compute_uv=False)[0]))

    return grad_norms, sigma_maxes, float(loss)



def measure_mean_tanh_deriv(weights, X):
    """Measure mean |tanh'(z)| at each layer to track saturation."""
    activations, _ = forward_tanh(weights, X)
    mean_derivs = []
    for layer in range(len(weights)):
        tanh_deriv = 1.0 - activations[layer + 1] ** 2
        mean_derivs.append(float(np.mean(np.abs(tanh_deriv))))
    return mean_derivs



def fit_alpha(grad_norms):
    """Fit gradient_norm(layer_i) ~ alpha^(L - 1 - i).

    Layer 0 is deepest (furthest from the output).
    Layer L-1 is closest to the output.
    If alpha < 1, gradients shrink as layers get further from the output.
    """
    num_layers = len(grad_norms)
    if num_layers < 2:
        return 1.0

    valid = [(i, gn) for i, gn in enumerate(grad_norms) if gn > 1e-30]
    if len(valid) < 2:
        return 0.0

    x = np.array([num_layers - 1 - i for i, _ in valid], dtype=float)
    y = np.array([np.log(gn) for _, gn in valid], dtype=float)

    A = np.vstack([np.ones_like(x), x]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return float(np.exp(coeffs[1]))


# =============================================================================
# EXPERIMENT HELPERS
# =============================================================================

def make_dataset(config):
    """Create the fixed random regression batch used by every method/depth."""
    rng = np.random.RandomState(config["seed"])
    X = rng.randn(config["input_dim"], config["batch_size"]) * 0.5
    Y = rng.randn(config["output_dim"], config["batch_size"]) * 0.5
    return X, Y



def train_method(method, weights_init, X, Y, config):
    """Dispatch training for a named method."""
    if method == "SGD":
        return train_sgd(weights_init, X, Y, config["num_steps"], config["lr_sgd"])
    if method == "Muon":
        return train_muon(
            weights_init,
            X,
            Y,
            config["num_steps"],
            config["lr_muon"],
            config["ns_iters"],
        )
    if method == "SGD+OrthoPen(0.003)":
        return train_sgd_ortho_penalty(
            weights_init,
            X,
            Y,
            config["num_steps"],
            config["lr_sgd"],
            config["ortho_lambda"],
        )
    if method == "SGD+OrthoPen(1.0)":
        return train_sgd_ortho_penalty(
            weights_init,
            X,
            Y,
            config["num_steps"],
            config["lr_sgd"],
            config["ortho_lambda_strong"],
        )
    if method == "SGD+HardOrtho":
        return train_sgd_hard_ortho(weights_init, X, Y, config["num_steps"], config["lr_sgd"])
    raise ValueError(f"Unknown method: {method}")



def summarize_result(depth, method, weight_seed, grad_norms, sigma_maxes, mean_derivs, loss):
    """Build a structured result record for one depth/method pair."""
    proxy_multipliers = [float(s * d) for s, d in zip(sigma_maxes, mean_derivs)]
    mean_sigma_max = float(np.mean(sigma_maxes))
    mean_tanh_deriv = float(np.mean(mean_derivs))
    mean_proxy_multiplier = float(np.mean(proxy_multipliers))
    alpha = float(fit_alpha(grad_norms))
    ratio = float(grad_norms[0] / grad_norms[-1]) if grad_norms[-1] > 1e-30 else float("inf")

    return {
        "depth": int(depth),
        "method": method,
        "weight_init_seed": int(weight_seed),
        "alpha": alpha,
        "loss": float(loss),
        "ratio": ratio,
        "grad_norms": [float(x) for x in grad_norms],
        "sigma_maxes": [float(x) for x in sigma_maxes],
        "mean_tanh_derivs": [float(x) for x in mean_derivs],
        "proxy_multipliers": proxy_multipliers,
        "mean_sigma_max": mean_sigma_max,
        "mean_tanh_deriv": mean_tanh_deriv,
        "mean_proxy_multiplier": mean_proxy_multiplier,
        "proxy_all_below_1": bool(all(p < 1.0 for p in proxy_multipliers)),
        "proxy_all_above_1": bool(all(p > 1.0 for p in proxy_multipliers)),
        "layer_labels": [f"L{i}" for i in range(depth)],
    }



def compute_diagnostics(results_by_depth_method):
    """Compute narrow toy-scope checks tied to the implemented proxy metrics."""
    per_depth = []
    overall_pass = True

    for depth in sorted(results_by_depth_method):
        by_method = results_by_depth_method[depth]
        sgd = by_method["SGD"]
        muon = by_method["Muon"]
        hard = by_method["SGD+HardOrtho"]

        checks = {
            "hard_ortho_mean_proxy_lt_1": hard["mean_proxy_multiplier"] < 1.0,
            "sgd_mean_proxy_gt_1": sgd["mean_proxy_multiplier"] > 1.0,
            "muon_mean_proxy_gt_1": muon["mean_proxy_multiplier"] > 1.0,
            "hard_ortho_mean_sigma_near_1": abs(hard["mean_sigma_max"] - 1.0) < 0.01,
            "hard_ortho_alpha_lt_muon_alpha": hard["alpha"] < muon["alpha"],
        }
        depth_pass = all(checks.values())
        overall_pass = overall_pass and depth_pass

        row = {
            "depth": int(depth),
            **checks,
            "all_toy_scope_checks_passed": bool(depth_pass),
            "observed_mean_proxy_sgd": float(sgd["mean_proxy_multiplier"]),
            "observed_mean_proxy_muon": float(muon["mean_proxy_multiplier"]),
            "observed_mean_proxy_hard_ortho": float(hard["mean_proxy_multiplier"]),
            "observed_mean_sigma_sgd": float(sgd["mean_sigma_max"]),
            "observed_mean_sigma_muon": float(muon["mean_sigma_max"]),
            "observed_mean_sigma_hard_ortho": float(hard["mean_sigma_max"]),
            "observed_alpha_sgd": float(sgd["alpha"]),
            "observed_alpha_muon": float(muon["alpha"]),
            "observed_alpha_hard_ortho": float(hard["alpha"]),
        }
        per_depth.append(row)

    return {
        "per_depth": per_depth,
        "all_toy_scope_checks_passed": bool(overall_pass),
        "scope_note": (
            "These checks only summarize the implemented proxy diagnostics. They do not "
            "constitute proof of exponential vanishing or generalization beyond this toy run."
        ),
    }



def format_layer_values(values, precision):
    """Format per-layer values for console printing."""
    return "  ".join([f"L{i}={value:.{precision}f}" for i, value in enumerate(values)])



def print_report(experiment):
    """Pretty-print a calibrated console report."""
    config = experiment["config"]
    summary_rows = experiment["summary_rows"]
    results_by_depth_method = experiment["results_by_depth_method"]
    diagnostics = experiment["diagnostics"]

    print("=" * 100)
    print(experiment["title"])
    print("=" * 100)
    print("Scope: single-seed, single-batch, full-batch toy regression probe.")
    print("Proxy note: sigma_max(W_l) * mean|tanh'(z_l)| is a layerwise heuristic, not a full Jacobian metric.")
    print("            MeanProxy in the summary is the arithmetic mean of those layerwise proxy values.")
    print()
    print(
        f"Config: width={config['width']}, depths={config['depths']}, steps={config['num_steps']}, "
        f"batch={config['batch_size']}"
    )
    print(
        f"        lr_sgd={config['lr_sgd']}, lr_muon={config['lr_muon']}, "
        f"ortho_lambda={config['ortho_lambda']}, ortho_lambda_strong={config['ortho_lambda_strong']}"
    )
    print(
        f"        ns_iters={config['ns_iters']}, seed={config['seed']}, "
        f"loss={config['loss_definition']}"
    )
    print(f"Runtime: {experiment['runtime_seconds']:.2f}s")
    print()

    print("SUMMARY TABLE")
    print("-" * 100)
    header = (
        f"{'Depth':<6} {'Method':<22} {'Alpha':>7} {'Loss':>11} {'MeanSig':>9} "
        f"{'Mean|tanh|':>11} {'MeanProxy':>10} {'Ratio(L0/LL)':>12}"
    )
    print(header)
    print("-" * 100)
    for row in summary_rows:
        print(
            f"{row['depth']:<6} {row['method']:<22} {row['alpha']:>7.4f} {row['loss']:>11.6f} "
            f"{row['mean_sigma_max']:>9.4f} {row['mean_tanh_deriv']:>11.4f} "
            f"{row['mean_proxy_multiplier']:>10.4f} {row['ratio']:>12.4f}"
        )
    print("-" * 100)

    for depth in config["depths"]:
        print(f"\nDepth {depth} layerwise diagnostics")
        print("-" * 100)
        for method in config["methods"]:
            row = results_by_depth_method[depth][method]
            print(f"{method}:")
            print(f"  grad norms   : {format_layer_values(row['grad_norms'], 6)}")
            print(f"  sigma_max    : {format_layer_values(row['sigma_maxes'], 4)}")
            print(f"  mean|tanh'|  : {format_layer_values(row['mean_tanh_derivs'], 4)}")
            print(f"  proxy s*d    : {format_layer_values(row['proxy_multipliers'], 4)}")
            print(
                f"  alpha={row['alpha']:.4f}, loss={row['loss']:.6f}, "
                f"mean_proxy={row['mean_proxy_multiplier']:.4f}"
            )

    print("\nTOY-SCOPE DIAGNOSTIC CHECKS")
    print("-" * 100)
    for row in diagnostics["per_depth"]:
        print(f"Depth {row['depth']}:")
        print(
            f"  HardOrtho mean proxy < 1       : {row['hard_ortho_mean_proxy_lt_1']} "
            f"({row['observed_mean_proxy_hard_ortho']:.4f})"
        )
        print(
            f"  SGD mean proxy > 1             : {row['sgd_mean_proxy_gt_1']} "
            f"({row['observed_mean_proxy_sgd']:.4f})"
        )
        print(
            f"  Muon mean proxy > 1            : {row['muon_mean_proxy_gt_1']} "
            f"({row['observed_mean_proxy_muon']:.4f})"
        )
        print(
            f"  HardOrtho mean sigma ~= 1      : {row['hard_ortho_mean_sigma_near_1']} "
            f"({row['observed_mean_sigma_hard_ortho']:.4f})"
        )
        print(
            f"  HardOrtho alpha < Muon alpha   : {row['hard_ortho_alpha_lt_muon_alpha']} "
            f"({row['observed_alpha_hard_ortho']:.4f} < {row['observed_alpha_muon']:.4f})"
        )
        print(f"  All toy-scope checks passed    : {row['all_toy_scope_checks_passed']}")
    print("-" * 100)
    print(
        "Overall toy-scope proxy checks passed: "
        f"{diagnostics['all_toy_scope_checks_passed']}"
    )
    print(diagnostics["scope_note"])
    print()

    print("CALIBRATED CONCLUSION")
    print("-" * 100)
    print("1. Hard orthogonality and the strong penalty keep mean sigma_max near 1 and")
    print("   usually push the proxy sigma_max * mean|tanh'| below 1 in this toy setup.")
    print("2. SGD and Muon allow larger singular values, and their mean proxy values stay")
    print("   above 1 across the tested depths in this run.")
    print("3. The fitted alpha values remain much closer to 1 than the strongest original")
    print("   narrative predicted; this implementation does not show alpha < 0.7 for")
    print("   HardOrtho and should not be presented as a definitive demonstration of")
    print("   severe exponential vanishing.")
    print("4. Treat the result as a mechanistic toy probe whose main value is comparing")
    print("   proxy behavior under different orthogonality constraints, not as a complete")
    print("   gradient-transport theory validation.")
    print("-" * 100)


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment(verbose=True, config_overrides=None):
    """Run the experiment and return structured results for downstream analysis."""
    config = get_default_config()
    if config_overrides:
        config.update(config_overrides)
        config["depths"] = list(config["depths"])
        config["methods"] = list(config["methods"])

    start_time = time.perf_counter()
    np.random.seed(config["seed"])
    X, Y = make_dataset(config)

    results_by_depth_method = {}
    summary_rows = []

    for depth in config["depths"]:
        depth_results = {}
        for method in config["methods"]:
            weight_seed = config["seed"] + depth * 100
            weights_init = init_weights(depth, config["width"], seed=weight_seed)
            weights_final = train_method(method, weights_init, X, Y, config)

            grad_norms, sigma_maxes, loss = measure_at_step(weights_final, X, Y)
            mean_derivs = measure_mean_tanh_deriv(weights_final, X)
            row = summarize_result(
                depth=depth,
                method=method,
                weight_seed=weight_seed,
                grad_norms=grad_norms,
                sigma_maxes=sigma_maxes,
                mean_derivs=mean_derivs,
                loss=loss,
            )
            depth_results[method] = row
            summary_rows.append(row)
        results_by_depth_method[depth] = depth_results

    diagnostics = compute_diagnostics(results_by_depth_method)
    runtime_seconds = float(time.perf_counter() - start_time)

    script_path = Path(__file__).resolve()
    notebook_path = script_path.with_suffix(".ipynb")

    experiment = {
        "experiment_id": "2.15",
        "title": "Experiment 2.15: Toy tanh gradient-transport probe under orthogonality constraints",
        "identity": {
            "pair_name": "TANH_VANISHING_ortho_penalty",
            "script_path": str(script_path),
            "notebook_path": str(notebook_path),
            "recommended_run_command": f"python {PAIR_RELATIVE_DIR / 'run_experiment.py'}",
        },
        "config": config,
        "seed_policy": {
            "global_seed": int(config["seed"]),
            "dataset_seed": int(config["seed"]),
            "weight_init_seed_formula": "seed + depth * 100 (same init reused across methods at fixed depth)",
        },
        "dataset": {
            "input_shape": [int(v) for v in X.shape],
            "target_shape": [int(v) for v in Y.shape],
            "reuse_policy": "same fixed random batch reused for every depth and method",
        },
        "methods": list(config["methods"]),
        "summary_rows": summary_rows,
        "results_by_depth_method": results_by_depth_method,
        "diagnostics": diagnostics,
        "runtime_seconds": runtime_seconds,
        "caveats": [
            "Single-seed, single-batch toy regression experiment.",
            "Measurements are taken after a fixed 200 training steps, not at verified convergence.",
            "No held-out data, uncertainty estimates, or trajectory logging are included.",
            "The proxy sigma_max(W) * mean|tanh'(z)| is heuristic and does not equal full Jacobian transport.",
        ],
    }

    if verbose:
        print_report(experiment)
    return experiment



def main():
    """CLI entrypoint."""
    run_experiment(verbose=True)


if __name__ == "__main__":
    main()
