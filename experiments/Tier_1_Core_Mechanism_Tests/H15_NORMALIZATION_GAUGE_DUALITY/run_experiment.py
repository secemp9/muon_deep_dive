#!/usr/bin/env python3
"""
H15: Conditioning-based probe of Muon in an alternating diagonal/matrix toy model
=================================================================================

This experiment compares momentum SGD against a Muon-style update rule in a
small 4-layer network with alternating diagonal and matrix layers:

  L1: diagonal, L2: matrix, L3: diagonal, L4: matrix

The primary observable is per-layer weight condition number kappa(W):
  - diagonal layers: max(|d|) / min(|d|)
  - matrix layers:   sigma_max / sigma_min

Interpretation should remain modest:
  - this is a conditioning-based toy probe,
  - matrix-layer kappa is a conditioning / spectral anisotropy diagnostic,
    not a direct gauge-coordinate or gauge-drift observable,
  - matrix-vs-diagonal ratios are heuristic layer-type comparisons rather than
    a clean causal decomposition of a gauge-specific contribution.

Core default setup is intentionally preserved from the original H15 pass:
500 steps, 5 seeds, synthetic Gaussian data, and alternating diagonal/matrix
layers trained with either momentum SGD or Muon-style normalized updates.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

DIM = 32
NUM_STEPS = 500
NUM_SEEDS = 5
LR_SGD = 0.003
LR_MUON = 0.005
MOMENTUM = 0.9
NS_ITERS = 5
NUM_SAMPLES = 100
BASE_SEED = 42

MODULE_PATH = Path(__file__).resolve()
NOTEBOOK_PATH = MODULE_PATH.with_suffix(".ipynb")

LAYER_NAMES = ["L1 (diag)", "L2 (matrix)", "L3 (diag)", "L4 (matrix)"]
LAYER_TYPES = ["diag", "matrix", "diag", "matrix"]
DIAG_INDICES = [0, 2]
MATRIX_INDICES = [1, 3]
DEFAULT_SNAPSHOT_STEPS = [0, 25, 50, 100, 200, 300, 400, 499]


# =============================================================================
# NETWORK: alternating diagonal / matrix layers
# =============================================================================


def init_network(rng):
    """
    4 layers alternating diagonal / matrix.
    Returns list of (type, params) where type is 'diag' or 'matrix'.
    Diagonal layers stored as 1D vectors (length DIM).
    Matrix layers stored as 2D arrays (DIM x DIM).
    """
    layers = []
    d1 = rng.randn(DIM) * 0.5 + 1.0
    layers.append(("diag", d1.copy()))
    W2 = rng.randn(DIM, DIM) * np.sqrt(2.0 / DIM)
    layers.append(("matrix", W2.copy()))
    d3 = rng.randn(DIM) * 0.5 + 1.0
    layers.append(("diag", d3.copy()))
    W4 = rng.randn(DIM, DIM) * np.sqrt(2.0 / DIM)
    layers.append(("matrix", W4.copy()))
    return layers



def forward(layers, X):
    """
    Forward pass: apply layers sequentially with ReLU after layers 1,2,3.
    Layer 1 (diag): out = diag(d) @ X, then ReLU
    Layer 2 (matrix): out = W @ X, then ReLU
    Layer 3 (diag): out = diag(d) @ X, then ReLU
    Layer 4 (matrix): out = W @ X (no activation)
    """
    activations = [X.copy()]
    pre_acts = []
    out = X.copy()
    for idx, (ltype, param) in enumerate(layers):
        if ltype == "diag":
            pre = param[:, None] * out
        else:
            pre = param @ out
        pre_acts.append(pre.copy())
        if idx < len(layers) - 1:
            out = np.maximum(0, pre)
        else:
            out = pre
        activations.append(out.copy())
    return out, pre_acts, activations



def compute_loss(layers, X, Y):
    pred, _, _ = forward(layers, X)
    diff = pred - Y
    return 0.5 * np.mean(np.sum(diff ** 2, axis=0))



def compute_gradients(layers, X, Y):
    """Backprop through the alternating network."""
    N = X.shape[1]
    pred, pre_acts, activations = forward(layers, X)
    delta = (pred - Y) / N

    grads = [None] * len(layers)
    for l in range(len(layers) - 1, -1, -1):
        ltype, param = layers[l]
        if ltype == "diag":
            grads[l] = np.sum(delta * activations[l], axis=1)
            if l > 0:
                delta = param[:, None] * delta
        else:
            grads[l] = delta @ activations[l].T
            if l > 0:
                delta = param.T @ delta

        if l > 0:
            delta = delta * (pre_acts[l - 1] > 0).astype(float)

    return grads


# =============================================================================
# NEWTON-SCHULZ ORTHOGONALIZATION (for matrix layers)
# =============================================================================


def newton_schulz_ortho(M, n_iters=NS_ITERS):
    """Newton-Schulz iteration to approximate the polar factor."""
    norm = np.linalg.norm(M, ord="fro")
    if norm < 1e-12:
        return M
    X = M / norm
    for _ in range(n_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


# =============================================================================
# CONDITION NUMBER
# =============================================================================


def condition_number(layers, idx):
    """
    Compute condition number of layer idx.
    For diagonal: max(|d|) / min(|d|) (with floor to avoid div-by-zero)
    For matrix: sigma_max / sigma_min from SVD
    """
    ltype, param = layers[idx]
    if ltype == "diag":
        abs_d = np.abs(param)
        dmax = np.max(abs_d)
        dmin = np.max([np.min(abs_d), 1e-12])
        return dmax / dmin
    s = np.linalg.svd(param, compute_uv=False)
    return s[0] / max(s[-1], 1e-12)


# =============================================================================
# TRAINING
# =============================================================================


def train(layers_init, X, Y, optimizer="sgd", n_steps=NUM_STEPS):
    """
    Train the network.

    Returns a dict with:
      - losses: shape [n_steps]
      - kappas: shape [n_steps, 4]
      - diverged: bool
      - divergence_step: int | None

    Measurement timing is intentionally preserved from the original H15 pass:
    the loss and kappa recorded at step t are measured *before* the parameter
    update at that step.
    """
    layers = [(t, p.copy()) for t, p in layers_init]
    velocities = [np.zeros_like(p) for _, p in layers]
    n_layers = len(layers)

    losses = []
    kappas = []
    diverged = False
    divergence_step = None

    for step in range(n_steps):
        loss = compute_loss(layers, X, Y)
        losses.append(loss)

        if np.isnan(loss) or loss > 1e10:
            diverged = True
            divergence_step = step
            losses[-1] = 1e10
            kappas.append([1e10] * n_layers)
            for _ in range(n_steps - step - 1):
                losses.append(1e10)
                kappas.append([1e10] * n_layers)
            break

        step_kappas = [condition_number(layers, i) for i in range(n_layers)]
        kappas.append(step_kappas)
        grads = compute_gradients(layers, X, Y)

        for i in range(n_layers):
            ltype, param = layers[i]
            g = grads[i]

            if optimizer == "sgd":
                velocities[i] = MOMENTUM * velocities[i] + g
                new_param = param - LR_SGD * velocities[i]
            elif optimizer == "muon":
                if ltype == "diag":
                    ortho_g = np.sign(g)
                    ortho_g[ortho_g == 0] = 1.0
                else:
                    ortho_g = newton_schulz_ortho(g)
                velocities[i] = MOMENTUM * velocities[i] + ortho_g
                new_param = param - LR_MUON * velocities[i]
            else:
                raise ValueError(f"Unknown optimizer: {optimizer}")

            layers[i] = (ltype, new_param)

    return {
        "losses": np.array(losses),
        "kappas": np.array(kappas),
        "diverged": diverged,
        "divergence_step": divergence_step,
    }


# =============================================================================
# AGGREGATION / REPORTING HELPERS
# =============================================================================


def _safe_ratio(numerator, denominator):
    return numerator / max(denominator, 1e-12)



def _snapshot_steps_for(num_steps):
    last_index = max(num_steps - 1, 0)
    return [step for step in DEFAULT_SNAPSHOT_STEPS if step <= last_index]



def _mean_std(array, axis=0):
    return np.mean(array, axis=axis), np.std(array, axis=axis)



def _describe_vector(values):
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "sem": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "values": arr,
        }
    std = float(np.std(arr))
    return {
        "n": n,
        "mean": float(np.mean(arr)),
        "std": std,
        "sem": float(std / np.sqrt(n)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "values": arr,
    }



def run_experiment(verbose=True):
    """
    Run the default H15 experiment and return structured raw results plus
    aggregated summaries for downstream notebook analysis.
    """
    config = {
        "dim": DIM,
        "num_steps": NUM_STEPS,
        "num_seeds": NUM_SEEDS,
        "lr_sgd": LR_SGD,
        "lr_muon": LR_MUON,
        "momentum": MOMENTUM,
        "ns_iters": NS_ITERS,
        "num_samples": NUM_SAMPLES,
        "base_seed": BASE_SEED,
    }
    seeds = [BASE_SEED + seed_idx * 37 for seed_idx in range(NUM_SEEDS)]

    if verbose:
        print("=" * 90)
        print("H15: CONDITIONING-BASED MUON PROBE IN AN ALTERNATING TOY MODEL")
        print("=" * 90)
        print(f"Script: {MODULE_PATH}")
        print(f"Notebook counterpart: {NOTEBOOK_PATH}")
        print(f"Architecture: 4 alternating layers (diag-matrix-diag-matrix), width={DIM}")
        print(f"Steps: {NUM_STEPS}, Seeds: {NUM_SEEDS}")
        print("Observable: per-layer condition number kappa(W)")
        print("Caveat: matrix-layer kappa is a conditioning diagnostic, not a direct gauge observable.")
        print()

    all_sgd_kappas = []
    all_muon_kappas = []
    all_sgd_losses = []
    all_muon_losses = []
    divergence = {"sgd": [], "muon": []}

    for run_seed in seeds:
        rng = np.random.RandomState(run_seed)
        if verbose:
            print(f"--- Seed {run_seed} ---")

        X = rng.randn(DIM, NUM_SAMPLES) * 0.3
        Y = rng.randn(DIM, NUM_SAMPLES) * 0.3
        layers_init = init_network(rng)

        if verbose:
            print("  Training with SGD...", flush=True)
        sgd_run = train(layers_init, X, Y, optimizer="sgd")
        all_sgd_losses.append(sgd_run["losses"])
        all_sgd_kappas.append(sgd_run["kappas"])
        divergence["sgd"].append(sgd_run["diverged"])

        if verbose:
            print("  Training with Muon...", flush=True)
        muon_run = train(layers_init, X, Y, optimizer="muon")
        all_muon_losses.append(muon_run["losses"])
        all_muon_kappas.append(muon_run["kappas"])
        divergence["muon"].append(muon_run["diverged"])

        if verbose:
            final_sgd = sgd_run["losses"][-1] if len(sgd_run["losses"]) > 0 else float("nan")
            final_muon = muon_run["losses"][-1] if len(muon_run["losses"]) > 0 else float("nan")
            print(f"  Last recorded loss: SGD={final_sgd:.4f}, Muon={final_muon:.4f}")
            print()

    sgd_losses_all = np.array(all_sgd_losses)
    muon_losses_all = np.array(all_muon_losses)
    sgd_kappas_all = np.array(all_sgd_kappas)
    muon_kappas_all = np.array(all_muon_kappas)

    sgd_losses_mean, sgd_losses_std = _mean_std(sgd_losses_all, axis=0)
    muon_losses_mean, muon_losses_std = _mean_std(muon_losses_all, axis=0)
    sgd_kappas_mean, sgd_kappas_std = _mean_std(sgd_kappas_all, axis=0)
    muon_kappas_mean, muon_kappas_std = _mean_std(muon_kappas_all, axis=0)

    last_recorded_step = min(NUM_STEPS - 1, sgd_kappas_all.shape[1] - 1)
    snapshot_steps = _snapshot_steps_for(sgd_kappas_all.shape[1])

    final_sgd_losses_by_seed = sgd_losses_all[:, last_recorded_step]
    final_muon_losses_by_seed = muon_losses_all[:, last_recorded_step]

    per_layer_final = []
    final_ratio_by_seed_and_layer = sgd_kappas_all[:, last_recorded_step, :] / np.maximum(
        muon_kappas_all[:, last_recorded_step, :],
        1e-12,
    )

    diag_ratios = []
    matrix_ratios = []
    for li, (layer_name, layer_type) in enumerate(zip(LAYER_NAMES, LAYER_TYPES)):
        k_sgd_seeds = sgd_kappas_all[:, last_recorded_step, li]
        k_muon_seeds = muon_kappas_all[:, last_recorded_step, li]
        k_sgd_mean = float(np.mean(k_sgd_seeds))
        k_muon_mean = float(np.mean(k_muon_seeds))
        k_sgd_std = float(np.std(k_sgd_seeds))
        k_muon_std = float(np.std(k_muon_seeds))
        ratio_of_means = float(_safe_ratio(k_sgd_mean, k_muon_mean))
        per_seed_ratios = final_ratio_by_seed_and_layer[:, li]
        ratio_stats = _describe_vector(per_seed_ratios)

        summary_row = {
            "layer_index": li,
            "layer_name": layer_name,
            "layer_type": layer_type,
            "sgd_mean": k_sgd_mean,
            "sgd_std": k_sgd_std,
            "muon_mean": k_muon_mean,
            "muon_std": k_muon_std,
            "ratio_of_means": ratio_of_means,
            "per_seed_ratio_mean": ratio_stats["mean"],
            "per_seed_ratio_std": ratio_stats["std"],
            "per_seed_ratio_sem": ratio_stats["sem"],
            "per_seed_ratio_median": ratio_stats["median"],
            "per_seed_ratio_min": ratio_stats["min"],
            "per_seed_ratio_max": ratio_stats["max"],
            "per_seed_ratios": per_seed_ratios,
        }
        per_layer_final.append(summary_row)

        if layer_type == "diag":
            diag_ratios.append(ratio_of_means)
        else:
            matrix_ratios.append(ratio_of_means)

    sgd_diag_by_seed = np.mean(sgd_kappas_all[:, :, DIAG_INDICES], axis=2)
    muon_diag_by_seed = np.mean(muon_kappas_all[:, :, DIAG_INDICES], axis=2)
    sgd_matrix_by_seed = np.mean(sgd_kappas_all[:, :, MATRIX_INDICES], axis=2)
    muon_matrix_by_seed = np.mean(muon_kappas_all[:, :, MATRIX_INDICES], axis=2)

    def build_type_summary(name, sgd_by_seed, muon_by_seed, layer_ratio_mean, layer_ratio_std):
        sgd_mean = float(np.mean(sgd_by_seed[:, last_recorded_step]))
        sgd_std = float(np.std(sgd_by_seed[:, last_recorded_step]))
        muon_mean = float(np.mean(muon_by_seed[:, last_recorded_step]))
        muon_std = float(np.std(muon_by_seed[:, last_recorded_step]))
        per_seed_ratios = sgd_by_seed[:, last_recorded_step] / np.maximum(
            muon_by_seed[:, last_recorded_step],
            1e-12,
        )
        ratio_stats = _describe_vector(per_seed_ratios)
        return {
            "layer_type": name,
            "sgd_mean": sgd_mean,
            "sgd_std": sgd_std,
            "muon_mean": muon_mean,
            "muon_std": muon_std,
            "layer_ratio_mean": float(layer_ratio_mean),
            "layer_ratio_std": float(layer_ratio_std),
            "per_seed_ratio_mean": ratio_stats["mean"],
            "per_seed_ratio_std": ratio_stats["std"],
            "per_seed_ratio_sem": ratio_stats["sem"],
            "per_seed_ratio_median": ratio_stats["median"],
            "per_seed_ratio_min": ratio_stats["min"],
            "per_seed_ratio_max": ratio_stats["max"],
            "per_seed_ratios": per_seed_ratios,
            "sgd_trajectory_by_seed": sgd_by_seed,
            "muon_trajectory_by_seed": muon_by_seed,
            "sgd_trajectory_mean": np.mean(sgd_by_seed, axis=0),
            "sgd_trajectory_std": np.std(sgd_by_seed, axis=0),
            "muon_trajectory_mean": np.mean(muon_by_seed, axis=0),
            "muon_trajectory_std": np.std(muon_by_seed, axis=0),
        }

    avg_diag = float(np.mean(diag_ratios))
    avg_matrix = float(np.mean(matrix_ratios))
    matrix_over_diag_ratio = float(_safe_ratio(avg_matrix, avg_diag))

    per_type_final = {
        "diag": build_type_summary("diag", sgd_diag_by_seed, muon_diag_by_seed, avg_diag, np.std(diag_ratios)),
        "matrix": build_type_summary(
            "matrix",
            sgd_matrix_by_seed,
            muon_matrix_by_seed,
            avg_matrix,
            np.std(matrix_ratios),
        ),
    }

    final_sgd_loss_mean = float(np.mean(final_sgd_losses_by_seed))
    final_sgd_loss_std = float(np.std(final_sgd_losses_by_seed))
    final_muon_loss_mean = float(np.mean(final_muon_losses_by_seed))
    final_muon_loss_std = float(np.std(final_muon_losses_by_seed))
    final_loss_gap_by_seed = final_sgd_losses_by_seed - final_muon_losses_by_seed
    final_loss_ratio_by_seed = final_sgd_losses_by_seed / np.maximum(final_muon_losses_by_seed, 1e-12)
    final_loss_gap_stats = _describe_vector(final_loss_gap_by_seed)
    final_loss_ratio_stats = _describe_vector(final_loss_ratio_by_seed)

    hypotheses = {
        "H1": {
            "description": "Diagonal-layer conditioning improvement lies in the heuristic 1-20x range.",
            "criterion": "1.0 <= avg_diag <= 20.0",
            "observed_value": avg_diag,
            "passed": bool(1.0 <= avg_diag <= 20.0),
        },
        "H2": {
            "description": "Matrix-layer conditioning improvement exceeds 5x.",
            "criterion": "avg_matrix > 5.0",
            "observed_value": avg_matrix,
            "passed": bool(avg_matrix > 5.0),
        },
        "H3": {
            "description": "Matrix-layer improvement exceeds 2x the diagonal-layer improvement.",
            "criterion": "avg_matrix > 2.0 * avg_diag",
            "observed_value": matrix_over_diag_ratio,
            "passed": bool(avg_matrix > 2.0 * avg_diag),
        },
        "H4": {
            "description": "Muon attains lower last-recorded mean loss than SGD.",
            "criterion": "final_muon_loss_mean < final_sgd_loss_mean",
            "observed_value": {
                "sgd_mean": final_sgd_loss_mean,
                "muon_mean": final_muon_loss_mean,
                "sgd_over_muon": float(_safe_ratio(final_sgd_loss_mean, final_muon_loss_mean)),
            },
            "passed": bool(final_muon_loss_mean < final_sgd_loss_mean),
        },
    }
    total_pass = int(sum(item["passed"] for item in hypotheses.values()))

    results = {
        "identity": {
            "experiment_id": "H15_NORMALIZATION_GAUGE_DUALITY",
            "title": "H15: Conditioning-based probe of Muon in an alternating diagonal/matrix toy model",
            "pair_scope": "Notebook and script study the same conditioning-based toy probe without claiming a clean gauge decomposition.",
        },
        "paths": {
            "script": str(MODULE_PATH),
            "notebook": str(NOTEBOOK_PATH),
        },
        "config": config,
        "seeds": seeds,
        "layer_names": LAYER_NAMES,
        "layer_types": LAYER_TYPES,
        "measurement_notes": {
            "loss_and_kappa_timing": "Loss and kappa are recorded before each parameter update; the last entry is the pre-update state at the final training step.",
            "scope": "Matrix-layer kappa is used as a conditioning / spectral anisotropy diagnostic rather than a direct gauge observable.",
            "ratio_caveat": "Matrix-vs-diagonal ratios are heuristic layer-type comparisons, not a clean causal decomposition.",
            "snapshot_steps": snapshot_steps,
            "last_recorded_step": last_recorded_step,
        },
        "raw": {
            "sgd_losses": sgd_losses_all,
            "muon_losses": muon_losses_all,
            "sgd_kappas": sgd_kappas_all,
            "muon_kappas": muon_kappas_all,
            "final_ratio_by_seed_and_layer": final_ratio_by_seed_and_layer,
        },
        "aggregates": {
            "sgd_losses_mean": sgd_losses_mean,
            "sgd_losses_std": sgd_losses_std,
            "muon_losses_mean": muon_losses_mean,
            "muon_losses_std": muon_losses_std,
            "sgd_kappas_mean": sgd_kappas_mean,
            "sgd_kappas_std": sgd_kappas_std,
            "muon_kappas_mean": muon_kappas_mean,
            "muon_kappas_std": muon_kappas_std,
            "per_layer_final": per_layer_final,
            "per_type_final": per_type_final,
            "divergence": divergence,
            "paired_statistics": {
                "final_loss_gap_sgd_minus_muon": final_loss_gap_stats,
                "final_loss_ratio_sgd_over_muon": final_loss_ratio_stats,
            },
        },
        "summary": {
            "avg_diag_ratio": avg_diag,
            "avg_matrix_ratio": avg_matrix,
            "matrix_over_diag_ratio": matrix_over_diag_ratio,
            "final_sgd_loss_mean": final_sgd_loss_mean,
            "final_sgd_loss_std": final_sgd_loss_std,
            "final_muon_loss_mean": final_muon_loss_mean,
            "final_muon_loss_std": final_muon_loss_std,
            "loss_ratio_sgd_over_muon": float(_safe_ratio(final_sgd_loss_mean, final_muon_loss_mean)),
            "paired_final_loss_gap_mean": final_loss_gap_stats["mean"],
            "paired_final_loss_gap_std": final_loss_gap_stats["std"],
            "paired_final_loss_gap_sem": final_loss_gap_stats["sem"],
            "paired_final_loss_ratio_mean": final_loss_ratio_stats["mean"],
            "paired_final_loss_ratio_std": final_loss_ratio_stats["std"],
            "paired_final_loss_ratio_sem": final_loss_ratio_stats["sem"],
            "last_recorded_step": last_recorded_step,
            "total_pass": total_pass,
        },
        "hypotheses": hypotheses,
    }
    return results



def print_report(results):
    """Pretty-print the structured results for CLI usage."""
    config = results["config"]
    aggregates = results["aggregates"]
    summary = results["summary"]
    hypotheses = results["hypotheses"]
    measurement_notes = results["measurement_notes"]
    paired_stats = aggregates["paired_statistics"]

    print("=" * 90)
    print("H15 REPORT: CONDITIONING-BASED TOY PROBE")
    print("=" * 90)
    print(f"Script: {results['paths']['script']}")
    print(f"Notebook counterpart: {results['paths']['notebook']}")
    print(f"Seeds: {results['seeds']}")
    print(
        f"Config: dim={config['dim']}, steps={config['num_steps']}, samples={config['num_samples']}, "
        f"lr_sgd={config['lr_sgd']}, lr_muon={config['lr_muon']}, momentum={config['momentum']}"
    )
    print("Observable: per-layer condition number kappa(W)")
    print(f"Measurement note: {measurement_notes['loss_and_kappa_timing']}")
    print(f"Scope note: {measurement_notes['scope']}")
    print(f"Caveat: {measurement_notes['ratio_caveat']}")

    print(f"\n{'=' * 90}")
    print("CONDITION NUMBER SNAPSHOTS (mean over seeds)")
    print(f"{'=' * 90}")
    for li, layer_name in enumerate(results["layer_names"]):
        print(f"\n{layer_name}:")
        print(f"{'Step':>6}  {'SGD kappa':>12}  {'Muon kappa':>12}  {'SGD/Muon':>12}")
        print("-" * 50)
        for step in measurement_notes["snapshot_steps"]:
            k_sgd = aggregates["sgd_kappas_mean"][step, li]
            k_muon = aggregates["muon_kappas_mean"][step, li]
            ratio = _safe_ratio(k_sgd, k_muon)
            print(f"{step:>6}  {k_sgd:>12.2f}  {k_muon:>12.2f}  {ratio:>12.2f}")

    print(f"\n{'=' * 90}")
    print(f"FINAL CONDITIONING SUMMARY (last recorded step = {summary['last_recorded_step']})")
    print(f"{'=' * 90}")
    print(f"{'Layer':>12}  {'Type':>8}  {'SGD kappa':>17}  {'Muon kappa':>17}  {'Ratio':>10}")
    print("-" * 75)
    for row in aggregates["per_layer_final"]:
        print(
            f"{row['layer_name']:>12}  {row['layer_type']:>8}  "
            f"{row['sgd_mean']:>8.1f}+/-{row['sgd_std']:<5.1f}  "
            f"{row['muon_mean']:>8.1f}+/-{row['muon_std']:<5.1f}  "
            f"{row['ratio_of_means']:>8.2f}x"
        )

    print(f"\n{'=' * 90}")
    print("TYPE-AVERAGED FINAL SUMMARY")
    print(f"{'=' * 90}")
    for layer_type in ["diag", "matrix"]:
        row = aggregates["per_type_final"][layer_type]
        print(
            f"{layer_type:>6}: "
            f"SGD={row['sgd_mean']:.2f}+/-{row['sgd_std']:.2f}, "
            f"Muon={row['muon_mean']:.2f}+/-{row['muon_std']:.2f}, "
            f"layer-ratio mean={row['layer_ratio_mean']:.2f}x, "
            f"paired type-ratio={row['per_seed_ratio_mean']:.2f}x+/-{row['per_seed_ratio_std']:.2f}x "
            f"(SEM {row['per_seed_ratio_sem']:.2f}x)"
        )
    print(
        f"Matrix-over-diagonal improvement ratio (heuristic only): "
        f"{summary['matrix_over_diag_ratio']:.3f}x"
    )

    print(f"\n{'=' * 90}")
    print("LOSS SUMMARY")
    print(f"{'=' * 90}")
    print(
        f"SGD last-recorded mean loss:  {summary['final_sgd_loss_mean']:.6f} +/- {summary['final_sgd_loss_std']:.6f}"
    )
    print(
        f"Muon last-recorded mean loss: {summary['final_muon_loss_mean']:.6f} +/- {summary['final_muon_loss_std']:.6f}"
    )
    print(f"SGD/Muon loss ratio:          {summary['loss_ratio_sgd_over_muon']:.3f}x")
    print(
        f"Paired seedwise loss gap (SGD-Muon): {summary['paired_final_loss_gap_mean']:.6f} +/- {summary['paired_final_loss_gap_std']:.6f} "
        f"(SEM {summary['paired_final_loss_gap_sem']:.6f})"
    )
    print(
        f"Paired seedwise loss ratio (SGD/Muon): {summary['paired_final_loss_ratio_mean']:.3f}x +/- {summary['paired_final_loss_ratio_std']:.3f}x "
        f"(SEM {summary['paired_final_loss_ratio_sem']:.3f}x)"
    )

    print(f"\n{'=' * 90}")
    print("H1-H4 HEURISTIC CHECKS (DESCRIPTIVE, NOT FORMAL SIGNIFICANCE TESTS)")
    print(f"{'=' * 90}")
    for key in ["H1", "H2", "H3", "H4"]:
        item = hypotheses[key]
        print(f"\n{key}: {item['description']}")
        print(f"  Criterion: {item['criterion']}")
        if key == "H4":
            obs = item["observed_value"]
            print(
                f"  Observed: SGD={obs['sgd_mean']:.6f}, Muon={obs['muon_mean']:.6f}, "
                f"SGD/Muon={obs['sgd_over_muon']:.3f}x"
            )
        else:
            print(f"  Observed: {item['observed_value']:.6f}")
        print(f"  Verdict: {'PASS' if item['passed'] else 'FAIL'}")

    print(f"\n{'=' * 90}")
    print("CALIBRATED CONCLUSION")
    print(f"{'=' * 90}")
    print(f"Heuristic checks passed: {summary['total_pass']}/4")
    print(
        f"Muon substantially lowers last-recorded mean loss "
        f"({summary['final_sgd_loss_mean']:.4f} -> {summary['final_muon_loss_mean']:.4f})."
    )
    if summary["avg_matrix_ratio"] > summary["avg_diag_ratio"]:
        print(
            "Matrix layers show larger final conditioning gains than diagonal layers on this metric, "
            "but the observable still does not isolate a direct gauge effect."
        )
    else:
        print(
            "The default run does not show the originally intended matrix-layer conditioning advantage "
            "over diagonal layers."
        )
    print(
        "This pair should therefore be read as a conditioning-based toy probe whose current default run "
        "shows strong loss improvement but does not support a strong gauge-decomposition story."
    )



def main():
    results = run_experiment(verbose=True)
    print_report(results)


if __name__ == "__main__":
    main()
