#!/usr/bin/env python3
"""
1.3b-ii-A -- Hessian: Top-5 Eigenvector Alignment With Update (2x2 net)

2-input, 2-hidden, 1-output deep linear net (6 total params).
Compute FULL 6x6 Hessian via finite differences.
Project SGD and Muon updates onto each Hessian eigenvector.

Hypothesis: Bottom 1-2 Hessian eigenvectors = gauge (flat) directions.
Muon update has near-zero projection there. SGD has uniform projection.

Context:
  - 1.2b-i showed Muon does NOT distinguish gauge from physical in Lyapunov terms
  - 1.3b-i-A showed Muon REVERSES the feedback loop (corr = -0.51)
  - This experiment checks whether Muon's step DIRECTION avoids flat Hessian directions
"""

import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Configuration
# ============================================================
np.random.seed(42)
N_SAMPLES = 20
INPUT_DIM = 2
HIDDEN_DIM = 2
OUTPUT_DIM = 1
N_PARAMS = INPUT_DIM * HIDDEN_DIM + OUTPUT_DIM * HIDDEN_DIM  # 4 + 2 = 6
EPSILON = 1e-4        # finite-difference step
LR_SGD = 0.01
LR_MUON = 0.01
N_TRAINING_STEPS = 200
MEASURE_STEPS = [0, 5, 10, 20, 40, 60, 80, 120, 160, 199]

# Fixed dataset
X = np.random.randn(N_SAMPLES, INPUT_DIM)
Y = np.random.randn(N_SAMPLES, OUTPUT_DIM)


# ============================================================
# Helpers
# ============================================================
def pack(W1, W2):
    """Flatten W1 (2x2) and W2 (1x2) into a 6-vector."""
    return np.concatenate([W1.ravel(), W2.ravel()])


def unpack(theta):
    """Reconstruct W1 (2x2) and W2 (1x2) from a 6-vector."""
    W1 = theta[:4].reshape(HIDDEN_DIM, INPUT_DIM)
    W2 = theta[4:].reshape(OUTPUT_DIM, HIDDEN_DIM)
    return W1, W2


def forward(W1, W2, X_in):
    """y = W2 @ W1 @ x for each sample."""
    # X_in: (N, 2), W1: (2, 2), W2: (1, 2)
    # hidden: (N, 2), out: (N, 1)
    hidden = X_in @ W1.T   # (N, 2)
    out = hidden @ W2.T    # (N, 1)
    return out


def loss_fn(theta):
    """MSE loss given flattened parameter vector."""
    W1, W2 = unpack(theta)
    pred = forward(W1, W2, X)
    return np.mean((pred - Y) ** 2)


def compute_gradient(theta):
    """Gradient of loss via central finite differences (6-vector)."""
    grad = np.zeros(N_PARAMS)
    for i in range(N_PARAMS):
        e_i = np.zeros(N_PARAMS)
        e_i[i] = EPSILON
        grad[i] = (loss_fn(theta + e_i) - loss_fn(theta - e_i)) / (2 * EPSILON)
    return grad


def compute_hessian(theta):
    """Full 6x6 Hessian via central finite differences."""
    H = np.zeros((N_PARAMS, N_PARAMS))
    for i in range(N_PARAMS):
        for j in range(i, N_PARAMS):
            e_i = np.zeros(N_PARAMS)
            e_j = np.zeros(N_PARAMS)
            e_i[i] = EPSILON
            e_j[j] = EPSILON
            fpp = loss_fn(theta + e_i + e_j)
            fpm = loss_fn(theta + e_i - e_j)
            fmp = loss_fn(theta - e_i + e_j)
            fmm = loss_fn(theta - e_i - e_j)
            H[i, j] = (fpp - fpm - fmp + fmm) / (4 * EPSILON ** 2)
            H[j, i] = H[i, j]
    return H


def muon_orthogonalize(G):
    """
    Newton-Schulz-style polar factor: ortho(G) = U @ V^T from SVD of G.
    This is the Muon update direction for a single weight matrix.
    For a matrix with fewer rows than columns (like W2: 1x2), we use
    the polar factor directly.
    """
    U, S, Vt = np.linalg.svd(G, full_matrices=False)
    return U @ Vt


def compute_sgd_update(theta):
    """SGD update direction: -gradient (unnormalized)."""
    grad = compute_gradient(theta)
    return -grad


def compute_muon_update(theta):
    """
    Muon update direction: for each weight matrix separately,
    compute gradient matrix, orthogonalize it via polar factor,
    then flatten all back to a 6-vector.
    """
    grad = compute_gradient(theta)
    # Split gradient into W1 and W2 parts
    grad_W1 = grad[:4].reshape(HIDDEN_DIM, INPUT_DIM)   # (2, 2)
    grad_W2 = grad[4:].reshape(OUTPUT_DIM, HIDDEN_DIM)  # (1, 2)

    # Orthogonalize each gradient matrix separately
    ortho_W1 = muon_orthogonalize(grad_W1)
    ortho_W2 = muon_orthogonalize(grad_W2)

    # Muon steps in the negative orthogonalized gradient direction
    update = np.concatenate([-ortho_W1.ravel(), -ortho_W2.ravel()])
    return update


def project_onto_eigenvectors(update, eigvecs):
    """
    Project update onto each Hessian eigenvector.
    Returns |<update, v_i>| / ||update|| for each eigenvector.
    eigvecs: columns are eigenvectors (shape N_PARAMS x N_PARAMS).
    """
    norm = np.linalg.norm(update)
    if norm < 1e-12:
        return np.zeros(N_PARAMS)
    update_hat = update / norm
    projections = np.abs(eigvecs.T @ update_hat)  # each row is one eigenvector
    return projections


# ============================================================
# Main experiment
# ============================================================
def run_experiment():
    print("=" * 90)
    print("1.3b-ii-A: Hessian Top-5 Eigenvector Alignment With Update (2x2 deep linear net)")
    print("=" * 90)
    print(f"Network: {INPUT_DIM} -> {HIDDEN_DIM} -> {OUTPUT_DIM} (deep linear)")
    print(f"Total params: {N_PARAMS}")
    print(f"Data: {N_SAMPLES} samples, MSE loss")
    print(f"Hessian: full {N_PARAMS}x{N_PARAMS} via finite differences (eps={EPSILON})")
    print(f"Training steps: {N_TRAINING_STEPS}, measured at steps: {MEASURE_STEPS}")
    print()

    # Initialize weights
    W1 = np.random.randn(HIDDEN_DIM, INPUT_DIM) * 0.5
    W2 = np.random.randn(OUTPUT_DIM, HIDDEN_DIM) * 0.5
    theta = pack(W1, W2)

    # Storage for results across steps
    all_results = []

    # We train with plain SGD and measure at specific steps
    for step in range(N_TRAINING_STEPS):
        if step in MEASURE_STEPS:
            # -- Compute Hessian --
            H = compute_hessian(theta)
            eigenvalues, eigvecs = np.linalg.eigh(H)
            # eigh returns ascending order; we want descending
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigvecs = eigvecs[:, idx]

            # -- Compute updates --
            sgd_update = compute_sgd_update(theta)
            muon_update = compute_muon_update(theta)

            # -- Project onto eigenvectors --
            proj_sgd = project_onto_eigenvectors(sgd_update, eigvecs)
            proj_muon = project_onto_eigenvectors(muon_update, eigvecs)

            current_loss = loss_fn(theta)

            result = {
                "step": step,
                "loss": current_loss,
                "eigenvalues": eigenvalues.copy(),
                "proj_sgd": proj_sgd.copy(),
                "proj_muon": proj_muon.copy(),
            }
            all_results.append(result)

            # -- Print table --
            print(f"--- Step {step:3d} | Loss = {current_loss:.6f} ---")
            print(f"  {'Eig#':>4s}  {'Eigenvalue':>12s}  {'|proj| SGD':>12s}  {'|proj| Muon':>12s}  {'Type':>10s}")
            print(f"  {'----':>4s}  {'----------':>12s}  {'----------':>12s}  {'-----------':>12s}  {'----':>10s}")

            # Label: top eigenvalues = physical, bottom = gauge (flat)
            for k in range(N_PARAMS):
                if k < 2:
                    label = "PHYSICAL"
                elif k >= N_PARAMS - 2:
                    label = "GAUGE"
                else:
                    label = "mid"
                print(f"  {k+1:4d}  {eigenvalues[k]:12.6f}  {proj_sgd[k]:12.6f}  {proj_muon[k]:12.6f}  {label:>10s}")

            # Summary stats
            phys_sgd = np.sum(proj_sgd[:2] ** 2)
            phys_muon = np.sum(proj_muon[:2] ** 2)
            gauge_sgd = np.sum(proj_sgd[-2:] ** 2)
            gauge_muon = np.sum(proj_muon[-2:] ** 2)
            print(f"  Sum-of-squares on PHYSICAL (top-2):  SGD={phys_sgd:.4f}  Muon={phys_muon:.4f}")
            print(f"  Sum-of-squares on GAUGE (bottom-2):  SGD={gauge_sgd:.4f}  Muon={gauge_muon:.4f}")
            ratio_sgd = phys_sgd / (gauge_sgd + 1e-15)
            ratio_muon = phys_muon / (gauge_muon + 1e-15)
            print(f"  Physical/Gauge ratio:                SGD={ratio_sgd:.4f}  Muon={ratio_muon:.4f}")
            print()

        # -- SGD training step (to advance the model) --
        grad = compute_gradient(theta)
        theta = theta - LR_SGD * grad

    # ============================================================
    # Aggregate analysis
    # ============================================================
    print("=" * 90)
    print("AGGREGATE ANALYSIS ACROSS ALL MEASURED STEPS")
    print("=" * 90)

    avg_proj_sgd = np.mean([r["proj_sgd"] for r in all_results], axis=0)
    avg_proj_muon = np.mean([r["proj_muon"] for r in all_results], axis=0)

    print(f"\n  Average |projection| across {len(all_results)} measurement steps:")
    print(f"  {'Eig#':>4s}  {'Avg |proj| SGD':>16s}  {'Avg |proj| Muon':>16s}  {'Muon/SGD ratio':>16s}  {'Type':>10s}")
    print(f"  {'----':>4s}  {'--------------':>16s}  {'---------------':>16s}  {'--------------':>16s}  {'----':>10s}")
    for k in range(N_PARAMS):
        if k < 2:
            label = "PHYSICAL"
        elif k >= N_PARAMS - 2:
            label = "GAUGE"
        else:
            label = "mid"
        ratio = avg_proj_muon[k] / (avg_proj_sgd[k] + 1e-15)
        print(f"  {k+1:4d}  {avg_proj_sgd[k]:16.6f}  {avg_proj_muon[k]:16.6f}  {ratio:16.4f}  {label:>10s}")

    # Physical vs Gauge mass (sum of squared projections, averaged)
    phys_mass_sgd = np.mean([np.sum(r["proj_sgd"][:2] ** 2) for r in all_results])
    phys_mass_muon = np.mean([np.sum(r["proj_muon"][:2] ** 2) for r in all_results])
    gauge_mass_sgd = np.mean([np.sum(r["proj_sgd"][-2:] ** 2) for r in all_results])
    gauge_mass_muon = np.mean([np.sum(r["proj_muon"][-2:] ** 2) for r in all_results])

    print(f"\n  Average sum-of-squared projections:")
    print(f"    PHYSICAL (top-2):   SGD={phys_mass_sgd:.6f}   Muon={phys_mass_muon:.6f}")
    print(f"    GAUGE (bottom-2):   SGD={gauge_mass_sgd:.6f}   Muon={gauge_mass_muon:.6f}")
    print(f"    Physical/Gauge:     SGD={phys_mass_sgd/(gauge_mass_sgd+1e-15):.4f}   Muon={phys_mass_muon/(gauge_mass_muon+1e-15):.4f}")

    # Hypothesis test: does Muon have lower gauge projection than SGD?
    gauge_proj_sgd_all = [np.sum(r["proj_sgd"][-2:] ** 2) for r in all_results]
    gauge_proj_muon_all = [np.sum(r["proj_muon"][-2:] ** 2) for r in all_results]
    muon_less_gauge = sum(1 for s, m in zip(gauge_proj_sgd_all, gauge_proj_muon_all) if m < s)

    print(f"\n  Hypothesis check: Muon has less gauge mass than SGD?")
    print(f"    Steps where Muon gauge-mass < SGD gauge-mass: {muon_less_gauge}/{len(all_results)}")

    phys_proj_sgd_all = [np.sum(r["proj_sgd"][:2] ** 2) for r in all_results]
    phys_proj_muon_all = [np.sum(r["proj_muon"][:2] ** 2) for r in all_results]
    muon_more_phys = sum(1 for s, m in zip(phys_proj_sgd_all, phys_proj_muon_all) if m > s)
    print(f"    Steps where Muon physical-mass > SGD physical-mass: {muon_more_phys}/{len(all_results)}")

    # Eigenvalue spectrum summary
    print(f"\n  Eigenvalue spectrum across steps (showing spread):")
    for k in range(N_PARAMS):
        evals_k = [r["eigenvalues"][k] for r in all_results]
        if k < 2:
            label = "PHYSICAL"
        elif k >= N_PARAMS - 2:
            label = "GAUGE"
        else:
            label = "mid"
        print(f"    Eig {k+1}: mean={np.mean(evals_k):10.6f}  std={np.std(evals_k):10.6f}  [{label}]")

    # ============================================================
    # Verdict
    # ============================================================
    print("\n" + "=" * 90)
    print("VERDICT")
    print("=" * 90)

    avg_gauge_ratio = gauge_mass_muon / (gauge_mass_sgd + 1e-15)
    avg_phys_ratio = phys_mass_muon / (phys_mass_sgd + 1e-15)

    if avg_gauge_ratio < 0.8 and muon_less_gauge >= len(all_results) * 0.7:
        print("  CONFIRMED: Muon consistently avoids gauge (flat Hessian) directions.")
        print(f"  Muon's gauge mass is {avg_gauge_ratio:.2f}x that of SGD (averaged).")
        verdict = "CONFIRMED"
    elif avg_gauge_ratio > 1.2 and muon_less_gauge <= len(all_results) * 0.3:
        print("  REFUTED: Muon has MORE mass on gauge directions than SGD.")
        print(f"  Muon's gauge mass is {avg_gauge_ratio:.2f}x that of SGD (averaged).")
        verdict = "REFUTED"
    else:
        print("  NUANCED: Muon's avoidance of gauge directions is not clear-cut.")
        print(f"  Muon's gauge mass is {avg_gauge_ratio:.2f}x that of SGD.")
        print(f"  Muon's physical mass is {avg_phys_ratio:.2f}x that of SGD.")
        print(f"  Steps with Muon < SGD on gauge: {muon_less_gauge}/{len(all_results)}")
        verdict = "NUANCED"

    print(f"\n  Context connection:")
    print(f"  - 1.2b-i found Muon does NOT distinguish gauge/physical in Lyapunov terms.")
    print(f"  - 1.3b-i-A found Muon REVERSES the feedback loop (corr = -0.51).")
    if verdict == "CONFIRMED":
        print(f"  - This result shows the mechanism: Muon's orthogonalization steers")
        print(f"    the update away from flat (gauge) Hessian directions.")
    elif verdict == "NUANCED":
        print(f"  - The Hessian alignment picture is nuanced, consistent with 1.2b-i's")
        print(f"    finding that Muon doesn't cleanly separate gauge from physical.")
        print(f"    The orthogonalization affects direction but not in a simple")
        print(f"    gauge-avoiding way.")
    print("=" * 90)

    # ============================================================
    # Plot
    # ============================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_steps = len(all_results)
        fig, axes = plt.subplots(2, 5, figsize=(22, 8), sharey=True)
        axes = axes.flatten()

        for idx_r, r in enumerate(all_results):
            ax = axes[idx_r]
            x_pos = np.arange(N_PARAMS)
            width = 0.35
            bars_sgd = ax.bar(x_pos - width / 2, r["proj_sgd"], width,
                              color="steelblue", alpha=0.8, label="SGD")
            bars_muon = ax.bar(x_pos + width / 2, r["proj_muon"], width,
                               color="firebrick", alpha=0.8, label="Muon")

            ax.set_title(f"Step {r['step']}\nLoss={r['loss']:.4f}", fontsize=9)
            ax.set_xlabel("Eigenvector (1=top)", fontsize=8)
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f"{k+1}\n({r['eigenvalues'][k]:.3f})"
                                for k in range(N_PARAMS)], fontsize=7)

            # Shade gauge region
            ax.axvspan(N_PARAMS - 2 - 0.5, N_PARAMS - 0.5, alpha=0.1,
                        color="gray", label="gauge zone")

            if idx_r == 0:
                ax.legend(fontsize=7, loc="upper right")

        axes[0].set_ylabel("|projection| / ||update||", fontsize=9)
        axes[5].set_ylabel("|projection| / ||update||", fontsize=9)

        fig.suptitle("1.3b-ii-A: Hessian Eigenvector Alignment -- SGD vs Muon\n"
                     "(Eigenvalues descending: left=physical/high-curvature, right=gauge/flat)",
                     fontsize=12, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.93])

        plot_path = os.path.join(SCRIPT_DIR, "hessian_eigenvector_alignment.png")
        plt.savefig(plot_path, dpi=150)
        print(f"\n  Plot saved to: {plot_path}")
        plt.close()

        # --- Additional summary plot: average projections ---
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        x_pos = np.arange(N_PARAMS)
        width = 0.35
        ax2.bar(x_pos - width / 2, avg_proj_sgd, width,
                color="steelblue", alpha=0.85, label="SGD (avg)")
        ax2.bar(x_pos + width / 2, avg_proj_muon, width,
                color="firebrick", alpha=0.85, label="Muon (avg)")

        avg_evals = np.mean([r["eigenvalues"] for r in all_results], axis=0)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f"v{k+1}\n(lam={avg_evals[k]:.3f})"
                             for k in range(N_PARAMS)], fontsize=8)
        ax2.set_ylabel("|projection| / ||update||")
        ax2.set_xlabel("Hessian eigenvector (1=highest curvature, 6=flattest)")
        ax2.set_title("Average Eigenvector Alignment: SGD vs Muon\n"
                       f"(across {n_steps} steps, {verdict})")
        ax2.axvspan(N_PARAMS - 2 - 0.5, N_PARAMS - 0.5, alpha=0.12,
                     color="gray", label="gauge zone")
        ax2.legend()
        plt.tight_layout()

        plot2_path = os.path.join(SCRIPT_DIR, "hessian_alignment_summary.png")
        fig2.savefig(plot2_path, dpi=150)
        print(f"  Summary plot saved to: {plot2_path}")
        plt.close()

    except ImportError:
        print("\n  [matplotlib not available; skipping plots]")

    return all_results


if __name__ == "__main__":
    run_experiment()
