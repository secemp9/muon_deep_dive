#!/usr/bin/env python3
"""
Experiment 2.18: Adaptive NS steps -- k(t) decreasing over training

Hypothesis: k=20 hurts (confirmed 1.4c-ii). Adaptive schedule
  k(t) = max(1, round(5 - 4*t/T))
outperforms fixed k=5 by >3% final loss.

Tests 5 NS schedules on 4-layer deep linear and ReLU networks:
  (a) Fixed k=1
  (b) Fixed k=5  (Muon default)
  (c) Fixed k=10
  (d) Adaptive-linear: k(t) = max(1, round(5 - 4*t/T))
  (e) Adaptive-erank: k(t) = max(1, round(5 * erank(G)/n))

Uses Muon's quintic NS coefficients: a=3.4445, b=-4.7750, c=2.0315
  X_{k+1} = a * X_k + b * X_k @ (X_k^T @ X_k) + c * X_k @ (X_k^T @ X_k)^2
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# Core functions
# ============================================================================

def newton_schulz_quintic(G, num_iters=5):
    """
    Muon's quintic Newton-Schulz iteration.
    Coefficients: a=3.4445, b=-4.7750, c=2.0315  (satisfy a+b+c=1)
    X_{k+1} = a*X + b*X@(X^T@X) + c*X@(X^T@X)^2

    Each iteration costs 4 matmuls:
      1) X^T @ X        (n x m) @ (m x n) = n x n
      2) X @ (X^T@X)    (m x n) @ (n x n) = m x n   [used for b-term]
      3) (X^T@X)^2      (n x n) @ (n x n) = n x n
      4) X @ (X^T@X)^2  (m x n) @ (n x n) = m x n   [used for c-term]
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    m, n = G.shape
    X = G / (np.linalg.norm(G, 'fro') + 1e-30)

    for _ in range(num_iters):
        XtX = X.T @ X             # matmul 1
        X_XtX = X @ XtX           # matmul 2
        XtX2 = XtX @ XtX          # matmul 3
        X_XtX2 = X @ XtX2         # matmul 4
        X = a * X + b * X_XtX + c * X_XtX2

    return X


def effective_rank(M):
    """
    Effective rank via Shannon entropy of normalized singular values.
    erank = exp(H(p)), where p_i = sigma_i / sum(sigma).
    """
    s = np.linalg.svd(M, compute_uv=False)
    s = s[s > 1e-30]
    if len(s) == 0:
        return 1.0
    p = s / s.sum()
    H = -np.sum(p * np.log(p + 1e-30))
    return np.exp(H)


def relu(x):
    return np.maximum(0, x)


def relu_deriv(x):
    return (x > 0).astype(float)


# ============================================================================
# Schedule definitions
# ============================================================================

def schedule_fixed(k_val):
    """Return a schedule function that always returns k_val."""
    def sched(t, T, G=None, n=None):
        return k_val
    sched.__name__ = f"Fixed k={k_val}"
    return sched


def schedule_adaptive_linear(t, T, G=None, n=None):
    """k(t) = max(1, round(5 - 4*t/T)) -- starts at 5, linearly to 1."""
    return max(1, round(5 - 4 * t / T))
schedule_adaptive_linear.__name__ = "Adaptive-linear"


def schedule_adaptive_erank(t, T, G=None, n=None):
    """k(t) = max(1, round(5 * erank(G)/n)) -- fewer steps when gradient is low-rank."""
    if G is None:
        return 5
    er = effective_rank(G)
    return max(1, round(5 * er / n))
schedule_adaptive_erank.__name__ = "Adaptive-erank"


# ============================================================================
# Training: Deep Linear Network
# ============================================================================

def train_deep_linear(depth=4, width=32, input_dim=32, output_dim=32,
                      n_steps=500, lr=0.02, schedule_fn=None, seed=42):
    """
    Train deep linear network W_1 @ W_2 @ ... @ W_depth with Muon.
    Returns: losses, eranks_per_step, k_per_step, total_ns_matmuls
    """
    if schedule_fn is None:
        schedule_fn = schedule_fixed(5)

    np.random.seed(seed)
    target = np.random.randn(output_dim, input_dim) * 0.5

    # Initialize weights
    dims = [input_dim] + [width] * (depth - 1) + [output_dim]
    weights = []
    for i in range(depth):
        fan_in, fan_out = dims[i], dims[i + 1]
        W = np.random.randn(fan_out, fan_in) * np.sqrt(2.0 / (fan_in + fan_out))
        weights.append(W)

    losses = []
    eranks = []
    k_values_used = []
    total_ns_matmuls = 0

    for step in range(n_steps):
        # Forward: product of all layers
        product = np.eye(input_dim)
        for W in weights:
            product = W @ product

        # Loss
        diff = product - target
        loss = 0.5 * np.linalg.norm(diff, 'fro') ** 2
        losses.append(loss)

        # Backward: gradients for each layer
        grads = []
        for k_layer in range(depth):
            left = np.eye(weights[k_layer].shape[0])
            for j in range(k_layer + 1, depth):
                left = weights[j] @ left

            right = np.eye(input_dim)
            for j in range(k_layer):
                right = weights[j] @ right

            grad = left.T @ diff @ right.T
            grads.append(grad)

        # Track gradient effective rank (use first layer's gradient)
        er = effective_rank(grads[0])
        eranks.append(er)
        n_dim = min(grads[0].shape)

        # Muon update with scheduled k
        for k_layer in range(depth):
            G = grads[k_layer]
            k_ns = schedule_fn(step, n_steps, G=G, n=min(G.shape))
            if k_layer == 0:
                k_values_used.append(k_ns)
            G_orth = newton_schulz_quintic(G, num_iters=k_ns)
            total_ns_matmuls += k_ns * 4  # 4 matmuls per NS iteration per layer
            weights[k_layer] -= lr * G_orth

    return losses, eranks, k_values_used, total_ns_matmuls


# ============================================================================
# Training: ReLU Network
# ============================================================================

def train_relu_net(depth=4, width=32, input_dim=32, output_dim=32,
                   n_steps=500, lr=0.01, schedule_fn=None, seed=42,
                   batch_size=64):
    """
    Train ReLU MLP: x -> ReLU(W1 x) -> ReLU(W2 ...) -> W_L ...
    Quadratic loss on random targets.
    Returns: losses, eranks, k_per_step, total_ns_matmuls
    """
    if schedule_fn is None:
        schedule_fn = schedule_fixed(5)

    np.random.seed(seed)

    # Random data
    X_data = np.random.randn(batch_size, input_dim)
    Y_data = np.random.randn(batch_size, output_dim) * 0.3

    # Initialize weights
    dims = [input_dim] + [width] * (depth - 1) + [output_dim]
    weights = []
    for i in range(depth):
        fan_in, fan_out = dims[i], dims[i + 1]
        W = np.random.randn(fan_out, fan_in) * np.sqrt(2.0 / fan_in)
        weights.append(W)

    losses = []
    eranks = []
    k_values_used = []
    total_ns_matmuls = 0

    for step in range(n_steps):
        # Forward pass with ReLU activations
        activations = [X_data.T]  # input_dim x batch
        for i in range(depth):
            z = weights[i] @ activations[-1]
            if i < depth - 1:
                a = relu(z)
            else:
                a = z  # no activation on last layer
            activations.append(a)

        # Loss: 0.5 * ||output - Y||^2
        output = activations[-1].T  # batch x output_dim
        diff = output - Y_data  # batch x output_dim
        loss = 0.5 * np.mean(np.sum(diff ** 2, axis=1))
        losses.append(loss)

        # Backward pass
        # dL/d(output) = diff / batch_size
        delta = diff.T / batch_size  # output_dim x batch

        grads = [None] * depth
        for i in range(depth - 1, -1, -1):
            # Gradient for weight i
            grads[i] = delta @ activations[i].T  # (dim_out x batch) @ (batch x dim_in)

            if i > 0:
                # Propagate through weight
                delta = weights[i].T @ delta
                # ReLU derivative
                pre_act = weights[i - 1] @ activations[i - 1] if i >= 1 else activations[0]
                # We need the pre-activation to apply relu_deriv
                # Actually activations[i] = relu(weights[i-1] @ activations[i-1]) for i>=1
                # We stored the post-activation, so we use the sign
                delta = delta * relu_deriv(weights[i - 1] @ activations[i - 1])

        # Track gradient effective rank
        er = effective_rank(grads[0])
        eranks.append(er)

        # Muon update
        for k_layer in range(depth):
            G = grads[k_layer]
            k_ns = schedule_fn(step, n_steps, G=G, n=min(G.shape))
            if k_layer == 0:
                k_values_used.append(k_ns)
            G_orth = newton_schulz_quintic(G, num_iters=k_ns)
            total_ns_matmuls += k_ns * 4
            weights[k_layer] -= lr * G_orth

    return losses, eranks, k_values_used, total_ns_matmuls


# ============================================================================
# Main experiment
# ============================================================================

def run_experiment(train_fn, net_type, n_steps=500, **train_kwargs):
    """Run all 5 schedules and collect results."""

    schedules = {
        "(a) Fixed k=1":       schedule_fixed(1),
        "(b) Fixed k=5":       schedule_fixed(5),
        "(c) Fixed k=10":      schedule_fixed(10),
        "(d) Adaptive-linear": schedule_adaptive_linear,
        "(e) Adaptive-erank":  schedule_adaptive_erank,
    }

    results = {}
    for name, sched in schedules.items():
        losses, eranks, k_used, total_matmuls = train_fn(
            n_steps=n_steps, schedule_fn=sched, **train_kwargs
        )
        results[name] = {
            'losses': losses,
            'eranks': eranks,
            'k_used': k_used,
            'total_matmuls': total_matmuls,
            'final_loss': losses[-1],
        }

    return results


def print_table(results, net_type):
    """Print the results table."""
    ref = results["(b) Fixed k=5"]
    ref_loss = ref['final_loss']
    ref_matmuls = ref['total_matmuls']

    print(f"\n{'='*100}")
    print(f"  Results: {net_type}")
    print(f"{'='*100}")
    print(f"  {'Schedule':<25s} {'Final loss':>12s} {'Total NS matmuls':>18s} "
          f"{'Pareto score':>14s} {'vs k=5 loss':>14s} {'vs k=5 flops':>14s}")
    print(f"  {'-'*25} {'-'*12} {'-'*18} {'-'*14} {'-'*14} {'-'*14}")

    for name, r in results.items():
        fl = r['final_loss']
        tm = r['total_matmuls']
        pareto = fl * tm
        vs_loss = (fl - ref_loss) / ref_loss * 100
        vs_flops = (tm - ref_matmuls) / ref_matmuls * 100
        print(f"  {name:<25s} {fl:12.6e} {tm:18d} {pareto:14.4e} "
              f"{vs_loss:+13.1f}% {vs_flops:+13.1f}%")

    print()


def print_erank_analysis(results, net_type):
    """Verify that gradient erank decreases over training."""
    print(f"\n  Gradient effective rank analysis ({net_type}):")
    print(f"  {'Schedule':<25s} {'erank(t=0)':>12s} {'erank(t=T/2)':>14s} "
          f"{'erank(t=T)':>12s} {'Decreasing?':>13s}")
    print(f"  {'-'*25} {'-'*12} {'-'*14} {'-'*12} {'-'*13}")

    for name, r in results.items():
        eranks = r['eranks']
        T = len(eranks)
        er_start = np.mean(eranks[:20])
        er_mid = np.mean(eranks[T//2 - 10: T//2 + 10])
        er_end = np.mean(eranks[-20:])
        decreasing = er_end < er_start
        print(f"  {name:<25s} {er_start:12.2f} {er_mid:14.2f} {er_end:12.2f} "
              f"{'YES' if decreasing else 'NO':>13s}")


def main():
    np.random.seed(42)
    N_STEPS = 500

    print("=" * 100)
    print("  Experiment 2.18: Adaptive NS steps -- k(t) decreasing over training")
    print("=" * 100)
    print()
    print("  Hypothesis: Adaptive schedule k(t) = max(1, round(5 - 4*t/T))")
    print("              outperforms fixed k=5 by >3% final loss.")
    print()
    print("  NS iteration: quintic (a=3.4445, b=-4.7750, c=2.0315)")
    print("  4 matmuls per NS iteration per layer")
    print()

    # ================================================================
    #  Part I: Deep Linear Network
    # ================================================================
    print("=" * 100)
    print("  PART I: 4-layer Deep Linear Network (32x32, 500 steps)")
    print("=" * 100)

    linear_results = run_experiment(
        train_deep_linear,
        "Deep Linear",
        n_steps=N_STEPS,
        depth=4, width=32, input_dim=32, output_dim=32,
        lr=0.02, seed=42
    )

    print_table(linear_results, "Deep Linear Network")
    print_erank_analysis(linear_results, "Deep Linear")

    # ================================================================
    #  Part II: ReLU Network
    # ================================================================
    print()
    print("=" * 100)
    print("  PART II: 4-layer ReLU Network (32x32, 500 steps)")
    print("=" * 100)

    relu_results = run_experiment(
        train_relu_net,
        "ReLU",
        n_steps=N_STEPS,
        depth=4, width=32, input_dim=32, output_dim=32,
        lr=0.01, seed=42, batch_size=64
    )

    print_table(relu_results, "ReLU Network")
    print_erank_analysis(relu_results, "ReLU")

    # ================================================================
    #  Hypothesis testing
    # ================================================================
    print()
    print("=" * 100)
    print("  HYPOTHESIS TESTING")
    print("=" * 100)

    for net_type, results in [("Deep Linear", linear_results), ("ReLU", relu_results)]:
        ref = results["(b) Fixed k=5"]
        ref_loss = ref['final_loss']
        ref_matmuls = ref['total_matmuls']
        ref_pareto = ref_loss * ref_matmuls

        print(f"\n  --- {net_type} ---")

        # Test 1: Adaptive-linear vs k=5 on final loss
        adl = results["(d) Adaptive-linear"]
        loss_improvement = (ref_loss - adl['final_loss']) / ref_loss * 100
        t1 = adl['final_loss'] < ref_loss * 0.97  # >3% better
        print(f"  T1: Adaptive-linear loss < k=5 by >3%?  "
              f"{'PASS' if t1 else 'FAIL'}  "
              f"(improvement={loss_improvement:+.2f}%, "
              f"adaptive={adl['final_loss']:.6e}, k=5={ref_loss:.6e})")

        # Test 2: Adaptive schedules beat k=5 on Pareto score by >10%
        for sname in ["(d) Adaptive-linear", "(e) Adaptive-erank"]:
            s = results[sname]
            s_pareto = s['final_loss'] * s['total_matmuls']
            pareto_improvement = (ref_pareto - s_pareto) / ref_pareto * 100
            t2 = s_pareto < ref_pareto * 0.90  # >10% better
            print(f"  T2: {sname} Pareto < k=5 by >10%?  "
                  f"{'PASS' if t2 else 'FAIL'}  "
                  f"(improvement={pareto_improvement:+.2f}%, "
                  f"pareto_adaptive={s_pareto:.4e}, pareto_k5={ref_pareto:.4e})")

        # Test 3: Gradient erank decreases over training
        eranks = ref['eranks']
        er_start = np.mean(eranks[:20])
        er_end = np.mean(eranks[-20:])
        t3 = er_end < er_start
        print(f"  T3: Gradient erank decreases over training?  "
              f"{'PASS' if t3 else 'FAIL'}  "
              f"(start={er_start:.2f}, end={er_end:.2f})")

        # Test 4: Adaptive uses fewer total NS matmuls than k=5
        for sname in ["(d) Adaptive-linear", "(e) Adaptive-erank"]:
            s = results[sname]
            flop_saving = (ref_matmuls - s['total_matmuls']) / ref_matmuls * 100
            t4 = s['total_matmuls'] < ref_matmuls
            print(f"  T4: {sname} uses fewer matmuls than k=5?  "
                  f"{'PASS' if t4 else 'FAIL'}  "
                  f"(saving={flop_saving:+.1f}%, "
                  f"adaptive={s['total_matmuls']}, k=5={ref_matmuls})")

    # ================================================================
    #  Plotting
    # ================================================================
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle("Exp 2.18: Adaptive NS Steps -- k(t) Decreasing Over Training",
                 fontsize=14, fontweight='bold')

    colors = {
        "(a) Fixed k=1":       '#1f77b4',
        "(b) Fixed k=5":       '#ff7f0e',
        "(c) Fixed k=10":      '#2ca02c',
        "(d) Adaptive-linear": '#d62728',
        "(e) Adaptive-erank":  '#9467bd',
    }
    linestyles = {
        "(a) Fixed k=1":       '-',
        "(b) Fixed k=5":       '-',
        "(c) Fixed k=10":      '-',
        "(d) Adaptive-linear": '--',
        "(e) Adaptive-erank":  ':',
    }

    for col, (net_type, results) in enumerate([
        ("Deep Linear", linear_results), ("ReLU", relu_results)
    ]):
        # Row 0: Loss curves
        ax = axes[0, col]
        for name, r in results.items():
            ax.semilogy(r['losses'], color=colors[name], linestyle=linestyles[name],
                        linewidth=1.8, label=name, alpha=0.85)
        ax.set_xlabel('Training step')
        ax.set_ylabel('Loss')
        ax.set_title(f'{net_type}: Loss over training')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Row 1: Gradient effective rank
        ax = axes[1, col]
        for name, r in results.items():
            ax.plot(r['eranks'], color=colors[name], linestyle=linestyles[name],
                    linewidth=1.5, label=name, alpha=0.7)
        ax.set_xlabel('Training step')
        ax.set_ylabel('Gradient effective rank')
        ax.set_title(f'{net_type}: Gradient erank over training')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Row 2: k(t) schedule used
        ax = axes[2, col]
        for name, r in results.items():
            ax.plot(r['k_used'], color=colors[name], linestyle=linestyles[name],
                    linewidth=1.5, label=name, alpha=0.7)
        ax.set_xlabel('Training step')
        ax.set_ylabel('NS steps k(t)')
        ax.set_title(f'{net_type}: NS steps used per training step')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(SCRIPT_DIR, "adaptive_ns_steps.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved: {plot_path}")

    # ── Pareto plot ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Exp 2.18: Pareto Efficiency (Final Loss x Total NS Matmuls)",
                 fontsize=13, fontweight='bold')

    for col, (net_type, results) in enumerate([
        ("Deep Linear", linear_results), ("ReLU", relu_results)
    ]):
        ax = axes[col]
        for name, r in results.items():
            ax.scatter(r['total_matmuls'], r['final_loss'],
                       c=colors[name], s=100, zorder=5, edgecolors='black')
            ax.annotate(name.split(')')[0] + ')', (r['total_matmuls'], r['final_loss']),
                        textcoords="offset points", xytext=(8, 5), fontsize=8)

        ax.set_xlabel('Total NS matmuls')
        ax.set_ylabel('Final loss')
        ax.set_title(f'{net_type}: Pareto plot')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    pareto_path = os.path.join(SCRIPT_DIR, "adaptive_ns_pareto.png")
    plt.savefig(pareto_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Pareto plot saved: {pareto_path}")

    print("\n" + "=" * 100)
    print("  EXPERIMENT COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
