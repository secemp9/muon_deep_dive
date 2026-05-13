#!/usr/bin/env python3
"""
H20b: Curvature dependence of checkpoint loss-ratio compounding (toy study)
==========================================================================

This exploratory experiment compares Muon-style Newton-Schulz gradient
normalization against Frobenius-normalized SGD ("NormSGD") across four small
toy surfaces:

  1. a retained legacy quadratic with two fake 1x4 "layers" (kept only for
     continuity with the first-pass baseline),
  2. a new same-shape constant-Hessian control with two 4x4 parameter layers,
  3. a two-layer deep linear network,
  4. a two-layer ReLU MLP.

The current implementation measures only:

  1. checkpoint losses after separate learning-rate sweeps,
  2. checkpoint ratios loss_normsgd / loss_muon,
  3. fitted slopes of log(ratio) vs checkpoint.

It does NOT directly measure Hessians, activation-boundary crossings, or
cosine-to-Newton alignment.

Important caveats:
  - The legacy quadratic is degenerate for Muon-vs-NormSGD comparison because
    each fake layer gradient is a 1x4 row vector.
  - On a 1x4 row vector, Newton-Schulz equals simple Frobenius normalization up
    to numerical precision.
  - The new constant-Hessian control removes that trivial identity by giving the
    optimizers full 4x4 layer gradients on a constant-Hessian objective, but it
    is still only a toy control and does not by itself prove curvature
    causality.
"""

import time
import numpy as np

DIM = 4
NUM_STEPS = 500
NUM_SEEDS = 5
BATCH_SIZE = 64
NS_ITERS = 5
MOMENTUM = 0.9
CHECKPOINTS = [100, 200, 300, 400, 500]
LR_SELECTION_STEP = 200
MUON_LR_GRID = np.logspace(-4, -1, 12)
NORMSGD_LR_GRID = np.logspace(-3, 0, 12)
LEGACY_CONTROL_NAME = 'LegacyQuadratic'
PRIMARY_CONTROL_NAME = 'ConstantHessian'
SURFACE_ORDER = (LEGACY_CONTROL_NAME, PRIMARY_CONTROL_NAME, 'DeepLinear', 'ReLU_MLP')
PRIMARY_SURFACE_ORDER = (PRIMARY_CONTROL_NAME, 'DeepLinear', 'ReLU_MLP')


def newton_schulz(M, n_iters=NS_ITERS):
    norm = np.linalg.norm(M, 'fro')
    if norm < 1e-15:
        return M
    X = M / norm
    for _ in range(n_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


def _finite_array(values):
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def safe_mean(values):
    finite = _finite_array(values)
    return float(np.mean(finite)) if finite.size else float('nan')


def safe_std(values):
    finite = _finite_array(values)
    return float(np.std(finite)) if finite.size else float('nan')


def safe_median(values):
    finite = _finite_array(values)
    return float(np.median(finite)) if finite.size else float('nan')


def fit_log_ratio_slope(checkpoints, ratios):
    xs = []
    ys = []
    for cp, ratio in zip(checkpoints, ratios):
        if np.isfinite(ratio) and ratio > 0:
            xs.append(cp)
            ys.append(np.log(ratio))
    if len(xs) < 2:
        return float('nan')
    slope, _ = np.polyfit(xs, ys, 1)
    return float(slope)


def row_vector_identity_check(dim=DIM, n_iters=NS_ITERS):
    rng = np.random.RandomState(12345)
    g = rng.randn(1, dim)
    muon_g = newton_schulz(g, n_iters=n_iters)
    norm_g = g / max(np.linalg.norm(g, 'fro'), 1e-15)
    diff = muon_g - norm_g
    return {
        'shape': [1, dim],
        'max_abs_diff': float(np.max(np.abs(diff))),
        'frobenius_diff': float(np.linalg.norm(diff, 'fro')),
    }


def matrix_layer_nonidentity_check(dim=DIM, n_iters=NS_ITERS):
    rng = np.random.RandomState(54321)
    g = rng.randn(dim, dim)
    muon_g = newton_schulz(g, n_iters=n_iters)
    norm_g = g / max(np.linalg.norm(g, 'fro'), 1e-15)
    diff = muon_g - norm_g
    return {
        'shape': [dim, dim],
        'max_abs_diff': float(np.max(np.abs(diff))),
        'frobenius_diff': float(np.linalg.norm(diff, 'fro')),
    }


def make_spd_matrix(rng, size, eig_low=0.25, eig_high=4.0):
    Q, _ = np.linalg.qr(rng.randn(size, size))
    eigvals = np.logspace(np.log10(eig_low), np.log10(eig_high), size)
    eigvals = eigvals[rng.permutation(size)]
    H = Q @ np.diag(eigvals) @ Q.T
    H = 0.5 * (H + H.T)
    return H, eigvals


def get_surface_metadata():
    return {
        LEGACY_CONTROL_NAME: {
            'display_name': 'LegacyQuadratic (1x4 fake layers)',
            'analysis_role': 'legacy_control',
            'family': 'Legacy constant-Hessian quadratic',
            'n_params': DIM * 2,
            'layer_shapes': [(1, DIM), (1, DIM)],
            'scope_note': (
                'Retained only for continuity with the first-pass baseline. '
                'Muon and Frobenius-normalized SGD coincide on each fake 1x4 '
                'layer gradient at a fixed learning rate.'
            ),
        },
        PRIMARY_CONTROL_NAME: {
            'display_name': 'ConstantHessian (4x4x2 control)',
            'analysis_role': 'primary_control',
            'family': 'Same-shape constant-Hessian quadratic',
            'n_params': 2 * DIM * DIM,
            'layer_shapes': [(DIM, DIM), (DIM, DIM)],
            'scope_note': (
                'Primary repaired control. Uses two 4x4 parameter layers with '
                'fixed SPD 16x16 Hessian blocks, so Muon and NormSGD no longer '
                'coincide trivially on the layer gradients.'
            ),
        },
        'DeepLinear': {
            'display_name': 'DeepLinear',
            'analysis_role': 'nonlinear_reference',
            'family': 'Two-layer deep linear network',
            'n_params': 2 * DIM * DIM,
            'layer_shapes': [(DIM, DIM), (DIM, DIM)],
            'scope_note': 'Factorized linear map with nonconvex parameterization.',
        },
        'ReLU_MLP': {
            'display_name': 'ReLU_MLP',
            'analysis_role': 'nonlinear_reference',
            'family': 'Two-layer ReLU MLP',
            'n_params': 2 * DIM * DIM,
            'layer_shapes': [(DIM, DIM), (DIM, DIM)],
            'scope_note': 'Piecewise-linear nonconvex surface with activation gating.',
        },
    }


# =========================================================================
# SURFACE (a): LEGACY QUADRATIC WITH FAKE 1x4 LAYERS
# =========================================================================
class LegacyQuadraticSurface:
    def __init__(self, rng):
        M = rng.randn(DIM * 2, DIM * 2)
        self.A = M.T @ M + 0.1 * np.eye(DIM * 2)
        self.n_params = DIM * 2
        self.n_layers = 2
        self.layer_size = DIM
        eigvals = np.linalg.eigvalsh(self.A)
        self._diagnostics = {
            'hessian_min_eig': float(np.min(eigvals)),
            'hessian_max_eig': float(np.max(eigvals)),
            'hessian_condition_number': float(np.max(eigvals) / max(np.min(eigvals), 1e-15)),
        }

    def loss(self, theta):
        return float(0.5 * theta @ self.A @ theta)

    def grad_matrices(self, theta):
        g = self.A @ theta
        return [g[:DIM].reshape(1, DIM), g[DIM:].reshape(1, DIM)]

    def diagnostics(self):
        return dict(self._diagnostics)


# =========================================================================
# SURFACE (b): SAME-SHAPE CONSTANT-HESSIAN CONTROL
# =========================================================================
class ConstantHessianSurface:
    def __init__(self, rng):
        self.n_params = 2 * DIM * DIM
        self.n_layers = 2
        self.layer_size = DIM * DIM
        self.hessian_blocks = []
        self.centers = []
        layer_stats = []

        for _ in range(self.n_layers):
            H, eigvals = make_spd_matrix(rng, self.layer_size)
            self.hessian_blocks.append(H)
            self.centers.append(rng.randn(DIM, DIM) * 0.25)
            layer_stats.append({
                'min_eig': float(np.min(eigvals)),
                'max_eig': float(np.max(eigvals)),
                'condition_number': float(np.max(eigvals) / max(np.min(eigvals), 1e-15)),
            })

        self._diagnostics = {
            'block_structure': 'Two independent SPD 16x16 Hessian blocks acting on 4x4 layer parameters',
            'layer_hessian_stats': layer_stats,
        }

    def _unpack(self, theta):
        split = DIM * DIM
        return [theta[:split].reshape(DIM, DIM), theta[split:].reshape(DIM, DIM)]

    def loss(self, theta):
        total = 0.0
        for W, C, H in zip(self._unpack(theta), self.centers, self.hessian_blocks):
            delta = (W - C).reshape(-1)
            total += 0.5 * delta @ H @ delta
        return float(total / self.n_layers)

    def grad_matrices(self, theta):
        grads = []
        for W, C, H in zip(self._unpack(theta), self.centers, self.hessian_blocks):
            delta = (W - C).reshape(-1)
            grad = (H @ delta) / self.n_layers
            grads.append(grad.reshape(DIM, DIM))
        return grads

    def diagnostics(self):
        return {
            'block_structure': self._diagnostics['block_structure'],
            'layer_hessian_stats': [dict(item) for item in self._diagnostics['layer_hessian_stats']],
        }


# =========================================================================
# SURFACE (c): DEEP LINEAR
# =========================================================================
class DeepLinearSurface:
    def __init__(self, rng):
        self.X = rng.randn(DIM, BATCH_SIZE) * 0.3
        self.Y = rng.randn(DIM, BATCH_SIZE) * 0.3
        self.n_params = 2 * DIM * DIM
        self.n_layers = 2
        self.layer_size = DIM * DIM

    def _unpack(self, theta):
        split = DIM * DIM
        return [theta[:split].reshape(DIM, DIM), theta[split:].reshape(DIM, DIM)]

    def loss(self, theta):
        W1, W2 = self._unpack(theta)
        pred = W2 @ W1 @ self.X
        return float(0.5 * np.mean(np.sum((pred - self.Y) ** 2, axis=0)))

    def grad_matrices(self, theta):
        W1, W2 = self._unpack(theta)
        N = self.X.shape[1]
        a1 = W1 @ self.X
        pred = W2 @ a1
        delta = (pred - self.Y) / N
        G2 = delta @ a1.T
        G1 = (W2.T @ delta) @ self.X.T
        return [G1, G2]


# =========================================================================
# SURFACE (d): RELU MLP
# =========================================================================
class ReLUMLPSurface:
    def __init__(self, rng):
        self.X = rng.randn(DIM, BATCH_SIZE) * 0.3
        self.Y = rng.randn(DIM, BATCH_SIZE) * 0.3
        self.n_params = 2 * DIM * DIM
        self.n_layers = 2
        self.layer_size = DIM * DIM

    def _unpack(self, theta):
        split = DIM * DIM
        return [theta[:split].reshape(DIM, DIM), theta[split:].reshape(DIM, DIM)]

    def loss(self, theta):
        W1, W2 = self._unpack(theta)
        h = np.maximum(0, W1 @ self.X)
        pred = W2 @ h
        return float(0.5 * np.mean(np.sum((pred - self.Y) ** 2, axis=0)))

    def grad_matrices(self, theta):
        W1, W2 = self._unpack(theta)
        N = self.X.shape[1]
        pre = W1 @ self.X
        h = np.maximum(0, pre)
        pred = W2 @ h
        delta = (pred - self.Y) / N
        G2 = delta @ h.T
        delta_h = W2.T @ delta
        delta_h *= (pre > 0).astype(float)
        G1 = delta_h @ self.X.T
        return [G1, G2]


def make_surfaces(seed):
    return {
        LEGACY_CONTROL_NAME: LegacyQuadraticSurface(np.random.RandomState(seed)),
        PRIMARY_CONTROL_NAME: ConstantHessianSurface(np.random.RandomState(seed + 500)),
        'DeepLinear': DeepLinearSurface(np.random.RandomState(seed + 1000)),
        'ReLU_MLP': ReLUMLPSurface(np.random.RandomState(seed + 2000)),
    }


def train_optimizer(
    surface,
    theta0,
    lr,
    opt_name,
    num_steps=NUM_STEPS,
    checkpoints=CHECKPOINTS,
    momentum=MOMENTUM,
    ns_iters=NS_ITERS,
):
    """Train and return losses at checkpoints."""
    theta = theta0.copy()
    mom = [np.zeros(surface.layer_size) for _ in range(surface.n_layers)]
    checkpoint_losses = {}

    for step in range(num_steps):
        loss = surface.loss(theta)
        if not np.isfinite(loss) or loss > 1e10:
            for cp in checkpoints:
                if cp not in checkpoint_losses:
                    checkpoint_losses[cp] = float('inf')
            break

        grads = surface.grad_matrices(theta)
        updates = []
        for i, G in enumerate(grads):
            if G.ndim == 1:
                G = G.reshape(1, -1)
            prev_mom = mom[i].reshape(G.shape)
            if opt_name == 'muon':
                transformed = newton_schulz(G, n_iters=ns_iters)
            elif opt_name == 'normsgd':
                transformed = G / max(np.linalg.norm(G, 'fro'), 1e-15)
            else:
                raise ValueError(f'Unknown optimizer: {opt_name}')

            mom[i] = momentum * prev_mom + transformed
            updates.append(mom[i].ravel())

        theta = theta - lr * np.concatenate(updates)

        if step + 1 in checkpoints:
            checkpoint_losses[step + 1] = float(surface.loss(theta))

    return checkpoint_losses


def select_best_lr(
    surface,
    theta0,
    lr_grid,
    opt_name,
    lr_selection_step=LR_SELECTION_STEP,
    num_steps=NUM_STEPS,
    checkpoints=CHECKPOINTS,
    momentum=MOMENTUM,
    ns_iters=NS_ITERS,
):
    best_lr = float(lr_grid[0])
    best_loss = float('inf')
    sweep = []

    for lr in lr_grid:
        checkpoint_losses = train_optimizer(
            surface,
            theta0,
            float(lr),
            opt_name,
            num_steps=num_steps,
            checkpoints=checkpoints,
            momentum=momentum,
            ns_iters=ns_iters,
        )
        selection_loss = float(checkpoint_losses.get(lr_selection_step, float('inf')))
        sweep.append({'lr': float(lr), 'selection_loss': selection_loss})
        if np.isfinite(selection_loss) and selection_loss < best_loss:
            best_loss = selection_loss
            best_lr = float(lr)

    return best_lr, best_loss, sweep


def summarize_checkpoint_values(values_by_checkpoint):
    summary = {}
    for cp, values in values_by_checkpoint.items():
        summary[cp] = {
            'values': [float(v) for v in values],
            'mean': safe_mean(values),
            'std': safe_std(values),
            'n_finite': int(np.isfinite(np.asarray(values, dtype=float)).sum()),
        }
    return summary


def _surface_diagnostics(surface):
    if hasattr(surface, 'diagnostics'):
        return surface.diagnostics()
    return None


def run_experiment(verbose=False):
    start_time = time.time()
    seeds = [42 + i * 137 for i in range(NUM_SEEDS)]
    surface_metadata = get_surface_metadata()

    config = {
        'dim': DIM,
        'num_steps': NUM_STEPS,
        'num_seeds': NUM_SEEDS,
        'batch_size': BATCH_SIZE,
        'ns_iters': NS_ITERS,
        'momentum': MOMENTUM,
        'checkpoints': list(CHECKPOINTS),
        'lr_selection_step': LR_SELECTION_STEP,
        'lr_grids': {
            'muon': [float(x) for x in MUON_LR_GRID],
            'normsgd': [float(x) for x in NORMSGD_LR_GRID],
        },
        'seeds': list(seeds),
    }

    aggregate_ratios = {
        sname: {cp: [] for cp in CHECKPOINTS} for sname in SURFACE_ORDER
    }
    aggregate_losses = {
        sname: {
            'muon': {cp: [] for cp in CHECKPOINTS},
            'normsgd': {cp: [] for cp in CHECKPOINTS},
        }
        for sname in SURFACE_ORDER
    }
    aggregate_best_lrs = {
        sname: {'muon': [], 'normsgd': []} for sname in SURFACE_ORDER
    }
    seed_results = []

    for seed in seeds:
        surfaces = make_surfaces(seed)
        seed_record = {'seed': seed, 'surface_results': {}}

        for sname in SURFACE_ORDER:
            if verbose:
                print(f'[seed {seed}] {sname}')

            surf = surfaces[sname]
            theta0 = np.random.RandomState(seed + 3000).randn(surf.n_params) * 0.3
            initial_loss = float(surf.loss(theta0))

            best_muon_lr, best_muon_loss, muon_sweep = select_best_lr(
                surf,
                theta0,
                MUON_LR_GRID,
                'muon',
            )
            best_norm_lr, best_norm_loss, norm_sweep = select_best_lr(
                surf,
                theta0,
                NORMSGD_LR_GRID,
                'normsgd',
            )

            losses_muon = train_optimizer(surf, theta0, best_muon_lr, 'muon')
            losses_norm = train_optimizer(surf, theta0, best_norm_lr, 'normsgd')

            checkpoint_ratios = {}
            for cp in CHECKPOINTS:
                loss_muon = float(losses_muon.get(cp, float('inf')))
                loss_norm = float(losses_norm.get(cp, float('inf')))
                ratio = float(loss_norm / max(loss_muon, 1e-30))

                checkpoint_ratios[cp] = ratio
                aggregate_ratios[sname][cp].append(ratio)
                aggregate_losses[sname]['muon'][cp].append(loss_muon)
                aggregate_losses[sname]['normsgd'][cp].append(loss_norm)

            per_seed_slope = fit_log_ratio_slope(
                CHECKPOINTS,
                [checkpoint_ratios[cp] for cp in CHECKPOINTS],
            )

            aggregate_best_lrs[sname]['muon'].append(best_muon_lr)
            aggregate_best_lrs[sname]['normsgd'].append(best_norm_lr)

            seed_record['surface_results'][sname] = {
                'initial_loss': initial_loss,
                'surface_diagnostics': _surface_diagnostics(surf),
                'best_lrs': {
                    'muon': float(best_muon_lr),
                    'normsgd': float(best_norm_lr),
                },
                'lr_selection_losses': {
                    'muon': float(best_muon_loss),
                    'normsgd': float(best_norm_loss),
                },
                'lr_sweep': {
                    'muon': muon_sweep,
                    'normsgd': norm_sweep,
                },
                'checkpoint_losses': {
                    'muon': {cp: float(losses_muon.get(cp, float('inf'))) for cp in CHECKPOINTS},
                    'normsgd': {cp: float(losses_norm.get(cp, float('inf'))) for cp in CHECKPOINTS},
                },
                'checkpoint_ratios': {cp: float(checkpoint_ratios[cp]) for cp in CHECKPOINTS},
                'per_seed_slope': per_seed_slope,
            }

        seed_results.append(seed_record)

    surface_summaries = {}
    for sname in SURFACE_ORDER:
        mean_ratios = {cp: safe_mean(aggregate_ratios[sname][cp]) for cp in CHECKPOINTS}
        std_ratios = {cp: safe_std(aggregate_ratios[sname][cp]) for cp in CHECKPOINTS}
        per_seed_slopes = [
            seed_record['surface_results'][sname]['per_seed_slope']
            for seed_record in seed_results
        ]
        slope_from_mean_ratios = fit_log_ratio_slope(
            CHECKPOINTS,
            [mean_ratios[cp] for cp in CHECKPOINTS],
        )

        surface_summaries[sname] = {
            'mean_ratios': mean_ratios,
            'std_ratios': std_ratios,
            'n_finite_ratios': {
                cp: int(np.isfinite(np.asarray(aggregate_ratios[sname][cp], dtype=float)).sum())
                for cp in CHECKPOINTS
            },
            'ratio_values': {
                cp: [float(v) for v in aggregate_ratios[sname][cp]] for cp in CHECKPOINTS
            },
            'checkpoint_losses': {
                'muon': summarize_checkpoint_values(aggregate_losses[sname]['muon']),
                'normsgd': summarize_checkpoint_values(aggregate_losses[sname]['normsgd']),
            },
            'best_lrs': {
                'muon': [float(v) for v in aggregate_best_lrs[sname]['muon']],
                'normsgd': [float(v) for v in aggregate_best_lrs[sname]['normsgd']],
            },
            'best_lr_medians': {
                'muon': safe_median(aggregate_best_lrs[sname]['muon']),
                'normsgd': safe_median(aggregate_best_lrs[sname]['normsgd']),
            },
            'per_seed_slopes': [float(v) for v in per_seed_slopes],
            'per_seed_slope_mean': safe_mean(per_seed_slopes),
            'per_seed_slope_std': safe_std(per_seed_slopes),
            'slope_from_mean_ratios': slope_from_mean_ratios,
        }

    legacy_rate = surface_summaries[LEGACY_CONTROL_NAME]['slope_from_mean_ratios']
    control_rate = surface_summaries[PRIMARY_CONTROL_NAME]['slope_from_mean_ratios']
    linear_rate = surface_summaries['DeepLinear']['slope_from_mean_ratios']
    relu_rate = surface_summaries['ReLU_MLP']['slope_from_mean_ratios']

    primary_t1 = bool(relu_rate > linear_rate > control_rate)
    primary_t2 = bool(relu_rate > 2 * control_rate) if control_rate > 0 else bool(relu_rate > 0)
    legacy_t1 = bool(relu_rate > linear_rate > legacy_rate)
    legacy_t2 = bool(relu_rate > 2 * legacy_rate) if legacy_rate > 0 else bool(relu_rate > 0)

    results = {
        'experiment_id': 'H20b_COMPOUNDING_CURVATURE_DEPENDENCE',
        'title': 'Curvature dependence of checkpoint loss-ratio compounding (toy study)',
        'measured_quantities': [
            'Checkpoint losses at selected steps after separate LR sweeps',
            'Checkpoint ratios loss_normsgd / loss_muon',
            'Slope fits of log(ratio) vs checkpoint',
        ],
        'not_measured_directly': [
            'Hessian spectra or curvature tensors along the trajectory',
            'Activation-boundary crossing counts',
            'Cosine-to-Newton alignment metrics',
        ],
        'surface_order': list(SURFACE_ORDER),
        'primary_surface_order': list(PRIMARY_SURFACE_ORDER),
        'analysis_targets': {
            'legacy_control': LEGACY_CONTROL_NAME,
            'primary_control': PRIMARY_CONTROL_NAME,
        },
        'surface_metadata': surface_metadata,
        'config': config,
        'method_checks': {
            'legacy_row_vector_muon_equals_frobenius': row_vector_identity_check(),
            'matrix_layer_muon_differs_from_frobenius': matrix_layer_nonidentity_check(),
        },
        'seed_results': seed_results,
        'surface_summaries': surface_summaries,
        'tests': {
            'T1_relu_gt_deep_linear_gt_constant_hessian_mean_slope': {
                'description': (
                    'Primary heuristic ordering test on slope_from_mean_ratios only; '
                    'this is not a direct proof that curvature causes compounding.'
                ),
                'passed': primary_t1,
                'values': {
                    PRIMARY_CONTROL_NAME: control_rate,
                    'DeepLinear': linear_rate,
                    'ReLU_MLP': relu_rate,
                },
            },
            'T2_relu_gt_2x_constant_hessian_mean_slope': {
                'description': (
                    'Primary heuristic test asking whether the ReLU mean-ratio slope exceeds '
                    'twice the same-shape constant-Hessian control slope. If the control slope '
                    'is non-positive, this reduces to checking whether the ReLU slope is positive.'
                ),
                'passed': primary_t2,
                'values': {
                    PRIMARY_CONTROL_NAME: control_rate,
                    'ReLU_MLP': relu_rate,
                },
            },
            'Legacy_reference_T1_relu_gt_deep_linear_gt_legacy_quadratic_mean_slope': {
                'description': (
                    'Legacy continuity check using the degenerate 1x4 quadratic retained from '
                    'the first-pass baseline.'
                ),
                'passed': legacy_t1,
                'values': {
                    LEGACY_CONTROL_NAME: legacy_rate,
                    'DeepLinear': linear_rate,
                    'ReLU_MLP': relu_rate,
                },
            },
            'Legacy_reference_T2_relu_gt_2x_legacy_quadratic_mean_slope': {
                'description': (
                    'Legacy continuity check comparing the ReLU slope to twice the retained '
                    'degenerate legacy quadratic slope.'
                ),
                'passed': legacy_t2,
                'values': {
                    LEGACY_CONTROL_NAME: legacy_rate,
                    'ReLU_MLP': relu_rate,
                },
            },
        },
        'caveats': [
            'This code measures checkpoint loss ratios and slope summaries, not curvature directly.',
            'The retained legacy quadratic has 8 parameters and fake 1x4 layers, so Muon and Frobenius normalization coincide there at fixed LR.',
            'The repaired constant-Hessian control matches the 4x4x2 layer shape of the network branches and removes that trivial identity, but it is still only a toy quadratic control.',
            'Learning rates are tuned separately on fixed grids, so apparent advantages reflect the full protocol rather than a pure transform-at-fixed-LR comparison.',
        ],
        'runtime_seconds': float(time.time() - start_time),
    }
    return results


def format_scalar(value, digits=6):
    if isinstance(value, (bool, np.bool_)):
        return 'True' if value else 'False'
    if not np.isfinite(value):
        return 'nan'
    return f'{float(value):.{digits}f}'


def format_ratio(value):
    if not np.isfinite(value):
        return 'nan'
    return f'{float(value):.2f}x'


def print_summary(results):
    config = results['config']
    legacy_check = results['method_checks']['legacy_row_vector_muon_equals_frobenius']
    matrix_check = results['method_checks']['matrix_layer_muon_differs_from_frobenius']
    surface_metadata = results['surface_metadata']

    print('=' * 110)
    print('H20b: checkpoint loss-ratio compounding across toy surfaces')
    print('=' * 110)
    print('Scope: exploratory comparison of Muon vs Frobenius-normalized SGD on toy controls and nonlinear references.')
    print('Measured quantities:')
    for item in results['measured_quantities']:
        print(f'  - {item}')
    print('Not measured directly:')
    for item in results['not_measured_directly']:
        print(f'  - {item}')
    print()
    print(
        f"Config: dim={config['dim']}, steps={config['num_steps']}, seeds={config['num_seeds']}, "
        f"batch={config['batch_size']}, momentum={config['momentum']}, ns_iters={config['ns_iters']}"
    )
    print(f"Checkpoints: {config['checkpoints']}; LR selection step: {config['lr_selection_step']}")
    print(f"Seeds: {config['seeds']}")
    print()
    print('Surface inventory:')
    for sname in results['surface_order']:
        meta = surface_metadata[sname]
        print(
            f"  - {meta['display_name']:<34} | role={meta['analysis_role']:<18} | "
            f"params={meta['n_params']:<2} | layers={meta['layer_shapes']}"
        )
    print()
    print('Checkpoint ratio summary (NormSGD loss / Muon loss; >1 favors Muon):')
    header = ['Surface'] + [f'T={cp}' for cp in config['checkpoints']] + ['slope(mean)']
    print('  ' + ' | '.join(f'{h:>14}' for h in header))
    print('  ' + '-' * (17 * len(header)))
    for sname in results['surface_order']:
        summary = results['surface_summaries'][sname]
        row = [f'{surface_metadata[sname]["display_name"]:>14}']
        row.extend(f'{format_ratio(summary["mean_ratios"][cp]):>14}' for cp in config['checkpoints'])
        row.append(f'{format_scalar(summary["slope_from_mean_ratios"], digits=6):>14}')
        print('  ' + ' | '.join(row))
    print()
    print('Slope summary (log-ratio vs checkpoint):')
    print('  ' + ' | '.join(f'{h:>20}' for h in ['Surface', 'mean-ratio slope', 'per-seed slope mean', 'per-seed slope std']))
    print('  ' + '-' * 92)
    for sname in results['surface_order']:
        summary = results['surface_summaries'][sname]
        print(
            '  ' + ' | '.join([
                f'{surface_metadata[sname]["display_name"]:>20}',
                f'{format_scalar(summary["slope_from_mean_ratios"], digits=6):>20}',
                f'{format_scalar(summary["per_seed_slope_mean"], digits=6):>20}',
                f'{format_scalar(summary["per_seed_slope_std"], digits=6):>20}',
            ])
        )
    print()
    print('Primary heuristic tests (same-shape constant-Hessian control):')
    for tname in [
        'T1_relu_gt_deep_linear_gt_constant_hessian_mean_slope',
        'T2_relu_gt_2x_constant_hessian_mean_slope',
    ]:
        payload = results['tests'][tname]
        status = 'PASS' if payload['passed'] else 'FAIL'
        print(f'  - {tname}: {status}')
        print(f'      {payload["description"]}')
        print(f'      values={payload["values"]}')
    print()
    print('Legacy continuity checks (degenerate 1x4 control retained only for comparison):')
    for tname in [
        'Legacy_reference_T1_relu_gt_deep_linear_gt_legacy_quadratic_mean_slope',
        'Legacy_reference_T2_relu_gt_2x_legacy_quadratic_mean_slope',
    ]:
        payload = results['tests'][tname]
        status = 'PASS' if payload['passed'] else 'FAIL'
        print(f'  - {tname}: {status}')
        print(f'      {payload["description"]}')
        print(f'      values={payload["values"]}')
    print()
    print('Method checks:')
    print(
        '  Legacy row-vector identity -> '
        f"shape={legacy_check['shape']}, max_abs_diff={legacy_check['max_abs_diff']:.3e}, "
        f"frobenius_diff={legacy_check['frobenius_diff']:.3e}"
    )
    print(
        '  Generic 4x4 matrix non-identity -> '
        f"shape={matrix_check['shape']}, max_abs_diff={matrix_check['max_abs_diff']:.3e}, "
        f"frobenius_diff={matrix_check['frobenius_diff']:.3e}"
    )
    print('  This confirms that the repaired control removes the legacy trivial Muon==NormSGD identity on 1x4 gradients.')
    print()
    print(f"Runtime: {results['runtime_seconds']:.2f} s")
    print('=' * 110)


def main():
    results = run_experiment(verbose=False)
    print_summary(results)


if __name__ == '__main__':
    main()
