"""
Neural Network Surrogate Model with Ensemble Learning
======================================================
Contaminant Transport in Groundwater Flow.
Lorién Crespo Gracia

Ensemble methods implemented:
  1. Baseline MLP (single network, architecture search)
  2. Bagging (with out-of-bag error estimation)
  3. Deep Ensembles (independently trained networks)
  4. Residual Boosting (sequential residual learners)

Evaluation: Stratified K-Fold Cross-Validation (K=5)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 0. CONFIGURATION
# =============================================================================

from config import *
try:
    from config_local import *  # local overrides — gitignored, never committed
except ImportError:
    pass

import os
if OUTPUT_DIR:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
_out = lambda fname: os.path.join(OUTPUT_DIR, fname) if OUTPUT_DIR else fname

np.random.seed(RANDOM_SEED)


# =============================================================================
# 1. DATA LOADING
# =============================================================================

print("=" * 60)
print("Loading data...")
M = pd.read_excel(DATA_FILE, skiprows=EXCEL_SKIPROWS).values

X = M[:, :-1]   # all columns except last  (samples × features)
y = M[:, -1]    # last column              (samples,)

Q          = len(y)
n_features = X.shape[1]
print(f"Dataset: {Q} samples  |  {n_features} features")
print(f"Evaluation: {K_FOLDS}-fold stratified cross-validation")

# Bin y into K_FOLDS quantiles for stratified splitting (regression proxy)
y_bins = pd.qcut(y, q=K_FOLDS, labels=False, duplicates='drop')


# =============================================================================
# HELPERS
# =============================================================================

def make_mlp(n_neurons: int, random_state: int = 0, alpha: float = 0.0001) -> MLPRegressor:
    """Return a configured single-hidden-layer MLP matching the MATLAB baseline."""
    return MLPRegressor(
        hidden_layer_sizes=(n_neurons,),
        activation='logistic',          # logsig → logistic sigmoid
        solver='lbfgs',                 # closest deterministic quasi-Newton solver
        max_iter=2000,
        random_state=random_state,
        early_stopping=False,
        alpha=alpha,                    # L2 regularisation strength
    )


def make_inv_y(scaler):
    """Return an inverse-transform callable bound to the given StandardScaler."""
    def inv(a):
        return scaler.inverse_transform(a.reshape(-1, 1)).ravel()
    return inv


def alpha_grid_search(X_tr_s, y_tr_s, y_tr, inv_y, n_neurons,
                      random_state: int = 0, val_frac: float = 0.2) -> float:
    """Select best L2 alpha via a held-out validation split within the training fold.

    Trains a single MLP for each candidate in L2_ALPHA_GRID on the inner-train
    portion and evaluates on the inner-val portion. Returns the alpha with the
    lowest validation MSE.
    """
    n_val = max(1, int(len(y_tr) * val_frac))
    rng   = np.random.RandomState(random_state)
    perm  = rng.permutation(len(y_tr))
    val_idx, trn_idx = perm[:n_val], perm[n_val:]

    best_alpha, best_mse = L2_ALPHA_GRID[0], np.inf
    for a in L2_ALPHA_GRID:
        net = make_mlp(n_neurons, random_state=random_state, alpha=a)
        net.fit(X_tr_s[trn_idx], y_tr_s[trn_idx])
        mse = mean_squared_error(y_tr[val_idx],
                                 inv_y(net.predict(X_tr_s[val_idx])))
        if mse < best_mse:
            best_mse  = mse
            best_alpha = a
    return best_alpha


def compute_metrics(y_true, y_pred):
    """Return (MSE, Pearson-R, R²) for a pair of arrays."""
    mse = mean_squared_error(y_true, y_pred)
    r   = pearsonr(y_true, y_pred)[0]
    r2  = r2_score(y_true, y_pred)
    return mse, r, r2


def scatter_plot(y_true, y_pred, title):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidths=0.4)
    lims = [0, max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', lw=1.5, label='1:1 line')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('True Values'); ax.set_ylabel('Predicted Values')
    ax.set_title(title); ax.legend(); ax.grid(True); ax.set_aspect('equal')
    plt.tight_layout()
    return fig


# =============================================================================
# 2. STRATIFIED K-FOLD CROSS-VALIDATION
# =============================================================================

print("\n" + "=" * 60)
print(f"STRATIFIED {K_FOLDS}-FOLD CROSS-VALIDATION")
print("=" * 60)

skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)

MODEL_NAMES = ['Baseline', 'Bagging', 'Deep Ensemble', 'Residual Boosting']
COLORS      = ['#4c72b0', '#dd8452', '#55a868', '#c44e52']

# Per-fold metric storage
fold_metrics = {name: {'mse': [], 'r': [], 'r2': []} for name in MODEL_NAMES}

# Out-of-fold (OOF) predictions — combined across all folds for scatter plots
oof_true  = np.zeros(Q)
oof_preds = {name: np.zeros(Q) for name in MODEL_NAMES}

# Fold-1 artefacts preserved for the architecture plot and parameter importance
arch_plot_data = None   # (mse_train_arr, mse_test_arr, best_n)
fold1_data     = {}

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_bins)):
    fold_num = fold_idx + 1
    print(f"\n{'─' * 60}")
    print(f"  FOLD {fold_num}/{K_FOLDS}   "
          f"(train={len(train_idx)}  test={len(test_idx)})")
    print(f"{'─' * 60}")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Scale — fit on training fold only
    scaler_X  = StandardScaler()
    scaler_y  = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s  = scaler_X.transform(X_test)
    y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    inv_y     = make_inv_y(scaler_y)

    # Unique seed per fold to avoid seed collision across folds
    fold_seed = RANDOM_SEED + fold_idx * 1000

    # ── Architecture Search ───────────────────────────────────────────────────
    print(f"  Architecture search (1–{MAX_NEURONS} neurons)...")
    mse_arch_train = np.zeros(MAX_NEURONS)
    mse_arch_test  = np.zeros(MAX_NEURONS)

    for n in range(MIN_NEURONS, MAX_NEURONS + 1):
        net = make_mlp(n, random_state=fold_seed)
        net.fit(X_train_s, y_train_s)
        mse_arch_train[n - 1] = mean_squared_error(
            y_train, inv_y(net.predict(X_train_s)))
        mse_arch_test[n - 1]  = mean_squared_error(
            y_test,  inv_y(net.predict(X_test_s)))

    best_n = int(np.argmin(mse_arch_test[MIN_NEURONS - 1:])) + MIN_NEURONS
    print(f"    → Best neurons: {best_n}")

    if fold_idx == 0:
        arch_plot_data = (mse_arch_train.copy(), mse_arch_test.copy(), best_n)

    # ── Baseline ──────────────────────────────────────────────────────────────
    best_alpha_base = alpha_grid_search(
        X_train_s, y_train_s, y_train, inv_y, best_n, random_state=fold_seed)
    best_net = make_mlp(best_n, random_state=fold_seed, alpha=best_alpha_base)
    best_net.fit(X_train_s, y_train_s)
    base_pred = inv_y(best_net.predict(X_test_s))
    mse, r, r2 = compute_metrics(y_test, base_pred)
    fold_metrics['Baseline']['mse'].append(mse)
    fold_metrics['Baseline']['r'].append(r)
    fold_metrics['Baseline']['r2'].append(r2)
    oof_preds['Baseline'][test_idx] = base_pred
    print(f"  Baseline         MSE={mse:.6f}  R={r:.4f}  R²={r2:.4f}"
          f"  (alpha={best_alpha_base})")

    # Preserve fold-1 state for downstream analyses
    if fold_idx == 0:
        fold1_data = {
            'X_train_s': X_train_s, 'X_test_s': X_test_s,
            'y_train_s': y_train_s, 'y_train':  y_train,
            'y_test':    y_test,    'inv_y':     inv_y,
            'best_n':    best_n,    'fold_seed': fold_seed,
        }

    # ── Bagging ───────────────────────────────────────────────────────────────
    best_alpha_bag = alpha_grid_search(
        X_train_s, y_train_s, y_train, inv_y, best_n, random_state=fold_seed)
    bag_preds_test = np.zeros((N_ENSEMBLE, len(y_test)))
    for m in range(N_ENSEMBLE):
        boot_idx = resample(np.arange(len(y_train)), replace=True,
                            random_state=fold_seed + m)
        net_b = make_mlp(best_n, random_state=fold_seed + m,
                         alpha=best_alpha_bag)
        net_b.fit(X_train_s[boot_idx], y_train_s[boot_idx])
        bag_preds_test[m] = net_b.predict(X_test_s)
    bag_pred_test = inv_y(bag_preds_test.mean(axis=0))
    mse, r, r2 = compute_metrics(y_test, bag_pred_test)
    fold_metrics['Bagging']['mse'].append(mse)
    fold_metrics['Bagging']['r'].append(r)
    fold_metrics['Bagging']['r2'].append(r2)
    oof_preds['Bagging'][test_idx] = bag_pred_test
    print(f"  Bagging          MSE={mse:.6f}  R={r:.4f}  R²={r2:.4f}")

    # ── Deep Ensembles ────────────────────────────────────────────────────────
    best_alpha_de = alpha_grid_search(
        X_train_s, y_train_s, y_train, inv_y, best_n, random_state=fold_seed)
    de_preds_test = np.zeros((N_ENSEMBLE, len(y_test)))
    for m in range(N_ENSEMBLE):
        net_d = make_mlp(best_n, random_state=fold_seed * 100 + m,
                         alpha=best_alpha_de)
        net_d.fit(X_train_s, y_train_s)
        de_preds_test[m] = net_d.predict(X_test_s)
    de_preds_orig = np.array([inv_y(de_preds_test[m]) for m in range(N_ENSEMBLE)])
    de_pred_test  = de_preds_orig.mean(axis=0)
    de_std_test   = de_preds_orig.std(axis=0)
    mse, r, r2 = compute_metrics(y_test, de_pred_test)
    fold_metrics['Deep Ensemble']['mse'].append(mse)
    fold_metrics['Deep Ensemble']['r'].append(r)
    fold_metrics['Deep Ensemble']['r2'].append(r2)
    oof_preds['Deep Ensemble'][test_idx] = de_pred_test
    print(f"  Deep Ensemble    MSE={mse:.6f}  R={r:.4f}  R²={r2:.4f}")

    if fold_idx == 0:
        fold1_data['de_pred_test'] = de_pred_test
        fold1_data['de_std_test']  = de_std_test

    # ── Residual Boosting ─────────────────────────────────────────────────────
    best_alpha_bst = alpha_grid_search(
        X_train_s, y_train_s, y_train, inv_y, best_n, random_state=fold_seed)
    cumulative_train = np.zeros(len(y_train))
    cumulative_test  = np.zeros(len(y_test))
    residual         = y_train_s.copy()
    for stage in range(N_BOOST_STAGES):
        net_r = make_mlp(best_n, random_state=fold_seed + stage * 37,
                         alpha=best_alpha_bst)
        net_r.fit(X_train_s, residual)
        cumulative_train += net_r.predict(X_train_s)
        cumulative_test  += net_r.predict(X_test_s)
        residual          = y_train_s - cumulative_train
    bst_pred_test = inv_y(cumulative_test)
    mse, r, r2 = compute_metrics(y_test, bst_pred_test)
    fold_metrics['Residual Boosting']['mse'].append(mse)
    fold_metrics['Residual Boosting']['r'].append(r)
    fold_metrics['Residual Boosting']['r2'].append(r2)
    oof_preds['Residual Boosting'][test_idx] = bst_pred_test
    print(f"  Res. Boosting    MSE={mse:.6f}  R={r:.4f}  R²={r2:.4f}")

    oof_true[test_idx] = y_test


# =============================================================================
# 3. PARAMETER IMPORTANCE (leave-one-out, fold-1 data)
# =============================================================================

print("\n" + "=" * 60)
print("PARAMETER IMPORTANCE — Leave-One-Out  (fold-1 split)")
print("=" * 60)

X_tr1  = fold1_data['X_train_s']
X_te1  = fold1_data['X_test_s']
y_tr1  = fold1_data['y_train']
y_te1  = fold1_data['y_test']
yts1   = fold1_data['y_train_s']
inv_y1 = fold1_data['inv_y']
bn1    = fold1_data['best_n']
fs1    = fold1_data['fold_seed']

impact = np.zeros(n_features)
for p in range(n_features):
    cols     = [c for c in range(n_features) if c != p]
    net_excl = make_mlp(bn1, random_state=fs1)
    net_excl.fit(X_tr1[:, cols], yts1)
    mse_tr = mean_squared_error(y_tr1, inv_y1(net_excl.predict(X_tr1[:, cols])))
    mse_te = mean_squared_error(y_te1, inv_y1(net_excl.predict(X_te1[:, cols])))
    impact[p] = (mse_tr + mse_te) / 2
    print(f"  Excl. Param {p + 1:2d}  TrainMSE={mse_tr:.6f}  TestMSE={mse_te:.6f}")

norm_impact   = impact / impact.sum()
sorted_idx    = np.argsort(norm_impact)[::-1]
sorted_labels = [f'Param {i + 1}' for i in sorted_idx]
sorted_values = norm_impact[sorted_idx]

fig_imp, ax = plt.subplots(figsize=(max(6, n_features), 4))
bars = ax.bar(range(n_features), sorted_values, color='steelblue')
ax.set_xticks(range(n_features))
ax.set_xticklabels(sorted_labels, rotation=45, ha='right')
ax.set_xlabel('Parameters'); ax.set_ylabel('Normalized Mean Performance Impact')
ax.set_title('Parameter Importance (Leave-One-Out, Fold 1)')
ax.grid(True, axis='y')
for bar, val in zip(bars, sorted_values):
    ax.text(bar.get_x() + bar.get_width() / 2, val,
            f'{val:.2f}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()


# =============================================================================
# 4. SUMMARY — Mean ± Std across folds
# =============================================================================

print("\n" + "=" * 60)
print(f"SUMMARY — {K_FOLDS}-Fold CV Test-Set Metrics (mean ± std)")
print("=" * 60)

summary_rows = []
for name in MODEL_NAMES:
    mses = fold_metrics[name]['mse']
    rs   = fold_metrics[name]['r']
    r2s  = fold_metrics[name]['r2']
    summary_rows.append({
        'Model':    name,
        'MSE mean': np.mean(mses), 'MSE std': np.std(mses),
        'R mean':   np.mean(rs),   'R std':   np.std(rs),
        'R² mean':  np.mean(r2s),  'R² std':  np.std(r2s),
    })

summary_df = pd.DataFrame(summary_rows).set_index('Model')

# Human-readable display
display_df = pd.DataFrame({
    'MSE (mean±std)': [
        f"{r['MSE mean']:.6f} ± {r['MSE std']:.6f}" for r in summary_rows],
    'R (mean±std)':   [
        f"{r['R mean']:.4f} ± {r['R std']:.4f}"     for r in summary_rows],
    'R² (mean±std)':  [
        f"{r['R² mean']:.4f} ± {r['R² std']:.4f}"   for r in summary_rows],
}, index=[r['Model'] for r in summary_rows])
print("\n", display_df.to_string())


# =============================================================================
# 5. PLOTS
# =============================================================================

# ── Architecture search — fold 1 ─────────────────────────────────────────────
mse_at, mse_te_f1, best_n_f1 = arch_plot_data
fig_arch, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(MIN_NEURONS, MAX_NEURONS + 1),
        mse_at[MIN_NEURONS - 1:],    'ro-', label='Train MSE')
ax.plot(range(MIN_NEURONS, MAX_NEURONS + 1),
        mse_te_f1[MIN_NEURONS - 1:], 'go-', label='Test MSE')
ax.plot(best_n_f1, mse_te_f1[best_n_f1 - 1], 'k*', ms=12, label='Best')
ax.set_xlabel('Number of Neurons'); ax.set_ylabel('MSE')
ax.set_title('ANN Performance vs. Number of Neurons (Fold 1)')
ax.legend(); ax.grid(True); plt.tight_layout()

# ── OOF scatter plots (one per model, all folds combined) ────────────────────
fig_scatters = {
    name: scatter_plot(oof_true, oof_preds[name],
                       f'{name} — OOF Predictions ({K_FOLDS} folds)')
    for name in MODEL_NAMES
}

# ── Deep Ensemble uncertainty plot — fold 1 ──────────────────────────────────
order = np.argsort(fold1_data['y_test'])
fig_unc, ax = plt.subplots(figsize=(8, 4))
ax.plot(fold1_data['y_test'][order],    label='True', color='black', lw=1.5)
ax.plot(fold1_data['de_pred_test'][order], label='DE Mean', color='steelblue', lw=1.5)
ax.fill_between(range(len(order)),
                fold1_data['de_pred_test'][order] - fold1_data['de_std_test'][order],
                fold1_data['de_pred_test'][order] + fold1_data['de_std_test'][order],
                alpha=0.3, color='steelblue', label='±1 std')
ax.set_xlabel('Sorted Test Sample Index')
ax.set_ylabel('Target Value')
ax.set_title('Deep Ensembles — Predictive Uncertainty (Fold 1 Test Set)')
ax.legend(); ax.grid(True); plt.tight_layout()

# ── Box plot: test MSE distribution across folds for all models ──────────────
fig_box, ax = plt.subplots(figsize=(8, 5))
box_data = [fold_metrics[name]['mse'] for name in MODEL_NAMES]
bp = ax.boxplot(box_data, labels=MODEL_NAMES, patch_artist=True, notch=False)
for patch, color in zip(bp['boxes'], COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
# Overlay individual fold points for transparency
for i, (name, color) in enumerate(zip(MODEL_NAMES, COLORS), start=1):
    ax.scatter([i] * K_FOLDS, fold_metrics[name]['mse'],
               color=color, zorder=3, edgecolors='k', linewidths=0.5, s=40)
ax.set_ylabel('Test MSE')
ax.set_title(f'Test MSE Distribution across {K_FOLDS} Folds')
ax.grid(True, axis='y')
plt.tight_layout()

# ── Summary bar chart with error bars ────────────────────────────────────────
metrics_cfg = [
    ('MSE mean', 'MSE std', 'Test MSE', True),
    ('R mean',   'R std',   'R',        False),
    ('R² mean',  'R² std',  'R²',       False),
]
fig_sum, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (mean_col, std_col, title, lower_better) in zip(axes, metrics_cfg):
    means = summary_df[mean_col].values
    stds  = summary_df[std_col].values
    bars  = ax.bar(summary_df.index, means, yerr=stds, capsize=4,
                   color=COLORS, alpha=0.85)
    best_i = np.argmin(means) if lower_better else np.argmax(means)
    bars[best_i].set_edgecolor('black')
    bars[best_i].set_linewidth(2)
    ax.set_title(title)
    ax.set_xticklabels(summary_df.index, rotation=30, ha='right')
    ax.grid(True, axis='y')
plt.suptitle(
    f'Ensemble Comparison — {K_FOLDS}-Fold CV Test Set (mean ± std)',
    fontsize=13)
plt.tight_layout()


# =============================================================================
# 6. SAVE ALL FIGURES
# =============================================================================

figures = {
    '01_architecture_search_fold1':   fig_arch,
    '02_parameter_importance':         fig_imp,
    '03_baseline_oof_scatter':         fig_scatters['Baseline'],
    '04_bagging_oof_scatter':          fig_scatters['Bagging'],
    '05_deep_ensemble_oof_scatter':    fig_scatters['Deep Ensemble'],
    '06_boosting_oof_scatter':         fig_scatters['Residual Boosting'],
    '07_deep_ensemble_uncertainty':    fig_unc,
    '08_cv_mse_boxplot':               fig_box,
    '09_summary_comparison':           fig_sum,
}

for fname, fig in figures.items():
    fig.savefig(_out(f'{fname}.png'), dpi=150, bbox_inches='tight')

print("\nAll figures saved as PNG files.")
plt.show()


# =============================================================================
# 7. EXPORT SUMMARY TABLE
# =============================================================================

summary_df.to_csv(_out('ensemble_summary_cv.csv'))
print("Summary table saved to ensemble_summary_cv.csv")
