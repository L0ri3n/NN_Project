"""
Neural Network Surrogate Model with Ensemble Learning
======================================================
Contaminant Transport in Groundwater Flow.
Lorién Crespo Gracia

Ensemble methods implemented:
  1. Baseline MLP (single network, architecture search)
  2. Bagging — Bootstrap (with replacement, OOB error estimation)
  2b. Bagging — Subsampling (65 % without replacement, OOB error estimation)
  3. Deep Ensembles (independently trained networks)
  4. Residual Boosting (sequential residual learners)

Evaluation: Stratified K-Fold Cross-Validation (K=5)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, gaussian_kde, skew, gumbel_r, lognorm, kstest, probplot
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

# ── Log10 preprocessing ───────────────────────────────────────────────────────
# Param 1 (index 0) and the target are right-skewed; log10 compresses the
# dynamic range and helps the MLP learn a more linear mapping.
# y_orig retains the untransformed target so that metrics (MSE, R²) are always
# reported in the original physical units after applying the inverse transform.
y_orig  = y.copy()
X_raw   = X.copy()           # raw features before log10 — used for distribution plots
X[:, 0] = np.log10(X[:, 0])
y       = np.log10(y)

Q          = len(y)
n_features = X.shape[1]
print(f"Dataset: {Q} samples  |  {n_features} features")
print("  Preprocessing: log10 applied to Param 1 and Target")
print(f"  Sample weighting: {SAMPLE_WEIGHT_SCHEME}")
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
    """Return an inverse-transform callable: StandardScaler^-1 then 10^x.

    The full forward chain is  y_orig → log10 → StandardScaler → y_s.
    The inverse chain is therefore  y_s → StandardScaler^-1 → 10^x → y_orig.
    Predictions returned by this callable are in the original physical units,
    so MSE and R² computed against y_orig are directly interpretable.
    """
    def inv(a):
        log_vals = scaler.inverse_transform(a.reshape(-1, 1)).ravel()
        return 10 ** log_vals
    return inv


def compute_sample_weights(y_train_orig):
    """Return per-sample training weights that emphasise high target values.

    Returns None when SAMPLE_WEIGHT_SCHEME == 'none'.
    Otherwise returns an array of positive weights normalised to mean = 1,
    which preserves average gradient magnitude so learning dynamics are not
    thrown off relative to the unweighted baseline.

    'rank'       — weight proportional to rank (1 = lowest, n = highest);
                   robust to outliers in the y scale.
    'log_value'  — weight proportional to log10(y) shifted to be strictly
                   positive; tied directly to the log10 preprocessing space.
    """
    if SAMPLE_WEIGHT_SCHEME == 'none':
        return None
    if SAMPLE_WEIGHT_SCHEME == 'rank':
        w = np.argsort(np.argsort(y_train_orig)).astype(float) + 1.0
    elif SAMPLE_WEIGHT_SCHEME == 'log_value':
        lv = np.log10(y_train_orig)
        w  = lv - lv.min() + 1.0
    return w / w.mean()


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
# 1b. INPUT DATA DISTRIBUTION & CLUSTERING ANALYSIS
# =============================================================================

print("\n" + "=" * 60)
print("INPUT DATA EXPLORATION")
print("=" * 60)

# ── Raw feature & target distributions (before log10) — 2×2 grid ─────────────
_param_labels = [f'Param {i + 1}' for i in range(n_features)] + ['Target']
_raw_data     = [X_raw[:, i] for i in range(n_features)] + [y_orig]
_raw_colors   = ['steelblue'] * n_features + ['tomato']

fig_raw_dist, _axr = plt.subplots(2, 2, figsize=(9, 7))
_axr = _axr.ravel()
for _i, (_vals, _lbl, _col) in enumerate(zip(_raw_data, _param_labels, _raw_colors)):
    _ax = _axr[_i]
    _ax.hist(_vals, bins=15, density=True, alpha=0.5,
             color=_col, edgecolor='white')
    _kde_x = np.linspace(_vals.min(), _vals.max(), 200)
    _ax.plot(_kde_x, gaussian_kde(_vals)(_kde_x), 'k-', lw=1.5)
    _ax.set_title(_lbl, fontsize=9)
    _ax.set_xlabel('Value (original units)', fontsize=7)
    _ax.set_ylabel('Density', fontsize=7)
    _ax.tick_params(labelsize=7)
    _ax.grid(True, alpha=0.3)
for _j in range(len(_raw_data), 4):
    _axr[_j].set_visible(False)
fig_raw_dist.suptitle('Raw Distributions — Before Preprocessing', fontsize=12)
plt.tight_layout()

# ── Feature & target distributions ───────────────────────────────────────────
n_cols_d = min(4, n_features + 1)
n_rows_d = int(np.ceil((n_features + 1) / n_cols_d))
fig_dist, axes_dist = plt.subplots(n_rows_d, n_cols_d,
                                   figsize=(n_cols_d * 3.5, n_rows_d * 2.8),
                                   squeeze=False)
axes_dist = axes_dist.ravel()

for i in range(n_features):
    ax = axes_dist[i]
    vals = X[:, i]
    ax.hist(vals, bins=15, density=True, alpha=0.5,
            color='steelblue', edgecolor='white')
    kde_x = np.linspace(vals.min(), vals.max(), 200)
    ax.plot(kde_x, gaussian_kde(vals)(kde_x), 'k-', lw=1.5)
    ax.set_title(f'Param {i + 1}', fontsize=9)
    ax.set_xlabel('Value', fontsize=7)
    ax.set_ylabel('Density', fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)

ax = axes_dist[n_features]
ax.hist(y, bins=15, density=True, alpha=0.5, color='tomato', edgecolor='white')
kde_ty = np.linspace(y.min(), y.max(), 200)
ax.plot(kde_ty, gaussian_kde(y)(kde_ty), 'k-', lw=1.5)
ax.set_title('Target', fontsize=9)
ax.set_xlabel('Value', fontsize=7)
ax.set_ylabel('Density', fontsize=7)
ax.tick_params(labelsize=7)
ax.grid(True, alpha=0.3)

for j in range(n_features + 1, len(axes_dist)):
    axes_dist[j].set_visible(False)

fig_dist.suptitle('Input Feature & Target Distributions', fontsize=12)
plt.tight_layout()

# Print basic stats
print(f"\n  {'':12s}{'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10} {'Skew':>8}")
for i in range(n_features):
    v = X[:, i]
    print(f"  Param {i + 1:>5d}   {v.min():10.4f} {v.max():10.4f}"
          f" {v.mean():10.4f} {v.std():10.4f} {skew(v):8.3f}")
v = y
print(f"  {'Target':>12s}  {v.min():10.4f} {v.max():10.4f}"
      f" {v.mean():10.4f} {v.std():10.4f} {skew(v):8.3f}")

# ── Pearson correlation heatmap ───────────────────────────────────────────────
labels_corr = [f'P{i + 1}' for i in range(n_features)] + ['Target']
corr_matrix = np.corrcoef(M.T)
fig_corr, ax = plt.subplots(figsize=(max(5, n_features + 2),
                                     max(4, n_features + 1)))
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, label='Pearson r')
ax.set_xticks(range(len(labels_corr)))
ax.set_yticks(range(len(labels_corr)))
ax.set_xticklabels(labels_corr, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(labels_corr, fontsize=8)
for row in range(len(labels_corr)):
    for col in range(len(labels_corr)):
        val = corr_matrix[row, col]
        ax.text(col, row, f'{val:.2f}', ha='center', va='center',
                fontsize=7, color='white' if abs(val) > 0.6 else 'black')
ax.set_title('Pearson Correlation Heatmap (Features + Target)')
plt.tight_layout()

# ── K-Means clustering ────────────────────────────────────────────────────────
print("\n  K-Means clustering (silhouette & elbow):")
scaler_clust = StandardScaler()
X_scaled_c   = scaler_clust.fit_transform(X)

max_k      = min(8, Q // 5)
k_range    = range(2, max_k + 1)
sil_scores = []
inertias   = []

for k in k_range:
    km       = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    labels_k = km.fit_predict(X_scaled_c)
    sil_scores.append(silhouette_score(X_scaled_c, labels_k))
    inertias.append(km.inertia_)
    print(f"    k={k}  Silhouette={sil_scores[-1]:.4f}  Inertia={inertias[-1]:.2f}")

best_k      = list(k_range)[int(np.argmax(sil_scores))]
km_best     = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
best_labels = km_best.fit_predict(X_scaled_c)
print(f"\n  Best k (silhouette) = {best_k}")

for c in range(best_k):
    mask = best_labels == c
    print(f"    Cluster {c}: {mask.sum():3d} samples  "
          f"y_mean={y[mask].mean():.4f}  y_std={y[mask].std():.4f}")

# Silhouette + elbow side-by-side
fig_clust, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(list(k_range), sil_scores, 'o-', color='steelblue')
ax1.axvline(best_k, color='red', ls='--', label=f'Best k={best_k}')
ax1.set_xlabel('k'); ax1.set_ylabel('Silhouette Score')
ax1.set_title('Silhouette Score vs k'); ax1.legend(); ax1.grid(True)

ax2.plot(list(k_range), inertias, 's-', color='tomato')
ax2.set_xlabel('k'); ax2.set_ylabel('Inertia (within-cluster SSE)')
ax2.set_title('Elbow Curve'); ax2.grid(True)

plt.suptitle('K-Means Clustering Analysis', fontsize=12)
plt.tight_layout()

# PCA 2D projection coloured by cluster label
pca    = PCA(n_components=2, random_state=RANDOM_SEED)
X_pca  = pca.fit_transform(X_scaled_c)

fig_pca, ax = plt.subplots(figsize=(6, 5))
for c in range(best_k):
    mask = best_labels == c
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Cluster {c}',
               alpha=0.75, edgecolors='k', linewidths=0.3, s=55)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var.)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var.)')
ax.set_title(f'PCA Projection — K-Means (k={best_k})')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()


# =============================================================================
# 1c. DISTRIBUTION DIAGNOSTICS — Gumbel vs Log-Normal
# =============================================================================
# These plots let you visually confirm which distribution better describes the
# target variable, so you can choose PREPROCESS_TARGET in config.py accordingly.
#
# Gumbel probability paper: if points follow a straight line, the target is
#   well-described by a Gumbel distribution.
# Log-normal probability paper: if log10(target) plots linearly against normal
#   quantiles, the target is well-described by a log-normal distribution.
# PDF overlay + KS test give a quantitative comparison.

print("\n" + "=" * 60)
print("DISTRIBUTION DIAGNOSTICS — Gumbel vs Log-Normal")
print("=" * 60)

_gfit_loc, _gfit_scale   = gumbel_r.fit(y_orig)
_ln_shape, _ln_loc, _ln_scale = lognorm.fit(y_orig, floc=0)

_ks_g  = kstest(y_orig, 'gumbel_r', args=(_gfit_loc, _gfit_scale))
_ks_ln = kstest(y_orig, 'lognorm',  args=(_ln_shape, _ln_loc, _ln_scale))
print(f"  KS test — Gumbel:     D={_ks_g.statistic:.4f}   p={_ks_g.pvalue:.4f}")
print(f"  KS test — Log-Normal: D={_ks_ln.statistic:.4f}   p={_ks_ln.pvalue:.4f}")
print(f"  Better fit (lower KS D): {'Gumbel' if _ks_g.statistic < _ks_ln.statistic else 'Log-Normal'}")

# ── PDF overlay — isolated figure ────────────────────────────────────────────
fig_pdf, _ax_pdf = plt.subplots(figsize=(5, 4))
_yr = np.linspace(y_orig.min() * 0.9, y_orig.max() * 1.1, 300)
_ax_pdf.hist(y_orig, bins=15, density=True, alpha=0.4,
             color='steelblue', edgecolor='white', label='Data')
_ax_pdf.plot(_yr, gumbel_r.pdf(_yr, loc=_gfit_loc, scale=_gfit_scale),
             'r-', lw=2, label=f'Gumbel  (KS D={_ks_g.statistic:.3f})')
_ax_pdf.plot(_yr, lognorm.pdf(_yr, _ln_shape, _ln_loc, _ln_scale),
             'g-', lw=2, label=f'Log-Normal  (KS D={_ks_ln.statistic:.3f})')
_ax_pdf.set_xlabel('Target (original units)')
_ax_pdf.set_ylabel('Density')
_ax_pdf.set_title('PDF Overlay — Target Variable')
_ax_pdf.legend(fontsize=8); _ax_pdf.grid(True, alpha=0.3)
plt.tight_layout()

fig_dist_diag, _axd = plt.subplots(1, 3, figsize=(15, 5))

# ── Panel 1: Normalization-only Q-Q (no log transform) ───────────────────────
# Shows that StandardScaler alone does not produce a normal distribution.
_ax = _axd[0]
(_osm_raw, _osr_raw), (_sl_raw, _ic_raw, _r_raw) = probplot(y_orig, dist='norm', fit=True)
_ax.scatter(_osm_raw, _osr_raw, alpha=0.6, s=18, color='steelblue',
            edgecolors='k', linewidths=0.3, label='Data (no log)')
_ax.plot(_osm_raw, _sl_raw * np.array(_osm_raw) + _ic_raw, 'r-', lw=2,
         label=f'Normal fit  R\u00b2={_r_raw**2:.4f}')
_ax.set_xlabel('Theoretical Normal Quantiles')
_ax.set_ylabel('Target (original units)')
_ax.set_title('Q-Q — Normalisation Only')
_ax.legend(fontsize=8); _ax.grid(True, alpha=0.3)

# ── Panel 2: Gumbel probability paper ────────────────────────────────────────
# Points lie on a straight line when data follows a Gumbel distribution.
_ax = _axd[1]
_n   = len(y_orig)
_sy  = np.sort(y_orig)
_p_g = (np.arange(1, _n + 1) - 0.44) / (_n + 0.12)   # Gringorten plotting positions
_rv  = -np.log(-np.log(_p_g))                          # Gumbel reduced variate
_ax.scatter(_rv, _sy, alpha=0.6, s=18, color='steelblue',
            edgecolors='k', linewidths=0.3, label='Data')
_zl = np.linspace(_rv.min() - 0.5, _rv.max() + 0.5, 100)
_ax.plot(_zl, _gfit_loc + _gfit_scale * _zl, 'r-', lw=2,
         label=f'Gumbel line  (KS D={_ks_g.statistic:.3f})')
_ax.set_xlabel('Gumbel Reduced Variate  −ln(−ln(p))')
_ax.set_ylabel('Target (original units)')
_ax.set_title('Gumbel Probability Paper')
_ax.legend(fontsize=8); _ax.grid(True, alpha=0.3)

# ── Panel 3: Log-normal probability paper ────────────────────────────────────
# Points lie on a straight line when log10(data) is normally distributed.
_ax = _axd[2]
(_osm, _osr), (_sl, _ic, _r) = probplot(np.log10(y_orig), dist='norm', fit=True)
_ax.scatter(_osm, _osr, alpha=0.6, s=18, color='steelblue',
            edgecolors='k', linewidths=0.3, label='log\u2081\u2080(data)')
_ax.plot(_osm, _sl * np.array(_osm) + _ic, 'g-', lw=2,
         label=f'Normal fit  R\u00b2={_r**2:.4f}')
_ax.set_xlabel('Theoretical Normal Quantiles')
_ax.set_ylabel('log\u2081\u2080(Target)')
_ax.set_title('Log-Normal Probability Paper')
_ax.legend(fontsize=8); _ax.grid(True, alpha=0.3)

fig_dist_diag.suptitle(
    'Target Distribution Diagnostics  —  Preprocessing: log10',
    fontsize=12)
plt.tight_layout()

# ── K_h (Param 1) distribution diagnostics ───────────────────────────────────
_kh = X_raw[:, 0]
_kh_gfit_loc, _kh_gfit_scale     = gumbel_r.fit(_kh)
_kh_ln_shape, _kh_ln_loc, _kh_ln_scale = lognorm.fit(_kh, floc=0)
_kh_ks_g  = kstest(_kh, 'gumbel_r', args=(_kh_gfit_loc, _kh_gfit_scale))
_kh_ks_ln = kstest(_kh, 'lognorm',  args=(_kh_ln_shape, _kh_ln_loc, _kh_ln_scale))
print(f"\nDISTRIBUTION DIAGNOSTICS — K_h (Param 1)")
print(f"  KS test — Gumbel:     D={_kh_ks_g.statistic:.4f}   p={_kh_ks_g.pvalue:.4f}")
print(f"  KS test — Log-Normal: D={_kh_ks_ln.statistic:.4f}   p={_kh_ks_ln.pvalue:.4f}")
print(f"  Better fit (lower KS D): {'Gumbel' if _kh_ks_g.statistic < _kh_ks_ln.statistic else 'Log-Normal'}")

fig_kh_diag, _axk = plt.subplots(1, 3, figsize=(15, 5))

_ax = _axk[0]
(_osm_kh, _osr_kh), (_sl_kh, _ic_kh, _r_kh) = probplot(_kh, dist='norm', fit=True)
_ax.scatter(_osm_kh, _osr_kh, alpha=0.6, s=18, color='steelblue',
            edgecolors='k', linewidths=0.3, label='Data (no log)')
_ax.plot(_osm_kh, _sl_kh * np.array(_osm_kh) + _ic_kh, 'r-', lw=2,
         label=f'Normal fit  R\u00b2={_r_kh**2:.4f}')
_ax.set_xlabel('Theoretical Normal Quantiles')
_ax.set_ylabel('$K_h$ (m$^2$, original units)')
_ax.set_title('Q-Q — Normalisation Only')
_ax.legend(fontsize=8); _ax.grid(True, alpha=0.3)

_ax = _axk[1]
_n_kh  = len(_kh)
_sy_kh = np.sort(_kh)
_p_kh  = (np.arange(1, _n_kh + 1) - 0.44) / (_n_kh + 0.12)
_rv_kh = -np.log(-np.log(_p_kh))
_ax.scatter(_rv_kh, _sy_kh, alpha=0.6, s=18, color='steelblue',
            edgecolors='k', linewidths=0.3, label='Data')
_zl_kh = np.linspace(_rv_kh.min() - 0.5, _rv_kh.max() + 0.5, 100)
_ax.plot(_zl_kh, _kh_gfit_loc + _kh_gfit_scale * _zl_kh, 'r-', lw=2,
         label=f'Gumbel line  (KS D={_kh_ks_g.statistic:.3f})')
_ax.set_xlabel('Gumbel Reduced Variate  −ln(−ln(p))')
_ax.set_ylabel('$K_h$ (m$^2$, original units)')
_ax.set_title('Gumbel Probability Paper')
_ax.legend(fontsize=8); _ax.grid(True, alpha=0.3)

_ax = _axk[2]
(_osm_khl, _osr_khl), (_sl_khl, _ic_khl, _r_khl) = probplot(np.log10(_kh), dist='norm', fit=True)
_ax.scatter(_osm_khl, _osr_khl, alpha=0.6, s=18, color='steelblue',
            edgecolors='k', linewidths=0.3, label='log\u2081\u2080($K_h$)')
_ax.plot(_osm_khl, _sl_khl * np.array(_osm_khl) + _ic_khl, 'g-', lw=2,
         label=f'Normal fit  R\u00b2={_r_khl**2:.4f}')
_ax.set_xlabel('Theoretical Normal Quantiles')
_ax.set_ylabel('log\u2081\u2080($K_h$)')
_ax.set_title('Log-Normal Probability Paper')
_ax.legend(fontsize=8); _ax.grid(True, alpha=0.3)

fig_kh_diag.suptitle(
    '$K_h$ Distribution Diagnostics  —  Preprocessing: log10',
    fontsize=12)
plt.tight_layout()


# =============================================================================
# 2. STRATIFIED K-FOLD CROSS-VALIDATION
# =============================================================================

print("\n" + "=" * 60)
print(f"STRATIFIED {K_FOLDS}-FOLD CROSS-VALIDATION")
print("=" * 60)

skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)

MODEL_NAMES = ['Baseline', 'Bagging', 'Bagging (Subsamp)',
               'Deep Ensemble', 'Residual Boosting']
COLORS      = ['#4c72b0', '#dd8452', '#8172b2', '#55a868', '#c44e52']

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

    X_train, X_test         = X[train_idx], X[test_idx]
    y_train, y_test         = y[train_idx], y[test_idx]        # log10-space
    y_orig_train, y_orig_test = y_orig[train_idx], y_orig[test_idx]  # original units

    # Scale — fit on training fold only
    scaler_X  = StandardScaler()
    scaler_y  = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s  = scaler_X.transform(X_test)
    y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    inv_y     = make_inv_y(scaler_y)
    sw        = compute_sample_weights(y_orig_train)

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
            y_orig_train, inv_y(net.predict(X_train_s)))
        mse_arch_test[n - 1]  = mean_squared_error(
            y_orig_test,  inv_y(net.predict(X_test_s)))

    best_n = int(np.argmin(mse_arch_test[MIN_NEURONS - 1:])) + MIN_NEURONS
    print(f"    → Best neurons: {best_n}")

    if fold_idx == 0:
        arch_plot_data = (mse_arch_train.copy(), mse_arch_test.copy(), best_n)

    # ── Baseline ──────────────────────────────────────────────────────────────
    best_alpha_base = alpha_grid_search(
        X_train_s, y_train_s, y_orig_train, inv_y, best_n, random_state=fold_seed)
    best_net = make_mlp(best_n, random_state=fold_seed, alpha=best_alpha_base)
    best_net.fit(X_train_s, y_train_s, sample_weight=sw)
    base_pred = inv_y(best_net.predict(X_test_s))
    mse, r, r2 = compute_metrics(y_orig_test, base_pred)
    fold_metrics['Baseline']['mse'].append(mse)
    fold_metrics['Baseline']['r'].append(r)
    fold_metrics['Baseline']['r2'].append(r2)
    oof_preds['Baseline'][test_idx] = base_pred
    print(f"  Baseline         MSE={mse:.6f}  R={r:.4f}  R²={r2:.4f}"
          f"  (alpha={best_alpha_base})")

    # Preserve fold-1 state for downstream analyses
    if fold_idx == 0:
        fold1_data = {
            'X_train_s': X_train_s,    'X_test_s':  X_test_s,
            'y_train_s': y_train_s,    'y_train':   y_orig_train,  # original units
            'y_test':    y_orig_test,  'inv_y':     inv_y,         # original units
            'best_n':    best_n,       'fold_seed': fold_seed,
            'train_idx': train_idx,    'test_idx':  test_idx,
        }

    # ── Bagging — Bootstrap (with replacement) ────────────────────────────────
    best_alpha_bag = alpha_grid_search(
        X_train_s, y_train_s, y_orig_train, inv_y, best_n, random_state=fold_seed)
    bag_preds_test = np.zeros((N_ENSEMBLE, len(y_test)))
    oob_sum_bag    = np.zeros(len(y_train))
    oob_cnt_bag    = np.zeros(len(y_train))
    for m in range(N_ENSEMBLE):
        boot_idx = resample(np.arange(len(y_train)), replace=True,
                            random_state=fold_seed + m)
        oob_mask = np.ones(len(y_train), dtype=bool)
        oob_mask[np.unique(boot_idx)] = False
        net_b = make_mlp(best_n, random_state=fold_seed + m,
                         alpha=best_alpha_bag)
        sw_boot = sw[boot_idx] if sw is not None else None
        net_b.fit(X_train_s[boot_idx], y_train_s[boot_idx], sample_weight=sw_boot)
        if oob_mask.any():
            oob_sum_bag[oob_mask] += net_b.predict(X_train_s[oob_mask])
            oob_cnt_bag[oob_mask] += 1
        bag_preds_test[m] = net_b.predict(X_test_s)
    bag_pred_test = inv_y(bag_preds_test.mean(axis=0))
    valid_oob_bag = oob_cnt_bag > 0
    oob_pred_bag  = np.where(valid_oob_bag,
                             oob_sum_bag / np.where(oob_cnt_bag > 0, oob_cnt_bag, 1), 0)
    oob_mse_bag   = mean_squared_error(
        y_orig_train[valid_oob_bag], inv_y(oob_pred_bag[valid_oob_bag]))
    mse, r, r2 = compute_metrics(y_orig_test, bag_pred_test)
    fold_metrics['Bagging']['mse'].append(mse)
    fold_metrics['Bagging']['r'].append(r)
    fold_metrics['Bagging']['r2'].append(r2)
    oof_preds['Bagging'][test_idx] = bag_pred_test
    print(f"  Bagging (Boot)   MSE={mse:.6f}  R={r:.4f}  R²={r2:.4f}"
          f"  (alpha={best_alpha_bag})"
          f"  OOB-MSE={oob_mse_bag:.6f}  OOB-cov={valid_oob_bag.mean():.0%}")

    # ── Bagging — Subsampling (without replacement, BAG_SUBSAMPLE_FRAC) ───────
    best_alpha_sub = alpha_grid_search(
        X_train_s, y_train_s, y_orig_train, inv_y, best_n, random_state=fold_seed)
    sub_size       = max(1, int(BAG_SUBSAMPLE_FRAC * len(y_train)))
    sub_preds_test = np.zeros((N_ENSEMBLE, len(y_test)))
    oob_sum_sub    = np.zeros(len(y_train))
    oob_cnt_sub    = np.zeros(len(y_train))
    for m in range(N_ENSEMBLE):
        rng_sub  = np.random.RandomState(fold_seed + m)
        sub_idx  = rng_sub.choice(len(y_train), size=sub_size, replace=False)
        oob_mask = np.ones(len(y_train), dtype=bool)
        oob_mask[sub_idx] = False
        net_s = make_mlp(best_n, random_state=fold_seed + m,
                         alpha=best_alpha_sub)
        sw_sub = sw[sub_idx] if sw is not None else None
        net_s.fit(X_train_s[sub_idx], y_train_s[sub_idx], sample_weight=sw_sub)
        if oob_mask.any():
            oob_sum_sub[oob_mask] += net_s.predict(X_train_s[oob_mask])
            oob_cnt_sub[oob_mask] += 1
        sub_preds_test[m] = net_s.predict(X_test_s)
    sub_pred_test = inv_y(sub_preds_test.mean(axis=0))
    valid_oob_sub = oob_cnt_sub > 0
    oob_pred_sub  = np.where(valid_oob_sub,
                             oob_sum_sub / np.where(oob_cnt_sub > 0, oob_cnt_sub, 1), 0)
    oob_mse_sub   = mean_squared_error(
        y_orig_train[valid_oob_sub], inv_y(oob_pred_sub[valid_oob_sub]))
    mse, r, r2 = compute_metrics(y_orig_test, sub_pred_test)
    fold_metrics['Bagging (Subsamp)']['mse'].append(mse)
    fold_metrics['Bagging (Subsamp)']['r'].append(r)
    fold_metrics['Bagging (Subsamp)']['r2'].append(r2)
    oof_preds['Bagging (Subsamp)'][test_idx] = sub_pred_test
    print(f"  Bag. (Subsamp)   MSE={mse:.6f}  R={r:.4f}  R²={r2:.4f}"
          f"  (alpha={best_alpha_sub})"
          f"  OOB-MSE={oob_mse_sub:.6f}  OOB-cov={valid_oob_sub.mean():.0%}")

    # ── Deep Ensembles ────────────────────────────────────────────────────────
    best_alpha_de = alpha_grid_search(
        X_train_s, y_train_s, y_orig_train, inv_y, best_n, random_state=fold_seed)
    de_preds_test = np.zeros((N_ENSEMBLE, len(y_test)))
    for m in range(N_ENSEMBLE):
        net_d = make_mlp(best_n, random_state=fold_seed * 100 + m,
                         alpha=best_alpha_de)
        net_d.fit(X_train_s, y_train_s, sample_weight=sw)
        de_preds_test[m] = net_d.predict(X_test_s)
    de_preds_orig = np.array([inv_y(de_preds_test[m]) for m in range(N_ENSEMBLE)])
    de_pred_test  = de_preds_orig.mean(axis=0)
    de_std_test   = de_preds_orig.std(axis=0)
    mse, r, r2 = compute_metrics(y_orig_test, de_pred_test)
    fold_metrics['Deep Ensemble']['mse'].append(mse)
    fold_metrics['Deep Ensemble']['r'].append(r)
    fold_metrics['Deep Ensemble']['r2'].append(r2)
    oof_preds['Deep Ensemble'][test_idx] = de_pred_test
    print(f"  Deep Ensemble    MSE={mse:.6f}  R={r:.4f}  R²={r2:.4f}"
          f"  (alpha={best_alpha_de})")

    if fold_idx == 0:
        fold1_data['de_pred_test'] = de_pred_test
        fold1_data['de_std_test']  = de_std_test

    # ── Residual Boosting ─────────────────────────────────────────────────────
    best_alpha_bst = alpha_grid_search(
        X_train_s, y_train_s, y_orig_train, inv_y, best_n, random_state=fold_seed)
    cumulative_train = np.zeros(len(y_train))
    cumulative_test  = np.zeros(len(y_test))
    residual         = y_train_s.copy()
    for stage in range(N_BOOST_STAGES):
        net_r = make_mlp(best_n, random_state=fold_seed + stage * 37,
                         alpha=best_alpha_bst)
        net_r.fit(X_train_s, residual, sample_weight=sw)
        cumulative_train += net_r.predict(X_train_s)
        cumulative_test  += net_r.predict(X_test_s)
        residual          = y_train_s - cumulative_train
    bst_pred_test = inv_y(cumulative_test)
    mse, r, r2 = compute_metrics(y_orig_test, bst_pred_test)
    fold_metrics['Residual Boosting']['mse'].append(mse)
    fold_metrics['Residual Boosting']['r'].append(r)
    fold_metrics['Residual Boosting']['r2'].append(r2)
    oof_preds['Residual Boosting'][test_idx] = bst_pred_test
    print(f"  Res. Boosting    MSE={mse:.6f}  R={r:.4f}  R²={r2:.4f}"
          f"  (alpha={best_alpha_bst})")

    oof_true[test_idx] = y_orig_test   # original units for OOF scatter plots


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

# ── Raw-input LOO importance (standardisation only, no log10 on any input) ───
print("\nPARAMETER IMPORTANCE — Leave-One-Out (raw inputs, fold-1 split)")
tr_idx1 = fold1_data['train_idx']
te_idx1 = fold1_data['test_idx']

X_raw_tr = X_raw[tr_idx1]
X_raw_te = X_raw[te_idx1]
scaler_x_raw = StandardScaler().fit(X_raw_tr)
X_raw_tr_s   = scaler_x_raw.transform(X_raw_tr)
X_raw_te_s   = scaler_x_raw.transform(X_raw_te)

# Target: y_orig scaled independently (no log10)
scaler_y_raw  = StandardScaler().fit(y_tr1.reshape(-1, 1))
y_raw_tr_s    = scaler_y_raw.transform(y_tr1.reshape(-1, 1)).ravel()
inv_y_raw     = lambda a: scaler_y_raw.inverse_transform(
                    np.array(a).reshape(-1, 1)).ravel()

impact_raw = np.zeros(n_features)
for p in range(n_features):
    cols       = [c for c in range(n_features) if c != p]
    net_raw    = make_mlp(bn1, random_state=fs1)
    net_raw.fit(X_raw_tr_s[:, cols], y_raw_tr_s)
    mse_tr_raw = mean_squared_error(y_tr1, inv_y_raw(net_raw.predict(X_raw_tr_s[:, cols])))
    mse_te_raw = mean_squared_error(y_te1, inv_y_raw(net_raw.predict(X_raw_te_s[:, cols])))
    impact_raw[p] = (mse_tr_raw + mse_te_raw) / 2
    print(f"  Excl. Param {p + 1:2d}  TrainMSE={mse_tr_raw:.6f}  TestMSE={mse_te_raw:.6f}")

norm_impact_raw   = impact_raw / impact_raw.sum()
sorted_idx_raw    = np.argsort(norm_impact_raw)[::-1]
sorted_labels_raw = [f'Param {i + 1}' for i in sorted_idx_raw]
sorted_values_raw = norm_impact_raw[sorted_idx_raw]

print("\nRaw-input importance (for comparison — not plotted):")
for i, (lbl, val) in enumerate(zip(sorted_labels_raw, sorted_values_raw)):
    print(f"  {lbl}: {val:.3f}")

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

# ── Fold-to-fold MSE variance report (log10 transform effect) ─────────────────
print("\n" + "=" * 60)
print("FOLD-TO-FOLD MSE VARIANCE  (metrics in original units)")
print("  (log10 preprocessing applied to Param 1 and Target)")
print("=" * 60)
print(f"\n  {'Model':<22} {'Fold MSE values':>52}  Variance")
for name in MODEL_NAMES:
    mse_vals  = fold_metrics[name]['mse']
    mse_var   = np.var(mse_vals)
    vals_str  = "  ".join(f"{v:.6f}" for v in mse_vals)
    print(f"  {name:<22} {vals_str:>52}  {mse_var:.4e}")
print()


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
    '00_raw_distributions':                fig_raw_dist,
    '00a_input_distributions':             fig_dist,
    '00b_correlation_heatmap':             fig_corr,
    '00c_kmeans_clustering':               fig_clust,
    '00d_pca_cluster_projection':          fig_pca,
    '00e_target_distribution_diagnostics': fig_dist_diag,
    '00e2_kh_distribution_diagnostics':   fig_kh_diag,
    '00f_pdf_overlay':                     fig_pdf,
    '01_architecture_search_fold1':        fig_arch,
    '02_parameter_importance':              fig_imp,
    '03_baseline_oof_scatter':              fig_scatters['Baseline'],
    '04_bagging_boot_oof_scatter':          fig_scatters['Bagging'],
    '04b_bagging_subsamp_oof_scatter':      fig_scatters['Bagging (Subsamp)'],
    '05_deep_ensemble_oof_scatter':         fig_scatters['Deep Ensemble'],
    '06_boosting_oof_scatter':              fig_scatters['Residual Boosting'],
    '07_deep_ensemble_uncertainty':         fig_unc,
    '08_cv_mse_boxplot':                    fig_box,
    '09_summary_comparison':                fig_sum,
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
