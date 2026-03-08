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
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
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
# 1. DATA LOADING & SPLITTING
# =============================================================================

print("=" * 60)
print("Loading data...")
M = pd.read_excel(DATA_FILE).values

X = M[:, :-1]   # all columns except last  (samples × features)
y = M[:, -1]    # last column              (samples,)

Q      = len(y)
Q1     = int(np.floor(Q * TRAIN_RATIO))
idx    = np.random.permutation(Q)

X_train, y_train = X[idx[:Q1]],  y[idx[:Q1]]
X_test,  y_test  = X[idx[Q1:]], y[idx[Q1:]]

n_features = X_train.shape[1]
print(f"Dataset: {Q} samples  |  {n_features} features")
print(f"Train: {Q1}  |  Test: {Q - Q1}")


# =============================================================================
# HELPERS
# =============================================================================

def make_mlp(n_neurons: int, random_state: int = 0) -> MLPRegressor:
    """Return a configured single-hidden-layer MLP matching the MATLAB baseline."""
    return MLPRegressor(
        hidden_layer_sizes=(n_neurons,),
        activation='logistic',          # logsig  → logistic sigmoid
        solver='lbfgs',                 # closest deterministic quasi-Newton solver
        max_iter=2000,
        random_state=random_state,
        early_stopping=False,
    )


def metrics(y_true, y_pred, label=''):
    mse = mean_squared_error(y_true, y_pred)
    r   = pearsonr(y_true, y_pred)[0]
    r2  = r2_score(y_true, y_pred)
    if label:
        print(f"  {label:30s}  MSE={mse:.6f}  R={r:.4f}  R²={r2:.4f}")
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
# 2. BASELINE: ARCHITECTURE SEARCH  (mirrors MATLAB loop)
# =============================================================================

print("\n" + "=" * 60)
print("BASELINE — Architecture Search (1 to", MAX_NEURONS, "neurons)")
print("=" * 60)

mse_train_base = np.zeros(MAX_NEURONS)
mse_test_base  = np.zeros(MAX_NEURONS)
trained_nets   = {}

for n in range(MIN_NEURONS, MAX_NEURONS + 1):
    net = make_mlp(n, random_state=RANDOM_SEED)
    net.fit(X_train, y_train)
    mse_train_base[n - 1] = mean_squared_error(y_train, net.predict(X_train))
    mse_test_base[n - 1]  = mean_squared_error(y_test,  net.predict(X_test))
    trained_nets[n] = net
    print(f"  Neurons={n:2d}  TrainMSE={mse_train_base[n-1]:.6f}  "
          f"TestMSE={mse_test_base[n-1]:.6f}")

best_n   = int(np.argmin(mse_test_base)) + MIN_NEURONS
best_net = trained_nets[best_n]
print(f"\nBest network: {best_n} hidden neuron(s)")
metrics(y_train, best_net.predict(X_train), 'Baseline Train')
metrics(y_test,  best_net.predict(X_test),  'Baseline Test')

# Architecture-search plot
fig_arch, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(MIN_NEURONS, MAX_NEURONS + 1), mse_train_base, 'ro-', label='Train MSE')
ax.plot(range(MIN_NEURONS, MAX_NEURONS + 1), mse_test_base,  'go-', label='Test MSE')
ax.plot(best_n, mse_test_base[best_n - MIN_NEURONS], 'k*', ms=12, label='Best')
ax.set_xlabel('Number of Neurons'); ax.set_ylabel('MSE')
ax.set_title('ANN Performance vs. Number of Neurons')
ax.legend(); ax.grid(True); plt.tight_layout()

# Scatter plots — baseline
fig_tr = scatter_plot(y_train, best_net.predict(X_train), 'Baseline — Training')
fig_te = scatter_plot(y_test,  best_net.predict(X_test),  'Baseline — Testing')


# =============================================================================
# 3. PARAMETER IMPORTANCE (leave-one-out, mirrors MATLAB section)
# =============================================================================

print("\n" + "=" * 60)
print("PARAMETER IMPORTANCE — Leave-One-Out")
print("=" * 60)

impact = np.zeros(n_features)
for p in range(n_features):
    cols = [c for c in range(n_features) if c != p]
    net_excl = make_mlp(best_n, random_state=RANDOM_SEED)
    net_excl.fit(X_train[:, cols], y_train)
    mse_tr = mean_squared_error(y_train, net_excl.predict(X_train[:, cols]))
    mse_te = mean_squared_error(y_test,  net_excl.predict(X_test[:, cols]))
    impact[p] = (mse_tr + mse_te) / 2
    print(f"  Excl. Param {p + 1:2d}  TrainMSE={mse_tr:.6f}  TestMSE={mse_te:.6f}")

norm_impact    = impact / impact.sum()
sorted_idx     = np.argsort(norm_impact)[::-1]
sorted_labels  = [f'Param {i + 1}' for i in sorted_idx]
sorted_values  = norm_impact[sorted_idx]

fig_imp, ax = plt.subplots(figsize=(max(6, n_features), 4))
bars = ax.bar(range(n_features), sorted_values, color='steelblue')
ax.set_xticks(range(n_features)); ax.set_xticklabels(sorted_labels, rotation=45, ha='right')
ax.set_xlabel('Parameters'); ax.set_ylabel('Normalized Mean Performance Impact')
ax.set_title('Parameter Importance (Leave-One-Out)')
ax.grid(True, axis='y')
for bar, val in zip(bars, sorted_values):
    ax.text(bar.get_x() + bar.get_width() / 2, val,
            f'{val:.2f}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()


# =============================================================================
# 4. ENSEMBLE — BAGGING  (with Out-of-Bag error estimation)
# =============================================================================

print("\n" + "=" * 60)
print(f"ENSEMBLE 1 — Bagging  ({N_ENSEMBLE} members, best_n={best_n} neurons)")
print("=" * 60)

bag_preds_train = np.zeros((N_ENSEMBLE, len(y_train)))
bag_preds_test  = np.zeros((N_ENSEMBLE, len(y_test)))

# Out-of-bag (OOB) bookkeeping
oob_accumulator = np.zeros(len(y_train))
oob_count       = np.zeros(len(y_train))

for m in range(N_ENSEMBLE):
    # Bootstrap sample (indices into training set)
    boot_idx = resample(np.arange(len(y_train)), replace=True, random_state=RANDOM_SEED + m)
    oob_idx  = np.setdiff1d(np.arange(len(y_train)), boot_idx)

    net_b = make_mlp(best_n, random_state=RANDOM_SEED + m)
    net_b.fit(X_train[boot_idx], y_train[boot_idx])

    bag_preds_train[m] = net_b.predict(X_train)
    bag_preds_test[m]  = net_b.predict(X_test)

    # OOB contribution
    if len(oob_idx) > 0:
        oob_accumulator[oob_idx] += net_b.predict(X_train[oob_idx])
        oob_count[oob_idx]       += 1

    mse_te = mean_squared_error(y_test, net_b.predict(X_test))
    print(f"  Member {m + 1:2d}  TestMSE={mse_te:.6f}")

# Aggregate predictions
bag_pred_train = bag_preds_train.mean(axis=0)
bag_pred_test  = bag_preds_test.mean(axis=0)

print("\nBagging ensemble aggregate:")
metrics(y_train, bag_pred_train, 'Bagging Train')
metrics(y_test,  bag_pred_test,  'Bagging Test')

# OOB estimate
valid_oob = oob_count > 0
oob_pred  = np.where(valid_oob, oob_accumulator / np.maximum(oob_count, 1), np.nan)
oob_mse   = mean_squared_error(y_train[valid_oob], oob_pred[valid_oob])
print(f"  {'OOB Error Estimate':30s}  MSE={oob_mse:.6f}")

fig_bag_tr = scatter_plot(y_train, bag_pred_train, 'Bagging — Training')
fig_bag_te = scatter_plot(y_test,  bag_pred_test,  'Bagging — Testing')


# =============================================================================
# 5. ENSEMBLE — DEEP ENSEMBLES  (independent random initialisations)
# =============================================================================

print("\n" + "=" * 60)
print(f"ENSEMBLE 2 — Deep Ensembles  ({N_ENSEMBLE} members, best_n={best_n} neurons)")
print("=" * 60)

de_preds_train = np.zeros((N_ENSEMBLE, len(y_train)))
de_preds_test  = np.zeros((N_ENSEMBLE, len(y_test)))

for m in range(N_ENSEMBLE):
    net_d = make_mlp(best_n, random_state=RANDOM_SEED * 100 + m)   # different seeds
    net_d.fit(X_train, y_train)
    de_preds_train[m] = net_d.predict(X_train)
    de_preds_test[m]  = net_d.predict(X_test)
    mse_te = mean_squared_error(y_test, net_d.predict(X_test))
    print(f"  Member {m + 1:2d}  TestMSE={mse_te:.6f}")

de_pred_train = de_preds_train.mean(axis=0)
de_pred_test  = de_preds_test.mean(axis=0)
# Predictive uncertainty from standard deviation across members
de_std_test   = de_preds_test.std(axis=0)

print("\nDeep Ensembles aggregate:")
metrics(y_train, de_pred_train, 'Deep Ensemble Train')
metrics(y_test,  de_pred_test,  'Deep Ensemble Test')

fig_de_tr = scatter_plot(y_train, de_pred_train, 'Deep Ensemble — Training')
fig_de_te = scatter_plot(y_test,  de_pred_test,  'Deep Ensemble — Testing')

# Uncertainty plot
fig_unc, ax = plt.subplots(figsize=(8, 4))
order = np.argsort(y_test)
ax.plot(y_test[order], label='True', color='black', lw=1.5)
ax.plot(de_pred_test[order], label='DE Mean', color='steelblue', lw=1.5)
ax.fill_between(range(len(order)),
                de_pred_test[order] - de_std_test[order],
                de_pred_test[order] + de_std_test[order],
                alpha=0.3, color='steelblue', label='±1 std')
ax.set_xlabel('Sorted Test Sample Index')
ax.set_ylabel('Target Value')
ax.set_title('Deep Ensembles — Predictive Uncertainty (Test Set)')
ax.legend(); ax.grid(True); plt.tight_layout()


# =============================================================================
# 6. ENSEMBLE — RESIDUAL BOOSTING
# =============================================================================

print("\n" + "=" * 60)
print(f"ENSEMBLE 3 — Residual Boosting  ({N_BOOST_STAGES} stages, best_n={best_n} neurons)")
print("=" * 60)

boost_nets      = []
cumulative_train = np.zeros(len(y_train))
cumulative_test  = np.zeros(len(y_test))
residual         = y_train.copy()

for stage in range(N_BOOST_STAGES):
    net_r = make_mlp(best_n, random_state=RANDOM_SEED + stage * 37)
    net_r.fit(X_train, residual)
    boost_nets.append(net_r)

    pred_tr   = net_r.predict(X_train)
    pred_te   = net_r.predict(X_test)
    cumulative_train += pred_tr
    cumulative_test  += pred_te
    residual          = y_train - cumulative_train   # update residual

    mse_tr = mean_squared_error(y_train, cumulative_train)
    mse_te = mean_squared_error(y_test,  cumulative_test)
    print(f"  Stage {stage + 1:2d}  TrainMSE={mse_tr:.6f}  TestMSE={mse_te:.6f}")

print("\nResidual Boosting final:")
metrics(y_train, cumulative_train, 'Boosting Train')
metrics(y_test,  cumulative_test,  'Boosting Test')

fig_bst_tr = scatter_plot(y_train, cumulative_train, 'Residual Boosting — Training')
fig_bst_te = scatter_plot(y_test,  cumulative_test,  'Residual Boosting — Testing')


# =============================================================================
# 7. SUMMARY COMPARISON
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY — Test-Set Comparison")
print("=" * 60)

results = {
    'Baseline':         best_net.predict(X_test),
    'Bagging':          bag_pred_test,
    'Deep Ensemble':    de_pred_test,
    'Residual Boosting': cumulative_test,
}

summary_rows = []
for name, pred in results.items():
    mse, r, r2 = metrics(y_test, pred, name)
    summary_rows.append({'Model': name, 'MSE': mse, 'R': r, 'R²': r2})

summary_df = pd.DataFrame(summary_rows).set_index('Model')
print("\n", summary_df.to_string())

# Bar chart comparison
fig_sum, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, col in zip(axes, ['MSE', 'R', 'R²']):
    vals = summary_df[col].values
    colors = ['#4c72b0', '#dd8452', '#55a868', '#c44e52']
    ax.bar(summary_df.index, vals, color=colors)
    ax.set_title(col); ax.set_xticklabels(summary_df.index, rotation=30, ha='right')
    ax.grid(True, axis='y')
    # Highlight best bar
    best_i = (np.argmin(vals) if col == 'MSE' else np.argmax(vals))
    ax.bar(summary_df.index[best_i], vals[best_i], color='gold', edgecolor='black', lw=1.5)
plt.suptitle('Ensemble Comparison — Test Set', fontsize=13)
plt.tight_layout()


# =============================================================================
# 8. SAVE ALL FIGURES
# =============================================================================

figures = {
    '01_architecture_search':      fig_arch,
    '02_parameter_importance':     fig_imp,
    '03_baseline_train_scatter':   fig_tr,
    '04_baseline_test_scatter':    fig_te,
    '05_bagging_train_scatter':    fig_bag_tr,
    '06_bagging_test_scatter':     fig_bag_te,
    '07_deep_ensemble_train':      fig_de_tr,
    '08_deep_ensemble_test':       fig_de_te,
    '09_deep_ensemble_uncertainty':fig_unc,
    '10_boosting_train_scatter':   fig_bst_tr,
    '11_boosting_test_scatter':    fig_bst_te,
    '12_summary_comparison':       fig_sum,
}

for fname, fig in figures.items():
    fig.savefig(_out(f'{fname}.png'), dpi=150, bbox_inches='tight')

print("\nAll figures saved as PNG files.")
plt.show()

# =============================================================================
# 9. EXPORT SUMMARY TABLE
# =============================================================================

summary_df.to_csv(_out('ensemble_summary.csv'))
print("Summary table saved to ensemble_summary.csv")
