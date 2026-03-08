# Modification Log — Neural Network Ensemble Project

---

## [1] L2 Regularisation Grid Search per Model

**Date:** 2026-03-08
**Files changed:** `config.py`, `neural_network_ensemble.py`

### What changed

Added per-model L2 regularisation tuning. Each of the four models (Baseline,
Bagging, Deep Ensembles, Residual Boosting) independently selects its optimal
regularisation strength before training.

### Implementation details

- **`config.py`** — added `L2_ALPHA_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0]`
- **`make_mlp`** — added `alpha` parameter (default `0.0001`), forwarded to
  `MLPRegressor(alpha=...)`
- **`alpha_grid_search`** — new helper; splits the training fold 80/20, trains
  one MLP per candidate alpha, returns the alpha with the lowest inner-validation
  MSE
- Each model section calls `alpha_grid_search` with its fold data, then passes
  the chosen alpha to every `make_mlp` call in that section
- **Summary table** — `Best Alpha` column added to both the console output and
  `ensemble_summary_cv.csv`
- The architecture search (neuron count loop) still uses the sklearn default
  alpha to avoid a 2-D joint search; alpha is tuned afterwards for each model
  using the already-selected `best_n`

### Motivation

Without regularisation tuning, the four models implicitly all used the sklearn
default (α = 0.0001). Different ensemble strategies have different effective
capacities, so they benefit from different regularisation strengths. Per-model
grid search makes the comparison fairer and reduces the risk of overfitting,
especially for Residual Boosting where the cumulative model complexity grows
with each stage.

---

## [2] Stratified K-Fold Cross-Validation + Subsampling Bagging

**Date:** 2026-03-08
**Files changed:** `config.py`, `neural_network_ensemble.py`

### What changed

Replaced the single 80/20 train/test split with stratified 5-fold
cross-validation. All five models are trained and evaluated on every fold.
Metrics in the summary table are now reported as mean ± std across folds.
A new box plot visualises the test MSE distribution across folds for all
models. A second Bagging variant (subsampling without replacement) was also
added for comparison.

### Implementation details

- **`config.py`**
  - Added `K_FOLDS = 5` (overridable via `config_local.py`)
  - Added `BAG_SUBSAMPLE_FRAC = 0.65` — fraction of training samples drawn
    per member in subsampling Bagging
- **Data loading** — no train/test split; raw `X` and `y` are kept whole
- **Stratification** — `pd.qcut(y, q=K_FOLDS)` bins the continuous target
  into `K_FOLDS` quantile buckets; `StratifiedKFold` uses these bins to
  ensure each fold covers the full target range
- **CV loop** — `StratifiedKFold(n_splits=5, shuffle=True)` iterates over
  folds; scalers (`StandardScaler` for X and y) are fit on the training split
  of each fold only
- **Per-fold seed** — `fold_seed = RANDOM_SEED + fold_idx * 1000` keeps
  random states distinct across folds while remaining reproducible
- **Architecture search** — runs inside every fold; fold-1 result is saved
  for the architecture search plot
- **`alpha_grid_search`** — refactored from global to pure function;
  accepts `X_tr_s`, `y_tr_s`, `y_tr`, `inv_y` as arguments
- **`make_inv_y(scaler)`** — new helper that returns an inverse-transform
  closure bound to the fold's fitted `scaler_y`, replacing the global
  `inv_y` lambda
- **Bagging (Bootstrap)** — OOB tracking updated to use a boolean mask
  (`oob_mask[unique(boot_idx)] = False`) instead of `np.setdiff1d`
- **Bagging (Subsampling)** — new variant drawing `BAG_SUBSAMPLE_FRAC` of
  training samples without replacement per member; OOB is the exact
  complement (`oob_mask[sub_idx] = False`); tracked as `'Bagging (Subsamp)'`
- **`fold_metrics`** — dict of lists collecting MSE, R, R² per fold per model
- **OOF predictions** — `oof_preds[name][test_idx]` accumulates each fold's
  test predictions into a full-dataset array for scatter plots
- **Summary table** — columns are now `MSE mean`, `MSE std`, `R mean`,
  `R std`, `R² mean`, `R² std`; a human-readable `mean ± std` display is
  printed separately
- **Box plot** (`08_cv_mse_boxplot.png`) — new figure; one box per model
  showing the distribution of test MSE across the 5 folds, with individual
  fold points overlaid
- **Summary bar chart** — updated to 3 panels (MSE, R, R²) with `±std`
  error bars; Overfit ratio removed (train metrics not tracked per fold)
- **Scatter plots** — replaced per-split scatter with one OOF scatter per
  model (all folds combined)
- **Parameter importance** — unchanged algorithm; runs once after the CV
  loop using fold-1 training data and `best_n`
- **Uncertainty plot** — unchanged; uses fold-1 Deep Ensemble predictions
- **Output CSV** renamed to `ensemble_summary_cv.csv`

### Motivation

A single 80/20 split gives a point estimate of performance that is highly
sensitive to which samples fall in the test set, especially with a small
dataset. Five-fold CV provides a more reliable estimate of generalisation
performance and quantifies its variability (std across folds). Stratified
splitting on quantile bins ensures each fold covers the full range of the
target, preventing folds that are dominated by low or high values. The
subsampling Bagging variant is added to compare the effect of sampling with
vs. without replacement on ensemble diversity and OOB coverage.

---

## [3] Subsampling Bagging (without replacement, 65 %) + OOB Estimation

**Date:** 2026-03-08
**Files changed:** `config.py`, `neural_network_ensemble.py`

### What changed

Added a second Bagging variant that draws 65 % of each fold's training set
**without replacement** per member, keeping the remaining 35 % as a
deterministic OOB set. OOB error estimation was also added to the original
Bootstrap Bagging. Both variants now print per-fold OOB-MSE and OOB coverage
alongside test-set metrics, and appear as separate rows in the summary table
and all comparison plots.

### Implementation details

- **`config.py`** — added `BAG_SUBSAMPLE_FRAC = 0.65`
- **Bootstrap Bagging** — inner loop extended:
  - `oob_mask[np.unique(boot_idx)] = False` identifies OOB samples per member
  - Predictions for OOB samples accumulated in `oob_sum_bag / oob_cnt_bag`
  - Final OOB-MSE computed over samples with `oob_cnt_bag > 0` (~90 % coverage)
  - Print line extended: `OOB-MSE=…  OOB-cov=…%`
- **Bagging (Subsamp)** — new section immediately after Bootstrap Bagging:
  - `sub_size = int(BAG_SUBSAMPLE_FRAC * len(y_train))` per fold
  - `np.random.RandomState(fold_seed + m).choice(…, replace=False)` draws the
    subsample; the complement is the exact OOB set (`oob_mask[sub_idx] = False`)
  - Same OOB accumulation pattern as Bootstrap Bagging; coverage is exactly
    `1 − BAG_SUBSAMPLE_FRAC = 35 %` per member, ~88 % aggregated over 5 members
  - Alpha selected via `alpha_grid_search` with the same `fold_seed`
  - Results stored under key `'Bagging (Subsamp)'` in `fold_metrics` and
    `oof_preds`
- **`MODEL_NAMES`** — `'Bagging (Subsamp)'` inserted after `'Bagging'`
- **`COLORS`** — purple `'#8172b2'` added for `'Bagging (Subsamp)'`
- **Scatter plots** — `'Bagging (Subsamp)'` OOF scatter generated automatically
  via the `MODEL_NAMES` loop; saved as `04b_bagging_subsamp_oof_scatter.png`
- **Box plot & summary bar chart** — automatically include the new model
  because both iterate over `MODEL_NAMES` / `COLORS`
- **figures dict** — `'04_bagging_oof_scatter'` renamed to
  `'04_bagging_boot_oof_scatter'`; `'04b_bagging_subsamp_oof_scatter'` added

### Motivation

Bootstrap sampling leaves ~36.8 % of the training set out per member on
average, but individual samples can be replicated multiple times inside the
bag, potentially inflating effective training variance. Subsampling without
replacement guarantees a clean split: each member trains on exactly 65 % of
the data and the remaining 35 % are a true held-out OOB set with no
replication artefacts. Comparing both strategies under identical CV and
architecture conditions isolates the effect of the sampling scheme on ensemble
diversity, generalisation, and OOB error quality.

---

## [4] Log10 Preprocessing for Param 1 and Target Variable

**Date:** 2026-03-08
**Files changed:** `neural_network_ensemble.py`

### What changed

Added a log10 preprocessing step applied to `Param 1` (feature index 0) and
the target variable immediately after data loading, before any splitting or
scaling. All predictions are inverse-transformed (`10^x`) before metrics are
computed, so MSE and R² remain interpretable in original physical units.

### Implementation details

- **Data loading** — after extracting `X` and `y` from the Excel file:
  - `y_orig = y.copy()` preserves the raw target for metric computation
  - `X[:, 0] = np.log10(X[:, 0])` transforms Param 1 in-place
  - `y = np.log10(y)` transforms the target in-place
  - Stratified binning (`pd.qcut`) and all CV splitting use the log-space `y`
    (monotonic transform preserves quantile order)
- **`make_inv_y(scaler)`** — updated to chain two inverse steps:
  `log_vals = scaler.inverse_transform(a)` then `return 10 ** log_vals`.
  The full forward pipeline is `y_orig → log10 → StandardScaler → y_s`;
  the inverse is `y_s → StandardScaler⁻¹ → 10^x → y_orig`
- **CV loop** — each fold extracts `y_orig_train, y_orig_test = y_orig[train_idx], y_orig[test_idx]`
  alongside the log-space `y_train`, `y_test`
- **Architecture search** — `mean_squared_error` calls replaced to use
  `y_orig_train` / `y_orig_test` (original units) vs `inv_y(predictions)`
- **`alpha_grid_search`** — `y_tr` argument changed from log-space `y_train`
  to `y_orig_train` so alpha ranking is consistent with the final metric space
- **OOB MSE** (both Bagging variants) — `y_train[valid_oob]` replaced with
  `y_orig_train[valid_oob]` to match original units
- **`compute_metrics`** calls — all five model sections now compare against
  `y_orig_test` instead of log-space `y_test`
- **`oof_true`** — assigned from `y_orig_test` so OOF scatter plots display
  original units on both axes
- **`fold1_data`** — `'y_train'` and `'y_test'` keys store `y_orig_train` /
  `y_orig_test`, keeping parameter importance metrics in original units
- **New variance report** — printed after the summary table; shows per-fold
  MSE values and their variance (`np.var`) for each model

### Results (5-fold CV, original units)

| Model | MSE mean ± std | R² mean ± std | Fold-MSE variance |
|---|---|---|---|
| Baseline | 52 581 ± 17 118 | 0.885 ± 0.042 | 2.93e+08 |
| Bagging (Bootstrap) | 48 670 ± 19 097 | 0.903 ± 0.024 | 3.65e+08 |
| Bagging (Subsamp) | 49 712 ± 19 001 | 0.899 ± 0.029 | 3.61e+08 |
| Deep Ensemble | **46 881 ± 14 858** | **0.903 ± 0.024** | **2.21e+08** |
| Residual Boosting | 52 114 ± 16 787 | 0.887 ± 0.041 | 2.82e+08 |

Deep Ensembles achieves both the lowest mean MSE and the lowest fold-to-fold
variance. Fold 3 remains the hardest fold for all models (Baseline MSE ~86k
vs. ~37–47k elsewhere), indicating the high-variance comes from data
distribution rather than preprocessing. The transform provides the most
benefit to Deep Ensembles; Bagging variants show higher variance than before,
likely because the compressed input range makes subsample diversity more
sensitive to which samples land in each bag.

### Motivation

Both `Param 1` and the target variable are right-skewed. Log10 compression
reduces the dynamic range, making the relationship between inputs and output
more linear and easier for the MLP's logistic activations to approximate.
Because the inverse transform is applied before every metric call, the
reported MSE and R² figures are directly comparable to values from earlier
runs (pre-transform) and remain physically meaningful.
