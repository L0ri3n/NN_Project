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

---

## [5] Distribution Diagnostics + Gumbel Exploration (tried and reverted)

**Date:** 2026-03-13
**Files changed:** `config.py`, `neural_network_ensemble.py`

### What changed

Added a distribution diagnostic section (1c) that compares how well the
target variable fits a Gumbel vs a log-normal distribution, and explored
a Gumbel reduced-variate preprocessing option. The Gumbel preprocessing was
reverted after diagnostics confirmed log-normal is the better fit; the
diagnostic plots were kept for future reference.

### Implementation details

- **Section 1c** — new diagnostic block inserted after the clustering/PCA
  section; generates `00e_target_distribution_diagnostics.png` with three panels:
  - PDF overlay: histogram of `y_orig` with fitted Gumbel and log-normal PDFs
  - Gumbel probability paper: sorted `y_orig` vs Gringorten reduced variate
    (`−ln(−ln(p))`); a straight line indicates a Gumbel fit
  - Log-normal probability paper: normal Q-Q plot of `log10(y_orig)`
  - KS test D-statistics and p-values printed for both distributions
- **Imports** — added `gumbel_r`, `lognorm`, `kstest`, `probplot` from
  `scipy.stats`; retained for the diagnostics even after the preprocessing
  option was removed

### Results

KS D = 0.064 (log-normal) vs D = 0.277 (Gumbel); log-normal probability
paper R² = 0.97. The Gumbel option was removed; log10 remains the only
preprocessing applied to the target.

---

## [6] Sample Weighting to Emphasise Extreme Target Values

**Date:** 2026-03-13
**Files changed:** `config.py`, `neural_network_ensemble.py`

### What changed

Added per-sample training weights so that high target values receive
disproportionately more influence during gradient updates, directly
addressing the heavy upper tail that causes high MSE on extreme samples.

### Implementation details

- **`config.py`** — added `SAMPLE_WEIGHT_SCHEME` with three options:
  - `'none'` — all samples equally weighted (default, original behaviour)
  - `'rank'` — weight proportional to rank within the training fold; robust
    to scale of `y`
  - `'log_value'` — weight proportional to `log10(y_orig)` shifted to be
    strictly positive; tied to the log10 preprocessing space
- **`compute_sample_weights(y_train_orig)`** — new helper in the HELPERS
  section; returns `None` for `'none'`, otherwise a weight array normalised
  to `mean = 1` (preserves average gradient magnitude)
- **CV loop** — `sw = compute_sample_weights(y_orig_train)` computed once
  per fold after scalers are fitted
- **Baseline** — `best_net.fit(..., sample_weight=sw)`
- **Bagging (Bootstrap)** — `sw_boot = sw[boot_idx]` passed to each member
- **Bagging (Subsampling)** — `sw_sub = sw[sub_idx]` passed to each member
- **Deep Ensemble** — `sw` passed to each member
- **Residual Boosting** — `sw` passed to each stage (target changes per
  stage, weights do not)
- Architecture search and `alpha_grid_search` are intentionally left
  unweighted to keep hyperparameter selection fast and stable

### Results (5-fold CV, `SAMPLE_WEIGHT_SCHEME = 'log_value'`)

| Model | MSE mean | Δ vs [4] | R² mean | Δ vs [4] |
|---|---|---|---|---|
| Baseline | 49,493 | −3,088 | 0.898 | +0.013 |
| Bagging (Bootstrap) | 48,606 | −64 | 0.901 | −0.002 |
| Bagging (Subsamp) | **43,490** | **−6,222** | **0.911** | **+0.012** |
| Deep Ensemble | 48,294 | +1,413 | 0.902 | −0.001 |
| Residual Boosting | 49,376 | −2,738 | 0.899 | +0.012 |

Bagging (Subsampling) becomes the best model. The combination of
without-replacement subsampling (each member sees a clean 65% subset) and
log_value weights (extreme samples get extra pull when they are in a
member's training set) creates stronger member diversity across the target
range than bootstrap or deep ensemble approaches. Fold-to-fold MSE variance
dropped substantially across all models, indicating more consistent
performance rather than lucky fold assignments.

---

## [7] K_h Distribution Diagnostics + Standard-Normalisation Baseline

**Date:** 2026-04-11
**Files changed:** `neural_network_ensemble.py`

### What changed

Two additions made to support the LaTeX report:

1. **K_h distribution diagnostic figure (section 1c, `fig_kh_diag`)** — mirrors
   the existing target distribution diagnostics but applied to Param 1 (aquifer
   permeability $K_h$). Three-panel figure: standard Q-Q plot against normal
   quantiles, Gumbel probability paper, and log-normal probability paper. KS test
   D-statistics printed to console. Saved as `00e2_kh_distribution_diagnostics.png`.

2. **Standard-normalisation baseline CV (section 1d)** — reruns the full
   stratified 5-fold CV for all five models using only `StandardScaler` on raw
   inputs and raw target (no log10 transform, no sample weights). Results are
   printed to console and used in the preprocessing comparison table in the report.
   This provides the "standard normalisation only" baseline row against which the
   full log10 + sample weight pipeline is compared.

### Implementation details

- `X_raw` preserved before log10 transforms for use in section 1d and the raw
  LOO importance check
- `fig_kh_diag` added to the `figures` dict under key `'00e2_kh_distribution_diagnostics'`
- Section 1d uses private-scope variables (prefixed `_`) to avoid contaminating
  the main CV namespace; architecture search is simplified (no alpha grid search)
  to keep the comparison run fast

### Results

K_h KS statistics: D = 0.243 (log-normal) vs D = 0.297 (Gumbel). Log-normal
remains the better fit, though the higher D compared to the target reflects the
near-uniform spacing of the ten K_h levels on the log scale.

Standard-normalisation baseline MSE (mean ± std across 5 folds):

| Model              | MSE mean   | MSE std    |
|--------------------|------------|------------|
| Baseline           | ~121,786   | ~118,119   |
| Bagging (Bootstrap)| ~120,724   | ~15,403    |
| Bagging (Subsamp)  | ~115,483   | ~17,196    |
| Deep Ensemble      | ~56,587    | ~18,009    |
| Residual Boosting  | ~120,621   | ~118,112   |

Bootstrap Bagging and Residual Boosting show fold std approaching the mean,
illustrating that the log10 + sample weight pipeline is a prerequisite for
stable operation of those methods.

### Motivation

The report's preprocessing comparison table requires a true "no-preprocessing"
baseline. Comparing log-only vs log+weights (as done in earlier development)
does not answer the question of how much preprocessing helps overall. The
standard-normalisation run provides that reference point.

---

## [8] Raw-Input Leave-One-Out Parameter Importance

**Date:** 2026-04-11
**Files changed:** `neural_network_ensemble.py`

### What changed

Added a second LOO parameter importance analysis that uses untransformed inputs
(raw `X_raw`, `StandardScaler` only, no log10) to verify that the importance
ranking produced by the main analysis is not an artefact of the log10 transform
applied to K_h.

### Implementation details

- Uses the fold-1 train/test split stored in `fold1_data['train_idx']` and
  `fold1_data['test_idx']` (added to `fold1_data` alongside this change)
- A fresh `StandardScaler` is fitted on `X_raw[train_idx]` for this analysis
- Architecture fixed to `fold1_data['best_n']`; alpha fixed to sklearn default
- Results printed to console only — no figure generated
- `fold1_data` extended with `'train_idx'` and `'test_idx'` keys

### Results

Raw-input importance (normalised): φ ≈ 0.57, L ≈ 0.26, K_h ≈ 0.17 — identical
ranking to the log-space analysis. K_h's lower importance is not a preprocessing
artefact; it genuinely has less influence on contaminant concentration than
porosity or source distance across the sampled parameter range.

### Motivation

K_h (aquifer permeability) is log10-transformed during preprocessing. There was
a concern that this transform might disproportionately compress the K_h signal
relative to the other two inputs, artificially suppressing its apparent importance.
The raw-input check rules this out.
