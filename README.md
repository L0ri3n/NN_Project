# Neural Network Ensemble for Contaminant Transport Prediction

Course project for *Legacy Contamination and Soil Remediation*, TU Bergakademie Freiberg, Winter Semester 2025.

## Overview

This project builds and evaluates ensemble learning strategies on top of a multi-layer perceptron (MLP) surrogate model for predicting contaminant concentrations in an aquifer subject to hydraulic fracturing fluid migration. The dataset originates from the numerical simulations of Taherdangkoo et al. (2019).

Four ensemble strategies are compared against a single-model baseline using stratified 5-fold cross-validation:

- Bootstrap Bagging
- Subsampling Bagging (without replacement, 65 %)
- Deep Ensembles (random initialisation diversity)
- Residual Boosting (sequential residual fitting)

The full analysis and results are documented in the accompanying LaTeX report.

## Repository structure

```
.
├── neural_network_ensemble.py   # Main script — all models, CV, figures
├── config.py                    # Default configuration parameters
├── config_local.py.template     # Template for local overrides (copy and rename)
├── MODIFICATIONS.md             # Chronological log of all code changes
├── Data/
│   └── parametric_study.xlsx    # Input dataset (1000-run parametric study)
└── output/                      # Generated figures and summary CSV
```

The `Latex_document/` directory is excluded from version control (see `.gitignore`).

## Requirements

Python 3.x with the following packages:

| Package      | Purpose                              |
|--------------|--------------------------------------|
| numpy        | Numerical operations                 |
| pandas       | Data loading and binning             |
| scikit-learn | MLP, scalers, CV, metrics, resampling|
| scipy        | Distribution fitting, KS tests       |
| matplotlib   | All figures                          |

Install via conda or pip:

```bash
pip install numpy pandas scikit-learn scipy matplotlib openpyxl
```

## Running the script

```bash
python neural_network_ensemble.py
```

All figures are saved as PNG files in `output/`. The cross-validation summary is written to `output/ensemble_summary_cv.csv`.

To adjust configuration (random seed, number of folds, ensemble size, etc.), copy `config_local.py.template` to `config_local.py` and edit that file. Local overrides take precedence over `config.py` and are gitignored.

## Configuration

Key parameters in `config.py`:

| Parameter             | Default   | Description                                      |
|-----------------------|-----------|--------------------------------------------------|
| `RANDOM_SEED`         | 138       | Global seed for reproducibility                  |
| `K_FOLDS`             | 5         | Number of CV folds                               |
| `N_ENSEMBLE`          | 5         | Members per ensemble                             |
| `BAG_SUBSAMPLE_FRAC`  | 0.65      | Fraction drawn per member in Subsampling Bagging |
| `N_BOOST_STAGES`      | 5         | Stages in Residual Boosting                      |
| `L2_ALPHA_GRID`       | see file  | Candidate regularisation strengths               |
| `SAMPLE_WEIGHT_SCHEME`| log_value | Sample weighting for high-value emphasis         |

## Data source

Taherdangkoo, R., Tatomir, A., Anighoro, T., & Sauter, M. (2019). Modeling fate and transport of hydraulic fracturing fluid in the presence of abandoned wells. *Journal of Contaminant Hydrology*, 221, 58–68. https://doi.org/10.1016/j.jconhyd.2018.12.003

## Author

Lorién Crespo Gracia — TU Bergakademie Freiberg, 2025–2026
