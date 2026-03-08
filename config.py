# =============================================================================
# PROJECT CONFIGURATION — default values
# To override locally, copy config_local.py.template → config_local.py
# and edit that file. config_local.py is gitignored.
# =============================================================================

# Path to the input data file (relative to project root, or absolute)
DATA_FILE = 'Data/parametric_study.xlsx'

# Directory where output PNGs and CSV will be saved
OUTPUT_DIR = 'output'

# Train / test split ratio
TRAIN_RATIO = 0.80

# Global random seed
RANDOM_SEED = 42

# Architecture search range (number of hidden neurons)
MIN_NEURONS = 1
MAX_NEURONS = 10

# Ensemble settings
N_ENSEMBLE     = 10   # members for Bagging / Deep Ensembles
N_BOOST_STAGES = 5    # stages for Residual Boosting
