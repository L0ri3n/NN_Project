# =============================================================================
# PROJECT CONFIGURATION — default values
# To override locally, copy config_local.py.template → config_local.py
# and edit that file. config_local.py is gitignored.
# =============================================================================

# Path to the input data file (relative to project root, or absolute).
DATA_FILE = 'Data/parametric_study.xlsx'

# Rows to skip at the top of the Excel file before the header row.
# Increase this if the file has a title, caption, or blank rows above the data.
EXCEL_SKIPROWS = 4

# Directory where output PNGs and the summary CSV will be saved.
OUTPUT_DIR = 'output'

# Fraction of samples used for training; the remainder becomes the test set.
# E.g. 0.80 → 80 % train, 20 % test.
TRAIN_RATIO = 0.80

# Global random seed for reproducibility (data shuffling and network init).
RANDOM_SEED = 42

# Range of hidden-layer sizes explored during the baseline architecture search.
# The script trains one network per integer in [MIN_NEURONS, MAX_NEURONS] and
# selects the size with the lowest test MSE.
MIN_NEURONS = 1
MAX_NEURONS = 30

# Number of ensemble members used by Bagging and Deep Ensembles.
# Higher values reduce variance but increase training time linearly.
N_ENSEMBLE = 5

# Number of sequential stages in Residual Boosting.
# Each stage fits a new network on the residuals left by the previous stages.
N_BOOST_STAGES = 5
