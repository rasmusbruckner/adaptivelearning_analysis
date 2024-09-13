""" Prior Predictions Sampling Model

    This script runs simulations with different parameter settings for the age groups:

        1. All same
        2. Criterion different
        3. Number of samples different
"""


import numpy as np
import pandas as pd
import random
import matplotlib
import matplotlib.pyplot as plt
import os
from al_plot_utils import latex_plt
from al_sampling_sim_fun import run_sampling_simulation

# Set random number generator for reproducible results
np.random.seed(123)
random.seed(123)

# Update matplotlib to use Latex and to change some defaults
matplotlib = latex_plt(matplotlib)

# Get home directory
paths = os.getcwd()
path = paths.split(os.path.sep)
home_dir = path[1]

# Load data for model input
df_exp2 = pd.read_pickle('al_data/data_prepr_2.pkl')
n_subj = len(np.unique(df_exp2['subj_num']))

# Age group for model input
df_grouped = df_exp2.groupby('subj_num')['age_group'].unique().str[0].reset_index()
age_group = df_grouped['age_group']

n_ch = np.sum(age_group == 1)
n_ya = np.sum(age_group == 3)
n_oa = np.sum(age_group == 4)

# ------------------
# 1. Run simulations
# ------------------

# 1. All same
# -----------

# Parameter values
criterion_params_ch = [0.01, 0.008]
criterion_params_ya = [0.01, 0.008]
criterion_params_oa = [0.01, 0.008]

# Parameter values
n_samples_params_ch = [5.5, 2.0]
n_samples_params_ya = [5.5, 2.0]
n_samples_params_oa = [5.5, 2.0]

file_name = "all_same"
print('All-same parameters')

run_sampling_simulation(df_exp2, criterion_params_ch, criterion_params_ya, criterion_params_oa, n_samples_params_ch,
                        n_samples_params_ya, n_samples_params_oa, n_subj, n_ch, n_ya, n_oa, age_group, file_name)

# 2. Criterion different
# ----------------------

# Parameter values
criterion_params_ch = [0.02, 0.008]
criterion_params_ya = [0.008, 0.008]
criterion_params_oa = [0.02, 0.008]

file_name = "crit_different"
print('Criterion-different parameters')

run_sampling_simulation(df_exp2, criterion_params_ch, criterion_params_ya, criterion_params_oa, n_samples_params_ch,
                        n_samples_params_ya, n_samples_params_oa, n_subj, n_ch, n_ya, n_oa, age_group, file_name)

# 3. Number of samples different
# ------------------------------

# Parameter values
criterion_params_ch = [0.01, 0.008]
criterion_params_ya = [0.01, 0.008]
criterion_params_oa = [0.01, 0.008]

# Parameter values
n_samples_params_ch = [5.5, 0.7]
n_samples_params_ya = [6, 3.5]
n_samples_params_oa = [5.5, 0.7]

file_name = "n_samples_different"
print('N-samples-different parameters')

run_sampling_simulation(df_exp2, criterion_params_ch, criterion_params_ya, criterion_params_oa, n_samples_params_ch,
                        n_samples_params_ya, n_samples_params_oa, n_subj, n_ch, n_ya, n_oa, age_group, file_name)

# Show plots
plt.show()
