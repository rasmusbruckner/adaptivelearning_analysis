""" Prior Prediction Resource-Only Model

    1. Load and prepare data
    2. Simulate baseline model
    3. Simulate perseveration and anchoring behavior
    4. Compute effects of interest

"""

import numpy as np
import pandas as pd
from sampling.al_simulation_sampling import simulation_loop_sampling
from al_utilities import safe_save_dataframe, compute_anchoring_bias
import os
from al_plot_utils import latex_plt
import matplotlib
import random

# Set random number generator for reproducible results
np.random.seed(123)
random.seed(123)

# Update matplotlib to use Latex and to change some defaults
matplotlib = latex_plt(matplotlib)

# Get home directory
paths = os.getcwd()
path = paths.split(os.path.sep)
home_dir = path[1]

# Load data
df_exp2 = pd.read_pickle('al_data/data_prepr_2.pkl')
n_subj = len(np.unique(df_exp2['subj_num']))  # ID for each participant

# Age group for model input
age_group = df_exp2.groupby(['subj_num'])['age_group'].first()
age_group = age_group.reset_index(drop=True)

# Choose baseline perseveration parameters
# Here, we take the average of younger adults
b0 = 0.85
b1 = -0.25

# --------------------------
# 2. Simulate baseline model
# --------------------------

model_exp2 = pd.DataFrame(columns=['subj_num', 'age_group', 'criterion', 'n_samples'], index=[np.arange(n_subj)])
model_exp2['subj_num'] = np.arange(n_subj) + 1
model_exp2['age_group'] = np.array(age_group[:n_subj])
model_exp2.loc[model_exp2['age_group'] == 1, 'criterion'] = np.nan
model_exp2.loc[model_exp2['age_group'] == 3, 'criterion'] = np.nan
model_exp2.loc[model_exp2['age_group'] == 4, 'criterion'] = np.nan
model_exp2.loc[model_exp2['age_group'] == 1, 'n_samples'] = 100
model_exp2.loc[model_exp2['age_group'] == 3, 'n_samples'] = 100
model_exp2.loc[model_exp2['age_group'] == 4, 'n_samples'] = 100
model_exp2.loc[model_exp2['age_group'] == 1, 'b_0'] = b0
model_exp2.loc[model_exp2['age_group'] == 3, 'b_0'] = b0
model_exp2.loc[model_exp2['age_group'] == 4, 'b_0'] = b0
model_exp2.loc[model_exp2['age_group'] == 1, 'b_1'] = b1
model_exp2.loc[model_exp2['age_group'] == 3, 'b_1'] = b1
model_exp2.loc[model_exp2['age_group'] == 4, 'b_1'] = b1

# Run simulation
n_sim = 1
sim_pers = False
sub_plot = False
all_pers_baseline, all_est_errs_baseline, df_data_baseline = simulation_loop_sampling(df_exp2, model_exp2, n_subj,
                                                                                      n_sim=n_sim,
                                                                                      model_sat=False,
                                                                                      resource_only=True)

# ------------------------------------------------
# 3. Simulate perseveration and anchoring behavior
# ------------------------------------------------

# Simulate perseveration model
model_exp2.loc[model_exp2['age_group'] == 1, 'n_samples'] = 2
model_exp2.loc[model_exp2['age_group'] == 3, 'n_samples'] = 10
model_exp2.loc[model_exp2['age_group'] == 4, 'n_samples'] = 2
all_pers_pers, all_est_errs_pers, df_data_pers = simulation_loop_sampling(df_exp2, model_exp2, n_subj, n_sim=n_sim,
                                                                          model_sat=False,
                                                                          resource_only=True)

# Simulate anchoring model
model_exp2.loc[model_exp2['age_group'] == 1, 'n_samples'] = 20
model_exp2.loc[model_exp2['age_group'] == 3, 'n_samples'] = 50
model_exp2.loc[model_exp2['age_group'] == 4, 'n_samples'] = 20
all_pers_anchor, all_est_errs_anchor, df_data_anchor = simulation_loop_sampling(df_exp2, model_exp2, n_subj,
                                                                                n_sim=n_sim,
                                                                                model_sat=False, resource_only=True)

# ------------------------------
# 4. Compute effects of interest
# ------------------------------

# Compute perseveration
# ---------------------

# Baseline parameters
all_pers_baseline.name = "priorpred_resource_only_pers_baseline"
safe_save_dataframe(all_pers_baseline, None, overleaf=False)

# Perseveration parameters
all_pers_pers.name = "priorpred_resource_only_pers_pers"
safe_save_dataframe(all_pers_pers, None, overleaf=False)

# Anchoring parameters
all_pers_anchor.name = "priorpred_resource_only_pers_anchor"
safe_save_dataframe(all_pers_anchor, None, overleaf=False)

# Compute estimation errors
# -------------------------

# Baseline parameters
all_est_errs_baseline.name = "priorpred_resource_only_est_errs_baseline"
safe_save_dataframe(all_est_errs_baseline, None, overleaf=False)

# Perseveration parameters
all_est_errs_pers.name = "priorpred_resource_only_est_errs_pers"
safe_save_dataframe(all_est_errs_pers, None, overleaf=False)

# Anchoring parameters
all_est_errs_anchor.name = "priorpred_resource_only_est_errs_anchor"
safe_save_dataframe(all_est_errs_anchor, None, overleaf=False)

# Compute anchoring bias
# ----------------------

# Baseline parameters
df_reg_baseline = compute_anchoring_bias(n_subj, df_data_baseline)
df_reg_baseline.name = "priorpred_resource_only_df_reg_baseline"
safe_save_dataframe(df_reg_baseline, None, overleaf=False)

# Perseveration parameters
df_reg_pers = compute_anchoring_bias(n_subj, df_data_pers)
df_reg_pers.name = "priorpred_resource_only_df_reg_pers"
safe_save_dataframe(df_reg_pers, None, overleaf=False)

# Anchoring parameters
df_reg_anchor = compute_anchoring_bias(n_subj, df_data_anchor)
df_reg_anchor.name = "priorpred_resource_only_df_reg_anchor"
safe_save_dataframe(df_reg_anchor, None, overleaf=False)
