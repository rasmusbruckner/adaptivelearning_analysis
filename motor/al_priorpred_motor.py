""" Prior Prediction Motor Model

    1. Load and prepare data
    2. Simulate baseline model
    3. Parameter search
    4. Simulate perseveration and anchoring behavior
    5. Save data

"""

import pandas as pd
import matplotlib
import numpy as np
import os
from al_plot_utils import latex_plt
from al_simulation_motor import simulation_loop_motor
from al_utilities import compute_anchoring_bias, find_nearest, safe_save_dataframe

# Set random number generator for reproducible results
np.random.seed(123)

# Update matplotlib to use Latex and to change some defaults
matplotlib = latex_plt(matplotlib)

# Get home directory
paths = os.getcwd()
path = paths.split(os.path.sep)
home_dir = path[1]

# ------------------------
# 1. Load and prepare data
# ------------------------

# Load data
df_exp2 = pd.read_pickle('al_data/data_prepr_2.pkl')
all_id = list(set(df_exp2['subj_num']))  # ID for each participant
n_subj = len(all_id)  # number of subjects

# Age group for model input
age_group = df_exp2.groupby(['subj_num'])['age_group'].first()
age_group = age_group.reset_index(drop=False)

# Determine if parameter-grid search should be done or not
with_optim = False

# --------------------------
# 2. Simulate baseline model
# --------------------------

# General simulation parameters
# -----------------------------

model_params = pd.DataFrame(columns=['omikron_0', 'omikron_1', 'h', 's', 'u', 'sigma_H', 'cost_const', 'cost_exp',
                                     'subj_num', 'age_group'], index=range(n_subj))
model_params['omikron_0'] = np.nan
model_params['omikron_1'] = np.nan
model_params['h'] = 0.1
model_params['s'] = 1
model_params['u'] = 0
model_params['sigma_H'] = 0
model_params['cost_unit'] = 0
model_params['cost_exp'] = 1
model_params['beta_0'] = 0.5
model_params['beta_1'] = -5
model_params['subj_num'] = all_id
model_params['age_group'] = age_group['age_group']

# Simulate baseline model
all_pers_opt, all_est_errs_opt, all_data_opt = simulation_loop_motor(df_exp2, model_params, n_subj, n_sim=1)
all_pers_opt.name, all_est_errs_opt.name = "priorpred_motor_pers_opt", "priorpred_motor_est_errs_opt"

# -------------------
# 3. Parameter search
# -------------------

# Take pre-defined parameter values...
if not with_optim:

    best_params_pers_ya = [0.5, 1.0]
    best_params_pers_ch_oa = [0.75, 1.2]
    best_params_anchor_ya = [0.1, 1.0]
    best_params_anchor_ch_oa = [0.1, 1.4]

# ... or grid-search-based values
else:

    # Parameter values
    cost_unit = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
    cost_exp = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

    # Number of simulations
    n_sim = len(cost_unit) * len(cost_exp)

    # Initialize variables
    pers = np.full(n_sim, np.nan)
    anchor = np.full(n_sim, np.nan)
    counter = 0
    param_list = []

    # Cycle over cost-unit parameters
    for i in range(0, len(cost_unit)):

        # Cycle over cost-exponent parameters
        for j in range(0, len(cost_exp)):
            # Extract parameters
            model_params['cost_unit'] = cost_unit[i]
            model_params['cost_exp'] = cost_exp[j]

            # Store parameter tuples
            param_list.append([cost_unit[i], cost_exp[j]])

            # Run simulation
            # Control random seed for reproducible results
            np.random.seed(seed=1)
            all_pers_pers, all_est_errs_pers, all_data_pers = simulation_loop_motor(df_exp2, model_params,
                                                                                    n_subj, n_sim=1)

            # Perseveration probability
            pers[counter] = np.mean(all_pers_pers['noPush'])

            # Anchoring bias
            df_reg_anchor = compute_anchoring_bias(n_subj, all_data_pers)
            anchor[counter] = np.mean(df_reg_anchor['bucket_bias'])

            # Update counter
            counter += 1

    # Find best parameters
    # --------------------

    # Perseveration younger adults
    pers_effect, idx = find_nearest(pers, value=0.2)
    best_params_pers_ya = param_list[idx]
    print("Best params for perseveration in younger adults (" + str(pers_effect) + "): " +
          str(best_params_pers_ya))

    # Perseveration children and older adults
    pers_effect, idx = find_nearest(pers, value=0.4)
    best_params_pers_ch_oa = param_list[idx]
    print("Best params for perseveration in children /older adults (" + str(pers_effect) + "): " +
          str(best_params_pers_ch_oa))

    # Anchoring bias younger adults
    anchor_effect, idx = find_nearest(anchor, value=0.0)
    best_params_anchor_ya = param_list[idx]
    print("Best params for anchoring bias in younger adults (" + str(anchor_effect) + "): " +
          str(best_params_anchor_ya))

    # Anchoring bias children and older adults
    anchor_effect, idx = find_nearest(anchor, value=0.2)
    best_params_anchor_ch_oa = param_list[idx]
    print("Best params for anchoring bias in children /older adults (" + str(anchor_effect) + "): " +
          str(best_params_anchor_ch_oa))

# ------------------------------------------------
# 4. Simulate perseveration and anchoring behavior
# ------------------------------------------------

# Simulate perseveration model
model_params.loc[model_params['age_group'] == 1, 'cost_unit'] = best_params_pers_ch_oa[0]
model_params.loc[model_params['age_group'] == 3, 'cost_unit'] = best_params_pers_ya[0]
model_params.loc[model_params['age_group'] == 4, 'cost_unit'] = best_params_pers_ch_oa[0]
model_params.loc[model_params['age_group'] == 1, 'cost_exp'] = best_params_pers_ch_oa[1]
model_params.loc[model_params['age_group'] == 3, 'cost_exp'] = best_params_pers_ya[1]
model_params.loc[model_params['age_group'] == 4, 'cost_exp'] = best_params_pers_ch_oa[1]
all_pers_pers, all_est_errs_pers, all_data_pers = simulation_loop_motor(df_exp2, model_params, n_subj, n_sim=1)
all_pers_pers.name, all_est_errs_pers.name = "priorpred_motor_pers_pers", "priorpred_motor_est_errs_pers"

# Simulate anchoring model
model_params.loc[model_params['age_group'] == 1, 'cost_unit'] = best_params_anchor_ch_oa[0]
model_params.loc[model_params['age_group'] == 3, 'cost_unit'] = best_params_anchor_ya[0]
model_params.loc[model_params['age_group'] == 4, 'cost_unit'] = best_params_anchor_ch_oa[0]
model_params.loc[model_params['age_group'] == 1, 'cost_exp'] = best_params_anchor_ch_oa[1]
model_params.loc[model_params['age_group'] == 3, 'cost_exp'] = best_params_anchor_ya[1]
model_params.loc[model_params['age_group'] == 4, 'cost_exp'] = best_params_anchor_ch_oa[1]
all_pers_anchor, all_est_errs_anchor, all_data_anchor = simulation_loop_motor(df_exp2, model_params, n_subj, n_sim=1)
all_pers_anchor.name, all_est_errs_anchor.name = \
    "priorpred_motor_pers_anchor", "priorpred_motor_est_errs_anchor"

# -------------
#  5. Save data
# -------------

# Baseline parameters
safe_save_dataframe(all_pers_opt, None, overleaf=False)

# Perseveration parameters
safe_save_dataframe(all_pers_pers, None, overleaf=False)

# Anchoring parameters
safe_save_dataframe(all_pers_anchor, None, overleaf=False)

# Baseline parameters
safe_save_dataframe(all_est_errs_opt, None, overleaf=False)

# Perseveration parameters
safe_save_dataframe(all_est_errs_pers, None, overleaf=False)

# Anchoring parameters
safe_save_dataframe(all_est_errs_anchor, None, overleaf=False)

# Baseline parameters
df_reg_opt = compute_anchoring_bias(n_subj, all_data_opt)
df_reg_opt.name = "priorpred_motor_df_reg_opt"
safe_save_dataframe(df_reg_opt, None, overleaf=False)

# Perseveration parameters
df_reg_pers = compute_anchoring_bias(n_subj, all_data_pers)
df_reg_pers.name = "priorpred_motor_df_reg_pers"
safe_save_dataframe(df_reg_pers, None, overleaf=False)

# Anchoring parameters
df_reg_anchor = compute_anchoring_bias(n_subj, all_data_anchor)
df_reg_anchor.name = "priorpred_motor_df_reg_anchor"
safe_save_dataframe(df_reg_anchor, None, overleaf=False)
