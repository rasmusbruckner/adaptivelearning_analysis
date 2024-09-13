""" Posterior predictions to validate models

1. Load data
2. Run simulations
    - First experiment:
        - With perseveration (3 cycles)
        - Without perseveration (3 cycles)
        - With perseveration (1 cycle) to plot single-trial updates and predictions
    - Follow-up experiment:
        - With perseveration (3 cycles)
        - With perseveration (1 cycle) to plot single-trial updates and predictions
"""

import numpy as np
import pandas as pd
from al_simulation_rbm import simulation_loop
from al_utilities import safe_save_dataframe


# Set random number generator for reproducible results
np.random.seed(123)

# ------------
# 1. Load data
# ------------

# Data first experiment
df_exp1 = pd.read_pickle('al_data/data_prepr_1.pkl')

# Data follow-up experiment
df_exp2 = pd.read_pickle('al_data/data_prepr_2.pkl')

# Data control experiment
df_exp3 = pd.read_pickle('al_data/data_prepr_3.pkl')

# Parameter estimates first experiment
model_exp1 = pd.read_pickle('al_data/estimates_first_exp_10_sp.pkl')

# Parameter estimates second experiment
model_exp2 = pd.read_pickle('al_data/estimates_follow_up_exp_10_sp.pkl')
model_exp2['q'] = 0

# ------------------
# 2. Run simulations
# ------------------

# First experiment
# ----------------

# Extract ID's and number of participants
sub_sel = df_exp1['subj_num']  # ID for each trial
n_subj = len(list(set(sub_sel)))  # Number of participants

# First experiment with perseveration
n_sim = 3  # determine number of simulation cycles
sim_pers = True
all_pers, all_est_errs, _ = simulation_loop(df_exp1, model_exp1, n_subj, sim_pers, sim_bucket_bias=False, n_sim=n_sim)

# Save perseveration data
all_pers.name = "postpred_exp1_pers"
safe_save_dataframe(all_pers, 'index', overleaf=False)

# Save estimation-error data
all_est_errs.name = "postpred_exp1_est_err"
safe_save_dataframe(all_est_errs, 'index', overleaf=False)

# First experiment without perseveration
# --------------------------------------

sim_pers = False
_, all_est_errs, _ = simulation_loop(df_exp1, model_exp1, n_subj, sim_pers, sim_bucket_bias=False, n_sim=n_sim)

# Save estimation-error data
all_est_errs.name = "hyp_est_errs_exp1_no_pers"
safe_save_dataframe(all_est_errs, 'index', overleaf=False)

# First experiment, one cycle with perseveration to plot actual and predicted single-trial updates and predictions
n_sim = 1
sim_pers = True
_, _, _ = simulation_loop(df_exp1, model_exp1, n_subj, sim_pers, sim_bucket_bias=False, n_sim=n_sim, plot_data=True)

# Second experiment
# -----------------

# Extract ID's and number of participants
sub_sel = df_exp2['subj_num']  # ID for each trial
n_subj = len(list(set(sub_sel)))  # number of participants

# Second experiment with perseveration
n_sim = 3
sim_pers = True
all_pers, all_est_errs, _ = simulation_loop(df_exp2, model_exp2, n_subj, sim_pers, which_exp=2, n_sim=n_sim)

# Save perseveration data
all_pers.name = "postpred_exp2_pers"
safe_save_dataframe(all_pers, 'index', overleaf=False)

# Save estimation error data
all_est_errs.name = "postpred_exp2_est_err"
safe_save_dataframe(all_est_errs, 'index', overleaf=False)

# Second experiment, one cycle with perseveration to plot actual and predicted single-trial updates and predictions
n_sim = 1
sim_pers = True
all_pers, all_est_errs, _ = simulation_loop(df_exp2, model_exp2, n_subj, sim_pers, which_exp=2, n_sim=n_sim,
                                            plot_data=True)
# Third experiment
# ----------------
# Simulations to get single-trial plots for validation purposes only. Also ignoring anchoring for now since we
# don't present simulations for experiment 3.


# Extract ID's and number of participants
sub_sel = df_exp3['subj_num']  # ID for each trial
n_subj = len(list(set(sub_sel)))  # number of participants

df_exp3['r_t'] = np.nan
df_exp3['sigma'] = 17.5/3
df_exp3['v_t'] = False
df_exp3['c_t'] = False

model_exp3 = pd.DataFrame(columns=['omikron_0', 'omikron_1', 'b_0', 'b_1', 'h', 's', 'u', 'q', 'sigma_H', 'age_group', 'subj_num'], index=range(n_subj))
model_exp3['omikron_0'] = 0.5
model_exp3['omikron_1'] = 0.0
model_exp3['b_0'] = 0.5
model_exp3['b_1'] = 0.0
model_exp3['h'] = 0.1
model_exp3['s'] = 1
model_exp3['u'] = 0.0
model_exp3['q'] = 0.0
model_exp3['sigma_H'] = 0.0
model_exp3['age_group'] = 3
model_exp3['subj_num'] = np.arange(n_subj)+1

n_sim = 1
sim_pers = False
simulation_loop(df_exp3, model_exp3, n_subj, sim_pers, which_exp=3, n_sim=n_sim, plot_data=True)

