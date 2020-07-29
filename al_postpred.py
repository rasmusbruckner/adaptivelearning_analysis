""" Posterior predictions

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
from al_simulation import simulation_loop

# Set random number generator for reproducible results
np.random.seed(123)

# ------------
# 1. Load data
# ------------

# Data first experiment
df_exp1 = pd.read_pickle('al_data/data_prepr_1.pkl')

# Data follow-up experiment
df_exp2 = pd.read_pickle('al_data/data_prepr_2.pkl')

# Parameter estimates first experiment
model_exp1 = pd.read_pickle('al_data/estimates_first_exp_25_sp.pkl')

# Parameter estimates second experiment
model_exp2 = pd.read_pickle('al_data/estimates_follow_up_exp_25_sp.pkl')

# ------------------
# 2. Run simulations
# ------------------

# First experiment
# ----------------

# Extract ID's and number of participants
sub_sel = df_exp1['subj_num']  # ID for each trial
n_subj = len(list(set(sub_sel))) # Number of participants

# First experiment with perseveration
n_sim = 3  # determine number of simulation cycles
sim_pers = True
all_pers, all_est_errs = simulation_loop(df_exp1, model_exp1, n_subj, sim_pers,
                                         which_exp=1, sim_bucket_bias=False, n_sim=n_sim)
all_pers.to_pickle('al_data/postpred_exp1_pers.pkl')
all_est_errs.to_pickle('al_data/postpred_exp1_est_err.pkl')

# First experiment without perseveration
sim_pers = False
_, all_est_errs = simulation_loop(df_exp1, model_exp1, n_subj, sim_pers,
                                  which_exp=1, sim_bucket_bias=False, n_sim=n_sim)
all_est_errs.to_pickle('al_data/hyp_est_errs_exp1_no_pers.pkl')

# First experiment, one cycle with perseveration to plot actual and predicted single-trial updates and predictions
n_sim = 1
sim_pers = True
_, _ = simulation_loop(df_exp1, model_exp1, n_subj, sim_pers, which_exp=1,
                       sim_bucket_bias=False, n_sim=n_sim, plot_data=True)

# Second experiment
# -----------------

# Extract ID's and number of participants
sub_sel = df_exp2['subj_num']  # ID for each trial
n_subj = len(list(set(sub_sel)))  # number of participants

# Second experiment with perseveration
n_sim = 3
sim_pers = True
all_pers, all_est_errs = simulation_loop(df_exp2, model_exp2, n_subj, sim_pers,
                                         which_exp=2, sim_bucket_bias=True, n_sim=n_sim)
all_pers.to_pickle('al_data/postpred_exp2_pers.pkl')
all_est_errs.to_pickle('al_data/postpred_exp2_est_err.pkl')

# Second experiment, one cycle with perseveration to plot actual and predicted single-trial updates and predictions
n_sim = 1
sim_pers = True
all_pers, all_est_errs = simulation_loop(df_exp2, model_exp2, n_subj, sim_pers,
                                         which_exp=2, sim_bucket_bias=True, n_sim=n_sim, plot_data=True)
