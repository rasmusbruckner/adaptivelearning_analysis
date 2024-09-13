""" Examining uncertainty underestimation using the RBM and the perseverative RBM

This is a control analysis to clarify link between uncertainty underestimation and perseveration,
revealing that perseveration can be interpreted as uncertainty underestimation if we don't account
for it during model fitting.

1. Load data
2. Run simulations
3. Fit simulations without perseveration

"""

import numpy as np
from al_simulation_rbm import simulation_loop
import pandas as pd
from AlEstimation import AlEstimation
from Estimation.AlEstVars import AlEstVars
from AlAgentVarsRbm import AgentVars
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

# Parameter estimates first experiment
model_exp1 = pd.read_pickle('al_data/estimates_first_exp_10_sp.pkl')

# Parameter estimates second experiment
model_exp2 = pd.read_pickle('al_data/estimates_follow_up_exp_10_sp.pkl')
model_exp2['q'] = 0

# -------------------------------------
# 2. Run simulations with perseveration
# -------------------------------------

# Extract ID's and number of participants
sub_sel = df_exp1['subj_num']  # ID for each trial
n_subj = len(list(set(sub_sel)))  # Number of participants

model_exp1['u'] = 0

# First experiment with perseveration
n_sim = 1  # determine number of simulation cycles
sim_pers = True
all_pers, all_est_errs, df_model = simulation_loop(df_exp1, model_exp1, n_subj, sim_pers,
                                            which_exp=1, sim_bucket_bias=False, n_sim=n_sim)

# Create new data frame for recovery
df_sim = pd.DataFrame(index=range(0, len(df_model)), dtype='float')
df_sim['subj_num'] = df_model['subj_num'].copy()
df_sim['age_group'] = df_model['age_group'].copy()
df_sim['new_block'] = df_model['new_block'].copy()
df_sim['x_t'] = df_model['x_t'].copy()
df_sim['a_t'] = df_model['sim_a_t'].copy()
df_sim['delta_t'] = df_model['delta_t'].copy()
df_sim['v_t'] = df_model['v_t'].copy()
df_sim['r_t'] = df_model['r_t'].copy()
df_sim['sigma'] = df_model['sigma'].copy()
df_sim['mu_t'] = df_model['mu_t'].copy()
df_sim['b_t'] = df_model['sim_b_t'].copy()
df_sim['cond'] = df_model['cond'].copy()

# Save 
df_sim.name = 'unc_underest_sim'
safe_save_dataframe(df_sim, 'index', overleaf=False)

# ----------------------------------------
# 3. Fit simulations without perseveration
# ----------------------------------------

# Call AgentVars Object
agent_vars = AgentVars()

# Call AlEstVars object
est_vars = AlEstVars()
est_vars.n_subj = n_subj  # number of subjects
est_vars.n_ker = 4  # number of kernels for estimation
est_vars.n_sp = 10  # number of random starting points
est_vars.rand_sp = True  # use random starting points
est_vars.use_prior = True  # use weakly informative prior for uncertainty underestimation

# Reduced Bayesian model without perseveration
# --------------------------------------------

# Free parameters
est_vars.which_vars = {est_vars.omikron_0: True,  # motor noise
                       est_vars.omikron_1: True,  # learning-rate noise
                       est_vars.b_0: False,  # logistic-function intercept
                       est_vars.b_1: False,  # logistic-function slope
                       est_vars.h: True,  # hazard rate
                       est_vars.s: True,  # surprise sensitivity
                       est_vars.u: True,  # uncertainty underestimation
                       est_vars.q: True,  # reward bias
                       est_vars.sigma_H: True,  # catch trials
                       est_vars.d: False,  # anchoring bias
                       }

# Call AlEstimation object
al_estimation = AlEstimation(est_vars)

# Estimate parameters and save data
results_df = al_estimation.parallel_estimation(df_sim, agent_vars)

results_df.name = 'estimates_unc_underest'
safe_save_dataframe(results_df, 'index', overleaf=False)

