""" This script runs the model estimation for Experiment 1 and 2
    In Experiment 1, we compare a model with and a model without perseveration
    In Experiment 2, we use a model that includes perseveration and a parameter
    that models the influence of the shifting bucket.

    1. Load data
    2. Prepare analysis
    3. Estimate models for experiment 1
    4. Estimate model for experiment 2
"""


import pandas as pd
from AlEstimation import AlEstimation
from AlEstVars import AlEstVars
from AlAgentVars import AgentVars

# -------------
# 1. Load data
# -------------

# Load data from first experiment
df_exp1 = pd.read_pickle('al_data/data_prepr_1.pkl')

# Load data from second experiment
df_exp2 = pd.read_pickle('al_data/data_prepr_2.pkl')

# Extract participant information of experiment 1
all_id_exp1 = list(set(df_exp1['subj_num']))  # ID for each participant
n_subj_exp1 = len(all_id_exp1)  # number of participants

# Extract participant information of experiment 2
all_id_exp2 = list(set(df_exp2['subj_num']))  # ID for each participant
n_subj_exp2 = len(all_id_exp2)  # number of participants

# -------------------
# 2. Prepare analysis
# -------------------

# Call AgentVars Object
agent_vars = AgentVars()

# Call AlEstVars object
est_vars = AlEstVars()
est_vars.n_subj = n_subj_exp1  # number of subjects
est_vars.n_ker = 4  # number of kernels for estimation
est_vars.n_sp = 25  # number of random starting points
est_vars.rand_sp = True  # use random starting points
est_vars.use_prior = True  # use weakly informative prior for uncertainty underestimation

# ------------------------------------
# 3. Estimate models for experiment 1
# ------------------------------------

# Reduced Bayesian model with perseveration
# -----------------------------------------

# Free parameters
est_vars.which_vars = {est_vars.omikron_0: True,  # motor noise
                       est_vars.omikron_1: True,  # learning-rate noise
                       est_vars.b_0: True,  # logistic-function intercept
                       est_vars.b_1: True,  # logistic-function slope
                       est_vars.h: True,  # hazard rate
                       est_vars.s: True,  # surprise sensitivity
                       est_vars.u: True,  # uncertainty underestimation
                       est_vars.q: True,  # reward bias
                       est_vars.sigma_H: True,  # catch trials
                       est_vars.d: False,  # bucket shift
                       }

# Specify that experiment 1 is modeled
est_vars.which_exp = 1

# Call AlEstimation object
al_estimation = AlEstimation(est_vars)

# Estimate parameters and save data
results_df = al_estimation.parallel_estimation(df_exp1, agent_vars)
savename = 'al_data/estimates_first_exp_' + str(est_vars.n_sp) + '_sp.pkl'
results_df.to_pickle(savename)

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
                       est_vars.d: False,  # bucket shift
                       }

# Call AlEstimation object
al_estimation = AlEstimation(est_vars)

# Estimate parameters and save data
results_df = al_estimation.parallel_estimation(df_exp1, agent_vars)
savename = 'al_data/estimates_first_exp_no_pers_' + str(est_vars.n_sp) + '_sp.pkl'
results_df.to_pickle(savename)

# ----------------------------------
# 4. Estimate model for experiment 2
# ----------------------------------

# Free parameters
est_vars.which_vars = {est_vars.omikron_0: True,  # motor noise
                       est_vars.omikron_1: True,  # learning-rate noise
                       est_vars.b_0: True,  # logistic-function intercept
                       est_vars.b_1: True,  # logistic-function slope
                       est_vars.h: True,  # hazard rate
                       est_vars.s: True,  # surprise sensitivity
                       est_vars.u: True,  # uncertainty underestimation
                       est_vars.q: False,  # reward bias
                       est_vars.sigma_H: True,  # catch trials
                       est_vars.d: True,  # bucket shift
                       }

# Update number of participants
est_vars.n_subj = n_subj_exp2

# Specify that experiment 2 is modeled
est_vars.which_exp = 2

# Call AlEstimation object
al_estimation = AlEstimation(est_vars)

# Estimate parameters and save data
results_df = al_estimation.parallel_estimation(df_exp2, agent_vars)
savename = 'al_data/estimates_follow_up_exp_' + str(est_vars.n_sp) + '_sp.pkl'
results_df.to_pickle(savename)
