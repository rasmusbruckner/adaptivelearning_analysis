""" Parameter recovery analysis

1. Load data
For both experiments:
2. Sample random parameter values for recovery and simulate data
3. Estimate parameters
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from AlEstimation import AlEstimation
from Estimation.AlEstVars import AlEstVars
from AlAgentVarsRbm import AgentVars
from al_simulation_rbm import simulation
from al_utilities import safe_save_dataframe


# Set random number generator for reproducible results
np.random.seed(123)

# ------------
# 1. Load data
# ------------

# Data of first experiment
df_exp1 = pd.read_pickle('al_data/data_prepr_1.pkl')

# Data of follow-up experiment
df_exp2 = pd.read_pickle('al_data/data_prepr_2.pkl')

# Extract participant information of experiment 1
all_id_exp1 = list(set(df_exp1['subj_num']))  # ID for each participant
n_subj_exp1 = len(all_id_exp1)  # number of participants

# Extract participant information of experiment 2
all_id_exp2 = list(set(df_exp2['subj_num']))  # ID for each participant
n_subj_exp2 = len(all_id_exp2)  # number of participants

# Model parameters of first experiment
model_results_exp1 = pd.read_pickle('al_data/estimates_first_exp_10_sp.pkl')

# Model parameters of follow-up experiment
model_results_exp2 = pd.read_pickle('al_data/estimates_follow_up_exp_10_sp.pkl')

# Parameter names for recovery analysis of first experiment
est_vars = AlEstVars()
param_names_exp1 = [est_vars.omikron_0, est_vars.omikron_1, est_vars.b_0, est_vars.b_1, est_vars.h, est_vars.s,
                    est_vars.u, est_vars.q, est_vars.sigma_H]

# Parameter names for recovery analysis of follow-up experiment
param_names_exp2 = [est_vars.omikron_0, est_vars.omikron_1, est_vars.b_0, est_vars.b_1, est_vars.h, est_vars.s,
                    est_vars.u, est_vars.sigma_H, est_vars.d]

# ----------------------------------------------------------------
# 2. Sample random parameter values for recovery and simulate data
# ----------------------------------------------------------------


def recovery_simulation(model_results, param_names, df_exp, n_subj, which_exp):
    """ This function simulates data for the recovery analysis

    :param model_results: Model parameters of the current experiment
    :param param_names: Parameter names for recovery analysis
    :param df_exp:
    :param n_subj: Number of participants
    :param which_exp: Current experiment
    :return: df_recov: Data frame for recovery
    """

    # Initialize data frame that contains random parameter values for recovery
    sim_params = pd.DataFrame(columns=param_names)

    # Cycle over parameters
    for i in range(0, len(param_names)):

        # Open figure to visually inspect distribution of real and randomly determined parameters
        plt.figure()
        real_params_dist = sns.histplot(model_results[param_names[i]], kde=True, stat="density", bins=int(180/5),
                                        kde_kws=dict(cut=3), line_kws={'linewidth': 4}, edgecolor='black',
                                        color='darkblue')
        real_params_density = real_params_dist.get_lines()[0].get_data()
        plt.title(param_names[i])
        x = np.nan
        y = np.nan

        # Adjust density to make sure parameters are drawn within the certain parameter boundaries:
        # Select only those parameters that are within the range of interest (x). For these values
        # extract density (y) and normalize density for parameter sampling done below.
        # ----------------------------------------------------------------------------------------
        if param_names[i] == 'omikron_0':
            x = real_params_density[0][(real_params_density[0] >= 0.0) & (real_params_density[0] <= 10.0)]
            y = real_params_density[1][(real_params_density[0] >= 0.0) & (real_params_density[0] <= 10.0)]
            y = y / sum(y)
        elif param_names[i] == 'omikron_1' or param_names[i] == 'h' or param_names[i] == 's':
            x = real_params_density[0][(real_params_density[0] >= 0.0) & (real_params_density[0] <= 1.0)]
            y = real_params_density[1][(real_params_density[0] >= 0.0) & (real_params_density[0] <= 1.0)]
            y = y / sum(y)
        elif param_names[i] == 'b_0':
            x = real_params_density[0][(real_params_density[0] >= -30.0) & (real_params_density[0] <= 30.0)]
            y = real_params_density[1][(real_params_density[0] >= -30.0) & (real_params_density[0] <= 30.0)]
            y = y / sum(y)
        elif param_names[i] == 'b_1':
            x = real_params_density[0][(real_params_density[0] >= -1.5) & (real_params_density[0] <= 1.0)]
            y = real_params_density[1][(real_params_density[0] >= -1.5) & (real_params_density[0] <= 1.0)]
            y = y / sum(y)
        elif param_names[i] == 'u':
            x = real_params_density[0][(real_params_density[0] >= -2.0) & (real_params_density[0] <= 20.0)]
            y = real_params_density[1][(real_params_density[0] >= -2.0) & (real_params_density[0] <= 20.0)]
            y = y / sum(y)
        elif param_names[i] == 'sigma_H':
            x = real_params_density[0][(real_params_density[0] >= 0.0) & (real_params_density[0] <= 32.0)]
            y = real_params_density[1][(real_params_density[0] >= 0.0) & (real_params_density[0] <= 32.0)]
            y = y / sum(y)
        elif which_exp == 1 and param_names[i] == 'q':
            x = real_params_density[0][(real_params_density[0] >= -0.5) & (real_params_density[0] <= 0.5)]
            y = real_params_density[1][(real_params_density[0] >= -0.5) & (real_params_density[0] <= 0.5)]
            y = y / sum(y)
        elif which_exp == 2 and param_names[i] == 'd':
            x = real_params_density[0][(real_params_density[0] >= 0.0) & (real_params_density[0] <= 0.5)]
            y = real_params_density[1][(real_params_density[0] >= 0.0) & (real_params_density[0] <= 0.5)]
            y = y / sum(y)

        # Randomly draw parameters based on density estimates
        samples = np.random.choice(x, len(model_results), p=y)
        sns.histplot(samples, kde=True, stat="density", bins=int(180/5), kde_kws=dict(cut=3), line_kws={'linewidth': 4},
                     edgecolor='black', color='darkblue')

        # Add sampled parameters to simulation data frame
        sim_params[param_names[i]] = samples

    # Add age group and subject number to simulation data frame
    sim_params['age_group'] = model_results['age_group'].copy()
    sim_params['subj_num'] = model_results['subj_num'].copy()

    # Set reward parameter to 0 for experiment 2 (not used there)
    if which_exp == 2:
        sim_params['q'] = 0

    # Simulate data based on sampled parameters
    # -----------------------------------------

    est_err, sim_pers_prob, df_sim, true_params = simulation(df_exp, sim_params, n_subj, sim_pers=True,
                                                             which_exp=which_exp)

    # Save true parameters that were used for recovery
    if which_exp == 1:
        true_params.to_pickle('al_data/true_params_recov_exp1.pkl')
    else:
        true_params.to_pickle('al_data/true_params_recov_exp2.pkl')

    # Create new data frame for recovery
    df_recov = pd.DataFrame(index=range(0, len(df_sim)), dtype='float')
    df_recov['subj_num'] = df_exp['subj_num']
    df_recov['age_group'] = df_exp['age_group']
    df_recov['new_block'] = df_exp['new_block']
    df_recov['x_t'] = df_exp['x_t']
    df_recov['a_t'] = df_sim['sim_a_t']
    df_recov['delta_t'] = df_sim['delta_t']
    df_recov['v_t'] = df_exp['v_t']
    df_recov['r_t'] = df_exp['r_t']
    df_recov['sigma'] = df_sim['sigma']
    df_recov['mu_t'] = df_exp['mu_t']
    df_recov['b_t'] = df_sim['sim_b_t']
    df_recov['cond'] = df_exp['cond']
    if which_exp == 2:
        df_recov['y_t'] = df_exp['y_t']

    return df_recov


# ----------------
# First experiment
# ----------------

df_recov_exp1 = recovery_simulation(model_results_exp1, param_names_exp1, df_exp1, n_subj_exp1, 1)

# Call AlEstVars Object
est_vars = AlEstVars()
est_vars.n_subj = n_subj_exp1
est_vars.n_ker = 4  # number of kernels for estimation
est_vars.n_sp = 10  # number of random starting points
est_vars.rand_sp = True
est_vars.which_vars = {est_vars.omikron_0: True,  # motor noise
                       est_vars.omikron_1: True,  # learning-rate noise
                       est_vars.b_0: True,  # logistic function intercept
                       est_vars.b_1: True,  # logistic function slope
                       est_vars.h: True,  # hazard rate
                       est_vars.s: True,  # surprise sensitivity
                       est_vars.u: True,  # uncertainty underestimation
                       est_vars.q: True,  # reward bias
                       est_vars.sigma_H: True,  # catch trials
                       est_vars.d: False,  # bucket shift
                       }

est_vars.use_prior = True
al_estimation = AlEstimation(est_vars)

# Call AgentVars object
agent_vars = AgentVars()

# Estimate parameters and save data
results_df = al_estimation.parallel_estimation(df_recov_exp1, agent_vars)

results_df.name = 'param_recov_exp1'
safe_save_dataframe(results_df, 'index', overleaf=False)

# -----------------
# Second experiment
# -----------------

df_recov_exp2 = recovery_simulation(model_results_exp2, param_names_exp2, df_exp2, n_subj_exp2, 2)

est_vars.which_vars = {est_vars.omikron_0: True,  # motor noise
                       est_vars.omikron_1: True,  # learning-rate noise
                       est_vars.b_0: True,  # logistic-function intercept
                       est_vars.b_1: True,  # logistic-function slope
                       est_vars.h: True,  # hazard rate
                       est_vars.s: True,  # surprise sensitivity
                       est_vars.u: True,  # uncertainty underestimation
                       est_vars.q: False,  # reward
                       est_vars.sigma_H: True,  # catch trials
                       est_vars.d: True,  # anchoring bias
                       }

# Update number of participants
est_vars.n_subj = n_subj_exp2

# Specify that experiment 2 is modeled
est_vars.which_exp = 2

# Call AlEstimation object
al_estimation = AlEstimation(est_vars)

# Estimate parameters and save data
results_df = al_estimation.parallel_estimation(df_recov_exp2, agent_vars)

results_df.name = 'param_recov_exp2'
safe_save_dataframe(results_df, 'index', overleaf=False)

# Show figures
plt.show()
