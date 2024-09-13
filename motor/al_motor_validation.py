""" Motor-Model Validation: Additional analyses to validate the model

    1. Load data
    2. Plot model simulations for different parameter settings
    3. Plot perseveration function of motor model
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.special import expit
from AlAgentVarsRbm import AgentVars
from AlAgentRbm import AlAgent
from motor.al_task_agent_int_motor import task_agent_int_motor
from al_utilities import get_df_subj
from al_plot_utils import latex_plt, plot_validation_results


# Update matplotlib to use Latex and to change some defaults
matplotlib = latex_plt(matplotlib)
matplotlib.use('Qt5Agg')


# Get home directory
paths = os.getcwd()
path = paths.split(os.path.sep)
home_dir = path[1]

# Turn on interactive plotting mode
plt.ion()

# ------------
# 1. Load data
# ------------

# Data of follow-up experiment
df_exp2 = pd.read_pickle('al_data/data_prepr_2.pkl')
df_exp2['v_t'] = 0  # turn off catch trials

# Extract subject-specific data frame
subject = 0
df_subj = get_df_subj(df_exp2, subject)

# ----------------------------------------------------------
# 2. Plot model simulations for different parameter settings
# ----------------------------------------------------------

# Baseline parameters: No motor costs
# -----------------------------------

sel_coeffs = [np.nan, np.nan, 0.5, -5, 0.1, 1, 0, 0, 0.0, 1]

# Set agent variables
agent_vars = AgentVars()
agent_vars.h = sel_coeffs[4]
agent_vars.s = sel_coeffs[5]
agent_vars.u = np.exp(sel_coeffs[6])
agent_vars.q = 0
agent_vars.sigma_H = sel_coeffs[7]

# Create agent-object instance
agent = AlAgent(agent_vars)

# Control random seed for reproducible results
np.random.seed(seed=1)

# Simulate results
df_data = task_agent_int_motor(df_subj, agent, agent_vars, sel_coeffs)

# Plot results
plot_validation_results(df_data, df_subj)
# --> Results show very low perseveration and anchoring effects

# Perseveration parameters with very high motor costs
# ----------------------------------------------------

sel_coeffs = [np.nan, np.nan, 0.5, -5, 0.1, 1, 0, 0, 0.75, 1.4]

# Set agent variables
agent_vars = AgentVars()
agent_vars.h = sel_coeffs[4]
agent_vars.s = sel_coeffs[5]
agent_vars.u = np.exp(sel_coeffs[6])
agent_vars.q = 0
agent_vars.sigma_H = sel_coeffs[7]

# Create agent-object instance
agent = AlAgent(agent_vars)

# Control random seed for reproducible results
np.random.seed(seed=1)

# Simulate results
df_data = task_agent_int_motor(df_subj, agent, agent_vars, sel_coeffs)

# Plot results
plot_validation_results(df_data , df_subj)
# --> Results show high perseveration but an anchoring effect that is way too strong

# Anchoring parameters with lower motor costs
# -------------------------------------------

sel_coeffs = [np.nan, np.nan, 0.5, -5, 0.1, 1, 0, 0, 0.1, 1.4]

# Set agent variables
agent_vars = AgentVars()
agent_vars.h = sel_coeffs[4]
agent_vars.s = sel_coeffs[5]
agent_vars.u = np.exp(sel_coeffs[6])
agent_vars.q = 0
agent_vars.sigma_H = sel_coeffs[7]

# Create agent-object instance
agent = AlAgent(agent_vars)

# Control random seed for reproducible results
np.random.seed(seed=1)

# Simulate results
df_data = task_agent_int_motor(df_subj, agent, agent_vars, sel_coeffs)

# Plot results
plot_validation_results(df_data, df_subj)
# --> Results show a realistic anchoring effect but perseveration is way too low

# ---------------------------------------------
# 3. Plot perseveration function of motor model
# ---------------------------------------------

# Initialize perseveration-frequency arrays
pers_prob = np.full(50, np.nan)

# Range of predicted update
pe = np.linspace(1, 50)

# Perseveration parameters
beta_0 = 0.5
beta_1 = -5

# Cycle over range of predicted updates
for i in range(0, len(pers_prob)):

    # Compute perseveration probability
    pers_prob[i] = expit(beta_1 * (i-beta_0))

# Plot logistic function
plt.figure()
plt.plot(pe, pers_prob, color="r")
plt.ylabel('Perseveration Probability')
plt.xlabel('Predicted Update')
plt.xlim([0, 20])
sns.despine()

# Show plots
plt.ioff()
plt.show()
