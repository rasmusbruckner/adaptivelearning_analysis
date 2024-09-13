""" RBM validation: Additional analyses to validate the model

    1. Load data
    2. Plot model simulation
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from AlAgentVarsRbm import AgentVars
from AlAgentRbm import AlAgent
from al_task_agent_int_rbm import task_agent_int
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

# ------------------------
# 2. Plot model simulation
# ------------------------

# Set agent variables
agent_vars = AgentVars()

# Parameters
sel_coeffs = [0.01, 0.0, -30, -1.5, agent_vars.h, agent_vars.s, agent_vars.u, agent_vars.q, agent_vars.sigma_H, 0]

# Update agent variables
agent_vars.h = sel_coeffs[4]
agent_vars.s = sel_coeffs[5]
agent_vars.u = np.exp(sel_coeffs[6])  # we take the exponent
agent_vars.q = sel_coeffs[7]
agent_vars.sigma_H = sel_coeffs[8]

# Create agent-object instance
agent = AlAgent(agent_vars)

# Control random seed for reproducible results
np.random.seed(seed=1)

# Simulate results
which_exp = 2
_, df_data = task_agent_int(which_exp, df_subj, agent, agent_vars, sel_coeffs, sim=True)

# Plot results
plot_validation_results(df_data, df_subj)

# Delete unnecessary axes
sns.despine()

# Show plot
plt.ioff()
plt.show()
