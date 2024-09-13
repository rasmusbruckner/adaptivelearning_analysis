""" Resource-only-model validation: Additional analyses to validate the model

    1. Load data
    2. Simulate data based on sampling model (SM) and show validation plots
    3. Plot total number of samples as a function of absolute prediction error
    4. Plot model across task
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
from AlAgentVarsSampling import AgentVarsSampling
from AlAgentSampling import AlAgentSampling
from al_task_agent_int_resource_only import task_agent_int_resource_only
from al_utilities import get_df_subj
from al_plot_utils import latex_plt, cm2inch, plot_validation_results


# Update matplotlib to use Latex and to change some defaults
os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2016/bin/x86_64-darwin'
matplotlib = latex_plt(matplotlib)

# Get home directory
paths = os.getcwd()
path = paths.split(os.path.sep)
home_dir = path[1]

# ------------
# 1. Load data
# ------------

# Data of follow-up experiment
df_exp2 = pd.read_pickle('al_data/data_prepr_2.pkl')
df_exp2['v_t'] = 0  # turn off catch trials

# -----------------------------------------------------------------------
# 2. Simulate data based on sampling model (SM) and show validation plots
# -----------------------------------------------------------------------

# Extract subject-specific data frame
subject = 0
df_subj = get_df_subj(df_exp2, subject)

# Agent variables object
agent_vars = AgentVarsSampling()

# Set agent parameters
agent_vars.criterion = 0.04
agent_vars.n_samples = 20
agent_vars.model_sat = False
agent_vars.burn_in = 0
agent_vars.sigma = 17.5
agent = AlAgentSampling(agent_vars)

# Run task-agent interaction
df_data = task_agent_int_resource_only(df_subj, agent, agent_vars, show_pbar=True)

# Plot validation
plot_validation_results(df_data, df_subj)

# --------------------------------------------------------------------------
# 3. Plot total number of samples as a function of absolute prediction error
# --------------------------------------------------------------------------

plt.figure()
plt.plot(abs(np.array(df_data['delta_t'])), np.array(df_data['tot_samples']),  'o')
plt.xlabel('Prediction Error')
plt.ylabel('Total Number of Samples')
sns.despine()

# -------------------------
# 4. Plot model across task
# -------------------------

# Extract subject-specific data frame
subject = 0
df_subj = get_df_subj(df_exp2, subject)

# Agent variables object
agent_vars = AgentVarsSampling()

# Set agent parameters
agent_vars.n_samples = 20
agent_vars.model_sat = False
agent_vars.burn_in = 0
agent_vars.sigma = 17.5
agent_vars.criterion = 0.04
agent = AlAgentSampling(agent_vars)

# Run task-agent interaction
df_data_sm = task_agent_int_resource_only(df_subj, agent, agent_vars, show_pbar=True, seed=1)

# Plot simulation
# ---------------

# Size of figure
fig_height = 8
fig_width = 15

# Y-label distance
ylabel_dist = -0.1

# Turn interactive plotting mode on for debugger
plt.ion()

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))
f.canvas.draw()

# Create plot grid
gs_0 = gridspec.GridSpec(1, 4, wspace=0.5, hspace=0.7, top=0.95, bottom=0.1, left=0.125, right=0.95)

# Create subplot grid
gs_01 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs_0[:, 0:3], hspace=0.5)

# Indicate plot range and x-axis
plot_range = (0, 400)
x = np.linspace(0, plot_range[1]-plot_range[0]-1, plot_range[1]-plot_range[0])

dark_green = "#1E6F5C"
light_green = "#289672"

# Mean, outcomes, and predictions
ax_10 = plt.Subplot(f, gs_01[0:2, 0])
f.add_subplot(ax_10)
ax_10.plot(x, np.array(df_exp2['mu_t'][plot_range[0]:plot_range[1]]), '--',
           x, np.array(df_exp2['x_t'][plot_range[0]:plot_range[1]]), '.', color="#090030")
ax_10.plot(x, np.array(df_data_sm['sim_b_t'][plot_range[0]:plot_range[1]]), linewidth=1, color=dark_green, alpha=1)
ax_10.set_ylabel('Position')
ax_10.yaxis.set_label_coords(ylabel_dist, 0.5)
ax_10.legend(["Helicopter", "Outcome", "RBM", "SM"], loc='center left', framealpha=0.8, bbox_to_anchor=(1, 0.5))
ax_10.set_ylim(-9, 309)
ax_10.set_xticklabels([])

# Prediction errors
ax_11 = plt.Subplot(f, gs_01[2, 0])
f.add_subplot(ax_11)
ax_11.plot(x, np.array(df_data_sm['delta_t'][plot_range[0]:plot_range[1]]), linewidth=1, color="#090030", alpha=1)
ax_11.set_xticklabels([])
ax_11.set_ylabel('Prediction Error')
ax_11.yaxis.set_label_coords(ylabel_dist, 0.5)
ax_11.legend(["RBM", "SM"],  loc='center left', framealpha=0.8, bbox_to_anchor=(1, 0.5))

# Relative uncertainty, changepoint probability, and learning rate
ax_12 = plt.Subplot(f, gs_01[3, 0])
f.add_subplot(ax_12)
ax_12.plot(x, np.array(df_data_sm['tau_t'][plot_range[0]:plot_range[1]]), linewidth=1, color="#04879c", alpha=1)
ax_12.plot(x, np.array(df_data_sm['omega_t'][plot_range[0]:plot_range[1]]), linewidth=1, color="#0c3c78", alpha=1)
ax_12.legend(['RU RBM', 'CPP RBM', 'RU SM', 'CPP SM'], loc='center left', framealpha=0.8, bbox_to_anchor=(1, 0.5))
ax_12.set_xlabel('Trial')
ax_12.set_ylabel('Variable')
ax_12.yaxis.set_label_coords(ylabel_dist, 0.5)

# Delete unnecessary axes
sns.despine()

# Show plot
plt.ioff()
plt.show()
