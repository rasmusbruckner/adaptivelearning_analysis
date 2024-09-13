""" Figure S16: Comparison between sampling model and reduced Bayesian model

    1. Load data
    2. Simulate data based on sampling model (SM)
    3. Simulate data based on reduced Bayesian model (RBM)
    4. Prepare figure
    5. Compare SM and RBM
    6. Save figure
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
from al_simulation_rbm import simulation
from sampling.AlAgentVarsSampling import AgentVarsSampling
from sampling.AlAgentSampling import AlAgentSampling
from sampling.al_task_agent_int_sampling import task_agent_int_sampling
from al_utilities import get_df_subj
from al_plot_utils import cm2inch, latex_plt


# Update matplotlib to use Latex and to change some defaults
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

# ---------------------------------------------
# 2. Simulate data based on sampling model (SM)
# ---------------------------------------------

# Extract subject-specific data frame
subject = 0
df_subj = get_df_subj(df_exp2, subject)

# Agent-variables object
agent_vars = AgentVarsSampling()

# Set agent parameters
agent_vars.n_samples = 2000
agent_vars.model_sat = False
agent_vars.burn_in = 200
agent_vars.sigma = 17.5
agent = AlAgentSampling(agent_vars)

# Run task-agent interaction
df_data_sm = task_agent_int_sampling(df_subj, agent, agent_vars, show_pbar=True, seed=1)

# ------------------------------------------------------
# 3. Simulate data based on reduced Bayesian model (RBM)
# ------------------------------------------------------

# Simulation parameters
n_sim = 1
model_params = pd.DataFrame(columns=['omikron_0', 'omikron_1', 'b_0', 'b_1', 'h', 's', 'u', 'q', 'sigma_H', 'd',
                                     'subj_num', 'age_group'])
model_params.loc[0, 'omikron_0'] = 0.01
model_params.loc[0, 'omikron_1'] = 0
model_params.loc[0, 'b_0'] = -30
model_params.loc[0, 'b_1'] = -1.5
model_params.loc[0, 'h'] = 0.1
model_params.loc[0, 's'] = 1
model_params.loc[0, 'u'] = 0
model_params.loc[0, 'q'] = 0
model_params.loc[0, 'sigma_H'] = 0.01
model_params.loc[0, 'd'] = 0.0
model_params.loc[0, 'subj_num'] = 1.0
model_params.loc[0, 'age_group'] = 0

# Normative model simulation
sim_pers = False  # no perseveration simulation
_, _, df_data_RBM, _, = simulation(df_exp2, model_params, n_sim, sim_pers)

# -----------------
# 4. Prepare figure
# -----------------

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

# ---------------------
# 5. Compare SM and RBM
# ---------------------

dark_green = "#1E6F5C"
light_green = "#289672"

# Mean, outcomes, and predictions
ax_10 = plt.Subplot(f, gs_01[0:2, 0])
f.add_subplot(ax_10)
ax_10.plot(x, np.array(df_exp2['mu_t'][plot_range[0]:plot_range[1]]), '--',
           x, np.array(df_exp2['x_t'][plot_range[0]:plot_range[1]]), '.', color="#090030")
ax_10.plot(x, np.array(df_data_RBM['sim_b_t'][plot_range[0]:plot_range[1]]), linestyle='-', linewidth=3,
           color=dark_green, alpha=0.4)
ax_10.plot(x, np.array(df_data_sm['sim_b_t'][plot_range[0]:plot_range[1]]), linewidth=1, color=dark_green, alpha=1)
ax_10.set_ylabel('Position')
ax_10.yaxis.set_label_coords(ylabel_dist, 0.5)
ax_10.legend(["Helicopter", "Outcome", "RBM", "SM"], loc='center left', framealpha=0.8, bbox_to_anchor=(1, 0.5))
ax_10.set_ylim(-9, 309)
ax_10.set_xticklabels([])

# Prediction errors
ax_11 = plt.Subplot(f, gs_01[2, 0])
f.add_subplot(ax_11)

ax_11.plot(x, np.array(df_data_RBM['delta_t'][plot_range[0]:plot_range[1]]), linestyle='-', linewidth=2,
           color="#090030", alpha=0.4)
ax_11.plot(x, np.array(df_data_sm['delta_t'][plot_range[0]:plot_range[1]]), linewidth=1, color="#090030", alpha=1)
ax_11.set_xticklabels([])
ax_11.set_ylabel('Prediction Error')
ax_11.yaxis.set_label_coords(ylabel_dist, 0.5)
ax_11.legend(["RBM", "SM"],  loc='center left', framealpha=0.8, bbox_to_anchor=(1, 0.5))

# Relative uncertainty, changepoint probability, and learning rate
ax_12 = plt.Subplot(f, gs_01[3, 0])
f.add_subplot(ax_12)
ax_12.plot(x, np.array(df_data_RBM['tau_t'][plot_range[0]:plot_range[1]]), linestyle='-', linewidth=2,
           color="#04879c", alpha=0.4)
ax_12.plot(x, np.array(df_data_RBM['omega_t'][plot_range[0]:plot_range[1]]), linestyle='-', linewidth=2,
           color="#0c3c78", alpha=0.4)
ax_12.plot(x, np.array(df_data_sm['tau_t'][plot_range[0]:plot_range[1]]), linewidth=1, color="#04879c", alpha=1)
ax_12.plot(x, np.array(df_data_sm['omega_t'][plot_range[0]:plot_range[1]]), linewidth=1, color="#0c3c78", alpha=1)
ax_12.legend(['RU RBM', 'CPP RBM', 'RU SM', 'CPP SM'], loc='center left', framealpha=0.8, bbox_to_anchor=(1, 0.5))
ax_12.set_xlabel('Trial')
ax_12.set_ylabel('Variable')
ax_12.yaxis.set_label_coords(ylabel_dist, 0.5)

# Delete unnecessary axes
sns.despine()

# --------------
# 6. Save figure
# --------------

# Save figure for manuscript
savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_S_figure_16.pdf"
plt.savefig(savename, transparent=False, dpi=400)

# Show plot
plt.ioff()
plt.show()
