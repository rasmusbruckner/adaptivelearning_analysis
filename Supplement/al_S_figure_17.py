""" Figure S17: Sampling model in standard and anchoring condition

    1. Load data
    2. Simulate data based on sampling model (SM) and show validation plot
    3. Save figure
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
from sampling.AlAgentVarsSampling import AgentVarsSampling
from sampling.AlAgentSampling import AlAgentSampling
from sampling.al_task_agent_int_sampling import task_agent_int_sampling
from al_utilities import get_df_subj
from al_plot_utils import latex_plt, plot_trial_validation


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

# ----------------------------------------------------------------------
# 2. Simulate data based on sampling model (SM) and show validation plot
# ----------------------------------------------------------------------

# Extract subject-specific data frame
subject = 0
df_subj = get_df_subj(df_exp2, subject)

# Agent variables object
agent_vars = AgentVarsSampling()

# Set agent parameters
agent_vars.criterion = 0.04
agent_vars.n_samples = 4
agent_vars.model_sat = True
agent_vars.burn_in = 0
agent_vars.sigma = 17.5
agent = AlAgentSampling(agent_vars)

# Run task-agent interaction
df_data = task_agent_int_sampling(df_subj, agent, agent_vars, show_pbar=True)

# Split data into push and no-push condition
df_data_no_push = df_data[df_data["cond"] == "main_noPush"]
df_data_push = df_data[df_data["cond"] == "main_push"]

# Compute perseveration
pers = df_data['sim_a_t'] == 0

# Plot simulation results
plot_trial_validation(df_subj, df_data, pers)

# --------------
# 3. Save figure
# --------------

savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_S_figure_17.pdf"
plt.savefig(savename, transparent=False, dpi=400)

# Show plot
plt.ioff()
plt.show()
