""" Figure S18: Illustration of the two biases in the sampling model

    1. Load data
    2. Prepare figure
    3. Simulate data based on sampling model (SM)
    4. Run logistic regression
    5. Plot logistic regression
    6. Partial regression plot anchoring bias
    7. Save figure
"""

import numpy as np
import statsmodels.formula.api as smf
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import statsmodels.api as sm
import os
from al_plot_utils import cm2inch, latex_plt, label_subplots
from sampling.AlAgentVarsSampling import AgentVarsSampling
from al_utilities import get_df_subj
from sampling.AlAgentSampling import AlAgentSampling
from sampling.al_task_agent_int_sampling import task_agent_int_sampling


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

# -----------------
# 2. Prepare figure
# -----------------

# Size of figure
fig_height = 6
fig_width = 10

# Turn interactive plotting mode on for debugger
plt.ion()

# Axis limits
axlim = 150

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))

# Create plot grid
gs_0 = gridspec.GridSpec(1, 1, bottom=0.2, top=0.7)

# ---------------------------------------------
# 3. Simulate data based on sampling model (SM)
# ---------------------------------------------

# Extract subject-specific data frame
subject = 0
df_subj = get_df_subj(df_exp2, subject)

# Run simulation
n_sim = 1
sim_pers = True

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
df_data = task_agent_int_sampling(df_subj, agent, agent_vars, show_pbar=True, seed=0)

# --------------------------
# 4. Run logistic regression
# --------------------------

# Create data frame for regression
df_data_no_push = df_data[df_data["cond"] == "main_noPush"]
df = pd.DataFrame()
df['pers'] = np.array(df_data_no_push['pers']).astype(int)
df['PE'] = abs(np.array(df_data_no_push['delta_t']))

# Run regression
logit_mod = smf.logit("pers ~ PE", data=df)
logit_res = logit_mod.fit()
print(logit_res.summary())

# Generate logistic regression line showing perseveration probability
df_pe = pd.DataFrame({'PE': np.linspace(df["PE"].min(), df["PE"].max(), 100)})
pred_pers = logit_res.predict(df_pe)

# ---------------------------
# 5. Plot logistic regression
# ---------------------------

# Create subplot grid
gs_01 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_0[:, :], wspace=0.5)

# Plot perseveration probability
ax_1 = plt.Subplot(f, gs_01[0, 0])
f.add_subplot(ax_1)
line = ax_1.plot(np.array(df_pe['PE']), np.array(pred_pers), '-', color='k', linewidth=2)
points = ax_1.plot(np.array(df["PE"]), np.array(df["pers"]), '.', color='k', markersize=8, alpha=0.20)
ax_1.set_xlabel('Prediction Error')
ax_1.set_ylabel('Perseveration Probability')
ax_1.set_title('Logistic Regression')
ax_1.legend(["Regression Slope", "Data"],  framealpha=0.8, loc='lower left', bbox_to_anchor=(0, 1.175))

# -----------------------------------------
# 6. Partial regression plot anchoring bias
# -----------------------------------------

# Run regression: Update = b_0 + b_1 * PE + b_2 * y_t
df_data_push = df_data[df_data["cond"] == "main_push"]
mod = smf.ols(formula='sim_a_t ~ delta_t + sim_y_t', data=df_data_push)
res = mod.fit()
print(res.summary())

ax_2 = plt.Subplot(f, gs_01[0, 1])
f.add_subplot(ax_2)

# Partial regression plot: update by push
sm.graphics.plot_partregress(endog='sim_a_t', exog_i='sim_y_t', exog_others='delta_t', data=df_data_push,
                             obs_labels=False, ax=ax_2, marker='.', color='k', markersize=8, alpha=0.20)
ax_2.plot([-axlim, axlim], [-axlim, axlim], color='gray', linestyle='--', zorder=0)
ax_2.axhline(y=0.5, color='gray', linestyle='--', zorder=0)
ax_2.set_xlim(-axlim, axlim)
ax_2.set_ylim(-axlim, axlim)
ax_2.set_xlabel('Anchor')
ax_2.set_ylabel('Anchoring Bias')
sns.despine()

# Delete unnecessary axes
sns.despine()

# -------------------------------------
# 7. Add subplot labels and save figure
# -------------------------------------

# Label letters
texts = ['a', 'b']

# Add labels
label_subplots(f, texts, x_offset=0.1, y_offset=0.0)

# Save figure for manuscript
savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_S_figure_18.pdf"
plt.savefig(savename, transparent=False, dpi=400)

# Show plot
plt.ioff()
plt.show()
