""" Figure S1

    1. Load data
    2. Prepare figure
    3. Plot mixture-model illustration
    4. Plot logistic-function illustration
    5. Add subplot labels and save figure
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import expit
import seaborn as sns
import os
from al_plot_utils import cm2inch, label_subplots, latex_plt
from AlAgentVars import AgentVars
from AlAgent import AlAgent


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

# Parameter estimates
model_results = pd.read_pickle('al_data/estimates_first_exp_25_sp.pkl')

# -----------------
# 2. Prepare figure
# -----------------

# Size of figure
fig_width = 10
fig_height = 4

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))

# Create plot grid
gs_0 = gridspec.GridSpec(1, 1, left=0.125, right=0.95, top=0.85, bottom=0.225)

# Plot colors
colors = ["#92e0a9", "#69b0c1", "#6d6192", "#352d4d"]
sns.set_palette(sns.color_palette(colors))

# ----------------------------------
# 3. Plot mixture-model illustration
# ----------------------------------

# Create subplot grid
gs_00 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_0[0], hspace=0.6, wspace=0.3)

# Plot perseveration frequency
exp = 1
ax_00 = plt.Subplot(f, gs_00[0, 0])
f.add_subplot(ax_00)

# Extract parameter estimates for the group of younger adults
ya_median_h = np.median(model_results[model_results['age_group'] == 3]['h'])
ya_median_s = np.median(model_results[model_results['age_group'] == 3]['s'])
ya_median_u = np.median(model_results[model_results['age_group'] == 3]['u'])

# Create agent-object instance for learning-rate illustration
agent_vars = AgentVars()
agent_vars.s = ya_median_s
agent_vars.h = ya_median_h

# Range of prediction errors
pe = np.linspace(1, 80, 80)

# Initialize learning-rate array
alpha = np.full(len(pe), np.nan)

# Cycle over range of prediction errors
for i in range(0, len(pe)):

    # Compute learning rate
    agent = AlAgent(agent_vars)
    agent.tau_t = 0.2/np.exp(ya_median_u)
    agent.learn(pe[i], np.nan, False, np.nan, False)
    alpha[i] = agent.alpha_t

# Plot reduced Bayesian model component
ax_00.set_title('Mixture components')
ax_00.plot(pe, alpha, color="#584b42")
ax_00.set_ylabel('Learning rate')
ax_00.set_xlabel('Prediction error')
ax_00.text(0, 0.8, "Estimated reduced\nBayesian model")
ax_00.set_ylim([-0.03, 1.02])

# Plot perseveration model component
ax_00.plot(pe, np.zeros(80), color="#a0855b")
ax_00.text(0, 0.05, "Perseveration")

# --------------------------------------
# 4. Plot logistic-function illustration
# --------------------------------------

ax_01 = plt.Subplot(f, gs_00[0, 1])
f.add_subplot(ax_01)

# Set simulation intercept and slope
b_0 = 10  # intercept
b_1 = -0.15  # slope

# Initialize vectors for perseveration probability and updating probability
pers_prob = np.full(80, np.nan)
up_prob = np.full(80, np.nan)

# Cycle over predicted updates
for i in range(0, 80):

    pers_prob[i] = expit(np.array(b_1*(i-b_0)))
    up_prob[i] = 1-pers_prob[i]

# Plot weights of the 2 components
ax_01.plot(np.arange(80), pers_prob, color="#a0855b")
ax_01.plot(np.arange(80), up_prob, color="#584b42")
ax_01.set_ylim([-0.02, 1.2])
ax_01.set_title('Mixture weights')
ax_01.set_xlabel('Predicted update')
ax_01.set_ylabel('Probability')
ax_01.set_ylim([-0.03, 1.02])
ax_01.text(25, 0.7, "Estimated reduced\nBayesian model")
ax_01.text(40, 0.05, "Perseveration")

# -------------------------------------
# 5. Add subplot labels and save figure
# -------------------------------------

# Delete unnecessary axes
sns.despine()

# Add labels
texts = ['a', 'b']  # label letters
label_subplots(f, texts, x_offset=0.1, y_offset=0.02)

# Save figure for manuscript
savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_S_figure_1.pdf"
plt.savefig(savename, transparent=True, dpi=400)

# Show plot
plt.show()
