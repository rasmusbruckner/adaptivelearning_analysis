""" Figure 4

    1. Load data
    2. Prepare figure
    3. Plot simulated perseveration frequency and performance
    4. Plot zero-perseveration illustration
    5. Add subplot labels and save figure
"""

import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from al_utilities import get_mean_voi
from al_plot_utils import cm2inch, label_subplots, plot_arrow, latex_plt, text_legend, custom_boxplot
from scipy.special import expit

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

# Load data from first experiment
df_exp1 = pd.read_pickle('al_data/data_prepr_1.pkl')

# Simulation data
# ---------------

# Posterior predictive checks, obtained from "al_postpred"
all_pers = pd.read_pickle('al_data/postpred_exp1_pers.pkl')
all_est_err = pd.read_pickle('al_data/postpred_exp1_est_err.pkl')

# simulation without perseveration, obtained from "al_postpred"
hyp_est_err_exp1 = pd.read_pickle('al_data/hyp_est_errs_exp1_no_pers.pkl')

# -----------------
# 2. Prepare figure
# -----------------

# Size of figure
fig_width = 15
fig_height = 10

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))

# Create plot grid
gs_0 = gridspec.GridSpec(1, 1, wspace=5, top=0.95, bottom=0.1)

# Plot colors
colors = ["#92e0a9", "#69b0c1", "#6d6192", "#352d4d"]
sns.set_palette(sns.color_palette(colors))

# Compute estimation errors
voi = 1
e_t = get_mean_voi(df_exp1, voi)

# Compute perseveration frequency
voi = 2
pers = get_mean_voi(df_exp1, voi)

# Range of predicted updates
pred_up = np.linspace(1, 80, 80)

# Set intercept and slope for logistic-function illustration
b_0 = 10
b_1 = -0.15

# Initialize vectors for perseveration probability
pers_prob = np.full(80, np.nan)

# Cycle over range of predicted updates and get logistic function
for i in range(0, len(pred_up)):
    pers_prob[i] = expit(np.array(b_1*(i-b_0)))

# ---------------------------------------------------------
# 3. Plot simulated perseveration frequency and performance
# ---------------------------------------------------------

# Create subplot grid
gs_00 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_0[0], hspace=0.7, wspace=0.3)

# Plot perseveration
ax_00 = plt.Subplot(f, gs_00[0, 0])
f.add_subplot(ax_00)
ax_00 = custom_boxplot(ax_00, pers, 'pers', alpha=0.2)
sns.swarmplot(x='age_group', y='main', data=all_pers, alpha=1, size=1, color='k')
plt.xticks(np.arange(4), ['CH', 'AD', 'YA', 'OA'], rotation=0)
ax_00.set_xlabel('Age group')
ax_00.set_ylabel('Perseveration')
text_legend(plt.gca(), "Boxplots: participants | Points: simulations")

# Plot estimation errors
ax_01 = plt.Subplot(f, gs_00[0, 1])
f.add_subplot(ax_01)
ax_01 = custom_boxplot(ax_01, e_t, 'e_t', alpha=0.2)
sns.swarmplot(x='age_group', y='main', data=all_est_err, alpha=1, size=1, color='k')
plt.xticks(np.arange(4), ['CH', 'AD', 'YA', 'OA'], rotation=0)
ax_01.set_xlabel('Age group')
ax_01.set_ylabel('Estimation error')
ax_01.set_ylim(8, 25)

# ---------------------------------------
# 4. Plot zero-perseveration illustration
# ---------------------------------------

ax_10 = plt.Subplot(f, gs_00[1, 0])
f.add_subplot(ax_10)
ax_10.plot(pred_up, pers_prob, color="#a0855b", alpha=1)
ax_10.set_ylim([-0.02, 1.2])
ax_10.set_title('No perseveration')
ax_10.set_xlabel('Reduced Bayesian model predicted update')
ax_10.plot(pred_up, np.zeros(80), color="#a0855b")
ax_10.set_ylabel('Probability')
ax_10.set_ylim([-0.03, 1.02])
plot_arrow(ax_10, 5, 0.66, 5, 0.01)
text = 'Does reducing perseveration\nrescue performance?'
x, y = 20, 0.7
ax_10.text(x, y, text)
plot_arrow(ax_10, 60, 0.725, 90, 0.725)

# Model simulated estimation errors
ax_11 = plt.Subplot(f, gs_00[1, 1])
f.add_subplot(ax_11)
sns.swarmplot(x='age_group', y='main', data=hyp_est_err_exp1, alpha=1, size=1, color='k')
plt.xticks(np.arange(4), ['CH', 'AD', 'YA', 'OA'], rotation=0)
ax_11.set_ylim(8, 25)
ax_11.set_title('Hypothesized performance\nassuming no perseveration')
ax_11.set_ylabel('Estimation error')
ax_11.set_xlabel('Age group')

# --------------------------------------
# 5. Add subplot labels and save figure
# --------------------------------------

# Delete unnecessary axes
sns.despine()

# Add labels
texts = ['a', 'b', 'c', 'd']  # label letters
label_subplots(f, texts, x_offset=0.08, y_offset=0.0)

# Save figure for manuscript
savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_figure_4.pdf"
plt.savefig(savename, transparent=True, dpi=400)

# Show plot
plt.show()
