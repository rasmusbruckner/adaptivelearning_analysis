""" Figure 3: Descriptive results first experiment

    1. Load data
    2. Compute estimation errors and perseveration probability
    3. Compute average learning rates
    4. Run statistical tests
    5. Prepare figure
    6. Plot performance, average learning rates, and perseveration
    7. Plot logistic function of each age group
    8. Add subplot labels and save figure
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import expit
import seaborn as sns
import os
from al_utilities import get_mean_voi, get_stats, safe_save_dataframe, compute_average_LR
from al_plot_utils import cm2inch, label_subplots, swarm_boxplot, latex_plt


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
df_exp1 = pd.read_pickle('al_data/data_prepr_1.pkl')
n_subj = len(np.unique(df_exp1['subj_num']))

# Parameter estimates
model_results = pd.read_pickle('al_data/estimates_first_exp_10_sp.pkl')

# ----------------------------------------------------------
# 2. Compute estimation errors and perseveration probability
# ----------------------------------------------------------

# Compute estimation errors
voi = 1
e_t = get_mean_voi(df_exp1, voi)

# Compute perseveration frequency
voi = 2
pers = get_mean_voi(df_exp1, voi)

# ---------------------------------
# 3. Compute average learning rates
# ---------------------------------

df_alpha = compute_average_LR(n_subj, df_exp1)

# ------------------------
# 4. Run statistical tests
# ------------------------

# Estimation errors
# -----------------

# Print out estimation error statistics for paper
print('\n\nEstimation error Experiment 1\n')
exp1_est_err, exp1_est_err_stat, exp1_est_err_effect_size = get_stats(e_t, 1, 'e_t')
exp1_est_err.name, exp1_est_err_stat.name, exp1_est_err_effect_size.name = "exp1_est_err_desc", "exp1_est_err_stat", "exp1_est_err_effect_size"
safe_save_dataframe(exp1_est_err, 'age_group')
safe_save_dataframe(exp1_est_err_stat, 'test')
safe_save_dataframe(exp1_est_err_effect_size, 'type')

# Average learning rates
# ----------------------

# Print out average learning rate statistics for paper
print('\n\nAlpha Experiment 1\n')
exp1_lr_desc, exp1_lr_stat, exp1_lr_effect_size = get_stats(df_alpha, 1, 'alpha')
exp1_lr_desc.name, exp1_lr_stat.name, exp1_lr_effect_size.name = "exp1_lr_desc", "exp1_lr_stat", "exp1_lr_effect_size"
safe_save_dataframe(exp1_lr_desc, 'age_group')
safe_save_dataframe(exp1_lr_stat, 'test')
safe_save_dataframe(exp1_lr_effect_size, 'type')

# Perseveration
# -------------

# Print out perseveration statistics for paper
print('\n\nPerseveration\n')
exp1_pers_desc, exp1_pers_stat, exp1_pers_effect_size = get_stats(pers, 1, 'pers')
exp1_pers_desc.name, exp1_pers_stat.name, exp1_pers_effect_size.name = "exp1_pers_desc", "exp1_pers_stat", "exp1_pers_effect_size"
safe_save_dataframe(exp1_pers_desc, 'age_group')
safe_save_dataframe(exp1_pers_stat, 'test')
safe_save_dataframe(exp1_pers_effect_size, 'type')

# -----------------
# 5. Prepare figure
# -----------------

# Size of figure
fig_height = 5
fig_width = 15

# Image format
saveas = "png"

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))

# Create plot grid
gs_0 = gridspec.GridSpec(1, 1, wspace=1, hspace=0.7, top=0.8, bottom=0.2, left=0.1, right=0.95)

# Plot colors
colors = ["#BBE1FA", "#3282B8", "#0F4C75", "#1B262C"]
sns.set_palette(sns.color_palette(colors, n_colors=4))

# --------------------------------------------------------------
# 6. Plot performance, average learning rates, and perseveration
# --------------------------------------------------------------

# Create subplot grid
gs_00 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_0[0], hspace=0.6, wspace=0.5)

# Plot estimation-error swarm-boxplot
exp = 1
ax_2 = plt.Subplot(f, gs_00[0, 0])
f.add_subplot(ax_2)
ax_2 = swarm_boxplot(ax_2, e_t, 'e_t', 'Estimation Error', exp)
ax_2.set_title('Participants')

# Plot learning-rate swarm-boxplot
exp = 1
ax_3 = plt.Subplot(f, gs_00[0, 1])
f.add_subplot(ax_3)
ax_3 = swarm_boxplot(ax_3, df_alpha, 'alpha', 'Learning Rate', exp)
ax_3.set_title('Participants')

# Plot perseveration frequency
exp = 1
ax_00 = plt.Subplot(f, gs_00[0, 2])
f.add_subplot(ax_00)
swarm_boxplot(ax_00, pers, 'pers', 'Perseveration Probability', exp)
ax_00.set_title('Participants')

# -------------------------------------------
# 7. Plot logistic function of each age group
# -------------------------------------------

# Compute empirical intercept and slope parameters
print('\n\nIntercept\n')
median_b_0, _, _ = get_stats(model_results, 1, 'b_0')
print('\n\nSlope\n')
median_b_1, _, _ = get_stats(model_results, 1, 'b_1')

# Initialize perseveration-frequency arrays
pers_prob_ch = np.full(50, np.nan)
pers_prob_ad = np.full(50, np.nan)
pers_prob_ya = np.full(50, np.nan)
pers_prob_oa = np.full(50, np.nan)

# Range of predicted update
pe = np.linspace(1, 50)

# Cycle over range of predicted updates
for i in range(0, len(pers_prob_ch)):

    pers_prob_ch[i] = expit(median_b_1['median']['ch']*(i-median_b_0['median']['ch']))
    pers_prob_ad[i] = expit(median_b_1['median']['ad']*(i-median_b_0['median']['ad']))
    pers_prob_ya[i] = expit(median_b_1['median']['ya']*(i-median_b_0['median']['ya']))
    pers_prob_oa[i] = expit(median_b_1['median']['oa']*(i-median_b_0['median']['oa']))

ax_12 = plt.Subplot(f, gs_00[0, 3])
f.add_subplot(ax_12)
ax_12.plot(pe, pers_prob_ch)
ax_12.plot(pe, pers_prob_ad)
ax_12.plot(pe, pers_prob_ya)
ax_12.plot(pe, pers_prob_oa)
ax_12.set_ylabel('Perseveration Probability')
ax_12.set_xlabel('Predicted Update')
ax_12.set_title('Model-Based Estimate')
plt.legend(['CH', 'AD', 'YA', 'OA'])

# -------------------------------------
# 8. Add subplot labels and save figure
# -------------------------------------

# Delete unnecessary axes
sns.despine()

# Add labels
texts = ['a', 'b', 'c', 'd']
label_subplots(f, texts, x_offset=0.08, y_offset=0.025)

# Save figure
# -----------

if saveas == "pdf":
    savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_figure_3.pdf"
else:
    savename = "/" + home_dir + "/rasmus/Dropbox/heli_lifespan/png/al_figure_3.png"

plt.savefig(savename, transparent=False, dpi=400)

# Show plot
plt.show()
