""" Figure S21: Resource-only model

    1. Load data
    2. Prepare figure
    3. Plot optimal model
    4. Plot perseveration parameters
    5. Plot anchoring parameters
    6. Add subplot labels and save figure
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import seaborn as sns
import os
from al_plot_utils import cm2inch, latex_plt, swarm_boxplot, custom_boxplot_condition, text_legend, plot_header, label_subplots


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

# Perseveration
# -------------

# Optimal model
df_pers_opt = pd.read_pickle('al_data/priorpred_resource_only_pers_baseline.pkl')
pers_noPush_opt = df_pers_opt[df_pers_opt['variable'] == 'noPush'].reset_index(drop=True)
pers_push_opt = df_pers_opt[df_pers_opt['variable'] == 'push'].reset_index(drop=True)

# Perseveration parameters
df_pers_pers = pd.read_pickle('al_data/priorpred_resource_only_pers_pers.pkl')
pers_noPush_pers = df_pers_pers[df_pers_pers['variable'] == 'noPush'].reset_index(drop=True)
pers_push_pers = df_pers_pers[df_pers_pers['variable'] == 'push'].reset_index(drop=True)

# Anchoring parameters
df_pers_anchor = pd.read_pickle('al_data/priorpred_resource_only_pers_anchor.pkl')
pers_noPush_anchor = df_pers_anchor[df_pers_anchor['variable'] == 'noPush'].reset_index(drop=True)
pers_push_anchor = df_pers_anchor[df_pers_anchor['variable'] == 'push'].reset_index(drop=True)

# Estimation errors
# -----------------

# Optimal model
df_est_err_opt = pd.read_pickle('al_data/priorpred_resource_only_est_errs_baseline.pkl')
est_err_noPush_opt = df_est_err_opt[df_est_err_opt['variable'] == 'noPush'].reset_index(drop=True)
est_err_push_opt = df_est_err_opt[df_est_err_opt['variable'] == 'push'].reset_index(drop=True)

# Perseveration parameters
df_est_err_pers = pd.read_pickle('al_data/priorpred_resource_only_est_errs_pers.pkl')
est_err_noPush_pers = df_est_err_pers[df_est_err_pers['variable'] == 'noPush'].reset_index(drop=True)
est_err_push_pers = df_est_err_pers[df_est_err_pers['variable'] == 'push'].reset_index(drop=True)

# Anchoring parameters
df_est_err_anchor = pd.read_pickle('al_data/priorpred_resource_only_est_errs_anchor.pkl')
est_err_noPush_anchor = df_est_err_anchor[df_est_err_anchor['variable'] == 'noPush'].reset_index(drop=True)
est_err_push_anchor = df_est_err_anchor[df_est_err_anchor['variable'] == 'push'].reset_index(drop=True)

# Anchoring
# ---------

# Optimal model
df_reg_opt = pd.read_pickle('al_data/priorpred_resource_only_df_reg_baseline.pkl')

# Perseveration parameters
df_reg_pers = pd.read_pickle('al_data/priorpred_resource_only_df_reg_pers.pkl')

# Anchoring parameters
df_reg_anchor = pd.read_pickle('al_data/priorpred_resource_only_df_reg_anchor.pkl')

# -----------------
# 2. Prepare figure
# -----------------

# Figure size
fig_height = 13
fig_width = 15

# Plot colors
colors = ["#2F4F4F", "#696969", "#000000"]
sns.set_palette(sns.color_palette(colors))

# Y-label distance
ylabel_dist_behav = -0.3

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))
f.canvas.draw()

# Create plot grid
gs_0 = gridspec.GridSpec(nrows=3, ncols=1, hspace=0.4, top=0.95)

# ---------------------
# 3. Plot optimal model
# ---------------------

# Create subplot and plot header
gs_00 = gridspec.GridSpecFromSubplotSpec(nrows=6, ncols=3, subplot_spec=gs_0[0, 0], wspace=0.5)
ax_00 = plt.Subplot(f, gs_00[0, :])
f.add_subplot(ax_00)
plot_header(f, ax_00, 'Baseline Parameters', patches)

# Add text legends
text_legend(plt.gca(), "Darker colors (left): Standard condition | Lighter colors (right): Anchoring condition",
            coords=[1.2, 2.95], loc='lower left')

# Behavioral effects
# ------------------

condition_colors = ["#BBE1FA", "#3282B8", "#1B262C", "#e3f3fd", "#adcde2", "#babdbf"]

# Plot perseveration probability
ax_01 = plt.Subplot(f, gs_00[1:, 0])
f.add_subplot(ax_01)
custom_boxplot_condition(ax_01, pers_noPush_opt, pers_push_opt, 'value', 'Perseveration\nProbability',
                         condition_colors, with_lines=False)
ax_01.yaxis.set_label_coords(ylabel_dist_behav, 0.5)

# Plot estimation errors
ax_02 = plt.Subplot(f, gs_00[1:, 1])
f.add_subplot(ax_02)
custom_boxplot_condition(ax_02, est_err_noPush_opt, est_err_push_opt, 'value', 'Estimation Error',
                         condition_colors, with_lines=False)
ax_02.yaxis.set_label_coords(ylabel_dist_behav, 0.5)

# Plot bucket bias
ax_03 = plt.Subplot(f, gs_00[1:, 2])
f.add_subplot(ax_03)
swarm_boxplot(ax_03, df_reg_opt, 'bucket_bias', ' ', 2)
ax_03.set_ylabel('Anchoring Bias')

# --------------------------------
# 4. Plot perseveration parameters
# --------------------------------

# Create subplot and plot header
gs_10 = gridspec.GridSpecFromSubplotSpec(nrows=6, ncols=3, subplot_spec=gs_0[1, 0], wspace=0.5)
ax_10 = plt.Subplot(f, gs_10[0, :])
f.add_subplot(ax_10)
plot_header(f, ax_10, 'Perseveration Parameters', patches)

# Behavioral effects
# ------------------

# Plot perseveration probability
ax_11 = plt.Subplot(f, gs_10[1:, 0])
f.add_subplot(ax_11)
custom_boxplot_condition(ax_11, pers_noPush_pers, pers_push_pers, 'value', 'Perseveration\nProbability',
                         condition_colors, with_lines=False)
ax_11.yaxis.set_label_coords(ylabel_dist_behav, 0.5)

# Plot estimation errors
ax_12 = plt.Subplot(f, gs_10[1:, 1])
f.add_subplot(ax_12)
custom_boxplot_condition(ax_12, est_err_noPush_pers, est_err_push_pers, 'value', 'Estimation Error',
                         condition_colors, with_lines=False)
ax_12.yaxis.set_label_coords(ylabel_dist_behav, 0.5)

# Plot bucket bias
ax_13 = plt.Subplot(f, gs_10[1:, 2])
f.add_subplot(ax_13)
swarm_boxplot(ax_13, df_reg_pers, 'bucket_bias', 'Anchoring Bias', 2)
ax_13.yaxis.set_label_coords(ylabel_dist_behav, 0.5)

# ----------------------------
# 5. Plot anchoring parameters
# ----------------------------

# Create subplot and plot header
gs_20 = gridspec.GridSpecFromSubplotSpec(nrows=6, ncols=3, subplot_spec=gs_0[2, 0], wspace=0.5)
ax_20 = plt.Subplot(f, gs_20[0, :])
f.add_subplot(ax_20)
ax_0 = plot_header(f, ax_20, 'Anchoring Parameters', patches)

# Behavioral effects
# ------------------

# Plot perseveration probability
ax_21 = plt.Subplot(f, gs_20[1:, 0])
f.add_subplot(ax_21)
custom_boxplot_condition(ax_21, pers_noPush_anchor, pers_push_anchor, 'value', 'Perseveration\nProbability',
                         condition_colors, with_lines=False)
ax_21.yaxis.set_label_coords(ylabel_dist_behav, 0.5)

# Plot estimation errors
ax_22 = plt.Subplot(f, gs_20[1:, 1])
f.add_subplot(ax_22)
custom_boxplot_condition(ax_22, est_err_noPush_anchor, est_err_push_anchor, 'value', 'Estimation Error',
                         condition_colors, with_lines=False)
ax_22.yaxis.set_label_coords(ylabel_dist_behav, 0.5)

# Plot bucket bias
ax_23 = plt.Subplot(f, gs_20[1:, 2])
f.add_subplot(ax_23)
swarm_boxplot(ax_23, df_reg_anchor, 'bucket_bias', 'Anchoring Bias', 2)
ax_23.yaxis.set_label_coords(ylabel_dist_behav, 0.5)

# -------------------------------------
# 6. Add subplot labels and save figure
# -------------------------------------

# Delete unnecessary axes
sns.despine()

# Add labels
texts = ['', 'a', 'b', 'c', '', 'd', 'e', 'f', '', 'g', 'h', 'i']
label_subplots(f, texts, x_offset=+0.1, y_offset=-0.02)

# Save plot
savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_S_figure_21.pdf"
plt.savefig(savename, transparent=True, dpi=400)

# Show plot
plt.show()
