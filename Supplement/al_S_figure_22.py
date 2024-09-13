""" Figure S22: Motor model

    1. Load data
    2. Prepare figure
    3. Plot optimal model
    4. Plot perseveration parameters
    5. Plot anchoring parameters
    6. Add subplot labels and save figure
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import seaborn as sns
import os
from al_plot_utils import (cm2inch, latex_plt, swarm_boxplot, custom_boxplot_condition, text_legend, plot_header,
                           label_subplots)
from al_utilities import trial_cost_func


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
df_pers_opt = pd.read_pickle('al_data/priorpred_motor_pers_opt.pkl')
pers_noPush_opt = df_pers_opt[df_pers_opt['variable'] == 'noPush'].reset_index(drop=True)
pers_push_opt = df_pers_opt[df_pers_opt['variable'] == 'push'].reset_index(drop=True)

# Perseveration parameters
df_pers_pers = pd.read_pickle('al_data/priorpred_motor_pers_pers.pkl')
pers_noPush_pers = df_pers_pers[df_pers_pers['variable'] == 'noPush'].reset_index(drop=True)
pers_push_pers = df_pers_pers[df_pers_pers['variable'] == 'push'].reset_index(drop=True)

# Anchoring parameters
df_pers_anchor = pd.read_pickle('al_data/priorpred_motor_pers_anchor.pkl')
pers_noPush_anchor = df_pers_anchor[df_pers_anchor['variable'] == 'noPush'].reset_index(drop=True)
pers_push_anchor = df_pers_anchor[df_pers_anchor['variable'] == 'push'].reset_index(drop=True)

# Estimation errors
# -----------------

# Optimal model
df_est_err_opt = pd.read_pickle('al_data/priorpred_motor_est_errs_opt.pkl')
est_err_noPush_opt = df_est_err_opt[df_est_err_opt['variable'] == 'noPush'].reset_index(drop=True)
est_err_push_opt = df_est_err_opt[df_est_err_opt['variable'] == 'push'].reset_index(drop=True)

# Perseveration parameters
df_est_err_pers = pd.read_pickle('al_data/priorpred_motor_est_errs_pers.pkl')
est_err_noPush_pers = df_est_err_pers[df_est_err_pers['variable'] == 'noPush'].reset_index(drop=True)
est_err_push_pers = df_est_err_pers[df_est_err_pers['variable'] == 'push'].reset_index(drop=True)

# Anchoring parameters
df_est_err_anchor = pd.read_pickle('al_data/priorpred_motor_est_errs_anchor.pkl')
est_err_noPush_anchor = df_est_err_anchor[df_est_err_anchor['variable'] == 'noPush'].reset_index(drop=True)
est_err_push_anchor = df_est_err_anchor[df_est_err_anchor['variable'] == 'push'].reset_index(drop=True)

# Anchoring
# ---------

# Optimal model
df_reg_opt = pd.read_pickle('al_data/priorpred_motor_df_reg_opt.pkl')

# Perseveration parameters
df_reg_pers = pd.read_pickle('al_data/priorpred_motor_df_reg_pers.pkl')

# Anchoring parameters
df_reg_anchor = pd.read_pickle('al_data/priorpred_motor_df_reg_anchor.pkl')

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
ylabel_dist_behav = -0.55

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))
f.canvas.draw()

# Create plot grid
gs_0 = gridspec.GridSpec(nrows=3, ncols=1, hspace=0.4)

# ---------------------
# 3. Plot optimal model
# ---------------------

# Create subplot and plot header
gs_00 = gridspec.GridSpecFromSubplotSpec(nrows=6, ncols=4, subplot_spec=gs_0[0, 0], wspace=1)
ax_00 = plt.Subplot(f, gs_00[0, :])
f.add_subplot(ax_00)
plot_header(f, ax_00, 'Baseline Parameters', patches)

# Cost illustration
# -----------------

# Grid for cost illustration
grid = np.linspace(0, 300, 301)
sim_z_t = 200
mu = 100

# Compute costs
dist_z_t = abs(grid - sim_z_t)
dist_mu_t = abs(grid - mu)
update_error = trial_cost_func(dist_mu_t, 0.5, 1.1)
motor_cost = trial_cost_func(dist_z_t, 0.0, 1)

# Cost illustration
ax_01 = plt.Subplot(f, gs_00[1:, 0])
f.add_subplot(ax_01)
ax_01.plot(grid, update_error, color=colors[0])
ax_01.plot(grid, motor_cost, color=colors[1])
ax_01.plot(grid[update_error == min(update_error)], update_error[update_error == min(update_error)], '.', color=colors[0])
ax_01.set_ylim([-30, 350])
ax_01.set_ylabel('Cost')
ax_01.set_xlim([-10, 310])
ax_01.set_xticks(np.arange(0, 301, step=100))

# Behavioral effects
# ------------------

# Condition colors
condition_colors = ["#BBE1FA", "#3282B8", "#1B262C", "#e3f3fd", "#adcde2", "#babdbf"]

# Plot perseveration probability
ax_02 = plt.Subplot(f, gs_00[1:, 1])
f.add_subplot(ax_02)
custom_boxplot_condition(ax_02, pers_noPush_opt, pers_push_opt, 'value', 'Perseveration\nProbability',
                         condition_colors, with_lines=False)
ax_02.yaxis.set_label_coords(ylabel_dist_behav, 0.5)

# Plot estimation errors
ax_03 = plt.Subplot(f, gs_00[1:, 2])
f.add_subplot(ax_03)
custom_boxplot_condition(ax_03, est_err_noPush_opt, est_err_push_opt, 'value', 'Estimation Error',
                         condition_colors, with_lines=False)
ax_03.yaxis.set_label_coords(ylabel_dist_behav, 0.5)

# Plot bucket bias
ax_04 = plt.Subplot(f, gs_00[1:, 3])
f.add_subplot(ax_04)
swarm_boxplot(ax_04, df_reg_opt, 'bucket_bias', ' ', 2)
ax_04.set_ylabel('Anchoring Bias')

# --------------------------------
# 4. Plot perseveration parameters
# --------------------------------

# Create subplot and plot header
gs_10 = gridspec.GridSpecFromSubplotSpec(nrows=6, ncols=4, subplot_spec=gs_0[1, 0], wspace=1)
ax_10 = plt.Subplot(f, gs_10[0, :])
f.add_subplot(ax_10)
plot_header(f, ax_10, 'Perseveration Parameters', patches)

# Cost illustration
# -----------------

# Create subplot
ax_11 = plt.Subplot(f, gs_10[1:, 0])
f.add_subplot(ax_11)

# Compute costs
update_error = trial_cost_func(dist_mu_t, 0.5, 1.1)
motor_cost = trial_cost_func(dist_z_t, 0.75, 1.2)
total_costs = update_error + motor_cost

# Cost illustration
ax_11.plot(grid, update_error, color=colors[0])
ax_11.plot(grid, motor_cost, color=colors[1])
ax_11.plot(grid, total_costs, color=colors[2])

# Add text legends
ax_11.legend(['Error cost', 'Motor cost', 'Total cost'], loc='lower left', bbox_to_anchor=(0, 2.95), framealpha=1,
             edgecolor="k")
text_legend(plt.gca(), "Darker colors (left): Standard condition | Lighter colors (right): Anchoring condition",
            coords=[1.2, 2.95], loc='lower left')

# Cost minima
ax_11.plot(grid[update_error == min(update_error)], update_error[update_error == min(update_error)], '.',
           color=colors[0])
ax_11.plot(grid[motor_cost == min(motor_cost)], motor_cost[motor_cost == min(motor_cost)], '.', color=colors[1])
ax_11.plot(grid[total_costs == min(total_costs)], total_costs[total_costs == min(total_costs)], '.', color=colors[2])
ax_11.set_ylim([-30, 350])
ax_11.set_ylabel('Cost')
ax_11.set_xlim([-10, 310])
ax_11.set_xticks(np.arange(0, 301, step=100))

# Behavioral effects
# ------------------

# Plot perseveration probability
ax_12 = plt.Subplot(f, gs_10[1:, 1])
f.add_subplot(ax_12)
custom_boxplot_condition(ax_12, pers_noPush_pers, pers_push_pers, 'value', 'Perseveration\nProbability',
                         condition_colors, with_lines=False)
ax_12.yaxis.set_label_coords(ylabel_dist_behav, 0.5)

# Plot estimation errors
ax_13 = plt.Subplot(f, gs_10[1:, 2])
f.add_subplot(ax_13)
custom_boxplot_condition(ax_13, est_err_noPush_pers, est_err_push_pers, 'value', 'Estimation Error',
                         condition_colors, with_lines=False)
ax_13.yaxis.set_label_coords(ylabel_dist_behav, 0.5)

# Plot bucket bias
ax_14 = plt.Subplot(f, gs_10[1:, 3])
f.add_subplot(ax_14)
swarm_boxplot(ax_14, df_reg_pers, 'bucket_bias', 'Anchoring Bias', 2)
ax_14.yaxis.set_label_coords(ylabel_dist_behav, 0.5)

# ----------------------------
# 5. Plot anchoring parameters
# ----------------------------

# Create subplot and plot header
gs_20 = gridspec.GridSpecFromSubplotSpec(nrows=6, ncols=4, subplot_spec=gs_0[2, 0], wspace=1)
ax_20 = plt.Subplot(f, gs_20[0, :])
f.add_subplot(ax_20)
ax_0 = plot_header(f, ax_20, 'Anchoring Parameters', patches)

# Cost illustration
# -----------------

# Create subplot
ax_21 = plt.Subplot(f, gs_20[1:, 0])
f.add_subplot(ax_21)

# Compute costs
update_error = trial_cost_func(dist_mu_t, 0.5, 1.1)
motor_cost = trial_cost_func(dist_z_t, 0.1, 1.4)
total_costs = update_error + motor_cost

# Cost illustration
ax_21.plot(grid, update_error, color=colors[0])
ax_21.plot(grid, motor_cost, color=colors[1])
ax_21.plot(grid, total_costs, color=colors[2])
ax_21.plot(grid[update_error == min(update_error)], update_error[update_error == min(update_error)], '.',
           color=colors[0])
ax_21.plot(grid[motor_cost == min(motor_cost)], motor_cost[motor_cost == min(motor_cost)], '.', color=colors[1])
ax_21.plot(grid[total_costs == min(total_costs)], total_costs[total_costs == min(total_costs)], '.', color=colors[2])
ax_21.set_ylim([-30, 350])
ax_21.set_ylabel('Cost')
ax_21.set_xlim([-10, 310])
ax_21.set_xticks(np.arange(0, 301, step=100))

# Behavioral effects
# ------------------

# Plot perseveration probability
ax_22 = plt.Subplot(f, gs_20[1:, 1])
f.add_subplot(ax_22)
custom_boxplot_condition(ax_22, pers_noPush_anchor, pers_push_anchor, 'value', 'Perseveration\nProbability',
                         condition_colors, with_lines=False)
ax_22.yaxis.set_label_coords(ylabel_dist_behav, 0.5)

# Plot estimation errors
ax_23 = plt.Subplot(f, gs_20[1:, 2])
f.add_subplot(ax_23)
custom_boxplot_condition(ax_23, est_err_noPush_anchor, est_err_push_anchor, 'value', 'Estimation Error',
                         condition_colors, with_lines=False)
ax_23.yaxis.set_label_coords(ylabel_dist_behav, 0.5)

# Plot bucket bias
ax_24 = plt.Subplot(f, gs_20[1:, 3])
f.add_subplot(ax_24)
swarm_boxplot(ax_24, df_reg_anchor, 'bucket_bias', 'Anchoring Bias', 2)
ax_24.yaxis.set_label_coords(ylabel_dist_behav, 0.5)

# -------------------------------------
# 6. Add subplot labels and save figure
# -------------------------------------

# Delete unnecessary axes
sns.despine()

# Add labels
texts = ['', 'a', 'b', 'c', 'd', '', 'e', 'f', 'g', 'h', '', 'i', 'j', 'k', 'l']
label_subplots(f, texts, x_offset=+0.1, y_offset=-0.02)

# Save plot
savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_S_figure_22.pdf"
plt.savefig(savename, transparent=True, dpi=400)

# Show plot
plt.show()
