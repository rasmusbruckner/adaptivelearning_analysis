""" Figure S20: Sampling model parameter comparison

    1. Load simulated data
    2. Plot simulations
    3. Add subplot labels and save figure
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
from al_plot_utils import cm2inch, latex_plt, label_subplots, plot_sampling_results_row


# Update matplotlib to use Latex and to change some defaults
matplotlib = latex_plt(matplotlib)

# Get home directory
paths = os.getcwd()
path = paths.split(os.path.sep)
home_dir = path[1]

# ------------------------------------------------------------
#  1. Load simulated data (simulated in al_priorpred_sampling)
# ------------------------------------------------------------

# 1. Same parameters for each age group
# -------------------------------------

# Perseveration
all_pers_all_same = pd.read_pickle('al_data/all_same_all_pers.pkl')
pers_noPush_all_same = all_pers_all_same[all_pers_all_same['variable'] == "noPush"].reset_index(drop=True)
pers_push_all_same = all_pers_all_same[all_pers_all_same['variable'] == "push"].reset_index(drop=True)

# Estimation errors
all_est_errs_all_same = pd.read_pickle('al_data/all_same_all_est_errs.pkl')

# Anchoring bias
df_reg_all_same = pd.read_pickle('al_data/all_same_df_reg.pkl')

# 2. Different criterion
# ----------------------

# Perseveration: Bring into correct format for custom boxplot with both conditions
all_pers_criterion_diff = pd.read_pickle('al_data/crit_different_all_pers.pkl')
pers_noPush_criterion_diff = all_pers_criterion_diff[all_pers_criterion_diff['variable'] == "noPush"].reset_index(drop=True)
pers_push_criterion_diff = all_pers_criterion_diff[all_pers_criterion_diff['variable'] == "push"].reset_index(drop=True)

# Estimation errors: Bring into correct format for custom boxplot with both conditions
all_est_errs_criterion_diff = pd.read_pickle('al_data/crit_different_all_est_errs.pkl')

# Anchoring bias
df_reg_criterion_diff = pd.read_pickle('al_data/crit_different_df_reg.pkl')

# 3. Different chunk size
# -----------------------

# Perseveration: Bring into correct format for custom boxplot with both conditions
all_pers_chunk_size_diff = pd.read_pickle('al_data/n_samples_different_all_pers.pkl')
pers_noPush_chunk_size_diff = all_pers_chunk_size_diff[all_pers_chunk_size_diff['variable'] == "noPush"].reset_index(drop=True)
pers_push_chunk_size_diff = all_pers_chunk_size_diff[all_pers_chunk_size_diff['variable'] == "push"].reset_index(drop=True)

# Estimation errors: Bring into correct format for custom boxplot with both conditions
all_est_errs_chunk_size_diff = pd.read_pickle('al_data/n_samples_different_all_est_errs.pkl')

# Anchoring bias
df_reg_chunk_size_diff = pd.read_pickle('al_data/n_samples_different_df_reg.pkl')

# -------------------
# 2. Plot simulations
# -------------------

# Figure size
fig_height = 12
fig_width = 15

# Image format
saveas = "pdf"  # pdf or png

# Turn interactive plotting mode on for debugger
plt.ion()

# Condition colors
condition_colors = ["#BBE1FA", "#3282B8", "#1B262C", "#e3f3fd", "#adcde2", "#babdbf"]

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))
f.canvas.draw()

# Create plot grid
gs_0 = gridspec.GridSpec(nrows=3, ncols=1, left=0.15, top=0.925, bottom=0.1, right=0.95, hspace=0.8) # nrows = 17

# Y-label distance
ylabel_dist = -0.3

# 1. Same parameters for each age group
# -------------------------------------

gs_3 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=3, subplot_spec=gs_0[0, :], hspace=0.5, wspace=0.5)
title = 'Same parameters:\nMean chunk size = 11, mean criterion = 0.01'
plot_sampling_results_row(gs_3, f, pers_noPush_all_same, pers_push_all_same, all_est_errs_all_same, df_reg_all_same,
                          condition_colors, ylabel_dist, plot_legend=False, title=title)

# 2. Different criterion
# ----------------------

# Create subplot
gs_3 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=3, subplot_spec=gs_0[1, :], hspace=0.5, wspace=0.5)
title = 'Different criterion: Mean chunk size = 11\nmean criterion = 0.02 (CH/OA) and 0.008 (YA)'
plot_sampling_results_row(gs_3, f, pers_noPush_criterion_diff, pers_push_criterion_diff, all_est_errs_criterion_diff,
                          df_reg_criterion_diff, condition_colors, ylabel_dist, plot_legend=False, title=title)

# 3. Different chunk size
# -----------------------

# Create subplot
gs_3 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=3, subplot_spec=gs_0[2, :], hspace=0.5, wspace=0.5)
title = 'Different chunk size:\nMean chunk size = 4 (CH/OA) and 22 (YA), mean criterion = 0.01'
plot_sampling_results_row(gs_3, f, pers_noPush_chunk_size_diff, pers_push_chunk_size_diff, all_est_errs_chunk_size_diff,
                          df_reg_chunk_size_diff, condition_colors, ylabel_dist, plot_legend=False, title=title)

# -------------------------------------
# 3. Add subplot labels and save figure
# -------------------------------------

# Delete unnecessary axes
sns.despine()

# Add labels
texts = ['a', '', '', 'b', '', '', 'c']
label_subplots(f, texts, x_offset=+0.11, y_offset=0.02)

savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_S_figure_20.pdf"
plt.savefig(savename, transparent=False, dpi=400)

# Show plot
plt.ioff()
plt.show()
