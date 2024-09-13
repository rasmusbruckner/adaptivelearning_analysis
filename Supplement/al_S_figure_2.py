""" Figure S2: This script runs the catch-trial analyses for experiment 1 and 2

    1. Load data
    2. Compute catch-trial descriptive results
    3. Plot the results
    4. Add subplot labels and save figure
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
from al_plot_utils import cm2inch, label_subplots, swarm_boxplot, latex_plt, get_catch_trial_results
from al_utilities import safe_save_dataframe


# Update matplotlib to use Latex and to change some defaults
os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2016/bin/x86_64-darwin'
matplotlib = latex_plt(matplotlib)

# Get home directory
paths = os.getcwd()
path = paths.split(os.path.sep)
home_dir = path[1]

# Turn on interactive mode for debugger
plt.ion()

# ------------
# 1. Load data
# ------------

df_exp1 = pd.read_pickle('al_data/data_prepr_1.pkl')
df_exp2 = pd.read_pickle('al_data/data_prepr_2.pkl')

# ------------------------------------------
# 2. Compute catch-trial descriptive results
# ------------------------------------------

exp = 1
(exp1_catch_e_t_diff, exp1_catch_e_t_diff_group_desc, exp1_catch_e_t_diff_group_stat, exp1_catch_e_t_diff_zero_stat,
 exp1_catch_e_t_effect_size, exp1_catch_e_t_zero_effect_size) = get_catch_trial_results(df_exp1, exp)

(exp1_catch_e_t_diff_group_desc.name, exp1_catch_e_t_diff_group_stat.name, exp1_catch_e_t_diff_zero_stat.name,
 exp1_catch_e_t_effect_size.name, exp1_catch_e_t_zero_effect_size.name) = \
    ("exp1_catch_e_t_diff_group_desc", "exp1_catch_e_t_diff_group_stat", "exp1_catch_e_t_diff_zero_stat",
     "exp1_catch_e_t_effect_size", "exp1_catch_e_t_zero_effect_size")

# Save statistics for Latex manuscript
safe_save_dataframe(exp1_catch_e_t_diff_group_desc, 'age_group')
safe_save_dataframe(exp1_catch_e_t_diff_group_stat, 'test')
safe_save_dataframe(exp1_catch_e_t_diff_zero_stat, 'age_group')
safe_save_dataframe(exp1_catch_e_t_effect_size, 'type')
safe_save_dataframe(exp1_catch_e_t_zero_effect_size, 'type')

exp = 2
(exp2_catch_e_t_diff, exp2_catch_e_t_diff_group_desc, exp2_catch_e_t_diff_group_stat, exp2_catch_e_t_diff_zero_stat,
 exp2_catch_e_t_effect_size, exp2_catch_e_t_zero_effect_size) = get_catch_trial_results(df_exp2, exp)

(exp2_catch_e_t_diff_group_desc.name, exp2_catch_e_t_diff_group_stat.name, exp2_catch_e_t_diff_zero_stat.name,
 exp2_catch_e_t_effect_size.name, exp2_catch_e_t_zero_effect_size.name) = \
    ("exp2_catch_e_t_diff_group_desc", "exp2_catch_e_t_diff_group_stat", "exp2_catch_e_t_diff_zero_stat",
     " exp2_catch_e_t_effect_size", "exp2_catch_e_t_zero_effect_size")

safe_save_dataframe(exp2_catch_e_t_diff_group_desc, 'age_group')
safe_save_dataframe(exp2_catch_e_t_diff_group_stat, 'test')
safe_save_dataframe(exp2_catch_e_t_diff_zero_stat, 'age_group')
safe_save_dataframe(exp2_catch_e_t_effect_size, 'type')
safe_save_dataframe(exp2_catch_e_t_zero_effect_size, 'type')

# -------------------
# 3. Plot the results
# -------------------

# Size of figure
fig_height = 5
fig_width = 8

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))

# Create plot grid
gs_0 = gridspec.GridSpec(1, 1, top=0.85, bottom=0.25, left=0.2)

# Plot colors
colors = ["#BBE1FA", "#3282B8", "#0F4C75", "#1B262C"]
sns.set_palette(sns.color_palette(colors))

# Create subplot grid
gs_00 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_0[0], wspace=0.7)

# Experiment 1
# ------------

# Create subplot
ax_01 = plt.Subplot(f, gs_00[0, 0])
f.add_subplot(ax_01)

# Plot results
exp = 1
ax_01 = swarm_boxplot(ax_01, exp1_catch_e_t_diff, 'e_t_diff', 'Estimation Error\nReduction Catch Trials', exp)
ax_01.set_title('First Experiment')
ax_01.set_ylim(-12.5, 5)  # outlier excluded here

# Experiment 2
# ------------

# Create subplot
ax_02 = plt.Subplot(f, gs_00[0, 1])
f.add_subplot(ax_02)

# Plot results
exp = 2
ax_02 = swarm_boxplot(ax_02, exp2_catch_e_t_diff, 'e_t_diff', '', exp)
ax_02.set_title('Follow-Up Experiment')

ax_02.set_ylim(-12.5, 5)  # outlier excluded here
sns.despine()

# -------------------------------------
# 4. Add subplot labels and save figure
# -------------------------------------

# Add labels
texts = ['a', 'b']
label_subplots(f, texts, x_offset=0.15, y_offset=0.0)

# Save for manuscript
savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_S_figure_2.png"
plt.savefig(savename, transparent=False, dpi=400)

# Show plots
plt.ioff()
plt.show()
