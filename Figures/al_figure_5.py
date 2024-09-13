""" Figure 5: Descriptive results second experiment

    1. Load data
    2. Perseveration differences between conditions
    3. Estimation-error differences between conditions
    4. Compute average learning rates and anchoring bias
    5. Prepare figure
    6. Plot perseveration, performance, and anchoring bias
    7. Plot robust linear regression of perseveration probability and anchoring bias
    8. Add subplot labels and save figure
"""


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os
from al_utilities import (get_mean_voi, get_stats, safe_save_dataframe, compute_rob_reg_effect_size,
                          compute_anchoring_bias, get_cond_diff)
from al_plot_utils import cm2inch, label_subplots, swarm_boxplot, latex_plt, custom_boxplot_condition, text_legend


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
n_subj = len(np.unique(df_exp2['subj_num']))

# -----------------------------------------------
# 2. Perseveration differences between conditions
# -----------------------------------------------

# Read out push and noPush trials
df_noPush = df_exp2[df_exp2['cond'] == 'main_noPush']
df_push = df_exp2[df_exp2['cond'] == 'main_push']

# Perseveration
voi = 2
pers_noPush = get_mean_voi(df_noPush, voi)
pers_push = get_mean_voi(df_push, voi)

# Compute perseveration differences
exp2_pers_diff, exp2_pers_diff_desc, exp2_pers_diff_stat, exp2_pers_diff_zero_stat, exp2_pers_diff_effect_size, \
    exp2_pers_diff_zero_effect_size = get_cond_diff(df_noPush, df_push, voi)

exp2_pers_diff_desc.name, exp2_pers_diff_stat.name, exp2_pers_diff_zero_stat.name, exp2_pers_diff_effect_size.name, \
    exp2_pers_diff_zero_effect_size.name = "exp2_pers_diff_desc", "exp2_pers_diff_stat", "exp2_pers_diff_zero_stat", \
    "exp2_pers_diff_effect_size", "exp2_pers_diff_zero_effect_size"

# Save statistics for Latex manuscript
safe_save_dataframe(exp2_pers_diff_desc, 'age_group')
safe_save_dataframe(exp2_pers_diff_stat, 'test')
safe_save_dataframe(exp2_pers_diff_zero_stat, 'age_group')
safe_save_dataframe(exp2_pers_diff_effect_size, 'type')
safe_save_dataframe(exp2_pers_diff_zero_effect_size, 'type')

# Compute perseveration descriptives for Figure 6 (comparison sampling model and empirical data)
print('\n\nPerseveration results standard condition\n')
print('Comparing absolute perseveration between groups:\n')
voi = 2
exp2_pers_abs_stand_desc, _, _ = get_stats(pers_noPush, voi, 'pers')
exp2_pers_abs_stand_desc.name = "exp2_pers_abs_stand_desc"
safe_save_dataframe(exp2_pers_abs_stand_desc, 'age_group')

# Compute absolut perseveration in anchoring condition for manuscript
print('\n\nPerseveration results anchoring condition:\n')
print('Comparing absolute perseveration between groups:\n')
voi = 2
exp2_pers_abs_anchor_desc, exp2_pers_abs_anchor_stat, exp2_pers_abs_anchor_effect_size =\
    get_stats(pers_push, voi, 'pers')
exp2_pers_abs_anchor_desc.name, exp2_pers_abs_anchor_stat.name, exp2_pers_abs_anchor_effect_size.name = \
    "exp2_pers_abs_anchor_desc", "exp2_pers_abs_anchor_stat", "exp2_pers_abs_anchor_effect_size"
safe_save_dataframe(exp2_pers_abs_anchor_desc, 'age_group')
safe_save_dataframe(exp2_pers_abs_anchor_stat, 'test')
safe_save_dataframe(exp2_pers_abs_anchor_effect_size, 'type')

# --------------------------------------------------
# 3. Estimation-error differences between conditions
# --------------------------------------------------

voi = 1
exp2_est_err_diff, exp2_est_err_diff_desc, exp2_est_err_diff_stat, exp2_est_err_diff_zero_stat, exp2_est_err_diff_effect_size, \
    exp2_est_err_diff_zero_effect_size = get_cond_diff(df_noPush, df_push, voi)

exp2_est_err_diff_desc.name, exp2_est_err_diff_stat.name, exp2_est_err_diff_zero_stat.name, exp2_est_err_diff_effect_size.name, \
    exp2_est_err_diff_zero_effect_size.name = "exp2_est_err_diff_desc", "exp2_est_err_diff_stat", "exp2_est_err_diff_zero_stat", "exp2_est_err_diff_effect_size", \
    "exp2_est_err_diff_zero_effect_size"

# Save statistics for Latex manuscript
safe_save_dataframe(exp2_est_err_diff_desc, 'age_group')
safe_save_dataframe(exp2_est_err_diff_stat, 'test')
safe_save_dataframe(exp2_est_err_diff_zero_stat, 'age_group')
safe_save_dataframe(exp2_est_err_diff_effect_size, 'type')
safe_save_dataframe(exp2_est_err_diff_zero_effect_size, 'type')

# Estimation error
voi = 1
est_err_noPush = get_mean_voi(df_noPush, voi)
est_err_push = get_mean_voi(df_push, voi)

# Print out estimation error stats for Figure 6 (comparison sampling model and empirical data)
print('\n\nEstimation error results standard condition\n')
print('Comparing absolute estimation error between groups:\n')
voi = 2
exp2_est_err_abs_stand_desc, _, _ = get_stats(est_err_noPush, voi, 'e_t')
exp2_est_err_abs_stand_desc.name = "exp2_est_err_abs_stand_desc"
safe_save_dataframe(exp2_est_err_abs_stand_desc, 'age_group')

print('\n\nEstimation error results anchoring condition\n')
print('Comparing absolute estimation error between groups:\n')
voi = 2
exp2_est_err_abs_anchor_desc, _, _ = get_stats(est_err_push, voi, 'e_t')
exp2_est_err_abs_anchor_desc.name = "exp2_est_err_abs_anchor_desc"
safe_save_dataframe(exp2_est_err_abs_anchor_desc, 'age_group')

# ----------------------------------------------------
# 4. Compute average learning rates and anchoring bias
# ----------------------------------------------------

a_t_name = 'a_t'
y_t_name = 'y_t'
df_reg = compute_anchoring_bias(n_subj, df_exp2, a_t_name, y_t_name)

# Print out average learning rate statistics for paper
print('\n\nAlpha Experiment 2\n')
exp2_alpha_desc, exp2_alpha_stat, exp2_alpha_effect_size = get_stats(df_reg, 2, 'alpha')

# Print out average anchoring-bias statistics for paper
print('\n\nAnchoring bias Experiment 2\n')
exp2_anchoring_desc, exp2_anchoring_stat, exp2_anchoring_effect_size = get_stats(df_reg, 2, 'bucket_bias')

exp2_df_reg_desc = pd.concat([exp2_alpha_desc.add_suffix('_alpha'), exp2_anchoring_desc.add_suffix('_bb')], axis=1)
exp2_df_reg_stat = pd.concat([exp2_alpha_stat.add_suffix('_alpha'), exp2_anchoring_stat.add_suffix('_bb')], axis=1)
exp2_df_reg_effect_size = pd.concat(
    [exp2_alpha_effect_size.add_suffix('_alpha'), exp2_anchoring_effect_size.add_suffix('_bb')], axis=1)
exp2_df_reg_desc.name, exp2_df_reg_stat.name, exp2_df_reg_effect_size.name = "exp2_df_reg_desc", "exp2_df_reg_stat", "exp2_df_reg_effect_size"

# Save statistics for Latex manuscript
safe_save_dataframe(exp2_df_reg_desc, 'age_group')
safe_save_dataframe(exp2_df_reg_stat, 'test')
safe_save_dataframe(exp2_df_reg_effect_size, 'type')

# -----------------
# 5. Prepare figure
# -----------------

# Turn interactive plotting mode on for debugger
plt.ion()

# Size of figure
fig_height = 5
fig_width = 15

# Image format
saveas = "pdf"  # "png"

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))

# Create plot grid
gs_0 = gridspec.GridSpec(1, 1, wspace=1, hspace=0.7, top=0.9, bottom=0.3, left=0.1, right=0.95)  # bottom=0.2

# Group colors
colors = ["#BBE1FA", "#0F4C75", "#1B262C"]
sns.set_palette(sns.color_palette(colors))

# Condition colors
condition_colors = ["#BBE1FA", "#3282B8", "#1B262C", "#e3f3fd", "#adcde2", "#babdbf"]

# ------------------------------------------------------
# 6. Plot perseveration, performance, and anchoring bias
# ------------------------------------------------------

# Create subplot grid
gs_00 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_0[0], hspace=0.6, wspace=0.5)

# Plot perseveration swarm-boxplot
exp = 1
ax_3 = plt.Subplot(f, gs_00[0, 0])
f.add_subplot(ax_3)

# Show perseveration probability
custom_boxplot_condition(ax_3, pers_noPush, pers_push, "pers", "Perseveration Probability", condition_colors)

# Plot custom legend
text_legend(plt.gca(), "Darker colors (left): Standard condition | Lighter colors (right): Anchoring condition",
            coords=[-0.5, -0.5])

# Plot estimation error swarm-boxplot
ax_2 = plt.Subplot(f, gs_00[0, 1])
f.add_subplot(ax_2)
custom_boxplot_condition(ax_2, est_err_noPush, est_err_push, "e_t", "Estimation Error", condition_colors)

# Anchoring bias
ax_20 = plt.Subplot(f, gs_00[0, 2])
f.add_subplot(ax_20)
swarm_boxplot(ax_20, df_reg, 'bucket_bias', 'Anchoring Bias', 2)
ax_20.set_ylim(-0.1, 0.52)

# --------------------------------------------------------------------------------
# 7. Plot robust linear regression of perseveration probability and anchoring bias
# --------------------------------------------------------------------------------

# Data frame for regression model
data = pd.DataFrame()
data['pers'] = pers_noPush['pers'].copy()
data['d'] = df_reg['bucket_bias']  # model_exp2['d'].copy()

# Robust linear regression
mod = smf.rlm(formula='d ~ pers', M=sm.robust.norms.TukeyBiweight(3), data=data)
res = mod.fit(conv="weights")
print(res.summary())

# Compute effect size
R_squared, r = compute_rob_reg_effect_size(res, data)

# Save statistics for Latex manuscript
exp2_pers_anchoring_reg = pd.DataFrame()
exp2_pers_anchoring_reg['params'] = np.array([round(res.params['Intercept'], 3), round(res.params['pers'], 3)])
exp2_pers_anchoring_reg['p'] = np.array([round(res.pvalues['Intercept'], 3), round(res.pvalues['pers'], 3)])
exp2_pers_anchoring_reg['z'] = np.array([round(res.tvalues['Intercept'], 3), round(res.tvalues['pers'], 3)])
exp2_pers_anchoring_reg = exp2_pers_anchoring_reg.rename({0: 'int', 1: 'pers'}, axis='index')
exp2_pers_anchoring_reg.name = "exp2_pers_anchoring_reg"
exp2_pers_anchoring_reg.index.name = 'coeff'
print(exp2_pers_anchoring_reg)
safe_save_dataframe(exp2_pers_anchoring_reg, 'coeff')

# Compute regression line
x_vals = np.array([0, 0.8])
reg_line = res.params['Intercept'] + res.params['pers'] * x_vals

# Plot results
ax_22 = plt.Subplot(f, gs_00[0, 3])
f.add_subplot(ax_22)
ax_22.plot(np.array(pers_noPush['pers']), np.array(df_reg['bucket_bias']), '.', color='gray', alpha=0.7, markersize=2)
ax_22.plot(x_vals, reg_line, '-', color="k")
ax_22.set_ylabel('Anchoring Bias')
ax_22.set_xlabel('Perseveration Probability')
ax_22.set_title('r = ' + str(np.round(r, 3)))
ax_22.set_xticks(np.arange(0, 1, 0.2))

# -------------------------------------
# 8. Add subplot labels and save figure
# -------------------------------------

# Delete unnecessary axes
sns.despine()

# Add labels
texts = ['a', 'b', 'c', 'd', 'e']  # label letters
label_subplots(f, texts, x_offset=0.08, y_offset=0.025)

# Save figure
# -----------

if saveas == "pdf":
    savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_figure_5.pdf"
else:
    savename = "/" + home_dir + "/rasmus/Dropbox/heli_lifespan/png/al_figure_5.png"

plt.savefig(savename, transparent=False, dpi=400)

# Show plot
plt.ioff()
plt.show()
