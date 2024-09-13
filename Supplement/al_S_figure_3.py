""" Figure S3: Supplementary results second experiment

1. Load and prepare data
2. Average learning rate
3. Prepare figure
4  Plot perseveration and estimation errors for reward conditions
5. Plot perseveration and estimation errors for anchoring conditions
6. Plot perseveration for edge trials
7. Add subplot labels and save figure
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import os
from al_utilities import get_mean_voi, safe_save_dataframe, get_cond_diff
from al_plot_utils import cm2inch, label_subplots, latex_plt


# Update matplotlib to use Latex and to change some defaults
matplotlib = latex_plt(matplotlib)

# Get home directory
paths = os.getcwd()
path = paths.split(os.path.sep)
home_dir = path[1]

# ------------------------
# 1. Load and prepare data
# ------------------------

df_exp2 = pd.read_pickle('al_data/data_prepr_2.pkl')
n_subj = len(np.unique(df_exp2['subj_num']))

# Read out high- and low reward trials
low = df_exp2[df_exp2['r_t'] == 0.25]
high = df_exp2[df_exp2['r_t'] == 1]

# Read out stable- and shifting bucket trials
df_noPush = df_exp2[df_exp2['cond'] == 'main_noPush']
df_push = df_exp2[df_exp2['cond'] == 'main_push']

# Perseveration-frequency differences between high- and low-reward condition
voi = 2
_, exp2_rew_cond_pers_desc, _, exp2_rew_cond_pers_zero_stat, _, exp2_rew_cond_pers_effect_size_zero, =\
    get_cond_diff(low, high, voi)

exp2_rew_cond_pers_desc.name, exp2_rew_cond_pers_zero_stat.name, exp2_rew_cond_pers_effect_size_zero.name, =\
    "exp2_rew_cond_pers_desc", "exp2_rew_cond_pers_zero_stat", "exp2_rew_cond_pers_effect_size_zero"

# Save statistics for Latex manuscript
safe_save_dataframe(exp2_rew_cond_pers_desc, 'age_group')
safe_save_dataframe(exp2_rew_cond_pers_zero_stat, 'age_group')
safe_save_dataframe(exp2_rew_cond_pers_effect_size_zero, 'type')

# Estimation-error differences between high- and low-reward condition
voi = 1
_, exp2_rew_cond_est_err_desc, _, exp2_rew_cond_est_err_zero_stat, _, exp2_rew_cond_est_err_effect_size_zero =\
    get_cond_diff(low, high, voi)

exp2_rew_cond_est_err_desc.name, exp2_rew_cond_est_err_zero_stat.name, exp2_rew_cond_est_err_effect_size_zero.name\
    = "exp2_rew_cond_est_err_desc", "exp2_rew_cond_est_err_zero_stat", "exp2_rew_cond_est_err_effect_size_zero"

# Save statistics for Latex manuscript
safe_save_dataframe(exp2_rew_cond_est_err_desc, 'age_group')
safe_save_dataframe(exp2_rew_cond_est_err_zero_stat, 'age_group')
safe_save_dataframe(exp2_rew_cond_est_err_effect_size_zero, 'type')

# Perseveration in high- and low reward condition
voi = 2
pers_high = get_mean_voi(high, voi)
pers_low = get_mean_voi(low, voi)
pers_high['Condition'] = 1
pers_low['Condition'] = 2

# Perseveration in stable- and shifting-bucket condition
voi = 2
pers_noPush = get_mean_voi(df_noPush, voi)
pers_push = get_mean_voi(df_push, voi)
pers_noPush['Condition'] = 1
pers_push['Condition'] = 2

# Estimation errors in high- and low-reward condition
voi = 1
est_err_high = get_mean_voi(high, voi)
est_err_low = get_mean_voi(low, voi)
est_err_high['Condition'] = 1
est_err_low['Condition'] = 2

# Estimation errors in stable- and shifting-bucket condition
voi = 1
est_err_noPush = get_mean_voi(df_noPush, voi)
est_err_push = get_mean_voi(df_push, voi)
est_err_noPush['Condition'] = 1
est_err_push['Condition'] = 2

# Perseveration edge effects in shifting-bucket condition
voi = 3
pers_edge = get_mean_voi(df_push, voi)

# Motor-perseveration edge effects in shifting-bucket condition
voi = 4
motor_pers_edge = get_mean_voi(df_push, voi)

# -----------------
# 3. Prepare figure
# -----------------

# Group colors
colors = ["#BBE1FA", "#0F4C75", "#1B262C"]
sns.set_palette(sns.color_palette(colors))

# Size of figure
fig_width = 15
fig_height = 10

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))

# Y-label distance
ylabel_dist = -0.175

# -----------------------------------------------------------------
# 4. Plot perseveration and estimation errors for reward conditions
# -----------------------------------------------------------------

# Perseveration frequency
plt.subplot(321)
ax_01 = plt.gca()
vertical_stack = pd.concat([pers_high, pers_low], axis=0, sort=False)
sns.boxplot(x='Condition', hue='age_group', y='pers', data=vertical_stack, notch=False, showfliers=False, showcaps=False,
            linewidth=0.8, width=0.4, boxprops=dict(alpha=1), ax=ax_01, palette=colors)

ax_01.set_ylabel('Perseveration\nProbability')
ax_01.yaxis.set_label_coords(ylabel_dist, 0.5)
ax_01.xaxis.set_ticks([0, 1])
ax_01.set_xticklabels(['High Reward', 'Low Reward'])

# Plot legend
A0 = Line2D([0], [0], color=colors[0], lw=2)
A1 = Line2D([0], [0], color=colors[1], lw=2)
A2 = Line2D([0], [0], color=colors[2], lw=2)
ax_01.legend([A0, A1, A2], ['CH', 'YA', 'OA'], loc='upper right', bbox_to_anchor=(1.25, 0.9))

# Estimation error
plt.subplot(322)
ax_02 = plt.gca()
vertical_stack = pd.concat([est_err_high, est_err_low], axis=0, sort=False)
sns.boxplot(x='Condition', hue='age_group', y='e_t', data=vertical_stack, notch=False, showfliers=False, showcaps=False,
            linewidth=0.8, width=0.4, boxprops=dict(alpha=1), ax=ax_02, palette=colors)
ax_02.xaxis.set_ticks([0, 1])
ax_02.set_xticklabels(['High Reward', 'Low Reward'])
ax_02.get_legend().remove()
ax_02.set_ylabel('Estimation Error')
ax_02.yaxis.set_label_coords(ylabel_dist, 0.5)

# --------------------------------------------------------------------
# 5. Plot perseveration and estimation errors for anchoring conditions
# --------------------------------------------------------------------

# Perseveration frequency
plt.subplot(323)
ax_03 = plt.gca()
vertical_stack = pd.concat([pers_noPush, pers_push], axis=0, sort=False)
sns.boxplot(x='Condition', hue='age_group', y='pers', data=vertical_stack, notch=False, showfliers=False, showcaps=False,
            linewidth=0.8, width=0.4, boxprops=dict(alpha=1), ax=ax_03, palette=colors)
ax_03.set_ylabel('Perseveration\nProbability')
ax_03.yaxis.set_label_coords(ylabel_dist, 0.5)
ax_03.xaxis.set_ticks([0, 1])
ax_03.set_xticklabels(['Standard', 'Anchoring'])
ax_03.get_legend().remove()

# Estimation error
plt.subplot(324)
ax_04 = plt.gca()
vertical_stack = pd.concat([est_err_noPush, est_err_push], axis=0, sort=False)
sns.boxplot(x='Condition', hue='age_group', y='e_t', data=vertical_stack, notch=False, showfliers=False, showcaps=False,
            linewidth=0.8, width=0.4, boxprops=dict(alpha=1), ax=ax_04, palette=colors)
ax_04.set_ylabel('Estimation Error')
ax_04.yaxis.set_label_coords(ylabel_dist, 0.5)
ax_04.xaxis.set_ticks([0, 1])
ax_04.set_xticklabels(['Standard', 'Anchoring'])
ax_04.get_legend().remove()

# -------------------------------------
# 6. Plot perseveration for edge trials
# -------------------------------------

# Perseveration probability
plt.subplot(325)
ax_05 = plt.gca()
sns.boxplot(x='edge', hue='age_group', y='pers', data=pers_edge, notch=False, showfliers=False, showcaps=False,
            linewidth=0.8, width=0.4, boxprops=dict(alpha=1), ax=ax_05, palette=colors)
ax_05.set_ylabel('Perseveration\nProbability')
ax_05.yaxis.set_label_coords(ylabel_dist, 0.5)
ax_05.xaxis.set_ticks([0, 1])
ax_05.set_xticklabels(['False', 'True'])
ax_05.set_xlabel('Edge Trial')
ax_05.get_legend().remove()
ax_05.set_title('Anchoring Condition')

# Motor-perseveration probability
plt.subplot(326)
ax_06 = plt.gca()
sns.boxplot(x='edge', hue='age_group', y='motor_pers', data=motor_pers_edge, notch=False, showfliers=False, showcaps=False,
            linewidth=0.8, width=0.4, boxprops=dict(alpha=1), ax=ax_06, palette=colors)
ax_06.set_ylabel('Motor-Perseveration\nProbability')
ax_06.yaxis.set_label_coords(ylabel_dist, 0.5)
ax_06.xaxis.set_ticks([0, 1])
ax_06.set_xticklabels(['False', 'True'])
ax_06.set_xlabel('Edge Trial')
ax_06.get_legend().remove()
ax_06.set_title('Anchoring Condition')

# -------------------------------------
# 7. Add subplot labels and save figure
# -------------------------------------

# Adjust space and axes
plt.subplots_adjust(left=0.15, bottom=0.1, right=None, top=0.95, wspace=0.5, hspace=0.7)
sns.despine()

texts = ['a', 'b', 'c', 'd', 'e', 'f']  # label letters
label_subplots(f, texts, x_offset=0.10, y_offset=0.0)

savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_S_figure_3.pdf"
plt.savefig(savename, transparent=True, dpi=400)

# Show figure
plt.show()
