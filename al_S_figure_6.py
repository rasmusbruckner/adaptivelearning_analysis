""" Figure S6

1. Load and prepare data
2. Prepare figure
3. Plot perseveration and estimation errors for reward conditions
4. Plot perseveration and estimation errors for bucket conditions
5. Add subplot labels and save figure
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import os
from al_utilities import get_mean_voi
from al_plot_utils import cm2inch, label_subplots, latex_plt, get_cond_diff


# Update matplotlib to use Latex and to change some defaults
os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2016/bin/x86_64-darwin'
matplotlib = latex_plt(matplotlib)

# Get home directory
paths = os.getcwd()
path = paths.split(os.path.sep)
home_dir = path[1]

# -------------------------
# 1. Load and prepare data
# -------------------------

df_2 = pd.read_pickle('al_data/data_prepr_2.pkl')

# Read out high- and low reward trials
low = df_2[df_2['r_t'] == 0.25]
high = df_2[df_2['r_t'] == 1]

#  Read out stable- and shifting bucket trials
df_noPush = df_2[df_2['cond'] == 'main_noPush']
df_push = df_2[df_2['cond'] == 'main_push']

# Perseveration-frequency differences between high- and low-reward condition
voi = 2
_, fig_S_6_a_desc, _, fig_S_6_a_zero_stat = get_cond_diff(low, high, voi)
fig_S_6_a_zero_stat.to_csv('~/Dropbox/Apps/Overleaf/al_manuscript/al_dataframes/fig_S_6_a_zero_stat.csv')
fig_S_6_a_desc.to_csv('~/Dropbox/Apps/Overleaf/al_manuscript/al_dataframes/fig_S_6_a_desc.csv')

# Estimation-error differences between high- and low-reward condition
voi = 1
_, fig_S_6_b_desc, _, fig_S_6_b_zero_stat = get_cond_diff(low, high, voi)
fig_S_6_b_zero_stat.to_csv('~/Dropbox/Apps/Overleaf/al_manuscript/al_dataframes/fig_S_6_b_zero_stat.csv')
fig_S_6_b_desc.to_csv('~/Dropbox/Apps/Overleaf/al_manuscript/al_dataframes/fig_S_6_b_desc.csv')

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
# 2. Prepare figure
# -----------------

colors = ["#92e0a9",  "#6d6192", "#352d4d"]
sns.set_palette(sns.color_palette(colors))

fig_height = 15
fig_witdh = 15
f = plt.figure(figsize=cm2inch(fig_witdh, fig_height))

# -----------------------------------------------------------------
# 3. Plot perseveration and estimation errors for reward conditions
# -----------------------------------------------------------------

# Perseveration frequency
plt.subplot(321)
ax_00 = plt.gca()
vertical_stack = pd.concat([pers_high, pers_low], axis=0, sort=False)
sns.boxplot(x='Condition', hue='age_group', y='pers', data=vertical_stack, notch=False, showfliers=False,
            linewidth=0.8, width=0.4, boxprops=dict(alpha=1), ax=ax_00)
ax_00.set_ylabel('Estimated\nperseveration probability')
colors = ["#92e0a9",  "#6d6192", "#352d4d"]
ax_00.set_xticklabels(['High reward', 'Low reward'])

# Plot legend
A0 = Line2D([0], [0], color=colors[0], lw=2)
A1 = Line2D([0], [0], color=colors[1], lw=2)
A2 = Line2D([0], [0], color=colors[2], lw=2)
ax_00.legend([A0, A1, A2], ['CH', 'YA', 'OA'])

# Estimation error
plt.subplot(322)
ax_01 = plt.gca()
vertical_stack = pd.concat([est_err_high, est_err_low], axis=0, sort=False)
sns.boxplot(x='Condition', hue='age_group', y='e_t', data=vertical_stack, notch=False, showfliers=False,
            linewidth=0.8, width=0.4, boxprops=dict(alpha=1), ax=ax_01)
ax_01.set_xticklabels(['High reward', 'Low reward'])
ax_01.get_legend().remove()
ax_01.set_ylabel('Estimation error')

# -----------------------------------------------------------------
# 4. Plot perseveration and estimation errors for bucket conditions
# -----------------------------------------------------------------

# Perseveration frequency
plt.subplot(323)
ax_10 = plt.gca()
vertical_stack = pd.concat([pers_noPush, pers_push], axis=0, sort=False)
sns.boxplot(x='Condition', hue='age_group', y='pers', data=vertical_stack, notch=False, showfliers=False,
            linewidth=0.8, width=0.4, boxprops=dict(alpha=1), ax=ax_10)
ax_10.set_ylabel('Estimated\nperseveration probability')
ax_10.set_xticklabels(['Stable bucket', 'Shifting bucket'])
ax_10.get_legend().remove()

# Estimation error
plt.subplot(324)
ax_11 = plt.gca()
vertical_stack = pd.concat([est_err_noPush, est_err_push], axis=0, sort=False)
sns.boxplot(x='Condition', hue='age_group', y='e_t', data=vertical_stack, notch=False, showfliers=False,
            linewidth=0.8, width=0.4, boxprops=dict(alpha=1), ax=ax_11)
ax_11.set_ylabel('Estimation error')
ax_11.set_xticklabels(['Stable bucket', 'Shifting bucket'])
ax_11.get_legend().remove()

# -----------------------------------------------------------------
# 4. Plot perseveration and estimation errors for bucket conditions
# -----------------------------------------------------------------

# Perseveration probability
plt.subplot(325)
ax_10 = plt.gca()
sns.boxplot(x='edge', hue='age_group', y='pers', data=pers_edge, notch=False, showfliers=False,
            linewidth=0.8, width=0.4, boxprops=dict(alpha=1), ax=ax_10)
ax_10.set_ylabel('Estimated \nperseveration probability')
ax_10.set_xticklabels(['False', 'True'])
ax_10.set_xlabel('Edge trial')
ax_10.get_legend().remove()
ax_10.set_title('Shifting bucket')

# Motor-perseveration probability
plt.subplot(326)
ax_10 = plt.gca()
sns.boxplot(x='edge', hue='age_group', y='motor_pers', data=motor_pers_edge, notch=False, showfliers=False,
            linewidth=0.8, width=0.4, boxprops=dict(alpha=1), ax=ax_10)
ax_10.set_ylabel('Estimated motor-\nperseveration probability')
ax_10.set_xticklabels(['False', 'True'])
ax_10.set_xlabel('Edge trial')
ax_10.get_legend().remove()
ax_10.set_title('Shifting bucket')

# -------------------------------------
# 5. Add subplot labels and save figure
# -------------------------------------

# Adjust space and axes
plt.subplots_adjust(left=0.15, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
sns.despine()

texts = ['a', 'b', 'c', 'd', 'e', 'f']  # label letters
label_subplots(f, texts, x_offset=0.08, y_offset=0.0)

savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_S_figure_6.pdf"
plt.savefig(savename, transparent=True, dpi=400)

# Show figure
plt.show()
