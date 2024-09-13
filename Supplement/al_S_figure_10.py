""" Figure S10: Model-fitting results experiment 2

    1. Load data
    2. Run statistical tests
    3. Prepare figure
    4. Plot parameter estimates
    5. Add subplot labels and save figure
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from al_utilities import get_stats, safe_save_dataframe
from al_plot_utils import cm2inch, label_subplots, swarm_boxplot, latex_plt


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

model_exp2 = pd.read_pickle('al_data/estimates_follow_up_exp_10_sp.pkl')

# ------------------------
# 2. Run statistical tests
# ------------------------

# Print out statistics for paper
print('\n\nOmikron 0\n')
exp2_omikron_0_desc, exp2_omikron_0_stat, exp2_omikron_0_effect_size = get_stats(model_exp2, 2, 'omikron_0')
print('\n\nOmikron 1\n')
exp2_omikron_1_desc, exp2_omikron_1_stat, exp2_omikron_1_effect_size = get_stats(model_exp2, 2, 'omikron_1')
print('\n\nb_0 parameter\n')
exp2_b_0_desc, exp2_b_0_stat, exp2_b_0_effect_size = get_stats(model_exp2, 2, 'b_0')
print('\n\nb_1 parameter\n')
exp2_b_1_desc, exp2_b_1_stat, exp2_b_1_effect_size = get_stats(model_exp2, 2, 'b_1')
print('\n\nUncertainty-underestimation parameter 1\n')
exp2_u_desc, exp2_u_stat, exp2_u_effect_size = get_stats(model_exp2, 2, 'u')
print('\n\nSurprise-sensitivity parameter 1\n')
exp2_s_desc, exp2_s_stat, exp2_s_effect_size = get_stats(model_exp2, 2, 's')
print('\n\nHazard-rate parameter 1\n')
exp2_h_desc, exp2_h_stat, exp2_h_effect_size = get_stats(model_exp2, 2, 'h')
print('\n\nCatch-trial parameter\n')
exp2_sigma_H_desc, exp2_sigma_H_stat, exp2_sigma_H_effect_size = get_stats(model_exp2, 2, 'sigma_H')
print('\n\nAnchoring-bias parameter\n')
exp2_d_desc, exp2_d_stat, exp2_d_effect_size = get_stats(model_exp2, 2, 'd')

exp2_model_fitting_desc = pd.concat([exp2_omikron_0_desc.add_suffix('_omikron_0'),
                                     exp2_omikron_1_desc.add_suffix('_omikron_1'),
                                     exp2_b_0_desc.add_suffix('_b_0'),
                                     exp2_b_1_desc.add_suffix('_b_1'),
                                     exp2_u_desc.add_suffix('_u'),
                                     exp2_s_desc.add_suffix('_s'),
                                     exp2_h_desc.add_suffix('_h'),
                                     exp2_sigma_H_desc.add_suffix('_sigma_h'),
                                     exp2_d_desc.add_suffix('_d')], axis=1)

exp2_model_fitting_stat = pd.concat([exp2_omikron_0_stat.add_suffix('_omikron_0'),
                                     exp2_omikron_1_stat.add_suffix('_omikron_1'),
                                     exp2_b_0_stat.add_suffix('_b_0'),
                                     exp2_b_1_stat.add_suffix('_b_1'),
                                     exp2_u_stat.add_suffix('_u'),
                                     exp2_s_stat.add_suffix('_s'),
                                     exp2_h_stat.add_suffix('_h'),
                                     exp2_sigma_H_stat.add_suffix('_sigma_H'),
                                     exp2_d_stat.add_suffix('_d')], axis=1)

exp2_model_fitting_effect_size = pd.concat([exp2_omikron_0_effect_size.add_suffix('_omikron_0'),
                                            exp2_omikron_1_effect_size.add_suffix('_omikron_1'),
                                            exp2_b_0_effect_size.add_suffix('_b_0'),
                                            exp2_b_1_effect_size.add_suffix('_b_1'),
                                            exp2_u_effect_size.add_suffix('_u'),
                                            exp2_s_effect_size.add_suffix('_s'),
                                            exp2_h_effect_size.add_suffix('_h'),
                                            exp2_sigma_H_effect_size.add_suffix('_sigma_h'),
                                            exp2_d_effect_size.add_suffix('_d')], axis=1)

exp2_model_fitting_desc.name, exp2_model_fitting_stat.name, exp2_model_fitting_effect_size.name =\
    "exp2_model_fitting_desc", "exp2_model_fitting_stat", "exp2_model_fitting_effect_size"

# Save statistics for Latex manuscript
safe_save_dataframe(exp2_model_fitting_desc, 'age_group')
safe_save_dataframe(exp2_model_fitting_stat, 'test')
safe_save_dataframe(exp2_model_fitting_effect_size, 'type')

# -----------------
# 3. Prepare figure
# -----------------

# Plot colors
colors = ["#BBE1FA", "#0F4C75", "#1B262C"]
sns.set_palette(sns.color_palette(colors))

# Create figure
fig_height = 12
fig_witdh = 15
f = plt.figure(figsize=cm2inch(fig_witdh, fig_height))

# Y-label distance
ylabel_dist = -0.325

# ---------------------------
# 4. Plot parameter estimates
# ---------------------------

# omikron_0
plt.subplot(341)
ax_00 = plt.gca()
swarm_boxplot(ax_00, model_exp2, 'omikron_0', 'Parameter Estimate', 2)
ax_00.yaxis.set_label_coords(ylabel_dist, 0.5)
ax_00.set_title('Motor Noise')

# omikron_1
plt.subplot(342)
ax_01 = plt.gca()
swarm_boxplot(ax_01, model_exp2, 'omikron_1', ' ', 2)
ax_01.set_title('Learning-Rate Noise')

# b_0
plt.subplot(343)
ax_02 = plt.gca()
swarm_boxplot(ax_02, model_exp2, 'b_0', ' ', 2)
ax_02.set_title('Intercept')

# b_1
plt.subplot(344)
ax_10 = plt.gca()
swarm_boxplot(ax_10, model_exp2, 'b_1', ' ', 2)
ax_10.set_title('Slope')

# u
plt.subplot(345)
ax_11 = plt.gca()
swarm_boxplot(ax_11, model_exp2, 'u', 'Parameter Estimate', 2)
ax_11.yaxis.set_label_coords(ylabel_dist, 0.5)
ax_11.set_title('Uncertainty\nUnderestimation')

# s
plt.subplot(346)
ax_20 = plt.gca()
swarm_boxplot(ax_20, model_exp2, 's', ' ', 2)
ax_20.set_title('Surprise Sensitivity')

# h
plt.subplot(347)
ax_12 = plt.gca()
swarm_boxplot(ax_12, model_exp2, 'h', ' ', 2)
ax_12.set_title('Hazard Rate')

# sigma_H
plt.subplot(348)
ax_21 = plt.gca()
swarm_boxplot(ax_21, model_exp2, 'sigma_H', ' ', 2)
ax_21.set_title('Catch Trial')

# d
plt.subplot(349)
ax_21 = plt.gca()
swarm_boxplot(ax_21, model_exp2, 'd', 'Parameter Estimate', 2)
ax_21.yaxis.set_label_coords(ylabel_dist, 0.5)
ax_21.set_title('Anchoring Bias')

# -------------------------------------
# 5. Add subplot labels and save figure
# -------------------------------------

# Adjust space and axes
plt.subplots_adjust(wspace=0.5, hspace=0.7, top=0.9, bottom=0.1, left=0.1, right=0.95)
sns.despine()

texts = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']  # label letters
label_subplots(f, texts, x_offset=0.08, y_offset=0.0)

savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_S_figure_10.pdf"
plt.savefig(savename, transparent=True, dpi=400)

# Show figure
plt.show()
