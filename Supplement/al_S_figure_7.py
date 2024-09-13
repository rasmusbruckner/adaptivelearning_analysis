""" Figure S7: Control model first experiment

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
from al_plot_utils import cm2inch, label_subplots, swarm_boxplot, latex_plt
from al_utilities import get_stats, safe_save_dataframe


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

model_results = pd.read_pickle('al_data/estimates_first_exp_no_pers_10_sp.pkl')

# -------------------------
# 2. Runs statistical tests
# -------------------------

# Print out statistics for paper
print('\n\nUncertainty-underestimation parameter\n')
exp1_control_desc, exp1_control_stat, exp1_control_effect_size = get_stats(model_results, 1, 'u')

exp1_control_desc, exp1_control_stat, exp1_control_effect_size = \
    exp1_control_desc.add_suffix('_u'), exp1_control_stat.add_suffix('_u'), exp1_control_effect_size.add_suffix('_u')
exp1_control_desc.name, exp1_control_stat.name, exp1_control_effect_size.name = (
    "exp1_control_desc", "exp1_control_stat", "exp1_control_effect_size")

# Save statistics for Latex manuscript
safe_save_dataframe(exp1_control_desc, 'age_group')
safe_save_dataframe(exp1_control_stat, 'test')
safe_save_dataframe(exp1_control_effect_size, 'type')

# -----------------
# 3. Prepare figure
# -----------------

# Plot colors
colors = ["#BBE1FA", "#3282B8", "#0F4C75", "#1B262C"]
sns.set_palette(sns.color_palette(colors))

# Size of figure
fig_width = 15
fig_height = 8

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))

# Y-label distance
ylabel_dist = -0.325

# ---------------------------
# 4. Plot parameter estimates
# ---------------------------

# omikron_0
plt.subplot(241)
ax_00 = plt.gca()
swarm_boxplot(ax_00,  model_results, 'omikron_0', 'Parameter Estimate', 1)
ax_00.yaxis.set_label_coords(ylabel_dist, 0.5)
ax_00.set_title('Motor Noise')

# omikron_1
plt.subplot(242)
ax_01 = plt.gca()
swarm_boxplot(ax_01,  model_results, 'omikron_1', '', 1)
ax_01.set_title('Learning-Rate Noise')

# u
plt.subplot(243)
ax_02 = plt.gca()
swarm_boxplot(ax_02,  model_results, 'u', '', 1)
ax_02.set_title('Uncertainty\nUnderestimation')

# s
plt.subplot(244)
ax_10 = plt.gca()
swarm_boxplot(ax_10,  model_results, 's', '', 1)
ax_10.set_title('Surprise Sensitivity')

# h
plt.subplot(245)
ax_11 = plt.gca()
swarm_boxplot(ax_11,  model_results, 'h', 'Parameter Estimate', 1)
ax_11.yaxis.set_label_coords(ylabel_dist, 0.5)
ax_11.set_title('Hazard Rate')

# q
plt.subplot(246)
ax_12 = plt.gca()
swarm_boxplot(ax_12,  model_results, 'q', '', 1)
ax_12.set_title('Reward Bias')

# sigma_H
plt.subplot(247)
ax_20 = plt.gca()
swarm_boxplot(ax_20,  model_results, 'sigma_H', '', 1)
ax_20.set_title('Catch Trial')

# -------------------------------------
# 5. Add subplot labels and save figure
# -------------------------------------

# Adjust space and axes
plt.subplots_adjust(wspace=0.5, hspace=0.7, top=0.9, bottom=0.125, left=0.1, right=0.95)
sns.despine()

texts = ['a', 'b', 'c', 'd', 'e', 'f', 'g']  # label letters
label_subplots(f, texts, x_offset=0.08, y_offset=0.0)

savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_S_figure_7.pdf"
plt.savefig(savename, transparent=True, dpi=400)

# Show figure
plt.show()
