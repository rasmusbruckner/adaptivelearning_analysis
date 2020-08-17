""" Figure S4

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
from al_utilities import get_stats


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

model_results = pd.read_pickle('al_data/estimates_first_exp_no_pers_25_sp.pkl')

# -------------------------
# 2. Runs statistical tests
# -------------------------

# Print out statistics for paper
print('\n\nUncertainty-underestimation parameter\n')
median_u, q1_u, q3_u, p_u, stat_u = get_stats(model_results, 1, 'u')

# Create data frames to save statistics for Latex manuscript
fig_S_4_desc = pd.DataFrame()
fig_S_4_stat = pd.DataFrame()

# Median parameter estimates
fig_S_4_desc['median_u'] = round(median_u, 3)

# First quartile
fig_S_4_desc['q1_u'] = round(q1_u, 3)

# Third quartile
fig_S_4_desc['q3_u'] = round(q3_u, 3)

# Rename index and groups
fig_S_4_desc.index.name = 'age_group'
fig_S_4_desc = fig_S_4_desc.rename({1: 'ch', 2: 'ad', 3: 'ya', 4: 'oa'}, axis='index')

# P-values and test statistics
fig_S_4_stat['p_u'] = p_u
fig_S_4_stat['stat_u'] = stat_u

# Rename index and tests
fig_S_4_stat.index.name = 'test'
fig_S_4_stat = fig_S_4_stat.rename({0: 'kw', 1: 'ch_ad', 2: 'ch_ya', 3: 'ch_oa', 4: 'ad_ya', 5: 'ad_oa', 6: 'ya_oa'},
                                     axis='index')

# Save statistics for Latex manuscript
fig_S_4_desc.to_csv('~/Dropbox/Apps/Overleaf/al_manuscript/al_dataframes/fig_S_4_desc.csv')
fig_S_4_stat.to_csv('~/Dropbox/Apps/Overleaf/al_manuscript/al_dataframes/fig_S_4_stat.csv')

# -----------------
# 3. Prepare figure
# -----------------

# Plot colors
colors = ["#92e0a9", "#69b0c1", "#6d6192", "#352d4d"]
sns.set_palette(sns.color_palette(colors))

# Create figure
fig_height = 13.5
fig_witdh = 15
f = plt.figure(figsize=cm2inch(fig_witdh, fig_height))

# ---------------------------
# 4. Plot parameter estimates
# ---------------------------

# omikron_0
plt.subplot(331)
ax_00 = plt.gca()
swarm_boxplot(ax_00,  model_results, 'omikron_0', 'Parameter estimate', 2)
ax_00.set_title('Motor noise')

# omikron_1
plt.subplot(332)
ax_01 = plt.gca()
swarm_boxplot(ax_01,  model_results, 'omikron_1', 'Parameter estimate', 2)
ax_01.set_title('Learning-rate noise')

# u
plt.subplot(333)
ax_02 = plt.gca()
swarm_boxplot(ax_02,  model_results, 'u', 'Parameter estimate', 2)
ax_02.set_title('Uncertainty underestimation')

# s
plt.subplot(334)
ax_10 = plt.gca()
swarm_boxplot(ax_10,  model_results, 's', 'Parameter estimate', 2)
ax_10.set_title('Surprise sensitivity')

# h
plt.subplot(335)
ax_11 = plt.gca()
swarm_boxplot(ax_11,  model_results, 'h', 'Parameter estimate', 2)
ax_11.set_title('Hazard rate')

# q
plt.subplot(336)
ax_12 = plt.gca()
swarm_boxplot(ax_12,  model_results, 'q', 'Parameter estimate', 2)
ax_12.set_title('Reward bias')

# sigma_H
plt.subplot(337)
ax_20 = plt.gca()
swarm_boxplot(ax_20,  model_results, 'sigma_H', 'Parameter estimate', 2)
ax_20.set_title('Catch trial')

# -------------------------------------
# 5. Add subplot labels and save figure
# -------------------------------------

# Adjust space and axes
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.6)
sns.despine()

texts = ['a', 'b', 'c', 'd', 'e', 'f', 'g']  # label letters
label_subplots(f, texts, x_offset=0.08, y_offset=0.0)

savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_S_figure_4.pdf"
plt.savefig(savename, transparent=True, dpi=400)

# Show figure
plt.show()
