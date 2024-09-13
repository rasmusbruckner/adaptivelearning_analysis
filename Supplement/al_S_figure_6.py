""" Figure S6: Additional model-fitting results experiment 1

1. Load data
2. Prepare figure
3. Plot parameter estimates
4. Add subplot labels and save figure
"""

import pandas as pd
import matplotlib
import seaborn as sns
import os
import matplotlib.pyplot as plt
from al_plot_utils import cm2inch, label_subplots, latex_plt, swarm_boxplot


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

model_exp1 = pd.read_pickle('al_data/estimates_first_exp_10_sp.pkl')

# -----------------
# 2. Prepare figure
# -----------------

# Size of figure
fig_width = 15
fig_height = 8

# Plot colors
colors = ["#BBE1FA", "#3282B8", "#0F4C75", "#1B262C"]
sns.set_palette(sns.color_palette(colors))

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))

# Y-label distance
ylabel_dist = -0.325

# ---------------------------
# 3. Plot parameter estimates
# ---------------------------

# omikron_0
plt.subplot(241)
ax_00 = plt.gca()
swarm_boxplot(ax_00,  model_exp1, 'omikron_0', 'Parameter Estimate', 1)
ax_00.yaxis.set_label_coords(ylabel_dist, 0.5)
ax_00.set_title('Motor Noise')

# omikron_1
plt.subplot(242)
ax_01 = plt.gca()
swarm_boxplot(ax_01,  model_exp1, 'omikron_1', ' ', 1)
ax_01.set_title('Learning-Rate Noise')

# omikron_0
plt.subplot(243)
ax_00 = plt.gca()
swarm_boxplot(ax_00,  model_exp1, 'b_0', ' ', 1)
ax_00.set_title('Intercept')

# omikron_1
plt.subplot(244)
ax_01 = plt.gca()
swarm_boxplot(ax_01,  model_exp1, 'b_1', ' ', 1)
ax_01.set_title('Slope')
ax_01.set_ylim([-1.5, 0.2])

# q
plt.subplot(245)
ax_10 = plt.gca()
swarm_boxplot(ax_10,  model_exp1, 'q', 'Parameter Estimate', 1)
ax_00.yaxis.set_label_coords(ylabel_dist, 0.5)
ax_10.set_title('Reward Bias')

# sigma_H
plt.subplot(246)
ax_11 = plt.gca()
swarm_boxplot(ax_11,  model_exp1, 'sigma_H', ' ', 1)
ax_11.set_title('Catch Trial')

# -------------------------------------
# 4. Add subplot labels and save figure
# -------------------------------------

# Adjust space and axes
plt.subplots_adjust(wspace=0.5, hspace=0.7, top=0.9, bottom=0.125, left=0.1, right=0.95)
sns.despine()

texts = ['a', 'b', 'c', 'd', 'e', 'f']
label_subplots(f, texts, x_offset=0.075, y_offset=0.0)

savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_S_figure_6.pdf"
plt.savefig(savename, transparent=True, dpi=400)

# Show figure
plt.show()
