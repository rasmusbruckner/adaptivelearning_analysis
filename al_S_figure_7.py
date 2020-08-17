""" Figure S7

1. Load data
2. Prepare figure
3. Plot parameter estimates
4. Add subplot labels and save figure
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
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

model_exp2 = pd.read_pickle('al_data/estimates_follow_up_exp_25_sp.pkl')

# -----------------
# 2. Prepare figure
# -----------------

# Plot colors
colors = ["#92e0a9",  "#6d6192", "#352d4d"]
sns.set_palette(sns.color_palette(colors))

# Create figure
fig_height = 13.5
fig_witdh = 15
f = plt.figure(figsize=cm2inch(fig_witdh, fig_height))

# ----------------------------
# 3. Plot parameter estimates
# ----------------------------

# omikron_0
plt.subplot(331)
ax_00 = plt.gca()
swarm_boxplot(ax_00,  model_exp2, 'omikron_0', 'Parameter estimate', 2)
ax_00.set_title('Motor noise')

# omikron_1
plt.subplot(332)
ax_01 = plt.gca()
swarm_boxplot(ax_01,  model_exp2, 'omikron_1', 'Parameter estimate', 2)
ax_01.set_title('Learning-rate noise')

# b_0
plt.subplot(333)
ax_02 = plt.gca()
swarm_boxplot(ax_02,  model_exp2, 'b_0', 'Parameter estimate', 2)
ax_02.set_title('Intercept')

# b_1
plt.subplot(334)
ax_10 = plt.gca()
swarm_boxplot(ax_10,  model_exp2, 'b_1', 'Parameter estimate', 2)
ax_10.set_title('Slope')

# u
plt.subplot(335)
ax_11 = plt.gca()
swarm_boxplot(ax_11,  model_exp2, 'u', 'Parameter estimate', 2)
ax_11.set_title('Uncertainty underestimation')

# s
plt.subplot(336)
ax_20 = plt.gca()
swarm_boxplot(ax_20,  model_exp2, 's', 'Parameter estimate', 2)
ax_20.set_title('Surprise sensitivity')

# h
plt.subplot(337)
ax_12 = plt.gca()
swarm_boxplot(ax_12,  model_exp2, 'h', 'Parameter estimate', 2)
ax_12.set_title('Hazard rate')

# sigma_H
plt.subplot(338)
ax_21 = plt.gca()
swarm_boxplot(ax_21,  model_exp2, 'sigma_H', 'Parameter estimate', 2)
ax_21.set_title('Catch trial')

# -------------------------------------
# 4. Add subplot labels and save figure
# -------------------------------------

# Adjust space and axes
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.6)
sns.despine()

texts = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']  # label letters
label_subplots(f, texts, x_offset=0.08, y_offset=0.0)

savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_S_figure_7.pdf"
plt.savefig(savename, transparent=True, dpi=400)

# Show figure
plt.show()
