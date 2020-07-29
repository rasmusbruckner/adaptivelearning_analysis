""" SM Figure 2

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

model_exp1 = pd.read_pickle('al_data/estimates_first_exp_25_sp.pkl')

# -----------------
# 2. Prepare figure
# -----------------

# Size of figure
fig_width = 10
fig_height = 9

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))

# Plot colors
colors = ["#92e0a9", "#69b0c1", "#6d6192", "#352d4d"]
sns.set_palette(sns.color_palette(colors))

# ---------------------------
# 3. Plot parameter estimates
# ---------------------------

# omikron_0
plt.subplot(221)
ax_00 = plt.gca()
swarm_boxplot(ax_00,  model_exp1, 'omikron_0', 'Parameter estimate', 1)
ax_00.set_title('Motor noise')

# omikron_1
plt.subplot(222)
ax_01 = plt.gca()
swarm_boxplot(ax_01,  model_exp1, 'omikron_1', 'Parameter estimate', 1)
ax_01.set_title('Learning-rate noise')

# q
plt.subplot(223)
ax_10 = plt.gca()
swarm_boxplot(ax_10,  model_exp1, 'q', 'Parameter estimate', 1)
ax_10.set_title('Reward bias')

# sigma_H
plt.subplot(224)
ax_11 = plt.gca()
swarm_boxplot(ax_11,  model_exp1, 'sigma_H', 'Parameter estimate', 1)
ax_11.set_title('Catch trial')

# -------------------------------------
# 4. Add subplot labels and save figure
# -------------------------------------

# Adjust space and axes
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.6)
sns.despine()

texts = ['a', 'b', 'c', 'd']
label_subplots(f, texts, x_offset=0.08, y_offset=0.0)

savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_SM_figure_2.pdf"
plt.savefig(savename, transparent=True, dpi=400)

# Show figure
plt.show()
