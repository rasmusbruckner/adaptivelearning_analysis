""" Figure S8: Uncertainty-underestimation simulation

1. Load data
2. Prepare figure
3. Plot uncertainty underestimation
4. Save figure
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from al_plot_utils import swarm_boxplot, latex_plt


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

model_results = pd.read_pickle('al_data/estimates_unc_underest.pkl')

# -----------------
# 2. Prepare figure
# -----------------

# Plot colors
colors = ["#BBE1FA", "#3282B8", "#0F4C75", "#1B262C"]
sns.set_palette(sns.color_palette(colors))

# Size of figure
fig_height = 6.4/2
fig_witdh = 4.8/2

# Create figure
f = plt.figure(figsize=(fig_height, fig_witdh))

# Y-label distance
ylabel_dist = -0.325

# Create figure axis
ax = plt.gca()

# -----------------------------------
# 3. Plot uncertainty underestimation
# -----------------------------------

swarm_boxplot(ax,  model_results, 'u', '', 1)
ax.set_ylabel('Uncertainty Underestimation')

# --------------
# 4. Save figure
# --------------

# Adjust size and axes
sns.despine()
f.tight_layout()

# Save figure
savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_S_figure_8.pdf"
plt.savefig(savename, transparent=True, dpi=400)

# Show figure
plt.show()
