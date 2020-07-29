""" SM Figure 5

1. Load and prepare data
2. Prepare figure
3. Plot comparison of simulations and empirical results
4. Add subplot labels and save figure
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from al_plot_utils import cm2inch, label_subplots, latex_plt, get_cond_diff, custom_boxplot, text_legend


# Update matplotlib to use Latex and to change some defaults
os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2016/bin/x86_64-darwin'
matplotlib = latex_plt(matplotlib)

# Get home directory
paths = os.getcwd()
path = paths.split(os.path.sep)
home_dir = path[1]

# ------------------------
# 1. Load and prepare data
# ------------------------

# Follow-up experiment
df_exp2 = pd.read_pickle('al_data/data_prepr_2.pkl')

# Read out push and noPush trials
df_noPush = df_exp2[df_exp2['cond'] == 'main_noPush']
df_push = df_exp2[df_exp2['cond'] == 'main_push']

# Perseveration difference
voi = 2
fig_5_c_desc, _, fig_5_c_stat, fig_5_c_zero_stat = get_cond_diff(df_noPush, df_push, voi)

# Estimation error difference
voi = 1
fig_5_d_desc, _, fig_5_d_stat, fig_5_d_zero_stat = get_cond_diff(df_noPush, df_push, voi)

# Simulated perseveration of follow-up experiment
all_pers = pd.read_pickle('al_data/postpred_exp2_pers.pkl')

# Simulated estimation errors of follow-up experiment
all_est_errs = pd.read_pickle('al_data/postpred_exp2_est_err.pkl')

# Difference between conditions
all_est_errs['diff'] = all_est_errs['push'] - all_est_errs['noPush']
all_pers['diff'] = all_pers['push'] - all_pers['noPush']

# -----------------
# 2. Prepare figure
# -----------------

# Plot colors
colors = ["#92e0a9",  "#6d6192", "#352d4d"]
sns.set_palette(sns.color_palette(colors))

# Create figure
fig_height = 6
fig_witdh = 15
f = plt.figure(figsize=cm2inch(fig_witdh, fig_height))

# -------------------------------------------------------
# 3. Plot comparison of simulations and empirical results
# -------------------------------------------------------

# perseveration
plt.subplot(121)
ax_0 = plt.gca()
sns.swarmplot(x='age_group', y='diff', data=all_pers, alpha=0.7, size=1.5, ax=ax_0, color='k')
ax_0.set_ylabel('Perseveration-\nprobability difference')
ax_0.set_ylim([-.9, 0.1])
ax_0.set_xlabel('Age group')
ax_0 = custom_boxplot(ax_0, fig_5_c_desc, 'pers', alpha=0.2)
plt.xticks([0, 1, 2], ('CH', 'YA', 'OA'))
text_legend(plt.gca(), "Boxplots: participants | Points: simulations")

# Estimation errors
plt.subplot(122)
ax_1 = plt.gca()
sns.swarmplot(x='age_group', y='diff', data=all_est_errs, alpha=0.7, size=1.5, ax=ax_1, color='k')
plt.ylim([-6, 15])
ax_1.set_ylabel('Estimation-\nerror difference')
plt.xticks(np.arange(3), ['CH', 'YA', 'OA'], rotation=0)
ax_1.set_xlabel('Age group')
ax_1 = custom_boxplot(ax_1, fig_5_d_desc, 'e_t', alpha=0.2)
plt.xticks([0, 1, 2], ('CH', 'YA', 'OA'))

# --------------------------------------
# 4. Add subplot labels and save figure
# --------------------------------------

# Delete unnecessary axes
plt.subplots_adjust(left=None, bottom=0.3, right=None, top=None, wspace=None, hspace=None)
sns.despine()

# Add labels
texts = ['a', 'b']  # label letters
label_subplots(f, texts, x_offset=0.08, y_offset=0.0)

# Save figure for manuscript
savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_SM_figure_5.pdf"
plt.savefig(savename, transparent=True, dpi=400)

# Show figure
plt.show()
