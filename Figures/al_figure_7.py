""" Figure 7: Summary of third experiment

    1. Load data
    2. Parametric statistics for Figure 7
    3. Prepare figure
    4. Plot experiment and main results
    5. Add subplot labels and save figure
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
from al_plot_utils import latex_plt, cm2inch, label_subplots, plot_image
from al_utilities import safe_save_dataframe
from al_compute_effects_exp3 import compute_effects_exp3, ttests_main_paper


# Update matplotlib to use Latex and to change some defaults
matplotlib = latex_plt(matplotlib)
matplotlib.use('Qt5Agg')

# Get home directory
paths = os.getcwd()
path = paths.split(os.path.sep)
home_dir = path[1]

# ------------
# 1. Load data
# ------------

df_exp3 = pd.read_pickle('al_data/data_prepr_3.pkl')
n_subj = len(np.unique(df_exp3['subj_num']))

# Rename original conditions for compatibility with second experiment
df_exp3 = df_exp3.rename(columns={"cond": "all_cond"})

pers_diff_bin_1_thres, anchoring_diff, _, _, _ = compute_effects_exp3(df_exp3)

# -------------------------------------
# 2. Parametric statistics for Figure 7
# -------------------------------------

#  Perseveration statistics
# -------------------------

# Difference in perseveration between high and low reward in bin 1 from above
pers_diff = pd.DataFrame(pers_diff_bin_1_thres[:])

# Compute perseveration statistics
exp3_pers_param_desc, exp3_pers_param_stat = ttests_main_paper(pers_diff)

# Rename files
exp3_pers_param_desc.name, exp3_pers_param_stat.name = "exp3_pers_param_desc", "exp3_pers_param_stat"

# Save statistics for Latex manuscript
safe_save_dataframe(exp3_pers_param_desc, 'age_group')
safe_save_dataframe(exp3_pers_param_stat, 'age_group')

# Anchoring statistics
# --------------------

# Difference in perseveration between high and low reward in bin 1 from above
anchoring_diff = pd.DataFrame(anchoring_diff)

# Compute anchoring statistics
exp3_anchoring_param_desc, exp3_anchoring_param_stat = ttests_main_paper(anchoring_diff)

# Rename files
exp3_anchoring_param_desc.name, exp3_anchoring_param_stat.name = "exp3_anchoring_param_desc", "exp3_anchoring_param_stat"

# Save statistics for Latex manuscript
safe_save_dataframe(exp3_anchoring_param_desc, 'age_group')
safe_save_dataframe(exp3_anchoring_param_stat, 'age_group')

# Prepare data frame for difference figure
# ----------------------------------------

# Rename the columns
df1_renamed = pers_diff.rename(columns={'pers': 'value'})
df2_renamed = anchoring_diff.rename(columns={'bucket_bias': 'value'})

# Add a new column with the original column names
df1_renamed['voi'] = 'Perseveration'
df2_renamed['voi'] = 'Anchoring'

# Concatenate the data frames
effects = pd.concat([df1_renamed, df2_renamed], ignore_index=True)

# -----------------
# 3. Prepare figure
# -----------------

# Size of figure
fig_height = 5
fig_width = 15

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))

# Create plot grid
gs_0 = gridspec.GridSpec(1, 3, top=0.9, bottom=0.2, left=0.025, right=0.95, hspace=0.5, wspace=0.5)

# Create subplot
gs_00 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_0[:, :2], wspace=0.0)
ax_0 = plt.Subplot(f, gs_00[0, 0])
f.add_subplot(ax_0)

# -----------------------------------
# 4. Plot experiment and main results
# -----------------------------------

# Picture path
path = 'al_figures/onlineHeliGold.png'  # Figure text and font size
text = 'High-Reward Condition'
fontsize = 8

# Initialize image coordinates
cell_x0 = 0.0
cell_x1 = 1
image_y = 0.4

# Initialize text coordinates
text_y_dist = 0.05
text_pos = 'left_top'

# Plot images and text
plot_image(f, path, cell_x0, cell_x1, image_y, ax_0, text_y_dist, text, text_pos, fontsize, zoom=0.0325)

# Delete unnecessary axes
ax_0.axis('off')

# Next subplot
ax_1 = plt.Subplot(f, gs_00[0, 1])
f.add_subplot(ax_1)

# Picture path
path = 'al_figures/onlineHeliSilver.png'  # Figure text and font size
text = 'Low-Reward Condition'
fontsize = 8

# Plot images and text
plot_image(f, path, cell_x0, cell_x1, image_y, ax_1, text_y_dist, text, text_pos, fontsize, zoom=0.0325)

# Delete unnecessary axes
ax_1.axis('off')

# Plot difference with 95% confidence interval
gs_00 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_0[:, 2], wspace=0.0)
ax_1 = plt.Subplot(f, gs_00[0, 0])
f.add_subplot(ax_1)
sns.pointplot(data=effects, x="voi", y="value", errorbar=("ci", 95), color=".5", linestyle="none")

# Add zero line
plt.axhline(color='k', linestyle='--', linewidth=1)

# Text
ax_1.set_xticks([0,1], ['Perseveration\nProbability', 'Anchoring\nBias'], rotation=0)
ax_1.set_ylabel('Difference High - Low Reward')
ax_1.set_xlabel('')

# -------------------------------------
# 5. Add subplot labels and save figure
# -------------------------------------

# Delete unnecessary axes
sns.despine()

# Add labels
texts = ['a', np.nan, 'b']  # label letters
x_offset = [0.01, np.nan, 0.12]
label_subplots(f, texts, x_offset=x_offset, y_offset=0.025)

savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_figure_7.pdf"
plt.savefig(savename, transparent=False, dpi=400)

# Show figure
plt.ioff()
plt.show()
