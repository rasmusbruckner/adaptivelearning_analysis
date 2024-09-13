""" Figure S23: Third experiment in supplementary materials

    1. Load data
    2. Prepare figure
    3. Plot perseveration for standard and anchoring condition
    4. Plot estimation error for standard and anchoring condition
    5. Plot anchoring bias
    6. Plot perseveration differences between high- and low-reward condition
    7. Plot estimation-error differences between high- and low-reward condition
    8. Add subplot labels and save figure
 """

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
from al_plot_utils import latex_plt, cm2inch, label_subplots
from al_compute_effects_exp3 import compute_effects_exp3


# Set random number generator for reproducible results
np.random.seed(123)

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

pers_diff_bin_1_thres, anchoring_diff, pers_prob, est_err, df_reg = compute_effects_exp3(df_exp3)

# -----------------
# 2. Prepare figure
# -----------------

# Size of figure
fig_height = 10
fig_width = 15

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))

# Create plot grid
gs_0 = gridspec.GridSpec(2, 1, top=0.9, bottom=0.1, left=0.1, right=0.95, hspace=0.5)

# Colors
colors = ["#0F4C75"]
sns.set_palette(sns.color_palette(colors))

# Line width
linewidth = 0.1

# ----------------------------------------------------------
# 3. Plot perseveration for standard and anchoring condition
# -----------------------------------------------------------

# Create subplot
gs_00 = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs_0[:, :], hspace=0.4, wspace=0.6)
ax_0 = plt.Subplot(f, gs_00[0, 0])
f.add_subplot(ax_0)

# Perseveration boxplot
sns.boxplot(x="cond", y="pers", data=pers_prob,
            notch=False, showfliers=False, linewidth=0.8, width=0.3,
            ax=ax_0, showcaps=False)

# Add single-subject lines with jitter
x = [0, 1]
y = [pers_prob[pers_prob['cond'] == 'main_noPush']['pers'], pers_prob[pers_prob['cond'] == 'main_push']['pers']]
jitter = np.random.uniform(-0.1, 0.1, n_subj)
x[0] += jitter
x[1] += jitter
ax_0.plot(x, y, color='gray', alpha=0.8, zorder=0, linewidth=linewidth)

# Text
ax_0.set_xticks(np.arange(2), ['Standard', 'Anchoring'], rotation=0)
ax_0.set_xlabel('Condition')
ax_0.set_ylabel('Perseveration Probability')

# -------------------------------------------------------------
# 4. Plot estimation error for standard and anchoring condition
# -------------------------------------------------------------

# Create subplot
ax_1 = plt.Subplot(f, gs_00[0, 1])
f.add_subplot(ax_1)

# Estimation-error boxplot
sns.boxplot(x="cond", y="e_t", data=est_err,
            notch=False, showfliers=False, linewidth=0.8, width=0.3,
            ax=ax_1, showcaps=False)

# Add single-subject lines with jitter
y = [est_err[est_err['cond'] == 'main_noPush']['e_t'], est_err[est_err['cond'] == 'main_push']['e_t']]
jitter = np.random.uniform(-0.1, 0.1, n_subj)
x[0] += jitter
x[1] += jitter
ax_1.plot(x, y, color='gray', alpha=0.8, zorder=0, linewidth=linewidth)
plt.ylim([6, 27])

# Text
ax_1.set_xticks(np.arange(2), ['Standard', 'Anchoring'], rotation=0)
ax_1.set_xlabel('Condition')
ax_1.set_ylabel('Estimation Error')

# -----------------------
# 5. Plot anchoring bias
# -----------------------

# Create subplot
ax_2 = plt.Subplot(f, gs_00[0, 2])
f.add_subplot(ax_2)

# Plot seaborn boxplot with dots
dotsize = 2
sns.boxplot(df_reg['bucket_bias'], ax=ax_2, notch=False, showfliers=False, linewidth=0.8, width=0.3,
            legend=False, showcaps=False)
sns.stripplot(df_reg['bucket_bias'], color='gray', alpha=0.7, size=dotsize)

# Text
ax_2.set_xticks([0], ['Anchoring Condition'], rotation=0)
ax_2.set_ylabel('Anchoring Bias')

# -------------------------------------------------------------------------
# 6. Plot perseveration differences between high- and low-reward condition
# -------------------------------------------------------------------------

# Illustrate prediction-error bins
# --------------------------------

# Create subplot
ax_3 = plt.Subplot(f, gs_00[1, 0])
f.add_subplot(ax_3)

# Extract perseveration in each bin
pers_prob_bins = df_exp3.groupby(['subj_num', 'cond', 'pe_bin'])['pers'].mean().reset_index(drop=False)
stable_bin_1 = pers_prob_bins[(pers_prob_bins['cond'] == 'main_noPush') & (pers_prob_bins['pe_bin'] == 1)]['pers']
stable_bin_2 = pers_prob_bins[(pers_prob_bins['cond'] == 'main_noPush') & (pers_prob_bins['pe_bin'] == 2)]['pers']
stable_bin_3 = pers_prob_bins[(pers_prob_bins['cond'] == 'main_noPush') & (pers_prob_bins['pe_bin'] == 3)]['pers']

# Plot perseveration for each bin
ax_3.bar(np.arange(3), [np.median(stable_bin_1), np.median(stable_bin_2), np.median(stable_bin_3)],
         color='gray', alpha=0.8, zorder=0)
ax_3.set_ylabel('Perseveration Probability')
ax_3.set_xlabel('Prediction-Error Bin')
ax_3.set_xticks(np.arange(3))

# Show perseveration differences in bin 1 and only for subjects above 5% threshold
# --------------------------------------------------------------------------------

# Create subplot
ax_3 = plt.Subplot(f, gs_00[1, 1])
f.add_subplot(ax_3)

# Perseveration boxplot with dots
plt.axhline(color='k', linestyle='--', linewidth=1)
sns.boxplot(pers_diff_bin_1_thres[:], ax=ax_3, notch=False, showfliers=False, linewidth=0.8, width=0.3,
            legend=False, showcaps=False)
sns.stripplot(pers_diff_bin_1_thres[:], color='gray', alpha=0.7, size=dotsize)

# Text
ax_3.set_xticks([0], ['Standard Condition'], rotation=0)
ax_3.set_ylabel('Perseveration Difference\nHigh - Low Reward')

# ---------------------------------------------------------------------------
# 7. Plot estimation-error differences between high- and low-reward condition
# ---------------------------------------------------------------------------

# Create subplot
ax_4 = plt.Subplot(f, gs_00[1, 2])
f.add_subplot(ax_4)

# Estimation-error boxplot with dots
plt.axhline(color='k', linestyle='--', linewidth=1)
sns.boxplot(anchoring_diff, ax=ax_4, notch=False, showfliers=False, linewidth=0.8, width=0.3,
            legend=False, showcaps=False)
sns.stripplot(anchoring_diff, color='gray', alpha=0.7, size=dotsize)

# Text
ax_4.set_xticks([0], ['Anchoring Condition'], rotation=0)
ax_4.set_ylabel('Anchoring Difference\nHigh - Low Reward')

# -------------------------------------
# 8. Add subplot labels and save figure
# -------------------------------------

# Delete unnecessary axes
sns.despine()

# Add labels
texts = ['a', 'b', 'c', 'd', 'e', 'f']  # label letters
label_subplots(f, texts, x_offset=0.08, y_offset=0.025)

# Save figure
savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_S_figure_23.pdf"
plt.savefig(savename, transparent=False, dpi=400)

# Show plot
plt.ioff()
plt.show()
