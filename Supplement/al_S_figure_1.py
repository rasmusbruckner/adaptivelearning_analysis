""" Figure S1: Identical Picture & Spot-a-Word results

    1. Load data
    2. Run statistical tests
    3. Prepare figure
    4. Plot IP-SAW for both experiments
    5. Add subplot labels and save figure
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

# Load data
df_exp1 = pd.read_pickle('al_data/data_prepr_1.pkl')
df_exp2 = pd.read_pickle('al_data/data_prepr_2.pkl')

# Extract variables of interest
ip_1 = df_exp1.groupby(['subj_num', 'age_group'])['ip'].first()
ip_1 = ip_1.reset_index(drop=False)
saw_1 = df_exp1.groupby(['subj_num', 'age_group'])['saw'].first()
saw_1 = saw_1.reset_index(drop=False)
ip_2 = df_exp2.groupby(['subj_num', 'age_group'])['ip'].first()
ip_2 = ip_2.reset_index(drop=False)
saw_2 = df_exp2.groupby(['subj_num', 'age_group'])['saw'].first()
saw_2 = saw_2.reset_index(drop=False)

# ------------------------
# 2. Run statistical tests
# ------------------------

# First experiment IP
# -------------------

# Print out identical pictures statistics for paper
print('\n\nIdentical Picture Experiment 1\n')
exp1_ip_desc, exp1_ip_stat, exp1_ip_effect_size = get_stats(ip_1, 1, 'ip')
exp1_ip_desc.name, exp1_ip_stat.name, exp1_ip_effect_size.name = "exp1_ip_desc", "exp1_ip_stat", "exp1_ip_effect_size"

# Save statistics for Latex manuscript
safe_save_dataframe(exp1_ip_desc, 'age_group')
safe_save_dataframe(exp1_ip_stat, 'test')
safe_save_dataframe(exp1_ip_effect_size, 'type')

# First experiment SAW
# --------------------

# Print out spot-a-word statistics for paper
print('\n\nSpot-a-Word Experiment 1\n')
exp1_saw_desc, exp1_saw_stat, exp1_saw_effect_size = get_stats(saw_1, 1, 'saw')
exp1_saw_desc.name, exp1_saw_stat.name, exp1_saw_effect_size.name = "exp1_saw_desc", "exp1_saw_stat", "exp1_saw_effect_size"

# Save statistics for Latex manuscript
safe_save_dataframe(exp1_saw_desc, 'age_group')
safe_save_dataframe(exp1_saw_stat, 'test')
safe_save_dataframe(exp1_saw_effect_size, 'type')

# Second experiment: IP
# ---------------------

# Print out identical pictures statistics for paper
print('\n\nIdentical Picture Experiment 2\n')
exp2_ip_desc, exp2_ip_stat, exp2_ip_effect_size = get_stats(ip_2, 2, 'ip')
exp2_ip_desc.name, exp2_ip_stat.name, exp2_ip_effect_size.name = "exp2_ip_desc", "exp2_ip_stat", "exp2_ip_effect_size"

# Save statistics for Latex manuscript
safe_save_dataframe(exp2_ip_desc, 'age_group')
safe_save_dataframe(exp2_ip_stat, 'test')
safe_save_dataframe(exp2_ip_effect_size, 'type')

# Second experiment: SAW
# ----------------------

# Print out spot-a-word statistics for paper
print('\n\nSpot-a-Word Experiment 2\n')
exp2_saw_desc, exp2_saw_stat, exp2_saw_effect_size = get_stats(saw_2, 2, 'saw')
exp2_saw_desc.name, exp2_saw_stat.name, exp2_saw_effect_size.name = "exp2_saw_desc", "exp2_saw_stat", "exp2_saw_effect_size"

# Save statistics for Latex manuscript
safe_save_dataframe(exp2_saw_desc, 'age_group')
safe_save_dataframe(exp2_saw_stat, 'test')
safe_save_dataframe(exp2_saw_effect_size, 'type')

# -----------------
# 3. Prepare figure
# -----------------

# Size of figure
fig_height = 5
fig_width = 15

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))

# Create plot grid
gs_0 = gridspec.GridSpec(1, 1, wspace=1, hspace=0.7, top=0.8, bottom=0.2, left=0.1, right=0.95)

# Plot colors
colors = ["#BBE1FA", "#3282B8", "#0F4C75", "#1B262C"]
sns.set_palette(sns.color_palette(colors))

# -----------------------------------
# 4. Plot IP-SAW for both experiments
# -----------------------------------

# Create subplot grid
gs_00 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_0[0], hspace=0.6, wspace=0.5)

# Plot IP1
exp = 1
ax_01 = plt.Subplot(f, gs_00[0, 0])
f.add_subplot(ax_01)
ax_01 = swarm_boxplot(ax_01, ip_1, 'ip', 'Identical Picture', exp)
ax_01.set_ylim([9, 45])
ax_01.set_title('First Experiment')

# Plot SAW1
exp = 1
ax_02 = plt.Subplot(f, gs_00[0, 1])
f.add_subplot(ax_02)
ax_02 = swarm_boxplot(ax_02, saw_1, 'saw', 'Spot-a-Word', exp)
ax_02.set_ylim([-5, 40])
ax_02.set_title('First Experiment')

# Update plot colors
colors = ["#BBE1FA", "#0F4C75", "#1B262C"]
sns.set_palette(sns.color_palette(colors))

# Plot IP2
exp = 2
ax_03 = plt.Subplot(f, gs_00[0, 2])
f.add_subplot(ax_03)
ax_03 = swarm_boxplot(ax_03, ip_2, 'ip', 'Identical Picture', exp)
ax_03.set_ylim([9, 45])
ax_03.set_title('Second Experiment')

# Plot SAW2
exp = 2
ax_04 = plt.Subplot(f, gs_00[0, 3])
f.add_subplot(ax_04)
ax_04 = swarm_boxplot(ax_04, saw_2, 'saw', 'Spot-a-Word', exp)
ax_04.set_ylim([-5, 40])
ax_04.set_title('Second Experiment')

# -------------------------------------
# 5. Add subplot labels and save figure
# -------------------------------------

# Delete unnecessary axes
sns.despine()

# Add labels
texts = ['a', 'b', 'c', 'd']  # label letters
label_subplots(f, texts, x_offset=0.07, y_offset=0.025)

# Save figure for manuscript
savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_S_figure_1.pdf"
plt.savefig(savename, transparent=True, dpi=400)

# Show plot
plt.show()
