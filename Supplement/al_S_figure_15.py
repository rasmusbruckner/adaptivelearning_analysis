""" Figure S15: Robust linear regression with age group

1. Load data
2. Compute perseveration and anchoring bias
3. Plot results
4. Add subplot labels and save figure
"""

import pandas as pd
import seaborn as sns
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from al_utilities import get_mean_voi
from al_plot_utils import cm2inch, latex_plt, plot_pers_est_err_reg, label_subplots


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

# Data of second experiment
df_exp2 = pd.read_pickle('al_data/data_prepr_2.pkl')

# Model parameter second experiment incl. anchoring bias
model_exp2 = pd.read_pickle('al_data/estimates_follow_up_exp_10_sp.pkl')

# Simulated data based on sampling model
df_reg = pd.read_pickle('al_data/n_samples_different_df_reg.pkl')
all_pers = pd.read_pickle('al_data/n_samples_different_all_pers.pkl')

# -------------------------------------------
# 2. Compute perseveration and anchoring bias
# -------------------------------------------

# Empirical data
# --------------

# Read out pdata of standard condition
df_noPush = df_exp2[df_exp2['cond'] == 'main_noPush']

# Perseveration in standard condition
voi = 2
emp_pers_data = get_mean_voi(df_noPush, voi)

# Simulated data
# --------------

# Perseveration in standard condition
sim_pers_data = all_pers[all_pers['variable'] == 'noPush'].reset_index(drop=True)
sim_pers_data = sim_pers_data.rename(columns={"value": "pers"})

# Simulated anchoring bias
df_reg = df_reg.rename(columns={"bucket_bias": "d"})

# ---------------
# 3. Plot results
# ---------------

# Create figure
fig_height = 6
fig_witdh = 15
f = plt.figure(figsize=cm2inch(fig_witdh, fig_height))

# Create plot grid
gs_0 = gridspec.GridSpec(1, 1, bottom=0.2)

# Create left subplot
gs_00 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_0[0, 0])
ax_0 = plt.Subplot(f, gs_00[0, 0])
f.add_subplot(ax_0)

# Plot empirical data
ax_0 = plot_pers_est_err_reg(emp_pers_data, model_exp2, ax_0)
ax_0.set_title("Empirical Data")

# Create right subplot
ax_1 = plt.Subplot(f, gs_00[0, 1])
f.add_subplot(ax_1)

# Plot simulated data
ax_1 = plot_pers_est_err_reg(sim_pers_data, df_reg, ax_1)
ax_1.set_title("Simulated Data")

# Adjust axes in line with left plot
ax_1.set_ylim([ax_0.viewLim.y0, ax_0.viewLim.y1])
ax_1.set_xlim([ax_0.viewLim.x0, ax_0.viewLim.x1])

# -------------------------------------
# 4. Add subplot labels and save figure
# -------------------------------------

# Deleted unnecessary axes
sns.despine()

# Add labels
texts = ['a', 'b']  # label letters
label_subplots(f, texts, x_offset=0.08, y_offset=0.0)

# Save figure
savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_S_figure_15.pdf"
plt.savefig(savename, transparent=True, dpi=400)

# Show plot
plt.show()
