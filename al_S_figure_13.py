""" Figure S13

 1. Load data
 2. Prepare figure
 3. Plot illustration of relevant variables
 4. Plot illustration of a block of trials
 5. Add subplot labels and save figure
"""

import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
from al_plot_utils import cm2inch, label_subplots, latex_plt, plot_arrow
from al_simulation_satisficing import simulation_satisficing


# Update matplotlib to use Latex and to change some defaults
os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2016/bin/x86_64-darwin'
matplotlib = latex_plt(matplotlib)

# Get home directory
paths = os.getcwd()
path = paths.split(os.path.sep)
home_dir = path[1]

# Set random number generator for reproducible results
np.random.seed(123)

# ------------
# 1. Load data
# ------------

# Data of follow-up experiment
df_exp2 = pd.read_pickle('al_data/data_prepr_2.pkl')

# -----------------
# 2. Prepare figure
# -----------------

# Figure size
fig_height = 10
fig_width = 15

# Plot colors
colors = ["#92e0a9", "#69b0c1", "#6d6192", "#352d4d"]
sns.set_palette(sns.color_palette(colors))

# Font size
fs = 4

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))
f.canvas.draw()

# Create plot grid
gs_0 = gridspec.GridSpec(3, 2, hspace=1)

# Create subplot grid
gs_01 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_0[0, :], wspace=0.4)

# ------------------------------------------
# 3. Plot illustration of relevant variables
# ------------------------------------------

# Create subplot axes
ax_00 = plt.Subplot(f, gs_01[0, 0])
f.add_subplot(ax_00)

# Plot initial bucket position
ax_00.plot([0, 0], [0, 1.2], color=colors[1])
ax_00.text(0.2, 1.2, 'Bucket', color=colors[1], fontsize=fs)

# Plot satisficing threshold
ax_00.plot([10, 10], [0, 1.2], color=colors[3])
ax_00.text(10.2, 1.2, 'Threshold', fontsize=fs)

# Plot optimal belief update
ax_00.plot([15, 15], [0, 1.2], color=colors[3])
ax_00.text(15.2, 1.2, 'Optimal\nupdate', fontsize=fs)

# Plot prior belief
ax_00.plot([5, 5], [0, 1.2], color=colors[3])
ax_00.text(5.2, 1.2, '$b_t$', fontsize=fs)

# Plot y_t arrow
plot_arrow(ax_00, 5, 0.1, 0, 0.1, color=colors[3])
ax_00.text(0.6, 0.15, '$y_t$', fontsize=fs, color=colors[3])

# Plot optimal a_t arrow
plot_arrow(ax_00, 5, 0.3, 15, 0.3, color=colors[3])
ax_00.text(13.5, 0.35, '$a^*_t$', fontsize=fs, color=colors[3])

# Plot m_t arrow
plot_arrow(ax_00, 0, 0.5, 15, 0.5, color=colors[3])
ax_00.text(13.5, 0.55, '$m_t$', fontsize=fs, color=colors[3])

# Plot actual movement arrow
plot_arrow(ax_00, 0, 0.7, 10, 0.7, color=colors[1])
ax_00.text(5.25, 0.75, 'actual movement', fontsize=fs, color=colors[1])

# Plot a_t arrow
plot_arrow(ax_00, 5, 0.9, 10, 0.9, color="#f30a49")
ax_00.text(8.5, 0.95, '$a_t$', fontsize=fs, color="#f30a49")

# ----------------------------------------------
# 3. Plot illustration of percent-point function
# ----------------------------------------------

# Create subplot axes
ax_01 = plt.Subplot(f, gs_01[0, 1])
f.add_subplot(ax_01)

# Mean and standard deviation of the illustrated PPF
mu = 1
st_dev = 0.8

# Create x-axis
range_x = np.linspace(0, 1, 101)
range_x = np.round(range_x, 2)

# Create ppf
ppf = norm.ppf(range_x, loc=mu, scale=st_dev)

# Plot ppf
ax_01.plot(range_x, ppf, 'k')
ax_01.plot([0.3, 0.3], [-1.0, ppf[range_x == 0.3]], color=colors[3], label="High satisficing")
ax_01.plot([0.0, 0.3], [ppf[range_x == 0.3], ppf[range_x == 0.3]], color=colors[3], label="High satisficing")
ax_01.plot([0.5, 0.5], [-1.0, ppf[range_x == 0.5]], color=colors[3], label="High satisficing")
ax_01.plot([0.0, 0.5], [ppf[range_x == 0.5], ppf[range_x == 0.5]], color=colors[3], label="High satisficing")
ax_01.set_title('Percent-point function')
ax_01.set_xlabel('Probability')
ax_01.set_ylabel('Update')
ax_01.set_xlim(0.0, 1)
ax_01.set_ylim(-1, 3)
plt.yticks(np.arange(-1, 3, 1))

# -----------------------------------------
# 4. Plot illustration of a block of trials
# -----------------------------------------

# Simulation parameters
n_sim = 1
model_params = pd.DataFrame(columns=['omikron_0', 'omikron_1', 'b_0', 'b_1', 'h', 's', 'u', 'q', 'sigma_H', 'd',
                                     'subj_num', 'age_group', 'high_satisficing'])
model_params.loc[0, 'omikron_0'] = 0.01
model_params.loc[0, 'omikron_1'] = 0
model_params.loc[0, 'b_0'] = -30
model_params.loc[0, 'b_1'] = -1.5
model_params.loc[0, 'h'] = 0.1
model_params.loc[0, 's'] = 1
model_params.loc[0, 'u'] = 0
model_params.loc[0, 'q'] = 0
model_params.loc[0, 'sigma_H'] = 0
model_params.loc[0, 'd'] = 0.0
model_params.loc[0, 'subj_num'] = 1.0
model_params.loc[0, 'age_group'] = 1
model_params.loc[0, 'high_satisficing'] = 0.1

# Run simulation
_, _, df_data, _, = simulation_satisficing(df_exp2, model_params, n_sim)

# Indicate plot range and x-axis
plot_range = (0, 400)
x = np.linspace(0, plot_range[1]-plot_range[0]-1, plot_range[1]-plot_range[0])

# Create subplot grid
gs_02 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_0[1:3, :], hspace=0.75)

# Plot block example
ax_10 = plt.Subplot(f, gs_02[0, 0])
f.add_subplot(ax_10)
ax_10.plot(x, df_exp2['mu_t'][plot_range[0]:plot_range[1]], '--',
           x, df_exp2['x_t'][plot_range[0]:plot_range[1]], '.', color="#090030")
ax_10.plot(x, df_data['sim_b_t'][plot_range[0]:plot_range[1]], linewidth=2, color="#f30a49", alpha=0.8)
ax_10.plot(x[200:plot_range[1]], df_data['sim_z_t'][200:plot_range[1]], '.', color="#04879c", markersize=5)
ax_10.set_ylabel('Screen unit')
ax_10.set_xlabel('Trial')
ax_10.legend(["Helicopter", "Outcome", "Model", "Bucket"], loc=1, framealpha=0.8)
ax_10.set_ylim(0, 309)
ax_10.set_xlim(150, 250)
ax_10.plot([200, 200], [0, 309], color='k')
ax_10.set_title('Stable- vs. shifting-bucket')

# Plot persveration
pers = df_data['sim_a_t'] == 0
ax_11 = plt.Subplot(f, gs_02[1, 0])
f.add_subplot(ax_11)
ax_11.plot(x, pers[plot_range[0]:plot_range[1]], '.', linewidth=2, color="#090030", alpha=1)
ax_11.set_ylabel('Perseveration')
ax_11.set_xlim(150, 250)
ax_11.set_xlabel('Trial')
ax_11.plot([200, 200], [0, 1], color='k')

# -------------------------------------
# 5. Add subplot labels and save figure
# -------------------------------------

# Adjust axes
sns.despine()

# Add labels
texts = ['a', 'b', 'c', 'd', 'e']
label_subplots(f, texts, x_offset=0.08, y_offset=0.0)

# Save figure
savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_S_figure_13.pdf"
plt.savefig(savename, transparent=True, dpi=400)

plt.show()
