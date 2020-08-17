""" Figure 1

    1. Load data and compute performance
    2. Prepare figure
    3. Plot task trial schematic
    4. Plot block example and model computations
    5. Add subplot labels and save figure
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
from al_simulation import simulation
from al_utilities import get_mean_voi, get_stats
from al_plot_utils import latex_plt, plot_image, cm2inch, label_subplots, swarm_boxplot


# Update matplotlib to use Latex and to change some defaults
os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2016/bin/x86_64-darwin'
matplotlib = latex_plt(matplotlib)

# Get home directory
paths = os.getcwd()
path = paths.split(os.path.sep)
home_dir = path[1]

# ------------------------------------
# 1. Load data and compute performance
# ------------------------------------

# Load data
df_exp1 = pd.read_pickle('al_data/data_prepr_1.pkl')
n_subj = len(np.unique(df_exp1['subj_num']))

# Compute estimation errors
voi = 1
e_t = get_mean_voi(df_exp1, voi)

# -----------------
# 2. Prepare figure
# -----------------

# Size of figure
fig_height = 10
fig_width = 8

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))
f.canvas.draw()
# f.canvas.tostring_argb()

# Create plot grid
gs_0 = gridspec.GridSpec(3, 1, wspace=0.5, hspace=0.7, top=0.95, bottom=0.085, left=0.18, right=0.95)

# Plot colors
colors = ["#92e0a9", "#69b0c1", "#6d6192", "#352d4d"]
sns.set_palette(sns.color_palette(colors))

# ----------------------------
# 3. Plot task trial schematic
# ----------------------------

# Create subplot grid and axis
gs_00 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_0[0])
ax_0 = plt.Subplot(f, gs_00[0, 0])
f.add_subplot(ax_0)

# Picture paths
path = ['al_figures/prediction.png', 'al_figures/outcome.png',
        'al_figures/prediction_error.png', 'al_figures/update.png']

# Figure text and font size
text = ['Prediction', 'Outcome\n(1.4s)', 'Prediction\nerror', 'Update (max. 6s)']
fontsize = 6

# Initialize image coordinates
cell_x0 = 0.0
cell_x1 = 0.2
image_y = 0.8

# Initialize text coordinates
text_y_dist = [0.1, 0.22, 0.22, 0.1]
text_pos = 'left_below'

# Cycle over images
for i in range(0, 4):

    # Plot images and text
    plot_image(f, path[i], cell_x0, cell_x1, image_y, ax_0, text_y_dist[i], text[i], text_pos, fontsize, zoom=0.05)

    # Update coordinates
    cell_x0 += 0.25
    cell_x1 += 0.25
    image_y += -0.2

# Delete unnecessary axes
ax_0.axis('off')

# --------------------------------------------
# 4. Plot block example and model computations
# --------------------------------------------

# Create subplot grid
gs_01 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs_0[1:3], hspace=0.5)

# Simulation parameters
n_sim = 1
model_params = pd.DataFrame(columns=['omikron_0', 'omikron_1', 'b_0', 'b_1', 'h', 's', 'u', 'q', 'sigma_H', 'd',
                                     'subj_num', 'age_group'])
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
model_params.loc[0, 'age_group'] = 0

# Normative model simulation
sim_pers = False  # no perseveration simulation
_, _, df_data, _, = simulation(df_exp1, model_params, n_sim, sim_pers)

# Indicate plot range and x-axis
plot_range = (200, 225)
x = np.linspace(0, plot_range[1]-plot_range[0]-1, plot_range[1]-plot_range[0])

# Mean, outcomes and predictions
ax_10 = plt.Subplot(f, gs_01[0:2, 0])
f.add_subplot(ax_10)
ax_10.plot(x, df_exp1['mu_t'][plot_range[0]:plot_range[1]], '--',
           x, df_exp1['x_t'][plot_range[0]:plot_range[1]], '.', color="#090030")
ax_10.plot(x, df_data['sim_b_t'][plot_range[0]:plot_range[1]], linewidth=2, color="#f30a49", alpha=0.8)
ax_10.set_ylabel('Screen unit')
ax_10.legend(["Helicopter", "Outcome", "Model"], loc=1, framealpha=0.8)
ax_10.set_ylim(0, 309)
ax_10.set_xticklabels([''])

# Prediction errors
ax_11 = plt.Subplot(f, gs_01[2, 0])
f.add_subplot(ax_11)
ax_11.plot(x, df_data['delta_t'][plot_range[0]:plot_range[1]], linewidth=2, color="#090030", alpha=1)
ax_11.set_xticklabels([''])
ax_11.set_ylabel('Pred. error')

# Relative uncertainty, changepoint probability and learning rate
ax_12 = plt.Subplot(f, gs_01[3, 0])
f.add_subplot(ax_12)
ax_12.plot(x, df_data['tau_t'][plot_range[0]:plot_range[1]], linewidth=2, color="#04879c", alpha=1)
ax_12.plot(x, df_data['omega_t'][plot_range[0]:plot_range[1]], linewidth=2, color="#0c3c78", alpha=1)
ax_12.plot(x, df_data['alpha_t'][plot_range[0]:plot_range[1]], linewidth=2, color="#f30a49", alpha=0.8)
ax_12.legend(['RU', 'CPP', 'LR'], loc=1)
ax_12.set_xlabel('Trial')
ax_12.set_ylabel('Variable')

# Delete unnecessary axes
sns.despine()

# -------------------------------------
# 5. Add subplot labels and save figure
# -------------------------------------

# Label letters
texts = ['a', 'b', ' ', ' ']

# Add labels
label_subplots(f, texts, x_offset=0.15, y_offset=0.0)

# Save figure
savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_figure_1.pdf"
plt.savefig(savename, transparent=True, dpi=400)

# Show plot
plt.show()
