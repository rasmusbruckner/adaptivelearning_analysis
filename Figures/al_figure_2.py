""" Figure 2: Task and model experiment 1

    1. Load data
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
from al_simulation_rbm import simulation
from al_plot_utils import latex_plt, cm2inch, label_subplots, plot_image


# Update matplotlib to use Latex and to change some defaults
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
df_exp1['v_t'] = 0  # no catch trials for illustration
n_subj = len(np.unique(df_exp1['subj_num']))

# -----------------
# 2. Prepare figure
# -----------------

# Size of figure
fig_height = 10
fig_width = 8

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))
f.canvas.draw()

# Image format
saveas = "pdf"

# Create plot grid
gs_0 = gridspec.GridSpec(3, 1, wspace=0.5, hspace=0.4, top=0.95, bottom=0.085, left=0.18, right=0.95)

# Y-label distance
ylabel_dist = -0.15

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
text = ['Prediction', 'Outcome\n(1.4s)', 'Prediction\nError', 'Update (max. 6s)']
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
model_params.loc[0, 'sigma_H'] = 0.01
model_params.loc[0, 'd'] = 0.0
model_params.loc[0, 'subj_num'] = 1.0
model_params.loc[0, 'age_group'] = 0

# Normative model simulation
sim_pers = False  # no perseveration simulation
_, _, df_data, _, = simulation(df_exp1, model_params, n_sim, sim_pers)

# Indicate plot range and x-axis
plot_range = (200, 225)
x = np.linspace(0, plot_range[1]-plot_range[0]-1, plot_range[1]-plot_range[0])

# Mean, outcomes, and predictions
ax_10 = plt.Subplot(f, gs_01[0:2, 0])
f.add_subplot(ax_10)
ax_10.plot(x, np.array(df_exp1['mu_t'][plot_range[0]:plot_range[1]]), '--',
           x, np.array(df_exp1['x_t'][plot_range[0]:plot_range[1]]), '.', color="k")
ax_10.plot(x, np.array(df_data['sim_b_t'][plot_range[0]:plot_range[1]]), linewidth=2, color="k", alpha=1)
ax_10.set_ylabel('Position')
ax_10.yaxis.set_label_coords(ylabel_dist, 0.5)
ax_10.legend(["Helicopter", "Outcome", "Model"], loc=1, framealpha=0.8)
ax_10.set_ylim(-9, 309)

# Remove tick parameters
ax_10.tick_params(
    axis='x',  # changes apply to the x-axis
    labelbottom=False)  # labels along the bottom edge are off

# Prediction errors
ax_11 = plt.Subplot(f, gs_01[2, 0])
f.add_subplot(ax_11)
ax_11.plot(x, np.array(df_data['delta_t'][plot_range[0]:plot_range[1]]), linewidth=2, color="k", alpha=1)
ax_11.set_ylabel('Prediction Error')
ax_11.yaxis.set_label_coords(ylabel_dist, 0.5)

# Remove tick parameters
ax_11.tick_params(
    axis='x',  # changes apply to the x-axis
    labelbottom=False)  # labels along the bottom edge are off

# Relative uncertainty, changepoint probability, and learning rate
ax_12 = plt.Subplot(f, gs_01[3, 0])
f.add_subplot(ax_12)
ax_12.plot(x, np.array(df_data['tau_t'][plot_range[0]:plot_range[1]]), linewidth=2, color="#3282B8", alpha=1)
ax_12.plot(x, np.array(df_data['omega_t'][plot_range[0]:plot_range[1]]), linewidth=2, color="#0F4C75", alpha=1)
ax_12.plot(x, np.array(df_data['alpha_t'][plot_range[0]:plot_range[1]]), linewidth=2, color="k", alpha=1)
ax_12.legend(['RU', 'CPP', 'LR'], loc=1)
ax_12.set_xlabel('Trial')
ax_12.set_ylabel('Variable')
ax_12.yaxis.set_label_coords(ylabel_dist, 0.5)

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
# -----------

if saveas == "pdf":
    savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_figure_2.pdf"
else:
    savename = "/" + home_dir + "/rasmus/Dropbox/heli_lifespan/png/al_figure_2.png"

plt.savefig(savename, transparent=False, dpi=400)

# Show plot
plt.show()
