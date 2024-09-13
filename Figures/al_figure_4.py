""" Figure 4: Task and model experiment 2

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
from al_plot_utils import latex_plt, plot_image, cm2inch, label_subplots


# Update matplotlib to use Latex and to change some defaults
matplotlib = latex_plt(matplotlib)

# Get home directory
paths = os.getcwd()
path = paths.split(os.path.sep)
home_dir = path[1]

# ------------
# 1. Load data
# ------------

# Data of follow-up experiment
df_exp2 = pd.read_pickle('al_data/data_prepr_2.pkl')
df_exp2['v_t'] = 0  # No catch trials for illustration
n_subj = len(np.unique(df_exp2['subj_num']))

# Follow-up experiment
model_exp2 = pd.read_pickle('al_data/estimates_follow_up_exp_10_sp.pkl')

# -----------------
# 2. Prepare figure
# -----------------

# Size of figure
fig_height = 10
fig_width = 8

# Image format
saveas = "pdf"

# Turn on interactive mode
plt.ion()

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))
f.canvas.draw()

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
path = ['al_figures/prediction_exp_2.png', 'al_figures/outcome_exp_2.png',
        'al_figures/prediction_error_exp_2.png', 'al_figures/update_exp_2.png']

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
bbox = np.nan

# Cycle over images
for i in range(0, 4):
    # Plot images and text
    _, bbox, _ = plot_image(f, path[i], cell_x0, cell_x1, image_y, ax_0, text_y_dist[i], text[i], text_pos, fontsize,
                            zoom=0.05)

    # Update coordinates
    cell_x0 += 0.25
    cell_x1 += 0.25
    image_y += -0.2

# Delete unnecessary axes
ax_0.axis('off')

# Add bucket illustration
y = -0.3
ax_0.text(bbox.x0, y, "Arrow: Anchor", color='#04879c', clip_on=False)

# Anchor example arrow
shrinkA = 1
shrinkB = 1
y1 = 0.15
y2 = 0.0
x = 0.9
ax_0.annotate("", xy=(x, y1), xycoords='data', xytext=(x, y2), textcoords='data',
              arrowprops=dict(arrowstyle="<-", color="#04879c", shrinkA=shrinkA, shrinkB=shrinkB,
                              patchA=None, patchB=None, connectionstyle="arc3,rad=0"), annotation_clip=False)

# Condition description
x = -0.07
y = -0.225
ax_0.text(x, y, "Block 1 and 3:\nAnchoring condition", color='k', clip_on=False)
y = -0.475
ax_0.text(x, y, "Block 2 and 4:\nStandard condition", color='gray', clip_on=False)

# --------------------------------------------
# 4. Plot block example and model computations
# --------------------------------------------

# Create subplot grid
gs_01 = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=gs_0[1:3], hspace=0.5)

N = 1
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
model_params.loc[0, 'd'] = 0.5
model_params.loc[0, 'subj_num'] = 1.0
model_params.loc[0, 'age_group'] = 0
sim_pers = False

# Normative model
_, _, df_data, _ = simulation(df_exp2, model_params, N, sim_pers, which_exp=2)

# Indicate plot range, x-axis and add subplot
plot_range = (200, 225)
x = np.linspace(0, plot_range[1] - plot_range[0] - 1, plot_range[1] - plot_range[0])

# Mean, outcomes, and predictions
ax_1 = plt.Subplot(f, gs_01[1:3, 0])
f.add_subplot(ax_1)

# Mean, outcomes and predictions
ax_1.plot(x, np.array(df_exp2['mu_t'][plot_range[0]:plot_range[1]]), '--', color="k")
ax_1.plot(x, np.array(df_exp2['x_t'][plot_range[0]:plot_range[1]]), '.', color="k")
ax_1.plot(x, np.array(df_data['sim_z_t'][plot_range[0]:plot_range[1]]), '.', color="#04879c")
ax_1.plot(x, np.array(df_data['sim_b_t'][plot_range[0]:plot_range[1]]), linewidth=2, color="k", alpha=1)
ax_1.set_ylim(-9, 309)
ax_1.set_ylabel('Screen Unit')
ax_1.yaxis.set_label_coords(ylabel_dist, 0.5)
ax_1.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
ax_1.legend(["Helicopter", "Outcome", "Bucket (Anchor)", "Model"], framealpha=0.8, loc='lower left',
            bbox_to_anchor=(0.55, 0.7))

# Bucket shift
ax_2 = plt.Subplot(f, gs_01[3:, 0])
f.add_subplot(ax_2)
ax_2.axhline(y=0, linestyle = '--', color='gray')
ax_2.plot(x, np.array(df_data['sim_y_t'][plot_range[0]:plot_range[1]]), linewidth=2, color="k")
ax_2.set_ylabel('Bucket Shift')
ax_2.yaxis.set_label_coords(ylabel_dist, 0.5)
ax_2.set_xlabel('Trial')
ax_2.set_ylim(-150, 150)

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
    savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_figure_4.pdf"
else:
    savename = "/" + home_dir + "/rasmus/Dropbox/heli_lifespan/png/al_figure_4.png"

plt.savefig(savename, transparent=False, dpi=400)

# Show plot
plt.ioff()
plt.show()
