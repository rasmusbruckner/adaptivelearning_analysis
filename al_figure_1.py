""" Figure 1

    1. Load data and compute performance
    2. Compute average learning rates
    3. Run statistical tests
    4. Prepare figure
    5. Plot task trial schematic
    6. Plot block example and model computations
    7. Plot performance and average learning rates
    8. Add subplot labels and save figure
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import statsmodels.api as sm
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

# ---------------------------------
# 2. Compute average learning rates
# ---------------------------------

# Initialize learning rate and age_group variables
alpha = np.full(n_subj, np.nan)
age_group = np.full(n_subj, np.nan)

# Cycle over participants
for i in range(0, n_subj):

    # Extract data of current participant
    df_subj = df_exp1[(df_exp1['subj_num'] == i + 1)].copy()
    x = np.linspace(0, len(df_subj) - 1, len(df_subj))
    df_subj.loc[:, 'trial'] = x.tolist()
    df_subj = df_subj.set_index('trial')

    # Extract prediction error and prediction update and add intercept to data frame
    X = df_subj['delta_t']
    Y = df_subj['a_t']
    X = X.dropna()
    Y = Y.dropna()
    X = sm.add_constant(X)  # adding a constant as intercept

    # Estimate model and extract learning rate parameter alpha (i.e., influence of delta_t on a_t)
    model = sm.OLS(Y, X).fit()
    alpha[i] = model.params['delta_t']
    age_group[i] = np.unique(df_subj['age_group'])

    # Uncomment for single-trial figure
    # plt.figure()
    # plt.plot(X, Y, '.')

# Add learning rate results to data frame
df_alpha = pd.DataFrame()
df_alpha['alpha'] = alpha
df_alpha['age_group'] = age_group

# ------------------------
# 3. Run statistical tests
# ------------------------

# Estimation errors
# -----------------

# Print out estimation error statistics for paper
print('\n\nEstimation error Experiment 1\n')
median_est_err, q1_est_err, q3_est_err, p_est_err, stat_est_err = get_stats(e_t, 1, 'e_t')

# Create data frames to save statistics for Latex manuscript
fig_1_c_desc = pd.DataFrame()
fig_1_c_stat = pd.DataFrame()

# Median estimation error
fig_1_c_desc['median'] = round(median_est_err, 3)

# First quartile
fig_1_c_desc['q1'] = round(q1_est_err, 3)

# Third quartile
fig_1_c_desc['q3'] = round(q3_est_err, 3)

# Make sure to have correct index and labels
fig_1_c_desc.index.name = 'age_group'
fig_1_c_desc = fig_1_c_desc.rename({1: 'ch', 2: 'ad', 3: 'ya', 4: 'oa'}, axis='index')

# P-values and test statistics
fig_1_c_stat['p'] = p_est_err
fig_1_c_stat['stat'] = stat_est_err
fig_1_c_stat.index.name = 'test'
fig_1_c_stat = fig_1_c_stat.rename({0: 'kw', 1: 'ch_ad', 2: 'ch_ya', 3: 'ch_oa', 4: 'ad_ya', 5: 'ad_oa', 6: 'ya_oa'},
                                   axis='index')

# Save estimation-error statistics for Latex manuscript
fig_1_c_desc.to_csv('~/Dropbox/Apps/Overleaf/al_manuscript/al_dataframes/fig_1_c_desc.csv')
fig_1_c_stat.to_csv('~/Dropbox/Apps/Overleaf/al_manuscript/al_dataframes/fig_1_c_stat.csv')

# Average learning rates
# ----------------------

# Print out average learning rate statistics for paper
print('\n\nAlpha Experiment 1\n')
median_alpha, q1_alpha, q3_alpha, p_alpha, stat_alpha = get_stats(df_alpha, 1, 'alpha')

# Create data frames to save statistics for Latex manuscript
fig_1_d_desc = pd.DataFrame()
fig_1_d_stat = pd.DataFrame()

# Median alpha
fig_1_d_desc['median'] = round(median_alpha, 3)

# First quartile
fig_1_d_desc['q1'] = round(q1_alpha, 3)

# Third quartile
fig_1_d_desc['q3'] = round(q3_alpha, 3)

# Make sure to have correct index and labels
fig_1_d_desc.index.name = 'age_group'
fig_1_d_desc = fig_1_d_desc.rename({1: 'ch', 2: 'ad', 3: 'ya', 4: 'oa'}, axis='index')

# P-values and test statistics
fig_1_d_stat['p'] = p_alpha
fig_1_d_stat['stat'] = stat_alpha
fig_1_d_stat.index.name = 'test'
fig_1_d_stat = fig_1_d_stat.rename({0: 'kw', 1: 'ch_ad', 2: 'ch_ya', 3: 'ch_oa', 4: 'ad_ya', 5: 'ad_oa', 6: 'ya_oa'},
                                   axis='index')

# Save learning-rate statistics for Latex manuscript
fig_1_d_desc.to_csv('~/Dropbox/Apps/Overleaf/al_manuscript/al_dataframes/fig_1_d_desc.csv')
fig_1_d_stat.to_csv('~/Dropbox/Apps/Overleaf/al_manuscript/al_dataframes/fig_1_d_stat.csv')

# -----------------
# 4. Prepare figure
# -----------------

# Size of figure
fig_height = 13.125
fig_width = 8

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))
f.canvas.draw()
# f.canvas.tostring_argb()

# Create plot grid
gs_0 = gridspec.GridSpec(4, 1, wspace=0.5, hspace=0.7, top=0.95, bottom=0.07, left=0.18, right=0.95)

# Plot colors
colors = ["#92e0a9", "#69b0c1", "#6d6192", "#352d4d"]
sns.set_palette(sns.color_palette(colors))

# ----------------------------
# 5. Plot task trial schematic
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
# 6. Plot block example and model computations
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
ax_12.legend(['RU', 'CPP', 'Learning\nrate'], loc=1)
ax_12.set_xlabel('Trial')
ax_12.set_ylabel('Variable')

# ----------------------------------------------
# 7. Plot performance and average learning rates
# ----------------------------------------------

# Create subplot grid
gs_02 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_0[3], hspace=0.1, wspace=0.5)

# Plot estimation-error swarm-boxplot
exp = 1
ax_2 = plt.Subplot(f, gs_02[0, 0])
f.add_subplot(ax_2)
ax_2 = swarm_boxplot(ax_2, e_t, 'e_t', 'Estimation error', exp)

# Plot learning-rate swarm-boxplot
exp = 1
ax_3 = plt.Subplot(f, gs_02[0, 1])
f.add_subplot(ax_3)
ax_3 = swarm_boxplot(ax_3, df_alpha, 'alpha', 'Learning rate', exp)

# Delete unnecessary axes
sns.despine()

# -------------------------------------
# 8. Add subplot labels and save figure
# -------------------------------------

# Label letters
texts = ['a', 'b', ' ', ' ', 'c', 'd']

# Add labels
label_subplots(f, texts, x_offset=0.15, y_offset=0.0)

# Save figure
savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_figure_1.pdf"
plt.savefig(savename, transparent=True, dpi=400)

# Show plot
plt.show()
