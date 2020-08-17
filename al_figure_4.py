""" Figure 4

 1. Load data
 2. Prepare figure
 3. Plot illustration of satisficing model
 4. Plot simulation parameters
 5. Simulate data with satisficing model
 6. Compute learning rate and bucket bias with regression model
 7. Plot bucket bias, perseveration probability and estimation errors
 8. Add subplot labels and save figure
"""

import numpy as np
import scipy.stats as stats
from scipy.stats import norm
import statsmodels.formula.api as smf
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
from al_plot_utils import cm2inch, label_subplots, latex_plt, swarm_boxplot, plot_arrow
from al_simulation_satisficing import simulation_loop_satisficing
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


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
n_subj = len(np.unique(df_exp2['subj_num']))

# Follow-up experiment modeling results
model_exp2 = pd.read_pickle('al_data/estimates_follow_up_exp_25_sp.pkl')

# -----------------
# 2. Prepare figure
# -----------------

# Satisficing parameters
high_satisficing = 0.1
low_satisficing = 0.02

# Figure size
fig_height = 10
fig_width = 15

# Plot colors
colors = ["#92e0a9", "#6d6192", "#352d4d", "#69b0c1", "#a0855b", "#505050"]
sns.set_palette(sns.color_palette(colors))

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))
f.canvas.draw()

# Create plot grid
gs_0 = gridspec.GridSpec(2, 2, left=0.05, top=0.95, hspace=0.5, wspace=0.3, right=0.95)

# -----------------------------------------
# 3. Plot illustration of satisficing model
# -----------------------------------------

# Create subplot grid
gs_01 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs_0[0:2, 0], hspace=0.75)

# Create subplot axes
ax_10 = plt.Subplot(f, gs_01[0, 0])
f.add_subplot(ax_10)

# Normative model
# ---------------

# Mean and standard deviation of posterior around update
mu = 10
sigma = 10

# Plot posterior distribution
x = np.linspace(mu - 3*sigma, mu + 5*sigma, 2001)
x = np.round(x, 3)
y = stats.norm.pdf(x, mu, sigma)
ax_10.plot(x, y, color='k')

# Plot threshold and update
threshold = norm.ppf(0.5, loc=mu, scale=sigma)
threshold = np.round(threshold, 2)
ax_10.plot([10, 10], [0, y[x == 10]], color='k')
fs = 4
ax_10.text(11, 0.001, 'Optimal\nupdate', color='k', fontsize=fs)

# Plot bucket position
ax_10.plot([0, 0], [0, y[x == 10]], color=colors[3])
ax_10.text(-10.5, 0.035, 'Default\n(bucket)', color=colors[3], fontsize=fs)

# Plot arrow that indicates update
plot_arrow(ax_10, 0, 0.005, mu, 0.005, color=colors[3])

# Title and axes
ax_10.set_title('Normative')
ax_10.tick_params(axis='y', which='both', left=False, labelleft=False)
ax_10.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

# Satisficing model without perseveration
# ---------------------------------------

# Create subplot axes
ax_11 = plt.Subplot(f, gs_01[1, 0])
f.add_subplot(ax_11)

# Plot posterior distribution
ax_11.plot(x, y, color='k')

# Plot threshold and update
threshold_low = norm.ppf(0.3, loc=mu, scale=sigma)
threshold_low = np.round(threshold_low, 2)
threshold_high = norm.ppf(0.7, loc=mu, scale=sigma)
threshold_high = np.round(threshold_high, 2)
adjust_factor = 1  # make sure bar is not wider than update
ax_11.plot([threshold_low+adjust_factor, threshold_high-adjust_factor], [y[x == threshold_low], y[x == threshold_low]],
           color=colors[-1], linewidth=5)
ax_11.plot([threshold_low, threshold_low], [0, y[x == threshold_low]], color='k')

# Plot bucket position
ax_11.plot([0, 0], [0, y[x == 10]], color=colors[3])
ax_11.text(-10.5, 0.035, 'Default\n(bucket)', color=colors[3], fontsize=fs)

# Plot arrow that indicates update
plot_arrow(ax_11, 0, 0.005, threshold_low, 0.005, color=colors[3])

# Title and axes
ax_11.set_title('Default belief without perseveration')
ax_11.text(5.76, 0.001, 'Satisfied\nupdate', fontsize=fs)
ax_11.tick_params(axis='y', which='both', left=False, labelleft=False)
ax_11.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

# Illustration of satisficing parameter
axins = inset_axes(ax_11, width="5%", height="60%", loc=1)
axins.bar(0, 0.2, color=colors[-1])
axins.set_ylim(0, 0.3)
axins.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axins.set_ylabel("Satisficing\ncriterion", fontsize=4)

# Satisficing model with perseveration
# ------------------------------------

# Create subplot axes
ax_12 = plt.Subplot(f, gs_01[2, 0])
f.add_subplot(ax_12)

# Plot posterior distribution
mu = 3
y = stats.norm.pdf(x, mu, sigma)
ax_12.plot(x, y, color='k')

# Plot threshold
threshold_low = norm.ppf(0.3, loc=mu, scale=sigma)
threshold_low = np.round(threshold_low, 1)
threshold_high = norm.ppf(0.7, loc=mu, scale=sigma)
threshold_high = np.round(threshold_high, 1)
ax_12.plot([threshold_low+adjust_factor, threshold_high-adjust_factor], [y[x == threshold_low], y[x == threshold_low]],
           color=colors[-1], linewidth=5)

# Plot bucket and threshold as mixed line
xx = np.zeros(11)
yy = np.linspace(0, 0.0425, 11)
points = np.array([xx, yy]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
curr_colors = ['k', colors[3]]
colorthing = np.tile(curr_colors, np.int(len(segments)/len(curr_colors)))
cmap = ListedColormap(colorthing)
normthing = np.linspace
b_norm = BoundaryNorm(yy, cmap.N)
lc = LineCollection(segments, cmap=cmap, norm=b_norm, zorder=10)
lc.set_array(yy)
line = ax_12.add_collection(lc)

# Title, axes, main message
ax_12.set_title('Default belief leads to perseveration')
ax_12.text(25, 0.025, 'Perseveration if update is\nwithin satisficing thresholds', fontsize=fs)
ax_12.text(1, 0.001, 'Default =\nSatisfied update', fontsize=fs)
ax_12.tick_params(axis='y', which='both', left=False, labelleft=False)
ax_12.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

# Satisficing model with bucket push
# ----------------------------------

# Create subplot axes
ax_13 = plt.Subplot(f, gs_01[3, 0])
f.add_subplot(ax_13)

# Plot posterior distribution
mu = 20
y = stats.norm.pdf(x, mu, sigma)
prior, = ax_13.plot(x, y, color='k')

# Plot threshold
threshold_low = norm.ppf(0.3, loc=mu, scale=sigma)
threshold_low = np.round(threshold_low, 2)
threshold_high = norm.ppf(0.7, loc=mu, scale=sigma)
threshold_high = np.round(threshold_high, 2)
ax_13.plot([threshold_low+adjust_factor, threshold_high-adjust_factor], [y[x == threshold_low], y[x == threshold_low]],
           color=colors[-1], linewidth=5)

# Plot bucket bias and threshold
low, = ax_13.plot([threshold_low+adjust_factor, mu-adjust_factor], [0.0285, 0.0285], color=colors[3], linewidth=5)
ax_13.text(threshold_low+1, 0.021, 'Bias', color=colors[3], fontsize=fs)
ax_13.plot([threshold_low, threshold_low], [0, y[x == threshold_low]], color='k')

# Plot bucket location
bucket, = ax_13.plot([0, 0], [0, 0.04], color=colors[3])
ax_13.text(-10.5, 0.035, 'Default\n(bucket)', color=colors[3], fontsize=fs)

# Plot arrow that indicates update
plot_arrow(ax_13, 0, 0.005, threshold_low, 0.005, color=colors[3])

# Title, axes, main message
ax_13.set_title('Default belief leads to belief-updating bias')
ax_13.text(35, 0.025, 'Large required update:\nNo perseveration but\nbias towards default', fontsize=fs)
ax_13.text(threshold_low+1, 0.001, 'Satisfied\nupdate', fontsize=fs)
ax_13.tick_params(axis='y', which='both', left=False, labelleft=False)
ax_13.set_xlabel('Update', fontsize=fs)

# Add labels
texts = ['a', '', 'b', 'c', 'd']
label_subplots(f, texts, x_offset=0.04, y_offset=0.0)

# -----------------------------
# 4. Plot simulation parameters
# -----------------------------

# Create subplot axes
gs_12 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_0[0:2, 1:], hspace=0.4, wspace=0.6)
ax_0 = plt.Subplot(f, gs_12[0, 0])
f.add_subplot(ax_0)

ax_0.bar(np.arange(3), [high_satisficing, low_satisficing, high_satisficing], color=colors)
ax_0.set_ylabel("Satisficing criterion")
ax_0.set_ylim(0, 0.15)
ax_0.set_xlabel('Age group')
plt.xticks([0, 1, 2], ('CH', 'YA', 'OA'))

# ---------------------------------------
# 5. Simulate data with satisficing model
# ---------------------------------------

# Adjust model parameters # todo: parameterize epsilon here
model_exp2['omikron_0'] = np.nan
model_exp2['omikron_1'] = np.nan
model_exp2['b_0'] = np.nan
model_exp2['b_1'] = np.nan
model_exp2['h'] = 0.1
model_exp2['s'] = 1
model_exp2['u'] = 0.0
model_exp2['q'] = 0.0
model_exp2['sigma_H'] = 1
model_exp2['d'] = 0
model_exp2['low_satisficing'] = low_satisficing
model_exp2['high_satisficing'] = high_satisficing

# Run simulation
n_sim = 1
sim_pers = True
all_pers, all_est_errs, df_data = simulation_loop_satisficing(df_exp2, model_exp2, n_subj, n_sim=n_sim)

# --------------------------------------------------------------
# 6. Compute learning rate and bucket bias with regression model
# --------------------------------------------------------------

# Initialize learning rate and age_group variables
alpha = np.full(n_subj, np.nan)
bucket_bias = np.full(n_subj, np.nan)
age_group = np.full(n_subj, np.nan)

# Cycle over participants
for i in range(0, n_subj):

    # Extract data of current participant
    df_subj = df_data[(df_data['subj_num'] == i + 1)].copy()
    df_subj_push = df_subj[df_subj['cond'] == 'main_push'].copy()
    df_subj_push = df_subj_push.reset_index()

    # Create data frame for regression
    data = pd.DataFrame()
    data['a_t'] = df_subj_push['sim_a_t'].copy()
    data['delta_t'] = df_subj_push['delta_t'].copy()
    data['y_t'] = df_subj_push['sim_y_t'].copy()
    data = data.dropna().reset_index()

    # Run regression
    mod = smf.ols(formula='a_t ~ delta_t + y_t', data=data)
    res = mod.fit()

    # Save results
    alpha[i] = res.params['delta_t']
    bucket_bias[i] = res.params['y_t']
    age_group[i] = np.unique(df_exp2[df_exp2['subj_num'] == i + 1]['age_group'])

# Add learning rate results to data frame
df_reg = pd.DataFrame()
df_reg['alpha'] = alpha
df_reg['bucket_bias'] = bucket_bias
df_reg['age_group'] = age_group

# --------------------------------------------------------------------
# 7. Plot bucket bias, perseveration probability and estimation errors
# --------------------------------------------------------------------

# Plot bucket bias
ax_1 = plt.Subplot(f, gs_12[0, 1])
f.add_subplot(ax_1)
swarm_boxplot(ax_1, df_reg, 'bucket_bias', ' ', 2)
ax_1.set_ylabel('Belief-updating bias\n(regression)')

# Difference between conditions
all_est_errs['diff'] = all_est_errs['push'] - all_est_errs['noPush']
all_pers['diff'] = all_pers['push'] - all_pers['noPush']

# Plot perseveration probability
ax_0 = plt.Subplot(f, gs_12[1, 0])
f.add_subplot(ax_0)
swarm_boxplot(ax_0, all_pers, "diff", "diff", 2)
ax_0.set_ylabel('Perseveration-\nprobability difference')
ax_0.set_xlabel('Age group')
plt.xticks([0, 1, 2], ('CH', 'YA', 'OA'))

# Plot estimation errors
ax_1 = plt.Subplot(f, gs_12[1, 1])
f.add_subplot(ax_1)
sns.swarmplot(x='age_group', y='diff', data=all_est_errs, alpha=0.7, size=1.5, ax=ax_1, color='k')
swarm_boxplot(ax_1, all_est_errs, "diff", "diff", 2)
ax_1.set_ylabel('Estimation-\nerror difference')
plt.xticks(np.arange(3), ['CH', 'YA', 'OA'], rotation=0)
ax_1.set_xlabel('Age group')
plt.xticks([0, 1, 2], ('CH', 'YA', 'OA'))

# Adjust axes
sns.despine()
ax_10.spines['left'].set_visible(False)
ax_11.spines['left'].set_visible(False)
ax_12.spines['left'].set_visible(False)
ax_13.spines['left'].set_visible(False)

# --------------------------------------
# 13. Add subplot labels and save figure
# --------------------------------------

# Add labels
texts = ['', '', '', '', '', 'e', 'f', 'g', 'h']
label_subplots(f, texts, x_offset=0.08, y_offset=0.0)

# Save figure
savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_figure_4.pdf"
plt.savefig(savename, transparent=True, dpi=400)

# Show plot
plt.show()
