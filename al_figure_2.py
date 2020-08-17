""" Figure 2

    1. Load data
    2. Compute estimation errors and perseveration probability
    3. Compute average learning rates
    4. Run statistical tests
    5. Prepare figure
    6. Plot performance, average learning rates, and perseveration
    7. Plot logistic function of each age group
    8. Add subplot labels and save figure
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import expit
import seaborn as sns
import statsmodels.api as sm
import os
from al_utilities import get_mean_voi, get_stats
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
n_subj = len(np.unique(df_exp1['subj_num']))

# Parameter estimates
model_results = pd.read_pickle('al_data/estimates_first_exp_25_sp.pkl')

# ----------------------------------------------------------
# 2. Compute estimation errors and perseveration probability
# ----------------------------------------------------------

# Compute estimation errors
voi = 1
e_t = get_mean_voi(df_exp1, voi)

# Compute perseveration frequency
voi = 2
pers = get_mean_voi(df_exp1, voi)

# ---------------------------------
# 3. Compute average learning rates
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
# 4. Run statistical tests
# ------------------------

# Estimation errors
# -----------------

# Print out estimation error statistics for paper
print('\n\nEstimation error Experiment 1\n')
median_est_err, q1_est_err, q3_est_err, p_est_err, stat_est_err = get_stats(e_t, 1, 'e_t')

# Create data frames to save statistics for Latex manuscript
fig_2_a_desc = pd.DataFrame()
fig_2_a_stat = pd.DataFrame()

# Median estimation error
fig_2_a_desc['median'] = round(median_est_err, 3)

# First quartile
fig_2_a_desc['q1'] = round(q1_est_err, 3)

# Third quartile
fig_2_a_desc['q3'] = round(q3_est_err, 3)

# Make sure to have correct index and labels
fig_2_a_desc.index.name = 'age_group'
fig_2_a_desc = fig_2_a_desc.rename({1: 'ch', 2: 'ad', 3: 'ya', 4: 'oa'}, axis='index')

# P-values and test statistics
fig_2_a_stat['p'] = p_est_err
fig_2_a_stat['stat'] = stat_est_err
fig_2_a_stat.index.name = 'test'
fig_2_a_stat = fig_2_a_stat.rename({0: 'kw', 1: 'ch_ad', 2: 'ch_ya', 3: 'ch_oa', 4: 'ad_ya', 5: 'ad_oa', 6: 'ya_oa'},
                                   axis='index')

# Save estimation-error statistics for Latex manuscript
fig_2_a_desc.to_csv('~/Dropbox/Apps/Overleaf/al_manuscript/al_dataframes/fig_2_a_desc.csv')
fig_2_a_stat.to_csv('~/Dropbox/Apps/Overleaf/al_manuscript/al_dataframes/fig_2_a_stat.csv')

# Average learning rates
# ----------------------

# Print out average learning rate statistics for paper
print('\n\nAlpha Experiment 1\n')
median_alpha, q1_alpha, q3_alpha, p_alpha, stat_alpha = get_stats(df_alpha, 1, 'alpha')

# Create data frames to save statistics for Latex manuscript
fig_2_b_desc = pd.DataFrame()
fig_2_b_stat = pd.DataFrame()

# Median alpha
fig_2_b_desc['median'] = round(median_alpha, 3)

# First quartile
fig_2_b_desc['q1'] = round(q1_alpha, 3)

# Third quartile
fig_2_b_desc['q3'] = round(q3_alpha, 3)

# Make sure to have correct index and labels
fig_2_b_desc.index.name = 'age_group'
fig_2_b_desc = fig_2_b_desc.rename({1: 'ch', 2: 'ad', 3: 'ya', 4: 'oa'}, axis='index')

# P-values and test statistics
fig_2_b_stat['p'] = p_alpha
fig_2_b_stat['stat'] = stat_alpha
fig_2_b_stat.index.name = 'test'
fig_2_b_stat = fig_2_b_stat.rename({0: 'kw', 1: 'ch_ad', 2: 'ch_ya', 3: 'ch_oa', 4: 'ad_ya', 5: 'ad_oa', 6: 'ya_oa'},
                                   axis='index')

# Save learning-rate statistics for Latex manuscript
fig_2_b_desc.to_csv('~/Dropbox/Apps/Overleaf/al_manuscript/al_dataframes/fig_2_b_desc.csv')
fig_2_b_stat.to_csv('~/Dropbox/Apps/Overleaf/al_manuscript/al_dataframes/fig_2_b_stat.csv')

# Perseveration
# -------------

# Print out perseveration statistics for paper
print('\n\nPerseveration\n')
median_pers, q1_pers, q3_pers, p_pers, stat_pers = get_stats(pers, 1, 'pers')

# Create data frame for descriptive results
fig_2_c_desc = pd.DataFrame()

# Median perseveration
fig_2_c_desc['median'] = round(median_pers, 3)

# First quartile
fig_2_c_desc['q1'] = round(q1_pers, 3)

# Third quartile
fig_2_c_desc['q3'] = round(q3_pers, 3)

# Adjust index
fig_2_c_desc.index.name = 'age_group'
fig_2_c_desc = fig_2_c_desc.rename({1: 'ch', 2: 'ad', 3: 'ya', 4: 'oa'}, axis='index')

# P-values and test statistics
fig_2_c_stat = pd.DataFrame()
fig_2_c_stat['p'] = p_pers
fig_2_c_stat['stat'] = stat_pers

# Adjust index
fig_2_c_stat.index.name = 'test'
fig_2_c_stat = fig_2_c_stat.rename({0: 'kw', 1: 'ch_ad', 2: 'ch_ya', 3: 'ch_oa', 4: 'ad_ya', 5: 'ad_oa', 6: 'ya_oa'},
                                   axis='index')

# Save perseveration statistics for Latex manuscript
fig_2_c_desc.to_csv('~/Dropbox/Apps/Overleaf/al_manuscript/al_dataframes/fig_2_c_desc.csv')
fig_2_c_stat.to_csv('~/Dropbox/Apps/Overleaf/al_manuscript/al_dataframes/fig_2_c_stat.csv')

# -----------------
# 5. Prepare figure
# -----------------

# Size of figure
fig_height = 8
fig_width = 8

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))

# Create plot grid
gs_0 = gridspec.GridSpec(1, 1, wspace=0.5, hspace=0.7, top=0.925, bottom=0.12, left=0.18, right=0.95)

# Plot colors
colors = ["#92e0a9", "#69b0c1", "#6d6192", "#352d4d"]
sns.set_palette(sns.color_palette(colors))

# --------------------------------------------------------------
# 6. Plot performance, average learning rates, and perseveration
# --------------------------------------------------------------

# Create subplot grid
gs_00 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_0[0], hspace=0.6, wspace=0.5)

# Plot estimation-error swarm-boxplot
exp = 1
ax_2 = plt.Subplot(f, gs_00[0, 0])
f.add_subplot(ax_2)
ax_2 = swarm_boxplot(ax_2, e_t, 'e_t', 'Estimation error', exp)
ax_2.set_title('Participants')

# Plot learning-rate swarm-boxplot
exp = 1
ax_3 = plt.Subplot(f, gs_00[0, 1])
f.add_subplot(ax_3)
ax_3 = swarm_boxplot(ax_3, df_alpha, 'alpha', 'Learning rate', exp)
ax_3.set_title('Participants')

# Plot perseveration frequency
exp = 1
ax_00 = plt.Subplot(f, gs_00[1, 0])
f.add_subplot(ax_00)
swarm_boxplot(ax_00, pers, 'pers', 'Perseveration\nprobability', exp)
ax_00.set_title('Participants')

# -------------------------------------------
# 7. Plot logistic function of each age group
# -------------------------------------------

# Compute empirical intercept and slope parameters
print('\n\nIntercept\n')
median_b_0, _, _, _, _ = get_stats(model_results, 1, 'b_0')
print('\n\nSlope\n')
median_b_1, _, _, _, _ = get_stats(model_results, 1, 'b_1')

# Initialize perseveration-frequency arrays
pers_prob_ch = np.full(50, np.nan)
pers_prob_ad = np.full(50, np.nan)
pers_prob_ya = np.full(50, np.nan)
pers_prob_oa = np.full(50, np.nan)

# Range of predicted update
pe = np.linspace(1, 50, 50)

# Cycle over range of predicted updates
for i in range(0, len(pers_prob_ch)):

    pers_prob_ch[i] = expit(median_b_1[1]*(i-median_b_0[1]))
    pers_prob_ad[i] = expit(median_b_1[2]*(i-median_b_0[2]))
    pers_prob_ya[i] = expit(median_b_1[3]*(i-median_b_0[3]))
    pers_prob_oa[i] = expit(median_b_1[4]*(i-median_b_0[4]))

ax_12 = plt.Subplot(f, gs_00[1, 1])
f.add_subplot(ax_12)
ax_12.plot(pe, pers_prob_ch)
ax_12.plot(pe, pers_prob_ad)
ax_12.plot(pe, pers_prob_ya)
ax_12.plot(pe, pers_prob_oa)
ax_12.set_ylabel('Perseveration\nprobability')
ax_12.set_xlabel('Predicted update')
ax_12.set_title('Model-based estimate')
plt.legend(['CH', 'AD', 'YA', 'OA'])

# -------------------------------------
# 8. Add subplot labels and save figure
# -------------------------------------

# Delete unnecessary axes
sns.despine()

# Add labels
texts = ['a', 'b', 'c', 'd']  # label letters
label_subplots(f, texts, x_offset=0.12, y_offset=0.025)

# Save figure for manuscript
savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_figure_2.pdf"
plt.savefig(savename, transparent=True, dpi=400)

# Show plot
plt.show()
