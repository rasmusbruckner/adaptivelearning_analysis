""" Figure 5

1. Load data
2. Compute average learning rates and bucket bias
3. Differences of perseveration frequency between shifting- and stable bucket condition
4. Differences of estimation errors between shifting- and stable bucket condition
5. Run statistical tests
6. Prepare figure
7. Plot task trial schematic
8. Plot block example and model computations
9. Plot perseveration and estimation errors
10. Plot learning rate and bucket bias (regression)
11. Plot model-based bucket-shift parameter estimate
12. Plot robust linear regression of perseveration probability on bucket-shift parameter
13. Add subplot labels and save figure
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sm
import statsmodels.formula.api as smf
from al_utilities import get_stats, get_mean_voi
from al_plot_utils import plot_image, cm2inch, label_subplots, latex_plt, swarm_boxplot, get_cond_diff
import matplotlib.gridspec as gridspec
from al_simulation import simulation


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

# Data of follow-up experiment
df_exp2 = pd.read_pickle('al_data/data_prepr_2.pkl')
n_subj = len(np.unique(df_exp2['subj_num']))

# Follow-up experiment
model_exp2 = pd.read_pickle('al_data/estimates_follow_up_exp_25_sp.pkl')

# -------------------------------------------------
# 2. Compute average learning rates and bucket bias
# -------------------------------------------------

# Initialize learning rate, bucket bias and age_group variables
alpha = np.full(n_subj, np.nan)
bucket_bias = np.full(n_subj, np.nan)
age_group = np.full(n_subj, np.nan)

# Cycle over participants
for i in range(0, n_subj):

    # Extract data of current participant
    df_subj = df_exp2[(df_exp2['subj_num'] == i + 1)].copy()
    df_subj_push = df_subj[df_subj['cond'] == 'main_push'].copy()
    df_subj_push = df_subj_push.reset_index()

    data = pd.DataFrame()
    data['a_t'] = df_subj_push['a_t'].copy()
    data['delta_t'] = df_subj_push['delta_t'].copy()
    data['y_t'] = df_subj_push['y_t'].copy()

    mod = smf.ols(formula='a_t ~ delta_t + y_t', data=data)
    res = mod.fit()

    alpha[i] = res.params['delta_t']
    bucket_bias[i] = res.params['y_t']
    age_group[i] = np.unique(df_subj['age_group'])

# Add learning rate results to data frame
df_reg = pd.DataFrame()
df_reg['alpha'] = alpha
df_reg['bucket_bias'] = bucket_bias
df_reg['age_group'] = age_group

# Print out average learning rate statistics for paper
print('\n\nAlpha Experiment 2\n')
median_alpha, q1_alpha, q3_alpha, p_alpha, stat_alpha = get_stats(df_reg, 2, 'alpha')

# Print out average bucket-bias statistics for paper
print('\n\nBucket bias Experiment 2\n')
median_bb, q1_bb, q3_bb, p_bb, stat_bb = get_stats(df_reg, 2, 'bucket_bias')

# Create data frames to save statistics for Latex manuscript
fig_5_ef_desc = pd.DataFrame()
fig_5_ef_stat = pd.DataFrame()

# Median parameter estimates
fig_5_ef_desc['median_alpha'] = round(median_alpha, 3)
fig_5_ef_desc['median_bb'] = round(median_bb, 3)

# First quartile
fig_5_ef_desc['q1_alpha'] = round(q1_alpha, 3)
fig_5_ef_desc['q1_bb'] = round(q1_bb, 3)

# Third quartile
fig_5_ef_desc['q3_alpha'] = round(q3_alpha, 3)
fig_5_ef_desc['q3_bb'] = round(q3_bb, 3)
fig_5_ef_desc.index.name = 'age_group'
fig_5_ef_desc = fig_5_ef_desc.rename({1: 'ch', 2: 'ad', 3: 'ya', 4: 'oa'}, axis='index')

# P-values and test statistics
fig_5_ef_stat['p_alpha'] = p_alpha
fig_5_ef_stat['stat_alpha'] = stat_alpha
fig_5_ef_stat['p_bb'] = p_bb
fig_5_ef_stat['stat_bb'] = stat_bb
fig_5_ef_stat.index.name = 'test'
fig_5_ef_stat = fig_5_ef_stat.rename({0: 'kw', 1: 'ch_ad', 2: 'ch_ya', 3: 'ch_oa', 4: 'ad_ya', 5: 'ad_oa', 6: 'ya_oa'},
                                     axis='index')

# Save statistics for Latex manuscript
fig_5_ef_desc.to_csv('~/Dropbox/Apps/Overleaf/al_manuscript/al_dataframes/fig_5_ef_desc.csv')
fig_5_ef_stat.to_csv('~/Dropbox/Apps/Overleaf/al_manuscript/al_dataframes/fig_5_ef_stat.csv')

# ---------------------------------------------------------------------------------------
# 3. Differences of perseveration frequency between shifting- and stable bucket condition
# ---------------------------------------------------------------------------------------

# Read out push and noPush trials
df_noPush = df_exp2[df_exp2['cond'] == 'main_noPush']
df_push = df_exp2[df_exp2['cond'] == 'main_push']

# Perseveration in stable-bucket condition
voi = 2
pers_noPush = get_mean_voi(df_noPush, voi)

# Compute differences
fig_5_c_cond_diff, fig_5_c_desc, fig_5_c_stat, fig_5_c_zero_stat = get_cond_diff(df_noPush, df_push, voi)
fig_5_c_desc.to_csv('~/Dropbox/Apps/Overleaf/al_manuscript/al_dataframes/fig_5_c_desc.csv')
fig_5_c_stat.to_csv('~/Dropbox/Apps/Overleaf/al_manuscript/al_dataframes/fig_5_c_stat.csv')
fig_5_c_zero_stat.to_csv('~/Dropbox/Apps/Overleaf/al_manuscript/al_dataframes/fig_5_c_zero_stat.csv')

# ----------------------------------------------------------------------------------
# 4. Differences of estimation errors between shifting- and stable bucket condition
# ----------------------------------------------------------------------------------

voi = 1
fig_5_d_cond_diff, fig_5_d_desc, fig_5_d_stat, fig_5_d_zero_stat = get_cond_diff(df_noPush, df_push, voi)
fig_5_d_desc.to_csv('~/Dropbox/Apps/Overleaf/al_manuscript/al_dataframes/fig_5_d_desc.csv')
fig_5_d_stat.to_csv('~/Dropbox/Apps/Overleaf/al_manuscript/al_dataframes/fig_5_d_stat.csv')
fig_5_d_zero_stat.to_csv('~/Dropbox/Apps/Overleaf/al_manuscript/al_dataframes/fig_5_d_zero_stat.csv')

# ------------------------
# 5. Run statistical tests
# ------------------------

# Print out statistics for paper
print('\n\nOmikron 0\n')
median_omikron_0, q1_omikron_0, q3_omikron_0, p_omikron_0, stat_omikron_0 = get_stats(model_exp2, 2, 'omikron_0')
print('\n\nOmikron 1\n')
median_omikron_1, q1_omikron_1, q3_omikron_1, p_omikron_1, stat_omikron_1 = get_stats(model_exp2, 2, 'omikron_1')
print('\n\nb_1 parameter\n')
median_b_0, q1_b_0, q3_b_0, p_b_0, stat_b_0 = get_stats(model_exp2, 2, 'b_0')
print('\n\nb_1parameter\n')
median_b_1, q1_b_1, q3_b_1, p_b_1, stat_b_1 = get_stats(model_exp2, 2, 'b_1')
print('\n\nUncertainty-underestimation parameter 1\n')
median_u, q1_u, q3_u, p_u, stat_u = get_stats(model_exp2, 2, 'u')
print('\n\nSurprise-sensitivity parameter 1\n')
median_s, q1_s, q3_s, p_s, stat_s = get_stats(model_exp2, 2, 's')
print('\n\nHazard-rate parameter 1\n')
median_h, q1_h, q3_h, p_h, stat_h = get_stats(model_exp2, 2, 'h')
print('\n\nCatch-trial parameter\n')
median_sigma_H, q1_sigma_H, q3_sigma_H, p_sigma_H, stat_sigma_H = get_stats(model_exp2, 2, 'sigma_H')
print('\n\nBucket-bias parameter\n')
median_d, q1_d, q3_d, p_d, stat_d = get_stats(model_exp2, 2, 'd')

# Create data frames to save statistics for Latex manuscript
fig_5_g_desc = pd.DataFrame()
fig_5_g_stat = pd.DataFrame()

# Median parameter estimates
fig_5_g_desc['median_omikron_0'] = round(median_omikron_0, 3)
fig_5_g_desc['median_omikron_1'] = round(median_omikron_1, 3)
fig_5_g_desc['median_b_0'] = round(median_b_0, 3)
fig_5_g_desc['median_b_1'] = round(median_b_1, 3)
fig_5_g_desc['median_u'] = round(median_u, 3)
fig_5_g_desc['median_s'] = round(median_s, 3)
fig_5_g_desc['median_h'] = round(median_h, 3)
fig_5_g_desc['median_sigma_H'] = round(median_sigma_H, 3)
fig_5_g_desc['median_d'] = round(median_d, 3)

# First quartile
fig_5_g_desc['q1_omikron_0'] = round(q1_omikron_0, 3)
fig_5_g_desc['q1_omikron_1'] = round(q1_omikron_1, 3)
fig_5_g_desc['q1_b_0'] = round(q1_b_0, 3)
fig_5_g_desc['q1_b_1'] = round(q1_b_1, 3)
fig_5_g_desc['q1_u'] = round(q1_u, 3)
fig_5_g_desc['q1_s'] = round(q1_s, 3)
fig_5_g_desc['q1_h'] = round(q1_h, 3)
fig_5_g_desc['q1_sigma_H'] = round(q1_sigma_H, 3)
fig_5_g_desc['q1_d'] = round(q1_d, 3)

# Third quartile
fig_5_g_desc['q3_omikron_0'] = round(q3_omikron_0, 3)
fig_5_g_desc['q3_omikron_1'] = round(q3_omikron_1, 3)
fig_5_g_desc['q3_b_0'] = round(q3_b_0, 3)
fig_5_g_desc['q3_b_1'] = round(q3_b_1, 3)
fig_5_g_desc['q3_u'] = round(q3_u, 3)
fig_5_g_desc['q3_s'] = round(q3_s, 3)
fig_5_g_desc['q3_h'] = round(q3_h, 3)
fig_5_g_desc['q3_sigma_H'] = round(q3_sigma_H, 3)
fig_5_g_desc['q3_d'] = round(q3_d, 3)
fig_5_g_desc.index.name = 'age_group'
fig_5_g_desc = fig_5_g_desc.rename({1: 'ch', 2: 'ad', 3: 'ya', 4: 'oa'}, axis='index')

# P-values and test statistics
fig_5_g_stat['p_omikron_0'] = p_omikron_0
fig_5_g_stat['stat_omikron_0'] = stat_omikron_0
fig_5_g_stat['p_omikron_1'] = p_omikron_1
fig_5_g_stat['stat_omikron_1'] = stat_omikron_1
fig_5_g_stat['p_b_0'] = p_b_0
fig_5_g_stat['stat_b_0'] = stat_b_0
fig_5_g_stat['p_b_1'] = p_b_1
fig_5_g_stat['stat_b_1'] = stat_b_1
fig_5_g_stat['p_u'] = p_u
fig_5_g_stat['stat_u'] = stat_u
fig_5_g_stat['p_s'] = p_s
fig_5_g_stat['stat_s'] = stat_s
fig_5_g_stat['p_h'] = p_h
fig_5_g_stat['stat_h'] = stat_h
fig_5_g_stat['p_sigma_H'] = p_sigma_H
fig_5_g_stat['stat_sigma_H'] = stat_sigma_H
fig_5_g_stat['p_d'] = p_d
fig_5_g_stat['stat_d'] = stat_d
fig_5_g_stat.index.name = 'test'
fig_5_g_stat = fig_5_g_stat.rename({0: 'kw', 1: 'ch_ad', 2: 'ch_ya', 3: 'ch_oa', 4: 'ad_ya', 5: 'ad_oa', 6: 'ya_oa'},
                                   axis='index')

# Save statistics for Latex manuscript
fig_5_g_desc.to_csv('~/Dropbox/Apps/Overleaf/al_manuscript/al_dataframes/fig_5_g_desc.csv')
fig_5_g_stat.to_csv('~/Dropbox/Apps/Overleaf/al_manuscript/al_dataframes/fig_5_g_stat.csv')

# -----------------
# 6. Prepare figure
# -----------------

# Adjust figure colors
colors = ["#92e0a9",  "#6d6192", "#352d4d"]
sns.set_palette(sns.color_palette(colors))

# Create figure
fig_height = 12
fig_witdh = 15
f = plt.figure(figsize=cm2inch(fig_witdh, fig_height))

# ----------------------------
# 7. Plot task trial schematic
# ----------------------------

# Create subplot grid and axis
gs0 = gridspec.GridSpec(5, 4, wspace=1, hspace=0.9, left=0.1, right=0.95, top=0.95, bottom=0.1)
gs00 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[0:2, 0:2])

# Create subplot
ax_00 = plt.Subplot(f, gs00[0, 0])
f.add_subplot(ax_00)

# Picture paths
path = ['al_figures/prediction_2.png', 'al_figures/outcome_2.png',
        'al_figures/prediction_error_2.png', 'al_figures/update_2.png']

# Figure text and font size
text = ['Prediction', 'Outcome\n(1.4s)', 'Prediction\nerror', 'Update (max. 6s)']
fontsize = 6

# Initialize image coordinates
cell_x0 = 0.0
cell_x1 = 0.2
image_y = 0.9

# Initialize text coordinates
text_y_dist = [0.05, 0.12, 0.12, 0.05]
text_pos = 'left_below'

# Cycle over images
for i in range(0, 4):

    # Plot images and text
    _, _, ab = plot_image(f, path[i], cell_x0, cell_x1, image_y, ax_00, text_y_dist[i], text[i],
                          text_pos, fontsize, zoom=0.1)

    # Update coordinates
    cell_x0 += 0.25
    cell_x1 += 0.25
    image_y += -0.2

# Delete unnecessary axes
ax_00.axis('off')

# Add bucket illustration
x = 0.735
y = -0.0
ax_00.text(x, y, "Bucket", color='k')

# Right arrow
shrinkA = 1
shrinkB = 1
x1 = 0.875
y1 = 0.02
x2 = 0.95
y2 = 0.02
ax_00.annotate("", xy=(x1, y1), xycoords='data', xytext=(x2, y2), textcoords='data',
               arrowprops=dict(arrowstyle="<-", color="k", shrinkA=shrinkA, shrinkB=shrinkB,
                               patchA=None, patchB=None, connectionstyle="arc3,rad=0"))

# Left arrow
x1 = 0.725
x2 = 0.65
ax_00.annotate("", xy=(x1, y1), xycoords='data', xytext=(x2, y2), textcoords='data',
               arrowprops=dict(arrowstyle="<-", color="k", shrinkA=shrinkA, shrinkB=shrinkB, patchA=None,
                               patchB=None, connectionstyle="arc3,rad=0"))

# Condition description
x = -0.1
y = 0.1
ax_00.text(x, y, "Block 1 and 3:\nShifting-bucket environment", color='k')
y = -0.05
ax_00.text(x, y, "Block 2 and 4:\nStable-bucket environment", color='gray')

# --------------------------------------------
# 8. Plot block example and model computations
# --------------------------------------------

# todo: bucket-shift simulationen so machen, dass bucket push nicht größer 300 und kleiner 0 ist
# also auch in supplementary figures nicht vergessen..

N = 1
model_params = pd.DataFrame(
    columns=['omikron_0', 'omikron_1', 'b_0', 'b_1', 'h', 's', 'u', 'q', 'sigma_H', 'd', 'subj_num', 'age_group'])
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
x = np.linspace(0, plot_range[1]-plot_range[0]-1, plot_range[1]-plot_range[0])
gs01 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[0:2, 2:4], hspace=0.1)
ax_01 = plt.Subplot(f, gs01[0, 0])
f.add_subplot(ax_01)

# Mean, outcomes and predictions
ax_01.plot(x, df_exp2['mu_t'][plot_range[0]:plot_range[1]], '--', color="#090030")
ax_01.plot(x, df_exp2['x_t'][plot_range[0]:plot_range[1]], 'k.', color="#090030")
ax_01.plot(x, df_data['sim_z_t'][plot_range[0]:plot_range[1]], '.', color="#04879c")
ax_01.plot(x, df_data['sim_b_t'][plot_range[0]:plot_range[1]], linewidth=2, color="#f30a49", alpha=0.8)
ax_01.set_ylabel('Screen unit')
ax_01.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
ax_01.legend(["Helicopter", "Outcome", "Bucket", "Model"], loc=1, framealpha=0.8)

# Bucket shift
ax_010 = plt.Subplot(f, gs01[1, 0])
f.add_subplot(ax_010)
ax_010.plot(x, df_data['sim_y_t'][plot_range[0]:plot_range[1]], linewidth=2, color="#090030")
ax_010.set_ylabel('Bucket shift')
ax_010.set_xlabel('Trial')

# -------------------------------------------
# 9. Plot perseveration and estimation errors
# -------------------------------------------

# Perseveration
gs02 = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs0[2:5, :], wspace=0.5, hspace=0.5)
ax_11 = plt.Subplot(f, gs02[0, 0])
f.add_subplot(ax_11)
ax_11 = swarm_boxplot(ax_11, fig_5_c_cond_diff, 'pers', 'Perseveration-\nprobability difference', 2)

# Estimation errors
ax_12 = plt.Subplot(f, gs02[0, 1])
f.add_subplot(ax_12)
swarm_boxplot(ax_12, fig_5_d_cond_diff, 'e_t', 'Estimation-\nerror difference', 2)

# ---------------------------------------------------
# 10. Plot learning rate and bucket bias (regression)
# ---------------------------------------------------

# Learning rate
ax_13 = plt.Subplot(f, gs02[0, 2])
f.add_subplot(ax_13)
swarm_boxplot(ax_13, df_reg, 'alpha', ' ', 2)
ax_13.set_ylabel('Learning rate')

# Bucket bias
ax_20 = plt.Subplot(f, gs02[1, 0])
f.add_subplot(ax_20)
swarm_boxplot(ax_20, df_reg, 'bucket_bias', ' ', 2)
ax_20.set_ylabel('Bucket bias\n(regression)')

# ----------------------------------------------------
# 11. Plot model-based bucket-shift parameter estimate
# ----------------------------------------------------

ax_21 = plt.Subplot(f, gs02[1, 1])
f.add_subplot(ax_21)
swarm_boxplot(ax_21, model_exp2, 'd', ' ', 2)
ax_21.set_ylabel('Bucket bias\n(Bayesian model)')

# ---------------------------------------------------------------------------------------
# 12. Plot robust linear regression of perseveration probability on bucket-shift parameter
# ---------------------------------------------------------------------------------------

# Data frame for regression model
data = pd.DataFrame()
data['pers'] = pers_noPush['pers'].copy()
data['d'] = model_exp2['d'].copy()

# Robust linear regression
mod = smf.rlm(formula='d ~ pers', M=sm.robust.norms.TukeyBiweight(3), data=data)
res = mod.fit(conv="weights")
print(res.summary())

# Plot results
ax_22 = plt.Subplot(f, gs02[1, 2])
f.add_subplot(ax_22)
x = pers_noPush['pers'].copy()
y = model_exp2['d'].copy()
ax_22.plot(x, y, '.', color='gray', alpha=0.7, markersize=2)
ax_22.plot(x, res.fittedvalues, '-', label="RLM", color="k")
ax_22.set_ylabel('Bucket bias\n(Bayesian model)')
ax_22.set_xlabel('Estimated\nperseveration probability')
ax_22.set_xticks(np.arange(0, 1, 0.2))

# --------------------------------------
# 13. Add subplot labels and save figure
# --------------------------------------

# Delete unnecessary axes
sns.despine()

# Add labels
texts = ['a', 'b', ' ', 'c', 'd', 'e', 'f', 'g', 'h']
label_subplots(f, texts, x_offset=0.09, y_offset=0.0)

# Save figure
savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_figure_5.pdf"
plt.savefig(savename, transparent=True, dpi=400)

# Show plot
plt.show()
