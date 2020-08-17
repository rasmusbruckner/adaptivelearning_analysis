""" Figure S2

    1. Load data
    2. Run statistical tests
    3. Prepare figure
    4. Plot learning-rate simulations
    5. Plot parameter estimates
    6. Add subplot labels and save figure
"""

import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import os
from al_utilities import get_stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from AlAgentVars import AgentVars
from AlAgent import AlAgent
from al_plot_utils import cm2inch, label_subplots, latex_plt, swarm_boxplot


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

model_results = pd.read_pickle('al_data/estimates_first_exp_25_sp.pkl')

# -------------------------
# 2. Runs statistical tests
# -------------------------

# Print out statistics for paper
print('\n\nMotor-noise parameter\n')
median_omikron_0, q1_omikron_0, q3_omikron_0, p_omikron_0, stat_omikron_0 = get_stats(model_results, 1, 'omikron_0')
print('\n\nLearning-rate-noise parameter\n')
median_omikron_1, q1_omikron_1, q3_omikron_1, p_omikron_1, stat_omikron_1 = get_stats(model_results, 1, 'omikron_1')
print('\n\nb_0 parameter 1\n')
median_b_0, q1_b_0, q3_b_0, p_b_0, stat_b_0 = get_stats(model_results, 1, 'b_0')
print('\n\nb_1 parameter\n')
median_b_1, q1_b_1, q3_b_1, p_b_1, stat_b_1 = get_stats(model_results, 1, 'b_1')
print('\n\nUncertainty-underestimation parameter\n')
median_u, q1_u, q3_u, p_u, stat_u = get_stats(model_results, 1, 'u')
print('\n\nSurprise-sensitivity parameter\n')
median_s, q1_s, q3_s, p_s, stat_s = get_stats(model_results, 1, 's')
print('\n\nHazard-rate parameter\n')
median_h, q1_h, q3_h, p_h, stat_h = get_stats(model_results, 1, 'h')
print('\n\nCatch-trial parameter\n')
median_sigma_H, q1_sigma_H, q3_sigma_H, p_sigma_H, stat_sigma_H = get_stats(model_results, 1, 'sigma_H')
print('\n\nReward-bias parameter\n')
median_q, q1_q, q3_q, p_q, stat_q = get_stats(model_results, 1, 'q')

# Create data frames to save statistics for Latex manuscript
fig_S_2_desc = pd.DataFrame()
fig_S_2_stat = pd.DataFrame()

# Median parameter estimates
fig_S_2_desc['median_omikron_0'] = round(median_omikron_0, 3)
fig_S_2_desc['median_omikron_1'] = round(median_omikron_1, 3)
fig_S_2_desc['median_b_0'] = round(median_b_0, 3)
fig_S_2_desc['median_b_1'] = round(median_b_1, 3)
fig_S_2_desc['median_u'] = round(median_u, 3)
fig_S_2_desc['median_s'] = round(median_s, 3)
fig_S_2_desc['median_h'] = round(median_h, 3)
fig_S_2_desc['median_q'] = round(median_q, 3)
fig_S_2_desc['median_sigma_H'] = round(median_sigma_H, 3)

# First quartile
fig_S_2_desc['q1_omikron_0'] = round(q1_omikron_0, 3)
fig_S_2_desc['q1_omikron_1'] = round(q1_omikron_1, 3)
fig_S_2_desc['q1_b_0'] = round(q1_b_0, 3)
fig_S_2_desc['q1_b_1'] = round(q1_b_1, 3)
fig_S_2_desc['q1_u'] = round(q1_u, 3)
fig_S_2_desc['q1_s'] = round(q1_s, 3)
fig_S_2_desc['q1_h'] = round(q1_h, 3)
fig_S_2_desc['q1_q'] = round(q1_q, 3)
fig_S_2_desc['q1_sigma_H'] = round(q1_sigma_H, 3)

# Third quartile
fig_S_2_desc['q3_omikron_0'] = round(q3_omikron_0, 3)
fig_S_2_desc['q3_omikron_1'] = round(q3_omikron_1, 3)
fig_S_2_desc['q3_b_0'] = round(q3_b_0, 3)
fig_S_2_desc['q3_b_1'] = round(q3_b_1, 3)
fig_S_2_desc['q3_u'] = round(q3_u, 3)
fig_S_2_desc['q3_s'] = round(q3_s, 3)
fig_S_2_desc['q3_h'] = round(q3_h, 3)
fig_S_2_desc['q3_q'] = round(q3_q, 3)
fig_S_2_desc['q3_sigma_H'] = round(q3_sigma_H, 3)

# Rename index and groups
fig_S_2_desc.index.name = 'age_group'
fig_S_2_desc = fig_S_2_desc.rename({1: 'ch', 2: 'ad', 3: 'ya', 4: 'oa'}, axis='index')

# P-values and test statistics
fig_S_2_stat['p_omikron_0'] = p_omikron_0
fig_S_2_stat['stat_omikron_0'] = stat_omikron_0
fig_S_2_stat['p_omikron_1'] = p_omikron_1
fig_S_2_stat['stat_omikron_1'] = stat_omikron_1
fig_S_2_stat['p_b_0'] = p_b_0
fig_S_2_stat['stat_b_0'] = stat_b_0
fig_S_2_stat['p_b_1'] = p_b_1
fig_S_2_stat['stat_b_1'] = stat_b_1
fig_S_2_stat['p_u'] = p_u
fig_S_2_stat['stat_u'] = stat_u
fig_S_2_stat['p_s'] = p_s
fig_S_2_stat['stat_s'] = stat_s
fig_S_2_stat['p_h'] = p_h
fig_S_2_stat['stat_h'] = stat_h
fig_S_2_stat['p_q'] = p_q
fig_S_2_stat['stat_q'] = stat_q
fig_S_2_stat['p_sigma_H'] = p_sigma_H
fig_S_2_stat['stat_sigma_H'] = stat_sigma_H

# Rename index and tests
fig_S_2_stat.index.name = 'test'
fig_S_2_stat = fig_S_2_stat.rename({0: 'kw', 1: 'ch_ad', 2: 'ch_ya', 3: 'ch_oa', 4: 'ad_ya', 5: 'ad_oa', 6: 'ya_oa'},
                               axis='index')
# Save statistics for Latex manuscript
fig_S_2_desc.to_csv('~/Dropbox/Apps/Overleaf/al_manuscript/al_dataframes/fig_S_2_desc.csv')
fig_S_2_stat.to_csv('~/Dropbox/Apps/Overleaf/al_manuscript/al_dataframes/fig_S_2_stat.csv')

# -----------------
# 3. Prepare figure
# -----------------

# Size of figure
fig_width = 15
fig_height = 9

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))

# Create plot grid
gs_0 = gridspec.GridSpec(1, 1, left=0.1, right=0.95)

# Plot colors
colors = ["#92e0a9", "#69b0c1", "#6d6192", "#352d4d"]
sns.set_palette(sns.color_palette(colors))

# ---------------------------------
# 4. Plot learning-rate simulations
# ---------------------------------

# Create subplot grid
gs_00 = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs_0[0], hspace=0.7, wspace=0.5)

# Agent object instance
agent_vars = AgentVars()
agent = AlAgent(agent_vars)

# Initialize arrays
pe = np.linspace(1, 80, 80)
alpha = np.full(80, np.nan)
alpha_u = np.full(80, np.nan)
alpha_s = np.full(80, np.nan)
alpha_h = np.full(80, np.nan)

# Cycle over prediction error range
# ---------------------------------

for i in range(0, 80):

    # Set agent variables
    agent_vars.h = 0.1
    agent_vars.s = 1
    agent_vars.u = np.exp(0)
    agent_vars.q = 0
    agent_vars.sigma_H = 0

    # Normative model
    agent = AlAgent(agent_vars)
    agent.tau_t = 0.2
    agent.learn(pe[i], np.nan, False, np.nan, False)
    alpha[i] = agent.alpha_t

    # Uncertainty underestimation
    agent = AlAgent(agent_vars)
    agent.tau_t = 0.2/10
    agent.learn(pe[i], np.nan, False, np.nan, False)
    alpha_u[i] = agent.alpha_t

    # Surprise sensitivity
    agent_vars.s = 0.3
    agent = AlAgent(agent_vars)
    agent.tau_t = 0.2
    agent.learn(pe[i], np.nan, False, np.nan, False)
    alpha_s[i] = agent.alpha_t

    # Increased hazard rate
    agent_vars.s = 1
    agent_vars.h = 0.5
    agent = AlAgent(agent_vars)
    agent.tau_t = 0.2
    agent.learn(pe[i], np.nan, False, np.nan, False)
    alpha_h[i] = agent.alpha_t

# Illustrate uncertainty underestimation
ax_00 = plt.Subplot(f, gs_00[0, 0])
f.add_subplot(ax_00)
ax_00.plot(np.arange(80), alpha, color="#f30a49")
ax_00.plot(np.arange(80), alpha_u, color="#584b42")
ax_00.set_ylim([-0.02, 1.02])
ax_00.set_ylabel('Learning rate')
ax_00.set_xlabel('Prediction error')
ax_00.set_title('Simulated\nuncertainty underestimation')

# Illustrate surprise sensitivity
ax_01 = plt.Subplot(f, gs_00[0, 1])
f.add_subplot(ax_01)
ax_01.plot(np.arange(80), alpha, color="#f30a49")
ax_01.plot(np.arange(80), alpha_s, color="#584b42")
ax_01.set_ylim([-0.02, 1.02])
ax_01.set_xlabel('Prediction error')
ax_01.set_title('Simulated\nsurprise sensitivity')

# Illustrate high hazard rate
ax_02 = plt.Subplot(f, gs_00[0, 2])
f.add_subplot(ax_02)
ax_02.plot(np.arange(80), alpha, color="#f30a49")
ax_02.plot(np.arange(80), alpha_h, color="#584b42")
ax_02.set_ylim([-0.02, 1.02])
ax_02.set_xlabel('Prediction error')
ax_02.set_title('Simulated\nhazard-rate impact')

# ---------------------------
# 5. Plot parameter estimates
# ---------------------------

# Plot properties
exp = 1

# Uncertainty underestimation
ax_10 = plt.Subplot(f, gs_00[1, 0])
f.add_subplot(ax_10)
swarm_boxplot(ax_10,  model_results, 'u', 'Parameter estimate', exp)
ax_10.set_ylim([-1.5, 8.5])
ax_10.set_title('Empirical\nuncertainty underestimation')

# Surprise sensitivity
ax_11 = plt.Subplot(f, gs_00[1, 1])
f.add_subplot(ax_11)
swarm_boxplot(ax_11,  model_results, 's', ' ', exp)
ax_11.set_title('Empirical\nsurprise sensitivity')

# Hazard rate
ax_12 = plt.Subplot(f, gs_00[1, 2])
f.add_subplot(ax_12)
swarm_boxplot(ax_12,  model_results, 'h', ' ', exp)
ax_12.set_title('Empirical\nhazard rate')
sns.despine()

# --------------------------------------
# 6. Add subplot labels and save figure
# --------------------------------------

# Add labels
texts = ['a', 'b', 'c', 'd', 'e', 'f']  # label letters
label_subplots(f, texts, x_offset=0.08, y_offset=0.0)

# Save figure for manuscript
savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_S_figure_2.pdf"
plt.savefig(savename, transparent=True, dpi=400)

plt.show()
