""" Figure 3

    1. Load data
    2. Prepare figure
    3. Plot perseveration frequency
    4. Plot mixture-model illustration
    5. Plot logistic-function illustration
    6. Plot intercept and slope estimates
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
import os
from al_utilities import get_mean_voi, get_stats
from al_plot_utils import cm2inch, label_subplots, swarm_boxplot, latex_plt
from AlAgentVars import AgentVars
from AlAgent import AlAgent


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

# Load data from first experiment
df_exp1 = pd.read_pickle('al_data/data_prepr_1.pkl')

# Parameter estimates
model_results = pd.read_pickle('al_data/estimates_first_exp_25_sp.pkl')

# Compute perseveration frequency
voi = 2
pers = get_mean_voi(df_exp1, voi)

# Print out perseveration statistics for paper
print('\n\nPerseveration\n')
median_pers, q1_pers, q3_pers, p_pers, stat_pers = get_stats(pers, 1, 'pers')

# Create data frame for descriptive results
fig_3_a_desc = pd.DataFrame()

# Median perseveration
fig_3_a_desc['median'] = round(median_pers, 3)

# First quartile
fig_3_a_desc['q1'] = round(q1_pers, 3)

# Third quartile
fig_3_a_desc['q3'] = round(q3_pers, 3)

# Adjust index
fig_3_a_desc.index.name = 'age_group'
fig_3_a_desc = fig_3_a_desc.rename({1: 'ch', 2: 'ad', 3: 'ya', 4: 'oa'}, axis='index')

# P-values and test statistics
fig_3_a_stat = pd.DataFrame()
fig_3_a_stat['p'] = p_pers
fig_3_a_stat['stat'] = stat_pers

# Adjust index
fig_3_a_stat.index.name = 'test'
fig_3_a_stat = fig_3_a_stat.rename({0: 'kw', 1: 'ch_ad', 2: 'ch_ya', 3: 'ch_oa', 4: 'ad_ya', 5: 'ad_oa', 6: 'ya_oa'},
                                   axis='index')

# Save perseveration statistics for Latex manuscript
fig_3_a_desc.to_csv('~/Dropbox/Apps/Overleaf/al_manuscript/al_dataframes/fig_3_a_desc.csv')
fig_3_a_stat.to_csv('~/Dropbox/Apps/Overleaf/al_manuscript/al_dataframes/fig_3_a_stat.csv')

# -----------------
# 2. Prepare figure
# -----------------

# Size of figure
fig_width = 15
fig_height = 8

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))

# Create plot grid
gs_0 = gridspec.GridSpec(1, 1, left=0.1, right=0.95, top=0.9, bottom=0.125)

# Plot colors
colors = ["#92e0a9", "#69b0c1", "#6d6192", "#352d4d"]
sns.set_palette(sns.color_palette(colors))

# -------------------------------
# 3. Plot perseveration frequency
# -------------------------------

# Create subplot grid
gs_00 = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs_0[0], hspace=0.6, wspace=0.5)

# Plot perseveration frequency
exp = 1
ax_00 = plt.Subplot(f, gs_00[0, 0])
f.add_subplot(ax_00)
swarm_boxplot(ax_00, pers, 'pers', 'Estimated probability', exp)
ax_00.set_title('Participant perseveration')

# ----------------------------------
# 4. Plot mixture-model illustration
# ----------------------------------

# Adjust data frame
model_results['subj_num'] = np.arange(0, len(model_results))

# Extract parameter estimates for the group of younger adults
ya_median_h = np.median(model_results[model_results['age_group'] == 3]['h'])
ya_median_s = np.median(model_results[model_results['age_group'] == 3]['s'])
ya_median_u = np.median(model_results[model_results['age_group'] == 3]['u'])

# Create agent-object instance for learning-rate illustration
agent_vars = AgentVars()
agent_vars.s = ya_median_s
agent_vars.h = ya_median_h

# Range of prediction errors
pe = np.linspace(1, 80, 80)

# Initialize learning-rate array
alpha = np.full(len(pe), np.nan)

# Cycle over range of prediction errors
for i in range(0, len(pe)):

    # Compute learning rate
    agent = AlAgent(agent_vars)
    agent.tau_t = 0.2/np.exp(ya_median_u)
    agent.learn(pe[i], np.nan, False, np.nan, False)
    alpha[i] = agent.alpha_t

# Plot reduced Bayesian model component
ax_01 = plt.Subplot(f, gs_00[0, 1])
f.add_subplot(ax_01)
ax_01.set_title('Mixture components')
ax_01.plot(pe, alpha, color="#584b42")
ax_01.set_ylabel('Learning rate')
ax_01.set_xlabel('Prediction error')
ax_01.text(0, 0.8, "Estimated reduced\nBayesian model")
ax_01.set_ylim([-0.03, 1.02])

# Plot perseveration model component
ax_01.plot(pe, np.zeros(80), color="#a0855b")
ax_01.text(0, 0.05, "Perseveration")

# --------------------------------------
# 5. Plot logistic-function illustration
# --------------------------------------

ax_02 = plt.Subplot(f, gs_00[0, 2])
f.add_subplot(ax_02)

# Set simulation intercept and slope
b_0 = 10  # intercept
b_1 = -0.15  # slope

# Initialize vectors for perseveration probability and updating probability
pers_prob = np.full(80, np.nan)
up_prob = np.full(80, np.nan)

# Cycle over predicted updates
for i in range(0, 80):

    pers_prob[i] = expit(np.array(b_1*(i-b_0)))
    up_prob[i] = 1-pers_prob[i]

# Plot weights of the 2 components
ax_02.plot(np.arange(80), pers_prob, color="#a0855b")
ax_02.plot(np.arange(80), up_prob, color="#584b42")
ax_02.set_ylim([-0.02, 1.2])
ax_02.set_title('Mixture weights')
ax_02.set_xlabel('Predicted update')
ax_02.set_ylabel('Probability')
ax_02.set_ylim([-0.03, 1.02])
ax_02.text(25, 0.7, "Estimated reduced\nBayesian model")
ax_02.text(40, 0.05, "Perseveration")

# -------------------------------------
# 6. Plot intercept and slope estimates
# -------------------------------------

# Intercept
ax_10 = plt.Subplot(f, gs_00[1, 0])
f.add_subplot(ax_10)
swarm_boxplot(ax_10,  model_results, 'b_0', ' ', exp)
ax_10.set_title('Intercept')
ax_10.set_ylabel('Parameter estimate')

# Slope
ax_11 = plt.Subplot(f, gs_00[1, 1])
f.add_subplot(ax_11)
swarm_boxplot(ax_11,  model_results, 'b_1', ' ', exp)
ax_11.set_title('Slope')
ax_11.set_ylabel('Parameter estimate')
ax_11.set_ylim([-1.5, 0.2])

# -------------------------------------------
# 7. Plot logistic function of each age group
# -------------------------------------------

# Compute empirical intercept and slope parameters
print('\n\nIntercept\n')
median_b_0, _, _, _, _ = get_stats(model_results, 1, 'b_0')
print('\n\nSlope\n')
median_b_1, _, _, _, _ = get_stats(model_results, 1, 'b_1')

# Initialize perseveration-frequency arrays
pers_prob_ch = np.full(80, np.nan)
pers_prob_ad = np.full(80, np.nan)
pers_prob_ya = np.full(80, np.nan)
pers_prob_oa = np.full(80, np.nan)

# Cycle over range of predicted updates
for i in range(0, len(pers_prob_ch)):

    pers_prob_ch[i] = expit(median_b_1[1]*(i-median_b_0[1]))
    pers_prob_ad[i] = expit(median_b_1[2]*(i-median_b_0[2]))
    pers_prob_ya[i] = expit(median_b_1[3]*(i-median_b_0[3]))
    pers_prob_oa[i] = expit(median_b_1[4]*(i-median_b_0[4]))

ax_12 = plt.Subplot(f, gs_00[1, 2])
f.add_subplot(ax_12)
ax_12.plot(pe, pers_prob_ch)
ax_12.plot(pe, pers_prob_ad)
ax_12.plot(pe, pers_prob_ya)
ax_12.plot(pe, pers_prob_oa)
ax_12.set_ylabel('Probability')
ax_12.set_xlabel('Predicted update')
ax_12.set_title('Estimated perseveration')
plt.legend(['CH', 'AD', 'YA', 'OA'])

# --------------------------------------
# 8. Add subplot labels and save figure
# --------------------------------------

# Delete unnecessary axes
sns.despine()

# Add labels
texts = ['a', 'b', 'c', 'd', 'e', 'f']  # label letters
label_subplots(f, texts, x_offset=0.08, y_offset=0.0)

# Save figure for manuscript
savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_figure_3.pdf"
plt.savefig(savename, transparent=True, dpi=400)

# Show plot
plt.show()
