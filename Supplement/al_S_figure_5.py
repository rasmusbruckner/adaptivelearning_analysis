""" Figure S5: Model-fitting illustration and results experiment 1

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
from AlAgentVarsRbm import AgentVars
from AlAgentRbm import AlAgent
from al_plot_utils import cm2inch, label_subplots, latex_plt, swarm_boxplot
from al_utilities import safe_save_dataframe

# Update matplotlib to use Latex and to change some defaults
matplotlib = latex_plt(matplotlib)

# Get home directory
paths = os.getcwd()
path = paths.split(os.path.sep)
home_dir = path[1]

# ------------
# 1. Load data
# ------------
model_results = pd.read_pickle('al_data/estimates_first_exp_10_sp.pkl')

# -------------------------
# 2. Runs statistical tests
# -------------------------

# Print out statistics for paper
print('\n\nMotor-noise parameter\n')
exp1_omikron_0_desc, exp1_omikron_0_stat, exp1_omikron_0_effect_size = get_stats(model_results, 1, 'omikron_0')
print('\n\nLearning-rate-noise parameter\n')
exp1_omikron_1_desc, exp1_omikron_1_stat, exp1_omikron_1_effect_size = get_stats(model_results, 1, 'omikron_1')
print('\n\nb_0 parameter\n')
exp1_b_0_desc, exp1_b_0_stat, exp1_b_0_effect_size = get_stats(model_results, 1, 'b_0')
print('\n\nb_1 parameter\n')
exp1_b_1_desc, exp1_b_1_stat, exp1_b_1_effect_size = get_stats(model_results, 1, 'b_1')
print('\n\nUncertainty-underestimation parameter\n')
exp1_u_desc, exp1_u_stat, exp1_u_effect_size = get_stats(model_results, 1, 'u')
print('\n\nSurprise-sensitivity parameter\n')
exp1_s_desc, exp1_s_stat, exp1_s_effect_size = get_stats(model_results, 1, 's')
print('\n\nHazard-rate parameter\n')
exp1_h_desc, exp1_h_stat, exp1_h_effect_size = get_stats(model_results, 1, 'h')
print('\n\nCatch-trial parameter\n')
exp1_sigma_H_desc, exp1_sigma_H_stat, exp1_sigma_H_effect_size = get_stats(model_results, 1, 'sigma_H')
print('\n\nReward-bias parameter\n')
exp1_q_desc, exp1_q_stat, exp1_q_effect_size = get_stats(model_results, 1, 'q')

exp1_model_fitting_desc = pd.concat([exp1_omikron_0_desc.add_suffix('_omikron_0'),
                                     exp1_omikron_1_desc.add_suffix('_omikron_1'),
                                     exp1_b_0_desc.add_suffix('_b_0'),
                                     exp1_b_1_desc.add_suffix('_b_1'),
                                     exp1_u_desc.add_suffix('_u'),
                                     exp1_s_desc.add_suffix('_s'),
                                     exp1_h_desc.add_suffix('_h'),
                                     exp1_sigma_H_desc.add_suffix('_sigma_h'),
                                     exp1_q_desc.add_suffix('_q')], axis=1)

exp1_model_fitting_stat = pd.concat([exp1_omikron_0_stat.add_suffix('_omikron_0'),
                                     exp1_omikron_1_stat.add_suffix('_omikron_1'),
                                     exp1_b_0_stat.add_suffix('_b_0'),
                                     exp1_b_1_stat.add_suffix('_b_1'),
                                     exp1_u_stat.add_suffix('_u'),
                                     exp1_s_stat.add_suffix('_s'),
                                     exp1_h_stat.add_suffix('_h'),
                                     exp1_sigma_H_stat.add_suffix('_sigma_h'),
                                     exp1_q_stat.add_suffix('_q')], axis=1)

exp1_model_fitting_effect_size = pd.concat([exp1_omikron_0_effect_size.add_suffix('_omikron_0'),
                                            exp1_omikron_1_effect_size.add_suffix('_omikron_1'),
                                            exp1_b_0_effect_size.add_suffix('_b_0'),
                                            exp1_b_1_effect_size.add_suffix('_b_1'),
                                            exp1_u_effect_size.add_suffix('_u'),
                                            exp1_s_effect_size.add_suffix('_s'),
                                            exp1_h_effect_size.add_suffix('_h'),
                                            exp1_sigma_H_effect_size.add_suffix('_sigma_h'),
                                            exp1_q_effect_size.add_suffix('_q')], axis=1)

exp1_model_fitting_desc.name, exp1_model_fitting_stat.name, exp1_model_fitting_effect_size.name =\
    "exp1_model_fitting_desc", "exp1_model_fitting_stat", "exp1_model_fitting_effect_size"

# Save statistics for Latex manuscript
safe_save_dataframe(exp1_model_fitting_desc, 'age_group')
safe_save_dataframe(exp1_model_fitting_stat, 'test')
safe_save_dataframe(exp1_model_fitting_effect_size, 'type')

# -----------------
# 3. Prepare figure
# -----------------

# Size of figure
fig_width = 15
fig_height = 9

# Plot colors
colors = ["#BBE1FA", "#3282B8", "#0F4C75", "#1B262C"]
sns.set_palette(sns.color_palette(colors))

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))

# Create plot grid
gs_0 = gridspec.GridSpec(1, 1, left=0.1, right=0.95)

# Y-label distance
ylabel_dist = -0.225

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

# Cycle over prediction-error range
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
    agent.tau_t = 0.2 / 10
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
ax_00.plot(np.arange(80), alpha, color="k")
ax_00.plot(np.arange(80), alpha_u, color="#0F4C75")
ax_00.set_ylim([-0.02, 1.02])
ax_00.set_ylabel('Learning Rate')
ax_00.yaxis.set_label_coords(ylabel_dist, 0.5)
ax_00.set_xlabel('Prediction Error')
ax_00.set_title('Simulated\nUncertainty Underestimation')
ax_00.legend(['RBM', 'Biased'], fontsize=5, loc='lower right')

# Illustrate surprise sensitivity
ax_01 = plt.Subplot(f, gs_00[0, 1])
f.add_subplot(ax_01)
ax_01.plot(np.arange(80), alpha, color="k")
ax_01.plot(np.arange(80), alpha_s, color="#0F4C75")
ax_01.set_ylim([-0.02, 1.02])
ax_01.set_xlabel('Prediction Error')
ax_01.set_title('Simulated\nSurprise Sensitivity')

# Illustrate high hazard rate
ax_02 = plt.Subplot(f, gs_00[0, 2])
f.add_subplot(ax_02)
ax_02.plot(np.arange(80), alpha, color="k")
ax_02.plot(np.arange(80), alpha_h, color="#0F4C75")
ax_02.set_ylim([-0.02, 1.02])
ax_02.set_xlabel('Prediction Error')
ax_02.set_title('Simulated\nHazard-Rate Impact')

# ---------------------------
# 5. Plot parameter estimates
# ---------------------------

# Plot properties
exp = 1

# Uncertainty underestimation
ax_10 = plt.Subplot(f, gs_00[1, 0])
f.add_subplot(ax_10)
swarm_boxplot(ax_10, model_results, 'u', 'Parameter Estimate', exp)
ax_10.yaxis.set_label_coords(ylabel_dist, 0.5)
ax_10.set_ylim([-1.5, 7.0])
ax_10.set_title('Empirical\nUncertainty Underestimation')

# Surprise sensitivity
ax_11 = plt.Subplot(f, gs_00[1, 1])
f.add_subplot(ax_11)
swarm_boxplot(ax_11, model_results, 's', ' ', exp)
ax_11.set_title('Empirical\nSurprise Sensitivity')

# Hazard rate
ax_12 = plt.Subplot(f, gs_00[1, 2])
f.add_subplot(ax_12)
swarm_boxplot(ax_12, model_results, 'h', ' ', exp)
ax_12.set_title('Empirical\nHazard Rate')
sns.despine()

# --------------------------------------
# 6. Add subplot labels and save figure
# --------------------------------------

# Add labels
texts = ['a', 'b', 'c', 'd', 'e', 'f']  # label letters
label_subplots(f, texts, x_offset=0.08, y_offset=0.0)

# Save figure for manuscript
savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_S_figure_5.pdf"
plt.savefig(savename, transparent=True, dpi=400)

plt.show()
