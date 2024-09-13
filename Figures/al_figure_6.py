""" Figure 6: Sampling model with simulation

 1. Prepare figure
 2. Schematic illustration of the sampling model
 3. Optimal model: Plot true and sampled posteriors
 4. Optimal model: Plot sampling trace
 5. Perseveration illustration: Plot true and sampled posteriors
 6. Perseveration illustration: Plot sampling trace
 7. Anchoring illustration: Plot true and sampled posteriors
 8. Anchoring illustration: Plot sampling trace
 9. Plot bucket bias, perseveration probability, and estimation errors
 10. Add subplot labels and save figure
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import seaborn as sns
import os
import scipy.stats as stats
from sampling.AlAgentVarsSampling import AgentVarsSampling
from sampling.AlAgentSampling import AlAgentSampling
from sampling.al_task_agent_int_sampling import task_agent_int_sampling
from al_plot_utils import cm2inch, latex_plt, plot_rec, plot_arrow, center_x, get_text_coords, \
    plot_centered_text, label_subplots, plot_sampling_results_row

# Update matplotlib to use Latex and to change some defaults
matplotlib = latex_plt(matplotlib)

# Get home directory
paths = os.getcwd()
path = paths.split(os.path.sep)
home_dir = path[1]

# -----------------
# 1. Prepare figure
# -----------------

# Figure size
fig_height = 12
fig_width = 15

# Image format
saveas = "pdf"  # pdf or png

# Turn interactive plotting mode on for debugger
plt.ion()

# Condition colors
condition_colors = ["#BBE1FA", "#3282B8", "#1B262C", "#e3f3fd", "#adcde2", "#babdbf"]

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))
f.canvas.draw()

# Create plot grid
gs_0 = gridspec.GridSpec(nrows=12, ncols=3, left=0.15, top=0.95, bottom=0.15, right=0.95, hspace=2)

# Y-label distance
ylabel_dist = -0.3

# -----------------------------------------------
# 2. Schematic illustration of the sampling model
# -----------------------------------------------

# Create subplot
gs_1 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_0[:3, :], hspace=0, wspace=0.5)
ax_1 = plt.Subplot(f, gs_1[0, :])
f.add_subplot(ax_1)
ax_1.axis('off')

# Compute subplot ratio
ll, ur = ax_1.get_position() * f.get_size_inches()  # get lower left and upper right of subplot
width, height = ur - ll  # difference
ratio = width / height  # ratio

# Determine fontsize for sampling-model plot
fontsize = 6

# Create four boxes for illustration
n_boxes = 4
box_size = 0.03
box_y0 = 0.7

# Compute distance between boxes
box_sum = n_boxes * box_size
whitespace = 1 - box_sum
dist = whitespace / (n_boxes + 1)

# List with text for schematic
printed_word = ["Starting Point", "Sample Chunk 1", "Sample Chunk 2", "Sample Chunk N"]
accept_text = ['If accepted:\nPerseveration', 'If rejected:\nAccept Starting Point',
               'If rejected:\nAccept Chunk 1', 'If rejected:\nAccept Chunk N-1']

# Arrow shrink parameter
shrink = 0.2

# Cycle over boxes
for i in range(0, n_boxes):

    # Plot current box
    cell_x0 = dist * (i + 1) + box_size * i  # x0-coordinate
    cell_y0, height = box_y0, box_size * ratio  # y0-coordinate and height
    ax_1 = plot_rec(ax_1, patches, cell_x0, box_size, cell_y0, height)

    # Plot corresponding text
    word_length, _, _ = get_text_coords(f, ax_1, cell_x0, cell_y0, printed_word[i], fontsize)
    ax_1.text(center_x(cell_x0, box_size, word_length), 1, printed_word[i])

    # Plot connecting arrows and text
    if i < n_boxes - 1:
        x1, y1 = box_size * (i + 1) + dist * (i + 1), cell_y0 + height / 2  # x and y coordinates
        ax_1 = plot_arrow(ax_1, x1, y1, x1 + dist, y1, shrink_a=shrink, shrink_b=shrink, arrow_style="<->")
        ax_1.text(x1 + dist / 2, box_y0 - 0.15, 'Compare\nAccuracy', horizontalalignment="center", fontsize=5)

    # Plot vertical arrow
    x1 = box_size * i + dist * (i + 1) + box_size / 2
    arrow_head = box_y0 - 0.3
    ax_1 = plot_arrow(ax_1, x1, box_y0, x1, box_y0 - 0.3, shrink_a=shrink, shrink_b=shrink)

    # Plot lower text
    text_y = arrow_head - 0.1  # y-coordinate
    plot_centered_text(f, ax_1, cell_x0 + box_size / 2, text_y, cell_x0 + box_size / 2, text_y, accept_text[i], 5,
                       c_type="other")

# --------------------------------------------------
# 3. Optimal model: Plot true and sampled posteriors
# --------------------------------------------------

# Create subplot
gs_2 = gridspec.GridSpecFromSubplotSpec(nrows=2, ncols=3, subplot_spec=gs_0[3:8, :], hspace=0, wspace=0.5)
ax_21 = plt.Subplot(f, gs_2[0, 0])
f.add_subplot(ax_21)
ax_21.axis('off')

# Agent-variables object
agent_vars = AgentVarsSampling()
agent_vars.model_sat = False  # no satisficing
agent_vars.burn_in = 0  # no burn-in
agent_vars.criterion = 0.0  # satisficing criterion
agent_vars.n_samples = 1000  # number of samples
agent_vars.mu_0 = 120  # prior belief
agent_vars.h = 0.0  # agent does not assume any changes
agent_vars.sample_std = 15  # standard deviation of samples from proposal distribution
# (for illustration here lower than usual)

# Simulated outcome
x_t = 150  # trial outcome

# Agent-object instance
agent = AlAgentSampling(agent_vars)
agent.x_t = x_t

#  Plot posterior distribution
x = np.arange(0, 300)  # outcome grid
prior = agent.compute_prior(x)
likelihood = agent.compute_likelihood(x)
posterior = agent.compute_posterior(prior, likelihood)
posterior = posterior / sum(posterior)
posterior_mean = np.sum(posterior * x)
ax_21.plot(posterior, color='k', alpha=0.4)

# Create data frame for simulation
data = {'subj_num': [1, 1], 'age_group': [1, 1], 'new_block': [1, 0], 'x_t': [x_t, x_t], 'mu_t': [x_t, x_t],
        'y_t': [0, 0], 'cond': ['main_noPush', 'main_noPush']}
df_subj = pd.DataFrame(data)

# Run sampling simulation
agent.reinitialize_agent(seed=1)
_ = task_agent_int_sampling(df_subj, agent, agent_vars, n_trials=2)

# Plot sample density
[n, bins] = np.histogram(agent.samples, bins=20)
density = stats.gaussian_kde(agent.samples)
ax_21.plot(bins, density(bins), color='k')

# Plot bin that is closest to the sample mean
index_min = np.argmin(abs(bins - np.mean(agent.samples)))
ax_21.plot(bins[index_min], density(bins[index_min]), 'o', color='k')
ax_21.vlines(bins[index_min], density(bins[index_min]) - 0.01, density(bins[index_min]) + 0.01, color='k')

# Adjust axes
x_range = 40
ax_21.set_xlim(posterior_mean - x_range, posterior_mean + x_range)
ax_21.set_ylim(-0.01, 0.07)

# Add text
ax_21.set_title("Optimal Sampling")

# Add custom legend
custom_lines = [Line2D([], [], color='k', marker='|', linestyle='None', markersize=5, markeredgewidth=1.5),
                Line2D([0], [0], marker='o', color='w', label='Scatter', markerfacecolor='k', markersize=5),
                Line2D([0], [0], color='k', lw=1)]
ax_21.legend(custom_lines, ['New Belief', 'Sample\nMean', 'Sample\nDistribution'],
             fontsize=5, loc='lower left', bbox_to_anchor=(0.7, 0.2))

# -------------------------------------
# 4. Optimal model: Plot sampling trace
# -------------------------------------

# Create subplot
ax_22 = plt.Subplot(f, gs_2[1, 0])
f.add_subplot(ax_22)

# Plot sampling trace
ax_22.plot(np.array(agent.samples), np.arange(len(agent.samples)), color="k", linewidth=1)
ax_22.set_xlim(posterior_mean - x_range, posterior_mean + x_range)

# Add axis labels
ax_22.set_xlabel("Sample Value")
ax_22.set_ylabel("Sample Number")
ax_22.yaxis.set_label_coords(ylabel_dist, 0.5)

# ---------------------------------------------------------------
# 7. Perseveration illustration: Plot true and sampled posteriors
# ---------------------------------------------------------------

# Create subplot
ax_23 = plt.Subplot(f, gs_2[0, 1])
f.add_subplot(ax_23)
ax_23.axis('off')

#  Plot posterior distribution
ax_23.plot(posterior, color='k', alpha=0.4)

# Agent-variables object
agent_vars = AgentVarsSampling()
agent_vars.model_sat = False  # no satisficing (bc here manually illustrated)
agent_vars.n_samples = 10  # number of samples
agent_vars.mu_0 = 120  # prior belief

# Agent-object instance
agent = AlAgentSampling(agent_vars)
agent.reinitialize_agent(seed=48)

# Run sampling simulation
df_data = task_agent_int_sampling(df_subj, agent, agent_vars, n_trials=2)

# Plot sample density
[_, bins] = np.histogram(agent.samples, bins=20)
density = stats.gaussian_kde(agent.samples)
ax_23.plot(bins, density(bins), color='k')

# Plot bin that is closest to the sample mean and the prior
index_min_prior = np.argmin(abs(bins - 120))
index_min = np.argmin(abs(bins - np.mean(agent.samples)))
ax_23.plot(bins[index_min], density(bins[index_min]), 'o', color='k')
ax_23.vlines(bins[index_min_prior], density(bins[index_min_prior]) - 0.01, density(bins[index_min_prior]) + 0.01,
             color='k')

# Adjust axes
ax_23.set_xlim(posterior_mean - x_range, posterior_mean + x_range)
ax_23.set_ylim(-0.01, 0.07)

# Add text
ax_23.text(115, 0.04, 'Prior', horizontalalignment='right', verticalalignment='center',
           clip_on=False, color='k', alpha=0.5, size=5)
ax_23.text(155, 0.04, 'New Belief =\nPrior Belief', horizontalalignment='center',
           verticalalignment='center', clip_on=False, color='k', alpha=0.5, size=5)
ax_23.set_title("Perseveration")

# Plot prior
ax_23.vlines(df_data['sim_z_t'][0], 0, max(posterior), color='k')

# --------------------------------------------------
# 4. Perseveration illustration: Plot sampling trace
# --------------------------------------------------

# Create subplot
ax_24 = plt.Subplot(f, gs_2[1, 1])
f.add_subplot(ax_24)

# Plot sampling trace
ax_24.plot(np.array(agent.samples), np.arange(len(agent.samples)), color="k", linewidth=1)
ax_24.set_xlim(posterior_mean - x_range, posterior_mean + x_range)
ax_24.set_yticks(np.arange(0, 10, step=3))

# Add axis labels
ax_24.set_xlabel("Sample Value")
ax_24.set_ylabel("Sample Number")
ax_24.yaxis.set_label_coords(ylabel_dist, 0.5)

# -----------------------------------------------------------
# 7. Anchoring illustration: Plot true and sampled posteriors
# -----------------------------------------------------------

# Create subplot
ax_25 = plt.Subplot(f, gs_2[0, 2])
f.add_subplot(ax_25)
ax_25.axis('off')

#  Plot posterior distribution
ax_25.plot(posterior, color='k', alpha=0.4)

# Agent-variables object
agent_vars = AgentVarsSampling()
agent_vars.model_sat = False  # no satisficing (bc here manually illustrated)
agent_vars.burn_in = 0  # no burn-in
agent_vars.criterion = 0.0  # satisficing criterion
agent_vars.n_samples = 20  # number of samples
agent_vars.mu_0 = 120  # prior belief
agent_vars.h = 0.0  # agent does not assume any changes
agent_vars.sample_std = 15  # 20 standard deviation of samples from proposal distribution. Chosen such that the

# Agent-object instance
agent = AlAgentSampling(agent_vars)
agent.reinitialize_agent(seed=19)

# Update data frame for simulation
df_subj.loc[1, 'y_t'] = -25

data = {'subj_num': [1, 1, 1], 'age_group': [1, 1, 1], 'new_block': [1, 0, 0], 'x_t': [x_t, x_t, x_t],
        'mu_t': [x_t, x_t, x_t], 'y_t': [0, -25, 0], 'cond': ['main_noPush', 'main_noPush', 'main_noPush']}
df_subj = pd.DataFrame(data)

# Run sampling simulation
df_data = task_agent_int_sampling(df_subj, agent, agent_vars, n_trials=3)

# Plot sample density
[_, bins] = np.histogram(agent.samples, bins=20)
density = stats.gaussian_kde(agent.samples)
ax_25.plot(bins, density(bins), color='k')

# Plot bin that is closest to the sample mean and the prior
index_min = np.argmin(abs(bins - np.mean(agent.samples)))
ax_25.plot(bins[index_min], density(bins[index_min]), 'o', color='k')
ax_25.vlines(bins[index_min], density(bins[index_min]) - 0.01, density(bins[index_min]) + 0.01, color='k')

# Adjust axes
ax_25.set_xlim(posterior_mean - x_range, posterior_mean + x_range)
ax_25.set_ylim(-0.010, 0.07)

# Add anchor line
ax_25.vlines(df_data['sim_z_t'][1], 0, max(posterior), color='k')

# Add text
ax_25.text(90, 0.04, 'Anchor', horizontalalignment='right',
           verticalalignment='center', clip_on=False, color='k', alpha=0.5, size=5)
ax_25.text(155, 0.04, 'New Belief =\nSample Mean', horizontalalignment='center',
           verticalalignment='center', clip_on=False, color='k', alpha=0.5, size=5)
ax_25.set_title("Anchoring")

# Plot mean of the distribution
ax_21.vlines(posterior_mean, 0, max(posterior), color="k", alpha=0.5)
ax_23.vlines(posterior_mean, 0, max(posterior), color="k", alpha=0.5)
ax_25.vlines(posterior_mean, 0, max(posterior), color="k", alpha=0.5)

# ----------------------------------------------
# 8. Anchoring illustration: Plot sampling trace
# ----------------------------------------------

# Create subplot
ax_26 = plt.Subplot(f, gs_2[1, 2])
f.add_subplot(ax_26)

# Plot sampling trace
ax_26.plot(np.array(agent.samples), np.arange(len(agent.samples)), color="k", linewidth=1)
ax_26.set_xlim(posterior_mean - x_range, posterior_mean + x_range)
ax_26.set_yticks(np.arange(0, 25, step=10))

# Add axis labels
ax_26.set_xlabel("Sample Value")
ax_26.set_ylabel("Sample Number")
ax_26.yaxis.set_label_coords(ylabel_dist, 0.5)

# ----------------------------------------------------------------------
#  9. Plot bucket bias, perseveration probability, and estimation errors
# ----------------------------------------------------------------------

# Create subplot
gs_3 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=3, subplot_spec=gs_0[9:12, :], hspace=0.5, wspace=0.5)

# Load data from "sampling/al_priorpred_sampling"
# -----------------------------------------------

# Perseveration: Bring into correct format for custom boxplot with both conditions
all_pers = pd.read_pickle('al_data/n_samples_different_all_pers.pkl')
pers_noPush = all_pers[all_pers['variable'] == "noPush"].reset_index(drop=True)
pers_push = all_pers[all_pers['variable'] == "push"].reset_index(drop=True)

# Estimation errors: Bring into correct format for custom boxplot with both conditions
all_est_errs = pd.read_pickle('al_data/n_samples_different_all_est_errs.pkl')

# Anchoring bias
df_reg = pd.read_pickle('al_data/n_samples_different_df_reg.pkl')

# Plot results
# ------------

plot_sampling_results_row(gs_3, f, pers_noPush, pers_push, all_est_errs, df_reg, condition_colors, ylabel_dist)

# --------------------------------------
# 10. Add subplot labels and save figure
# --------------------------------------

# Add labels
texts = ['a', 'b', '', 'c', '', 'd', '', 'e', 'f', 'g']
label_subplots(f, texts, x_offset=+0.1, y_offset=0.02)

# Adjust axes
sns.despine()

# Save figure
# -----------

if saveas == "pdf":
    savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_figure_6.pdf"
else:
    savename = "/" + home_dir + "/rasmus/Dropbox/heli_lifespan/png/al_figure_6.png"

plt.savefig(savename, transparent=False, dpi=400)

# Show plot
plt.ioff()
plt.show()
