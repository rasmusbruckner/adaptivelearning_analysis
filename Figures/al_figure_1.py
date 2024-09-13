""" Figure 1: Illustration of sampling and satisficing idea

    1. Prepare figure
    2. Plot restaurant example
    3. Plot true and sampled posteriors
    4. Plot sampling traces
    5. Illustrate cost differences between age groups
    6. Illustrate sampling differences between age groups
    7. Add subplot labels and save figure
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import seaborn as sns
import scipy.stats as stats
import os
from sampling.AlAgentVarsSampling import AgentVarsSampling
from sampling.AlAgentSampling import AlAgentSampling
from al_plot_utils import cm2inch, label_subplots, latex_plt, plot_image

# Update matplotlib to use Latex and to change some defaults
matplotlib = latex_plt(matplotlib)
matplotlib.use('Qt5Agg')

# Get home directory
paths = os.getcwd()
path = paths.split(os.path.sep)
home_dir = path[1]

# -----------------
# 1. Prepare figure
# -----------------

# Figure size
fig_height = 6.5
fig_width = 14

# Image format
saveas = "pdf"

# Turn interactive plotting mode on for debugger
plt.ion()

# Distribution colors
dark_green = "#1E6F5C"
light_green = "#289672"
group_colors = ["#BBE1FA", "#0F4C75", "#1B262C"]

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))

# Create plot grid
gs_0 = gridspec.GridSpec(5, 4, left=0.05, top=0.9, bottom=0.2, right=0.95, hspace=0.0, wspace=0.25)

# --------------------------
# 2. Plot restaurant example
# --------------------------

# Create upper subplot
gs_00 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_0[0:2, 0])
ax_0 = plt.Subplot(f, gs_00[0, 0])
f.add_subplot(ax_0)
ax_0.set_aspect('1')  # x:y ratio = 1:1

# Plot restaurant icon
text_pos = 'above'
fontsize = 6
cell_x0 = 0.6
cell_x1 = 0.6
image_y = 0.0
path = 'al_figures/icons8-restaurantgebäude-100.png'
plot_image(f, path, cell_x0, cell_x1, image_y, ax_0, 0.1, '', 'above', fontsize, zoom=0.3)

# Add circle
restaurant_circle = plt.Circle((0.6, 0.0), 0.35, fill=False, clip_on=False)
ax_0.add_artist(restaurant_circle)

# Delete unnecessary axes
ax_0.axis('off')

# Create lower subplot
gs_00 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_0[2:, 0])
ax_1 = plt.Subplot(f, gs_00[0, 0])
f.add_subplot(ax_1)
ax_1.set_aspect('1')

# Plot thinking woman icon
cell_x0 = 0.0
cell_x1 = 0.0
image_y = -0.7
path = 'al_figures/icons8-woman-head-100.png'
plot_image(f, path, cell_x0, cell_x1, image_y, ax_1, 0.1, '', text_pos, fontsize, zoom=0.3)

# Add thinking bubbles
thinking_bubble = plt.Circle((0.5, -0.25), 0.075, fill=False, clip_on=False)
ax_1.add_artist(thinking_bubble)
thinking_bubble = plt.Circle((0.65, 0.05), 0.1, fill=False, clip_on=False)
ax_1.add_artist(thinking_bubble)

# Delete unnecessary axes and adjust y-axis
ax_1.axis('off')
ax_1.set_ylim(-1, 1)

# -----------------------------------
# 3. Plot true and sampled posteriors
# -----------------------------------

# Create subplot
gs_01 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_0[0:2, 1:3])
ax_2 = plt.Subplot(f, gs_01[0, 0])
f.add_subplot(ax_2)
ax_2.axis('off')

# Agent variables object
agent_vars = AgentVarsSampling()
agent_vars.model_sat = False  # no satisficing (bc here manually illustrated)
agent_vars.burn_in = 0  # no burn-in
agent_vars.criterion = 1.0  # satisficing criterion
agent_vars.n_samples = 1000  # number of samples
agent_vars.mu_0 = 120  # prior belief
agent_vars.h = 0.0  # agent does not assume any changes

# Simulated outcome
x_t = 150

# Agent-object instance
agent = AlAgentSampling(agent_vars)
agent.sample_curr = agent_vars.mu_0
agent.x_t = x_t

#  Plot posterior distribution
x = np.arange(0, 300)  # outcome grid
prior = agent.compute_prior(x)
likelihood = agent.compute_likelihood(x)
posterior = agent.compute_posterior(prior, likelihood)
posterior = posterior / sum(posterior)
posterior_mean = np.sum(posterior * x)
ax_2.plot(posterior, color='k', alpha=0.4)
ax_2.vlines(posterior_mean, 0, max(posterior), 'k', alpha=0.5)

# Create data frame for sampling simulation
data = [{'subj_num': 1, 'age_group': 1, 'x_t': x_t, 'mu_t': x_t, 'y_t': 0, 'cond': 'main_noPush'}]
df_subj = pd.DataFrame(data)

# Manually define the samples for illustration
samples = [[125, 115, 130, 119, 120, 121, 129, 115, 111, 118],  # perseveration samples
           [160, 155, 149, 149, 152, 145, 142, 150, 140, 140, 135, 132, 139, 135, 133]]  # anchoring samples

# Initialize variables
bins = []
index_min = []
index_min_prior = []
density = []

# Colors of distributions
dist_colors = [dark_green, light_green]

# Cycle over perseveration and anchoring samples
for i in range(2):

    # Plot sample density
    [_, b] = np.histogram(samples[i], bins=20)
    bins.append(b)
    density.append(stats.gaussian_kde(samples[i]))
    ax_2.plot(bins[i], density[i](bins[i]), color=dist_colors[i])

    # Identify bin that is closest to the sample mean and the prior
    index_min.append(np.argmin(abs(bins[i] - np.mean(samples[i]))))
    index_min_prior.append(np.argmin(abs(bins[i] - 125)))

    # Plot bin representing sample mean
    ax_2.plot(bins[i][index_min[i]], density[i](bins[i][index_min[i]]), 'o', color='k')

# Plot bins representing new belief
ax_2.vlines(bins[0][index_min_prior[0]], density[0](bins[0][index_min_prior[0]]) - 0.01,  # perseveration: prior
            density[0](bins[0][index_min_prior[0]]) + 0.01, color='k')
ax_2.vlines(bins[1][index_min[1]], density[1](bins[1][index_min[1]]) - 0.01,  # anchoring: sample mean
            density[1](bins[1][index_min[1]]) + 0.01, color='k')

# Plot anchor
ax_2.vlines(bins[1][-1], density[1](bins[1][-1]) - 0.01, density[1](bins[1][-1]) + 0.01, color=light_green)

# Plot arrows that show the comparison between the hypothetical beliefs
# ---------------------------------------------------------------------
x1 = bins[0][index_min[0]]
x2 = bins[0][index_min_prior[0]]
delta = x2 - x1
fraction = 0.8
arrow_height = delta * fraction
ax_2.annotate("", xy=(x1, 1.25 * max(posterior)), xycoords='data',
              xytext=(x2, 1.25 * max(posterior)), textcoords='data',
              arrowprops=dict(arrowstyle="<->", color="0.5", connectionstyle="bar, fraction="+str(fraction), shrinkA=1,
              shrinkB=1))

x1 = bins[1][index_min[1]]
x2 = 160
delta = x2 - x1
fraction = arrow_height/delta
ax_2.annotate("", xy=(x1, 1.025 * max(posterior)), xycoords='data',
              xytext=(x2, 1.025 * max(posterior)), textcoords='data', arrowprops=dict(arrowstyle="<->",
              color="0.5", connectionstyle="bar, fraction="+str(fraction), shrinkA=1, shrinkB=1))
# Adjust axes
ax_2.set_xlim(90, 165)
ax_2.set_ylim(-0.01, 1.5 * max(posterior))

# Add text
ax_2.text(90, 0.1 * max(posterior), 'Optimal\nBelief Dist.', horizontalalignment='left', verticalalignment='bottom',
          clip_on=False, color='k', alpha=0.5, size=5)
ax_2.text(105, 1.2 * max(posterior), 'Prior belief better\nthan sample\nmean', horizontalalignment='center',
          verticalalignment='center', clip_on=False, color='k', alpha=0.5, size=5)
ax_2.text((bins[1][index_min[1]] + 160) / 2, 1.6 * max(posterior), 'Sample mean better\n than anchor',
          horizontalalignment='center', verticalalignment='center', clip_on=False, color='k', alpha=0.5, size=5)
ax_2.text(161, 0.1 * max(posterior), 'Anchor', horizontalalignment='left', verticalalignment='bottom',
          clip_on=False, color=light_green, alpha=1, size=5)

# Add custom legend
custom_lines = [Line2D([], [], color='k', marker='|', linestyle='None', markersize=5, markeredgewidth=1.5),
                Line2D([0], [0], marker='o', color='w', label='Scatter', markerfacecolor='k', markersize=5),
                Line2D([0], [0], color='k', lw=1)]
ax_2.legend(custom_lines, ['New Belief', 'Sample\nMean', 'Sample\nDistribution'], fontsize=5,
            loc='lower left', ncol=2, bbox_to_anchor=(1, 0.4))

# -----------------------
# 4. Plot sampling traces
# -----------------------

# Create subplot
gs_02 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_0[2:, 1:3], wspace=2, hspace=0.5)
ax_3 = plt.Subplot(f, gs_02[0, 0])
f.add_subplot(ax_3)

# Match axes to upper plot
ax_3.set_xlim(90, 165)
ax_3.set_ylim(-1, 15)
sns.despine()

# Plot sampling traces
ax_3.plot(np.array(samples[0]), np.arange(len(samples[0])), color=dark_green, linewidth=1)
ax_3.plot(np.array(samples[1]), np.arange(len(samples[1])), color=light_green, linewidth=1)

# Plot satisficing thresholds
ax_3.hlines(10, 90, 125, linestyle='--', color='k', alpha=0.5, linewidth=0.5, clip_on=False)
ax_3.hlines(15, 125, 165, linestyle='--', color='k', alpha=0.5, linewidth=0.5, clip_on=False)

# Add arrow to y-axis
ax_3.arrow(90, 14, 0, 0, head_width=2, head_length=1, fc='k', ec='k', clip_on=False)

# Remove tick parameters
ax_3.tick_params(
    axis='both',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom=False,
    left=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    labelbottom=False,
    labelleft=False)  # labels along the bottom edge are off

# Add text
pers_text = r"$\bf{Perseveration}$" + ": Sampling\nstarts from prior but\nsamples are rejected "
ax_3.text(110, -3.5, pers_text, horizontalalignment='center', verticalalignment='top', clip_on=False, color='k', size=5)
anchoring_text = r"$\bf{Anchoring}$" + ": Sampling\nstarts from anchor and\nsamples are accepted "
ax_3.text(145, -3.5, anchoring_text, horizontalalignment='center', verticalalignment='top', clip_on=False, size=5)
ax_3.text(91, 10, 'Stopping Criterion', horizontalalignment='left', verticalalignment='bottom', clip_on=False, size=5)
ax_3.set_ylabel("Number of Samples")
ax_3.set_xlabel("Restaurant Quality")
ax_3.text(95, 0, 'Perseveration\nExample', horizontalalignment='left', verticalalignment='bottom', clip_on=False,
          size=5, color=dark_green)
ax_3.text(153, 5, 'Anchoring\nExample', horizontalalignment='left', verticalalignment='bottom', clip_on=False, size=5,
          color=light_green)

# -----------------------------------------------------
# 5. Illustrate cost differences between age groups
# -----------------------------------------------------

# Create subplot
gs_03 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_0[2:, 3], wspace=0.5)
ax_4 = plt.Subplot(f, gs_03[0, 0])
f.add_subplot(ax_4)

# Add bars
ax_4.bar(np.arange(3), [15, 8, 15], edgecolor=group_colors,  # nach oben?
         color=group_colors, alpha=1)
ax_4.set_xticks([0, 1, 2])
ax_4.set_xticks(np.arange(3), ['CH', 'YA', 'OA'], rotation=0)
ax_4.set_xlabel('Age Group')
ax_4.set_title("Sampling\nCost", y=1.0)

# Remove tick parameters
ax_4.tick_params(
    axis='y',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom=False,
    left=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    labelbottom=False,
    labelleft=False)  # labels along the bottom edge are off

# -----------------------------------------------------
# 6. Illustrate sampling differences between age groups
# -----------------------------------------------------

# Create subplot
ax_5 = plt.Subplot(f, gs_03[0, 1])
f.add_subplot(ax_5)

# Add bars
ax_5.bar(np.arange(3), [8, 15, 8], edgecolor=group_colors,
         color=group_colors, alpha=1)
ax_5.set_xticks([0, 1, 2])
ax_5.set_xticks([0, 1, 2])
ax_5.set_xticks(np.arange(3), ['CH', 'YA', 'OA'], rotation=0)
ax_5.set_xlabel('Age Group')
ax_5.set_title("Number of\nSamples", y=1.01)  # with 1.0, titles were slightly misaligned

# Remove tick parameters
ax_5.tick_params(
    axis='y',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom=False,
    left=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    labelbottom=False,
    labelleft=False)  # labels along the bottom edge are off

# Delete unnecessary axes
sns.despine()

# -------------------------------------
# 7. Add subplot labels and save figure
# -------------------------------------

# Add labels
texts = ['a', '', 'b', '', 'c', 'd']
x_offset = [0.06, np.nan, 0.06, np.nan, 0.03, 0.03]
y_offset = [-0.1, np.nan, -0.1, np.nan, -0.05, -0.05]
label_subplots(f, texts, x_offset=x_offset, y_offset=y_offset)

# Save figure
# -----------

if saveas == "pdf":
    savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_figure_1.pdf"
else:
    savename = "/" + home_dir + "/rasmus/Dropbox/heli_lifespan/png/al_figure_1.png"

plt.savefig(savename, transparent=False, dpi=400)

# Show plot
plt.ioff()
plt.show()
