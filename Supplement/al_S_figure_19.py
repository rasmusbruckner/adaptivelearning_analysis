""" Figure S19: Sampling model correlations between biases and parameters

    1. Load data
    2. Choose parameters
    3. Plot simulation
    4. Add subplot labels and save figure
"""

import numpy as np
import pandas as pd
from sampling.al_simulation_sampling import simulation_loop_sampling
import matplotlib.gridspec as gridspec
from al_utilities import compute_anchoring_bias, safe_save_dataframe
import matplotlib.pyplot as plt
from time import sleep
import seaborn as sns
import os
from al_plot_utils import cm2inch, latex_plt, label_subplots
import matplotlib
import random

# Set random number generator for reproducible results
# futuretodo: use only one random function
np.random.seed(123)
random.seed(123)

# Update matplotlib to use Latex and to change some defaults
matplotlib = latex_plt(matplotlib)

# Get home directory
paths = os.getcwd()
path = paths.split(os.path.sep)
home_dir = path[1]

# Should data be re-simulated?
sim_data = True

# ------------
# 1. Load data
# ------------

# Data of follow-up experiment
df_exp2 = pd.read_pickle('al_data/data_prepr_2.pkl')
n_subj = len(np.unique(df_exp2['subj_num']))

# Extract age group
df_grouped = df_exp2.groupby('subj_num')['age_group'].unique().str[0].reset_index()
age_group = df_grouped['age_group']

# --------------------
# 2. Choose parameters
# --------------------

# We will have six simulations: 1-3 will have fixed chunk size (n_samples) and 4-6 same criterion.
# It is important that we sample from a distribution, so below we will have a mean and a scale value for each case.

# Mean chunk size
n_samples = [2, 2, 2, 1, 3, 15]

# Scale of chunk size (gamma distribution)
n_samples_scale = [3, 3, 3, 3, 3, 3, 3, 3, 3]

# Mean of criterion
criterion = [0.02, 0.01, 0.001, 0.01, 0.01, 0.01]

# Scale of criterion (Gaussian distribution)
criterion_scale = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]

# Initialize all_sets variable
all_sets = np.nan

if sim_data:

    # Cycle over parameters
    for i in range(len(n_samples)):

        # Inform user
        sleep(0.1)
        print('\nSimulation: ' + str(i))
        sleep(0.1)

        # Create a new data frame with all relevant model parameters
        model_exp2 = pd.DataFrame(columns=['subj_num', 'age_group', 'criterion', 'n_samples'],
                                  index=[np.arange(n_subj)])
        model_exp2['subj_num'] = np.arange(n_subj) + 1
        model_exp2['age_group'] = np.array(age_group[:n_subj])

        # Select the current parameters
        if i < 3:
            all_n_samples = np.round(np.random.gamma(shape=n_samples[i], scale=n_samples_scale[i], size=n_subj), 0)
            all_criterion = np.repeat(criterion[i], n_subj)
        else:
            all_n_samples = np.repeat(n_samples[i], n_subj)
            all_criterion = np.random.normal(loc=criterion[i], scale=criterion_scale[i], size=n_subj)

        # Ensure parameters are not below minimum
        all_n_samples[all_n_samples < 1] = 1
        all_criterion[all_criterion < 0.001] = 0.001

        # Add parameters to data frame
        model_exp2.loc[:, 'criterion'] = all_criterion
        model_exp2.loc[:, 'n_samples'] = all_n_samples

        # Run simulation
        n_sim = 1
        sim_pers = True
        all_pers, all_est_errs, df_data = simulation_loop_sampling(df_exp2, model_exp2, n_subj, n_sim=n_sim)

        # Compute anchoring bias
        df_reg = compute_anchoring_bias(n_subj, df_data)

        # Put everything in simulation data frame
        df_sim = pd.DataFrame()
        df_sim['pers'] = all_pers[all_pers['variable'] == 'noPush']['value'].reset_index(drop=True)
        df_sim['d'] = df_reg['bucket_bias'].copy()
        df_sim['age_group'] = all_pers['age_group'].copy()
        df_sim['n_samples'] = all_n_samples.copy()
        df_sim['criterion'] = all_criterion.copy()
        df_sim['sim'] = i

        if i == 0:
            all_sets = df_sim.copy()
        elif i > 0:
            all_sets = pd.concat([all_sets, df_sim.copy()])

    all_sets.name = 'all_sets'
    safe_save_dataframe(all_sets, 'index', overleaf=False)

else:

    all_sets = pd.read_pickle('al_data/all_sets.pkl')

# ------------------
# 3. Plot simulation
# ------------------

# Figure size
fig_height = 15
fig_width = 10

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))

# Create plot grid
gs_0 = gridspec.GridSpec(3, 1, wspace=1, hspace=0.5)

row = [0, 1, 2]
column = [0, 0, 0]
n_samples_examples = n_samples[3:]

# Cycle over the 3 chunk-size examples
for i in range(len(n_samples_examples)):

    # Extract data of current simulations
    df_sim = all_sets[all_sets['sim'] == i + 3]

    # Create subplot
    gs_00 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_0[row[i], 0], wspace=0.5)
    ax_0 = plt.Subplot(f, gs_00[0, 0])
    f.add_subplot(ax_0)
    ax_1 = plt.Subplot(f, gs_00[0, 1])
    f.add_subplot(ax_1)

    # Plot regression-style plot
    sns.regplot(data=df_sim, x="criterion", y="pers", ax=ax_0, marker='o', color='k',
                scatter_kws={'s': 20, 'alpha': 0.2})
    sns.regplot(data=df_sim, x="criterion", y="d", ax=ax_1, marker='o', color='k',
                scatter_kws={'s': 20, 'alpha': 0.2})
    ax_0.set_title("Chunk Size = " + str(n_samples_examples[i]))
    ax_1.set_title("Chunk Size = " + str(n_samples_examples[i]))
    ax_0.set_xlabel("Criterion")
    ax_1.set_xlabel("Criterion")
    ax_0.set_ylabel("Perseveration Probability")
    ax_1.set_ylabel("Anchoring Bias")
    ax_0.set_ylim(-0.1, 1)
    ax_1.set_ylim(-0.1, 1)

# -------------------------------------
# 4. Add subplot labels and save figure
# -------------------------------------

# Deleted unnecessary axes
sns.despine()

# Add labels
texts = ['a', 'b', 'c', 'd', 'e', 'f']  # label letters
label_subplots(f, texts, x_offset=0.1)

savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_S_figure_19.pdf"
plt.savefig(savename, transparent=False, dpi=400)

# Show plot
plt.show()
