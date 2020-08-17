""" Figure S5

1. Load data and compute cumulated BIC's
2. Prepare figure
3. Plot BIC and pEP
4. Add subplot labels and save figure
"""

import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import os
import matplotlib.pyplot as plt
from al_plot_utils import cm2inch, label_subplots, latex_plt


# Update matplotlib to use Latex and to change some defaults
os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2016/bin/x86_64-darwin'
matplotlib = latex_plt(matplotlib)

# Get home directory
paths = os.getcwd()
path = paths.split(os.path.sep)
home_dir = path[1]

# ----------------------------------------
# 1. Load data and compute cumulated BIC's
# ----------------------------------------

# model_1 = pd.read_pickle('al_data/model_exp_1_log_25sp.pkl')
model_1 = pd.read_pickle('al_data/estimates_first_exp_25_sp.pkl')
model_2 = pd.read_pickle('al_data/estimates_first_exp_no_pers_25_sp.pkl')

# Compute for how many participants perseveration model fit the data better
model_diff = model_1['BIC'] - model_2['BIC']
print(sum(model_diff > 0))  # in 124 out of 129, model with perseveration fit the data better.

# Save BIC as csv for Bayesian model comparison
# ---------------------------------------------
n_ch = sum(model_1['age_group'] == 1)
n_ad = sum(model_1['age_group'] == 2)
n_ya = sum(model_1['age_group'] == 3)
n_oa = sum(model_1['age_group'] == 4)

ch_bic_mat = np.full([n_ch, 2], np.nan)
ad_bic_mat = np.full([n_ad, 2], np.nan)
ya_bic_mat = np.full([n_ya, 2], np.nan)
oa_bic_mat = np.full([n_oa, 2], np.nan)

ch_bic_mat[:, 0] = model_1[model_1['age_group'] == 1]['BIC']
ch_bic_mat[:, 1] = model_2[model_2['age_group'] == 1]['BIC']
np.savetxt('al_data/ch_bic_mat.csv', ch_bic_mat, delimiter=',')

ad_bic_mat[:, 0] = model_1[model_1['age_group'] == 2]['BIC']
ad_bic_mat[:, 1] = model_2[model_2['age_group'] == 2]['BIC']
np.savetxt('al_data/ad_bic_mat.csv', ad_bic_mat, delimiter=',')

ya_bic_mat[:, 0] = model_1[model_1['age_group'] == 3]['BIC']
ya_bic_mat[:, 1] = model_2[model_2['age_group'] == 3]['BIC']
np.savetxt('al_data/ya_bic_mat.csv', ya_bic_mat, delimiter=',')

oa_bic_mat[:, 0] = model_1[model_1['age_group'] == 4]['BIC']
oa_bic_mat[:, 1] = model_2[model_2['age_group'] == 4]['BIC']
np.savetxt('al_data/oa_bic_mat.csv', oa_bic_mat, delimiter=',')

# Compute BIC sum for plotting
BIC_1_CH = sum(model_1[model_1['age_group'] == 1]['BIC'])/10000
BIC_1_AD = sum(model_1[model_1['age_group'] == 2]['BIC'])/10000
BIC_1_YA = sum(model_1[model_1['age_group'] == 3]['BIC'])/10000
BIC_1_OA = sum(model_1[model_1['age_group'] == 4]['BIC'])/10000

BIC_2_CH = sum(model_2[model_2['age_group'] == 1]['BIC'])/10000
BIC_2_AD = sum(model_2[model_2['age_group'] == 2]['BIC'])/10000
BIC_2_YA = sum(model_2[model_2['age_group'] == 3]['BIC'])/10000
BIC_2_OA = sum(model_2[model_2['age_group'] == 4]['BIC'])/10000

# ------------------
# 2. Prepare figure
# ------------------

# Plot colors
colors = ["#92e0a9", "#69b0c1", "#6d6192", "#352d4d"]
sns.set_palette(sns.color_palette(colors))

# Create figure
fig_height = 7.5
fig_width = 7.5
f = plt.figure(figsize=cm2inch(fig_width, fig_height))

BIC = pd.DataFrame(columns=['age_group', 'model', 'BIC'])
BIC['age_group'] = ['CH', 'AD', 'YA', 'OA', 'CH', 'AD', 'YA', 'OA']
BIC['Cumulated BIC'] = [BIC_1_CH, BIC_1_AD, BIC_1_YA, BIC_1_OA, BIC_2_CH, BIC_2_AD, BIC_2_YA, BIC_2_OA]
BIC['pEP'] = [1, 1, 1, 1, 0, 0, 0, 0]  # pEP are really that clear
BIC['Model'] = ['With perseveration', 'With perseveration', 'With perseveration', 'With perseveration',
                'Without perseveration', 'Without perseveration', 'Without perseveration', 'Without perseveration']

# -------------------
# 3. Plot BIC and pEP
# -------------------

# Create subplot for BIC
plt.subplot(211)
ax_0 = plt.gca()

# Plot BIC
sns.barplot(x='Model', hue='age_group', y='Cumulated BIC', data=BIC, ax=ax_0)

# Add y-axis text
f.text(0.1, 0.93, r'$\times 10^4$', size=8, rotation=0, color='k', ha="center", va="center")

# Remove legend
legend = ax_0.get_legend().remove()

# Create subplot for pEP
plt.subplot(212)
ax_1 = plt.gca()

# Plot pEP
sns.barplot(x='Model', hue='age_group', y='pEP', data=BIC, ax=ax_1)

# Adjust legend
handles, labels = ax_1.get_legend_handles_labels()
ax_1.legend(handles=handles[:], labels=labels[:])

# -------------------------------------
# 4. Add subplot labels and save figure
# -------------------------------------

# Adjust space and axes
plt.subplots_adjust(left=0.15, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
sns.despine()

texts = ['a', 'b']  # label letters
label_subplots(f, texts, x_offset=0.15, y_offset=0.0)

savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_S_figure_5.pdf"
plt.savefig(savename, transparent=True, dpi=400)

# Show figure
plt.show()
