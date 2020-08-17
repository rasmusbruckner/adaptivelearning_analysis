""" Figure S12

1. Load data
2. Differences of perseveration frequency between shifting- and stable bucket condition
3. Prepare figure
4. Plot robust linear regression of perseveration probability on bucket-shift parameter
5. Add subplot labels and save figure
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sm
import statsmodels.formula.api as smf
from al_utilities import get_mean_voi
from al_plot_utils import cm2inch, latex_plt


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

# Follow-up experiment
model_exp2 = pd.read_pickle('al_data/estimates_follow_up_exp_25_sp.pkl')

# ---------------------------------------------------------------------------------------
# 2. Differences of perseveration frequency between shifting- and stable bucket condition
# ---------------------------------------------------------------------------------------

# Read out push and noPush trials
df_noPush = df_exp2[df_exp2['cond'] == 'main_noPush']

# Perseveration in stable-bucket condition
voi = 2
pers_noPush = get_mean_voi(df_noPush, voi)

# -----------------
# 3. Prepare figure
# -----------------

# Adjust figure colors
colors = ["#92e0a9",  "#6d6192", "#352d4d"]
sns.set_palette(sns.color_palette(colors))

# Create figure
fig_height = 6
fig_witdh = 8
f = plt.figure(figsize=cm2inch(fig_witdh, fig_height))

# ---------------------------------------------------------------------------------------
# 4. Plot robust linear regression of perseveration probability on bucket-shift parameter
# ---------------------------------------------------------------------------------------

# Data frame for regerssion model
data = pd.DataFrame()
data['pers'] = pers_noPush['pers'].copy()
data['d'] = model_exp2['d'].copy()
data['age_group'] = pers_noPush['age_group'].copy()

# Recode age dummy variable in reverse order, i.e., with older adults as reference because they seem to
# have the strongest effect
data.loc[data['age_group'] == 3, 'age_group'] = 2  # YA in the middle
data.loc[data['age_group'] == 1, 'age_group'] = 3  # CH last variable
data.loc[data['age_group'] == 4, 'age_group'] = 1  # OA reference

# Robust linear regression
mod = smf.rlm(formula='d ~ pers + C(age_group, Treatment) + pers * C(age_group, Treatment)',
              M=sm.robust.norms.TukeyBiweight(3), data=data)
res = mod.fit(conv="weights")
print(res.summary())

# Plot results
plt.plot(pers_noPush[pers_noPush['age_group'] == 1]['pers'].copy(),
         model_exp2[pers_noPush['age_group'] == 1]['d'].copy(), '.', color=colors[0], alpha=1, markersize=5)
plt.plot(pers_noPush[pers_noPush['age_group'] == 3]['pers'].copy(),
         model_exp2[pers_noPush['age_group'] == 3]['d'].copy(), '.', color=colors[1], alpha=1, markersize=5)
plt.plot(pers_noPush[pers_noPush['age_group'] == 4]['pers'].copy(),
         model_exp2[pers_noPush['age_group'] == 4]['d'].copy(), '.', color=colors[2], alpha=1, markersize=5)
plt.plot(pers_noPush[pers_noPush['age_group'] == 1]['pers'].copy(),
         res.fittedvalues[pers_noPush['age_group'] == 1], '-', label="CH", color=colors[0])
plt.plot(pers_noPush[pers_noPush['age_group'] == 3]['pers'].copy(),
         res.fittedvalues[pers_noPush['age_group'] == 3], '-', label="YA", color=colors[1])
plt.plot(pers_noPush[pers_noPush['age_group'] == 4]['pers'].copy(),
         res.fittedvalues[pers_noPush['age_group'] == 4], '-', label="OA", color=colors[2])
plt.ylabel('Bucket-shift parameter estimate')
plt.xlabel('Estimated perseveration probability')
plt.legend()
plt.tight_layout()

# -------------------------------------
# 5. Add subplot labels and save figure
# -------------------------------------

# Delete unnecessary axes
sns.despine()

# Save figure
savename = "/" + home_dir + "/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_figures/al_S_figure_12.pdf"
plt.savefig(savename, transparent=True, dpi=400)

# Show plot
plt.show()
