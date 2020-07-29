# This script (1) contains a function to preprocess adaptive learning BIDS data
# and (2) applies the preprocessing to data of Experiment 1 and 2

import numpy as np
from al_utilities import load_data, sorted_nicely, get_file_paths
import pandas as pd


def preprocessing(exp):
    """ Data preprocessing

    This function loads and preprocesses the adaptive learning BIDS data for further analyses.

        :param exp: Current experiment of which data will be preprocessed

    """

    # Data folder
    if exp == 1:
        folder_path = 'al_data/first_experiment'
    else:
        folder_path = 'al_data/follow_up_experiment'

    # Get all file names
    identifier = "*behav.tsv"
    file_paths = get_file_paths(folder_path, identifier)

    # Sort all file names according to participant ID
    file_paths = sorted_nicely(file_paths)

    # Load pseudonomized BIDS data
    data_pn = load_data(file_paths)

    # In participant 40, we used higher rewards, which was subsequently adjusted to avoid too high boni -
    # here we adjust it to adjust it to the rest of the data
    if exp == 2:
        data_pn.loc[(data_pn.loc[:, 'subj_num'] == 40) & (data_pn.loc[:, 'r_t'] == 1), 'r_t'] = 0.25
        data_pn.loc[(data_pn.loc[:, 'subj_num'] == 40) & (data_pn.loc[:, 'r_t'] == 4), 'r_t'] = 1

    # -----------------------------
    # Compute variables of interest
    # -----------------------------

    # Extract relevant variables
    x_t = data_pn['x_t'].values  # outcomes
    mu_t = data_pn['mu_t'].values  # mean of outcome-generating distribution (helicopter location)
    sigma = data_pn['sigma'].values  # standard deviations of outcome-generating distribution (wind)
    b_t = data_pn['b_t'].values  # participant predictions
    new_block = data_pn['new_block'].values  # block change indicator

    # Recode last entry in variables of interest of each block to nan
    # to avoid that data between different participants are mixed
    to_nan = np.zeros(len(new_block))
    to_nan[:-1] = new_block[1:]
    to_nan[-1] = 1  # manually added, because no new block after last trial

    # General variables
    # -----------------

    # Prediction error: delta_t := x_t - b_t
    delta_t = x_t.copy().astype(float) - b_t.copy().astype(float)
    delta_t[to_nan == 1] = np.nan

    # Estimation error: e_t := |mu_t - b_t|
    e_t = abs(mu_t.copy().astype(float) - b_t.copy().astype(float))

    # Update: a_t := b_{t+1} - b_t
    # I.e., at trial t, we analyze the difference between belief at t and t+1
    a_t = np.full(len(data_pn), np.nan)
    a_t[:-1] = b_t[1:].copy().astype(float) - b_t[:-1].copy().astype(float)
    a_t[to_nan == 1] = np.nan

    # Perseveration: pers := 1, if a_t=0; 0, else
    pers = a_t == 0

    # Experiment 2 specific: shift in initial bucket location
    # -------------------------------------------------------

    if exp == 2:

        # Initial bucket location
        z_t = data_pn['z_t'][0:].values

        # In the task code, we used a Gaussian distribution which did not take into account the edges of the screen
        # while saving the location. However, subjects saw buckets between 0 and 300. Here we therefore correct for
        # the edge
        z_t[z_t > 300] = 300
        z_t[z_t < 0] = 0

        # Read out edge trials, i.e., trials in which bucket is at the very left or right
        edge = np.full(len(z_t), False)
        edge[z_t == 300] = True
        edge[z_t == 0] = True

        # Bucket shift: y_t := z_{t+1} - b_t
        # I.e., for influence of bucket on update (a_t := b_{t+1} - b_t), we consider difference between prediction b_t
        # and shifted position at trial t+1 (z_{t+1}).
        y_t = np.full(len(delta_t), np.nan)
        y_t[:-1] = z_t[1:].copy().astype(float) - b_t[:-1].copy().astype(float)
        y_t[to_nan == 1] = np.nan

        # Compute motor perseveration, i.e., prediction at shifted bucket location
        motor_pers = b_t == z_t

    else:

        # Set all edge trials to False in first experiment
        edge = np.full(len(pers), False)

    # Preprocessing for additional analyses not reported in the paper
    # ---------------------------------------------------------------

    # Mean error
    me = mu_t.copy().astype(float) - b_t.copy().astype(float)
    me[to_nan == 1] = np.nan

    # Squared center error
    # ce = 150 - b_t
    # ce_sqrd = np.sign(ce) * ce ** 2

    # Noise dummy variables
    sigma_dummy = np.full(len(b_t), np.nan)
    sigma_dummy[sigma == 10] = -0.5
    sigma_dummy[sigma == 25] = 0.5
    sigma_dummy[sigma == 17.5] = 0.0

    # Save computed variables
    # -----------------------
    data_pn['a_t'] = a_t
    data_pn['delta_t'] = delta_t
    # data_pn['me'] = me
    data_pn['e_t'] = e_t
    data_pn['pers'] = pers
    data_pn['edge'] = edge

    # data_pn['ce_sqrd'] = ce_sqrd
    # data_pn['sigma_dummy'] = sigma_dummy

    if exp == 2:
        data_pn['z_t'] = z_t
        data_pn['y_t'] = y_t
        data_pn['motor_pers'] = motor_pers

    if exp == 1:
        data_pn.to_pickle('al_data/data_prepr_1.pkl')  # data for experiment 1
    else:
        data_pn.to_pickle('al_data/data_prepr_2.pkl')  # data for experiment 1


# Preprocessing of first experiment
# ---------------------------------
preprocessing(1)

# Get participant information from BIDS metadata .json file
file_path = get_file_paths('al_data/first_experiment', "participants.tsv")
exp_1_participants = pd.read_csv(file_path[0], sep='\t', header=0)

# Preprocessing of follow-up experiment
# -------------------------------------
preprocessing(2)

# Get participant information from BIDS metadata .json file
file_path = get_file_paths('al_data/follow_up_experiment', "participants.tsv")
exp_2_participants = pd.read_csv(file_path[0], sep='\t', header=0)

# Put participant statistics in data frames for Latex
# ----------------------------------------------------

# Initialize data frame
df_participants = pd.DataFrame(index=[1, 2], columns=['min_age_ch', 'min_age_ad', 'min_age_ya', 'min_age_oa',
                                                      'max_age_ch', 'max_age_ad', 'max_age_ya', 'max_age_oa',
                                                      'median_age_ch', 'median_age_ad', 'median_age_ya',
                                                      'median_age_oa', 'n_ch', 'n_ad', 'n_ya', 'n_oa',
                                                      'n_female_ch', 'n_female_ad', 'n_female_ya', 'n_female_oa'])

# First experiment: minimum age
df_participants.loc[1, 'min_age_ch'] = exp_1_participants[exp_1_participants['age_group'] == 1]['age'].min()
df_participants.loc[1, 'min_age_ad'] = exp_1_participants[exp_1_participants['age_group'] == 2]['age'].min()
df_participants.loc[1, 'min_age_ya'] = exp_1_participants[exp_1_participants['age_group'] == 3]['age'].min()
df_participants.loc[1, 'min_age_oa'] = exp_1_participants[exp_1_participants['age_group'] == 4]['age'].min()

# First experiment: maximum age
df_participants.loc[1, 'max_age_ch'] = exp_1_participants[exp_1_participants['age_group'] == 1]['age'].max()
df_participants.loc[1, 'max_age_ad'] = exp_1_participants[exp_1_participants['age_group'] == 2]['age'].max()
df_participants.loc[1, 'max_age_ya'] = exp_1_participants[exp_1_participants['age_group'] == 3]['age'].max()
df_participants.loc[1, 'max_age_oa'] = exp_1_participants[exp_1_participants['age_group'] == 4]['age'].max()

# First experiment: median age
df_participants.loc[1, 'median_age_ch'] = \
    np.int(exp_1_participants[exp_1_participants['age_group'] == 1]['age'].median())
df_participants.loc[1, 'median_age_ad'] = \
    np.int(exp_1_participants[exp_1_participants['age_group'] == 2]['age'].median())
df_participants.loc[1, 'median_age_ya'] =\
    np.int(exp_1_participants[exp_1_participants['age_group'] == 3]['age'].median())
df_participants.loc[1, 'median_age_oa'] = \
    np.int(exp_1_participants[exp_1_participants['age_group'] == 4]['age'].median())

# First experiment: number of subject
df_participants.loc[1, 'n_ch'] = len(exp_1_participants[(exp_1_participants['age_group'] == 1)])
df_participants.loc[1, 'n_ad'] = len(exp_1_participants[(exp_1_participants['age_group'] == 2)])
df_participants.loc[1, 'n_ya'] = len(exp_1_participants[(exp_1_participants['age_group'] == 3)])
df_participants.loc[1, 'n_oa'] = len(exp_1_participants[(exp_1_participants['age_group'] == 4)])

# First experiment: number of females
df_participants.loc[1, 'n_female_ch'] = len(exp_1_participants[(exp_1_participants['age_group'] == 1)
                                                               & (exp_1_participants['sex'] == 'female')])
df_participants.loc[1, 'n_female_ad'] = len(exp_1_participants[(exp_1_participants['age_group'] == 2)
                                                               & (exp_1_participants['sex'] == 'female')])
df_participants.loc[1, 'n_female_ya'] = len(exp_1_participants[(exp_1_participants['age_group'] == 3)
                                                               & (exp_1_participants['sex'] == 'female')])
df_participants.loc[1, 'n_female_oa'] = len(exp_1_participants[(exp_1_participants['age_group'] == 4)
                                                               & (exp_1_participants['sex'] == 'female')])

# Follow-up experiment: minimum age
df_participants.loc[2, 'min_age_ch'] = exp_2_participants[exp_2_participants['age_group'] == 1]['age'].min()
df_participants.loc[2, 'min_age_ya'] = exp_2_participants[exp_2_participants['age_group'] == 3]['age'].min()
df_participants.loc[2, 'min_age_oa'] = exp_2_participants[exp_2_participants['age_group'] == 4]['age'].min()

# Follow-up experiment: maximum age
df_participants.loc[2, 'max_age_ch'] = exp_2_participants[exp_2_participants['age_group'] == 1]['age'].max()
df_participants.loc[2, 'max_age_ya'] = exp_2_participants[exp_2_participants['age_group'] == 3]['age'].max()
df_participants.loc[2, 'max_age_oa'] = exp_2_participants[exp_2_participants['age_group'] == 4]['age'].max()

# Follow-up experiment: median age
df_participants.loc[2, 'median_age_ch'] = \
    np.int(exp_2_participants[exp_2_participants['age_group'] == 1]['age'].median())
df_participants.loc[2, 'median_age_ya'] = \
    np.int(exp_2_participants[exp_2_participants['age_group'] == 3]['age'].median())
df_participants.loc[2, 'median_age_oa'] = \
    np.int(exp_2_participants[exp_2_participants['age_group'] == 4]['age'].median())

# Follow-up experiment: number of subject
df_participants.loc[2, 'n_ch'] = len(exp_2_participants[(exp_2_participants['age_group'] == 1)])
df_participants.loc[2, 'n_ya'] = len(exp_2_participants[(exp_2_participants['age_group'] == 3)])
df_participants.loc[2, 'n_oa'] = len(exp_2_participants[(exp_2_participants['age_group'] == 4)])

# Follow-up experiment: number of females
df_participants.loc[2, 'n_female_ch'] = len(exp_2_participants[(exp_2_participants['age_group'] == 1)
                                                               & (exp_2_participants['sex'] == 'female')])
df_participants.loc[2, 'n_female_ya'] = len(exp_2_participants[(exp_2_participants['age_group'] == 3)
                                                               & (exp_2_participants['sex'] == 'female')])
df_participants.loc[2, 'n_female_oa'] = len(exp_2_participants[(exp_2_participants['age_group'] == 4)
                                                               & (exp_2_participants['sex'] == 'female')])

# Give index name "exp"
df_participants.index.name = 'exp'

# Save data frame to Latex folder
df_participants.to_csv('~/Dropbox/Apps/Overleaf/al_manuscript/al_dataframes/participants.csv')
