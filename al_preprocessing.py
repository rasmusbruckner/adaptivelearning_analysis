""" Data Preprocessing: This script (1) contains a function to preprocess adaptive learning BIDS data
    and (2) applies the preprocessing to data of Experiment 1, 2 and 3 """

import numpy as np
from al_utilities import load_data, sorted_nicely, get_file_paths, get_sub_stats, safe_save_dataframe
import pandas as pd
import sys


def preprocessing(exp):
    """ This function loads and preprocesses the adaptive learning BIDS data for further analyses

        futuretodo: - create function for key parameters such as prediction errors and run unit tests on them
                    - run integration test based on data of a subject

        :param exp: Current experiment for which data will be preprocessed
    """

    # Data folder
    if exp == 1:
        folder_path = 'al_data/first_experiment'
    elif exp == 2:
        folder_path = 'al_data/follow_up_experiment'
    else:
        folder_path = 'al_data/third_experiment'

    # Get all file names
    identifier = "*behav.tsv"
    file_paths = get_file_paths(folder_path, identifier)

    # Sort all file names according to participant ID
    file_paths = sorted_nicely(file_paths)

    # Load pseudonymized BIDS data
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
    new_block = data_pn['new_block'].values  # block change indicator
    b_t = data_pn['b_t'].values  # participant predictions

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
        z_t = data_pn['z_t'].values

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

        # Initiation and reaction time:
        # We take values 1: instead of 0: This is analogous to the updates, where update
        # on trial 0 indicates update from 0 to 1. For IT and RT we do the same. E.g.,
        # rt_0 reflects update RT from trial 0 to 1.

        # Copy data frame to shift them up
        init_rt = data_pn.loc[:, 'init_rt'].copy()
        rt = data_pn.loc[:, 'rt'].copy()
        init_rt_shifted = data_pn.loc[:, 'init_rt'].copy()
        rt_shifted = data_pn.loc[:, 'rt'].copy()

        # Take shifted data
        init_rt_shifted[:-1] = init_rt[1:].values
        rt_shifted[:-1] = rt[1:].values

        # Deal with initiation reaction times that are = nan when update = 0:
        # In this case, RT = IT
        init_rt_shifted[np.isnan(init_rt_shifted)] = rt_shifted[np.isnan(init_rt_shifted)]

        # Set some values to nan, consistent with other cases
        init_rt_shifted[to_nan == 1] = np.nan
        rt_shifted[to_nan == 1] = np.nan

        # Take new values
        data_pn.loc[:, 'init_rt'] = init_rt_shifted
        data_pn.loc[:, 'rt'] = rt_shifted

    elif exp == 3:

        # Compute prediction-error bin for perseveration analysis
        pe_bin = np.full(len(delta_t), np.nan)
        delta_quantile_1 = np.nanquantile(abs(delta_t), .33)
        delta_quantile_2 = np.nanquantile(abs(delta_t), .66)
        quantile_1 = abs(delta_t) <= delta_quantile_1
        quantile_2 = np.logical_and(abs(delta_t) >= delta_quantile_1, abs(delta_t) <= delta_quantile_2)
        quantile_3 = abs(delta_t) > delta_quantile_2
        pe_bin[quantile_1] = 1
        pe_bin[quantile_2] = 2
        pe_bin[quantile_3] = 3

        # Initial bucket location
        z_t = data_pn['z_t'].values

        # In online heli, edges should be taken into account but nevertheless correcting for it just to be safe
        # the edge
        #z_t[z_t > 100] = 100
        z_t[z_t > 300] = 300
        z_t[z_t < 0] = 0

        # Read out edge trials, i.e., trials in which bucket is at the very left or right
        edge = np.full(len(z_t), False)
        #edge[z_t == 100] = True
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

    # Save computed variables
    # -----------------------
    data_pn['a_t'] = a_t
    data_pn['delta_t'] = delta_t
    data_pn['e_t'] = e_t
    data_pn['pers'] = pers

    if exp == 1:
        data_pn['edge'] = edge
    if exp == 2:
        data_pn['z_t'] = z_t
        data_pn['y_t'] = y_t
        data_pn['motor_pers'] = motor_pers
        data_pn['edge'] = edge
    elif exp == 3:
        data_pn['z_t'] = z_t
        data_pn['y_t'] = y_t
        data_pn['motor_pers'] = motor_pers
        data_pn['edge'] = edge
        data_pn['pe_bin'] = pe_bin

    all_id = list(set(data_pn['subj_num']))  # ID for each participant
    n_subj = len(all_id)  # number of participants

    # Test if expected values appear in preprocessed data frames
    for i in range(n_subj):

        df_subj = data_pn[(data_pn['subj_num'] == i + 1)].copy()

        if exp == 1:
            if not np.sum(np.isnan(df_subj['delta_t'])) == 2:
                sys.exit("Unexpected NaN's in delta_t")
            if not np.sum(np.isnan(df_subj['a_t'])) == 2:
                sys.exit("Unexpected NaN's in a_t")
            if not np.sum(df_subj['new_block']) == 2:
                sys.exit("Unexpected NaN's in new_block")
        elif exp == 2:
            if not np.sum(np.isnan(df_subj['delta_t'])) == 4:
                sys.exit("Unexpected NaN's in delta_t")
            if not np.sum(np.isnan(df_subj['a_t'])) == 4:
                sys.exit("Unexpected NaN's in a_t")
            if not np.sum(df_subj['new_block']) == 4:
                sys.exit("Unexpected NaN's in new_block")
            if not np.sum(np.isnan(df_subj['y_t'])) == 4:
                sys.exit("Unexpected NaN's in y_t")
            if not np.sum(np.isnan(df_subj['init_rt'])) == 4:
                sys.exit("Unexpected NaN's in init_rt")
            if not np.sum(np.isnan(df_subj['rt'])) == 4:
                sys.exit("Unexpected NaN's in rt")
            if not np.sum(np.isnan(df_subj['z_t'])) == 0:
                sys.exit("Unexpected NaN's in z_t")
            if not np.sum(np.isnan(df_subj['motor_pers'])) == 0:
                sys.exit("Unexpected NaN's in motor_pers")
        elif exp == 3:
            if not np.sum(np.isnan(df_subj['delta_t'])) == 4:
                sys.exit("Unexpected NaN's in delta_t")
            if not np.sum(np.isnan(df_subj['a_t'])) == 4:
                sys.exit("Unexpected NaN's in a_t")
            if not np.sum(df_subj['new_block']) == 4:
                sys.exit("Unexpected NaN's in new_block")
            if not np.sum(np.isnan(df_subj['y_t'])) == 4:
                sys.exit("Unexpected NaN's in y_t")
            if not np.sum(np.isnan(df_subj['z_t'])) == 0:
                sys.exit("Unexpected NaN's in z_t")
            if not np.sum(np.isnan(df_subj['motor_pers'])) == 0:
                sys.exit("Unexpected NaN's in motor_pers")

        if exp == 1 or exp == 2:
            if not np.sum(np.isnan(df_subj['subj_num'])) == 0:
                sys.exit("Unexpected NaN's in subj_num")
            if not np.sum(np.isnan(df_subj['age_group'])) == 0:
                sys.exit("Unexpected NaN's in age_group")
            if not np.sum(np.isnan(df_subj['x_t'])) == 0:
                sys.exit("Unexpected NaN's in x_t")
            if not np.sum(np.isnan(df_subj['b_t'])) == 0:
                sys.exit("Unexpected NaN's in b_t")
            if not np.sum(np.isnan(df_subj['mu_t'])) == 0:
                sys.exit("Unexpected NaN's in mu_t")
            if not np.sum(np.isnan(df_subj['c_t'])) == 0:
                sys.exit("Unexpected NaN's in c_t")
            if not np.sum(np.isnan(df_subj['r_t'])) == 0:
                sys.exit("Unexpected NaN's in r_t")
            if not np.sum(np.isnan(df_subj['sigma'])) == 0:
                sys.exit("Unexpected NaN's in sigma")
            if not np.sum(np.isnan(df_subj['v_t'])) == 0:
                sys.exit("Unexpected NaN's in v_t")
            if not np.sum(np.isnan(df_subj['e_t'])) == 0:
                sys.exit("Unexpected NaN's in e_t")
            if not np.sum(np.isnan(df_subj['pers'])) == 0:
                sys.exit("Unexpected NaN's in pers")
            if not np.sum(np.isnan(df_subj['edge'])) == 0:
                sys.exit("Unexpected NaN's in edge")
        else:
            if not np.sum(np.isnan(df_subj['subj_num'])) == 0:
                sys.exit("Unexpected NaN's in subj_num")
            if not np.sum(np.isnan(df_subj['age_group'])) == 0:
                sys.exit("Unexpected NaN's in age_group")
            if not np.sum(np.isnan(df_subj['x_t'])) == 0:
                sys.exit("Unexpected NaN's in x_t")
            if not np.sum(np.isnan(df_subj['b_t'])) == 0:
                sys.exit("Unexpected NaN's in b_t")
            if not np.sum(np.isnan(df_subj['mu_t'])) == 0:
                sys.exit("Unexpected NaN's in mu_t")
            if not np.sum(np.isnan(df_subj['c_t'])) == 0:
                sys.exit("Unexpected NaN's in c_t")
            if not np.sum(np.isnan(df_subj['e_t'])) == 0:
                sys.exit("Unexpected NaN's in e_t")
            if not np.sum(np.isnan(df_subj['pers'])) == 0:
                sys.exit("Unexpected NaN's in pers")
            if not np.sum(np.isnan(df_subj['edge'])) == 0:
                sys.exit("Unexpected NaN's in edge")

    return data_pn


# Preprocessing of first experiment
# ---------------------------------
data_pn_exp1 = preprocessing(1)

# Load previous file for comparison
expected_data_pn_exp1 = pd.read_pickle('al_data/data_prepr_1.pkl')

# Save data frame
data_pn_exp1.name = "data_prepr_1"
safe_save_dataframe(data_pn_exp1, np.nan, overleaf=False)

# Get participant information from BIDS metadata .json file
file_path = get_file_paths('al_data/first_experiment', "participants.tsv")
exp_1_participants = pd.read_csv(file_path[0], sep='\t', header=0)

# Preprocessing of second experiment
# ----------------------------------
data_pn_exp2 = preprocessing(2)

# Load previous file for comparison
expected_data_pn_exp2 = pd.read_pickle('al_data/data_prepr_2.pkl')

# Save data frame
data_pn_exp2.name = "data_prepr_2"
safe_save_dataframe(data_pn_exp2, np.nan, overleaf=False)

# Get participant information from BIDS metadata .json file
file_path = get_file_paths('al_data/follow_up_experiment', "participants.tsv")
exp_2_participants = pd.read_csv(file_path[0], sep='\t', header=0)

# Preprocessing of third experiment
# ---------------------------------
data_pn_exp3 = preprocessing(3)

# Save data frame
data_pn_exp3.name = "data_prepr_3"
safe_save_dataframe(data_pn_exp3, np.nan, overleaf=False)

# Get participant information from BIDS metadata .json file
file_path = get_file_paths('al_data/third_experiment', "participants.tsv")
exp_3_participants = pd.read_csv(file_path[0], sep='\t', header=0)

# Put participant statistics in data frames for Latex
# ---------------------------------------------------

# Initialize data frame
df_participants = pd.DataFrame(index=[1, 2, 3], columns=['min_age_ch', 'min_age_ad', 'min_age_ya', 'min_age_oa',
                                                      'max_age_ch', 'max_age_ad', 'max_age_ya', 'max_age_oa',
                                                      'median_age_ch', 'median_age_ad', 'median_age_ya',
                                                      'median_age_oa', 'n_ch', 'n_ad', 'n_ya', 'n_oa',
                                                      'n_female_ch', 'n_female_ad', 'n_female_ya', 'n_female_oa'])

# Compute stats

# First experiment
# ----------------
df_participants = get_sub_stats(exp_1_participants, df_participants)

# Second experiment
# -----------------
df_participants = get_sub_stats(exp_2_participants, df_participants, exp=2)

# Third experiment
# -----------------
df_participants.loc[3, 'min_age_ya'] = exp_3_participants['age'].min()
df_participants.loc[3, 'max_age_ya'] = exp_3_participants['age'].max()
df_participants.loc[3, 'median_age_ya'] = exp_3_participants['age'].median()
df_participants.loc[3, 'n_ya'] = len(exp_3_participants['age'])
df_participants.loc[3, 'n_female_ya'] = len(exp_3_participants[exp_3_participants['sex'] == 'female'])

# Save data frame
df_participants.index.name = 'exp'
df_participants.name = "participants"
safe_save_dataframe(df_participants, 'exp', sub_stats=True)
