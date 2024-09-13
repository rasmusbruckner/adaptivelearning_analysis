import numpy as np
import scipy.stats as stats
import statsmodels.stats.api as sm
import pandas as pd
from al_utilities import compute_anchoring_bias, safe_save_dataframe, get_stats_cont_exp


def compute_effects_exp3(df_exp3):
    """ This function compute the effects for the third experiment for plotting and reporting in paper

    :param df_exp3: Data frame experiment 3
    :return: pers_diff_bin_1_thres: Perseveration for smaller prediction errors
             anchoring_diff: Difference in anchoring between conditions
             pers_prob: Overall perseveration probability
             est_err: Overall estimation error
             df_reg: Anchoring bias based on regression analysis
    """

    # ---------------
    # 1. Prepare data
    # ---------------

    # Compute number of subjects
    n_subj = len(np.unique(df_exp3['subj_num']))

    # Divide condition info into two parts: cond (push vs. no-push) and rew_cond (high vs. low)
    df_exp3["cond"] = np.nan
    df_exp3["rew_cond"] = np.nan

    # "cond" part
    df_exp3.loc[df_exp3["all_cond"] == "stable_high", 'cond'] = "main_noPush"
    df_exp3.loc[df_exp3["all_cond"] == "stable_low", 'cond'] = "main_noPush"
    df_exp3.loc[df_exp3["all_cond"] == "push_high", 'cond'] = "main_push"
    df_exp3.loc[df_exp3["all_cond"] == "push_low", 'cond'] = "main_push"

    # "rew_cond" part
    df_exp3.loc[df_exp3["all_cond"] == "stable_high", 'rew_cond'] = "high"
    df_exp3.loc[df_exp3["all_cond"] == "stable_low", 'rew_cond'] = "low"
    df_exp3.loc[df_exp3["all_cond"] == "push_high", 'rew_cond'] = "high"
    df_exp3.loc[df_exp3["all_cond"] == "push_low", 'rew_cond'] = "low"

    # Extract push vs. no-push
    df_push = df_exp3[df_exp3['cond'] == 'main_push']

    # ----------------------------------------------------------------
    # 2. Perseveration differences between push and no-push conditions
    # ----------------------------------------------------------------

    # Compute perseveration probability for both conditions
    pers_prob = df_exp3.groupby(['subj_num', 'cond'])['pers'].mean().reset_index(drop=False)

    # Compute difference between the conditions
    pers_diff = pers_prob[pers_prob['cond'] == 'main_noPush']['pers'].reset_index(drop=True) - \
                pers_prob[pers_prob['cond'] == 'main_push']['pers'].reset_index(drop=True)

    # Get stats
    exp3_pers_desc, exp3_pers_stat, exp3_pers_effect_size = get_stats_cont_exp(pers_diff)
    exp3_pers_desc.name, exp3_pers_stat.name, exp3_pers_effect_size.name = \
        "exp3_pers_desc", "exp3_pers_stat", "exp3_pers_effect_size"

    # Save statistics for Latex manuscript
    safe_save_dataframe(exp3_pers_desc, 'age_group')
    safe_save_dataframe(exp3_pers_stat, 'age_group')
    safe_save_dataframe(exp3_pers_effect_size, 'type')

    # -----------------------------------------
    # 3. Perseveration differences in first bin
    # -----------------------------------------

    # Extract perseveration probability from first bin
    df_pers_binned = df_exp3.groupby(['subj_num', 'cond', 'rew_cond', 'pe_bin'])['pers'].mean().reset_index(drop=False)

    # Divide into low and high reward
    stable_low_bin_1 = df_pers_binned[(df_pers_binned['cond'] == 'main_noPush') & (df_pers_binned['pe_bin'] == 1) &
                                      (df_pers_binned['rew_cond'] == 'low')]['pers'].reset_index(drop=True)
    stable_high_bin_1 = df_pers_binned[(df_pers_binned['cond'] == 'main_noPush') & (df_pers_binned['pe_bin'] == 1) &
                                       (df_pers_binned['rew_cond'] == 'high')]['pers'].reset_index(drop=True)

    # Same but only for subjects with minimum 5% perseveration
    stable_high_bin_1_thres = stable_high_bin_1[stable_low_bin_1 > 0.05]
    stable_low_bin_1_thres = stable_low_bin_1[stable_low_bin_1 > 0.05]

    # Difference high and low for all subjects
    pers_diff_bin_1 = stable_high_bin_1 - stable_low_bin_1

    # Difference for perseverative subjects
    pers_diff_bin_1_thres = stable_high_bin_1_thres - stable_low_bin_1_thres

    # Stats for all subjects
    # ----------------------

    exp1_pers_bin_1_desc, exp1_pers_bin_1_stat, exp1_pers_bin_1_effect_size = get_stats_cont_exp(pers_diff_bin_1)
    exp1_pers_bin_1_desc.name, exp1_pers_bin_1_stat.name, exp1_pers_bin_1_effect_size.name = \
        "exp3_pers_bin_1_desc", "exp3_pers_bin_1_stat", "exp3_pers_bin_1_effect_size"

    # Save statistics for Latex manuscript
    safe_save_dataframe(exp1_pers_bin_1_desc, 'age_group')
    safe_save_dataframe(exp1_pers_bin_1_stat, 'age_group')
    safe_save_dataframe(exp1_pers_bin_1_effect_size, 'type')

    # Stats for subjects above threshold
    # ----------------------------------

    exp3_pers_bin_1_thres_desc, exp3_pers_bin_1_thres_stat, exp3_pers_bin_1_thres_effect_size = get_stats_cont_exp(
        pers_diff_bin_1_thres)
    exp3_pers_bin_1_thres_desc.name, exp3_pers_bin_1_thres_stat.name, exp3_pers_bin_1_thres_effect_size.name = \
        "exp3_pers_bin_1_thres_desc", "exp3_pers_bin_1_thres_stat", "exp3_pers_bin_1_thres_effect_size"

    # Save statistics for Latex manuscript
    safe_save_dataframe(exp3_pers_bin_1_thres_desc, 'age_group')
    safe_save_dataframe(exp3_pers_bin_1_thres_stat, 'age_group')
    safe_save_dataframe(exp3_pers_bin_1_thres_effect_size, 'type')

    # -------------------------------------------------------------------
    # 4. Estimation-error differences between push and no-push conditions
    # -------------------------------------------------------------------

    # Compute estimation error for both conditions
    est_err = df_exp3.groupby(['subj_num', 'cond', 'c_t'])['e_t'].mean().reset_index(drop=False)
    est_err = est_err[est_err['c_t'] == 0]  # Drop cp trials
    est_err = est_err.reset_index(drop=False)  # Reset index

    # Compute difference between the conditions
    est_err_diff = est_err[est_err['cond'] == 'main_noPush']['e_t'].reset_index(drop=True) - \
                   est_err[est_err['cond'] == 'main_push']['e_t'].reset_index(drop=True)

    # Get stats
    exp3_est_err_desc, exp3_est_err_stat, exp3_est_err_effect_size = get_stats_cont_exp(est_err_diff)
    exp3_est_err_desc.name, exp3_est_err_stat.name, exp3_est_err_effect_size.name = \
        "exp3_est_err_desc", "exp3_est_err_stat", "exp3_est_err_effect_size"

    # Save statistics for Latex manuscript
    safe_save_dataframe(exp3_est_err_desc, 'age_group')
    safe_save_dataframe(exp3_est_err_stat, 'age_group')
    safe_save_dataframe(exp3_est_err_effect_size, 'type')

    # ---------------------------------------------------
    # 5. Estimation-error differences high vs. low reward
    # ---------------------------------------------------

    # Extract estimation errors for high and low reward
    est_err_rew = df_exp3.groupby(['subj_num', 'rew_cond'])['e_t'].mean().reset_index(drop=False)

    # Compute difference
    est_err_rew_diff = est_err_rew[est_err_rew['rew_cond'] == 'high']['e_t'].reset_index(drop=True) - \
                       est_err_rew[est_err_rew['rew_cond'] == 'low']['e_t'].reset_index(drop=True)

    # Get stats
    est_err_rew_desc, est_err_rew_stat, est_err_rew_effect_size = get_stats_cont_exp(est_err_rew_diff)

    # -----------------------------------
    # 6. Anchoring bias in push condition
    # -----------------------------------

    # Compute anchoring bias for push condition
    a_t_name = 'a_t'
    y_t_name = 'y_t'
    df_reg = compute_anchoring_bias(n_subj, df_push, a_t_name, y_t_name)

    # Get stats
    exp3_df_reg_desc, exp3_df_reg_stat, exp3_df_reg_effect_size = get_stats_cont_exp(df_reg['bucket_bias'])
    exp3_df_reg_desc.name, exp3_df_reg_stat.name, exp3_df_reg_effect_size.name = \
        "exp3_df_reg_desc", "exp3_df_reg_stat", "exp3_df_reg_effect_size"

    # Save statistics for Latex manuscript
    safe_save_dataframe(exp3_df_reg_desc, 'age_group')
    safe_save_dataframe(exp3_df_reg_stat, 'age_group')
    safe_save_dataframe(exp3_df_reg_effect_size, 'type')

    # ---------------------------------------------------------
    # 7. Anchoring-bias differences between high and low reward
    # ---------------------------------------------------------

    # Divide into high- vs. low-reward condition
    df_push_high_rew = df_push[(df_push['rew_cond'] == 'high')]
    df_push_low_rew = df_push[(df_push['rew_cond'] == 'low')]

    # Compute bias separately for high and low conditions
    df_reg_high = compute_anchoring_bias(n_subj, df_push_high_rew, a_t_name, y_t_name)
    df_reg_low = compute_anchoring_bias(n_subj, df_push_low_rew, a_t_name, y_t_name)

    # Compute difference between the conditions
    anchoring_diff = df_reg_high['bucket_bias'] - df_reg_low['bucket_bias']

    # Get stats
    exp3_anchoring_diff_desc, exp3_anchoring_diff_stat, exp3_anchoring_diff_effect_size = get_stats_cont_exp(
        anchoring_diff)
    exp3_anchoring_diff_desc.name, exp3_anchoring_diff_stat.name, exp3_anchoring_diff_effect_size.name = \
        "exp3_anchoring_diff_desc", "exp3_anchoring_diff_stat", "exp3_anchoring_diff_effect_size"

    # Save statistics for Latex manuscript
    safe_save_dataframe(exp3_anchoring_diff_desc, 'age_group')
    safe_save_dataframe(exp3_anchoring_diff_stat, 'age_group')
    safe_save_dataframe(exp3_anchoring_diff_effect_size, 'type')

    # Extract perseveration in each bin
    pers_prob_bins = df_exp3.groupby(['subj_num', 'cond', 'pe_bin'])['pers'].mean().reset_index(drop=False)
    stable_bin_1 = pers_prob_bins[(pers_prob_bins['cond'] == 'main_noPush') & (pers_prob_bins['pe_bin'] == 1)]['pers']
    stable_bin_2 = pers_prob_bins[(pers_prob_bins['cond'] == 'main_noPush') & (pers_prob_bins['pe_bin'] == 2)]['pers']
    stable_bin_3 = pers_prob_bins[(pers_prob_bins['cond'] == 'main_noPush') & (pers_prob_bins['pe_bin'] == 3)]['pers']

    return pers_diff_bin_1_thres, anchoring_diff, pers_prob, est_err, df_reg


def ttests_main_paper(df_diff_values):
    """ This function computes descriptive and inferential statistics for the third experiment

    :param df_diff_values: Difference values
    :return: diff_results_desc: Descriptive results
             diff_results_stat: Inferential statistics
    """

    # Perform the one-sample t-test on the difference scores
    t_stat_diff, p_val_diff = stats.ttest_1samp(df_diff_values, 0)
    print(f"One-sample t-test on difference scores: t-statistic = {t_stat_diff}, p-value = {p_val_diff}")

    # Confidence interval perseveration difference
    stats_conf_int = stats.t.interval(0.95, len(df_diff_values) - 1, loc=np.mean(df_diff_values),
                                      scale=stats.sem(df_diff_values))

    # Just checking ci using alternative function from statsmodels api: results consistent
    stats_conf_int_alt = sm.DescrStatsW(df_diff_values).tconfint_mean()

    # Put descriptive results in data frame
    diff_results_desc = pd.DataFrame()
    diff_results_desc['mean'] = [round(np.mean(df_diff_values), 3)]
    diff_results_desc['lower_int'] = [round(stats_conf_int[0][0], 3)]
    diff_results_desc['upper_int'] = [round(stats_conf_int[1][0], 3)]
    diff_results_desc.index.name = 'age_group'
    diff_results_desc = diff_results_desc.rename({0: 'ya'}, axis='index')

    # Print out descriptives
    print("\nDescriptives")
    print(diff_results_desc)

    # Put stats in data frame
    diff_results_stat = pd.DataFrame()

    # Save all p-values
    diff_results_stat['p'] = [round(p_val_diff[0], 3)]

    # Save all test statistics
    diff_results_stat['stat'] = [round(t_stat_diff[0], 3)]

    diff_results_stat.index.name = 'age_group'
    diff_results_stat = diff_results_stat.rename({0: 'ya'}, axis='index')

    # Print out comparison against zero
    print("\nZero stats")
    print(diff_results_stat)

    return diff_results_desc, diff_results_stat
