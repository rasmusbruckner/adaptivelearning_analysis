""" Utilities: This module contains utility functions that are repeatedly used throughout the project """

import numpy as np
import pandas as pd
from scipy import stats
from fnmatch import fnmatch
from scipy.special import expit
import statsmodels.api as sm
import os
import re
import statsmodels.formula.api as smf
import sys
import math


def sorted_nicely(input_list):
    """ This function sorts the given iterable in the way that is expected

        Obtained from:
        https://arcpy.wordpress.com/2012/05/11/sorting-alphanumeric-strings-in-python/

        :param input_list: The iterable to be sorted
        :return: Sorted iterable
    """

    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(input_list, key=alphanum_key)


def get_df_subj(df, i):
    """ This function creates a subject-specific data frame with adjusted index

    :param df: Data frame containing all data
    :param i: Current subject number
    :return: df_subj: Index-adjusted subject-specific data frame
    """

    df_subj = df[(df['subj_num'] == i + 1)].copy()
    df_subj = df_subj.reset_index(drop=True)  # adjust index

    return df_subj


def get_sim_est_err(df_subj, df_data, exp=2):
    """ This function computes the simulated estimation errors

    :param df_subj: Subject-specific data
    :param df_data: Data frame containing simulation results
    :param exp: Current experiment
    :return: Conditional output:
                sim_est_err_no_push: Simulated estimation error standard condition
                sim_est_err_push: Simulated estimation error anchoring condition
                sim_est_err: Simulated estimation error
    """

    # Extract no-change-point trials
    no_cp = df_subj['c_t'] == 0

    # Extract true helicopter location for estimation-error computation
    real_mu = df_subj['mu_t']

    # Extract model prediction for estimation-error computation
    sim_pred = df_data['sim_b_t']
    sim_pred = sim_pred.reset_index(drop=True)  # adjust index

    # Compute estimation error
    sim_est_err_all = real_mu - sim_pred
    sim_est_err_nocp = sim_est_err_all[no_cp]  # estimation error without change points

    if exp == 2:

        # Extract shifting- and stable-bucket conditions
        cond_1 = df_subj['cond'] == "main_noPush"
        cond_1 = cond_1[no_cp]
        cond_2 = df_subj['cond'] == "main_push"
        cond_2 = cond_2[no_cp]

        # Compute average estimation errors for both conditions
        sim_est_err_no_push = np.mean(abs(sim_est_err_nocp[cond_1]))
        sim_est_err_push = np.mean(abs(sim_est_err_nocp[cond_2]))

        return sim_est_err_no_push, sim_est_err_push

    else:

        sim_est_err = np.mean(abs(sim_est_err_nocp))

        return sim_est_err


def correct_push(mu, sim_y_t):
    """ This function ensures that simulated bucket pushes stay within the screen range

    :param mu: Model belief
    :param sim_y_t: Simulated bucket push
    :return: sim_y_t: Adjusted y_t
             sim_z_t: Adjusted z_t
    """

    # For absolute position z_t, we compute the difference
    # between shift (y_t := z_t - b_{t-1}) and b_{t-1}: z_t = b_{t-1} + y_t
    sim_z_t = mu + sim_y_t

    # Adjust for edges of the screen. This is necessary because the model makes different trial-by-trial
    # predictions than participants, where we corrected for this already during preprocessing

    if sim_z_t > 300:  # push exceeds screen on the right side

        # Compute distance to screen edge
        right_dist = 300 - sim_z_t

        # Adjust y_t and z_t accordingly
        sim_y_t = sim_y_t + right_dist
        sim_z_t = mu + sim_y_t

    elif sim_z_t < 0:  # push undermines 0 (on the left side)

        # Compute distance to screen edge
        left_dist = 0 + sim_z_t

        # Adjust y_t and z_t accordingly
        sim_y_t = sim_y_t - left_dist
        sim_z_t = mu + sim_y_t

    return sim_y_t, sim_z_t


def safe_div(x, y):
    """ This function divides two numbers and avoids division by zero

        Obtained from:
        https://www.yawintutor.com/zerodivisionerror-division-by-zero/

    :param x: x-value
    :param y: y-value
    :return: c: result
    """

    if y == 0:
        c = 0.0
    else:
        c = x / y
    return c


def safe_div_list(x, y):
    """ This function divides two numbers in lists and avoids division by zero

    :param x: x-values
    :param y: y-values
    :return: c: result
    """

    c = np.full(len(y), np.nan)
    is_zero = y == 0
    c[is_zero] = 0.0

    # The suggested reformatting to "is_zero is not True" does not work (comparison not element-wise)
    c[is_zero == False] = x[is_zero == False] / y[is_zero == False]

    return c


def round_to_nearest_half_int(number):
    """ This functions rounds a number to the closest half integer

        Taken from https://bobbyhadz.com/blog/python-round-float-to-nearest-0-5

        :return: Result
    """

    return np.round(number * 2) / 2


def compute_persprob(intercept, slope, abs_pred_up):
    """ This function computes perseveration probability

    :param intercept: Logistic function intercept
    :param slope: Logistic function slope
    :param abs_pred_up: Absolute predicted update
    :return: Computed perseveration probability
    """

    # expit(x) = 1/(1+exp(-x)), i.e., (1/(1+exp(-slope*(abs_pred_up-int)))) (eq. 20)
    return expit(slope * (abs_pred_up - intercept))


def trial_cost_func(m, c, exp):
    """ This function computes error and motor costs for the motor model

    :param m: Cost grid
    :param c: Cost unit
    :param exp: Cost exponent
    :return: Computed costs
    """

    # curr_cost = c * m^exp
    return c * m ** exp


def find_nearest(array, value):
    """ This function finds the closet value in an array

    :param array: Array with all values
    :param value: Target value
    :return: Nearest value, id of nearest value
    """

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def get_mean_voi(df_int, voi):
    """ This function computes mean estimation errors and perseveration

    :param df_int: Data frame with single-trial data
    :param voi: Variable of interest: 1 = estimation error, 2 = perseveration, 3 = motor_perseveration
    :return: mean_voi: Data frame containing mean estimation errors
    """

    if voi == 1:
        # mean estimation errors
        mean_voi = df_int.groupby(['subj_num', 'age_group', 'c_t'])['e_t'].mean()
    elif voi == 2:
        # mean perseveration frequency
        mean_voi = df_int.groupby(['subj_num', 'age_group'])['pers'].mean()
    elif voi == 3:
        # mean perseveration frequency (edge trial)
        mean_voi = df_int.groupby(['subj_num', 'age_group', 'edge'])['pers'].mean()
    else:
        # mean motor-perseveration frequency (edge trial)
        mean_voi = df_int.groupby(['subj_num', 'age_group', 'edge'])['motor_pers'].mean()

    # Reset index
    mean_voi = mean_voi.reset_index(drop=False)

    if voi == 1:
        mean_voi = mean_voi[mean_voi['c_t'] == 0]  # Drop cp trials
        mean_voi = mean_voi.reset_index(drop=False)  # Reset index

    return mean_voi


def get_stats(voi, exp, voi_name, test=1):
    """ This function computes the statistical hypothesis tests

    :param voi: Variable of interest
    :param exp: Current experiment
    :param voi_name: Name of voi
    :param test: Which test to compute. 1: Comparison between the age groups. 2: Test against zero
    :return: voi_median, voi_q1, voi_q3, p_values, stat: Median, 1st and 3rd quartile, p-values and test statistics
    """

    # Compute median, first and third quartile
    voi_median = voi.groupby(['age_group'])[voi_name].median()
    voi_q1 = voi.groupby(['age_group'])[voi_name].quantile(0.25)
    voi_q3 = voi.groupby(['age_group'])[voi_name].quantile(0.75)

    # Put descriptive results in data frame
    desc = pd.DataFrame()
    desc['median'] = round(voi_median, 3)
    desc['q1'] = round(voi_q1, 3)
    desc['q3'] = round(voi_q3, 3)
    desc.index.name = 'age_group'
    desc = desc.rename({1: 'ch', 2: 'ad', 3: 'ya', 4: 'oa'}, axis='index')

    # Print out descriptives
    print("\nDescriptives")
    print(desc)

    if test == 1:

        # Test null hypothesis that two groups have the same distribution of their voi using the nonparametric
        # Mann-Whitney U test (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html)

        # Children and younger adults
        ch_ya_u, ch_ya_p = stats.mannwhitneyu(voi[voi['age_group'] == 1][voi_name],
                                              voi[voi['age_group'] == 3][voi_name])

        # Compute effect sizes
        n_ch = sum(voi['age_group'] == 1)
        n_ya = sum(voi['age_group'] == 3)
        cl_ch_ya, rank_biserial_ch_ya, r_square_ch_ya, r_ch_ya = mannwhitneyu_effectsize(ch_ya_u, n_ch, n_ya, ch_ya_p)

        # Children and older adults
        ch_oa_u, ch_oa_p = stats.mannwhitneyu(voi[voi['age_group'] == 1][voi_name],
                                              voi[voi['age_group'] == 4][voi_name])

        # Compute effect sizes
        n_oa = sum(voi['age_group'] == 4)
        cl_ch_oa, rank_biserial_ch_oa, r_square_ch_oa, r_ch_oa = mannwhitneyu_effectsize(ch_oa_u, n_ch, n_oa, ch_oa_p)

        # Younger and older adults
        ya_oa_u, ya_oa_p = stats.mannwhitneyu(voi[voi['age_group'] == 3][voi_name],
                                              voi[voi['age_group'] == 4][voi_name])

        # Compute effect sizes
        cl_ya_oa, rank_biserial_ya_oa, r_square_ya_oa, r_ya_oa = mannwhitneyu_effectsize(ya_oa_u, n_ya, n_oa, ya_oa_p)

        if exp == 1:

            # Test null hypothesis that the population median of all age groups is equal using the nonparametric
            # Kruskal Wallis H test (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html)
            kw_h, kw_p = stats.kruskal(voi[voi['age_group'] == 1][voi_name],
                                       voi[voi['age_group'] == 2][voi_name],
                                       voi[voi['age_group'] == 3][voi_name],
                                       voi[voi['age_group'] == 4][voi_name])

            # Test null hypothesis that two groups have the same distribution of their voi using the nonparametric
            # Mann-Whitney U test
            # ----------------------------------------------------------------------------------------------------

            # Children and adolescents
            ch_ad_u, ch_ad_p = stats.mannwhitneyu(voi[voi['age_group'] == 1][voi_name],
                                                  voi[voi['age_group'] == 2][voi_name])

            # Compute effect sizes
            n_ad = sum(voi['age_group'] == 2)
            cl_ch_ad, rank_biserial_ch_ad, r_square_ch_ad, r_ch_ad = mannwhitneyu_effectsize(ch_ad_u, n_ch, n_ad,
                                                                                             ch_ad_p)

            # Adolescents and younger adults
            ad_ya_u, ad_ya_p = stats.mannwhitneyu(voi[voi['age_group'] == 2][voi_name],
                                                  voi[voi['age_group'] == 3][voi_name])

            # Compute effect sizes
            cl_ad_ya, rank_biserial_ad_ya, r_square_ad_ya, r_ad_ya = mannwhitneyu_effectsize(ad_ya_u, n_ad, n_ya,
                                                                                             ad_ya_p)

            # Adolescents and older adults
            ad_oa_u, ad_oa_p = stats.mannwhitneyu(voi[voi['age_group'] == 2][voi_name],
                                                  voi[voi['age_group'] == 4][voi_name])

            # Compute effect sizes
            cl_ad_oa, rank_biserial_ad_oa, r_square_ad_oa, r_ad_oa = mannwhitneyu_effectsize(ad_oa_u, n_ad, n_oa,
                                                                                             ad_oa_p)

        else:

            # Test null hypothesis that the population median of all age groups is equal using the nonparametric
            # Kruskal Wallis H test (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html)
            kw_h, kw_p = stats.kruskal(voi[voi['age_group'] == 1][voi_name],
                                       voi[voi['age_group'] == 3][voi_name],
                                       voi[voi['age_group'] == 4][voi_name])

            # Set comparisons involving adolescents to nan
            ch_ad_p, ad_ya_p, ad_oa_p, ch_ad_u, ad_ya_u, ad_oa_u = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            cl_ch_ad, cl_ad_ya, cl_ad_oa = np.nan, np.nan, np.nan
            rank_biserial_ch_ad, rank_biserial_ad_ya, rank_biserial_ad_oa = np.nan, np.nan, np.nan
            r_square_ch_ad, r_square_ad_ya, r_square_ad_oa, r_ch_ad, r_ad_ya, r_ad_oa = \
                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        # Put stats in data frame
        stat = pd.DataFrame()

        # Save all p-values
        stat['p'] = np.array([round(kw_p, 3), round(ch_ad_p, 3), round(ch_ya_p, 3), round(ch_oa_p, 3),
                              round(ad_ya_p, 3), round(ad_oa_p, 3), round(ya_oa_p, 3)])

        # Save all test statistics
        stat['stat'] = np.array([round(kw_h, 3), round(ch_ad_u, 3), round(ch_ya_u, 3), round(ch_oa_u, 3),
                                 round(ad_ya_u, 3), round(ad_oa_u, 3), round(ya_oa_u, 3)])
        stat.index.name = 'test'
        stat = stat.rename({0: 'kw', 1: 'ch_ad', 2: 'ch_ya', 3: 'ch_oa', 4: 'ad_ya', 5: 'ad_oa', 6: 'ya_oa'},
                           axis='index')

        # Print out stats
        print("\nStats")
        print(stat)

        # Put all effect sizes in data frame
        effect_size = np.array([[round(cl_ch_ad, 3), round(rank_biserial_ch_ad, 3), round(r_square_ch_ad, 3),
                                 round(r_ch_ad, 3)],
                                [round(cl_ch_ya, 3), round(rank_biserial_ch_ya, 3), round(r_square_ch_ya, 3),
                                 round(r_ch_ya, 3)],
                                [round(cl_ch_oa, 3), round(rank_biserial_ch_oa, 3), round(r_square_ch_oa, 3),
                                 round(r_ch_oa, 3)],
                                [round(cl_ad_ya, 3), round(rank_biserial_ad_ya, 3), round(r_square_ad_ya, 3),
                                 round(r_ad_ya, 3)],
                                [round(cl_ad_oa, 3), round(rank_biserial_ad_oa, 3), round(r_square_ad_oa, 3),
                                 round(r_ad_oa, 3)],
                                [round(cl_ya_oa, 3), round(rank_biserial_ya_oa, 3), round(r_square_ya_oa, 3),
                                 round(r_ya_oa, 3)]])

        effect_size = pd.DataFrame(effect_size, columns=['cl', 'bi_c', 'r_sq', 'r'])
        effect_size.index.name = 'type'
        effect_size = effect_size.rename({0: 'ch_ad', 1: 'ch_ya', 2: 'ch_oa', 3: 'ad_ya', 4: 'ad_oa', 5: 'ya_oa'},
                                         axis='index')

        # Print out effect sizes
        print("\nEffect sizes")
        print(effect_size)

    elif test == 2:

        # Test null hypothesis that the distribution of the differences between bucket and no-bucket-shift conditions
        # is symmetric about zero with the nonparametric Wilcoxon sign rank test
        # (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html)

        # Children
        # -------
        res = stats.wilcoxon(voi[voi['age_group'] == 1][voi_name], method='approx')
        ch_stat, ch_p, ch_z = res.statistic, res.pvalue, res.zstatistic

        # Compute effect sizes
        n_ch = sum(voi['age_group'] == 1)
        cl_ch, rank_biserial_ch, r_square_ch, r_ch = wilcoxon_sign_rank_effectsize(n_ch, ch_stat, ch_z, ch_p)

        # Younger adults
        # --------------
        res = stats.wilcoxon(voi[voi['age_group'] == 3][voi_name], method='approx')
        ya_stat, ya_p, ya_z = res.statistic, res.pvalue, res.zstatistic

        # Compute effect sizes
        n_ya = sum(voi['age_group'] == 3)
        cl_ya, rank_biserial_ya, r_square_ya, r_ya = wilcoxon_sign_rank_effectsize(n_ya, ya_stat, ya_z, ya_p)

        # Older adults
        # ------------
        res = stats.wilcoxon(voi[voi['age_group'] == 4][voi_name], method='approx')
        oa_stat, oa_p, oa_z = res.statistic, res.pvalue, res.zstatistic

        # Compute effect sizes
        n_oa = sum(voi['age_group'] == 4)
        cl_oa, rank_biserial_oa, r_square_oa, r_oa = wilcoxon_sign_rank_effectsize(n_oa, oa_stat, oa_z, oa_p)

        if exp == 1:

            # Adolescents
            # -----------
            res = stats.wilcoxon(voi[voi['age_group'] == 2][voi_name], method='approx')
            ad_stat, ad_p, ad_z = res.statistic, res.pvalue, res.zstatistic

            # Compute effect sizes
            n_ad = sum(voi['age_group'] == 2)
            cl_ad, rank_biserial_ad, r_square_ad, r_ad = wilcoxon_sign_rank_effectsize(n_ad, ad_stat, ad_z, ad_p)

        else:

            # Set comparisons involving adolescents to nan
            ad_stat, ad_p, ad_z = np.nan, np.nan, np.nan
            cl_ad, rank_biserial_ad, r_square_ad, r_ad = np.nan, np.nan, np.nan, np.nan

        # Put stats in data frame
        stat = pd.DataFrame()

        # Save all p values
        stat['p'] = np.array([round(ch_p, 3), round(ad_p, 3), round(ya_p, 3), round(oa_p, 3)])

        # Save all test statistics
        stat['stat'] = np.array([round(ch_stat, 3), round(ad_stat, 3), round(ya_stat, 3), round(oa_stat, 3)])

        # Save all z values
        stat['z'] = np.array([round(ch_z, 3), round(ad_z, 3), round(ya_z, 3), round(oa_z, 3)])
        stat.index.name = 'age_group'
        stat = stat.rename({0: 'ch', 1: 'ad', 2: 'ya', 3: 'oa'}, axis='index')

        # Print out comparison against zero
        print("\nZero stats")
        print(stat)

        # Put all effect sizes in data frame
        effect_size = np.array([[round(cl_ch, 3), round(rank_biserial_ch, 3), round(r_square_ch, 3), round(r_ch, 3)],
                                [round(cl_ad, 3), round(rank_biserial_ad, 3), round(r_square_ad, 3), round(r_ad, 3)],
                                [round(cl_ya, 3), round(rank_biserial_ya, 3), round(r_square_ya, 3), round(r_ya, 3)],
                                [round(cl_oa, 3), round(rank_biserial_oa, 3), round(r_square_oa, 3), round(r_oa, 3)]])

        effect_size = pd.DataFrame(effect_size, columns=['cl', 'bi_c', 'r_sq', 'r'])
        effect_size.index.name = 'type'
        effect_size = effect_size.rename({0: 'ch', 1: 'ad', 2: 'ya', 3: 'oa'}, axis='index')

        # Print out effect sizes
        print("\nEffect sizes")
        print(effect_size)

    return desc, stat, effect_size


def get_stats_cont_exp(voi):
    """ This function computes the statistical hypothesis tests for the control experiment

        For more documentation, see get_stats in utilities

    :param voi: Variable of interest
    :return: desc: Descriptive statistics
             stat: Inferential statistics
             effect_size: Effect size
    """

    # Number of subjects
    n_subj = len(voi)

    # Compute median, first and third quartile
    voi_median = voi.median()
    voi_q1 = voi.quantile(0.25)
    voi_q3 = voi.quantile(0.75)

    # Put descriptive results in data frame
    desc = pd.DataFrame()
    desc['median'] = [round(voi_median, 3)]
    desc['q1'] = [round(voi_q1, 3)]
    desc['q3'] = [round(voi_q3, 3)]
    desc.index.name = 'age_group'
    desc = desc.rename({0: 'ya'}, axis='index')

    # Print out descriptives
    print("\nDescriptives")
    print(desc)

    # Wilcoxon signed-rank test for null hypothesis that scores come from the same distribution
    res = stats.wilcoxon(voi, method='approx')
    voi_stat, voi_p, voi_z = res.statistic, res.pvalue, res.zstatistic

    # Compute effect size
    cl_voi, rank_biserial_voi, r_square_voi, r_voi = wilcoxon_sign_rank_effectsize(n_subj, voi_stat, voi_z, voi_p)

    # Put stats in data frame
    stat = pd.DataFrame()

    # Save all p values
    stat['p'] = [round(voi_p, 3)]

    # Save all test statistics
    stat['stat'] = [round(voi_stat, 3)]

    # Save all z values
    stat['z'] = [round(voi_z, 3)]
    stat.index.name = 'age_group'
    stat = stat.rename({0: 'ya'}, axis='index')

    # Print out comparison against zero
    print("\nZero stats")
    print(stat)

    # Put all effect sizes in data frame
    effect_size = np.array([[round(cl_voi, 3), round(rank_biserial_voi, 3), round(r_square_voi, 3), round(r_voi, 3)]])
    effect_size = pd.DataFrame(effect_size, columns=['cl', 'bi_c', 'r_sq', 'r'])
    effect_size.index.name = 'type'
    effect_size = effect_size.rename({0: 'ya'}, axis='index')

    # Print out effect sizes
    print("\nEffect sizes")
    print(effect_size)

    return desc, stat, effect_size


def mannwhitneyu_effectsize(u1, nx, ny, p):
    """ This function compute several effect-size measures for the Mann-Whitney U Test

        Useful literature:
        Kirby (2014): https://journals.sagepub.com/doi/full/10.2466/11.IT.3.1
        Lakens (2013): https://www.frontiersin.org/articles/10.3389/fpsyg.2013.00863/full
        Wikipedia page: https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test

        Futuretodo: https://www.statsmodels.org/devel/generated/statsmodels.stats.nonparametric.rank_compare_2indep.html

    :param u1: Mann-Whitney U statistic
    :param nx: Number of subjects first group (sample x)
    :param ny: Number of subjects second group (sample y)
    :param p: P-value of Mann-Whitney U function
    :return: common_language: Common-language effect size
             rank_biserial: Rank-biserial correlation
             r_square: R-square (coefficient of determination)
             r: Pearson correlation

    """

    # Compute U statistic for second group (sample y)
    u2 = nx * ny - u1

    # Compute z statistic
    # z is the standardized U value, based on which p-value is computed
    # (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html)
    u = min(u1, u2)
    n = nx + ny
    z = (u - nx * ny / 2 + 0.5) / np.sqrt(nx * ny * (n + 1) / 12)
    p_based_on_z = 2 * stats.norm.cdf(z)  # use CDF to get p-value from smaller statistic

    # Check if z-value is consistent with p-value
    if not math.isclose(p_based_on_z, p, abs_tol=1e-2):
        sys.exit("P-values don't match")

    # Compute common-language effect size (or probability of superiority) in favor of a positive effect for group 1
    # When assuming a more positive effect for group 1, this is often called favorable evidence (f)
    common_language = u1 / (nx * ny)

    # Rank-biserial correlation
    # This can be expressed as r = f - u, where u denotes unfavorable evidence
    rank_biserial = common_language - (1 - common_language)

    # Coefficient of determination (r_square)
    r_square = z ** 2 / (nx + ny)
    # Often, this effect size is interpreted in the following way (e.g., in Lakens (2013)):
    # r^2 = 0.01 small
    # r^2 = 0.06 medium
    # r^2 = 0.14 large

    # Pearson correlation coefficient r = (z/sqrt(N)). Often interpreted in the following way:
    # |r| < 0.3: small
    # |r| > 0.3 and < 0.5: medium
    # |r| > 0.5: large
    r = z / np.sqrt(nx + ny)

    return common_language, rank_biserial, r_square, r


def wilcoxon_sign_rank_effectsize(n, w, z, p):
    """ This function compute several effect-size measures for the Wilcoxon sign-rank Test

        Useful literature:
        Kirby (2014): https://journals.sagepub.com/doi/full/10.2466/11.IT.3.1
        Lakens (2013): https://www.frontiersin.org/articles/10.3389/fpsyg.2013.00863/full
        Wikipedia page: https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test

    :param n: Number of subjects
    :param w: Test statistic
    :param z: z statistic
    :param p: p value
    :return: common_language: Common-language effect size
             rank_biserial: Rank-biserial correlation
             r_square: R-square (coefficient of determination)
             r: Pearson correlation
    """

    # Compute z based on p-value
    z_based_on_p = stats.norm.ppf(p / 2)

    # Check if z-value is consistent with p-value
    if not math.isclose(z_based_on_p, z, abs_tol=1e-5):
        sys.exit("Z-values don't match")

    # Compute total rank sum
    total_rank_sum = np.sum(np.arange(n) + 1)

    # In Scipy, w is the "sum of the ranks of the differences above or below zero, whichever is smaller".
    # To compute the larger w, we do
    w_large = total_rank_sum - w

    # The common-language effect size is the proportion of the larger rank sum
    # This is in line with the favorable evidence (f), assuming w_large is what we want
    common_language = (w_large / total_rank_sum)

    # Compute the "unfavorable" evidence u
    u = (w / total_rank_sum)

    # Compute the biserial-rank correlation, in line with r = f - u
    rank_biserial = u - common_language

    # Compute Pearson correlation
    r = z / np.sqrt(n)

    # Compute R^2 based on Pearson correlation
    r_square = r ** 2

    return common_language, rank_biserial, r_square, r


def compute_rob_reg_effect_size(res, data):
    """ This function compute effect sizes for the robust regression

        Useful literature: https://en.wikipedia.org/wiki/Coefficient_of_determination

    :param res: Regression results
    :param data: Data frame for regression model
    :return: r_square: Coefficient of determination
             r: Pearson correlation
    """

    # Sum of squared errors
    sse = sum(res.resid.values ** 2)

    # Total sum of squares
    sst = sum((data['d'] - np.mean(data['d'])) ** 2)

    # R-squared
    r_squared = 1 - (sse / sst)

    # Correlation coefficient
    r = np.sqrt(1 - (sse / sst))

    # Print out effect sizes
    print("\n\nR-squared = " + str(r_squared))
    print("r = " + str(r))

    return r_squared, r


def get_cond_diff(cond_1, cond_2, voi):
    """ This function computes the differences in perseveration and estimation errors between
        two task conditions in the follow-up experiment

    :param cond_1: First condition of interest
    :param cond_2: Second condition of interest
    :param voi: Variable of interest
    :return: desc: Descriptive statistics
             stat: Inferential statistics
             zero_stat: Inferential statistics against zero
    """

    # Identify variable of interest
    if voi == 1:
        voi_name = 'e_t'
        print_name = 'Estimation error'
    elif voi == 2:
        voi_name = 'pers'
        print_name = 'Perseveration'
    else:
        voi_name = 'motor_pers'
        print_name = 'Motor perseveration'

    # Compute mean of variable of interest
    voi_cond_1 = get_mean_voi(cond_1, voi)
    voi_cond_2 = get_mean_voi(cond_2, voi)

    # Compute difference between conditions
    cond_diff = voi_cond_2.copy()
    cond_diff[voi_name] = cond_diff[voi_name] - voi_cond_1[voi_name]

    print('\n' + print_name + ' difference')
    desc, stat, effect_size_diff = get_stats(cond_diff, 2, voi_name)

    print('\n' + print_name + ' difference test against zero')
    _, zero_stat, effect_size_zero = get_stats(cond_diff, 2, voi_name, test=2)

    return cond_diff, desc, stat, zero_stat, effect_size_diff, effect_size_zero


def get_sub_stats(bids_participants, df_participants, exp=1):
    """ This function computes useful subject descriptive statistics for the methods section

    :param bids_participants: Participant information from BIDS metadata
    :param df_participants: (Initialized) data frame with participant stats
    :param exp: Current experiment
    :return: df_participants: Updated data frame with participant stats
    """

    def sub_min(sub_stats, sub_data, curr_exp):
        """ This sub-function computes the minimum age

        :param sub_stats: Current data frame with participant stats
        :param sub_data: Participant information from BIDS metadata
        :param curr_exp: Current experiment
        :return: Updated data frame with participant stats
        """
        # First experiment: minimum age
        sub_stats.loc[exp, 'min_age_ch'] = sub_data[sub_data['age_group'] == 1]['age'].min()
        sub_stats.loc[exp, 'min_age_ya'] = sub_data[sub_data['age_group'] == 3]['age'].min()
        sub_stats.loc[exp, 'min_age_oa'] = sub_data[sub_data['age_group'] == 4]['age'].min()

        if curr_exp == 1:
            sub_stats.loc[exp, 'min_age_ad'] = sub_data[sub_data['age_group'] == 2]['age'].min()

        return sub_stats

    def sub_max(sub_stats, sub_data, curr_exp):
        """ This sub-function computes the maximum age

        :param sub_stats: Current data frame with participant stats
        :param sub_data: Participant information from BIDS metadata
        :param curr_exp: Current experiment
        :return: Updated data frame with participant stats
        """

        # First experiment: maximum age
        sub_stats.loc[exp, 'max_age_ch'] = sub_data[sub_data['age_group'] == 1]['age'].max()
        sub_stats.loc[exp, 'max_age_ya'] = sub_data[sub_data['age_group'] == 3]['age'].max()
        sub_stats.loc[exp, 'max_age_oa'] = sub_data[sub_data['age_group'] == 4]['age'].max()

        if curr_exp == 1:
            sub_stats.loc[exp, 'max_age_ad'] = sub_data[sub_data['age_group'] == 2]['age'].max()

        return sub_stats

    def sub_median(sub_stats, sub_data, curr_exp):
        """ This sub-function computes the median age

        :param sub_stats: Current data frame with participant stats
        :param sub_data: Participant information from BIDS metadata
        :param curr_exp: Current experiment
        :return: Updated data frame with participant stats
        """

        # First experiment: maximum age
        sub_stats.loc[exp, 'median_age_ch'] = sub_data[sub_data['age_group'] == 1]['age'].median()
        sub_stats.loc[exp, 'median_age_ya'] = sub_data[sub_data['age_group'] == 3]['age'].median()
        sub_stats.loc[exp, 'median_age_oa'] = sub_data[sub_data['age_group'] == 4]['age'].median()

        if curr_exp == 1:
            sub_stats.loc[exp, 'median_age_ad'] = sub_data[sub_data['age_group'] == 2]['age'].median()

        return sub_stats

    def sub_n(sub_stats, sub_data, curr_exp):
        """ This sub-function compute the number of subjects

        :param sub_stats: Current data frame with participant stats
        :param sub_data: Participant information from BIDS metadata
        :param curr_exp: Current experiment
        :return: Updated data frame with participant stats
        """

        sub_stats.loc[exp, 'n_ch'] = len(sub_data[(sub_data['age_group'] == 1)])
        sub_stats.loc[exp, 'n_ya'] = len(sub_data[(sub_data['age_group'] == 3)])
        sub_stats.loc[exp, 'n_oa'] = len(sub_data[(sub_data['age_group'] == 4)])

        if curr_exp == 1:
            sub_stats.loc[exp, 'n_ad'] = len(sub_data[(sub_data['age_group'] == 2)])

        return sub_stats

    def sub_n_female(sub_stats, sub_data, curr_exp):
        """ This sub-function computes the number of females

        :param sub_stats: Current data frame with participant stats
        :param sub_data: Participant information from BIDS metadata
        :param curr_exp: Current experiment
        :return: Updated data frame with participant stats
        """

        sub_stats.loc[exp, 'n_female_ch'] = len(sub_data[(sub_data['age_group'] == 1)
                                                         & (sub_data['sex'] == 'female')])
        sub_stats.loc[exp, 'n_female_ya'] = len(sub_data[(sub_data['age_group'] == 3)
                                                         & (sub_data['sex'] == 'female')])
        sub_stats.loc[exp, 'n_female_oa'] = len(sub_data[(sub_data['age_group'] == 4)
                                                         & (sub_data['sex'] == 'female')])

        if curr_exp == 1:
            sub_stats.loc[exp, 'n_female_ad'] = len(sub_data[(sub_data['age_group'] == 2)
                                                             & (sub_data['sex'] == 'female')])

        return sub_stats

    # Put participant statistics in data frame for Latex
    # --------------------------------------------------

    # Minimum age
    df_participants = sub_min(df_participants, bids_participants, exp)

    # Maximum age
    df_participants = sub_max(df_participants, bids_participants, exp)

    # Median age
    df_participants = sub_median(df_participants, bids_participants, exp)

    # Number of subject
    df_participants = sub_n(df_participants, bids_participants, exp)

    # Number of females
    df_participants = sub_n_female(df_participants, bids_participants, exp)

    return df_participants


def compute_average_LR(n_subj, df_exp):
    """ This function compute the average learning rate

    :param n_subj: Number of subjects
    :param df_exp: Data frame with all data
    :return: df_alpha: Regression results
    """

    # Initialize learning-rate and age-group variables
    alpha = np.full(n_subj, np.nan)
    age_group = np.full(n_subj, np.nan)
    subj_num = np.full(n_subj, np.nan)

    # Cycle over participants
    for i in range(0, n_subj):

        df_subj = get_df_subj(df_exp, i)

        # Extract prediction error and prediction update and add intercept to data frame
        predictor = df_subj['delta_t']
        dependent_var = df_subj['a_t']
        predictor = predictor.dropna()
        dependent_var = dependent_var.dropna()
        predictor = sm.add_constant(predictor)  # adding a constant as intercept

        # Estimate model and extract learning rate parameter alpha (i.e., influence of delta_t on a_t)
        model = sm.OLS(dependent_var, predictor).fit()
        alpha[i] = model.params['delta_t']
        age_group[i] = np.unique(df_subj['age_group'])
        subj_num[i] = i + 1

        # Uncomment for single-trial figure
        # plt.figure()
        # plt.plot(predictor, dependent_var, '.')

    # Add learning rate results to data frame
    df_alpha = pd.DataFrame()
    df_alpha['alpha'] = alpha
    df_alpha['age_group'] = age_group
    df_alpha['subj_num'] = subj_num

    return df_alpha


def compute_anchoring_bias(n_subj, all_data_anchor, a_t_name='sim_a_t', y_t_name='sim_y_t'):
    """ This function computes the anchoring bias

    :param n_subj: Number of subjects
    :param all_data_anchor: Data frame with all subjects
    :param a_t_name: Name of update (string)
    :param y_t_name: Name of push (string)
    :return: df_reg: Regression results
    """

    # Initialize learning rate, bucket bias and age_group variables
    alpha = np.full(n_subj, np.nan)
    bucket_bias = np.full(n_subj, np.nan)
    age_group = np.full(n_subj, np.nan)

    # Cycle over participants
    for i in range(0, n_subj):

        # Extract data of current participant
        df_subj = all_data_anchor[(all_data_anchor['subj_num'] == i + 1)].copy()
        df_subj_push = df_subj[df_subj['cond'] == 'main_push'].copy()
        df_subj_push = df_subj_push.reset_index()

        # Regression data
        data = pd.DataFrame()
        data['a_t'] = df_subj_push[a_t_name].copy()
        data['delta_t'] = df_subj_push['delta_t'].copy()
        data['y_t'] = df_subj_push[y_t_name].copy()

        # Run regression model
        mod = smf.ols(formula='a_t ~ delta_t + y_t', data=data)
        res = mod.fit()

        # Save results
        alpha[i] = res.params['delta_t']
        bucket_bias[i] = res.params['y_t']
        age_group[i] = np.unique(df_subj['age_group'])

    # Add learning rate results to data frame
    df_reg = pd.DataFrame()
    df_reg['alpha'] = alpha
    df_reg['bucket_bias'] = bucket_bias
    df_reg['age_group'] = age_group

    return df_reg


def compute_pers_anchoring_relation(pers_noPush, model_exp2):
    """ This function computes the association between perseveration and estimation errors

    :param pers_noPush: Perseveration in no-push condition
    :param model_exp2: Estimation error
    :return: res: Regression results
    """

    # Data frame for regression model
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

    compute_rob_reg_effect_size(res, data)

    return res


def compute_rt(rt_type, df_exp):
    """ This function computes different reaction times for the task conditions in the follow-up experiment

    :param rt_type: Type of reaction-time transformation ("speed", "log", "standard")
    :param df_exp: Data frame
    :return: mean_init_rt_push: IT for push condition
             mean_init_rt_no_push: IT for standard condition
             mean_rt_push: RT for push condition
             mean_rt_no_push: RT for standard condition
    """

    # What to compute?
    init_rt, rt = get_rt_voi(rt_type)

    # Age-related differences in IT
    mean_init_rt = df_exp.groupby(['subj_num', 'age_group', 'cond'])[init_rt].mean().reset_index(drop=False)
    mean_init_rt_push = mean_init_rt[mean_init_rt['cond'] == 'main_push'].reset_index(drop=True)
    mean_init_rt_no_push = mean_init_rt[mean_init_rt['cond'] == 'main_noPush'].reset_index(drop=True)

    # Age-related differences in RT
    mean_rt = df_exp.groupby(['subj_num', 'age_group', 'cond'])[rt].mean().reset_index(drop=False)
    mean_rt_push = mean_rt[mean_rt['cond'] == 'main_push'].reset_index(drop=True)
    mean_rt_no_push = mean_rt[mean_rt['cond'] == 'main_noPush'].reset_index(drop=True)

    return mean_init_rt_push, mean_init_rt_no_push, mean_rt_push, mean_rt_no_push


def get_rt_voi(rt_type):
    """ This function maps rt_type input to variable name in data frames

    :param rt_type: Type of reaction-time transformation ("speed", "log", "standard")
    :return: init_rt: Name of IT
             rt: Name of RT
    """

    # What to compute?
    if rt_type == "speed":
        init_rt = "init_rt_speed"
        rt = "rt_speed"
    elif rt_type == "log":
        init_rt = "init_rt_log"
        rt = "rt_log"
    elif rt_type == "log_corrected":
        init_rt = "init_rt_log"  # since this is initiation, we don't take corrected one
        rt = "rt_log_corrected"
    else:
        init_rt = "init_rt"
        rt = "rt"

    return init_rt, rt


def load_data(f_names):
    """ This function loads the adaptive learning BIDS data and checks if they are complete

    :param f_names: List with all file names
    :return: all_data: Data frame that contains all data
    """

    # Initialize arrays
    n_trials = np.full(len(f_names), np.nan)  # number of trials

    # Put data in data frame
    all_data = np.nan
    for i in range(0, len(f_names)):

        if i == 0:

            # Load data of participant 0
            all_data = pd.read_csv(f_names[0], sep='\t', header=0)
            new_data = all_data
        else:

            # Load data of participant 1,..,N
            new_data = pd.read_csv(f_names[i], sep='\t', header=0)

        # Count number of respective trials
        n_trials[i] = len(new_data)
        new_data['trial'] = np.int64(np.arange(n_trials[i]))

        # Indicate if less than 400 trials
        if n_trials[i] < 400:
            print("Only %i trials found" % n_trials[i])

        # Append data frame
        if i > 0:
            all_data = pd.concat([all_data, new_data], ignore_index=True)

    return all_data


def get_file_paths(folder_path, identifier):
    """ This function extracts the file path

    :param folder_path: Relative path to current folder
    :param identifier: Identifier for file of interest
    :return: file_path: Absolute path to file
    """

    file_paths = []
    for path, subdirs, files in os.walk(folder_path):
        for name in files:
            if fnmatch(name, identifier):
                file_paths.append(os.path.join(path, name))

    return file_paths


def safe_save_dataframe(dataframe, index_col, overleaf=True, sub_stats=False):
    """ This function saves a data frame for the Latex manuscript and ensures that values don't change unexpectedly

    :param dataframe: Data frame that we want to save
    :param index_col: Column index
    :param overleaf: If true, use overleaf folder, otherwise al_data
    :param sub_stats: If used in preprocessing for sub stats, assume object type
    :return: None
    """

    # Get home directory
    paths = os.getcwd()
    path = paths.split(os.path.sep)
    home_dir = path[1]

    # Initialize expected data frame
    expected_df = np.nan

    # Load previous file for comparison
    if overleaf:

        # File path
        df_name = '/' + home_dir + '/rasmus/Dropbox/Apps/Overleaf/al_manuscript/al_dataframes/' + dataframe.name + '.csv'

        # Check if file exists
        path_exist = os.path.exists(df_name)

        # If so, load file for comparison
        if path_exist:
            expected_df = pd.read_csv(df_name, index_col=index_col)

    else:

        # File path
        df_name = 'al_data/' + dataframe.name + '.pkl'

        # Check if file exists
        path_exist = os.path.exists(df_name)

        # If so, load file for comparison
        if path_exist:
            expected_df = pd.read_pickle(df_name)

    # If we have the file already, check if as expected
    if path_exist:
        # Test if equal and save data
        if sub_stats:
            expected_df = expected_df.astype(object)
        same = dataframe.equals(expected_df)
        print("\nActual and expected " + dataframe.name + " equal:", same, "\n")

    # If new, we'll create the file
    else:
        same = True
        print("\nCreating new data frame: " + dataframe.name + "\n")

    # Save file
    if overleaf:
        if not same:
            dataframe.to_csv(
                '~/Dropbox/Apps/Overleaf/al_manuscript/al_dataframes/' + dataframe.name + '_unexpected.csv')
        else:
            dataframe.to_csv('~/Dropbox/Apps/Overleaf/al_manuscript/al_dataframes/' + dataframe.name + '.csv')
    else:
        if not same:
            dataframe.to_pickle('al_data/' + dataframe.name + '_unexpected.pkl')
        else:
            dataframe.to_pickle('al_data/' + dataframe.name + '.pkl')
