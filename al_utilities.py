import numpy as np
import pandas as pd
from scipy import stats
from fnmatch import fnmatch
from scipy.special import expit
import os
import re


def load_data(f_names):
    """ This function loads the adaptive learning BIDS data and checks if they are complete

    :param f_names: List with all file names
    :return: all_data: Data frame that contains all data
    """

    # Initialize arrays
    n_trials = np.full(len(f_names), np.nan)  # number of trials

    # Put data in data frame
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

        # new_data['id'] = i

        # Indicate if less than 400 trials
        if n_trials[i] < 400:
            print("Only %i trials found" % n_trials[i])

        # Append data frame
        if i > 0:
            all_data = all_data.append(new_data, ignore_index=True)

    return all_data


def sorted_nicely(l):
    """ This function sorts the given iterable in the way that is expected

        Obtained from:
        https://arcpy.wordpress.com/2012/05/11/sorting-alphanumeric-strings-in-python/

        :param l: The iterable to be sorted
        :return: Sorted iterable
    """

    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(l, key=alphanum_key)


def get_df_subj(df, i):
    """ This function creates a subject-specific data frame with adjusted index

    :param df: Data frame containing all data
    :param i: Current subject number
    :return: df_subj: Index-adjusted subject-spedific data frame
    """
    df_subj = df[(df['subj_num'] == i + 1)].copy()
    x = np.linspace(0, len(df_subj) - 1, len(df_subj))
    df_subj.loc[:, 'trial'] = x.tolist()
    df_subj = df_subj.set_index('trial')

    return df_subj


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


def get_mean_voi(df_int, voi):
    """ This function computes mean estimation errors and perseveration

    :param df_int: Data frame with single-trial data
    :param voi: Variable of interest: 1 = estimation error, 2 = perseveration, 3 = motor_perseveration
    :return: mean_voi: Data frame containing mean estimation errors
    """

    if voi == 1:
        # mean estimation errors
        mean_voi = df_int.groupby(['subj_num', 'age_group',  'c_t'])['e_t'].mean()
    elif voi == 2:
        # mean perseveration frequency
        mean_voi = df_int.groupby(['subj_num', 'age_group'])['pers'].mean()
    elif voi == 3:
        # mean motor-perseveration frequency
        mean_voi = df_int.groupby(['subj_num', 'age_group', 'edge'])['pers'].mean()
    else:
        # mean motor-perseveration frequency
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

    if test == 1:

        # Test null hypothesis that two groups have the same distribution of their voi using the nonparametric
        # Mann-Whitney U test (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html)

        # Children and younger adults
        ch_ya_u, ch_ya_p = stats.mannwhitneyu(voi[voi['age_group'] == 1][voi_name],
                                              voi[voi['age_group'] == 3][voi_name], alternative='two-sided')

        # Children and older adults
        ch_oa_u, ch_oa_p = stats.mannwhitneyu(voi[voi['age_group'] == 1][voi_name],
                                              voi[voi['age_group'] == 4][voi_name], alternative='two-sided')

        # Younger and older adults
        ya_oa_u, ya_oa_p = stats.mannwhitneyu(voi[voi['age_group'] == 3][voi_name],
                                              voi[voi['age_group'] == 4][voi_name], alternative='two-sided')

        if exp == 1:

            # Test null hypothesis that that the population median of all age groups is equal using the nonparametric
            # Kruskal Wallis H test (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html)
            kw_H, kw_p = stats.kruskal(voi[voi['age_group'] == 1][voi_name],
                                       voi[voi['age_group'] == 2][voi_name],
                                       voi[voi['age_group'] == 3][voi_name],
                                       voi[voi['age_group'] == 4][voi_name])

            # Test null hypothesis that two groups have the same distribution of their voi using the nonparametric
            # Mann-Whitney U test
            # ----------------------------------------------------------------------------------------------------

            # Children and adolescents
            ch_ad_u, ch_ad_p = stats.mannwhitneyu(voi[voi['age_group'] == 1][voi_name],
                                                  voi[voi['age_group'] == 2][voi_name], alternative='two-sided')
            # Adolescents and younger adults
            ad_ya_u, ad_ya_p = stats.mannwhitneyu(voi[voi['age_group'] == 2][voi_name],
                                                  voi[voi['age_group'] == 3][voi_name], alternative='two-sided')
            # Adolescents and older adults
            ad_oa_u, ad_oa_p = stats.mannwhitneyu(voi[voi['age_group'] == 2][voi_name],
                                                  voi[voi['age_group'] == 4][voi_name], alternative='two-sided')

        else:

            # Test null hypothesis that that the population median of all age groups is equal using the nonparametric
            # Kruskal Wallis H test (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html)
            kw_H, kw_p = stats.kruskal(voi[voi['age_group'] == 1][voi_name],
                                       voi[voi['age_group'] == 3][voi_name],
                                       voi[voi['age_group'] == 4][voi_name])

            # Set comparisons involving adolescents to nan
            ch_ad_p = np.nan
            ad_ya_p = np.nan
            ad_oa_p = np.nan
            ch_ad_u = np.nan
            ad_ya_u = np.nan
            ad_oa_u = np.nan

        # Save all p values
        p_values = np.array([round(kw_p, 3), round(ch_ad_p, 3), round(ch_ya_p, 3), round(ch_oa_p, 3), round(ad_ya_p, 3),
                             round(ad_oa_p, 3), round(ya_oa_p, 3)])
        # Save all test statistics
        stat = np.array([round(kw_H, 3), round(ch_ad_u, 3), round(ch_ya_u, 3), round(ch_oa_u, 3), round(ad_ya_u, 3),
                         round(ad_oa_u, 3), round(ya_oa_u, 3)])

        # Print results to console
        # -------------------------
        print('Kruskal-Wallis: H = %.3f, p = %.3f' % (round(kw_H, 3), round(kw_p, 3)))
        print('Children - adolescents: u = %.3f, p = %.3f' % (round(ch_ad_u, 3), round(ch_ad_p, 3)))
        print('Children - younger adults: u = %.3f, p = %.3f' % (round(ch_ya_u, 3), round(ch_ya_p, 3)))
        print('Children - older adults: u = %.3f, p = %.3f' % (round(ch_oa_u, 3), round(ch_oa_p, 3)))
        print('Adolescents - younger adults: u = %.3f, p = %.3f' % (round(ad_ya_u, 3), round(ad_ya_p, 3)))
        print('Adolescents - older adults: u = %.3f, p = %.3f' % (round(ad_oa_u, 3), round(ad_oa_p, 3)))
        print('Younger adults - older adults: u = %.3f, p = %.3f' % (round(ya_oa_u, 3), round(ya_oa_p, 3)))

        print('Children: median = %.3f , IQR = (%.3f - %.3f)'
              % (round(voi_median[1], 3), round(voi_q1[1], 3), round(voi_q3[1], 3)))
        if exp == 1:
            print('Adolescents: median = %.3f , IQR = (%.3f - %.3f)'
                  % (round(voi_median[2], 3), round(voi_q1[2], 3), round(voi_q3[2], 3)))
        print('Younger adults: median = %.3f , IQR = (%.3f - %.3f)'
              % (round(voi_median[3], 3), round(voi_q1[3], 3), round(voi_q3[3], 3)))
        print('Older adults: median = %.3f , IQR = (%.3f - %.3f)'
              % (round(voi_median[4], 3), round(voi_q1[4], 3), round(voi_q3[4], 3)))

    elif test == 2:

        # Test null hypothesis that the distribution of the differences between bucket and no bucket shift conditions
        # is symmetric about zero with the nonparametric Wilcoxon sign rank test
        # (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html)

        ch_stat, ch_p = stats.wilcoxon(voi[voi['age_group'] == 1][voi_name], y=None, zero_method='wilcox',
                                       correction=False, alternative='two-sided')
        ya_stat, ya_p = stats.wilcoxon(voi[voi['age_group'] == 3][voi_name], y=None, zero_method='wilcox',
                                       correction=False, alternative='two-sided')
        oa_stat, oa_p = stats.wilcoxon(voi[voi['age_group'] == 4][voi_name], y=None, zero_method='wilcox',
                                       correction=False, alternative='two-sided')

        # Save all p values
        p_values = np.array([round(ch_p, 3), round(ya_p, 3), round(oa_p, 3)])

        # Save all test statistics
        stat = np.array([round(ch_stat, 3), round(ya_stat, 3), round(oa_stat, 3)])

        # Print results to console
        # -------------------------
        print('Children: w = %.3f, p = %.3f' % (round(ch_stat, 3), round(ch_p, 3)))
        print('Younger adults: w = %.3f, p = %.3f' % (round(ya_stat, 3), round(ya_p, 3)))
        print('Older adults: w = %.3f, p = %.3f' % (round(oa_stat, 3), round(oa_p, 3)))

    return voi_median, voi_q1, voi_q3, p_values, stat


def safe_div(x, y):
    """ This function divides two numbers and avoids division by zero

    :param x: x-value
    :param y: y-value
    :return: result
    """

    if y < 1.e-5:
        y = 1.e-5
    if x < 1.e-5:
        x = 1.e-5

    return x / y


def compute_persprob(intercept, slope, abs_pred_up):
    """ This function computes the perseveration probability

    :param intercept: Logistic function intercept
    :param slope: Logistic function slope
    :param abs_pred_up: Absolute predicted update
    :return: computed perseveration probability
    """

    # expit(x) = 1/(1+exp(-x)), i.e., (1/(1+exp(-slope*(abs_pred_up-int))))
    return expit(slope*(abs_pred_up-intercept))
