""" Test Utilities: Utilities unit tests """

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch
from unittest.mock import Mock
from al_utilities import (sorted_nicely, safe_div, safe_div_list, get_sim_est_err, correct_push,
                          round_to_nearest_half_int, get_df_subj, compute_persprob, trial_cost_func, find_nearest,
                          get_mean_voi, get_stats, get_sub_stats, compute_average_LR, compute_anchoring_bias,
                          mannwhitneyu_effectsize, wilcoxon_sign_rank_effectsize, compute_rob_reg_effect_size)


class MockResid:
    """ This mock function is used in the MockRes mock function to mock the residuals of a regression """

    def __init__(self):
        self.values = np.arange(10) * 0.1


class MockRes:
    """This mock function mocks a regression function """

    def __init__(self):
        mock_resid = MockResid()
        self.resid = mock_resid


class TestUtilities(unittest.TestCase):

    class Wilcoxon:
        """ This class definition is used to mock-out the scipy wilcoxon function to determine the mock output """

        def __init__(self, statistic, pvalue, zstatistic):
            self.statistic = statistic
            self.pvalue = pvalue
            self.zstatistic = zstatistic

    class OLS:
        """ This class definition is used to mock-out the scipy OLS function to determine the mock output """

        def __init__(self, formula=1, data=0):
            self.data = data
            self.formula = formula
            self.params = pd.DataFrame({'delta_t': [1], 'y_t': [2]})

        def fit(self):
            """ This is a mock fit() function for OLS """

            return self

    def test_sorted_nicely(self):
        """ This function tests the sorting function used for sorting the file paths """

        file_paths = ['al_data/first_experiment/sub_5/behav/sub-5_task-helicopter_behav.tsv',
                      'al_data/first_experiment/sub_1/behav/sub-1_task-helicopter_behav.tsv',
                      'al_data/first_experiment/sub_8/behav/sub-8_task-helicopter_behav.tsv',
                      'al_data/first_experiment/sub_4/behav/sub-4_task-helicopter_behav.tsv',
                      'al_data/first_experiment/sub_2/behav/sub-2_task-helicopter_behav.tsv',
                      'al_data/first_experiment/sub_6/behav/sub-6_task-helicopter_behav.tsv',
                      'al_data/first_experiment/sub_9/behav/sub-9_task-helicopter_behav.tsv',
                      'al_data/first_experiment/sub_3/behav/sub-3_task-helicopter_behav.tsv',
                      'al_data/first_experiment/sub_10/behav/sub-10_task-helicopter_behav.tsv',
                      'al_data/first_experiment/sub_7/behav/sub-7_task-helicopter_behav.tsv']

        # Sort all file names according to participant ID
        file_paths_sorted = sorted_nicely(file_paths)

        # Expected test results
        file_paths_expected = ['al_data/first_experiment/sub_1/behav/sub-1_task-helicopter_behav.tsv',
                               'al_data/first_experiment/sub_2/behav/sub-2_task-helicopter_behav.tsv',
                               'al_data/first_experiment/sub_3/behav/sub-3_task-helicopter_behav.tsv',
                               'al_data/first_experiment/sub_4/behav/sub-4_task-helicopter_behav.tsv',
                               'al_data/first_experiment/sub_5/behav/sub-5_task-helicopter_behav.tsv',
                               'al_data/first_experiment/sub_6/behav/sub-6_task-helicopter_behav.tsv',
                               'al_data/first_experiment/sub_7/behav/sub-7_task-helicopter_behav.tsv',
                               'al_data/first_experiment/sub_8/behav/sub-8_task-helicopter_behav.tsv',
                               'al_data/first_experiment/sub_9/behav/sub-9_task-helicopter_behav.tsv',
                               'al_data/first_experiment/sub_10/behav/sub-10_task-helicopter_behav.tsv']

        self.assertTrue(file_paths_sorted == file_paths_expected)

    def test_get_df_subj(self):
        """ This function tests the function that selects the subject-specific data frames """

        # Create input data frame
        df_subj = pd.DataFrame(index=np.arange(10) + 20)
        df_subj['subj_num'] = np.array([30, 30, 30, 30, 30, 31, 31, 31, 31, 31])
        df_subj['test'] = np.arange(10)

        # Extract subject-specific data frame
        df_subj = get_df_subj(df_subj, 30)

        # Create expected data frame
        expected_df = pd.DataFrame(index=np.arange(5))
        expected_df['subj_num'] = np.array([31, 31, 31, 31, 31])  # expected bc. Python starts counting at 0
        expected_df['test'] = np.arange(5) + 5

        # Test output
        df_subj.equals(expected_df)

    def test_get_sim_est_err(self):
        """ This function tests the get_sim_est_err function used in the simulations """

        # Create input data frames (exp = 1)
        # ----------------------------------

        # df_subj with task data
        df_subj = pd.DataFrame(index=range(0, 10), dtype='float')
        df_subj['c_t'] = [1, 0, 0, 1, 0, 1, 0, 0, 1, 1]
        df_subj['mu_t'] = [150, 150, 150, 150, 150, 150, 150, 150, 150, 150]
        df_subj['cond'] = ['main_noPush', 'main_noPush', 'main_noPush', 'main_noPush', 'main_noPush',
                           'main_push', 'main_push', 'main_push', 'main_push', 'main_push']

        # df_data with predictions
        df_data = pd.DataFrame(index=range(0, 10), dtype='float')
        df_data['sim_b_t'] = [160, 140, 130, 160, 145, 160, 160, 180, 160, 150]

        # Run function and test output (exp = 1)
        # --------------------------------------
        sim_est_err_no_push, sim_est_err_push = get_sim_est_err(df_subj, df_data)
        self.assertEqual(sim_est_err_no_push, 11.666666666666666)
        self.assertEqual(sim_est_err_push, 20)

        sim_est_err = get_sim_est_err(df_subj, df_data, exp=1)
        self.assertEqual(sim_est_err, 15)

    def test_correct_push(self):
        """ This function tests the correct_push function used in the anchoring simulations """

        # Push is beyond 300 and has to be adjusted
        mu = 290
        sim_y_t = 20
        [sim_y_t, sim_z_t] = correct_push(mu, sim_y_t)
        self.assertEqual(sim_y_t, 10)
        self.assertEqual(sim_z_t, 300)

        # Push is below 0 and has to be adjusted
        mu = 10
        sim_y_t = -20
        [sim_y_t, sim_z_t] = correct_push(mu, sim_y_t)
        self.assertEqual(sim_y_t, -10)
        self.assertEqual(sim_z_t, 0)

        # Push should not be corrected
        mu = 150
        sim_y_t = 20
        [sim_y_t, sim_z_t] = correct_push(mu, sim_y_t)
        self.assertEqual(sim_y_t, 20)
        self.assertEqual(sim_z_t, 170)

        # Push should not be corrected
        mu = 150
        sim_y_t = -30
        [sim_y_t, sim_z_t] = correct_push(mu, sim_y_t)
        self.assertEqual(sim_y_t, -30)
        self.assertEqual(sim_z_t, 120)

    def test_safe_div(self):
        """ This function tests the safe_div function """

        # Division by zero, so return 0
        x = 1
        y = 0
        c = safe_div(x, y)
        self.assertEqual(c, 0)

        # Division by 2, so return regular result
        x = 1
        y = 2
        c = safe_div(x, y)
        self.assertEqual(c, 1 / 2)

        # Division by 2, so return regular result
        x = 0
        y = 2
        c = safe_div(x, y)
        self.assertEqual(c, 0 / 2)

    def test_safe_div_list(self):
        """ This function tests the safe_div_list function """

        # Length 1 and division by zero, so return 0
        x = np.array([1])
        y = np.array([0])
        c = safe_div_list(x, y)
        self.assertEqual(c, 0)

        # Length 3 and division by zero, so return 0
        x = np.array([1, 1, 1])
        y = np.array([0, 0, 0])
        c = safe_div_list(x, y)
        self.assertTrue(np.array_equal(c, np.array([0, 0, 0])))

        # Length 3 and 2 times division by zero, so return 0 in these 2 cases
        x = np.array([1, 1, 1])
        y = np.array([0, 2, 0])
        c = safe_div_list(x, y)
        self.assertTrue(np.array_equal(c, np.array([0, 1 / 2, 0])))

        # Length 3 and division by 2, so return regular results
        x = np.array([1, 1, 1])
        y = np.array([2, 2, 2])
        c = safe_div_list(x, y)
        self.assertTrue(np.array_equal(c, np.array([1 / 2, 1 / 2, 1 / 2])))

        # Length 3 and division by 2, so return regular results
        x = np.array([0, 0, 0])
        y = np.array([2, 2, 2])
        c = safe_div_list(x, y)
        self.assertTrue(np.array_equal(c, np.array([0 / 2, 0 / 2, 0 / 2])))

        # Length 3 and division by 0 in 2 cases, so return 0
        x = np.array([1, 0, 1])
        y = np.array([0, 0, 2])
        c = safe_div_list(x, y)
        self.assertTrue(np.array_equal(c, np.array([0, 0, 1 / 2])))

    def test_round_to_nearest_half_int(self):
        """ This function tests the function that rounds a number to the nearest half integer """

        # Test different example cases
        self.assertEqual(round_to_nearest_half_int(1.3), 1.5)
        self.assertEqual(round_to_nearest_half_int(2.6), 2.5)
        self.assertEqual(round_to_nearest_half_int(3.0), 3.0)
        self.assertEqual(round_to_nearest_half_int(4.1), 4.0)

    def test_compute_pers_prob(self):
        """ This function test the perseveration function of the RBM """

        # Test parameters that bring down perseveration to zero
        persprob = compute_persprob(-30, -1.5, 1)
        self.assertTrue(persprob < 1.e-10)

        # Test parameters that lead to 0.5
        persprob = compute_persprob(0, 0, 10)
        self.assertEqual(persprob, 0.5)

        # Test parameters that lead to low perseveration probability
        persprob = compute_persprob(0, -0.1, 10)
        self.assertEqual(persprob, 0.2689414213699951, 6)

        # Test parameters that lead to low perseveration probability
        persprob = compute_persprob(20, -0.1, 10)
        self.assertEqual(persprob, 0.7310585786300049, 6)

    def test_trial_cost_func(self):
        """ This function tests the cost function of the motor model """

        # Cost grid
        grid = np.linspace(0, 300, 301)

        # Test squared case
        dist = abs(grid - 0)
        cost = trial_cost_func(dist, 1, 2)
        self.assertTrue((cost == dist ** 2).all())

        # Test linear case
        cost = trial_cost_func(dist, 1, 1)
        self.assertTrue((cost == dist).all())

        # Test no-cost case
        cost = trial_cost_func(dist, 0, 1)
        self.assertTrue((cost == 0).all())

    def test_find_nearest(self):
        """ This function tests the function finds the closet value in an array """

        # Cost grid
        grid = np.linspace(0, 300, 301)

        # Test several example cases
        nearest_value, idx = find_nearest(grid, 150.1)
        self.assertEqual(nearest_value, 150)
        nearest_value, idx = find_nearest(grid, 138.6)
        self.assertEqual(nearest_value, 139)
        nearest_value, idx = find_nearest(grid, 0)
        self.assertEqual(nearest_value, 0)
        nearest_value, idx = find_nearest(grid, 300)
        self.assertEqual(nearest_value, 300)

    def test_get_mean_voi(self):
        """ This function tests the function that computes the mean variable of interest """

        # Create Data frame
        data = {'subj_num': [0, 1, 2, 3],
                'age_group': [1, 1, 2, 2],
                'c_t': [0, 1, 0, 0],
                'edge': [0, 0, 0, 0],
                'e_t': [2, 3, 4, 5],
                'pers': [1, 0, 1, 0],
                'motor_pers': [1, 0, 1, 0]}

        df_int = pd.DataFrame(data)

        # Test case voi == 1, where cp is discarded
        # -----------------------------------------

        mean_voi = get_mean_voi(df_int, 1)

        # Create Data frame
        data = {'index': [0, 2, 3],
                'subj_num': [0, 2, 3],
                'age_group': [1, 2, 2],
                'c_t': [0, 0, 0],
                'e_t': [2.0, 4.0, 5.0]}
        df_expected = pd.DataFrame(data)

        self.assertTrue(mean_voi.equals(df_expected))

        # Test case voi == 2, where cp is not discarded
        # ---------------------------------------------

        mean_voi = get_mean_voi(df_int, 2)

        # Create Data frame
        data = {'subj_num': [0, 1, 2, 3],
                'age_group': [1, 1, 2, 2],
                'pers': [1.0, 0.0, 1.0, 0.0]}
        df_expected = pd.DataFrame(data)

        self.assertTrue(mean_voi.equals(df_expected))

    @patch('scipy.stats.mannwhitneyu', new=Mock(side_effect=[[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]))
    @patch('scipy.stats.kruskal', new=Mock(side_effect=[[1, 1]]))
    @patch('al_utilities.mannwhitneyu_effectsize',
           new=Mock(side_effect=[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6]]))
    def test_get_stats_exp1_test1(self):
        """ This function tests the function that computes the stats

            Here: experiment = 1 and test = 1
        """

        # Create voi data frame as input
        data = {'subj_num': [0, 1, 2, 3],
                'age_group': [1, 1, 2, 2],
                'pers': [1.0, 0.0, 1.0, 0.0]}
        voi = pd.DataFrame(data)

        # Run function
        exp = 1
        voi_name = 'pers'
        test = 1
        desc, stat, effect_size = get_stats(voi, exp, voi_name, test)

        # Test descriptives
        # -----------------
        data = {'median': [0.5, 0.5],
                'q1': [0.25, 0.25],
                'q3': [0.75, 0.75]}
        exp_desc = pd.DataFrame(data)
        exp_desc.index.name = 'age_group'
        exp_desc = exp_desc.rename({0: 'ch', 1: 'ad', 2: 'ya', 3: 'oa'}, axis='index')

        self.assertTrue(desc.equals(exp_desc))

        # Test stats
        # ----------
        data = {'p': [1, 4, 1, 2, 5, 6, 3],
                'stat': [1, 4, 1, 2, 5, 6, 3]}
        exp_stat = pd.DataFrame(data)
        exp_stat.index.name = 'test'
        exp_stat = exp_stat.rename({0: 'kw', 1: 'ch_ad', 2: 'ch_ya', 3: 'ch_oa', 4: 'ad_ya', 5: "ad_oa", 6: "ya_oa"},
                                   axis='index')

        self.assertTrue(stat.equals(exp_stat))

        # Test effect sizes
        # -----------------
        data = {'cl': [4, 1, 2, 5, 6, 3],
                'bi_c': [4, 1, 2, 5, 6, 3],
                'r_sq': [4, 1, 2, 5, 6, 3],
                'r': [4, 1, 2, 5, 6, 3]}

        exp_effect_size = pd.DataFrame(data)
        exp_effect_size.index.name = 'type'
        exp_effect_size = exp_effect_size.rename(
            {0: 'ch_ad', 1: 'ch_ya', 2: 'ch_oa', 3: 'ad_ya', 4: "ad_oa", 5: "ya_oa"}, axis='index')

        self.assertTrue(effect_size.equals(exp_effect_size))

    @patch('scipy.stats.mannwhitneyu', new=Mock(side_effect=[[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]))
    @patch('scipy.stats.kruskal', new=Mock(side_effect=[[1, 1]]))
    @patch('al_utilities.mannwhitneyu_effectsize',
           new=Mock(side_effect=[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6]]))
    def test_get_stats_exp2_test1(self):
        """ This function tests the function that computes the stats

            Here: experiment = 2 and test = 1
        """

        # Create voi data frame as input
        data = {'subj_num': [0, 1, 2, 3],
                'age_group': [1, 1, 2, 2],
                'pers': [1.0, 0.0, 1.0, 0.0]}
        voi = pd.DataFrame(data)

        # Run function
        exp = 2
        voi_name = 'pers'
        test = 1
        desc, stat, effect_size = get_stats(voi, exp, voi_name, test)

        # Test descriptives not necessary (same as above)
        # -----------------------------------------------

        # Test stats
        # ----------
        data = {'p': [1, np.nan, 1, 2, np.nan, np.nan, 3],
                'stat': [1, np.nan, 1, 2, np.nan, np.nan, 3]}
        exp_stat = pd.DataFrame(data)
        exp_stat.index.name = 'test'
        exp_stat = exp_stat.rename({0: 'kw', 1: 'ch_ad', 2: 'ch_ya', 3: 'ch_oa', 4: 'ad_ya', 5: "ad_oa", 6: "ya_oa"},
                                   axis='index')

        self.assertTrue(stat.equals(exp_stat))

        # Test effect sizes
        # -----------------
        data = {'cl': [np.nan, 1, 2, np.nan, np.nan, 3],
                'bi_c': [np.nan, 1, 2, np.nan, np.nan, 3],
                'r_sq': [np.nan, 1, 2, np.nan, np.nan, 3],
                'r': [np.nan, 1, 2, np.nan, np.nan, 3]}

        exp_effect_size = pd.DataFrame(data)
        exp_effect_size.index.name = 'type'
        exp_effect_size = exp_effect_size.rename(
            {0: 'ch_ad', 1: 'ch_ya', 2: 'ch_oa', 3: 'ad_ya', 4: "ad_oa", 5: "ya_oa"}, axis='index')

        self.assertTrue(effect_size.equals(exp_effect_size))

    @patch('scipy.stats.wilcoxon', new=Mock(
        side_effect=[Wilcoxon(1, 1, 1), Wilcoxon(2, 2, 2),
                     Wilcoxon(3, 3, 3), Wilcoxon(4, 4, 4)]))
    @patch('al_utilities.wilcoxon_sign_rank_effectsize',
           new=Mock(side_effect=[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]))
    def test_get_stats_exp1_test2(self):
        """ This function tests the function that computes the stats

            Here: experiment = 1 and test = 2
        """

        # Create voi data frame as input
        data = {'subj_num': [0, 1, 2, 3],
                'age_group': [1, 1, 2, 2],
                'pers': [1.0, 0.0, 1.0, 0.0]}
        voi = pd.DataFrame(data)

        # Run function
        exp = 1
        voi_name = 'pers'
        test = 2
        desc, stat, effect_size = get_stats(voi, exp, voi_name, test)

        # Test descriptives not necessary (same as above)
        # -----------------------------------------------

        # Test stats
        # ----------
        data = {'p': [1, 4, 2, 3],
                'stat': [1, 4, 2, 3],
                'z': [1, 4, 2, 3]}
        exp_stat = pd.DataFrame(data)
        exp_stat.index.name = 'age_group'
        exp_stat = exp_stat.rename({0: 'ch', 1: 'ad', 2: 'ya', 3: 'oa'},
                                   axis='index')

        self.assertTrue(stat.equals(exp_stat))

        # Test effect sizes
        # -----------------
        data = {'cl': [1, 4, 2, 3],
                'bi_c': [1, 4, 2, 3],
                'r_sq': [1, 4, 2, 3],
                'r': [1, 4, 2, 3]}

        exp_effect_size = pd.DataFrame(data)
        exp_effect_size.index.name = 'type'
        exp_effect_size = exp_effect_size.rename(
            {0: 'ch', 1: 'ad', 2: 'ya', 3: 'oa'}, axis='index')

        self.assertTrue(effect_size.equals(exp_effect_size))

    @patch('scipy.stats.wilcoxon', new=Mock(
        side_effect=[Wilcoxon(1, 1, 1), Wilcoxon(2, 2, 2),
                     Wilcoxon(3, 3, 3), Wilcoxon(4, 4, 4)]))
    @patch('al_utilities.wilcoxon_sign_rank_effectsize',
           new=Mock(side_effect=[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]))
    def test_get_stats_exp2_test2(self):
        """ This function tests the function that computes the stats

            Here: experiment = 1 and test = 1
        """

        # Create voi data frame as input
        data = {'subj_num': [0, 1, 2, 3],
                'age_group': [1, 1, 2, 2],
                'pers': [1.0, 0.0, 1.0, 0.0]}
        voi = pd.DataFrame(data)

        # Run function
        exp = 2
        voi_name = 'pers'
        test = 2
        desc, stat, effect_size = get_stats(voi, exp, voi_name, test)

        # Test descriptives not necessary (same as above)
        # -----------------------------------------------

        # Test stats
        # ----------
        data = {'p': [1, np.nan, 2, 3],
                'stat': [1, np.nan, 2, 3],
                'z': [1, np.nan, 2, 3]}
        exp_stat = pd.DataFrame(data)
        exp_stat.index.name = 'age_group'
        exp_stat = exp_stat.rename({0: 'ch', 1: 'ad', 2: 'ya', 3: 'oa'},
                                   axis='index')

        self.assertTrue(stat.equals(exp_stat))

        # Test effect sizes
        # -----------------
        data = {'cl': [1, np.nan, 2, 3],
                'bi_c': [1, np.nan, 2, 3],
                'r_sq': [1, np.nan, 2, 3],
                'r': [1, np.nan, 2, 3]}

        exp_effect_size = pd.DataFrame(data)
        exp_effect_size.index.name = 'type'
        exp_effect_size = exp_effect_size.rename(
            {0: 'ch', 1: 'ad', 2: 'ya', 3: 'oa'}, axis='index')

        self.assertTrue(effect_size.equals(exp_effect_size))

    def test_mannwhitneyu_effectsize(self):
        """ This function tests the function that computes effect sizes for the Mann-Whitney U test """

        # Input values, usually based on Mann-Whitney U test
        u1 = 860
        nx = 30
        ny = 30
        p = 1.e-3

        # Apply function
        common_language, rank_biserial, r_square, r = mannwhitneyu_effectsize(u1, nx, ny, p)

        self.assertEqual(common_language, 0.9555555555555556)
        self.assertEqual(rank_biserial, 0.9111111111111112)
        self.assertEqual(r_square, 0.6108934426229509)
        self.assertEqual(r, -0.7815967263384301)

    @patch('math.isclose', new=Mock(return_value=1))
    def test_sign_rank_effectsize(self):
        """ This function test the function that computes effect sizes for the sign-rank test """

        # In put values, usually based on sign-rank test
        n = 30
        stat = 30
        z = -3
        p = 1.7e-6

        # Apply function
        common_language, rank_biserial, r_square, r = wilcoxon_sign_rank_effectsize(n, stat, z, p)

        self.assertEqual(common_language, 0.9354838709677419)
        self.assertEqual(rank_biserial, -0.8709677419354838)
        self.assertEqual(r_square, 0.29999999999999993)
        self.assertEqual(r, -0.5477225575051661)

    def test_compute_rob_reg_effect_size(self):
        """ This function test the function that computes effect sizes for the robust regression """

        # Call function with mock regression results
        res = MockRes()

        # Data frame with mock data
        data = pd.DataFrame({'d': np.arange(10)})

        # Apply function
        r_squared, r = compute_rob_reg_effect_size(res, data)

        self.assertEqual(r_squared, 0.9654545454545455)
        self.assertEqual(r, 0.982575465526463)

    def test_get_sub_stats(self):
        """ This function computes the function that computes subject stats for the methods """

        # BIDS data frame for input
        data = {'participant_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                'age': [8, 9, 10, 16, 17, 18, 27, 28, 29, 65, 66, 67],
                'sex': ['female', 'male', 'female', 'female', 'male', 'female', 'female', 'male', 'female', 'female',
                        'male', 'female'],
                'age_group': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
                }
        bids_participants = pd.DataFrame(data)

        # Initialize data frame with stats
        df_participants = pd.DataFrame(index=[1, 2], columns=['min_age_ch', 'min_age_ad', 'min_age_ya', 'min_age_oa',
                                                              'max_age_ch', 'max_age_ad', 'max_age_ya', 'max_age_oa',
                                                              'median_age_ch', 'median_age_ad', 'median_age_ya',
                                                              'median_age_oa', 'n_ch', 'n_ad', 'n_ya', 'n_oa',
                                                              'n_female_ch', 'n_female_ad', 'n_female_ya',
                                                              'n_female_oa'])

        # Run function for experiments 1 and 2
        df_participants = get_sub_stats(bids_participants, df_participants)
        df_participants = get_sub_stats(bids_participants, df_participants, exp=2)

        # Expected results
        data = {'min_age_ch': [8, 8], 'min_age_ad': [16, np.nan], 'min_age_ya': [27, 27], 'min_age_oa': [65, 65],
                'max_age_ch': [10, 10], 'max_age_ad': [18, np.nan], 'max_age_ya': [29, 29], 'max_age_oa': [67, 67],
                'median_age_ch': [9, 9], 'median_age_ad': [17, np.nan], 'median_age_ya': [28, 28],
                'median_age_oa': [66, 66],
                'n_ch': [3, 3], 'n_ad': [3, np.nan], 'n_ya': [3, 3], 'n_oa': [3, 3],
                'n_female_ch': [2, 2], 'n_female_ad': [2, np.nan], 'n_female_ya': [2, 2], 'n_female_oa': [2, 2]}
        df_part_exp = pd.DataFrame(data)
        df_part_exp = df_part_exp.rename({0: 1, 1: 2}, axis='index')
        df_part_exp = df_part_exp.astype(object)

        self.assertTrue(df_participants.equals(df_part_exp))

    @patch('statsmodels.api.OLS.fit', new=OLS)
    @patch('al_utilities.get_df_subj',
           new=Mock(return_value=pd.DataFrame({'delta_t': [1], 'a_t': [1], 'age_group': [2]})))
    def test_compute_average_LR(self):
        """ This function tests the function that computes average learning rates """

        # Run function
        n_subj = 1
        df_exp = 1
        df_alpha = compute_average_LR(n_subj, df_exp)

        # Compute expected output
        df_alpha_exp = pd.DataFrame({'alpha': [1.0], 'age_group': [2.0], 'subj_num': [1.0]})

        self.assertTrue(df_alpha.equals(df_alpha_exp))

    @patch('statsmodels.formula.api.ols', new=OLS)
    def test_compute_anchoring_bias(self):
        """ This function tests the function that computes the anchoring bias  """

        # Run function
        n_subj = 1
        alll_data_anchor = pd.DataFrame({'sim_a_t': [1], 'sim_y_t': [2], 'delta_t': [3], 'subj_num': [1],
                                         'cond': 'main_push', 'age_group': [3]})
        df_reg = compute_anchoring_bias(n_subj, alll_data_anchor)

        # Compute expected output
        df_reg_exp = pd.DataFrame({'alpha': [1.0], 'bucket_bias': [2.0], 'age_group': [3.0]})

        self.assertTrue(df_reg.equals(df_reg_exp))


# Run unit test
if __name__ == '__main__':
    unittest.main()
