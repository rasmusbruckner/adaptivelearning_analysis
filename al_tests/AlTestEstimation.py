""" Al Test Estimation: Estimation unit and integration tests """

import unittest
from unittest.mock import patch
from unittest.mock import Mock
import numpy as np
import pandas as pd
from Estimation.AlEstimation import AlEstimation
from Estimation.AlEstVars import AlEstVars
from AlAgentVarsRbm import AgentVars


class MockMinimizeRes:
    """ This class definition is used to mock-out the scipy minimize function to determine the mock output """

    def __init__(self):

        self.fun = 100
        self.x = np.array(np.arange(9))


def minimize(*args, **kwargs):
    """ This function is used to mock-out the scipy minimize function

    :param args: Input arguments
    :param kwargs: Input arguments
    :return: None
    """
    return MockMinimizeRes()


class TestEstimation(unittest.TestCase):
    """ This class definition implements the Estimation unit tests
        and an integration test in order to test the parameter estimation

        We have the following test functions

            test_estimation_init: Estimation-object initialization
            test_parallel_estimation: Parallelization routines
            test_model_estimation: Model-estimation parameters and subject information
            test_llh: Computed likelihood
            test_llh_nan: Computed likelihood for the case that a likelihood = nan
            test_llh_np_prior: Computed likelihood for the case that no prior is used
            test_bic: Bayesian information criterion
            test_integration_estimation_exp1: Integration test based on first participant,
                specifically for experiment 1
            test_integration_estimation_exp2: Integration test based on first participant,
                specifically for experiment 2
    """

    @staticmethod
    def model_estimation(*args):
        """ This function is used as a model_estimation mock function

        :param args: Input arguments
        :return: Mocked estimation output
        """
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    def test_estimation_init(self):
        """ This function tests the estimation initialization based on EstVars """

        # Initialize estimation based on est_vars
        est_vars = AlEstVars()
        estimation = AlEstimation(est_vars)

        # Test variable name strings
        self.assertEqual(estimation.omikron_0, 'omikron_0')
        self.assertEqual(estimation.omikron_1, 'omikron_1')
        self.assertEqual(estimation.b_0, 'b_0')
        self.assertEqual(estimation.b_1, 'b_1')
        self.assertEqual(estimation.h, 'h')
        self.assertEqual(estimation.s, 's')
        self.assertEqual(estimation.u, 'u')
        self.assertEqual(estimation.q, 'q')
        self.assertEqual(estimation.sigma_H, 'sigma_H')
        self.assertEqual(estimation.d, 'd')

        # Test starting points
        self.assertEqual(estimation.omikron_0_x0, 5.0)
        self.assertEqual(estimation.omikron_1_x0, 0.0)
        self.assertEqual(estimation.b_0_x0, 0.5)
        self.assertEqual(estimation.b_1_x0, 0)
        self.assertEqual(estimation.h_x0, 0.1)
        self.assertEqual(estimation.s_x0, 0.999)
        self.assertEqual(estimation.u_x0, 0.0)
        self.assertEqual(estimation.q_x0, 0.0)
        self.assertEqual(estimation.sigma_H_x0, 10.0)
        self.assertEqual(estimation.d_x0, 0.0)

        # Test starting-point range
        self.assertEqual(estimation.omikron_0_x0_range, (1, 10))
        self.assertEqual(estimation.omikron_1_x0_range, (0.001, 1))
        self.assertEqual(estimation.b_0_x0_range, (-30, 30))
        self.assertEqual(estimation.b_1_x0_range, (-1.5, 1))
        self.assertEqual(estimation.h_x0_range, (0.001, 0.99))
        self.assertEqual(estimation.s_x0_range, (0.001, 0.99))
        self.assertEqual(estimation.u_x0_range, (1, 10))
        self.assertEqual(estimation.q_x0_range, (0, 0.1))
        self.assertEqual(estimation.sigma_H_x0_range, (1, 32))
        self.assertEqual(estimation.d_x0_range, (-1, 1))

        # Test boundaries
        self.assertEqual(estimation.omikron_0_bnds, (0.1, 10))
        self.assertEqual(estimation.omikron_1_bnds, (0.001, 1))
        self.assertEqual(estimation.b_0_bnds, (-30, 30))
        self.assertEqual(estimation.b_1_bnds, (-1.5, 1))
        self.assertEqual(estimation.h_bnds, (0.001, 0.99))
        self.assertEqual(estimation.s_bnds, (0.001, 1))
        self.assertEqual(estimation.u_bnds, (-2, 15))
        self.assertEqual(estimation.q_bnds, (-0.5, 0.5))
        self.assertEqual(estimation.sigma_H_bnds, (0, 32))
        self.assertEqual(estimation.d_bnds, (-1, 1))

        # Test free parameters and fixed parameter values
        which_vars = {'omikron_0': True,
                      'omikron_1': True,
                      'b_0': True,
                      'b_1': True,
                      'h': True,
                      's': True,
                      'u': True,
                      'q': True,
                      'sigma_H': True,
                      'd': False}

        fixed_mod_coeffs = {'omikron_0': 10.0,
                            'omikron_1': 0.0,
                            'b_0': -30,
                            'b_1': -1.5,
                            'h': 0.1,
                            's': 1.0,
                            'u': 0.0,
                            'q': 0.0,
                            'sigma_H': 0.0,
                            'd': 0.0}

        self.assertEqual(estimation.which_vars, which_vars)
        self.assertEqual(estimation.fixed_mod_coeffs, fixed_mod_coeffs)

        # Test other attributes
        self.assertTrue(np.isnan(estimation.n_subj))
        self.assertEqual(estimation.n_ker, 4)
        self.assertEqual(estimation.which_exp, 1)
        self.assertTrue(estimation.rand_sp)
        self.assertEqual(estimation.n_sp, 10)
        self.assertTrue(estimation.use_prior)

    @patch.object(AlEstimation, 'model_estimation', new=model_estimation)
    def test_parallel_estimation(self):
        """ This function tests the parallelization routines.

            We mock out "model_estimation" and use a patched function defined above that returns
            pre-specified parameter estimates.
        """

        # Load input variables
        # --------------------

        # Load data from first experiment
        df_exp1 = pd.read_pickle('al_data/data_prepr_1.pkl')
        n_subj_exp1 = 4  # number of participants

        # Call AgentVars Object
        agent_vars = AgentVars()

        # Call AlEstVars object
        est_vars = AlEstVars()
        est_vars.n_subj = n_subj_exp1  # number of subjects

        # Call AlEstimation object
        al_estimation = AlEstimation(est_vars)

        # Estimate parameters
        results_df = al_estimation.parallel_estimation(df_exp1, agent_vars)

        # Create expected results data frame
        output = [self.model_estimation(),  self.model_estimation(),  self.model_estimation(), self.model_estimation()]
        columns = [est_vars.omikron_0, est_vars.omikron_1, est_vars.b_0, est_vars.b_1, est_vars.h,  est_vars.s,
                   est_vars.u, est_vars.q, est_vars.sigma_H, 'llh', 'BIC', 'age_group', 'subj_num']
        expected_df = pd.DataFrame(output, columns=columns)

        # Test function output
        self.assertTrue(expected_df.equals(results_df))

    @patch('Estimation.AlEstimation.minimize', new=minimize)
    def test_model_estimation(self):
        """ This function tests the model_estimation function

            We mock out "scipy minimize" using  a patched function defined above to have
            pre-specified parameter estimates.
        """

        # Call AlEstVars object
        est_vars = AlEstVars()
        est_vars.n_sp = 1

        # Call AlEstimation object
        al_estimation = AlEstimation(est_vars)

        # Load data from first experiment as in "al_estimation.py"
        df_exp1 = pd.read_pickle('al_data/data_prepr_1.pkl')
        df = df_exp1[(df_exp1['subj_num'] == 1)].copy()

        # Call AgentVars Object
        agent_vars = AgentVars()

        # Get model estimation results
        results_list = al_estimation.model_estimation(df, agent_vars)

        # Created expected results list
        expected_list = self.model_estimation()[:9]
        expected_list.append(100)
        expected_list.append(-126.93903402377997)  # Futuretodo: mock BIC out in this context
        expected_list.append(1.0)
        expected_list.append(1.0)

        # Test function output
        self.assertEqual(results_list, expected_list)

    @patch('Estimation.AlEstimation.task_agent_int', new=Mock(return_value=(np.array([-1, -2, -3]), 2)))
    def test_llh(self):
        """ This function tests the likelihood function

            We mock out "task_agent_int" to work with pre-specified likelihood values.
        """

        # Call AlEstVars object
        est_vars = AlEstVars()

        # Call AlEstimation object
        al_estimation = AlEstimation(est_vars)

        # Call agent variables
        agent_vars = AgentVars()

        # Load data from first experiment as in "al_estimation.py"
        df_exp1 = pd.read_pickle('al_data/data_prepr_1.pkl')
        df = df_exp1[(df_exp1['subj_num'] == 1)].copy()

        # Set current coefficients
        coeffs = np.array(np.arange(9))

        # Compute likelihood
        llh_sum = al_estimation.llh(coeffs, df, agent_vars)

        # Test function output
        self.assertEqual(llh_sum, 9.248376445638772)

    @patch('Estimation.AlEstimation.task_agent_int', new=Mock(return_value=(np.array([-1, np.nan, -3]), 2)))
    def test_llh_nan(self):
        """ This function tests the likelihood function

            We mock out "task_agent_int" to work with pre-specified likelihood values.
            Here we test if the function returns "nan" if a likelihood value is equal to nan.
        """

        # Call AlEstVars object
        est_vars = AlEstVars()

        # Call AlEstimation object
        al_estimation = AlEstimation(est_vars)

        # Call agent variables
        agent_vars = AgentVars()

        # Load data from first experiment as in "al_estimation.py"
        df_exp1 = pd.read_pickle('al_data/data_prepr_1.pkl')
        df = df_exp1[(df_exp1['subj_num'] == 1)].copy()

        # Set current coefficients
        coeffs = np.array(np.arange(9))

        # Compute likelihood
        llh_sum = al_estimation.llh(coeffs, df, agent_vars)

        # Test function output
        self.assertTrue(np.isnan(llh_sum))

    @patch('Estimation.AlEstimation.task_agent_int', new=Mock(return_value=(np.array([-1, -2, -3]), 2)))
    def test_llh_np_prior(self):
        """ This function tests the likelihood function

            We mock out "task_agent_int" to work with pre-specified likelihood values. Here we
            test the function output when no prior for uncertainty underestimation is used.
        """

        # Call AlEstVars object
        est_vars = AlEstVars()
        est_vars.use_prior = False

        # Call AlEstimation object
        al_estimation = AlEstimation(est_vars)

        # Call agent variables
        agent_vars = AgentVars()

        # Load data from first experiment as in "al_estimation.py"
        df_exp1 = pd.read_pickle('al_data/data_prepr_1.pkl')
        df = df_exp1[(df_exp1['subj_num'] == 1)].copy()

        # Set current coefficients
        coeffs = np.array(np.arange(9))

        # Compute likelihood
        llh_sum = al_estimation.llh(coeffs, df, agent_vars)

        # Test function output
        self.assertEqual(llh_sum, 6)

    def test_bic(self):
        """ This function tests the computation of the Bayesian information criterion """

        llh = -120
        n_params = 9
        n_trials = 398
        est_vars = AlEstVars()
        al_estimation = AlEstimation(est_vars)
        bic = al_estimation.compute_bic(llh, n_params, n_trials)

        # Test function output
        self.assertEqual(bic, 93.06096597622003)

    def test_integration_estimation_exp1(self):
        """ This function performs an integration test for experiment 1

            The expected test output was generated on an Ubuntu machine.
        """

        # Load input variables
        # -----------------------

        # Load data from first experiment
        df_exp1 = pd.read_pickle('al_data/data_prepr_1.pkl')

        n_subj_exp1 = 4  # number of participants

        # Call AgentVars Object
        agent_vars = AgentVars()

        # Call AlEstVars object
        est_vars = AlEstVars()
        est_vars.n_subj = n_subj_exp1  # number of subjects
        est_vars.n_ker = 4  # number of kernels for estimation
        est_vars.n_sp = 1  # number of random starting points
        est_vars.rand_sp = True  # use random starting points
        est_vars.use_prior = True  # use weakly informative prior for uncertainty underestimation

        # Free parameters
        est_vars.which_vars = {est_vars.omikron_0: True,  # motor noise
                               est_vars.omikron_1: True,  # learning-rate noise
                               est_vars.b_0: True,  # logistic-function intercept
                               est_vars.b_1: True,  # logistic-function slope
                               est_vars.h: True,  # hazard rate
                               est_vars.s: True,  # surprise sensitivity
                               est_vars.u: True,  # uncertainty underestimation
                               est_vars.q: True,  # reward bias
                               est_vars.sigma_H: True,  # catch trials
                               est_vars.d: False,  # bucket shift
                               }

        # Specify that experiment 1 is modeled
        est_vars.which_exp = 1

        # Call AlEstimation object
        al_estimation = AlEstimation(est_vars)

        # Estimate parameters
        results_df = al_estimation.parallel_estimation(df_exp1, agent_vars)

        # savename = 'al_tests/test_estimates_exp1.pkl'
        # results_df.to_pickle(savename)

        # Load test data
        test_data = pd.read_pickle('al_tests/test_estimates_exp1.pkl')

        # Test function output
        self.assertTrue(test_data.equals(results_df))

    def test_integration_estimation_exp2(self):
        """ This function performs an integration test for experiment 2

            The expected test output was generated on an Ubuntu machine.
        """

        # Load input variables
        # -----------------------

        # Load data from first experiment
        df_exp2 = pd.read_pickle('al_data/data_prepr_2.pkl')
        n_subj_exp2 = 4  # number of participants

        # Call AgentVars Object
        agent_vars = AgentVars()

        # Call AlEstVars object
        est_vars = AlEstVars()
        est_vars.n_subj = n_subj_exp2  # number of subjects
        est_vars.n_ker = 4  # number of kernels for estimation
        est_vars.n_sp = 1  # number of random starting points
        est_vars.rand_sp = True  # use random starting points
        est_vars.use_prior = True  # use weakly informative prior for uncertainty underestimation

        # Free parameters
        est_vars.which_vars = {est_vars.omikron_0: True,  # motor noise
                               est_vars.omikron_1: True,  # learning-rate noise
                               est_vars.b_0: True,  # logistic-function intercept
                               est_vars.b_1: True,  # logistic-function slope
                               est_vars.h: True,  # hazard rate
                               est_vars.s: True,  # surprise sensitivity
                               est_vars.u: True,  # uncertainty underestimation
                               est_vars.q: False,  # reward bias
                               est_vars.sigma_H: True,  # catch trials
                               est_vars.d: True,  # bucket shift
                               }

        # Specify that experiment 1 is modeled
        est_vars.which_exp = 2

        # Call AlEstimation object
        al_estimation = AlEstimation(est_vars)

        # Estimate parameters
        results_df = al_estimation.parallel_estimation(df_exp2, agent_vars)

        # savename = 'al_tests/test_estimates_exp2.pkl'
        # results_df.to_pickle(savename)

        # Load test data
        test_data = pd.read_pickle('al_tests/test_estimates_exp2.pkl')

        # Test function output
        self.assertTrue(test_data.equals(results_df))


# Run unit test
if __name__ == '__main__':
    unittest.main()
