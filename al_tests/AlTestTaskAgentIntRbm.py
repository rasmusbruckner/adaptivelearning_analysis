""" Task-Agent Interaction RBM: Unit and integration tests

    Tests work on Rasmus Bruckner's Ubuntu laptop.
    Integration tests can fail on other operating systems.

"""

import numpy as np
import unittest
from unittest.mock import patch, Mock
import pandas as pd
from itertools import compress
from AlAgentVarsRbm import AgentVars
from AlAgentRbm import AlAgent
from Estimation.AlEstVars import AlEstVars
from al_task_agent_int_rbm import task_agent_int


class TestTaskAgentInt(unittest.TestCase):
    """ This class definition implements the task_agent_int unit tests
        and two integration tests based on a participant data set from
        experiment 1 and 2.

        We run the following tests:
            - test_first_trial_exp1: First trial of experiment 1
            - test_first_trial_exp1_sim: First trial of experiment 1 (simulation)
            - test_second_trial_exp1: Second trial of a block in experiment 1
            - test_second_trial_exp1_sim: Second trial of a block in experiment 1 (simulation)
            - test_last_block_trial_exp1: Last trial of a block in experiment 1
            - test_last_block_trial_exp1_sim: Last trial of a block in experiment 1 (simulation)
            - test_first_trial_exp2: First trial of experiment 2
            - test_first_trial_exp2_sim: First trial of experiment 2 (simulation)
            - test_second_trial_exp2: Second trial of a block in experiment 2
            - test_second_trial_exp2_sim: Second trial of a block in experiment 1 (simulation)
            - test_last_block_trial_exp2: Last trial of a block in experiment 2
            - test_pers_trial: Perseveration trial in experiment 1
            - test_integration_task_agent_int_exp1: Integration test based on data set
                of first participant (experiment 1)
            - test_integration_task_agent_int_exp2: Integration test based on data set
                of first participant (experiment 2)

        In the unit tests, we mock out the agent and use predefined values
            as defined in function mock_learn.
            - https://stackoverflow.com/questions/34406848/mocking-a-class-method-and-changing-some-object-attributes-in-python
            - https://stackoverflow.com/questions/38579535/how-to-supply-a-mock-class-method-for-python-unit-test

        The tests were created on Ubuntu and in the integration tests, the data frames generated on a different
        operating system might lead to fail.
    """

    def mock_learn(self, *args):
        """ This function mocks out the agent's learning function

        :param args: All input arguments
        :return: None
        """

        self.tau_t = 0.2
        self.mu_t = 122
        self.omega_t = 0.7
        self.alpha_t = 0.35
        self.a_t = -30
        self.sigma_t_sq = 50

    @patch('__main__.AlAgent.learn', new=mock_learn)
    def test_first_trial_exp1(self):
        """ This function implements a unit test of the task-agent interaction
            in the first trial of experiment 1.

            The test covers the first trial with the new_block[t] case,
            where the agent is initialized.
        """

        # Load function input
        which_exp, df, agent, agent_vars, sel_coeffs = self.load_default_input()

        # Extract first and second trial for unit test
        df_trial = df[0:3].copy()
        df_trial.loc[2, 'new_block'] = 1

        # Run first trial
        llh_mix, df_data = task_agent_int(which_exp, df_trial, agent, agent_vars, sel_coeffs)

        # Test function output
        self.assertEqual(df_data['delta_t'][0], -37.0)  # input value
        self.assertEqual(df_data['tau_t'][0], 0.5)  # new block initialization
        self.assertEqual(df_data['mu_t'][0], 122)  # belief according to mock function
        self.assertEqual(df_data['omega_t'][0], 0.7)  # omega mock function
        self.assertEqual(df_data['alpha_t'][0], 0.35)  # alpha mock function
        self.assertEqual(df_data['a_t_hat'][0], -30)  # update mock function
        self.assertEqual(df_data['sigma_t_sq'][0], 100)  # new block initialization
        self.assertAlmostEqual(llh_mix[0], -4.841523619865609, 6)  # computed llh

    @patch('__main__.AlAgent.learn', new=mock_learn)
    @patch('numpy.random.normal', new=Mock(return_value=1))
    @patch('numpy.random.binomial', new=Mock(return_value=0))
    def test_first_trial_exp1_sim(self):
        """ This function implements a unit test of the task-agent interaction
            in the first trial of experiment 1 when data are simulated

            The test covers the first trial with the new_block[t] case,
            where the agent is initialized. We additionally mock-out the
            Gaussian and binomial distributions to have consistent
            simulated updates sim_a_t.
        """

        # Load function input
        which_exp, df, agent, agent_vars, sel_coeffs = self.load_default_input()

        # Extract first and second trial for unit test
        df_trial = df[0:3].copy()
        df_trial.loc[2, 'new_block'] = 1

        # Run first trial
        llh_mix, df_data = task_agent_int(which_exp, df_trial, agent, agent_vars, sel_coeffs, sim=True)

        # Test function output
        self.assertEqual(df_data['delta_t'][0], -3.0)  # 147 (outcome) - 150 (belief trial 1)
        self.assertEqual(df_data['tau_t'][0], 0.5)  # new block initialization
        self.assertEqual(df_data['mu_t'][0], 122)  # belief according to mock function
        self.assertEqual(df_data['omega_t'][0], 0.7)  # omega mock function
        self.assertEqual(df_data['alpha_t'][0], 0.35)  # alpha mock function
        self.assertEqual(df_data['a_t_hat'][0], -30)  # update mock function
        self.assertAlmostEqual(llh_mix[0], -4.841523619865609, 6)  # computed llh
        self.assertEqual(df_data['sim_a_t'][0], 1.0)  # update mock Gaussian
        self.assertEqual(df_data['sim_b_t'][0], 150.0)  # initial belief
        self.assertEqual(df_data['sim_y_t'][0], 0.0)  # no push
        self.assertEqual(df_data['sim_z_t'][0], 150.0)  # no push
        self.assertTrue(df_data['cond'][0] == "main")  # main condition
        self.assertEqual(df_data['sigma'][0], 10)  # low-noise condition
        self.assertEqual(df_data['sigma_t_sq'][0], 100)  # new block initialization

    @patch('__main__.AlAgent.learn', new=mock_learn)
    def test_second_trial_exp1(self):
        """ This function implements a unit test of the task-agent interaction
            in the second trial of experiment 1

            The test covers the second trial, which is a regular trial without
            new_block[t] or new_block[t+1].
        """

        # Load function input
        which_exp, df, agent, agent_vars, sel_coeffs = self.load_default_input()

        # Extract second and third trial for unit test
        df_trial = df[1:4].copy()
        df_trial.loc[3, 'new_block'] = 1
        df_trial = df_trial.reset_index(drop=True)

        # Run second trial
        llh_mix, df_data = task_agent_int(which_exp, df_trial, agent, agent_vars, sel_coeffs)

        # Test function output
        self.assertEqual(df_data['delta_t'][0], 8.0)  # input value
        self.assertEqual(df_data['tau_t'][0], 0.5)  # new block initialization
        self.assertEqual(df_data['mu_t'][0], 122)  # belief according to mock function
        self.assertEqual(df_data['omega_t'][0], 0.7)  # omega mock function
        self.assertEqual(df_data['alpha_t'][0], 0.35)  # alpha mock function
        self.assertEqual(df_data['a_t_hat'][0], -30)  # update mock function
        self.assertEqual(df_data['sigma_t_sq'][0], 100)  # new block initialization
        self.assertAlmostEqual(llh_mix[0], -23.718769305380253, 6)  # computed llh

    @patch('__main__.AlAgent.learn', new=mock_learn)
    @patch('numpy.random.normal', new=Mock(return_value=1))
    @patch('numpy.random.binomial', new=Mock(return_value=0))
    def test_second_trial_exp1_sim(self):
        """ This function implements a unit test of the task-agent interaction
            in the second trial when data are simulated

            The test covers the second trial, which is a regular trial without
            new_block[t] or new_block[t+1]. We additionally mock-out the
            Gaussian and binomial distributions to have consistent
            simulated updates sim_a_t.
        """

        # Load function input
        which_exp, df, agent, agent_vars, sel_coeffs = self.load_default_input()

        # Extract second and third trial for unit test
        df_trial = df[0:4].copy()
        df_trial.loc[3:4, 'new_block'] = 1
        df_trial = df_trial.reset_index(drop=True)

        # Run second trial
        llh_mix, df_data = task_agent_int(which_exp, df_trial, agent, agent_vars, sel_coeffs, sim=True)

        # Test function output
        self.assertEqual(df_data['delta_t'][1], 2.0)  # 153 (outcome) - 151 (belief trial 1)
        self.assertEqual(df_data['tau_t'][1], 0.2)  # tau mock function
        self.assertEqual(df_data['mu_t'][1], 122)  # belief according to mock function
        self.assertEqual(df_data['omega_t'][1], 0.7)  # omega mock function
        self.assertEqual(df_data['sigma_t_sq'][1], 50)  # second trial estimation uncertainty
        self.assertAlmostEqual(df_data['a_t_hat'][1], -30, 6)  # update mock function
        self.assertAlmostEqual(llh_mix[1], -23.718769305380253, 6)  # computed llh
        self.assertEqual(df_data['sim_b_t'][1], 151.0)  # update mock Gaussian
        self.assertEqual(df_data['sim_a_t'][1], 1)  # update mock Gaussian
        self.assertTrue(np.isnan(df_data['sim_y_t'][1]))  # no simulation
        self.assertTrue(np.isnan(df_data['sim_z_t'][1]))  # no simulation
        self.assertTrue(df_data['cond'][0] == "main")  # input value
        self.assertEqual(df_data['sigma'][0], 10)  # input value

    @patch('__main__.AlAgent.learn', new=mock_learn)
    def test_last_block_trial_exp1(self):
        """ This function implements a unit test of the task-agent interaction
            in the last trial of a block of experiment 1

            The test covers the last trial of a block with the new_block[t+1] case,
            where agent.learn is not called.
        """

        # Load function input
        which_exp, df, agent, agent_vars, sel_coeffs = self.load_default_input()

        # Extract first and second trial for unit test
        df_trial = df[0:3].copy()
        df_trial.loc[2, 'new_block'] = 1

        # Run last block trial
        #   futuretodo: mock out compute_persprob
        llh_mix, df_data = task_agent_int(which_exp, df_trial, agent, agent_vars, sel_coeffs)

        # Test function output
        self.assertEqual(df_data['delta_t'][1], 8.0)  # input value
        self.assertAlmostEqual(df_data['tau_t'][1], 0.2, 6)  # tau mock function
        self.assertAlmostEqual(df_data['sigma_t_sq'][1], 50, 6)  # second trial estimation uncertainty
        self.assertTrue(np.isnan(df_data['a_t_hat'][1]))  # last trial should be nan
        self.assertTrue(np.isnan(df_data['mu_t'][1]))  # last trial should be nan
        self.assertTrue(np.isnan(df_data['omega_t'][1]))  # last trial should be nan
        self.assertTrue(np.isnan(df_data['alpha_t'][1]))  # last trial should be nan
        self.assertAlmostEqual(llh_mix, -4.841523619865609, 6)  # computed llh

    @patch('__main__.AlAgent.learn', new=mock_learn)
    @patch('numpy.random.normal', new=Mock(return_value=1))
    @patch('numpy.random.binomial', new=Mock(return_value=0))
    def test_last_block_trial_exp1_sim(self):
        """ This function implements a unit test of the task-agent interaction
            in the last block of a trial of experiment 1 and when data
            are simulated

            The test covers the last trial of a block with the new_block[t+1] case,
            where agent.learn is not called. We additionally mock-out the
            Gaussian and binomial distributions to have consistent
            simulated updates sim_a_t.
        """

        # Load function input
        which_exp, df, agent, agent_vars, sel_coeffs = self.load_default_input()

        # Extract first and second trial for unit test
        df_trial = df[0:3].copy()
        df_trial.loc[2, 'new_block'] = 1

        # Run last block trial
        llh_mix, df_data = task_agent_int(which_exp, df_trial, agent, agent_vars, sel_coeffs, sim=True)

        # Test function output
        self.assertTrue(np.isnan(df_data['delta_t'][1]))  # last trial should be nan
        self.assertAlmostEqual(df_data['tau_t'][1], 0.2, 6)  # tau mock function
        self.assertAlmostEqual(df_data['sigma_t_sq'][1], 50, 6)  # second trial estimation uncertainty
        self.assertTrue(np.isnan(df_data['a_t_hat'][1]))  # last trial should be nan
        self.assertTrue(np.isnan(df_data['mu_t'][1]))  # last trial should be nan
        self.assertTrue(np.isnan(df_data['omega_t'][1]))  # last trial should be nan
        self.assertTrue(np.isnan(df_data['alpha_t'][1]))  # last trial should be nan
        self.assertAlmostEqual(llh_mix, -4.841523619865609, 6)  # computed llh
        self.assertEqual(df_data['sim_b_t'][1], 151.0)  # update mock Gaussian
        self.assertTrue(np.isnan(df_data['sim_a_t'][1]))  # last trial should be nan
        self.assertTrue(np.isnan(df_data['sim_y_t'][1]))  # last trial should be nan
        self.assertTrue(np.isnan(df_data['sim_z_t'][1]))  # last trial should be nan
        self.assertTrue(df_data['cond'][0] == "main")  # input value
        self.assertEqual(df_data['sigma'][0], 10)  # input value

    @patch('__main__.AlAgent.learn', new=mock_learn)
    def test_first_trial_exp2(self):
        """ This function implements a unit test of the task-agent interaction
            on the first trial of experiment 2

            The test covers the first trial with the new_block[t] case,
            where the agent is initialized.
        """

        # Load function input
        which_exp, df, agent, agent_vars, sel_coeffs = self.load_default_input(which_exp=2)

        # Extract first and second trial for unit test
        df_trial = df[200:205].copy()
        df_trial.loc[202:205, 'new_block'] = 1
        df_trial = df_trial.reset_index(drop=True)

        # Run second trial
        llh_mix, df_data = task_agent_int(which_exp, df_trial, agent, agent_vars, sel_coeffs)

        # Test function output
        self.assertEqual(df_data['delta_t'][0], 79)  # input value
        self.assertEqual(df_data['tau_t'][0], 0.5)  # new block initialization
        self.assertEqual(df_data['sigma_t_sq'][0], 100)  # new block initialization
        self.assertEqual(df_data['mu_t'][0], 122)  # belief according to mock function
        self.assertEqual(df_data['omega_t'][0], 0.7)  # omega mock function
        self.assertEqual(df_data['alpha_t'][0], 0.35)  # alpha mock function
        self.assertEqual(df_data['a_t_hat'][0], -30.0)  # update mock function
        self.assertAlmostEqual(llh_mix[0], -46.05170, 5)  # computed llh

    @patch('__main__.AlAgent.learn', new=mock_learn)
    @patch('numpy.random.normal', new=Mock(return_value=1))
    @patch('numpy.random.binomial', new=Mock(return_value=0))
    def test_first_trial_exp2_sim(self):
        """ This function implements a unit test of the task-agent interaction
            in the first trial of experiment 2 when data are simulated

            The test covers the first trial with the new_block[t] case,
            where the agent is initialized. We additionally mock-out the
            Gaussian and binomial distributions to have consistent
            simulated updates sim_a_t.
         """

        # Load function input
        which_exp, df, agent, agent_vars, sel_coeffs = self.load_default_input(which_exp=2)

        # Extract first and second trial for unit test
        df_trial = df[200:205].copy()
        df_trial.loc[202:205, 'new_block'] = 1
        df_trial = df_trial.reset_index(drop=True)

        # Run second trial
        llh_mix, df_data = task_agent_int(which_exp, df_trial, agent, agent_vars, sel_coeffs, sim=True)

        # Test function output
        self.assertAlmostEqual(df_data['delta_t'][0], 118.0, 5)  # 268 (outcome) - 151 (belief trial 1)
        self.assertEqual(df_data['tau_t'][0], 0.5)  # new block initialization
        self.assertEqual(df_data['sigma_t_sq'][0], 100)  # new block initialization
        self.assertEqual(df_data['mu_t'][0], 122)  # belief according to mock function
        self.assertEqual(df_data['omega_t'][0], 0.7)  # omega mock function
        self.assertEqual(df_data['alpha_t'][0], 0.35)  # alpha mock function
        self.assertEqual(df_data['a_t_hat'][0], -30.0)  # update mock function
        self.assertAlmostEqual(llh_mix[0], -46.051701859880914, 6)  # computed llh
        self.assertEqual(df_data['sim_b_t'][0], 150.0)  # initial belief
        self.assertEqual(df_data['sim_a_t'][0], 1)  # update mock Gaussian
        self.assertEqual(df_data['sim_y_t'][0], 0)  # no push
        self.assertEqual(df_data['sim_z_t'][0], 150.0)  # no push
        self.assertTrue(df_data['cond'][0] == "main_push")  # input value
        self.assertEqual(df_data['sigma'][0], 17.5)  # input value

    @patch('__main__.AlAgent.learn', new=mock_learn)
    def test_second_trial_exp2(self):
        """ This function implements a unit test of the task-agent interaction
            in the second trial.

            The test covers the second trial, which is a regular trial without
            new_block[t] or new_block[t+1].
        """

        # Load function input
        which_exp, df, agent, agent_vars, sel_coeffs = self.load_default_input(which_exp=2)

        # Extract second and third trial for unit test
        # df_trial = df[1:4]
        # df_trial = df_trial.reset_index(drop=True)

        df_trial = df[201:206].copy()
        df_trial.loc[203:205, 'new_block'] = 1
        df_trial = df_trial.reset_index(drop=True)

        # Run second trial
        llh_mix, df_data = task_agent_int(which_exp, df_trial, agent, agent_vars, sel_coeffs)

        # Test function output
        self.assertEqual(df_data['delta_t'][0], -6)  # input value
        self.assertEqual(df_data['tau_t'][0], 0.5)  # new block initialization
        self.assertEqual(df_data['sigma_t_sq'][0], 100)  # new block initialization
        self.assertEqual(df_data['mu_t'][0], 122)  # belief according to mock function
        self.assertEqual(df_data['omega_t'][0], 0.7)  # omega mock function
        self.assertEqual(df_data['alpha_t'][0], 0.35)  # alpha mock function
        self.assertEqual(df_data['a_t_hat'][0], -42.2)  # mocked update + bucket bias
        self.assertAlmostEqual(llh_mix[0], -5.26618, 5)  # computed llh

    @patch('__main__.AlAgent.learn', new=mock_learn)
    @patch('numpy.random.normal', new=Mock(return_value=1))
    @patch('numpy.random.binomial', new=Mock(return_value=0))
    def test_second_trial_exp2_sim(self):
        """ This function implements a unit test of the task-agent interaction
            in the second trial when data are simulated.

            The test covers the second trial, which is a regular trial without
            new_block[t] or new_block[t+1].
        """

        # Load function input
        which_exp, df, agent, agent_vars, sel_coeffs = self.load_default_input(which_exp=2)

        # Extract second and third trial for unit test
        df_trial = df[200:206].copy()
        df_trial.loc[203:206, 'new_block'] = 1
        df_trial = df_trial.reset_index(drop=True)

        # Run second trial
        llh_mix, df_data = task_agent_int(which_exp, df_trial, agent, agent_vars, sel_coeffs, sim=True)

        # Test function output
        self.assertEqual(df_data['delta_t'][1], 107)  # # 258 (outcome) - 151 (belief trial 1)
        self.assertEqual(df_data['tau_t'][1], 0.2)  # tau mock function
        self.assertEqual(df_data['sigma_t_sq'][1], 50)  # second trial estimation uncertainty
        self.assertEqual(df_data['mu_t'][1], 122)  # belief according to mock function
        self.assertEqual(df_data['omega_t'][1], 0.7)  # omega mock function
        self.assertEqual(df_data['alpha_t'][1], 0.35)  # alpha mock function
        self.assertEqual(df_data['a_t_hat'][1], -42.2)  # mock output + bucket bias
        self.assertAlmostEqual(llh_mix[1], -5.26618, 5)  # computed llh
        self.assertEqual(df_data['sim_b_t'][1], 151.0)  # update mock Gaussian
        self.assertEqual(df_data['sim_a_t'][1], 1)  # update mock Gaussian
        self.assertEqual(df_data['sim_y_t'][1], -61)  # input value
        self.assertEqual(df_data['sim_z_t'][1], 90)  # computed position (correct_push can be mocked out in future)
        self.assertTrue(df_data['cond'][0] == "main_push")  # input value
        self.assertEqual(df_data['sigma'][0], 17.5)  # input value

    @patch('__main__.AlAgent.learn', new=mock_learn)
    def test_last_block_trial_exp2(self):
        """ This function implements a unit test of the task-agent interaction
            in the last block of a trial of experiment 2

            The test covers the last trial of a block with the new_block[t+1] case,
            where agent.learn is not called.
        """

        # Load function input
        which_exp, df, agent, agent_vars, sel_coeffs = self.load_default_input(which_exp=2)

        # Extract first and second trial for unit test
        df_trial = df[200:205].copy()
        df_trial.loc[202:205, 'new_block'] = 1
        df_trial = df_trial.reset_index(drop=True)

        # Run last block trial
        llh_mix, df_data = task_agent_int(which_exp, df_trial, agent, agent_vars, sel_coeffs)

        # Test function output
        self.assertEqual(df_data['delta_t'][1], -6)  # input value
        self.assertAlmostEqual(df_data['tau_t'][1], 0.2, 6)  # tau mock function
        self.assertAlmostEqual(df_data['sigma_t_sq'][1], 50, 6)  # second trial estimation uncertainty
        self.assertTrue(np.isnan(df_data['a_t_hat'][1]))  # last trial should be nan
        self.assertTrue(np.isnan(df_data['mu_t'][1]))  # last trial should be nan
        self.assertTrue(np.isnan(df_data['omega_t'][1]))  # last trial should be nan
        self.assertTrue(np.isnan(df_data['alpha_t'][1]))  # last trial should be nan
        self.assertAlmostEqual(llh_mix, -46.051701859880914, 6)  # computed llh

    @patch('__main__.AlAgent.learn', new=mock_learn)
    def test_pers_trial(self):
        """ This function implements a unit test of the task-agent interaction
            in a perseveration trial.

            The test covers a perseveration trial, where perseveration is directly
            considered in the likelihood function.
        """

        # Load function input
        which_exp, df, agent, agent_vars, sel_coeffs = self.load_default_input()

        # Extract trials for unit test
        df_trial = df[24:27].copy()
        df_trial.loc[27, 'new_block'] = 1
        df_trial = df_trial.reset_index(drop=True)

        # Run second trial
        llh_mix, df_data = task_agent_int(which_exp, df_trial, agent, agent_vars, sel_coeffs)

        # Test function output
        self.assertEqual(df_data['delta_t'][0], -8.0) # input value
        self.assertEqual(df_data['tau_t'][0], 0.5)  # new block initialization
        self.assertEqual(df_data['sigma_t_sq'][0], 100)  # new block initialization
        self.assertEqual(df_data['mu_t'][0], 122)  # belief according to mock function
        self.assertEqual(df_data['omega_t'][0], 0.7)  # omega mock function
        self.assertEqual(df_data['alpha_t'][0], 0.35)  # alpha mock function
        self.assertEqual(df_data['a_t_hat'][0], -30)  # update mock function
        self.assertAlmostEqual(llh_mix[0], -0.6931471794447687, 6)  # computed llh

    def test_integration_task_agent_int_exp1(self):
        """ This function implements an integration test of the task-agent interaction
            across 400 trials in experiment 1

            The test was programmed on Ubuntu, and data frames generated on a different
            operating system might lead to failure.
         """

        # Load function input
        which_exp, df, agent, agent_vars, sel_coeffs = self.load_default_input()

        # Run all trials
        llh_mix, df_data = task_agent_int(which_exp, df, agent, agent_vars, sel_coeffs)

        # savename = 'al_tests/test_task_agent_int.pkl'
        # df_data.to_pickle(savename)

        # Load test data
        test_data = pd.read_pickle('al_tests/test_task_agent_int.pkl')

        # Test function output
        self.assertTrue(test_data.equals(df_data))  # , atol=1e-04

    def test_integration_task_agent_int_exp2(self):
        """ This function implements an integration test of the task-agent interaction
            across 400 trials in experiment 2

            The test was programmed on Ubuntu, and data frames generated on a different
            operating system might lead to failure.
        """

        # Load function input
        which_exp, df, agent, agent_vars, sel_coeffs = self.load_default_input(which_exp=2)

        # Run all trials
        llh_mix, df_data = task_agent_int(which_exp, df, agent, agent_vars, sel_coeffs)

        # savename = 'al_tests/test_task_agent_int_exp_2.pkl'
        # df_data.to_pickle(savename)

        # Load test data
        test_data = pd.read_pickle('al_tests/test_task_agent_int_exp_2.pkl')

        # Test function output
        self.assertTrue(test_data.equals(df_data))

    @staticmethod
    def load_default_input(which_exp=1):
        """ This function loads the default input to task_agent_int.py.

        :return: which_exp, df, agent, agent_vars, sel_coeffs
        """

        #  which_exp input argument
        # -------------------------

        # Specify experiment as in "al_estimation.py"
        # which_exp = 1

        # df input argument
        # -----------------

        # Load data from first experiment as in "al_estimation.py"
        if which_exp == 1:
            df_exp1 = pd.read_pickle('al_data/data_prepr_1.pkl')
            df = df_exp1[(df_exp1['subj_num'] == 1)].copy()
        else:
            df_exp2 = pd.read_pickle('al_data/data_prepr_2.pkl')
            df = df_exp2[(df_exp2['subj_num'] == 1)].copy()

        # agent and agent_vars input arguments
        # ------------------------------------

        # Initialize agent based on agent_vars
        agent_vars = AgentVars()
        agent_vars.u = np.exp(0)
        agent = AlAgent(agent_vars)

        # sel_coeffs input argument
        # -------------------------

        # est_vars as loaded in "al_estimation.py"
        est_vars = AlEstVars()

        # Number of subjects as added in "al_estimation.py"
        est_vars.n_subj = 1  # here just 1 test subject

        # Free parameters as specified in "al_estimation.py"
        if which_exp == 1:
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
        else:
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

        # Fixed parameter values as spe
        fixed_coeffs = est_vars.fixed_mod_coeffs

        # Use fixed starting points
        if which_exp == 1:
            x0 = [est_vars.omikron_0_x0,
                  est_vars.omikron_1_x0,
                  est_vars.b_0_x0,
                  est_vars.b_1_x0,
                  est_vars.h_x0,
                  est_vars.s_x0,
                  est_vars.u_x0,
                  est_vars.q_x0,
                  est_vars.sigma_H_x0,
                  est_vars.d_x0]
        else:
            x0 = [est_vars.omikron_0_x0,
                  est_vars.omikron_1_x0,
                  est_vars.b_0_x0,
                  est_vars.b_1_x0,
                  est_vars.h_x0,
                  est_vars.s_x0,
                  est_vars.u_x0,
                  est_vars.q_x0,
                  est_vars.sigma_H_x0,
                  0.2]

        # Extract free parameters
        values = est_vars.which_vars.values()

        # Select starting points according to free parameters
        coeffs = np.array(list(compress(x0, values)))

        # Futuretodo: create a function for this
        #   Initialize parameter list and counters
        sel_coeffs = []
        i = 0

        # Put selected coefficients in list that is used for model estimation
        for key, value in est_vars.which_vars.items():
            if value:
                sel_coeffs.append(coeffs[i])
                i += 1
            else:
                sel_coeffs.append(fixed_coeffs[key])

        return which_exp, df, agent, agent_vars, sel_coeffs


# Run unit test
if __name__ == '__main__':
    unittest.main()
