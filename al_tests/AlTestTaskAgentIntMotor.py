""" Task-Agent-Interaction Motor Model: Unit and integration tests

    Tests work on Rasmus Bruckner's Ubuntu laptop.
    Integration tests can fail on other operating systems.
"""

import numpy as np
import unittest
import pandas as pd
from unittest.mock import patch, Mock
from AlAgentVarsRbm import AgentVars
from AlAgentRbm import AlAgent
from motor.al_task_agent_int_motor import task_agent_int_motor


class TestTaskAgentInt(unittest.TestCase):
    """ This class definition implements the task_agent_int (motor model) unittests
        and integration tests based on a participant data set from experiment 2

        We run the following tests:
            - test_first_trial_no_pers: First trial no perseveration
            - test_first_trial_pers_push: First trial no perseveration and push condition
            - test_second_trial_no_pers: Second trial no perseveration
            - test_second_trial_pers_push: Second trial with perseveration and push condition
            - test_second_trial_pers_no_push: Second trial with perseveration and no-push condition
            - test_last_block_trial: Last trial of a block
            - test_integration_task_agent_int_motor: Integration test based on data set

        In the unit tests, we mock out the agent and use predefined values
            as defined in fun mock_learn.
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
        self.a_t = 100

    @patch('__main__.AlAgent.learn', new=mock_learn)
    @patch('motor.al_task_agent_int_motor.find_nearest', new=Mock(return_value=[31, np.nan]))
    @patch('motor.al_task_agent_int_motor.correct_push', new=Mock(return_value=[150, 0]))
    @patch('numpy.random.binomial', new=Mock(return_value=0))
    def test_first_trial_no_pers(self):
        """ This function implements a unit test of the task-agent interaction
            in the first trial without perseveration

            The test covers the first trial with the new_block[t] case,
            where the agent is initialized.
         """

        # Load function input
        df, agent, agent_vars, sel_coeffs = self.load_default_input()

        # Extract first and second trial for unit test
        df_trial = df[200:202].copy()
        df_trial = df_trial.reset_index(drop=True)

        # Run first trial
        df_data = task_agent_int_motor(df_trial, agent, agent_vars, sel_coeffs)

        # Test function output
        self.assertEqual(df_data['delta_t'][0], 118, 5)  # 268 (outcome) - 150 (belief trial 1)
        self.assertEqual(df_data['tau_t'][0], 0.5)  # new block initialization
        self.assertEqual(df_data['mu_t'][0], 122)  # belief according to mock function
        self.assertEqual(df_data['omega_t'][0], 0.7)  # omega mock function
        self.assertEqual(df_data['alpha_t'][0], 0.35)  # alpha mock function
        self.assertEqual(df_data['sim_b_t'][0], 150)  # initial belief
        self.assertEqual(df_data['a_t_hat'][0], -119)  # difference mocked belief (find_nearest) and initial belief
        self.assertEqual(df_data['sim_a_t'][0], -119)  # difference mocked belief (find_nearest) and initial belief
        self.assertEqual(df_data['sim_y_t'][0], 0)  # first trial simulation without push
        self.assertEqual(df_data['sim_z_t'][0], 150.0)  # first trial simulation without push
        self.assertEqual(df_data['sigma'][0], 17.5)  # input value
        self.assertTrue(df_data['cond'][0] == "main_push")  # input value
        self.assertEqual(df_data['age_group'][0], 1)  # input value

    @patch('__main__.AlAgent.learn', new=mock_learn)
    @patch('motor.al_task_agent_int_motor.find_nearest', new=Mock(return_value=[31, np.nan]))
    @patch('motor.al_task_agent_int_motor.correct_push', new=Mock(return_value=[120, -30]))
    @patch('numpy.random.binomial', new=Mock(return_value=1))
    def test_first_trial_pers_push(self):
        """ This function implements a unit test of the task-agent interaction
            in the first trial with perseveration in the push condition

            The test covers the first trial with the new_block[t] case,
            where the agent is initialized.
         """

        # Load function input
        df, agent, agent_vars, sel_coeffs = self.load_default_input()

        # Extract first and second trial for unit test
        df_trial = df[200:202].copy()
        df_trial = df_trial.reset_index(drop=True)

        # Run second trial
        df_data = task_agent_int_motor(df_trial, agent, agent_vars, sel_coeffs)

        # Test function output
        self.assertEqual(df_data['delta_t'][0], 118, 5)  # 268 (outcome) - 150 (belief trial 1)
        self.assertEqual(df_data['tau_t'][0], 0.5)  # new block initialization
        self.assertEqual(df_data['mu_t'][0], 122)  # belief according to mock function
        self.assertEqual(df_data['omega_t'][0], 0.7)  # omega mock function
        self.assertEqual(df_data['alpha_t'][0], 0.35)  # alpha mock function
        self.assertEqual(df_data['sim_b_t'][0], 150)  # initial belief
        self.assertEqual(df_data['a_t_hat'][0], -119)  # difference mocked belief (find_nearest) and initial belief
        self.assertEqual(df_data['sim_a_t'][0], 0)  # perseveration case
        self.assertEqual(df_data['sim_y_t'][0], 0)  # first trial simulation without push
        self.assertEqual(df_data['sim_z_t'][0], 150.0)  # first trial simulation without push
        self.assertEqual(df_data['sigma'][0], 17.5)   # input value
        self.assertTrue(df_data['cond'][0] == "main_push")  # input value
        self.assertEqual(df_data['age_group'][0], 1)  # input value

    @patch('__main__.AlAgent.learn', new=mock_learn)
    @patch('motor.al_task_agent_int_motor.find_nearest', new=Mock(side_effect=[[20, np.nan], [50, np.nan]]))
    @patch('motor.al_task_agent_int_motor.correct_push', new=Mock(return_value=[-80, 25]))
    @patch('numpy.random.binomial', new=Mock(return_value=0))
    def test_second_trial_no_pers(self):
        """ This function implements a unit test of the task-agent interaction
            in the second trial without perseveration

            The test covers the second trial, which is a regular trial without
            new_block[t] or new_block[t+1].
        """

        # Load function input
        df, agent, agent_vars, sel_coeffs = self.load_default_input()

        # Extract second and third trial for unit test
        df_trial = df[200:204].copy()
        df_trial.loc[203, 'new_block'] = 1
        df_trial = df_trial.reset_index(drop=True)

        # Run second trial
        df_data = task_agent_int_motor(df_trial, agent, agent_vars, sel_coeffs)

        # Test function output
        self.assertEqual(df_data['delta_t'][1], 238)  # 258 (outcome) - 20 (belief trial 2)
        self.assertEqual(df_data['tau_t'][1], 0.2)  # tau mock function
        self.assertEqual(df_data['mu_t'][1], 122)  # belief according to mock function
        self.assertEqual(df_data['omega_t'][1], 0.7)  # omega mock function
        self.assertEqual(df_data['alpha_t'][1], 0.35)  # alpha mock function
        self.assertEqual(df_data['a_t_hat'][1], 30)  # difference mocked beliefs (find_nearest)
        self.assertEqual(df_data['sim_a_t'][1], 30)  # difference mocked beliefs (find_nearest)
        self.assertEqual(df_data['sim_b_t'][1], 20)  # mocked belief
        self.assertEqual(df_data['sim_y_t'][1], -80)  # mocked output correct push
        self.assertEqual(df_data['sim_z_t'][1], 25)  # mocked output correct push
        self.assertEqual(df_data['sigma'][1], 17.5)  # input value
        self.assertTrue(df_data['cond'][1] == "main_push")  # input value
        self.assertEqual(df_data['age_group'][1], 1)  # input value

    @patch('__main__.AlAgent.learn', new=mock_learn)
    @patch('motor.al_task_agent_int_motor.find_nearest', new=Mock(side_effect=[[20, np.nan], [50, np.nan]]))
    @patch('motor.al_task_agent_int_motor.correct_push', new=Mock(return_value=[-80, 25]))
    @patch('numpy.random.binomial', new=Mock(return_value=1))
    def test_second_trial_pers_push(self):
        """ This function implements a unit test of the task-agent interaction
            in the second trial with perseveration in the push condition

            The test covers the second trial, which is a regular trial without
            new_block[t] or new_block[t+1].
        """

        # Load function input
        df, agent, agent_vars, sel_coeffs = self.load_default_input()

        # Extract second and third trial for unit test
        df_trial = df[200:204].copy()
        df_trial.loc[203, 'new_block'] = 1
        df_trial = df_trial.reset_index(drop=True)

        # Run second trial
        df_data = task_agent_int_motor(df_trial, agent, agent_vars, sel_coeffs)

        # Test function output
        self.assertEqual(df_data['delta_t'][1], 108)  # 258 (outcome) - 150 (belief trial 2)
        self.assertEqual(df_data['tau_t'][1], 0.2)  # tau mock function
        self.assertEqual(df_data['mu_t'][1], 122)  # belief according to mock function
        self.assertEqual(df_data['omega_t'][1], 0.7)  # omega mock function
        self.assertEqual(df_data['alpha_t'][1], 0.35)  # alpha mock function
        self.assertEqual(df_data['a_t_hat'][1], -100)  # difference mocked beliefs (find_nearest)
        self.assertEqual(df_data['sim_a_t'][1], -80)  # this is the critical case bc indicates motor pers
        self.assertEqual(df_data['sim_b_t'][1], 150)  # test trial 2 to check if bucket push = updated belief
        self.assertEqual(df_data['sim_b_t'][2], 25)  # also test trial 2 to check if bucket push = updated belief
        self.assertEqual(df_data['sim_y_t'][1], -80)  # mocked output correct push
        self.assertEqual(df_data['sim_z_t'][1], 25)  # mocked output correct push
        self.assertEqual(df_data['sigma'][1], 17.5)  # input value
        self.assertTrue(df_data['cond'][1] == "main_push")  # input value
        self.assertEqual(df_data['age_group'][1], 1)  # input value

    @patch('__main__.AlAgent.learn', new=mock_learn)
    @patch('motor.al_task_agent_int_motor.find_nearest', new=Mock(side_effect=[[20, np.nan], [50, np.nan]]))
    @patch('motor.al_task_agent_int_motor.correct_push', new=Mock(return_value=[-80, 25]))
    @patch('numpy.random.binomial', new=Mock(return_value=1))
    def test_second_trial_pers_no_push(self):
        """ This function implements a unit test of the task-agent interaction
            in the second trial with perseveration in the no-push condition

            The test covers the second trial, which is a regular trial without
            new_block[t] or new_block[t+1].
        """

        # Load function input
        df, agent, agent_vars, sel_coeffs = self.load_default_input()

        # Extract second and third trial for unit test
        df_trial = df[200:204].copy()
        df_trial.loc[203, 'new_block'] = 1
        df_trial = df_trial.reset_index(drop=True)
        df_trial['cond'] = 'main_noPush'

        # Run second trial
        df_data = task_agent_int_motor(df_trial, agent, agent_vars, sel_coeffs)

        # Test function output
        self.assertEqual(df_data['delta_t'][1], 108)  # 258 (outcome) - 150 (belief trial 2)
        self.assertEqual(df_data['tau_t'][1], 0.2)  # tau mock function
        self.assertEqual(df_data['mu_t'][1], 122)  # belief according to mock function
        self.assertEqual(df_data['omega_t'][1], 0.7)  # omega mock function
        self.assertEqual(df_data['alpha_t'][1], 0.35)  # alpha mock function
        self.assertEqual(df_data['a_t_hat'][1], -100)  # difference mocked beliefs (find_nearest)
        self.assertEqual(df_data['sim_a_t'][1], 0)  # this is the critical case bc indicates pers
        self.assertEqual(df_data['sim_b_t'][1], 150)  # test trial 2 to check if bucket push = updated belief
        self.assertEqual(df_data['sim_b_t'][2], 25)  # also test trial 2 to check if bucket push = updated belief
        self.assertEqual(df_data['sim_y_t'][1], -80)  # mocked output correct push
        self.assertEqual(df_data['sim_z_t'][1], 25)  # mocked output correct push
        self.assertEqual(df_data['sigma'][1], 17.5)  # input value
        self.assertTrue(df_data['cond'][1] == "main_noPush")  # input value
        self.assertEqual(df_data['age_group'][1], 1)  # input value

    @patch('__main__.AlAgent.learn', new=mock_learn)
    @patch('motor.al_task_agent_int_motor.find_nearest', new=Mock(side_effect=[[20, np.nan], [50, np.nan]]))
    @patch('motor.al_task_agent_int_motor.correct_push', new=Mock(return_value=[-80, 25]))
    def test_last_block_trial(self):
        """ This function implements a unit test of the task-agent interaction
            in the last trial of a block

            The test covers the last trial of a block with the new_block[t+1] case,
            where agent.learn is not called.
        """

        # Load function input
        df, agent, agent_vars, sel_coeffs = self.load_default_input()

        # Extract first and second trial for unit test
        df_trial = df[200:203].copy()
        df_trial.loc[202:, 'new_block'] = 1
        df_trial = df_trial.reset_index(drop=True)

        # Run last block of trial
        df_data = task_agent_int_motor(df_trial, agent, agent_vars, sel_coeffs)

        # Test function output
        self.assertTrue(np.isnan(df_data['delta_t'][1]))  # last trial should be nan
        self.assertEqual(df_data['tau_t'][1], 0.2)  # tau mock function
        self.assertTrue(np.isnan(df_data['a_t_hat'][1]))  # last trial should be nan
        self.assertTrue(np.isnan(df_data['mu_t'][1]))  # last trial should be nan
        self.assertTrue(np.isnan(df_data['omega_t'][1]))  # last trial should be nan
        self.assertTrue(np.isnan(df_data['alpha_t'][1]))  # last trial should be nan
        self.assertEqual(df_data['sim_b_t'][1], 20)  # mocked output correct push
        self.assertTrue(np.isnan(df_data['sim_a_t'][1]))  # last trial should be nan
        self.assertEqual(df_data['sim_y_t'][1], -80)  # mocked output correct push
        self.assertEqual(df_data['sim_z_t'][1], 25)  # mocked output correct push
        self.assertEqual(df_data['sigma'][1], 17.5)  # input value
        self.assertTrue(df_data['cond'][1] == "main_push")  # input value
        self.assertEqual(df_data['age_group'][1], 1)  # input value

    def test_integration_task_agent_int_motor(self):
        """ This function implements an integration test of the task-agent interaction
            across 400 trials
        """

        # Load function input
        df, agent, agent_vars, sel_coeffs = self.load_default_input()

        # Run all trials
        np.random.seed(seed=1)
        df_data = task_agent_int_motor(df, agent, agent_vars, sel_coeffs)

        # savename = 'al_tests/test_task_agent_int_motor.pkl'
        # df_data.to_pickle(savename)

        # Load test data
        test_data = pd.read_pickle('al_tests/test_task_agent_int_motor.pkl')

        # Test function output
        self.assertTrue(test_data.equals(df_data))

    @staticmethod
    def load_default_input():
        """ This function loads the default input to task_agent_int.py.

        :return: df: Data frame
                 agent: Agent-object instance
                 agent_vars: Agent-variables-object instance
                 sel_coeffs: Agent coefficients
        """

        # Dataset for test
        df_exp2 = pd.read_pickle('al_data/data_prepr_2.pkl')
        df = df_exp2[(df_exp2['subj_num'] == 1)].copy()

        # Agent coefficients
        sel_coeffs = [np.nan, np.nan, 0.5, -5, 0.1, 1, 0, 0, 0.5, 1.1]

        # Initialize agent variables and agent
        agent_vars = AgentVars()
        agent_vars.h = sel_coeffs[4]
        agent_vars.s = sel_coeffs[5]
        agent_vars.u = np.exp(sel_coeffs[6])
        agent_vars.q = 0
        agent_vars.sigma_H = sel_coeffs[7]
        agent = AlAgent(agent_vars)

        return df, agent, agent_vars, sel_coeffs


# Run unit test
if __name__ == '__main__':
    unittest.main()
