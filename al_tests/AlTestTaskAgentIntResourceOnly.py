""" Task-Agent-Interaction Resource-Only Model: Unit and integration tests """

import unittest
from sampling.AlAgentSampling import AlAgentSampling
from sampling.AlAgentVarsSampling import AgentVarsSampling
from unittest.mock import patch
from unittest.mock import Mock
from sampling.al_task_agent_int_resource_only import task_agent_int_resource_only
import pandas as pd
import numpy as np
from al_utilities import get_df_subj


class TestTaskAgentIntResourceOnly(unittest.TestCase):
    """ This class definition implements the unit tests for the task-agent-interaction of the resource-only model

        We have the following test functions

            - test_first_trial_pers: First trial with perseveration
            - test_first_trial_pers_large_b_t: First trial without perseveration and adjusted b_t
            - test_first_trial_pers_small_b_t: First trial without perseveration and adjusted b_t
            - test_second_trial_pers: Second trial after initialization
            - test_second_trial_pers_large_push_pos: Second trial with strong positive push that is corrected
            - test_second_trial_pers_large_push_neg: Second trial with strong negative push that is corrected
            - test_last_trial_pers: Last trial of a block where we mostly expect nans
            - test_first_trial_no_pers: First trial without perseveration
            - test_integration: Integration test based on data set of first participant (experiment 2)
    """

    def sampling(self):
        """ This mock function replaces the sampling function for unit testing """

        # Make sure that belief is updated to get no perseveration
        self.mu_t = 201
        self.omega_t = 0.64
        self.tot_samples = 10

    @patch('sampling.AlAgentSampling.AlAgentSampling')
    @patch('numpy.random.binomial', new=Mock(return_value=1))
    def test_first_trial_pers(self, agent):
        """ This function implements a unit test of the task-agent interaction
            in the first trial when perseveration takes place. We mock out the
            sampling agent.

            The test covers the first trial with the new_block[t] case,
            where the agent is initialized.
        """

        # Load function input
        df_subj = self.load_default_input()

        # Initialize agent variables
        agent_vars = AgentVarsSampling()
        agent_vars.mu_0 = 34
        agent.delta_t = -21

        # Run interaction function and test results
        df_data = task_agent_int_resource_only(df_subj, agent, agent_vars, n_trials=2)
        self.assertEqual(df_data['x_t'][0], 176)  # expected bc defined in df_subj
        self.assertEqual(df_data['mu_t'][0], 34)  # expected bc defined here
        self.assertEqual(df_data['delta_t'][0], -21)  # expected bc defined here
        self.assertEqual(df_data['omega_t'][0], 1)  # expected bc initial value
        self.assertEqual(df_data['tau_t'][0], 0.5)  # expected bc initial value
        self.assertEqual(df_data['alpha_t'][0], 0.0)  # expected bc perseveration
        self.assertEqual(df_data['sim_b_t'][0], 34)  # expected bc perseveration
        self.assertEqual(df_data['sim_a_t'][0], 0)  # expected bc perseveration
        self.assertEqual(df_data['sim_y_t'][0], 0)  # expected bc defined here
        self.assertEqual(df_data['sim_z_t'][0], 34)  # expected bc defined here and no push
        self.assertEqual(df_data['pers'][0], True)  # expected bc perseveration
        self.assertEqual(df_data['cond'][0], 'main_noPush')  # expected bc defined here
        self.assertEqual(df_data['tot_samples'][0], 1)  # expected bc mock-function output

    @patch('sampling.AlAgentSampling.AlAgentSampling')
    @patch('numpy.random.binomial', new=Mock(return_value=1))
    def test_first_trial_pers_large_b_t(self, agent):
        """ This function implements a unit test of the task-agent interaction
            in the first trial when perseveration takes place. We mock out the
            sampling agent. b_t is > 300 and corrected.

            The test covers the first trial with the new_block[t] case,
            where the agent is initialized.
        """

        # Initialize agent variables
        agent_vars = AgentVarsSampling()
        agent_vars.mu_0 = 350
        agent.delta_t = -21

        # Load function input
        df_subj = self.load_default_input()

        # Run interaction function and test results
        df_data = task_agent_int_resource_only(df_subj, agent, agent_vars, n_trials=2)
        self.assertEqual(df_data['x_t'][0], 176)  # expected bc defined in df_subj
        self.assertEqual(df_data['mu_t'][0], 350)  # expected bc defined here
        self.assertEqual(df_data['delta_t'][0], -21)  # expected bc defined here
        self.assertEqual(df_data['omega_t'][0], 1)  # expected bc initial value
        self.assertEqual(df_data['tau_t'][0], 0.5)  # expected bc initial value
        self.assertEqual(df_data['alpha_t'][0], 0.0)  # expected bc perseveration
        self.assertEqual(df_data['sim_b_t'][0], 350)  # expected bc perseveration
        self.assertEqual(df_data['sim_b_t'][1], 300)  # expected bc perseveration
        self.assertEqual(df_data['sim_a_t'][0], 0)  # expected bc perseveration
        self.assertEqual(df_data['sim_y_t'][0], 0)  # expected bc defined here
        self.assertEqual(df_data['sim_z_t'][0], 350)  # expected bc defined here and no push
        self.assertEqual(df_data['pers'][0], True)  # expected bc perseveration
        self.assertEqual(df_data['cond'][0], 'main_noPush')  # expected bc defined here
        self.assertEqual(df_data['tot_samples'][0], 1)  # expected bc mock-function output

    @patch('sampling.AlAgentSampling.AlAgentSampling')
    @patch('numpy.random.binomial', new=Mock(return_value=1))
    def test_first_trial_pers_small_b_t(self, agent):
        """ This function implements a unit test of the task-agent interaction
            in the first trial when perseveration takes place. We mock out the
            sampling agent. b_t is < 0 and corrected.

            The test covers the first trial with the new_block[t] case,
            where the agent is initialized.
        """

        # Load function input
        df_subj = self.load_default_input()

        # Initialize agent variables
        agent_vars = AgentVarsSampling()
        agent_vars.mu_0 = -50
        agent.delta_t = -21

        # Run interaction function and test results
        df_data = task_agent_int_resource_only(df_subj, agent, agent_vars, n_trials=2)
        self.assertEqual(df_data['x_t'][0], 176)  # expected bc defined in df_subj
        self.assertEqual(df_data['mu_t'][0], -50)  # expected bc defined here
        self.assertEqual(df_data['delta_t'][0], -21)  # expected bc defined here
        self.assertEqual(df_data['omega_t'][0], 1)  # expected bc initial value
        self.assertEqual(df_data['tau_t'][0], 0.5)  # expected bc initial value
        self.assertEqual(df_data['alpha_t'][0], 0.0)  # expected bc perseveration
        self.assertEqual(df_data['sim_b_t'][0], -50)  # expected bc perseveration
        self.assertEqual(df_data['sim_b_t'][1], 0)  # expected bc perseveration
        self.assertEqual(df_data['sim_a_t'][0], 0)  # expected bc perseveration
        self.assertEqual(df_data['sim_y_t'][0], 0)  # expected bc defined here
        self.assertEqual(df_data['sim_z_t'][0], -50)  # expected bc defined here and no push
        self.assertEqual(df_data['pers'][0], True)  # expected bc perseveration
        self.assertEqual(df_data['cond'][0], 'main_noPush')  # expected bc defined here
        self.assertEqual(df_data['tot_samples'][0], 1)  # expected bc mock-function output

    @patch('sampling.AlAgentSampling.AlAgentSampling')
    def test_second_trial_pers(self, agent):
        """ This function implements a unit test of the task-agent interaction
            in the second trial when perseveration takes place. We mock out the
            sampling agent.
        """

        # Load function input
        df_subj = self.load_default_input()
        df_subj.loc[1, "y_t"] = 30

        # Initialize agent variables
        agent_vars = AgentVarsSampling()
        agent_vars.mu_0 = 34
        agent.delta_t = -21

        # Run interaction function and test results
        df_data = task_agent_int_resource_only(df_subj, agent, agent_vars, n_trials=3)
        self.assertEqual(df_data['x_t'][1], 176)  # expected bc defined in df_subj
        self.assertEqual(df_data['mu_t'][1], 34)  # expected bc defined here
        self.assertEqual(df_data['delta_t'][1], -21)  # expected bc defined here
        self.assertEqual(df_data['omega_t'][1], 1)  # expected bc initial value
        self.assertEqual(df_data['tau_t'][1], 0.5)  # expected bc initial value
        self.assertEqual(df_data['alpha_t'][1], 0.0)  # expected bc perseveration
        self.assertEqual(df_data['sim_b_t'][1], 34)  # expected bc perseveration
        self.assertEqual(df_data['sim_a_t'][1], 0)  # expected bc perseveration
        self.assertEqual(df_data['sim_y_t'][1], 30)  # expected bc defined here
        self.assertEqual(df_data['sim_z_t'][1], 64)  # expected bc defined here and no push
        self.assertEqual(df_data['pers'][1], True)  # expected bc perseveration
        self.assertEqual(df_data['cond'][1], 'main_noPush')  # expected bc defined here
        self.assertEqual(df_data['tot_samples'][1], 1)  # expected bc mock-function output

    @patch('sampling.AlAgentSampling.AlAgentSampling')
    def test_second_trial_pers_large_push_pos(self, agent):
        """ This function implements a unit test of the task-agent interaction
            in the second trial when perseveration takes place. We mock out the
            sampling agent.

            
            The test covers the second trial with an extremely large (positive) push
            that is corrected to avoid that it exceeds the screen coordinates.
        """

        # Load function input
        df_subj = self.load_default_input()
        df_subj.loc[1, "y_t"] = 300

        # Initialize agent variables
        agent_vars = AgentVarsSampling()
        agent_vars.mu_0 = 34
        agent.delta_t = -21

        # Run interaction function and test results
        df_data = task_agent_int_resource_only(df_subj, agent, agent_vars, n_trials=3)
        self.assertEqual(df_data['x_t'][1], 176)  # expected bc defined in df_subj
        self.assertEqual(df_data['mu_t'][1], 34)  # expected bc defined here
        self.assertEqual(df_data['delta_t'][1], -21)  # expected bc defined here
        self.assertEqual(df_data['omega_t'][1], 1)  # expected bc initial value
        self.assertEqual(df_data['tau_t'][1], 0.5)  # expected bc initial value
        self.assertEqual(df_data['alpha_t'][1], 0.0)  # expected bc perseveration
        self.assertEqual(df_data['sim_b_t'][1], 34)  # expected bc perseveration
        self.assertEqual(df_data['sim_a_t'][1], 0)  # expected bc perseveration
        self.assertEqual(df_data['sim_y_t'][1], 266)  # expected bc very large positive push
        self.assertEqual(df_data['sim_z_t'][1], 300)  # expected bc very large positive push
        self.assertEqual(df_data['pers'][1], True)  # expected bc perseveration
        self.assertEqual(df_data['cond'][1], 'main_noPush')  # expected bc defined here
        self.assertEqual(df_data['tot_samples'][1], 1)  # expected bc mock-function output

    @patch('sampling.AlAgentSampling.AlAgentSampling')
    def test_second_trial_pers_large_push_neg(self, agent):
        """ This function implements a unit test of the task-agent interaction
            in the second trial when perseveration takes place. We mock out the
            sampling agent.

            The test covers the second trial with an extremely large (negative) push 
            that is corrected avoit that it exceeds the screen coordinates.
        """

        # Load function input
        df_subj = self.load_default_input()
        df_subj.loc[1, "y_t"] = -300

        # Initialize agent variables
        agent_vars = AgentVarsSampling()
        agent_vars.mu_0 = 34
        agent.delta_t = -21

        # Run interaction function and test results
        df_data = task_agent_int_resource_only(df_subj, agent, agent_vars, n_trials=3)
        self.assertEqual(df_data['x_t'][1], 176)  # expected bc defined in df_subj
        self.assertEqual(df_data['mu_t'][1], 34)  # expected bc defined here
        self.assertEqual(df_data['delta_t'][1], -21)  # expected bc defined here
        self.assertEqual(df_data['omega_t'][1], 1)  # expected bc initial value
        self.assertEqual(df_data['tau_t'][1], 0.5)  # expected bc initial value
        self.assertEqual(df_data['alpha_t'][1], 0.0)  # expected bc perseveration
        self.assertEqual(df_data['sim_b_t'][1], 34)  # expected bc perseveration
        self.assertEqual(df_data['sim_a_t'][1], 0)  # expected bc perseveration
        self.assertEqual(df_data['sim_y_t'][1], -34)  # expected bc very large negative push
        self.assertEqual(df_data['sim_z_t'][1], 0)  # expected bc very large negative push
        self.assertEqual(df_data['pers'][1], True)  # expected bc perseveration
        self.assertEqual(df_data['cond'][1], 'main_noPush')  # expected bc defined here
        self.assertEqual(df_data['tot_samples'][1], 1)  # expected bc mock-function output

    @patch('sampling.AlAgentSampling.AlAgentSampling')
    def test_last_trial_pers(self, agent):
        """ This function implements a unit test of the task-agent interaction
            in the last trial when perseveration takes place. We mock out the
            sampling agent.

            We mostly expect nans here.
        """

        # Load function input
        df_subj = self.load_default_input()
        df_subj.loc[1, "y_t"] = 0

        # Initialize agent variables
        agent_vars = AgentVarsSampling()
        agent_vars.mu_0 = 34
        agent.delta_t = -21

        # Run interaction function and test results
        df_data = task_agent_int_resource_only(df_subj, agent, agent_vars, n_trials=3)
        self.assertTrue(np.isnan(df_data['x_t'][2]))
        self.assertTrue(np.isnan(df_data['mu_t'][2]))
        self.assertTrue(np.isnan(df_data['delta_t'][2]))
        self.assertTrue(np.isnan(df_data['omega_t'][2]))
        self.assertTrue(np.isnan(df_data['tau_t'][2]))
        self.assertTrue(np.isnan(df_data['alpha_t'][2]))
        self.assertEqual(df_data['sim_b_t'][2], 34)
        self.assertTrue(np.isnan(df_data['sim_a_t'][2]))
        self.assertTrue(np.isnan(df_data['sim_y_t'][2]))
        self.assertTrue(np.isnan(df_data['sim_z_t'][2]))
        self.assertEqual(df_data['pers'][2], 0.0)
        self.assertEqual(df_data['cond'][2], 'main_noPush')  # expected bc defined here
        self.assertTrue(np.isnan(df_data['tot_samples'][2]))

    @patch.object(AlAgentSampling, 'sampling', new=sampling)
    @patch.object(AlAgentSampling, 'reinitialize_agent', new=Mock(return_value=None))
    @patch.object(AlAgentSampling, 'compute_delta', new=Mock(return_value=None))
    @patch.object(AlAgentSampling, 'compute_cpp_samples', new=Mock(return_value=None))
    @patch.object(AlAgentSampling, 'compute_eu', new=Mock(return_value=None))
    @patch.object(AlAgentSampling, 'compute_ru', new=Mock(return_value=None))
    @patch('numpy.random.binomial', new=Mock(return_value=0))
    def test_first_trial_no_pers(self):
        """ This function implements a unit test of the task-agent interaction
            in the first trial when no perseveration takes place. We mock out the
            individual agent functions to simulate an update of mu_t to have
            a no-perseveration trial

            The test covers the first trial with the new_block[t] case,
            where the agent is initialized.
        """

        # Load function input
        df_subj = self.load_default_input()

        # Initialize agent based on agent_vars
        agent_vars = AgentVarsSampling()
        agent = AlAgentSampling(agent_vars)
        agent.delta_t = 100

        # Run interaction function and test results
        df_data = task_agent_int_resource_only(df_subj, agent, agent_vars, n_trials=2)
        self.assertEqual(df_data['x_t'][0], 176)  # expected bc defined in df_subj
        self.assertEqual(df_data['mu_t'][0], 201)  # expected bc defined in sampling mock function
        self.assertEqual(df_data['delta_t'][0], 100)  # expected bc defined here
        self.assertEqual(df_data['omega_t'][0], 0.64)  # expected bc defined in sampling mock function
        self.assertEqual(df_data['tau_t'][0], 0.5)  # expected bc initial value
        self.assertEqual(df_data['alpha_t'][0], 51/100)  # expected results of single-trial LR equation
        self.assertEqual(df_data['sim_b_t'][0], 150)  # expected bc initial value
        self.assertEqual(df_data['sim_a_t'][0], 51)  # mu_t = 201 - sim_b_t = 150
        self.assertEqual(df_data['sim_y_t'][0], 0)  # expected bc no push defined
        self.assertEqual(df_data['pers'][0], False)  # expected bc no update takes place
        self.assertEqual(df_data['cond'][0], 'main_noPush')  # expected bc defined here
        self.assertEqual(df_data['tot_samples'][0], 10)  # expected bc defined in sampling mock function

    def test_integration(self):
        """ This function implements an integration test based on data of the first participant (experiment 2) """

        df_exp2 = pd.read_pickle('al_data/data_prepr_2.pkl')

        # Extract subject-specific data frame
        df_subj = get_df_subj(df_exp2, 0)

        # Set agent parameters
        agent_vars = AgentVarsSampling()
        agent_vars.criterion = np.nan
        agent_vars.n_samples = 5
        agent_vars.model_sat = False
        agent = AlAgentSampling(agent_vars)

        # Run task-agent interaction
        agent.reinitialize_agent(seed=1)
        df_data = task_agent_int_resource_only(df_subj, agent, agent_vars, seed=1)

        # savename = 'al_tests/test_task_agent_int_resource_only.pkl'
        # df_data.to_pickle(savename)

        # Load test data
        test_data = pd.read_pickle('al_tests/test_task_agent_int_resource_only.pkl')

        # Test function output
        self.assertTrue(test_data.equals(df_data))

    @staticmethod
    def load_default_input():
        """ This function loads the default df_subj for task_agent_int_sampling.py.

        :return: df_subj: Data frame with synthetic subject data
        """

        # Create data frame for simulation
        data = {'subj_num': [1, 1, 1], 'age_group': [1, 1, 1], 'new_block': [1, 0, 0],
                'x_t': [176, 176, 176], 'mu_t': [176, 176, 176], 'y_t': [0, 0, 0],
                'cond': ['main_noPush', 'main_noPush', 'main_noPush']}
        df_subj = pd.DataFrame(data)

        return df_subj


# Run unit test
if __name__ == '__main__':
    unittest.main()
