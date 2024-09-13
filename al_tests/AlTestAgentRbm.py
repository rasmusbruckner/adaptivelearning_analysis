""" Test Agent RBM: Unit tests for the RBM """

import numpy as np
import unittest
import pandas as pd
from AlAgentVarsRbm import AgentVars
from AlAgentRbm import AlAgent


class TestAgent(unittest.TestCase):
    """ This class implements the unit test in order to test the agent class

        We have the following test functions:

            - test_agent_init: Initialization of AlAgentRBM
            - test_agent_learn: Learning function of the agent model
            - test_agent_learn_with_rew_bias: Learning function of the agent model with reward bias
            - test_agent_learn_with_ct: Learning function of the agent model with catch trials
            - test_agent_learn_with_nan_delta: Learning function of the agent model when delta_t = nan
    """

    def test_agent_init(self):
        # This function tests the agent initialization based on AgentVars

        # Initialize agent based on agent_vars
        agent_vars = AgentVars()
        agent = AlAgent(agent_vars)

        # Test initial agent variables
        self.assertEqual(agent.s, 1)
        self.assertEqual(agent.h, 0.1)
        self.assertEqual(agent.u, 0.0)
        self.assertEqual(agent.q, 0.0)
        self.assertEqual(agent.sigma, 10)
        self.assertEqual(agent.sigma_t_sq, 100)
        self.assertEqual(agent.sigma_H, 1)
        self.assertEqual(agent.tau_t, 0.5)
        self.assertEqual(agent.omega_t, 1)
        self.assertEqual(agent.mu_t, 150)
        self.assertTrue(np.isnan(agent.a_t))
        self.assertTrue(np.isnan(agent.alpha_t))
        self.assertTrue(np.isnan(agent.tot_var))
        self.assertTrue(np.isnan(agent.C))

    def test_agent_learn(self):
        # This function tests the learning function of the agent model without catch trials

        # Initialize agent based on agent_vars
        agent_vars = AgentVars()
        agent_vars.u = np.exp(0)
        agent = AlAgent(agent_vars)

        # In task_agent_int, input is based on data frame
        df = pd.DataFrame(index=range(0, 1), dtype='float')  # create this frame here
        df['delta_t'] = 50  # add prediction error
        df['b_t'] = 150  # add participant prediction
        df['r_t'] = 0  # add high-reward index
        df['v_t'] = 0  # add helicopter visibility
        df['mu_t'] = 0  # add true heli location (here zero, bc we don't test catch trials)

        # Extract delta and high_val as in task_agent_int
        delta = df['delta_t']
        high_val = df['r_t'] == 1  # indicates high value trials

        # Apply learning function
        agent.learn(delta[0], df['b_t'][0], df['v_t'][0], df['mu_t'][0], high_val[0])

        self.assertEqual(agent.s, 1)
        self.assertEqual(agent.h, 0.1)
        self.assertEqual(agent.u, 1)
        self.assertEqual(agent.q, 0)
        self.assertEqual(agent.sigma, 10)
        self.assertAlmostEqual(agent.sigma_t_sq, 163.43733523, 6)
        self.assertEqual(agent.sigma_H, 1)
        self.assertAlmostEqual(agent.tau_t, 0.62040308, 6)
        self.assertAlmostEqual(agent.omega_t, 0.8718136, 6)
        self.assertAlmostEqual(agent.mu_t, 196.79533994, 6)
        self.assertTrue(np.isnan(agent.C))
        self.assertAlmostEqual(agent.a_t, 46.79533994, 6)
        self.assertAlmostEqual(agent.alpha_t, 0.9359068, 6)
        self.assertAlmostEqual(agent.tot_var, 200, 6)

    def test_agent_learn_with_rew_bias(self):
        # This function tests the learning function of the agent model without catch trials
        # but including the reward bias

        # Initialize agent based on agent_vars
        agent_vars = AgentVars()
        agent_vars.u = np.exp(0)
        agent_vars.q = 0.025
        agent = AlAgent(agent_vars)

        # In task_agent_int, input is based on data frame
        df = pd.DataFrame(index=range(0, 1), dtype='float')  # create this frame here
        df['delta_t'] = 50  # add prediction errors
        df['b_t'] = 150  # add participant predictions
        df['r_t'] = 1  # add high-reward index
        df['v_t'] = 0  # add helicopter visibility
        df['mu_t'] = 0  # add true heli location (here zero, bc we don't test catch trials)

        # Extract delta and high_val as in task_agent_int
        delta = df['delta_t']
        high_val = df['r_t'] == 1  # indicates high value trials

        # Apply learning function
        agent.learn(delta[0], df['b_t'][0], df['v_t'][0], df['mu_t'][0], high_val[0])

        self.assertEqual(agent.s, 1)
        self.assertEqual(agent.h, 0.1)
        self.assertEqual(agent.u, 1)
        self.assertEqual(agent.q, 0.025)
        self.assertEqual(agent.sigma, 10)
        self.assertAlmostEqual(agent.sigma_t_sq, 163.43733523, 6)
        self.assertEqual(agent.sigma_H, 1)
        self.assertAlmostEqual(agent.tau_t, 0.62040308, 6)
        self.assertAlmostEqual(agent.omega_t, 0.8718136, 6)
        self.assertAlmostEqual(agent.mu_t, 198.04533994275727, 6)
        self.assertTrue(np.isnan(agent.C))
        self.assertAlmostEqual(agent.a_t, 48.04533994275727, 6)
        self.assertAlmostEqual(agent.alpha_t, 0.9609067988551455, 6)
        self.assertAlmostEqual(agent.tot_var, 200, 6)

    def test_agent_learn_with_ct(self):
        # This function tests the learning function of the agent model with catch trials

        # Initialize agent based on agent_vars
        agent_vars = AgentVars()
        agent_vars.u = np.exp(0)
        agent = AlAgent(agent_vars)

        # In task_agent_int, input is based on data frame
        df = pd.DataFrame(index=range(0, 1), dtype='float')  # create this frame here
        df['delta_t'] = 50  # add prediction errors
        df['b_t'] = 150  # add participant predictions
        df['r_t'] = 0  # add high-reward index
        df['v_t'] = 1  # add helicopter visibility
        df['mu_t'] = 190  # add true heli location

        # Extract delta and high_val as in task_agent_int
        delta = df['delta_t']
        high_val = df['r_t'] == 1  # indicates high value trials

        # Apply learning function
        agent.learn(delta[0], df['b_t'][0], df['v_t'][0], df['mu_t'][0], high_val[0])

        self.assertEqual(agent.s, 1)
        self.assertEqual(agent.h, 0.1)
        self.assertEqual(agent.u, 1)
        self.assertEqual(agent.q, 0)
        self.assertEqual(agent.sigma, 10)
        self.assertAlmostEqual(agent.sigma_t_sq, 361.24233883, 6)
        self.assertEqual(agent.sigma_H, 1)
        self.assertAlmostEqual(agent.tau_t, 0.78319423, 6)
        self.assertAlmostEqual(agent.omega_t, 0.8718136, 6)
        self.assertAlmostEqual(agent.mu_t, 190.06728059, 6)
        self.assertAlmostEqual(agent.C, 0.99009901, 6)
        self.assertAlmostEqual(agent.a_t, 40.06728059, 6)
        self.assertAlmostEqual(agent.alpha_t, 0.9359068, 6)
        self.assertAlmostEqual(agent.tot_var, 200, 6)

    def test_agent_learn_with_nan_delta(self):
        """ This function tests the learning function of the agent model when delta_t = nan
            and the agent should crash with warning message
        """

        # Initialize agent based on agent_vars
        agent_vars = AgentVars()
        agent_vars.u = np.exp(0)
        agent = AlAgent(agent_vars)

        # In task_agent_int, input is based on data frame
        df = pd.DataFrame(index=range(0, 1), dtype='float')  # create this frame here
        df['delta_t'] = np.nan  # add prediction error = nan
        df['b_t'] = 150  # add participant predictions
        df['r_t'] = 0  # add high-reward index
        df['v_t'] = 0  # add helicopter visibility
        df['mu_t'] = 150  # add true heli location

        # Extract delta and high_val as in task_agent_int
        delta = df['delta_t']
        high_val = df['r_t'] == 1  # indicates high value trials

        # Apply learning function
        with self.assertRaises(SystemExit):
            agent.learn(delta[0], df['b_t'][0], df['v_t'][0], df['mu_t'][0], high_val[0])


# Run unit test
if __name__ == '__main__':
    unittest.main()
