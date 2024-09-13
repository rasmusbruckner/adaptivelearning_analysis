""" Test Simulation RBM: Unit test of the simulation function """

import unittest
import numpy as np
from unittest.mock import patch
from unittest.mock import Mock
import pandas as pd
from al_simulation_rbm import simulation


class TestSimulationRBM(unittest.TestCase):
    """ This class implements the unit test of the simulation function (RBM)

            We have the following test functions:

            - test_simulation_rbm_exp1: Experiment 1
            - test_simulation_rbm_exp2: Experiment 2
    """

    def task_agent_int_exp1(*args, sim=1):
        """ This function mocks out the task-agent-interaction function for unit testing (experiment 1) """

        # Negative log-likelihodd
        llh_mix = -10

        # Create data frame with function output for unit testing
        df_data = pd.DataFrame(index=range(0, 5))
        df_data['sim_b_t'] = [160, 140, 130, 160, 145]
        df_data['c_t'] = [1, 0, 0, 1, 0]
        df_data['mu_t'] = [150, 150, 150, 150, 150]
        df_data['cond'] = ['main', 'main', 'main', 'main', 'main']
        df_data['pers'] = [1, 0, 1, 1, 0]
        df_data['sim_a_t'] = [0.0, 1, 0.0, 0.0, 1]

        return llh_mix, df_data

    def task_agent_int_exp2(*args, sim=1):
        """ This function mocks out the task-agent-interaction function for unit testing (experiment 2) """

        # Negative log-likelihodd
        llh_mix = -10

        # Create data frame with function output for unit testing
        df_data = pd.DataFrame(index=range(0, 10))
        df_data['sim_b_t'] = [160, 140, 130, 160, 145, 160, 160, 180, 160, 150]
        df_data['c_t'] = [1, 0, 0, 1, 0, 1, 0, 0, 1, 1]
        df_data['mu_t'] = [150, 150, 150, 150, 150, 150, 150, 150, 150, 150]
        df_data['cond'] = ['main_noPush', 'main_noPush', 'main_noPush', 'main_noPush', 'main_noPush',
                           'main_push', 'main_push', 'main_push', 'main_push', 'main_push']
        df_data['pers'] = [1, 0, 1, 1, 0, 1, 0, 0, 0, 1]
        df_data['sim_a_t'] = [0.0, 1, 0.0, 0.0, 1, 0.0, 1, 1, 1, 0.0]

        return llh_mix, df_data

    @patch('al_simulation_rbm.AlAgent', new=Mock(return_value=None))
    @patch('al_simulation_rbm.task_agent_int', new=task_agent_int_exp1)
    @patch('al_simulation_rbm.get_sim_est_err', new=Mock(return_value=np.array([1])))
    def test_simulation_rbm_exp1(self):
        """ This function tests the simulation function for the RBM in experiment 1 """

        # Create data frame with input data
        df_exp = pd.DataFrame(index=range(0, 5))
        df_exp['c_t'] = [1, 0, 0, 1, 0]
        df_exp['mu_t'] = [150, 150, 150, 150, 150]
        df_exp['cond'] = ['main', 'main', 'main', 'main', 'main']
        df_exp['subj_num'] = np.repeat(1, 5).tolist()

        # Create data frame with model parameters
        df_model = pd.DataFrame(index=np.arange(1))
        df_model['subj_num'] = 1
        df_model['age_group'] = 1
        df_model['omikron_0'] = 1
        df_model['omikron_1'] = 0.04
        df_model['b_0'] = 0.04
        df_model['b_1'] = 0.04
        df_model['h'] = 0.04
        df_model['s'] = 0.04
        df_model['u'] = 0.04
        df_model['q'] = 0.04
        df_model['d'] = 0.04
        df_model['sigma_H'] = 0.04

        # Run simulation function
        sim_pers = True
        sim_est_err, sim_pers_prob, df_sim, true_params = simulation(df_exp, df_model, 1, sim_pers)

        # Expected perseveration probability data frame
        expected_sim_pers_prob = pd.DataFrame(index=range(0, 1))
        expected_sim_pers_prob['main'] = [0.6]  # 3/5 in above mock function
        expected_sim_pers_prob['age_group'] = [1.0]  # defined in model input

        # Expected estimation error data frame
        expected_sim_est_err = pd.DataFrame(index=range(0, 1))
        expected_sim_est_err['main'] = [1.0]  # mock output of get_sim_est_err
        expected_sim_est_err['age_group'] = [1.0]  # defined in model input

        # Expected simulation data frame from mock function above
        _, expected_df_sim = self.task_agent_int_exp1()
        expected_df_sim['subj_num'] = 1

        # Test output
        self.assertTrue(expected_sim_pers_prob.equals(sim_pers_prob))
        self.assertTrue(expected_sim_est_err.equals(sim_est_err))
        self.assertTrue(expected_df_sim.equals(df_sim))
        self.assertTrue(df_model.equals(true_params))

    @patch('al_simulation_rbm.AlAgent', new=Mock(return_value=None))
    @patch('al_simulation_rbm.task_agent_int', new=task_agent_int_exp2)
    @patch('al_simulation_rbm.get_sim_est_err', new=Mock(return_value=np.array([1, 2])))
    def test_simulation_rbm_exp2(self):
        """ This function tests the simulation function for the RBM in experiment 2 """

        # Create data frame with input data
        df_exp = pd.DataFrame(index=range(0, 10), dtype='float')
        df_exp['c_t'] = [1, 0, 0, 1, 0, 1, 0, 0, 1, 1]
        df_exp['mu_t'] = [150, 150, 150, 150, 150, 150, 150, 150, 150, 150]
        df_exp['cond'] = ['main_noPush', 'main_noPush', 'main_noPush', 'main_noPush', 'main_noPush',
                          'main_push', 'main_push', 'main_push', 'main_push', 'main_push']
        df_exp['subj_num'] = np.repeat(1, 10).tolist()

        # Create data frame with model parameters
        df_model = pd.DataFrame(index=[np.arange(1)])
        df_model['subj_num'] = 1
        df_model['age_group'] = 1
        df_model['omikron_0'] = 1
        df_model['omikron_1'] = 0.04
        df_model['b_0'] = 0.04
        df_model['b_1'] = 0.04
        df_model['h'] = 0.04
        df_model['s'] = 0.04
        df_model['u'] = 0.04
        df_model['q'] = 0.04
        df_model['d'] = 0.04
        df_model['sigma_H'] = 0.04

        # Run simulation function
        sim_pers = True
        sim_est_err, sim_pers_prob, df_sim, true_params = simulation(df_exp, df_model, 1, sim_pers, which_exp=2)

        # Expected perseveration probability data frame
        expected_sim_pers_prob = pd.DataFrame(index=range(0, 1), dtype='float')
        expected_sim_pers_prob['noPush'] = [0.6]  # 3/5 in above mock function
        expected_sim_pers_prob['push'] = [0.4]  # 2/5 in above mock function
        expected_sim_pers_prob['age_group'] = [1.0]  # defined in model input

        # Expected estimation error data frame
        expected_sim_est_err = pd.DataFrame(index=range(0, 1), dtype='float')
        expected_sim_est_err['noPush'] = [1.0]  # mock output of get_sim_est_err
        expected_sim_est_err['push'] = [2.0]  # mock output of get_sim_est_err
        expected_sim_est_err['age_group'] = [1.0]  # defined in model input

        # Expected simulation data frame from mock function above
        _, expected_df_sim = self.task_agent_int_exp2()
        expected_df_sim['subj_num'] = 1

        # Test output
        self.assertTrue(expected_sim_pers_prob.equals(sim_pers_prob))
        self.assertTrue(expected_sim_est_err.equals(sim_est_err))
        self.assertTrue(expected_df_sim.equals(df_sim))
        self.assertTrue(df_model.equals(true_params))


# Run unit test
if __name__ == '__main__':
    unittest.main()
