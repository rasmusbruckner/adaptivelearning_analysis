""" Test Agent Sampling: Unit and integration tests for the sampling model """

import unittest
from sampling.AlAgentSampling import AlAgentSampling
from sampling.AlAgentVarsSampling import AgentVarsSampling
import numpy as np
from al_utilities import safe_div
from unittest.mock import patch
from unittest.mock import Mock


class TestAgentSampling(unittest.TestCase):
    """ This class definition implements the sampling-agent unit tests
        in order to test the critical learning and sampling functions

        We have the following test functions

            - test_agent_init: Agent initialization parameters
            - test_reinitialize_agent: Reinitialization of the agent
            - test_compute_delta: Computation of prediction error
            - test_compute_likelihood: Likelihood function
            - test_compute_prior: Prior function
            - test_compute_posterior: Posterior function
            - test_compute_cpp_samples: Computation of changepoint probability
            - test_compute_eu: Computation of estimation uncertainty
            - test_compute_ru: Computation of relative uncertainty
            - test_metropolis_hastings: Metropolis-Hastings algorithm
            - test_satisficing: Satisficing mechanism
            - test_sampling: Sampling process combining Metropolis-Hastings and satisficing
            - test_integration_sampling_sat: Integration test sampling function with satisficing
            - test_integration_sampling_opt: Integration test sampling function no satisficing
    """

    def metropolis_hastings(self):
        """ This function mocks out the metropolis-hastings function for unit testing """

        self.sample_curr = 200
        self.n_acc = 10

    def test_agent_init(self):
        # This function tests the agent initialization based on AgentVars

        # Initialize agent based on agent_vars
        agent_vars = AgentVarsSampling()
        agent = AlAgentSampling(agent_vars)

        # RBM variables
        self.assertEqual(agent.mu_0, 150)
        self.assertEqual(agent.sigma_0, 100)
        self.assertEqual(agent.sigma_t_sq, agent.sigma_0)
        self.assertEqual(agent.sigma, 17.5)
        self.assertEqual(agent.h, 0.1)
        self.assertEqual(agent.tau_0, 0.5)
        self.assertEqual(agent.tau_t, 0.5)
        self.assertEqual(agent.omega_0, 1)
        self.assertEqual(agent.omega_t, 1)

        # Sampling-model variables
        self.assertTrue(agent.model_sat)
        self.assertEqual(agent.n_samples, 30)
        self.assertEqual(agent.criterion, 1.0)
        self.assertEqual(agent.burn_in, 0)
        self.assertEqual(agent.sample_std, 30)

        # Directly initialized variables
        self.assertTrue(np.isnan(agent.x_t))
        self.assertTrue(np.isnan(agent.delta_t))
        self.assertTrue(np.isnan(agent.tot_var))
        self.assertTrue(np.isnan(agent.sample_curr))
        self.assertTrue(np.isnan(agent.sample_new))
        self.assertTrue(np.isnan(agent.mu_sampling))
        self.assertTrue(np.isnan(agent.mu_satisficing))
        self.assertTrue(np.isnan(agent.n_acc))
        self.assertTrue(np.isnan(agent.frac_acc))
        self.assertTrue(np.isnan(agent.tot_samples))
        self.assertEqual(agent.samples, [])
        self.assertEqual(agent.r_satisficing, [])

    def test_reinitialize_agent(self):
        """ This function tests the reinitialization function of the sampling agent """

        # Initialize agent based on agent_vars
        agent_vars = AgentVarsSampling()
        agent = AlAgentSampling(agent_vars)

        # Call reinitialization function
        agent.reinitialize_agent()

        self.assertEqual(agent.samples, [])
        self.assertEqual(agent.r_satisficing, [])
        self.assertEqual(agent.mu_t, agent.mu_0)
        self.assertEqual(agent.sigma_t_sq, agent.sigma_0)
        self.assertEqual(agent.omega_t, agent.omega_0)
        self.assertEqual(agent.tau_t, agent.tau_0)
        self.assertEqual(agent.sample_curr, agent.mu_0)
        self.assertEqual(agent.tot_var, agent.sigma ** 2 + agent.sigma_t_sq)

    def test_compute_delta(self):
        """" This function tests the prediction-error function of the agent """

        # Initialize agent based on agent_vars
        agent_vars = AgentVarsSampling()
        agent = AlAgentSampling(agent_vars)

        # Set test outcome and mean, and call PE function
        agent.x_t = 200
        agent.mu_t = 150
        agent.compute_delta()

        self.assertEqual(agent.delta_t, agent.x_t - agent.mu_t)

    def test_compute_likelihood(self):
        """ This function tests the likelihood function of the agent """

        # Initialize agent based on agent_vars
        agent_vars = AgentVarsSampling()
        agent = AlAgentSampling(agent_vars)

        # Set test outcome and sample, and call likelihood function
        agent.x_t = 200
        x = 150
        likelihood = agent.compute_likelihood(x)

        self.assertEqual(likelihood, 0.00038480568429887104)

    def test_compute_prior(self):
        """ This function tests the prior function of the agent """

        # Initialize agent based on agent_vars
        agent_vars = AgentVarsSampling()
        agent = AlAgentSampling(agent_vars)

        # Set test mu_0 and sample, and call prior function
        agent.mu_0 = 100
        x = 105
        prior = agent.compute_prior(x)

        self.assertEqual(prior, 0.03520653267642995)

    def test_compute_posterior(self):
        """ This function tests the posterior function of the agent """

        # Initialize agent based on agent_vars
        agent_vars = AgentVarsSampling()
        agent = AlAgentSampling(agent_vars)

        # Set test outcome and sample, and call likelihood function
        agent.x_t = 110
        x = 99
        likelihood = agent.compute_likelihood(x)

        # Set test mu_t and call prior function
        agent.mu_t = 100
        prior = agent.compute_prior(x)

        # Compute posterior
        posterior = agent.compute_posterior(prior, likelihood)
        expected_posterior = np.multiply(prior, likelihood) * (1 - agent.h) + np.multiply((1 / 300), likelihood) * agent.h

        self.assertEqual(posterior, expected_posterior)

    def test_compute_cpp_samples(self):
        """ This function tests the changepoint-probability function where we approximate
            CPP based on the samples 
        
            We test different levels of resulting changepoint-probability values
        """

        # Initialize agent based on agent_vars
        agent_vars = AgentVarsSampling()
        agent = AlAgentSampling(agent_vars)

        # Test small PE (sample approximation)
        samples_delta = [1, -1, 1]
        agent.tot_var = 10
        agent.compute_cpp_samples(samples_delta)
        self.assertEqual(agent.omega_t, 0.003076823519660333)

        # Test medium PE (sample approximation)
        samples_delta = [11, 10, 13]
        agent.tot_var = 10
        agent.compute_cpp_samples(samples_delta)
        self.assertEqual(agent.omega_t, 0.5967190622018047)

        # Test large PE (sample approximation)
        samples_delta = [29, 30, 31]
        agent.tot_var = 10
        agent.compute_cpp_samples(samples_delta)
        self.assertEqual(agent.omega_t, 1.0)

    def test_compute_eu(self):
        """ This function tests the estimation-uncertainty function

            We test the function using different omega_t values. Here the idea is that when omega_t = 0,
            estimation uncertainty is equal to the updated variance of a Gaussian distribution in standard
            Bayesian inference. When omega_t = 1, estimation is fully determined by the outcome variability
            sigma ** 2. Finally, we also test a mixture of the two cases (omega_t = 0.5). For details on the
            Gaussian case, see for example: https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
        """

        # Initialize agent based on agent_vars
        agent_vars = AgentVarsSampling()
        agent = AlAgentSampling(agent_vars)

        # Initialize PE
        agent.delta_t = 5

        # Test the omega_t = 0 case
        # -------------------------

        # Set parameters and compute estimation uncertainty
        agent.omega_t = 0.0  # now we only expect a standard Gaussian variance update
        agent.tau_t = agent.sigma_0 / (agent.sigma_0 + agent.sigma ** 2)  # Set relative uncertainty
        # so that we can straightforwardly compare the equations
        sigma_t_sq = agent.sigma_0  # ensure that estimation uncertainty is set to its initial value
        agent.compute_eu()  # update estimation uncertainty

        # Equation for Gaussian variance update
        omega_0_case = (agent.sigma ** 2 * sigma_t_sq) / (1 * sigma_t_sq + agent.sigma ** 2)

        # Test result
        self.assertEqual(agent.sigma_t_sq, omega_0_case)

        # Test the omega_t = 1 case
        # -------------------------

        # Set parameters and compute estimation uncertainty
        agent.omega_t = 1.0  # now we expect sigma ** 2
        agent.tau_t = agent.sigma_0 / (agent.sigma_0 + agent.sigma ** 2)  # Set relative uncertainty
        agent.compute_eu()  # update estimation uncertainty

        # Expected result for omega_t = 1 case
        omega_1_case = agent.sigma ** 2

        # Test result
        self.assertEqual(agent.sigma_t_sq, omega_1_case)

        # Test the mixture case
        # ---------------------

        # Set parameters and compute estimation uncertainty
        agent.omega_t = 0.5  # now we expect a mixture
        agent.tau_t = agent.sigma_0 / (agent.sigma_0 + agent.sigma ** 2)  # Set relative uncertainty
        agent.compute_eu()  # update estimation uncertainty

        # Test result
        self.assertEqual(agent.sigma_t_sq, 194.36908284023667)

    def test_compute_ru(self):
        """ This function tests the relative-uncertainty function """

        # Initialize agent based on agent_vars
        agent_vars = AgentVarsSampling()
        agent = AlAgentSampling(agent_vars)

        # Compute relative uncertainty
        agent.compute_ru()

        tau_t = safe_div(agent.sigma_t_sq, (agent.sigma_t_sq + agent.sigma ** 2))
        self.assertEqual(agent.tau_t, tau_t)

    @patch('random.normalvariate', new=Mock(return_value=180))
    @patch('random.uniform', new=Mock(return_value=0.9))
    @patch.object(AlAgentSampling, 'compute_prior', new=Mock(return_value=[np.nan, np.nan]))
    @patch.object(AlAgentSampling, 'compute_likelihood', new=Mock(return_value=[np.nan, np.nan]))
    @patch.object(AlAgentSampling, 'compute_posterior', new=Mock(side_effect=[[1, 2], [2, 1]]))
    def test_metropolis_hastings(self):
        """ This function tests the Metropolis-Hastings function

            We test two cases: Agent accepts sample and agent rejects sample.
        """

        # Initialize agent based on agent_vars
        agent_vars = AgentVarsSampling()
        agent = AlAgentSampling(agent_vars)

        # Test the case where sample is accepted
        agent.sample_curr = 170
        agent.n_acc = 10
        agent.metropolis_hastings()
        self.assertEqual(agent.sample_curr, 180)
        self.assertEqual(agent.n_acc, 11)

        # Test the case where sample is rejected
        agent.sample_curr = 170
        agent.metropolis_hastings()
        self.assertEqual(agent.sample_curr, 170)
        self.assertEqual(agent.n_acc, 11)

    @patch.object(AlAgentSampling, 'compute_prior', new=Mock(return_value=[np.nan, np.nan]))
    @patch.object(AlAgentSampling, 'compute_likelihood', new=Mock(return_value=[np.nan, np.nan]))
    @patch.object(AlAgentSampling, 'compute_posterior', new=Mock(side_effect=[[1, 3], [3, 1]]))
    def test_satisficing(self):
        """ This function tests the satisficing function

            We test two cases: Agent satisfices and agent keeps sampling
        """

        # Initialize agent based on agent_vars
        agent_vars = AgentVarsSampling()
        agent = AlAgentSampling(agent_vars)

        # Test the case where satisficing takes place
        agent.mu_t = 100
        agent.mu_satisficing = 110
        stop_sampling = agent.satisficing()
        self.assertEqual(agent.mu_t, 110)
        self.assertTrue(stop_sampling)
        self.assertEqual(agent.r_satisficing[0], 1 / 4)

        # Test the case where model keeps sampling
        agent.criterion = 0.1
        agent.mu_t = 100
        agent.mu_satisficing = 110
        agent.r_satisficing = []
        stop_sampling = agent.satisficing()
        self.assertEqual(agent.mu_t, 100)
        self.assertFalse(stop_sampling)
        self.assertEqual(agent.r_satisficing[0], 0.75)

    @patch.object(AlAgentSampling, 'metropolis_hastings', new=metropolis_hastings)
    @patch.object(AlAgentSampling, 'satisficing', new=Mock(side_effect=[True, False, True, True, False, True]))
    def test_sampling(self):
        """ This function tests the sampling function

            We test different combinations of perseveration, push, and satisficing. The Metropolis-Hastings
            and satisficing functions are mocked out.
        """

        # Initialize agent based on agent_vars
        agent_vars = AgentVarsSampling()
        agent = AlAgentSampling(agent_vars)

        # 1. Perseveration, no push, model_sat = True
        # -------------------------------------------

        agent.reinitialize_agent()
        agent.mu_t = 50  # current belief
        push = 0.0  # no push
        agent.sample_curr = agent.mu_t + push  # initial sample (current belief and push)
        agent.n_samples = 1  # number of samples

        # Run sampling function and test results
        agent.sampling()
        self.assertEqual(agent.mu_satisficing, 50)  # expected bc 50 is starting point for sampling
        self.assertEqual(agent.n_acc, 10)  # expected bc of mock Metropolis-Hastings function
        self.assertEqual(agent.mu_sampling, 200)  # expected bc of mock Metropolis-Hastings function
        self.assertEqual(agent.mu_0, 50)  # expected bc agent perseverates
        self.assertEqual(agent.mu_t, 50)  # expected bc agent perseverates
        self.assertEqual(agent.frac_acc, 10)  # expected bc of mock Metropolis-Hastings function
        self.assertCountEqual(agent.samples, [200])  # expected bc of mock Metropolis-Hastings function

        # 2. No perseveration, no push, model_sat = True
        # ----------------------------------------------

        agent.reinitialize_agent()
        agent.mu_t = 80  # current belief
        push = 0.0  # no push
        agent.sample_curr = agent.mu_t + push  # initial sample (current belief and push)
        agent.n_samples = 1  # number of samples

        # Run sampling function and test results
        agent.sampling()
        self.assertEqual(agent.mu_satisficing, 200)  # expected bc no perseveration
        self.assertEqual(agent.n_acc, 10)  # expected bc of mock Metropolis-Hastings function
        self.assertEqual(agent.mu_sampling, 200)  # expected bc of mock Metropolis-Hastings function
        self.assertEqual(agent.mu_0, 80)  # expected bc sampling function is mocked out
        # and mu_t not updated (just stop_sampling variable)
        self.assertEqual(agent.mu_t, 80)  # same here
        self.assertEqual(agent.frac_acc, 10 / 2)  # expected bc of mock Metropolis-Hastings function
        self.assertCountEqual(agent.samples, [200, 200])  # expected bc of mock Metropolis-Hastings function

        # 3. Perseveration, with push, model_sat = True
        # ---------------------------------------------

        agent.reinitialize_agent()
        agent.mu_t = 250  # current belief
        push = 20.83  # with push
        agent.sample_curr = agent.mu_t + push  # initial sample (current belief and push)
        agent.n_samples = 1  # number of samples

        # Run sampling function and test results
        agent.sampling()
        self.assertEqual(agent.mu_satisficing, 270.83)  # expected bc 270.83 is starting point for sampling
        self.assertEqual(agent.n_acc, 10)  # expected bc of mock Metropolis-Hastings function
        self.assertEqual(agent.mu_sampling, 200)  # expected bc of mock Metropolis-Hastings function
        self.assertEqual(agent.mu_0, 250)  # expected bc agent perseverates
        self.assertEqual(agent.mu_t, 250)  # expected bc agent perseverates
        self.assertEqual(agent.frac_acc, 10)  # expected bc of mock Metropolis-Hastings function
        self.assertCountEqual(agent.samples, [200])  # expected bc of mock Metropolis-Hastings function

        # 4. No perseveration, with push, model_sat = True
        # ------------------------------------------------

        agent.reinitialize_agent()
        agent.mu_t = 44  # current belief
        push = 20.83  # with push
        agent.sample_curr = agent.mu_t + push  # initial sample (current belief and push)
        agent.n_samples = 1  # number of samples

        # Run sampling function and test results
        agent.sampling()
        self.assertEqual(agent.mu_satisficing, 200)  # expected bc no perseveration
        self.assertEqual(agent.n_acc, 10)  # expected bc of mock Metropolis-Hastings function
        self.assertEqual(agent.mu_sampling, 200)  # expected bc of mock Metropolis-Hastings function
        self.assertEqual(agent.mu_0, 44)  # expected bc sampling function is mocked out
        # and mu_t not updated (just stop_sampling variable)
        self.assertEqual(agent.mu_t, 44)  # same here
        self.assertEqual(agent.frac_acc, 10 / 2)  # expected bc of mock Metropolis-Hastings function
        self.assertCountEqual(agent.samples, [200, 200])  # expected bc of mock Metropolis-Hastings function

        # 5. No perseveration, no push, model_sat = False
        # -----------------------------------------------

        agent.reinitialize_agent()
        agent.model_sat = False
        agent.mu_t = 230  # current belief
        push = 0.0  # no push
        agent.sample_curr = agent.mu_t + push  # initial sample (current belief and push)
        agent.n_samples = 1  # number of samples

        # Run sampling function and test results
        agent.sampling()
        self.assertEqual(agent.mu_satisficing, 230)  # expected bc 230 is starting point for sampling
        self.assertEqual(agent.n_acc, 10)  # expected bc of mock Metropolis-Hastings function
        self.assertEqual(agent.mu_sampling, 200)  # expected bc satisficing off and mu_t = mu_sampling
        self.assertEqual(agent.mu_0, 200)  # expected bc of mock Metropolis-Hastings function
        self.assertEqual(agent.mu_t, 200)  # expected bc satisficing off and mu_t = mu_sampling
        self.assertEqual(agent.frac_acc, 10)  # expected bc of mock Metropolis-Hastings function
        self.assertCountEqual(agent.samples, [200])  # expected bc of mock Metropolis-Hastings function

        # 6. No perseveration, with push, model_sat = False
        # -------------------------------------------------

        agent.reinitialize_agent()
        agent.model_sat = False
        agent.mu_t = 170  # current belief
        push = 20.83  # with push
        agent.sample_curr = agent.mu_t + push  # initial sample (current belief and push)
        agent.n_samples = 1  # number of samples

        # Run sampling function and test results
        agent.sampling()
        self.assertAlmostEqual(agent.mu_satisficing, 190.83)  # expected bc 190.83 is starting point for sampling
        self.assertEqual(agent.n_acc, 10)  # expected bc of mock Metropolis-Hastings function
        self.assertEqual(agent.mu_sampling, 200)  # expected bc of mock Metropolis-Hastings function
        self.assertEqual(agent.mu_0, 200)  # expected bc of mock Metropolis-Hastings function
        self.assertEqual(agent.mu_t, 200)  # expected bc of mock Metropolis-Hastings function
        self.assertEqual(agent.frac_acc, 10)  # expected bc of mock Metropolis-Hastings function
        self.assertCountEqual(agent.samples, [200])  # expected bc of mock Metropolis-Hastings function

    def test_integration_sampling_sat(self):
        """ This function implements an integration test for the sampling function with satisficing """

        # Initialize agent based on agent_vars
        agent_vars = AgentVarsSampling()
        agent_vars.n_samples = 5
        agent = AlAgentSampling(agent_vars)

        # 1. Perseveration, no push
        # -------------------------

        agent.reinitialize_agent(seed=1)
        agent.x_t = 120

        # Run sampling function and test results
        agent.sampling()
        self.assertAlmostEqual(agent.mu_satisficing, 150)  # expected bc 150 is starting point for sampling
        self.assertEqual(agent.n_acc, 1)  # expected bc in given this seed, agent accepts 1 sample out of 5
        self.assertEqual(agent.mu_sampling, 144.03115499744735)  # expected bc sample mean
        self.assertEqual(agent.mu_0, 150)  # expected bc agent perseverates
        self.assertEqual(agent.mu_t, 150)  # expected bc agent perseverates
        self.assertEqual(agent.frac_acc, 1 / 5)  # expected bc in given this seed, agent accepts 1 sample out of 5
        self.assertTrue(agent.r_satisficing == [0.5866643525553739])  # expected bc of this seed
        self.assertTrue(agent.samples == [150.0, 142.53894374680922, 142.53894374680922,  # expected bc
                                          142.53894374680922, 142.53894374680922])  # of this seed

        # 2. No perseveration, no push
        # ----------------------------

        agent.reinitialize_agent(seed=1)
        agent.criterion = 0.02
        agent.x_t = 120

        # Run sampling function and test results
        agent.sampling()
        self.assertAlmostEqual(agent.mu_satisficing, 144.03115499744735)  # expected bc no perseveration
        self.assertEqual(agent.n_acc, 4)  # expected bc in given this seed, agent accepts 4 samples out of 10
        self.assertEqual(agent.mu_sampling, 140.79208480682306)  # expected bc sample mean
        self.assertEqual(agent.mu_0, 144.03115499744735)  # expected bc agent perseverates
        self.assertEqual(agent.mu_t, 144.03115499744735)  # expected bc agent perseverates
        self.assertEqual(agent.frac_acc, 4 / 10)  # expected bc in given this seed, agent accepts 4 samples out of 10

        # 3. Perseveration, with push
        # ---------------------------

        agent.reinitialize_agent(seed=1)
        agent.x_t = 120
        agent.criterion = 10
        agent.sample_curr = 180

        # Run sampling function and test results
        agent.sampling()
        self.assertAlmostEqual(agent.mu_satisficing, 180)  # expected bc 180 is starting point for sampling
        self.assertEqual(agent.n_acc, 1)  # expected bc in given this seed, agent accepts 1 sample out of 5
        self.assertEqual(agent.mu_sampling, 174.03115499744735)  # expected bc sample mean
        self.assertEqual(agent.mu_0, 180)  # expected bc agent perseverates
        self.assertEqual(agent.mu_t, 180)  # expected bc agent perseverates
        self.assertEqual(agent.frac_acc, 1 / 5)  # expected bc in given this seed, agent accepts 1 sample out of 5

    def test_integration_sampling_opt(self):
        """ This function implements an integration test for the sampling function without satisficing """

        # Initialize agent based on agent_vars
        agent_vars = AgentVarsSampling()
        agent_vars.n_samples = 100
        agent_vars.model_sat = False
        agent = AlAgentSampling(agent_vars)

        # Reinitialized and set outcome
        agent.reinitialize_agent(seed=1)
        agent.x_t = 120

        # Run sampling function and test results
        agent.sampling()
        self.assertAlmostEqual(agent.mu_satisficing, 150)  # expected bc 150 is starting point for sampling
        self.assertEqual(agent.n_acc, 37)  # expected bc in given this seed, agent accepts 37 samples out of 100
        self.assertEqual(agent.mu_sampling, 141.50417272900464)  # expected bc sample mean
        self.assertEqual(agent.mu_0, 141.50417272900464)  # expected bc agent does not perseverate
        self.assertEqual(agent.mu_t, 141.50417272900464)  # expected bc agent does not perseverate
        self.assertEqual(agent.frac_acc, 37 / 100)  # expected bc in given this seed, agent accepts 37 samples out of 100


# Run unit test
if __name__ == '__main__':
    unittest.main()
