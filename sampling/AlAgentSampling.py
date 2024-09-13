""" AgentSampling: Implementation of the reduced Bayesian model as a sampling model """


import numpy as np
from scipy.stats import norm
from al_utilities import safe_div, safe_div_list
import random


class AlAgentSampling:
    """ This class definition specifies the properties of the object that implements the sampling model.
        The model infers the mean of the outcome generating distribution by MCMC sampling.
    """

    def __init__(self, agent_vars):
        # This function creates an agent object of class AgentMH based on the agent initialization input

        # RBM variables
        self.mu_0 = agent_vars.mu_0
        self.mu_t = agent_vars.mu_0
        self.sigma_0 = agent_vars.sigma_0
        self.sigma_t_sq = agent_vars.sigma_0
        self.sigma = agent_vars.sigma
        self.h = agent_vars.h
        self.tau_0 = agent_vars.tau_0
        self.tau_t = agent_vars.tau_0
        self.omega_0 = agent_vars.omega_0
        self.omega_t = agent_vars.omega_0

        # Sampling-model variables
        self.model_sat = agent_vars.model_sat
        self.n_samples = agent_vars.n_samples
        self.criterion = agent_vars.criterion
        self.burn_in = agent_vars.burn_in
        self.sample_std = agent_vars.sample_std

        # Initialize variables
        self.x_t = np.nan
        self.delta_t = np.nan
        self.tot_var = np.nan
        self.sample_curr = np.nan
        self.sample_new = np.nan
        self.mu_sampling = np.nan
        self.mu_satisficing = np.nan
        self.n_acc = np.nan
        self.frac_acc = np.nan
        self.tot_samples = np.nan
        self.samples = []
        self.r_satisficing = []

    def reinitialize_agent(self, **kwargs):
        """ This function re-initializes the sampling model """

        # Optional input for random seed
        seed = kwargs.get('seed', None)
        if seed is not None:
            random.seed(seed)

        # Empty lists
        self.samples = []
        self.r_satisficing = []

        # Initial belief
        self.mu_t = self.mu_0

        # Initial estimation uncertainty
        self.sigma_t_sq = self.sigma_0

        # Initial changepoint probability
        self.omega_t = self.omega_0

        # Initial relative uncertainty
        self.tau_t = self.tau_0

        # Current sample is equal to initial belief
        self.sample_curr = self.mu_0

        # Initialize total variance
        self.tot_var = self.sigma ** 2 + self.sigma_t_sq  # part of eq.8

    def compute_delta(self):
        """ This function computes the prediction error """

        self.delta_t = self.x_t - self.mu_t

    def compute_likelihood(self, x):
        """ This function computes the likelihood of the outcome

        :param x: Mean of likelihood
        :return: Likelihood
        """

        return norm.pdf(self.x_t, loc=x, scale=self.sigma)  # N(x_t, x, sigma^2)

    def compute_prior(self, x):
        """ This function computes the prior

        :param x: Location in which prior is evaluated
        :return: Prior
        """

        return norm.pdf(x, loc=self.mu_0, scale=np.sqrt(self.sigma_t_sq))  # N(x, mu_t, sigma_t^2)

    def compute_posterior(self, prior, lik):
        """ This function computes the posterior given the hazard rate

            The posterior is a combination of the no-changepoint case (p=1-h) and
            the changepoint case (p=h). The no-changepoint case combines the prior from the previous trial
            and the likelihood of the outcome. The changepoint case assume a flat prior and also takes
            the likelihood of the outcome into account.

        :param prior: Prior probability
        :param lik: Likelihood
        :return: Posterior probability
        """

        return np.multiply(prior, lik) * (1 - self.h) + np.multiply((1 / 300), lik) * self.h

    def compute_cpp(self, delta_t):
        """ This function computes changepoint probability based on the prediction error

            The function is currently only used for model validation in al_sampling_validation.py.

        :param delta_t: Prediction error
        :return: None
        """

        # Likelihood of prediction error given that changepoint occurred: (U(delta_t;[0,300]))^s * h
        term_1 = (1/300) * self.h  # numerator eq. 8

        # Likelihood of prediction error given that no changepoint occurred: (N(delta_t;0,sigma^2_t+sigma^2))^s * (1-h)
        term_2 = (norm.pdf(delta_t, 0, np.sqrt(self.tot_var))) * (1 - self.h)  # part of denominator of eq. 8

        # Compute changepoint probability
        self.omega_t = safe_div(term_1, (term_2 + term_1))  # eq. 8

    def compute_cpp_samples(self, samples_delta):
        """ This function computes changepoint probability based on the samples

            Taking the difference between samples and beliefs (sample-belief) is a sampling-based approximation of the
            prediction error.

        :param samples_delta: Samples - agent belief approximating the PE
        :return: None
        """

        # Likelihood of each sample given that changepoint occurred: (U(sample-belief; [0,300])) * h
        term_1 = np.repeat((1 / 300) * self.h, len(samples_delta))  # numerator eq. 8

        # Likelihood of prediction error given that no changepoint occurred:
        # (N(sample-belief; 0, sigma^2_t+sigma^2))^s * (1-h)
        term_2 = (norm.pdf(samples_delta, 0, np.sqrt(self.tot_var))) * (1 - self.h)  # part of denominator of eq. 8

        # Compute changepoint probability
        self.omega_t = np.mean(safe_div_list(term_1, (term_2 + term_1)))  # eq. 8

    def compute_eu(self):
        """ This function computes estimation uncertainty """

        # In principle, this can be combined with the RBM function. Although, this has additional parameters such as
        # uncertainty underestimation, which have to be included

        term_1 = self.omega_t * (self.sigma ** 2)
        term_2 = (1 - self.omega_t) * self.tau_t * (self.sigma ** 2)
        term_3 = self.omega_t * (1 - self.omega_t) * ((self.delta_t * (1 - self.tau_t)) ** 2)
        self.sigma_t_sq = term_1 + term_2 + term_3  # eq. 9

    def compute_ru(self):
        """ This function computes relative uncertainty """

        # In principle this can be combined with the RBM function.

        # Update relative uncertainty:
        # tau_{t+1} := sigma_{t+1}^2 / (sigma_{t+1}^2 + sigma^2)
        self.tau_t = safe_div(self.sigma_t_sq, (self.sigma_t_sq + self.sigma ** 2))  # eq. 10

    def metropolis_hastings(self):
        """ This function draws and evaluates samples according to the Metropolis-Hastings algorithm

            See for example "Machine Learning: A Probabilistic Perspective" (1st Edition) by Kevin Murphy
            24.3, page 850.
        """

        # futuretodo: for consistency w. other cases, use np.random
        # Sample from proposal distribution
        self.sample_new = random.normalvariate(self.sample_curr, self.sample_std)  # q(x_new | x_curr)

        # Evaluate target distribution (posterior)
        # ----------------------------------------

        # To obtain posterior, compute unnormalized prior probability for old and new sample...
        prior = self.compute_prior([self.sample_curr, self.sample_new])  # p^*_prior(x)
        sample_curr_prior, sample_new_prior = prior[0], prior[1]

        # ...and the unnormalized likelihood for old and new sample
        lik = self.compute_likelihood([self.sample_curr, self.sample_new])  # p^*_lik(x)
        sample_curr_lik, sample_new_lik = lik[0], lik[1]

        # Compute posterior by combining the two under consideration of the hazard rate
        posterior = self.compute_posterior([sample_curr_prior, sample_new_prior], [sample_curr_lik, sample_new_lik])

        # p^*_post(x)
        sample_curr_post, sample_new_post = posterior[0], posterior[1]

        # Accept or reject new sample
        # ---------------------------

        # Compute acceptance probability based on LR-test
        r = min(1.0, sample_new_post / sample_curr_post)

        # Sample u from U(0, 1)
        u = random.uniform(0, 1)

        # Accept or reject new sample
        if u < r:
            self.sample_curr = self.sample_new
            self.n_acc += 1

    def satisficing(self):
        """ This function implements the satisficing mechanism

            The function compares the mean given all samples (mu_sampling) and given the previous mean (mu_satisficing)
            and, based on the satisficing criterion, determines if it keeps sampling or not.
        """

        # To obtain posterior, compute unnormalized prior probability for previous and current mean...
        prior = self.compute_prior([self.mu_sampling, self.mu_satisficing])
        mu_sampling_prior, mu_satisficing_prior = prior[0], prior[1]

        # ...and the unnormalized likelihood for previous and current mean
        lik = self.compute_likelihood([self.mu_sampling, self.mu_satisficing])
        mu_sampling_lik, mu_satisficing_lik = lik[0], lik[1]

        # Compute posterior by combining the two under consideration of the hazard rate
        posterior = self.compute_posterior([mu_sampling_prior, mu_satisficing_prior],
                                           [mu_sampling_lik, mu_satisficing_lik])
        mu_sampling_post, mu_satisficing_post = posterior[0], posterior[1]

        # Compare previous and current mean by computing relative posterior probability:
        # Higher values indicate that mu_sampling_post is better than mu_statisficing_post
        r = mu_sampling_post / (mu_satisficing_post + mu_sampling_post)
        self.r_satisficing.append(r)

        # Compare lr to the case where new and old are equal (r = 1) suggesting that more sampling
        # does not sufficiently improve the estimate
        lr_diff = 0.5 - r

        # To determine when to stop sampling, compare lr_diff to satisficing criterion: If absolute difference is below
        # or equal to criterion, do satisficing. That is, we are not satisfied when new samples substantially change
        # the estimate in both directions.
        if abs(lr_diff) <= self.criterion:

            # New posterior mean is the previous mean (mu_satisficing)
            self.mu_t = self.mu_satisficing

            # ...and we stop sampling
            stop_sampling = True

        # Otherwise, keep sampling
        else:
            stop_sampling = False

        return stop_sampling

    def sampling(self):
        """ This function runs the sampling process by augmenting the Metropolis-Hastings algorithm
            with a satisficing mechanism
        """

        # Satisficing mean is initially equal to previous mean
        self.mu_satisficing = self.sample_curr

        # Initialize counters
        i = 0  # counter for total number of samples
        j = 0  # counter for number of samples per sampling round

        # Initialize number of accepted samples
        self.n_acc = 0

        # Cycle over samples
        while True:

            # Draw new sample according to Metropolis-Hastings algorithm
            self.metropolis_hastings()

            # Store effective samples and compute mean
            if i >= self.burn_in:

                # Store all samples after burn_in period in list
                self.samples.append(self.sample_curr)

                # Compute mean of samples
                self.mu_sampling = np.mean(self.samples)

            # Update counters
            i += 1  # total number of samples
            j += 1  # number of samples per round

            # Determine when to stop sampling
            # -------------------------------

            if j == self.n_samples:

                # After cycling over samples, decide whether to accept or reject sampled posterior
                if self.model_sat:

                    # Satisficing mechanism
                    stop_sampling = self.satisficing()

                    # Either stop sampling...
                    if stop_sampling:
                        break

                    # ...or, if new mean (mu_sampling) was above criterion, continue sampling
                    else:

                        # Set satisficing mean to current mean for next round
                        self.mu_satisficing = self.mu_sampling

                        # Reset counter for number of samples to keep sampling
                        j = 0

                # If no satisficing criterion is applied...
                else:

                    # ...posterior mean is equal to sampling mean when desired number of samples has been drawn
                    self.mu_t = self.mu_sampling

                    # Stop sampling when all samples have been drawn
                    break

        # Update prior mean and fraction of accepted samples
        self.mu_0 = self.mu_t
        self.tot_samples = i
        self.frac_acc = self.n_acc / self.tot_samples
