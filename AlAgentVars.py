import numpy as np


class AgentVars:
    # This class specifies the AgentVars object for the reduced optimal model

    def __init__(self):
        # This function determines the default agent variables

        self.s = 1  # ss
        self.h = 0.1  # hazard rate
        self.u = 0.0  # uncertainty underestimation
        self.q = 0  # reward bias
        self.sigma_H = 1  # catch-trial variability of helicopter cue
        self.sigma = 10  # noise in the environment (standard deviation)
        self.sigma_0 = 100  # initial variance of predictive distribution
        self.sigma_t_sq = np.nan  # variance of predictive distribution
        self.tau_0 = 0.5  # initial relative uncertainty
        self.tau_t = np.nan  # initialization of relative uncertainty for trials >= 1
        self.omega_0 = 1  # initial change point probability
        self.omega_t = np.nan  # initialization of changepoint probability for trials >= 1
        self.mu_0 = 150  # initial belief about mean
        self.mu_t = np.nan  # initialization of mu for trials >= 1
        self.C = np.nan  # initial C variable for catch-trial computations
