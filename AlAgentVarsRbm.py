""" AgentVars: Initialization of the reduced Bayesian model """


class AgentVars:
    """ This class specifies the AgentVars object for the reduced optimal model

        futuretodo: Consider using a data class here
    """

    def __init__(self):
        # This function determines the default agent variables

        self.s = 1  # surprise sensitivity
        self.h = 0.1  # hazard rate
        self.u = 0.0  # uncertainty underestimation
        self.q = 0  # reward bias
        self.sigma = 10  # noise in the environment (standard deviation)
        self.sigma_0 = 100  # initial variance of predictive distribution
        self.sigma_H = 1  # catch-trial standard deviation (uncertainty) of helicopter cue
        self.tau_0 = 0.5  # initial relative uncertainty
        self.omega_0 = 1  # initial change-point probability
        self.mu_0 = 150  # initial belief about mean
