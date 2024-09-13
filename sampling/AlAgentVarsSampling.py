""" AgentVarsSampling: Agent variables of the sampling model """


class AgentVarsSampling:
    """ This class specifies the agent-variables object for the sampling model

        Will change this to data class in the future:
        https://docs.python.org/3/library/dataclasses.html
    """

    def __init__(self):
        # This function determines the default sampling-agent variables

        # RBM variables
        self.mu_0 = 150.0  # initial belief about mean
        self.sigma_0 = 100  # initial variance of predictive distribution
        self.sigma = 17.5  # noise in the environment (standard deviation)
        self.h = 0.1  # hazard rate
        self.tau_0 = 0.5  # initial relative uncertainty
        self.omega_0 = 1  # initial change-point probability

        # Sampling-model variables
        self.model_sat = True  # satisficing on or off
        self.n_samples = 30  # number of samples per chunk (n)
        self.criterion = 1.0  # satisficing criterion (c)
        self.burn_in = 0  # number of burn-in samples
        self.sample_std = 30  # standard deviation of samples from proposal distribution. Chosen such that the
        # acceptance rate is between 25% and 40%, which is a good range according to the Murphy book (sigma_mh)
