import numpy as np


class AlEstVars:
    # This function defines the instance variables unique to each instance

    def __init__(self):
        # This function determines the default estimation variables

        # Parameter names for data frame
        self.omikron_0 = 'omikron_0'  # motor noise
        self.omikron_1 = 'omikron_1'  # learning-rate noise
        self.b_0 = 'b_0'  # logistic-function intercept
        self.b_1 = 'b_1'  # logistic-function slope
        self.h = 'h'  # hazard rate
        self.s = 's'  # surprise sensitivity
        self.u = 'u'  # uncertainty underestimation
        self.q = 'q'  # reward bias
        self.sigma_H = 'sigma_H'  # catch-trial
        self.d = 'd'  # bucket bias

        # Select staring points (used if rand_sp = False)
        self.omikron_0_x0 = 5.0
        self.omikron_1_x0 = 0.0
        self.b_0_x0 = 0.5
        self.b_1_x0 = 0
        self.h_x0 = 0.1
        self.s_x0 = 0.999
        self.u_x0 = 0.0
        self.q_x0 = 0.0
        self.sigma_H_x0 = 10.0
        self.d_x0 = 0.0

        # Select range of random starting point values (used if rand_sp = True)
        self.omikron_0_x0_range = (1, 10)
        self.omikron_1_x0_range = (0.001, 1)
        self.b_0_x0_range = (-30, 30)
        self.b_1_x0_range = (-1.5, 1)
        self.h_x0_range = (0.001, 0.99)
        self.s_x0_range = (0.001, 0.99)
        self.u_x0_range = (1, 10)
        self.q_x0_range = (0, 0.1)
        self.sigma_H_x0_range = (1, 32)
        self.d_x0_range = (-1, 1)

        # Select boundaries for estimation
        self.omikron_0_bnds = (0.1, 10)
        self.omikron_1_bnds = (0.001, 1)
        self.b_0_bnds = (-30, 30)
        self.b_1_bnds = (-1.5, 1)
        self.h_bnds = (0.001, 0.99)
        self.s_bnds = (0.001, 1)
        self.u_bnds = (-2, 15)
        self.q_bnds = (-0.5, 0.5)
        self.sigma_H_bnds = (0, 32)
        self.d_bnds = (-1, 1)

        # Free parameters
        self.which_vars = {self.omikron_0: True,
                           self.omikron_1: True,
                           self.b_0: True,
                           self.b_1: True,
                           self.h: True,
                           self.s: True,
                           self.u: True,
                           self.q: True,
                           self.sigma_H: True,
                           self.d: False,
                           }

        # Fixed values for fixed parameters
        self.fixed_mod_coeffs = {self.omikron_0: 10.0,
                                 self.omikron_1: 0.0,
                                 self.b_0: -30,
                                 self.b_1: -1.5,
                                 self.h: 0.1,
                                 self.s: 1.0,
                                 self.u: 0.0,
                                 self.q: 0.0,
                                 self.sigma_H: 0.0,
                                 self.d: 0.0}

        # Other attributes
        self.n_subj = np.nan  # number of participants
        self.n_ker = 4  # number of kernels for estimation
        self.which_exp = 1  # current experiment
        self.rand_sp = True  # use of random starting points during estimation
        self.n_sp = 25  # number of starting points
