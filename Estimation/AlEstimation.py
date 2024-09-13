""" Estimation Class: This class estimates the reduced Bayesian model """

import numpy as np
from AlAgentRbm import AlAgent
from scipy.optimize import minimize
from time import sleep
from tqdm import tqdm
from multiprocessing import Pool
from itertools import compress
import pandas as pd
from scipy.stats import norm
import random
from al_task_agent_int_rbm import task_agent_int


class AlEstimation:
    """ This class specifies the instance variables and methods of the parameter estimation """

    def __init__(self, est_vars):
        """ This function defines the instance variables unique to each instance

                   See AlEstVars for documentation

               :param est_vars: Estimation variables object instance
        """

        # Parameter names for data frame
        self.omikron_0 = est_vars.omikron_0
        self.omikron_1 = est_vars.omikron_1
        self.b_0 = est_vars.b_0
        self.b_1 = est_vars.b_1
        self.h = est_vars.h
        self.s = est_vars.s
        self.u = est_vars.u
        self.q = est_vars.q
        self.sigma_H = est_vars.sigma_H
        self.d = est_vars.d

        # Select fixed staring points (used if not rand_sp)
        self.omikron_0_x0 = est_vars.omikron_0_x0
        self.omikron_1_x0 = est_vars.omikron_1_x0
        self.b_0_x0 = est_vars.b_0_x0
        self.b_1_x0 = est_vars.b_1_x0
        self.h_x0 = est_vars.h_x0
        self.s_x0 = est_vars.s_x0
        self.u_x0 = est_vars.u_x0
        self.q_x0 = est_vars.q_x0
        self.sigma_H_x0 = est_vars.sigma_H_x0
        self.d_x0 = est_vars.d_x0

        # Select range of random starting point values (if rand_sp)
        self.omikron_0_x0_range = est_vars.omikron_0_x0_range
        self.omikron_1_x0_range = est_vars.omikron_1_x0_range
        self.b_0_x0_range = est_vars.b_0_x0_range
        self.b_1_x0_range = est_vars.b_1_x0_range
        self.h_x0_range = est_vars.h_x0_range
        self.s_x0_range = est_vars.s_x0_range
        self.u_x0_range = est_vars.u_x0_range
        self.q_x0_range = est_vars.q_x0_range
        self.sigma_H_x0_range = est_vars.sigma_H_x0_range
        self.d_x0_range = est_vars.d_x0_range

        # Select boundaries for estimation
        self.omikron_0_bnds = est_vars.omikron_0_bnds
        self.omikron_1_bnds = est_vars.omikron_1_bnds
        self.b_0_bnds = est_vars.b_0_bnds
        self.b_1_bnds = est_vars.b_1_bnds
        self.h_bnds = est_vars.h_bnds
        self.s_bnds = est_vars.s_bnds
        self.u_bnds = est_vars.u_bnds
        self.q_bnds = est_vars.q_bnds
        self.sigma_H_bnds = est_vars.sigma_H_bnds
        self.d_bnds = est_vars.d_bnds

        # Free parameter indexes
        self.which_vars = est_vars.which_vars

        # Fixed parameter values
        self.fixed_mod_coeffs = est_vars.fixed_mod_coeffs

        # Other attributes
        self.n_subj = est_vars.n_subj
        self.n_ker = est_vars.n_ker
        self.which_exp = est_vars.which_exp
        self.rand_sp = est_vars.rand_sp
        self.n_sp = est_vars.n_sp
        self.use_prior = est_vars.use_prior

    def parallel_estimation(self, df, agent_vars):
        """ This function manages the parallelization of the model estimation

        :param df: Data frame containing the data
        :param agent_vars: Agent variables object instance
        :return: results_df: Data frame containing regression results
        """

        # Inform user
        sleep(0.1)
        print('\nModel estimation:')
        sleep(0.1)

        # Initialize progress bar
        pbar = tqdm(total=self.n_subj)

        # Function for progress bar update
        def callback(x):
            """ This function updates the progress bar """

            pbar.update(1)

        # Initialize pool object for parallel processing
        pool = Pool(processes=self.n_ker)

        # Parallel parameter estimation
        results = [pool.apply_async(self.model_estimation,
                                    args=(df[(df['subj_num'] == i + 1)].copy(), agent_vars),
                                    callback=callback) for i in range(0, self.n_subj)]
        output = [p.get() for p in results]
        pool.close()
        pool.join()

        # Select parameters according to selected variables and create data frame
        prior_columns = [self.omikron_0, self.omikron_1, self.b_0, self.b_1, self.h, self.s, self.u, self.q,
                         self.sigma_H, self.d]

        # Add estimation results to data frame output
        columns = list(compress(prior_columns, self.which_vars.values()))
        columns.append('llh')
        columns.append('BIC')
        columns.append('age_group')
        columns.append('subj_num')
        results_df = pd.DataFrame(output, columns=columns)

        # Make sure that we keep the same order of participants
        results_df = results_df.sort_values(by=['subj_num'])

        # Close progress bar
        pbar.close()

        return results_df

    def model_estimation(self, df_subj, agent_vars):
        """ This function estimates the free parameters of the model

        :param df_subj: Data frame with data of current participants
        :param agent_vars: Agent variables object instance
        :return: results_list: List containing estimates, llh, bic and age group
        """

        # Control random number generator for reproducible results
        random.seed(123)

        # Extract age group and subject number for output
        age_group = list(set(df_subj['age_group']))
        age_group = float(age_group[0])
        subj_num = list(set(df_subj['subj_num']))
        subj_num = float(subj_num[0])

        # Extract free parameters
        values = self.which_vars.values()

        # Select starting points and boundaries
        # -------------------------------------

        bnds = [self.omikron_0_bnds, self.omikron_1_bnds, self.b_0_bnds, self.b_1_bnds, self.h_bnds, self.s_bnds,
                self.u_bnds, self.q_bnds, self.sigma_H_bnds, self.d_bnds]

        # Select boundaries according to selected free parameters
        bnds = np.array(list(compress(bnds, values)))

        # Initialize with unrealistically high likelihood
        min_llh = 100000  # futuretodo: set to inf?
        min_x = np.nan

        # Cycle over starting points
        for r in range(0, self.n_sp):

            if self.rand_sp:

                # Draw starting points from uniform distribution
                x0 = [random.uniform(self.omikron_0_x0_range[0], self.omikron_0_x0_range[1]),
                      random.uniform(self.omikron_1_x0_range[0], self.omikron_1_x0_range[1]),
                      random.uniform(self.b_0_x0_range[0], self.b_0_x0_range[1]),
                      random.uniform(self.b_1_x0_range[0], self.b_1_x0_range[1]),
                      random.uniform(self.h_x0_range[0], self.h_x0_range[1]),
                      random.uniform(self.s_x0_range[0], self.s_x0_range[1]),
                      random.uniform(self.u_x0_range[0], self.u_x0_range[1]),
                      random.uniform(self.q_x0_range[0], self.q_x0_range[1]),
                      random.uniform(self.sigma_H_x0_range[0], self.sigma_H_x0_range[1]),
                      random.uniform(self.d_x0_range[0], self.d_x0_range[1])]
            else:

                # Use fixed starting points
                x0 = [self.omikron_0_x0,
                      self.omikron_1_x0,
                      self.b_0_x0,
                      self.b_1_x0,
                      self.h_x0,
                      self.s_x0,
                      self.u_x0,
                      self.q_x0,
                      self.sigma_H_x0,
                      self.d_x0]

            # Select starting points according to free parameters
            x0 = np.array(list(compress(x0, values)))

            # Estimate parameters
            res = minimize(self.llh, x0, args=(df_subj, agent_vars), method='L-BFGS-B', bounds=bnds,
                           options={'disp': False})  # options={'disp': False}

            # Extract minimized log likelihood
            f_llh_max = res.fun

            # Check if negative log-likelihood is lower than the previous one and select the lowest
            if f_llh_max < min_llh:
                min_llh = f_llh_max
                min_x = res.x

        # Compute BIC
        if self.which_exp == 1:
            bic = self.compute_bic(min_llh, sum(self.which_vars.values()), len(df_subj)-2)
        else:
            bic = self.compute_bic(min_llh, sum(self.which_vars.values()), len(df_subj) - 4)

        # Save data to list of results
        min_x = min_x.tolist()
        results_list = list()
        results_list = results_list + min_x
        results_list.append(float(min_llh))
        results_list.append(float(bic))
        results_list.append(float(age_group))
        results_list.append(float(subj_num))

        return results_list

    def llh(self, coeffs, df, agent_vars):
        """ This function computes the cumulated negative log likelihood of the data under the model

        :param coeffs: Free parameters
        :param df: Data frame of current subject
        :param agent_vars: Agent variables object
        :return: llh_sum: Cumulated negative log-likelihood
        """

        # Get fixed parameters
        fixed_coeffs = self.fixed_mod_coeffs

        # Initialize parameter list and counters
        sel_coeffs = []
        i = 0

        # Put selected coefficients in list that is used for model estimation
        #   futuretodo: create function for this
        for key, value in self.which_vars.items():
            if value:
                sel_coeffs.append(coeffs[i])
                i += 1
            else:
                sel_coeffs.append(fixed_coeffs[key])

        # Adjust data frame indices for estimation
        x = np.linspace(0, len(df) - 1, len(df))
        df.loc[:, 'trial'] = x.tolist()
        df = df.set_index('trial')

        # Reduced Bayesian model variables
        # potentialtodo: function bc this is used repeatedly and can then also be tested
        agent_vars.h = sel_coeffs[4]
        agent_vars.s = sel_coeffs[5]
        agent_vars.u = np.exp(sel_coeffs[6])
        agent_vars.q = sel_coeffs[7]
        agent_vars.sigma_H = sel_coeffs[8]

        # Call AlAgent object
        agent = AlAgent(agent_vars)

        # Estimate parameters
        llh_mix, _ = task_agent_int(self.which_exp, df, agent, agent_vars, sel_coeffs)

        # Consider prior over uncertainty underestimation coefficient
        if self.use_prior:
            u_prob = np.log(norm.pdf(sel_coeffs[6], 0, 5))
        else:
            u_prob = 0

        # Compute cumulated negative log-likelihood
        llh_sum = -1 * (np.sum(llh_mix) + u_prob)

        return llh_sum

    @staticmethod
    def compute_bic(llh, n_params, n_trials):
        """ This function computes the Bayesian information criterion (BIC)

            See Stephan et al. (2009). Bayesian model selection for group studies. NeuroImage

        :param llh: Negative log-likelihood
        :param n_params: Number of free parameters
        :param n_trials: Number of trials
        :return: bic
        """

        return (-1 * llh) - (n_params / 2) * np.log(n_trials)  # eq. 24
