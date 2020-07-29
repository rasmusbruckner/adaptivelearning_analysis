import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import logsumexp
from al_utilities import compute_persprob


def task_agent_int(which_exp, df, agent, agent_vars, sel_coeffs, sim=False):
    """ This function models the interaction between task and agent

    :param which_exp: Current experiment: 1: first experiment; 2: follow-up experiment
    :param df: data frame with relevant data
    :param agent: agent object instance
    :param agent_vars: agent variables object instance
    :param sel_coeffs: free parameters
    :param sim: Indicates if function is currently used for simulations or not
    :return: llh_mix, df_data: negative log-likelihoods of mixture model and data frame with interaction variables
    """

    # Extract and initialize relevant variables
    n_trials = len(df)  # number of trials
    if not sim:
        delta = df['delta_t']
    else:
        delta = np.full(len(df), np.nan)

    pers = df['a_t'] == 0  # indicates perseveration trials
    high_val = df['r_t'] == 1  # indicates high value trials
    mu = np.full([n_trials], np.nan)  # inferred mean of the outcome-generating distribution
    mu_bias = np.full([n_trials], np.nan)  # inferred mean with bucket bias
    llh_rbm = np.full([n_trials-1], np.nan)  # log-likelihood of reduced Bayesian model
    llh_mix = np.full([n_trials-1], np.nan)  # log-likelihood of mixture model
    a_hat = np.full(n_trials, np.nan)  # predicted update according to reduced Bayesian model
    epsilon = np.full(n_trials, np.nan)  # response noise
    omega = np.full(n_trials, np.nan)  # changepoint probability
    tau = np.full(n_trials, np.nan)  # relative uncertainty
    alpha = np.full(n_trials, np.nan)  # learning rate

    # Initialize variables related to simulations
    sim_b_t = np.full(n_trials, np.nan)  # simulated prediction
    sim_z_t = np.full(n_trials, np.nan)  # simulated initial bucket location
    sim_y_t = np.full(n_trials, np.nan)  # simulated shift of the bucket
    sim_a_t = np.full(n_trials, np.nan)  # simulated update

    # Cycle over trials (last trial not considered because update would concern trial T+1, e.g., 401)
    # ------------------------------------------------------------------------------------------------

    for t in range(0, n_trials-1):

        # For experiment 2, indicate current block
        if which_exp == 2:
            no_push_cond = df['cond'][t] == 'main_noPush'

        # Extract noise condition
        agent.sigma = df['sigma'][t]

        # For first trial of new block
        if df['new_block'][t]:

            # Initialize estimation uncertainty, relative uncertainty and changepoint probability
            agent.sigma_t_sq = agent_vars.sigma_0
            agent.tau_t = agent_vars.tau_0
            agent.omega_t = agent_vars.omega_0

            if sim:

                # Set initial prediction to 150
                sim_b_t[t] = 150.0

                # Set initial bucket location z_t to 150 and and shift to 0
                sim_y_t[t] = 0.0
                sim_z_t[t] = 150.0

        # For all other trials
        else:
            if sim and which_exp == 2:

                # For simulations, we take the actual shift in the bucket location
                sim_y_t[t] = df['y_t'][t]

                # Here we check if every trial in the shifting-bucket condition is a push trial.
                # In "edge" trials (see preprocessing), push is absent because either at 0 or 300 (the edges of the
                # screen). If we have such a trial, simulate push at this stage.
                if df['cond'][t] == 'main_push' and df['edge'][t]:
                    sim_y_t[t] = np.random.normal(0, 20, 1)

                # Last trial of block is nan, therefore set to 0 to in these simulations
                if np.isnan(sim_y_t[t]):
                    sim_y_t[t] = 0.0

                # For absolute position z_t, we compute the difference
                # between shift (y_t := z_t - b_{t-1}) and b_{t-1}: z_t = b_{t-1} + y_t
                sim_z_t[t] = sim_b_t[t] + sim_y_t[t]

                # Adjust for edges of the screen. This is necessary because the model makes different trial-by-trial
                # predictions than participants, where we corrected for this already during preprocessing
                if sim_z_t[t] > 300:
                    sim_z_t[t] = 300
                elif sim_z_t[t] < 0:
                    sim_z_t[t] = 0

        # Record relative uncertainty of current trial
        tau[t] = agent.tau_t

        # Sequential belief update
        if sim:
            delta[t] = df['x_t'][t] - sim_b_t[t]
            agent.learn(delta[t], sim_b_t[t], df['v_t'][t], df['mu_t'][t], high_val[t])
        else:
            agent.learn(delta[t], df['b_t'][t], df['v_t'][t], df['mu_t'][t], high_val[t])

        # Record updated belief
        mu[t] = agent.mu_t

        # Record predicted update according to reduced Bayesian model
        a_hat[t] = agent.a_t

        # Record changepoint probability
        omega[t] = agent.omega_t

        # Record learning rate
        alpha[t] = agent.alpha_t

        # Model bucket bias:  hat{a}_t = hat{a}_t + d * y_t
        if which_exp == 2:

            if sim:
                if df['new_block'][t]:
                    # At the beginning of a block,
                    # we assume that last prediction (from previous block has no influence on the update)
                    bucket_bias = 0.0
                else:
                    bucket_bias = sel_coeffs[9] * sim_y_t[t]
            else:
                if df['new_block'][t]:
                    # At the beginning of a block,
                    # we assume that last prediction (from previous block has no influence on the update)
                    bucket_bias = 0.0
                else:
                    bucket_bias = sel_coeffs[9] * df['y_t'][t]

            # Add bucket bias to prediction update
            a_hat[t] = a_hat[t] + bucket_bias

            # Record updated belief with bucket bias
            mu_bias[t] = agent.mu_t + bucket_bias

        # Compute likelihood for updates according to reduced Bayesian model
        # ------------------------------------------------------------------

        # Compute absolute predicted update
        # |hat{a}_t|
        abs_pred_up = abs(a_hat[t])

        # Compute response noise
        # epsilon := omikron_0 + omikron_1 * |hat{t}_t|
        epsilon[t] = sel_coeffs[0] + sel_coeffs[1] * abs_pred_up

        # Compute likelihood of predicted update
        # p(a_t) := N(a_t; hat{a}_t, epsilon_t^2)
        p_a_t = norm.pdf(x=df['a_t'][t], loc=a_hat[t], scale=epsilon[t])

        # Adjust probability of update for numerical stability
        if p_a_t == 0.0 or np.isinf(p_a_t) or np.isnan(p_a_t):
            p_a_t = 1.e-5

        # Compute negative log-likelihood of predicted update according to reduced Bayesian model
        llh_rbm[t] = np.log(p_a_t)

        # In the first experiment, we model perseveration for all trials
        # In the follow-up experiment, we model perseveration only in the no_push condition
        if which_exp == 1:
            lambda_t = compute_persprob(sel_coeffs[2], sel_coeffs[3], abs_pred_up)
        else:
            if no_push_cond:
                lambda_t = compute_persprob(sel_coeffs[2], sel_coeffs[3], abs_pred_up)
            else:
                lambda_t = 0.0

        # Adjust lambda_t for numerical stability
        if lambda_t == 0.0 or np.isnan(lambda_t):
            lambda_t = 1.e-5
        elif lambda_t == 1:
            lambda_t = 1 - 1.e-5

        # Adjust delta function for numerical stability during computation of likelihood
        # delta(a_t) := 1 if a_t = 0, 0 else
        if pers[t] == 1:
            delta_fun = 1 - 1.e-5
        elif pers[t] == 0:
            delta_fun = 1.e-5

        # Compute negative log-likelihood of mixture based on
        # p^{lambda_t,delta}(a_t|x_{1:T}):= lambda_t * delta(a_t) + (1-lambda_t)p(a_t|x_{1:T})
        llh_mix[t] = logsumexp([np.log(delta_fun) + np.log(lambda_t), np.log((1 - lambda_t)) + llh_rbm[t]])

        # Simulate perseveration and updates
        if sim:

            # Randomly sample perseveration trials
            rand_pers = np.random.binomial(1, lambda_t)
            if rand_pers == 0:
                sim_a_t[t] = np.random.normal(a_hat[t], epsilon[t], 1)
            else:
                sim_a_t[t] = 0.0

            # Updated prediction
            sim_b_t[t+1] = sim_b_t[t] + sim_a_t[t]
            if sim_b_t[t+1] >= 300.0:
                sim_b_t[t+1] = 300.0
            elif sim_b_t[t+1] <= 0.0:
                sim_b_t[t+1] = 0.0

    # Attach model variables to data frame
    df_data = pd.DataFrame(index=range(0, n_trials), dtype='float')
    df_data['a_t_hat'] = a_hat
    df_data['mu_t'] = mu
    df_data['mu_t_bias'] = mu_bias
    df_data['delta_t'] = delta
    df_data['omega_t'] = omega
    df_data['tau_t'] = tau
    df_data['alpha_t'] = alpha

    if sim:

        # Save simulation-related variables
        df_data['sim_b_t'] = sim_b_t
        df_data['sim_a_t'] = sim_a_t
        df_data['sim_y_t'] = sim_y_t
        df_data['sim_z_t'] = sim_z_t
        df_data['sigma'] = df['sigma']
        if which_exp == 2:
            df_data['cond'] = df['cond']

    return llh_mix, df_data
