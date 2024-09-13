""" Task-Agent Interaction Sampling: Interaction between sampling model and predictive inference task """

import numpy as np
import pandas as pd
from tqdm import tqdm
from al_utilities import correct_push, safe_div


def task_agent_int_sampling(df_subj, agent, agent_vars, **kwargs):
    """ This function models the interaction between task and agent (sampling model)

    :param df_subj: Data frame with relevant data
    :param agent: Agent-object instance
    :param agent_vars: Agent-variables-object instance
    :param kwargs: Optional input arguments
    :return: df_data: Data frame with simulation results
    """

    # Optional input for number of trials
    n_trials = kwargs.get('n_trials', None)
    if n_trials is None:
        n_trials = len(df_subj)

    # Optional input for progress bar
    show_pbar = kwargs.get('show_pbar', None)

    # Optional input for random seed
    seed = kwargs.get('seed', None)
    if seed is not None:
        np.random.seed(seed)

    # Initialize relevant variables
    pers = np.zeros(n_trials)  # perseveration
    mu = np.full([n_trials], np.nan)  # inferred mean of the outcome-generating distribution
    a_hat = np.full(n_trials, np.nan)  # predicted update
    omega = np.full(n_trials, np.nan)  # changepoint probability
    tau = np.full(n_trials, np.nan)  # relative uncertainty
    alpha = np.full(n_trials, np.nan)  # learning rate
    delta = np.full(n_trials, np.nan)  # prediction error
    tot_samples = np.full(n_trials, np.nan)  # total number of samples
    x_t = np.full(n_trials, np.nan)  # outcome

    # Initialize variables related to simulations
    sim_b_t = np.full(n_trials, np.nan)  # simulated prediction
    sim_y_t = np.full(n_trials, np.nan)  # push
    sim_z_t = np.full(n_trials, np.nan)  # simulated initial bucket location

    # Reinitialize agent for each simulation
    agent.reinitialize_agent(seed=seed)

    # Initialize progress bar
    if show_pbar:
        pbar = tqdm(total=n_trials)

    # Cycle over trials
    for t in range(n_trials-1):

        # For first trial of new block
        # Futuretodo: create function to re-initialize agent on new block
        if df_subj['new_block'][t]:

            # Initialize mu, estimation uncertainty, relative uncertainty and changepoint probability
            agent.mu_t = agent_vars.mu_0
            agent.sigma_t_sq = agent_vars.sigma_0
            agent.tau_t = agent_vars.tau_0
            agent.omega_t = agent_vars.omega_0

            # Set initial bucket location, prediction, and push
            sim_z_t[t] = agent_vars.mu_0
            sim_b_t[t] = agent_vars.mu_0
            sim_y_t[t] = 0.0

        # For all other trials
        else:

            # For experiment 2, we take the actual shift in the bucket location
            sim_y_t[t] = df_subj.loc[t, 'y_t']

            # Here we check if every trial in the shifting-bucket condition is a push trial.
            # In "edge" trials (see preprocessing), push is absent because either at 0 or 300 (the edges of the
            # screen). If we have such a trial, simulate push at this stage.
            if df_subj['cond'][t] == 'main_push' and df_subj['edge'][t]:
                sim_y_t[t] = np.random.normal(0, 20, 1)

            # Ensure that push is within screen range [0, 300]
            sim_y_t[t], sim_z_t[t] = correct_push(sim_b_t[t], sim_y_t[t])

        # Set sampling starting point to initial bucket location
        agent.sample_curr = sim_z_t[t]

        # Record relative uncertainty of current trial
        tau[t] = agent.tau_t

        # For all but last trial of a block:
        if not df_subj['new_block'][t+1]:

            # Empty list on each trial - otherwise samples are being added to the list
            agent.samples = []
            agent.r_satisficing = []

            # Outcome
            agent.x_t = df_subj.loc[t, 'x_t'].copy()
            x_t[t] = df_subj.loc[t, 'x_t'].copy()

            # Prediction error
            agent.compute_delta()
            delta[t] = agent.delta_t

            # Sequential belief update through sampling
            agent.sampling()

            # Record updated belief
            mu[t] = agent.mu_t

            # Update and perseveration
            # futuretodo: add motor_pers as well
            a_hat[t] = agent.mu_t - sim_b_t[t]
            if a_hat[t] == 0.0:
                pers[t] = True

            # Approximate change-point probability based on samples
            samples_delta = np.array(agent.samples) - sim_b_t[t]
            agent.compute_cpp_samples(samples_delta)
            omega[t] = agent.omega_t

            # Compute learning rate
            alpha[t] = safe_div(a_hat[t], delta[t])

            # Compute total number of samples
            tot_samples[t] = agent.tot_samples

            # Compute estimation uncertainty
            agent.compute_eu()

            # Compute relative uncertainty
            agent.compute_ru()

            # Update total variance
            agent.tot_var = agent.sigma ** 2 + agent.sigma_t_sq  # part of eq.8

            # Updated prediction
            sim_b_t[t + 1] = agent.mu_t
            if sim_b_t[t + 1] >= 300.0:
                sim_b_t[t + 1] = 300.0
            elif sim_b_t[t + 1] <= 0.0:
                sim_b_t[t + 1] = 0.0

            # Update progress bar
            if show_pbar:
                pbar.update()

    # Attach model variables to data frame
    df_data = pd.DataFrame(index=range(0, n_trials), dtype='float')

    df_data['x_t'] = x_t
    df_data['mu_t'] = mu
    df_data['delta_t'] = delta
    df_data['omega_t'] = omega
    df_data['tau_t'] = tau
    df_data['alpha_t'] = alpha
    df_data['sim_b_t'] = sim_b_t
    df_data['sim_a_t'] = a_hat
    df_data['sim_y_t'] = sim_y_t
    df_data['sim_z_t'] = sim_z_t
    df_data['pers'] = pers
    df_data['cond'] = df_subj['cond'].copy()
    df_data['age_group'] = df_subj['age_group'].copy()
    df_data['tot_samples'] = tot_samples

    return df_data
