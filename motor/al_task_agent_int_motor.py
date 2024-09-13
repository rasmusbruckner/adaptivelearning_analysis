""" Task-Agent Interaction Motor Model: Interaction between motor model and predictive inference task """

import numpy as np
import pandas as pd
from al_utilities import compute_persprob, correct_push, trial_cost_func, find_nearest


def task_agent_int_motor(df, agent, agent_vars, sel_coeffs):
    """ This function models the interaction between task and agent (motor model)

    :param df: Data frame with relevant data
    :param agent: Agent-object instance
    :param agent_vars: Agent-variables-object instance
    :param sel_coeffs: Free parameters
    :return: df_data: Data frame with simulation results
    """

    # Extract and initialize relevant variables
    # -----------------------------------------
    n_trials = len(df)  # number of trials
    high_val = df['r_t'] == 1  # indicates high value trials
    mu = np.full([n_trials], np.nan)  # inferred mean of the outcome-generating distribution
    a_hat = np.full(n_trials, np.nan)  # predicted update according to reduced Bayesian model
    a_hat_movement = np.full(n_trials, np.nan)  # predicted update according to reduced Bayesian model
    omega = np.full(n_trials, np.nan)  # changepoint probability
    tau = np.full(n_trials, np.nan)  # relative uncertainty
    alpha = np.full(n_trials, np.nan)  # learning rate
    delta = np.full(len(df), np.nan)  # prediction error

    # Initialize variables related to simulations
    sim_b_t = np.full(n_trials, np.nan)  # simulated prediction
    sim_z_t = np.full(n_trials, np.nan)  # simulated initial bucket location
    sim_y_t = np.full(n_trials, np.nan)  # simulated shift of the bucket
    sim_a_t = np.full(n_trials, np.nan)  # simulated update
    grid = np.linspace(0, 300, 301)  # grid for cost computations

    # Cycle over trials
    # -----------------

    for t in range(0, n_trials-1):

        # Extract current push condition
        no_push_cond = df['cond'][t] == 'main_noPush'

        # Extract noise condition
        agent.sigma = df['sigma'][t]

        # For first trial of new block
        # Futuretodo: create function to re-initialize agent on new block
        if df['new_block'][t]:

            # Initialize estimation uncertainty, relative uncertainty and changepoint probability
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
            sim_y_t[t] = df['y_t'][t]

            # Here we check if every trial in the shifting-bucket condition is a push trial.
            # In "edge" trials (see preprocessing), push is absent because either at 0 or 300 (the edges of the
            # screen). If we have such a trial, simulate push at this stage.
            if df['cond'][t] == 'main_push' and df['edge'][t]:
                sim_y_t[t] = np.random.normal(0, 20, 1)  # futuretodo: parameterize

            # Ensure that push is within screen range [0, 300]
            sim_y_t[t], sim_z_t[t] = correct_push(sim_b_t[t], sim_y_t[t])

        # Record relative uncertainty of current trial
        tau[t] = agent.tau_t

        # For all but last trial of a block:
        if not df['new_block'][t+1]:

            # Sequential belief update
            delta[t] = df['x_t'][t] - sim_b_t[t]
            agent.learn(delta[t], sim_b_t[t], df['v_t'][t], df['mu_t'][t], high_val[t])

            # Record updated belief
            mu[t] = agent.mu_t

            # Record change-point probability
            omega[t] = agent.omega_t

            # Record learning rate
            alpha[t] = agent.alpha_t

            # Compute distance to starting location and optimal belief
            dist_z_t = abs(grid - sim_z_t[t])
            dist_mu_t = abs(grid - mu[t])

            # Compute costs
            error_cost = trial_cost_func(dist_mu_t, 0.5, 1.1)  # underadjustment costs (fixed)
            motor_cost = trial_cost_func(dist_z_t, sel_coeffs[8], sel_coeffs[9])  # motor costs (variable)
            total_costs = error_cost + motor_cost  # total costs

            # Take belief that minimizes costs
            sim_b_t[t+1], _ = find_nearest(grid[total_costs == np.min(total_costs)], sim_z_t[t])

            # Compute updates
            a_hat[t] = sim_b_t[t+1] - sim_b_t[t]  # belief update
            a_hat_movement[t] = sim_b_t[t+1] - sim_z_t[t]  # movement on screen used for perseveration

            # Compute absolute predicted movement on screen for perseveration
            abs_pred_movement = abs(a_hat_movement[t])

            # Compute perseveration probability
            lambda_t = compute_persprob(sel_coeffs[2], sel_coeffs[3], abs_pred_movement)

            # Randomly sample perseveration trials
            rand_pers = np.random.binomial(1, lambda_t)

            # Compute update and belief
            if rand_pers == 0:

                # If no perseveration, belief and belief update follow directly from cost-benefit computation
                sim_a_t[t] = a_hat[t]

            elif rand_pers and no_push_cond:

                # If perseveration and no-push condition, update is zero and belief is equal to starting location
                # (in this case previous belief)
                sim_a_t[t] = 0.0
                sim_b_t[t+1] = sim_z_t[t]

            elif rand_pers and not no_push_cond:

                # If perseveration and push condition, update is equal to bucket push (motor perseveration) and
                # belief is equal to starting location
                sim_a_t[t] = sim_y_t[t]
                sim_b_t[t+1] = sim_z_t[t]

            # Updated prediction
            if sim_b_t[t+1] >= 300.0:
                sim_b_t[t+1] = 300.0
            elif sim_b_t[t+1] <= 0.0:
                sim_b_t[t+1] = 0.0

    # Attach model variables to data frame
    df_data = pd.DataFrame(index=range(0, n_trials), dtype='float')
    df_data['a_t_hat'] = a_hat
    df_data['mu_t'] = mu
    df_data['delta_t'] = delta
    df_data['omega_t'] = omega
    df_data['tau_t'] = tau
    df_data['alpha_t'] = alpha

    # Save simulation-related variables
    df_data['sim_b_t'] = sim_b_t
    df_data['sim_a_t'] = sim_a_t
    df_data['sim_y_t'] = sim_y_t
    df_data['sim_z_t'] = sim_z_t
    df_data['sigma'] = df['sigma']
    df_data['cond'] = df['cond']
    df_data['age_group'] = df['age_group']

    return df_data
