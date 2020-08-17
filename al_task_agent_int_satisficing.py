import numpy as np
import pandas as pd
from scipy.stats import norm


def task_agent_int_satisficing(df, agent, agent_vars, sel_coeffs):
    """ This function models the interaction between task and default-belief model

    This function can not be used for model estimation, only simulations. In the future,
    the function may be extended or combined with task_agent_int or the reduced Bayesian model.
    Both functions currently share many lines of code, which will be fixed in future versions. However, the advantage
    of separate functions at this point is that the reduced Bayesian model and the default-belief model can
    easily be separately applied.

    :param df: data frame with relevant data
    :param agent: agent object instance
    :param agent_vars: agent variables object instance
    :param sel_coeffs: free parameters
    :return: df_data: data frame with interaction variables
    """

    # Extract and initialize relevant variables
    n_trials = len(df)  # number of trials
    delta = np.full(len(df), np.nan)  # prediction error
    high_val = df['r_t'] == 1  # indicates high value trials
    mu = np.full([n_trials], np.nan)  # inferred mean of the outcome-generating distribution
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

        # Extract noise condition
        agent.sigma = df['sigma'][t]

        # For first trial of new block
        if df['new_block'][t]:

            # Initialize estimation uncertainty, relative uncertainty and changepoint probability
            agent.sigma_t_sq = agent_vars.sigma_0
            agent.tau_t = agent_vars.tau_0
            agent.omega_t = agent_vars.omega_0

            # Set initial prediction to 150
            sim_b_t[t] = 150.0

            # Set initial bucket location z_t to 150 and and shift to 0
            sim_y_t[t] = 0.0
            sim_z_t[t] = 150.0

        # For all other trials
        else:

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
        delta[t] = df['x_t'][t] - sim_b_t[t]
        agent.learn(delta[t], sim_b_t[t], df['v_t'][t], df['mu_t'][t], high_val[t])

        # Record updated belief
        mu[t] = agent.mu_t

        # Record predicted update according to reduced Bayesian model
        a_hat[t] = agent.a_t

        # Record changepoint probability
        omega[t] = agent.omega_t

        # Record learning rate
        alpha[t] = agent.alpha_t

        # Compute response noise
        epsilon[t] = 5  # todo: parameterize epsilon

        # Default-belief model additions
        # ------------------------------

        # Compute spatial movement that is required to reach the optimal update
        m_t = - sim_y_t[t] + a_hat[t]

        # Compute the satisficing threshold
        if m_t >= 0:
            w_t = norm.ppf(0.5 - sel_coeffs[-1], loc=m_t, scale=20)
        else:
            w_t = norm.ppf(0.5 + sel_coeffs[-1], loc=m_t, scale=20)

        # Compute reported belief update
        if np.sign(m_t) == np.sign(w_t):
            sim_a_t[t] = np.random.normal(sim_y_t[t] + w_t, epsilon[t], 1)
        else:
            sim_a_t[t] = sim_y_t[t]

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

    return df_data
