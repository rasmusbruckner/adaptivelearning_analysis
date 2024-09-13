""" Simulations Reduced Bayesian Model: Run simulations across whole data set, e.g., for posterior predictive checks """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep
from tqdm import tqdm
from AlAgentRbm import AlAgent
from AlAgentVarsRbm import AgentVars
from al_task_agent_int_rbm import task_agent_int
from al_utilities import get_df_subj, get_sim_est_err


def simulation(df_exp, df_model, n_subj, sim_pers, plot_data=False, which_exp=1, sim_bucket_bias=True, sim=True):
    """ This function simulates data using the RBM mixture model

    :param df_exp: Data frame containing participant data
    :param df_model: Data frame containing model parameters
    :param n_subj: Number of participants
    :param sim_pers: Indicates if perseveration should be simulated or not
    :param plot_data: Indicates if single-trial plots for updates and predictions should be generated
    :param which_exp: Current experiment
    :param sim_bucket_bias: Indicates if bucket bias should be simulated or not
    :param sim: Indicates if prediction errors are simulated or not
    :return: sim_est_err, sim_pers_prob, df_sim, true_params: Simulated estimation errors,
             simulated perseveration probability, set of true parameters
    """

    # Inform user
    sleep(0.1)
    print('\nModel simulation:')
    sleep(0.1)

    # Initialize progress bar
    pbar = tqdm(total=n_subj)

    # Agent-variables object
    agent_vars = AgentVars()

    if which_exp == 3:
        agent_vars.mu_0 = 50

    # Initialize data frame for data that will be recovered
    df_sim = pd.DataFrame()

    # Initialize group vector
    group = np.full(n_subj, np.nan)

    # Initialize data frames for estimation errors and perseveration
    if which_exp == 1 or which_exp == 3:
        sim_est_err = pd.DataFrame(columns=['main', 'age_group'], index=np.arange(n_subj), dtype=float)
        sim_pers_prob = pd.DataFrame(columns=['main', 'age_group'], index=np.arange(n_subj), dtype=float)
    else:
        sim_est_err = pd.DataFrame(columns=['noPush', 'push', 'age_group'], index=np.arange(n_subj), dtype=float)
        sim_pers_prob = pd.DataFrame(columns=['noPush', 'push', 'age_group'], index=np.arange(n_subj), dtype=float)

    # Cycle over participants
    # -----------------------
    for i in range(0, n_subj):

        # Extract subject-specific data frame
        df_subj = get_df_subj(df_exp, i)

        # Extract model parameters from model data frame
        sel_coeffs = df_model[df_model['subj_num'] == i + 1].copy()

        # Extract age group of current participant
        group[i] = sel_coeffs[['age_group']].values

        # Save parameters for parameter-recovery analysis
        if i == 0:
            true_params = sel_coeffs
        elif i > 0:
            true_params = pd.concat([true_params, sel_coeffs], ignore_index=True)

        # Select relevant variables from parameter data frame
        if which_exp == 1 or which_exp == 3:
            sel_coeffs = sel_coeffs[['omikron_0', 'omikron_1', 'b_0', 'b_1', 'h', 's',
                                     'u', 'q', 'sigma_H']].values.tolist()[0]
        else:
            sel_coeffs = sel_coeffs[['omikron_0', 'omikron_1', 'b_0', 'b_1', 'h', 's',
                                     'u', 'q', 'sigma_H', 'd']].values.tolist()[0]

        # Adjust coefficients for "no-perseveration" and "no-bucket-bias" simulations
        if not sim_pers:
            sel_coeffs[2] = -30
            sel_coeffs[3] = -1.5
        elif which_exp == 2 and not sim_bucket_bias:
            sel_coeffs[9] = 0.0

        # Set agent variables of current participant
        # futuretodo: in function bc also used in other cases
        agent_vars.h = sel_coeffs[4]
        agent_vars.s = sel_coeffs[5]
        agent_vars.u = np.exp(sel_coeffs[6])
        agent_vars.q = sel_coeffs[7]
        agent_vars.sigma_H = sel_coeffs[8]

        # Agent object
        agent = AlAgent(agent_vars)

        # Run task-agent interaction
        _, df_data = task_agent_int(which_exp, df_subj, agent, agent_vars, sel_coeffs, sim=sim)

        # Record subject number
        df_data['subj_num'] = i+1

        # Add data to data frame
        df_sim = pd.concat([df_sim, df_data], ignore_index=True)

        if plot_data:

            # futuretodo: function that can be used in al_simulation_sampling
            # Plot updates
            plt.figure()
            plt.plot(np.arange(len(df_subj['a_t'])), df_subj['a_t'])
            plt.plot(np.arange(len(df_data['a_t_hat'])), df_data['a_t_hat'])
            plt.plot(np.arange(len(df_data['sim_a_t'])), df_data['sim_a_t'])
            if which_exp == 2:
                plt.plot(np.arange(len(df_data['sim_y_t'])), df_data['sim_y_t'], '.')
                plt.legend(["a_t", "a_t_hat", "sim_a_t", "sim_y_t"], loc=1, framealpha=0.8)
            else:
                plt.legend(["a_t", "a_t_hat", "sim_a_t"], loc=1, framealpha=0.8)

            # Save the plot
            savename = 'al_figures/single_trial/exp%s_up_%s.pdf' % (which_exp, i)
            plt.savefig(savename)
            plt.close()

            # Plot predictions
            plt.figure()
            plt.plot(np.arange(len(df_subj['x_t'])), df_subj['x_t'], '.')
            plt.plot(np.arange(len(df_subj['b_t'])), df_subj['b_t'])
            plt.plot(np.arange(len(df_data['sim_b_t'])), df_data['sim_b_t'])
            if which_exp == 2:
                plt.plot(np.arange(len(df_data['sim_z_t'])), df_data['sim_z_t'], '.')
                plt.legend(["x_t", "b_t", "sim_b_t", "sim_z_t"], loc=1, framealpha=0.8)
            else:
                plt.legend(["x_t", "b_t", "sim_b_t"], loc=1, framealpha=0.8)

            # Save the plot
            savename = 'al_figures/single_trial/exp%s_bel_%s.pdf' % (which_exp, i)
            plt.savefig(savename)
            plt.close()

        # Extract model prediction for perseveration and estimation-error computation
        if sim:

            # Compute perseveration
            df_data['pers'] = df_data['sim_a_t'] == 0

            # Save estimation errors and perseveration
            if which_exp == 1 or which_exp == 3:

                # sim_pers_prob['main'][i] = np.mean(df_data['pers'])
                # sim_pers_prob['age_group'][i] = group[i]
                # sim_est_err_main = get_sim_est_err(df_subj, df_data, exp=1)
                # sim_est_err['main'][i] = sim_est_err_main
                # sim_est_err['age_group'][i] = group[i]

                sim_pers_prob.loc[i, 'main'] = np.mean(df_data['pers'])
                sim_pers_prob.loc[i, 'age_group'] = group[i]
                sim_est_err_main = get_sim_est_err(df_subj, df_data, exp=1)
                sim_est_err.loc[i, 'main'] = sim_est_err_main
                sim_est_err.loc[i, 'age_group'] = group[i]

            else:

                # Save perseveration for both conditions and add age
                sim_pers_prob.loc[i, 'noPush'] = np.mean(df_data[(df_data["cond"] == "main_noPush")]['pers'])
                sim_pers_prob.loc[i, 'push'] = np.mean(df_data[(df_data["cond"] == "main_push")]['pers'])
                sim_pers_prob.loc[i, 'age_group'] = group[i]

                # Save estimation error for both conditions and add age
                sim_est_err_noPush, sim_est_err_push = get_sim_est_err(df_subj, df_data)
                sim_est_err.loc[i, 'noPush'] = sim_est_err_noPush
                sim_est_err.loc[i, 'push'] = sim_est_err_push
                sim_est_err.loc[i, 'age_group'] = group[i]

        # Update progress bar
        pbar.update(1)

        # Close progress bar
        if i == n_subj - 1:
            pbar.close()

    return sim_est_err, sim_pers_prob, df_sim, true_params


def simulation_loop(df_exp, df_model, n_subj, sim_pers, plot_data=False, which_exp=1, sim_bucket_bias=True, n_sim=10):
    """ This function runs the simulation across multiple cycles

    :param df_exp: Data frame containing participant data
    :param df_model: Data frame containing model parameters
    :param n_subj: Number of participants
    :param sim_pers: Indicates if perseveration should be simulated or not
    :param plot_data: Indicates if single-trial plots for updates and predictions should be generated
    :param which_exp: Current experiment
    :param sim_bucket_bias: Indicates if bucket bias should be simulated or not
    :param n_sim: Number of simulations
    :return: all_sim_pers, all_sim_est_errs: Simulated perseveration and estimation errors of all cycles
    """

    # Cycle over simulations
    for i in range(0, n_sim):

        # Simulate the data
        sim_est_err, sim_pers_prob, df_sim, _ = simulation(df_exp, df_model, n_subj, sim_pers, which_exp=which_exp,
                                                           sim_bucket_bias=sim_bucket_bias, plot_data=plot_data)

        # Put all data in data frames for estimation errors and perseveration
        if i == 0:
            all_sim_pers = sim_pers_prob
            all_sim_est_errs = sim_est_err
            all_data = df_sim.copy()
        else:
            all_sim_pers = pd.concat([all_sim_pers, sim_pers_prob])
            all_sim_est_errs = pd.concat([all_sim_est_errs, sim_est_err])
            all_data = pd.concat([all_data, df_sim])

    return all_sim_pers, all_sim_est_errs, all_data
