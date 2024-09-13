""" Simulations Motor Model: Run simulations across whole data set """

import numpy as np
import pandas as pd
from time import sleep
from tqdm import tqdm
from AlAgentRbm import AlAgent
from AlAgentVarsRbm import AgentVars
from motor.al_task_agent_int_motor import task_agent_int_motor
from al_utilities import get_df_subj, get_sim_est_err


def simulation_motor(df_exp, df_model, n_subj):
    """ This function simulates data using the motor model

    :param df_exp: Data frame containing participant data
    :param df_model: Data frame containing model parameters
    :param n_subj: Number of participants
    :return: sim_est_err: Simulated estimation errors
             sim_pers_prob: Simulated perseveration probability
             df_sim: Data frame with simulation results
    """

    # Inform user
    sleep(0.1)
    print('\nModel simulation:')
    sleep(0.1)

    # Initialize progress bar
    pbar = tqdm(total=n_subj)

    # Agent variables object
    agent_vars = AgentVars()

    # Initialize data frame for data that will be recovered
    df_sim = pd.DataFrame()

    # Initialize group vector
    group = np.full(n_subj, np.nan)

    # Initialize data frames from estimation errors and perseveration
    sim_est_err = pd.DataFrame(columns=['noPush', 'push', 'age_group', 'subj_num'], index=np.arange(n_subj),
                               dtype=float)
    sim_pers_prob = pd.DataFrame(columns=['noPush', 'push', 'age_group', 'subj_num'], index=np.arange(n_subj),
                                 dtype=float)

    # Cycle over participants
    # -----------------------
    for i in range(0, n_subj):

        # Extract subject-specific data frame
        df_subj = get_df_subj(df_exp, i)

        # Extract model parameters from model data frame
        sel_coeffs = df_model[df_model['subj_num'] == i + 1].copy()

        # Extract age group of current participant
        group[i] = sel_coeffs[['age_group']].values

        # Save parameters
        if i == 0:
            true_params = sel_coeffs
        else:
            true_params = pd.concat([true_params, sel_coeffs], ignore_index=True)

        # Select agent coefficients from input data frame
        sel_coeffs = sel_coeffs[['omikron_0', 'omikron_1', 'beta_0', 'beta_1', 'h', 's', 'u', 'sigma_H',
                                 'cost_unit', 'cost_exp']].values.tolist()[0]

        # Set agent variables of current participant
        agent_vars.h = sel_coeffs[4]
        agent_vars.s = sel_coeffs[5]
        agent_vars.u = np.exp(sel_coeffs[6])
        agent_vars.q = 0
        agent_vars.sigma_H = sel_coeffs[7]

        # Agent object
        agent = AlAgent(agent_vars)

        # Run task-agent interaction
        df_data = task_agent_int_motor(df_subj, agent, agent_vars, sel_coeffs)

        # Add subject number to data frame
        df_data['subj_num'] = i + 1

        # Add data to data frame
        df_sim = pd.concat([df_sim, df_data], ignore_index=True)

        # Compute perseveration
        df_data['pers'] = df_data['sim_a_t'] == 0

        # Save perseveration for both conditions and add age
        sim_pers_prob.loc[i, 'noPush'] = np.mean(df_data[(df_data["cond"] == "main_noPush")]['pers'])
        sim_pers_prob.loc[i, 'push'] = np.mean(df_data[(df_data["cond"] == "main_push")]['pers'])
        sim_pers_prob.loc[i, 'age_group'] = group[i]
        sim_pers_prob.loc[i, 'subj_num'] = i+1

        # Save estimation error for both conditions and add age
        sim_est_err_no_push, sim_est_err_push = get_sim_est_err(df_subj, df_data)
        sim_est_err.loc[i, 'noPush'] = sim_est_err_no_push
        sim_est_err.loc[i, 'push'] = sim_est_err_push
        sim_est_err.loc[i, 'age_group'] = group[i]
        sim_est_err.loc[i, 'subj_num'] = i+1

        # Update progress bar
        pbar.update(1)

        # Close progress bar
        if i == n_subj - 1:
            pbar.close()

    return sim_est_err, sim_pers_prob, df_sim


def simulation_loop_motor(df_exp, df_model, n_subj, n_sim=10):
    """ This function runs the simulation across multiple cycles

    :param df_exp: Data frame containing participant data
    :param df_model: Data frame containing model parameters
    :param n_subj: Number of participants
    :param n_sim: Number of simulations
    :return: all_sim_pers, all_sim_est_errs, all_data: Simulated perseveration and estimation errors of all cycles
             and all data concatenated
    """

    # Initialize variables
    all_sim_pers = np.nan
    all_sim_est_errs = np.nan
    all_data = np.nan

    # Cycle over simulations
    for i in range(n_sim):

        # Simulate the data
        sim_est_err, sim_pers_prob, df_sim = simulation_motor(df_exp, df_model, n_subj)

        # Put all data in data frames for estimation errors and perseveration
        # Also concatenate all data
        if i == 0:
            all_sim_pers = sim_pers_prob.copy()
            all_sim_est_errs = sim_est_err.copy()
            all_data = df_sim.copy()
        else:
            all_sim_pers = pd.concat([all_sim_pers, sim_pers_prob])
            all_sim_est_errs = pd.concat([all_sim_est_errs, sim_est_err])
            all_data = pd.concat([all_data, df_sim])

        all_sim_pers = all_sim_pers.melt(id_vars=['age_group', 'subj_num'])
        all_sim_est_errs = all_sim_est_errs.melt(id_vars=['age_group', 'subj_num'])

    return all_sim_pers, all_sim_est_errs, all_data
