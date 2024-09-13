""" Simulations Sampling Model: Run simulations across whole data set, e.g., for posterior predictive checks """

import numpy as np
import pandas as pd
from time import sleep
from tqdm import tqdm
from sampling.AlAgentVarsSampling import AgentVarsSampling
from sampling.AlAgentSampling import AlAgentSampling
from sampling.al_task_agent_int_sampling import task_agent_int_sampling
from sampling.al_task_agent_int_resource_only import task_agent_int_resource_only
from al_utilities import get_df_subj, get_sim_est_err


def simulation_sampling(df_exp, df_model, n_subj, model_sat=True, resource_only=False):
    """ This function simulates data using the mixture model

    :param df_exp: Data frame containing participant data
    :param df_model: Data frame containing model parameters
    :param n_subj: Number of participants
    :param model_sat: If true, model satisficing
    :param resource_only: If true, runs the resource-only version of the sampling model
    :return: sim_est_err, sim_pers_prob, df_sim, true_params: Simulated estimation errors,
             simulated perseveration probability, set of true parameters
    """

    # Inform user
    sleep(0.1)
    print('\nModel simulation:')
    sleep(0.1)

    # Initialize progress bar
    pbar = tqdm(total=n_subj)

    # Agent variables object
    agent_vars = AgentVarsSampling()
    agent_vars.model_sat = model_sat

    # Initialize data frame for data that will be recovered
    df_sim = pd.DataFrame()

    # Initialize group vector
    group = np.full(n_subj, np.nan)

    # Initialize data frames from estimation errors and perseveration
    sim_est_err = pd.DataFrame(columns=['noPush', 'push', 'age_group', 'subj_num'],
                               index=np.arange(n_subj), dtype=float)
    sim_pers_prob = pd.DataFrame(columns=['noPush', 'push', 'age_group', 'subj_num'],
                                 index=np.arange(n_subj), dtype=float)

    # Cycle over participants
    # -----------------------
    for i in range(0, n_subj):

        # Extract subject-specific data frame
        df_subj = get_df_subj(df_exp, i)

        # Extract model parameters from model data frame
        sel_coeffs = df_model[df_model['subj_num'] == i + 1].copy()

        # Extract age group of current participant
        group[i] = sel_coeffs[['age_group']].values

        if not resource_only:
            sel_coeffs = sel_coeffs[['criterion', 'n_samples']].values.tolist()[0]
        else:
            sel_coeffs = sel_coeffs[['criterion', 'n_samples', 'b_0', 'b_1']].values.tolist()[0]

        # Set agent parameters
        agent_vars.criterion = sel_coeffs[0]
        agent_vars.n_samples = sel_coeffs[1]

        agent = AlAgentSampling(agent_vars)

        # Run task-agent interaction
        if not resource_only:
            df_data = task_agent_int_sampling(df_subj, agent, agent_vars)
        else:
            df_data = task_agent_int_resource_only(df_subj, agent, agent_vars, b_0=sel_coeffs[2], b1=sel_coeffs[3])

        # Add subject number to data frame
        df_data['subj_num'] = i+1

        # Add data to data frame
        df_sim = pd.concat([df_sim, df_data], ignore_index=True)

        # Save perseveration for both conditions and add age
        sim_pers_prob.loc[i, 'noPush'] = np.mean(df_data[(df_data["cond"] == "main_noPush")]['pers'])
        sim_pers_prob.loc[i, 'push'] = np.mean(df_data[(df_data["cond"] == "main_push")]['pers'])
        sim_pers_prob.loc[i, 'age_group'] = group[i]
        sim_pers_prob.loc[i, 'subj_num'] = i+1

        # Save estimation error for both conditions and add age
        [sim_est_err_noPush, sim_est_err_push] = get_sim_est_err(df_subj, df_data)
        sim_est_err.loc[i, 'noPush'] = sim_est_err_noPush
        sim_est_err.loc[i, 'push'] = sim_est_err_push
        sim_est_err.loc[i, 'age_group'] = group[i]
        sim_est_err.loc[i, 'subj_num'] = i+1

        # Update progress bar
        pbar.update(1)

        # Close progress bar
        if i == n_subj - 1:
            pbar.close()

    return sim_est_err, sim_pers_prob, df_sim


def simulation_loop_sampling(df_exp, df_model, n_subj, n_sim=10, model_sat=True, resource_only=False):
    """ This function runs the simulation across multiple cycles

    :param df_exp: Data frame containing participant data
    :param df_model: Data frame containing model parameters
    :param n_subj: Number of participants
    :param n_sim: Number of simulations
    :param model_sat: If true, model satisficing
    :param resource_only: If true, runs the resource-only version of the sampling model
    :return: all_sim_pers, all_sim_est_errs, all_data: Simulated perseveration and estimation errors of all cycles
             and all data concatenated
    """

    # Initialize variables
    all_sim_pers = np.nan
    all_sim_est_errs = np.nan
    all_data = np.nan

    # Cycle over simulations
    for i in range(0, n_sim):

        # Simulate the data
        sim_est_err, sim_pers_prob, df_sim = simulation_sampling(df_exp, df_model, n_subj, model_sat=model_sat,
                                                                 resource_only=resource_only)

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
