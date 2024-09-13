import numpy as np
import matplotlib.pyplot as plt
from al_plot_utils import plot_sampling_params
import pandas as pd
from al_simulation_sampling import simulation_loop_sampling
from al_utilities import compute_anchoring_bias, safe_save_dataframe


def run_sampling_simulation(df_exp2, criterion_params_ch, criterion_params_ya, criterion_params_oa, n_samples_params_ch,
                            n_samples_params_ya, n_samples_params_oa, n_subj, n_ch, n_ya, n_oa, age_group, file_name):
    # Sample from Gaussian
    criterion_ch = np.random.normal(loc=criterion_params_ch[0], scale=criterion_params_ch[1], size=n_ch)
    criterion_ya = np.random.normal(loc=criterion_params_ya[0], scale=criterion_params_ya[1], size=n_ya)
    criterion_oa = np.random.normal(loc=criterion_params_oa[0], scale=criterion_params_oa[1], size=n_oa)

    # Don't undermine 0.001
    criterion_ch[criterion_ch < 0.001] = 0.001
    criterion_ya[criterion_ya < 0.001] = 0.001
    criterion_oa[criterion_oa < 0.001] = 0.001

    crit_mean_ch = np.mean(criterion_ch)
    crit_mean_ya = np.mean(criterion_ya)
    crit_mean_oa = np.mean(criterion_oa)

    crit_min_ch = np.min(criterion_ch)
    crit_min_ya = np.min(criterion_ya)
    crit_min_oa = np.min(criterion_oa)

    crit_max_ch = np.max(criterion_ch)
    crit_max_ya = np.max(criterion_ya)
    crit_max_oa = np.max(criterion_oa)

    print('CH crit: mean = ' + str(crit_mean_ch) + '; range = ' + str(crit_min_ch) + '-' + str(crit_max_ch))
    print('YA crit: mean = ' + str(crit_mean_ya) + '; range = ' + str(crit_min_ya) + '-' + str(crit_max_ya))
    print('OA crit: mean = ' + str(crit_mean_oa) + '; range = ' + str(crit_min_oa) + '-' + str(crit_max_oa))

    # Plot parameter distribution
    plt.figure()
    plt.subplot(311)
    plot_sampling_params(criterion_params_ch[0], criterion_params_ch[1], criterion_ch, 2)
    plt.subplot(312)
    plot_sampling_params(criterion_params_ya[0], criterion_params_ya[1], criterion_ya, 2)
    plt.subplot(313)
    plot_sampling_params(criterion_params_oa[0], criterion_params_oa[1], criterion_oa, 2)

    # Chunk size
    # ----------

    # Sample from gamma distribution
    n_samples_ch = np.round(np.random.gamma(shape=n_samples_params_ch[0], scale=n_samples_params_ch[1], size=n_ch), 0)
    n_samples_ya = np.round(np.random.gamma(shape=n_samples_params_ya[0], scale=n_samples_params_ya[1], size=n_ya), 0)
    n_samples_oa = np.round(np.random.gamma(shape=n_samples_params_oa[0], scale=n_samples_params_oa[1], size=n_oa), 0)

    # Don't undermine 1
    n_samples_ch[n_samples_ch < 1] = 1
    n_samples_ya[n_samples_ya < 1] = 1
    n_samples_oa[n_samples_oa < 1] = 1

    samples_mean_ch = np.mean(n_samples_ch)
    samples_mean_ya = np.mean(n_samples_ya)
    samples_mean_oa = np.mean(n_samples_oa)

    samples_min_ch = np.min(n_samples_ch)
    samples_min_ya = np.min(n_samples_ya)
    samples_min_oa = np.min(n_samples_oa)

    samples_max_ch = np.max(n_samples_ch)
    samples_max_ya = np.max(n_samples_ya)
    samples_max_oa = np.max(n_samples_oa)

    print('CH N samples: mean = ' + str(samples_mean_ch) + '; range = ' + str(samples_min_ch) + '-' + str(samples_max_ch))
    print('YA N samples: mean = ' + str(samples_mean_ya) + '; range = ' + str(samples_min_ya) + '-' + str(samples_max_ya))
    print('OA N samples: mean = ' + str(samples_mean_oa) + '; range = ' + str(samples_min_oa) + '-' + str(samples_max_oa))

    # Plot parameter distribution
    plt.figure()
    plt.subplot(311)
    plot_sampling_params(n_samples_params_ch[0], n_samples_params_ch[1], n_samples_ch, 1)
    plt.subplot(312)
    plot_sampling_params(n_samples_params_ya[0], n_samples_params_ya[1], n_samples_ya, 1)
    plt.subplot(313)
    plot_sampling_params(n_samples_params_oa[0], n_samples_params_oa[1], n_samples_oa, 1)

    # -----------------
    # 2. Run simulation
    # -----------------

    # Adjust model parameters
    model_exp2 = pd.DataFrame(columns=['subj_num', 'age_group', 'criterion', 'n_samples'], index=[np.arange(n_subj)])
    model_exp2['subj_num'] = np.arange(n_subj) + 1
    model_exp2['age_group'] = np.array(age_group[:n_subj])
    model_exp2.loc[model_exp2['age_group'] == 1, 'criterion'] = criterion_ch
    model_exp2.loc[model_exp2['age_group'] == 3, 'criterion'] = criterion_ya
    model_exp2.loc[model_exp2['age_group'] == 4, 'criterion'] = criterion_oa
    model_exp2.loc[model_exp2['age_group'] == 1, 'n_samples'] = n_samples_ch
    model_exp2.loc[model_exp2['age_group'] == 3, 'n_samples'] = n_samples_ya
    model_exp2.loc[model_exp2['age_group'] == 4, 'n_samples'] = n_samples_oa

    # Run simulation
    n_sim = 1
    all_pers, all_est_errs, df_data = simulation_loop_sampling(df_exp2, model_exp2, n_subj, n_sim=n_sim)

    # Check motor perseveration
    df_data['motor_pers'] = df_data['sim_y_t'] == df_data['sim_a_t']

    # Compute anchoring bias
    df_reg = compute_anchoring_bias(n_subj, df_data)

    # -----------------------
    # 3. Save simulation data
    # -----------------------

    df_reg.name = file_name + "_df_reg"
    safe_save_dataframe(df_reg, None, overleaf=False)

    all_pers.name = file_name + "_all_pers"
    safe_save_dataframe(all_pers, None, overleaf=False)

    all_est_errs.name = file_name + "_all_est_errs"
    safe_save_dataframe(all_est_errs, None, overleaf=False)

