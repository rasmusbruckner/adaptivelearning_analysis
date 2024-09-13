""" Sampling-model validation: Additional analyses to validate the model

    1. Load data
    2. Simulate data based on sampling model (SM) and show validation plots
    3. Plot total number of samples as a function of absolute prediction error
    4. Compare analytical solution to unnormalized numerical updates (SM style) without changes
    5. Compare analytical solution (RBM style)to unnormalized numerical updates (SM style) with changes
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from AlAgentVarsSampling import AgentVarsSampling
from AlAgentSampling import AlAgentSampling
from al_task_agent_int_sampling import task_agent_int_sampling
from al_utilities import get_df_subj
from al_plot_utils import latex_plt, plot_validation_results


# Update matplotlib to use Latex and to change some defaults
os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2016/bin/x86_64-darwin'
matplotlib = latex_plt(matplotlib)

# Get home directory
paths = os.getcwd()
path = paths.split(os.path.sep)
home_dir = path[1]

# ------------
# 1. Load data
# ------------

# Data of follow-up experiment
df_exp2 = pd.read_pickle('al_data/data_prepr_2.pkl')
df_exp2['v_t'] = 0  # turn off catch trials

# -----------------------------------------------------------------------
# 2. Simulate data based on sampling model (SM) and show validation plots
# -----------------------------------------------------------------------

# Extract subject-specific data frame
subject = 0
df_subj = get_df_subj(df_exp2, subject)

# Agent variables object
agent_vars = AgentVarsSampling()

# Set agent parameters
agent_vars.criterion = 0.01
agent_vars.n_samples = 4
agent_vars.model_sat = True
agent_vars.burn_in = 0
agent_vars.sigma = 17.5
agent = AlAgentSampling(agent_vars)

# Run task-agent interaction
df_data = task_agent_int_sampling(df_subj, agent, agent_vars, show_pbar=True)

# Plot validation
plot_validation_results(df_data, df_subj)

# --------------------------------------------------------------------------
# 3. Plot total number of samples as a function of absolute prediction error
# --------------------------------------------------------------------------

plt.figure()
plt.plot(abs(np.array(df_data['delta_t'])), np.array(df_data['tot_samples']),  'o')
plt.xlabel('Prediction Error')
plt.ylabel('Total Number of Samples')
sns.despine()

# -------------------------------------------------------------------------------------------
# 4. Compare analytical solution to unnormalized numerical updates (SM style) without changes
# -------------------------------------------------------------------------------------------

# Set mean and outcome
mu = 20
x_t = 40

# Set different potential variances of the distributions
sigma = np.sqrt(70)

# Define x-axis
x = np.round(np.linspace(-20, 70, 100), 3)

# Reinitialize agent
agent.reinitialize_agent()

# Set agent parameters
agent.x_t = x_t  # current outcome = x_t
agent.sigma = sigma  # environmental noise is sigma
agent.h = 0  # agent does not assume changes
agent.mu_0 = mu  # prior belief is mu
agent.sigma_t_sq = sigma ** 2  # estimation uncertainty is sigma^2

# Compute likelihood of outcome
lik = agent.compute_likelihood(x)

# Compute prior belief distribution
prior = agent.compute_prior(x)

# Compute posterior
post = agent.compute_posterior(prior, lik)

# Plot (normalized) distributions
plt.figure()
lik /= np.sum(lik)
plt.plot(x, lik, color="blue")
prior /= np.sum(prior)
plt.plot(x, prior, color="red")
post /= np.sum(post)
plt.plot(x, post, color="k")
sns.despine()

# --> Since variance of prior and posterior is the same, posterior is in the middle

# Grid based
# ----------
print('Hazard rate = ' + str(agent.h))
print('Grid-based expected value: ' + str(np.sum(post*x)))

# Analytical solution
# -------------------

# Relative uncertainty
tau_t = sigma ** 2 / (sigma ** 2 + sigma ** 2)

# Update prediction using relative uncertainty as learning rate
# mu_{t+1} := mu_t + tau_t * (x_t - mu_t)
updated_belief = mu + tau_t * (x_t - mu)
print('Analytically computed expected value: ' + str(updated_belief))

# ---------------------------------------------------------------------------------------------------
# 5. Compare analytical solution (RBM style) to unnormalized numerical updates (SM style) with changes
# ---------------------------------------------------------------------------------------------------

# Set agent parameters
agent.h = 0.9
agent.mu_0 = mu
agent.sigma_t_sq = sigma ** 2

# Compute likelihood of outcome
lik = agent.compute_likelihood(x)

# Compute prior belief distribution
prior = agent.compute_prior(x)

# Compute posterior
post = agent.compute_posterior(prior, lik)

# Plot (normalized) distributions
plt.figure()
prior /= np.sum(prior)
plt.plot(x, prior, color="red")
lik /= np.sum(lik)
plt.plot(x, lik, color="blue")
post /= np.sum(post)
plt.plot(x, post, color="k")
sns.despine()

# --> Since model assumes a high hazard rate, posterior is quite close to likelihood

# Grid based
# ----------
print('\nHazard rate = ' + str(agent.h))
print('Grid-based expected value: ' + str(np.sum(post*x)))

# Analytical solution
# -------------------

# Relative uncertainty
tau_t = sigma ** 2 / (sigma ** 2 + sigma ** 2)

# Update prediction using relative uncertainty as learning rate
# mu_{t+1} := mu_t + tau_t * (x_t - mu_t)
agent.tot_var = sigma ** 2 + sigma ** 2

# Take into account environmental changes by computing changepoint probability
agent.compute_cpp(x_t - mu)
updated_belief = ((mu + tau_t * (x_t - mu)) * (1-agent.omega_t)) + (x_t * agent.omega_t)
print('Analytically computed expected value: ' + str(updated_belief))

# Data of follow-up experiment
df_exp2 = pd.read_pickle('al_data/data_prepr_2.pkl')
df_exp2['v_t'] = 0  # turn off catch trials


# Show plot
plt.ioff()
plt.show()
