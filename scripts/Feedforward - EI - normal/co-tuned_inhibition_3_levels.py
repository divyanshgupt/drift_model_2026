import sys, os
sys.path.append("../../src/")

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

import scipy.stats as stats
from tqdm import tqdm
from network import FeedForward
from helper_functions import circular_gaussian


plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

inh_1_scale = 0.3
inh_2_scale = 0.7
inh_3_scale = 1.1

## Simulate three networks with different levels of co-tuned inhibition

network_inh_1 = FeedForward(inh="on", inh_type='co-tuned', inh_scale = inh_1_scale)
POs_1 = network_inh_1.get_POs_over_trials(network_inh_1.w_ef_baseline, network_inh_1.n_steps, 'baseline')
drift_mag_inh_1, drift_rate_inh_1, convergence_inh_1 = network_inh_1.get_metrics(network_inh_1.N, network_inh_1.n_days, network_inh_1.theta_stim, network_inh_1.POs)

network_inh_2 = FeedForward(inh="on", inh_type='co-tuned', inh_scale = inh_2_scale)
POs_2 = network_inh_2.get_POs_over_trials(network_inh_2.w_ef_baseline, network_inh_2.n_steps, 'baseline')
drift_mag_inh_2, drift_rate_inh_2, convergence_inh_2 = network_inh_2.get_metrics(network_inh_2.N, network_inh_2.n_days, network_inh_2.theta_stim, network_inh_2.POs)

network_inh_3 = FeedForward(inh="on", inh_type='co-tuned', inh_scale = inh_3_scale)
POs_3 = network_inh_3.get_POs_over_trials(network_inh_3.w_ef_baseline, network_inh_3.n_steps, 'baseline')
drift_mag_inh_3, drift_rate_inh_3, convergence_inh_3 = network_inh_3.get_metrics(network_inh_3.N, network_inh_3.n_days, network_inh_3.theta_stim, network_inh_3.POs)


## Plot drift magnitude and drift rate for the three levels of co-tuned inhibition

eo = 2

fig, axs = plt.subplots(1, 2, figsize=(6, 2), dpi=400)
axs[0].plot(np.arange(1, network_inh_1.n_days)[::eo], np.median(drift_mag_inh_1, axis=1)[:-1][::eo], ls='-', marker='o', ms=4, label=f'I = {inh_1_scale}', clip_on=False)
axs[0].plot(np.arange(1, network_inh_2.n_days)[::eo], np.median(drift_mag_inh_2, axis=1)[:-1][::eo], ls='-', marker='o', ms=4, label=f'I = {inh_2_scale}', clip_on=False)
axs[0].plot(np.arange(1, network_inh_3.n_days)[::eo], np.median(drift_mag_inh_3, axis=1)[:-1][::eo], ls='-', marker='o', ms=4, label=f'I = {inh_3_scale}', clip_on=False)

axs[0].set_ylim([0, 5]); axs[0].set_yticks([0, 5])
axs[0].set_xlabel('time since start [days]')
axs[0].set_ylabel(r'drift magnitude $ \; [\degree]$')
axs[0].set_xlim(0, 30)
axs[0].legend(frameon=False, fontsize=8)


axs[1].plot(np.mean(drift_rate_inh_1, axis=1)[:-1], ls='-', marker='o', ms=4, label=f'I = {inh_1_scale}', clip_on=False)
axs[1].plot(np.mean(drift_rate_inh_2, axis=1)[:-1], ls='-', marker='o', ms=4, label=f'I = {inh_2_scale}', clip_on=False)
axs[1].plot(np.mean(drift_rate_inh_3, axis=1)[:-1], ls='-', marker='o', ms=4, label=f'I = {inh_3_scale}', clip_on=False)

axs[1].set_ylim([0, 5]); axs[1].set_yticks([0, 5])
axs[1].set_xlabel('time since start [days]')
axs[1].set_ylabel(r'drift rate $ \; [\degree / $ day $]$')
axs[1].set_xlim(0, 30)  
axs[1].legend(frameon=False, fontsize=8)

fig.tight_layout()
fig.show()

fig.savefig("../../results/co-tuned_inhibition/drift_co-tuned_inhibition_3_levels.svg")


## Estimate tuning curves before and after learning for the three levels of co-tuned inhibition

num_stimuli = 500
N = 500
theta_list = np.linspace(0, 180, 500)

tuning_curves_initial_1 = np.empty((N, num_stimuli))
tuning_curves_final_1 = np.empty((N, num_stimuli))

tuning_curves_initial_2 = np.empty((N, num_stimuli))
tuning_curves_final_2 = np.empty((N, num_stimuli))

tuning_curves_initial_3 = np.empty((N, num_stimuli))
tuning_curves_final_3 = np.empty((N, num_stimuli))


for stim_num, theta in enumerate(theta_list):
    u = circular_gaussian(N, theta)
    i = network_inh_1.w_if.T.dot(u)
    e_initial = network_inh_1.w_ef_init.T.dot(u) - network_inh_1.w_ei.T.dot(i)
    e_initial[e_initial < 0] = 0
    tuning_curves_initial_1[:, stim_num] = e_initial
    
for stim_num, theta in enumerate(theta_list):
    u = circular_gaussian(N, theta)
    i = network_inh_1.w_if.T.dot(u)
    e_final = network_inh_1.W[:, :, -1].T.dot(u) - network_inh_1.w_ei.T.dot(i)
    e_final[e_final < 0] = 0
    tuning_curves_final_1[:, stim_num] = e_final

### network 2

for stim_num, theta in enumerate(theta_list):
    u = circular_gaussian(N, theta)
    i = network_inh_2.w_if.T.dot(u)
    e_initial = network_inh_2.w_ef_init.T.dot(u) - network_inh_2.w_ei.T.dot(i)
    e_initial[e_initial < 0] = 0
    tuning_curves_initial_2[:, stim_num] = e_initial
    
for stim_num, theta in enumerate(theta_list):
    u = circular_gaussian(N, theta)
    i = network_inh_2.w_if.T.dot(u)
    e_final = network_inh_2.W[:, :, -1].T.dot(u) - network_inh_2.w_ei.T.dot(i)
    e_final[e_final < 0] = 0
    tuning_curves_final_2[:, stim_num] = e_final

### network 3

for stim_num, theta in enumerate(theta_list):
    u = circular_gaussian(N, theta)
    i = network_inh_3.w_if.T.dot(u)
    e_initial = network_inh_3.w_ef_init.T.dot(u) - network_inh_3.w_ei.T.dot(i)
    e_initial[e_initial < 0] = 0
    tuning_curves_initial_3[:, stim_num] = e_initial
    
for stim_num, theta in enumerate(theta_list):
    u = circular_gaussian(N, theta)
    i = network_inh_3.w_if.T.dot(u)
    e_final = network_inh_3.W[:, :, -1].T.dot(u) - network_inh_3.w_ei.T.dot(i)
    e_final[e_final < 0] = 0
    tuning_curves_final_3[:, stim_num] = e_final


## Plot tuning curves before learning for the three levels of co-tuned inhibition

fig, axs = plt.subplots(1, 2, figsize=(7, 2.5), dpi=200)
i = 250

axs[0].plot(tuning_curves_initial_1[i], label=f'I = {inh_1_scale}')
axs[0].plot(tuning_curves_initial_2[i], label=f'I = {inh_2_scale}')
axs[0].plot(tuning_curves_initial_3[i], label=f'I = {inh_3_scale}')

axs[0].set_xticks(np.arange(0, N+1, 125), np.linspace(0, 2*np.pi, 5))
axs[0].set_xticklabels([0, 45, 90, 135, 180])
axs[0].set_xlabel("Orientation (in deg.)")

axs[0].set_title("E tuning curves")
axs[0].legend()

axs[1].plot(tuning_curves_initial_1[i]/np.max(tuning_curves_initial_1[i]), label=f'I = {inh_1_scale}')
axs[1].plot(tuning_curves_initial_2[i]/np.max(tuning_curves_initial_2[i]), label=f'I = {inh_2_scale}')
axs[1].plot(tuning_curves_initial_3[i]/np.max(tuning_curves_initial_3[i]), label=f'I = {inh_3_scale}')

axs[1].set_xticks(np.arange(0, N+1, 125), np.linspace(0, 2*np.pi, 5))
axs[1].set_xticklabels([0, 45, 90, 135, 180])
axs[1].set_xlabel("Orientation (in deg.)")

axs[1].set_title("Normalized excitatory response")
axs[1].legend()

fig.tight_layout()
fig.savefig("../../results/co-tuned_inhibition/E_tuning_curve_across_I_scales.svg")

