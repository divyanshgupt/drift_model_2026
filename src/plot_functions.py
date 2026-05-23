
import numpy as np
from matplotlib import pyplot as plt


def plot_drift_across_i_scales(location):

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
    fig.savefig(location+"/drift_across_i_scales.svg")

def plot_tuning_curve_norm_response_across_i_scales(location, i):
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.5), dpi=200)
    i = 249

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
    fig.savefig("../../results/tuned_blanket_inhibition/E_tuning_curve_across_I_scales.svg")


def plot_animated_tuning_curve_norm_response_across_i_scales(location, i):





def plot_animated_