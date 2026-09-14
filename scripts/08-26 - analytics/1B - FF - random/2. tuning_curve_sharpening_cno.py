
import numpy as np
import h5py
from matplotlib import pyplot as plt
import sys, os
sys.path.append("../../../src/")


import argparse
from tqdm import tqdm
from network import FeedForward


plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2

def generate(cno_inh_input_scales, seed, hebb_k, eta_k,
             n_theta, N, day, tuning_curves_neuron, tuning_widths_neuron, save_loc):
    for input_idx, cno_inh_input_scale in enumerate(cno_inh_input_scales):
        network = FeedForward(inh='on', inh_type='blanket',
                        inh_mod_type="hyperpolarizing", inh_input_scale=cno_inh_input_scale,
                        hebb_scaling=hebb_k, rand_scaling=eta_k, n_days=2,
                        seed=seed)

        network.run_analysis(saveloc=None, save_results=False, save_anims=False,
                            save_weights=False, save_tuning=False, plot_metrics=False)
        
        tuning_curves_neuron[input_idx, :] = network.tuning_curves_over_days_E[day][int(N//2), :]

        tuning_widths_neuron[input_idx, :] = network.tuning_widths_over_days_E[day][int(N//2)]

    # save tuning curves of the center neuron
    with h5py.File(save_loc + "tuning_curves_center_neuron.hdf5", "w") as f:
        f.create_dataset("tuning_curves", data=tuning_curves_neuron)
        f.create_dataset("cno_inh_input_scales", data=cno_inh_input_scales)
        f.create_dataset("tuning_widths", data=tuning_widths_neuron)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FF - random: run simulations or generate plots")
    parser.add_argument("--stage", choices=["generate", "plot"], default="plot")

    args = parser.parse_args()

    save_loc = "../../../results/08-26 - analytics/1B - FF - random/2. tuning_curve_sharpening_cno/"

    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
        
    cno_inh_input_scales = [0.0, 0.5, 1.0, 1.5, 2.0]  # simulate different strengths of the inhibitory manipulation
    seed = 10
    hebb_k = 0.3
    eta_k = 1.0
    n_theta = 100
    N = 500
    day = 1
    tuning_curves_neuron = np.full((len(cno_inh_input_scales), n_theta), np.nan)
    tuning_widths_neuron = np.full((len(cno_inh_input_scales), 1), np.nan)
    if args.stage == "generate":
        generate(cno_inh_input_scales, seed, hebb_k, eta_k,
                  n_theta, N, day, 
                  tuning_curves_neuron, tuning_widths_neuron, save_loc)

    if args.stage == "plot":
        # load the tuning curves of the center neuron
        with h5py.File(save_loc + "tuning_curves_center_neuron.hdf5", "r") as f:
            tuning_curves_neuron = f["tuning_curves"][:]
            tuning_widths_neuron = f["tuning_widths"][:]
            cno_inh_input_scales = f["cno_inh_input_scales"][:]

        x_ticks = np.linspace(0, 180, 5)
        xtick_labels = [0, 45, 90, 135, 180]
        color = 'blue'
        fig, axs = plt.subplots(1, 3, figsize=(10, 3))
        axs[0].set_xticks(x_ticks)
        axs[0].set_xticklabels(xtick_labels)
        axs[1].set_xticks(x_ticks)
        axs[1].set_xticklabels(xtick_labels)

        scaled_cno_inh_input_scales = 0.1 * cno_inh_input_scales
        for input_idx, scaled_cno_inh_input_scale in enumerate(scaled_cno_inh_input_scales):
            axs[0].plot(np.linspace(0, 180, n_theta), tuning_curves_neuron[input_idx, :],
                         label=rf"$I_{{CNO}}$: {scaled_cno_inh_input_scale:.2f}", color=color, alpha=0.2 + input_idx * 0.2)
            axs[1].plot(np.linspace(0, 180, n_theta), tuning_curves_neuron[input_idx, :] / np.max(tuning_curves_neuron[input_idx, :]),
                         label=rf"$I_{{CNO}}$: {scaled_cno_inh_input_scale:.2f}", color=color, alpha=0.2 + input_idx * 0.2)
        axs[0].set_xlabel(r"$\theta$")
        axs[0].set_ylabel("Tuning")
        axs[0].set_title("Tuning Curves")
        axs[0].legend(frameon=False)

        axs[1].set_xlabel(r"$\theta$")
        axs[1].set_ylabel("Normalized Tuning")
        axs[1].set_title("Normalized Tuning Curves")
        axs[1].legend(frameon=False)

        axs[2].plot(scaled_cno_inh_input_scales, tuning_widths_neuron.flatten(), marker='o', color='black')
        axs[2].set_xlabel(r"$I_{CNO}$")
        axs[2].set_ylabel("Tuning Width (°)")
        axs[2].set_title("Tuning Widths")
        axs[2].legend(frameon=False)

        fig.suptitle("Feedforward - random")
        fig.tight_layout()
        fig.savefig(save_loc + "tuning_curves_center_neuron.svg")

