
## run the network at different levels of hyperpolarizing
## input to the inh population. 
# Plot avg E and I firing rates on y-axis
# and the hyperpolarizing input to the inh population on x-axis.



import json
import numpy as np
import h5py
from matplotlib import pyplot as plt
import sys, os
sys.path.append("../../../src/")

import argparse
from tqdm import tqdm
from baseline_network import BaselineNetwork

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2


def generate(save_loc, cno_inh_input_scales, seed, hebb_k, eta_k,
             n_theta, N, n_days, day, avg_E_firing_rates, avg_I_firing_rates):
    
    for input_idx, cno_inh_input_scale in enumerate(cno_inh_input_scales):

        network = BaselineNetwork(inh_type='co-tuned', E_to_E="on", E_to_I="on", I_to_I="on",
                                plasticity_E_to_E="off", plasticity_E_to_I="off", plasticity_I_to_E="off", plasticity_I_to_I="off",
                                norm=True, set_seed=True, seed=int(seed),
                                inh_scale=1.0, train_sigma=25,
                                inh_mod_type="hyperpolarizing", inh_input_scale=cno_inh_input_scale,
                                hebb_scaling=hebb_k, rand_scaling=eta_k, n_days=n_days,
                                save_location=save_loc)

        network.run_analysis(save_results=False, save_anims=False,
                            save_weights=False, save_tuning=False, plot_metrics=False)

        # avg across cells
        avg_E_firing_rates[input_idx] = np.mean(network.tuning_curves_over_days[day, :])
        avg_I_firing_rates[input_idx] = np.mean(network.tuning_curves_over_days_I[day, :])


    with h5py.File(save_loc + "cno_intensity_vs_avg_firing_rate.hdf5", "w") as f:
        f.create_dataset("avg_E_firing_rates", data=avg_E_firing_rates)
        f.create_dataset("avg_I_firing_rates", data=avg_I_firing_rates)
        f.create_dataset("cno_inh_input_scales", data=cno_inh_input_scales)

    # save hyperparameters as a json file
    params = {
        "seed": seed,
        "hebb_k": hebb_k,
        "eta_k": eta_k,
        "n_theta": n_theta,
        "n_days": n_days,
        "N": N,
        "day": day,
        "cno_inh_input_scales": cno_inh_input_scales.tolist()
    }

    with open(save_loc + "hyperparameters.json", 'w') as f:
        json.dump(params, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rec - F to E plastic - cotuned: run simulations or generate plots")
    parser.add_argument("--stage", choices=["generate", "plot"], default="plot")

    args = parser.parse_args()

    save_loc = "../../../results/08-26 - analytics/2A - rec - F to E plastic - cotuned/1. ISN - cno_intensity_vs_avg_firing_rate/"

    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    cno_inh_input_scales = np.linspace(0, 2, 10)  # simulate different strengths of the inhibitory manipulation
    seed = 10
    hebb_k = 0.3
    eta_k = 1.0
    n_theta = 100
    n_days = 2
    N = 500
    day = 0

    avg_E_firing_rates = np.full((len(cno_inh_input_scales),), np.nan)
    avg_I_firing_rates = np.full((len(cno_inh_input_scales),), np.nan)

    if args.stage == "generate":
        generate(save_loc, cno_inh_input_scales, seed, hebb_k, eta_k,
                 n_theta, N, n_days, day, avg_E_firing_rates, avg_I_firing_rates)

    if args.stage == "plot":

        with h5py.File(save_loc + "cno_intensity_vs_avg_firing_rate.hdf5", "r") as f:
            avg_E_firing_rates = f["avg_E_firing_rates"][:]
            avg_I_firing_rates = f["avg_I_firing_rates"][:]
            cno_inh_input_scales = f["cno_inh_input_scales"][:]

        plt.figure(figsize=(5, 3))
        plt.plot(cno_inh_input_scales, avg_E_firing_rates, label="Avg E firing rate", color='blue', marker='o')
        plt.plot(cno_inh_input_scales, avg_I_firing_rates, label="Avg I firing rate", color='red', marker='o')
        plt.xlabel("Hyperpolarizing input to inh population")
        plt.ylabel("Average firing rate (Hz)")
        plt.title("Recurrent - F to E plastic - cotuned")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(save_loc + "cno_intensity_vs_avg_firing_rate.svg", dpi=300)
        plt.show()