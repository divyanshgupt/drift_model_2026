
# save the weights over time for a single simulation

# plot them in various ways to visualize the evolution of the weights over time
## 1. incoming weights on a single neuron over time (y-axis), different line for each weight
## 2. row of the weight matrix over pre-synaptic neurons (different time snapshots as different lines)
## 3. heatmap of the weight matrix over time (y:axis - time, x:axis - pre-synaptic neurons)



import json
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


def generate(save_loc, seed, hebb_k, eta_k, prop_shift,
             weight_clip, n_theta, N, n_days, baseline_input_scale, cno_input_scale):

        save_loc_baseline = save_loc + "baseline/"
        if not os.path.exists(save_loc_baseline):
            os.makedirs(save_loc_baseline)
        save_loc_cno = save_loc + "cno/"
        if not os.path.exists(save_loc_cno):
            os.makedirs(save_loc_cno)

        network = FeedForward(inh='on', inh_type='co-tuned',
                        inh_mod_type="hyperpolarizing", inh_input_scale=0,
                        hebb_scaling=hebb_k, rand_scaling=eta_k, n_days=n_days,
                        seed=seed, pre_run=False,
                          prop_shift=prop_shift, weight_clipping=weight_clip)
        network.run_analysis(saveloc=save_loc_baseline, save_results=True, save_anims=False,
                            save_weights=True, save_tuning=False, plot_metrics=False)

        network_CNO = FeedForward(inh='on', inh_type='co-tuned',
                        inh_mod_type="hyperpolarizing", inh_input_scale=cno_input_scale,
                        hebb_scaling=hebb_k, rand_scaling=eta_k, n_days=n_days,
                        seed=seed, pre_run=False,
                          prop_shift=prop_shift, weight_clipping=weight_clip)
        network_CNO.run_analysis(saveloc=save_loc_cno, save_results=True, save_anims=False,
                            save_weights=True, save_tuning=False, plot_metrics=False)

        json_params = {
            "seed": seed,
            "hebb_k": hebb_k,
            "eta_k": eta_k,
            "prop_shift": prop_shift,
            "weight_clip": weight_clip,
            "n_theta": n_theta,
            "n_days": n_days,
            "N": N,
            "baseline_input_scale": baseline_input_scale,
            "cno_input_scale": cno_input_scale,
        }
        with open(save_loc + "params.json", "w") as f:
            json.dump(json_params, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FF - cotuned: run simulations or generate plots")
    parser.add_argument("--stage", choices=["generate", "plot"], default="plot")

    args = parser.parse_args()

    seed = 1
    prop_shift = 0
    weight_clip = False
    n_theta = 100
    n_days = 100
    N = 500
    baseline_input_scale = 0
    cno_input_scale = 1

    hebb_eta_comb_list = [(2.4, 0.2), 
                          (2.4, 0.5),
                          (2.4, 1.2),
                          (2.4, 2.3),
                          (2.4, 3.7)]


    if args.stage == "generate":

        for (hebb_k, eta_k) in hebb_eta_comb_list:
            save_loc = f"../../../results/08-26 - analytics/1A - FF - cotuned/6. slice_hebb_vs_eta_weight_analysis/hebb_{hebb_k}_eta_{eta_k}/"
            if not os.path.exists(save_loc):
                        os.makedirs(save_loc)

            generate(save_loc, seed, hebb_k, eta_k, prop_shift,
                 weight_clip, n_theta, N, n_days,
                 baseline_input_scale, cno_input_scale)

    if args.stage == "plot":

        for (hebb_k, eta_k) in hebb_eta_comb_list:
            save_loc = f"../../../results/08-26 - analytics/1A - FF - cotuned/6. slice_hebb_vs_eta_weight_analysis/hebb_{hebb_k}_eta_{eta_k}/"

            save_loc_baseline = save_loc + "baseline/"
            save_loc_cno = save_loc + "cno/"

            with h5py.File(save_loc_baseline + "results.hdf5", "r") as f:
                W_baseline = f["W"][:]
                POs_baseline = f["POs"][:]

            with h5py.File(save_loc_cno + "results.hdf5", "r") as f:
                W_cno = f["W"][:]
                POs_cno = f["POs"][:]

            plots_loc = save_loc + "plots/"
            if not os.path.exists(plots_loc):
                os.makedirs(plots_loc)

            # 1. weights over time for single post neuron
            cell_idx = int(N/2)  # example post neuron index
            incoming_weights_baseline = W_baseline[:, cell_idx, :]
            incoming_weights_cno = W_cno[:, cell_idx, :]
            time_array = np.linspace(0, n_days, incoming_weights_baseline.shape[1])
            fig, axs = plt.subplots(1, 2, figsize=(8, 3), dpi=300)
    
            for pre_idx in range(0, incoming_weights_baseline.shape[0], 5):
                axs[0].plot(time_array, incoming_weights_baseline[pre_idx, :], alpha=0.3)
                axs[1].plot(time_array, incoming_weights_cno[pre_idx, :], alpha=0.3)
    
            plt.xlabel("Time (days)")
            plt.ylabel("Weight Value")
            plt.title("Incoming Weights for Post Neuron {}".format(cell_idx))
            axs[0].set_title("Baseline")
            axs[1].set_title("CNO")
            fig.suptitle(rf"$K_{{hebb}}$ = {hebb_k}, $K_{{eta}}$ = {eta_k}")
            fig.tight_layout()
            fig.savefig(plots_loc + "incoming_weights_post_neuron_{}.svg".format(cell_idx))
    
            # 2. row of the weight matrix over pre-synaptic neurons (different time snapshots as different lines)
            cell_idx = int(N/2)  # example post neuron index
            fig, axs = plt.subplots(1, 2, figsize=(8, 3), dpi=300)
            for time_idx in range(0, incoming_weights_baseline.shape[1], 100):
                axs[0].plot(np.arange(N), incoming_weights_baseline[:, time_idx], alpha=0.3)
                axs[1].plot(np.arange(N), incoming_weights_cno[:, time_idx], alpha=0.3)
            plt.xlabel("Pre-synaptic Neuron Index")
            plt.ylabel("Weight Value")
            axs[0].set_title("Baseline")
            axs[1].set_title("CNO")
            fig.suptitle(rf"$K_{{hebb}}$ = {hebb_k}, $K_{{eta}}$ = {eta_k}")
            fig.tight_layout()
            fig.savefig(plots_loc + "weight_matrix_row_post_neuron_{}.svg".format(cell_idx))

            
    
    