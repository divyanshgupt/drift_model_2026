
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

def generate(save_loc, seed, hebb_k, eta_k, 
             n_theta, N, n_days, propensity_values):

        network = FeedForward(inh='on', inh_type='co-tuned',
                        inh_mod_type="hyperpolarizing", inh_input_scale=0,
                        hebb_scaling=hebb_k, rand_scaling=eta_k, n_days=n_days,
                        seed=seed)
        network.run_analysis(saveloc=save_loc, save_results=True, save_anims=False,
                            save_weights=True, save_tuning=False, plot_metrics=False)

        json_params = {
            "seed": seed,
            "hebb_k": hebb_k,
            "eta_k": eta_k,
            "n_theta": n_theta,
            "n_days": n_days,
            "N": N,
        }
        with open(save_loc + "params.json", "w") as f:
            json.dump(json_params, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FF - cotuned: run simulations or generate plots")
    parser.add_argument("--stage", choices=["generate", "plot"], default="plot")

    args = parser.parse_args()

    seed = 1
    hebb_k = 0.3
    eta_k = 1.0
    n_theta = 100
    n_days = 50
    N = 500
    inh_input_scale = 0
    propensity_values = np.linspace(0.0, 1.0, 10) * 1e-2

    save_loc = f"../../../results/08-26 - analytics/1A - FF - cotuned/4. weights_over_time/hebb_{hebb_k}_eta_{eta_k}_inh_input_{inh_input_scale}/"
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    if args.stage == "generate":
        generate(save_loc, seed, hebb_k, eta_k,
                  n_theta, N, n_days, propensity_values)

    if args.stage == "plot":
        with h5py.File(save_loc + "results.hdf5", "r") as f:
            weights_over_time = f["W"][:]

        cell_idx = int(N//2) 
        # incoming weights for a single neuron
        incoming_weights = weights_over_time[:, cell_idx, :]
        print(incoming_weights.shape)  # (n_days, n_theta)

        # plot incoming weights for a single neuron over time
        plt.figure(figsize=(6, 4))
        for pre_idx in range(0, incoming_weights.shape[0], 10 ):
            if pre_idx in np.arange(200, 300):
                plt.plot(incoming_weights[pre_idx, :], label=f"Pre {pre_idx}", alpha=0.3)
        plt.xlabel("Time (days)")
        plt.ylabel("Weight value")
        plt.title("Incoming weights for a single neuron over time")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(save_loc + "1. incoming_weights_over_time.svg")

        # row of the weight matrix over pre-synaptic neurons (different time snapshots as different lines)
        plt.figure(figsize=(6, 4))
        for time_idx in range(0, weights_over_time.shape[1], 100):
             plt.plot(weights_over_time[:, cell_idx, time_idx], label=f"Time {time_idx}", alpha=0.3)
        plt.xlabel("Pre-synaptic neuron index")
        plt.ylabel("Weight value")
        plt.title("Incoming weights for neuron at different times")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(save_loc + "2. weight_matrix_row_over_time.svg")

        # heatmap of the weight matrix over time (y:axis - time, x:axis - pre-synaptic neurons)
        plt.figure(figsize=(6, 4))
        plt.imshow(weights_over_time[:, cell_idx, :].T, origin='lower', aspect='auto', cmap='viridis',
                   vmax=0.005, vmin=0.0)
        plt.colorbar(label="Weight value")
        plt.xlabel("Pre-synaptic neuron index")
        plt.ylabel("Time (days)")
        plt.title("Heatmap of incoming weights over time")
        plt.tight_layout()
        plt.savefig(save_loc + "3. heatmap_incoming_weights_over_time.svg")