
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
             weight_clip, n_theta, N, n_days):

        network = FeedForward(inh='on', inh_type='co-tuned',
                        inh_mod_type="hyperpolarizing", inh_input_scale=0,
                        hebb_scaling=hebb_k, rand_scaling=eta_k, n_days=n_days,
                        seed=seed, pre_run=False,
                          prop_shift=prop_shift, weight_clipping=weight_clip)
        network.run_analysis(saveloc=save_loc, save_results=True, save_anims=False,
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
    prop_shift = 0
    weight_clip = True
    n_theta = 100
    n_days = 100
    N = 500
    inh_input_scale = 0

    save_loc = f"../../../results/08-26 - analytics/1A - FF - cotuned/3. weights_over_time/hebb_{hebb_k}_eta_{eta_k}_inh_input_{inh_input_scale}_prop_shift_{prop_shift}_weight_clip_{weight_clip}/"
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    if args.stage == "generate":
        generate(save_loc, seed, hebb_k, eta_k, prop_shift,
                 weight_clip, n_theta, N, n_days)

    if args.stage == "plot":
        with h5py.File(save_loc + "results.hdf5", "r") as f:
            weights_over_time = f["W"][:]
            POs = f["POs"][:]

        cell_idxs = [int(N//2), int(N//4), int(3*N//4)]
        for cell_idx in cell_idxs:
            incoming_weights = weights_over_time[:, cell_idx, :]
            # plot incoming weights for a single neuron over time
            plt.figure(figsize=(6, 4))
            time_array = np.linspace(0, n_days, incoming_weights.shape[1])
            for pre_idx in range(0, incoming_weights.shape[0], 5):
                if pre_idx in np.arange(0, incoming_weights.shape[0]):
                    plt.plot(time_array,incoming_weights[pre_idx, :], label=f"Pre {pre_idx}", alpha=0.3)
            plt.xlabel("Time (days)")
            plt.ylabel("Weight value")
            plt.title("Incoming weights for a single neuron over time")
            plt.xticks(np.arange(0, n_days, 20), np.arange(0, n_days, 20))
            # plt.legend(frameon=False)
            plt.tight_layout()
            plt.savefig(save_loc + f"1. incoming_weights_over_time_{cell_idx}.svg")

        # row of the weight matrix over pre-synaptic neurons (different time snapshots as different lines)
        cell_idx = int(N//2)
        plt.figure(figsize=(6, 4))
        for time_idx in range(0, weights_over_time.shape[1], 100):
            plt.plot(weights_over_time[:, cell_idx, time_idx], label=f"Time {time_idx}", alpha=0.3)
        plt.xlabel("Pre-synaptic neuron index")
        plt.ylabel("Weight value")
        plt.title("Incoming weights for neuron at different times")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(save_loc + "2. weight_matrix_row_over_time.svg")

        skip_time = 30
        # heatmap of the weight matrix over time (y:axis - time, x:axis - pre-synaptic neurons)
        plt.figure(figsize=(6, 4))
        plt.imshow(weights_over_time[:, cell_idx, ::skip_time].T, origin='lower', aspect='auto', cmap='viridis',
                   vmax=0.005, vmin=0.0)
        plt.colorbar(label="Weight value")
        plt.xlabel("Pre-synaptic neuron index")
        plt.ylabel("Time (days)")
        plt.title("Heatmap of incoming weights over time")
        plt.tight_layout()
        plt.savefig(save_loc + "3. heatmap_incoming_weights_over_time.svg")


        # animation of incoming weights on a single neuron over time

        from matplotlib.animation import FuncAnimation

        post_neuron = int(N//2)
        incoming_weights = weights_over_time[:, post_neuron, :]
        print(f"Incoming weights shape: {incoming_weights.shape}")  # (n_days, n_theta)
        n_pre = incoming_weights.shape[0]
        fig, ax = plt.subplots(figsize=(6, 4))
        line, = ax.plot([], [], lw=2)
        ax.set_xlim(0, n_pre)
        ax.set_ylim(0, 0.1)
        ax.set_xlabel("Pre-neuron index")
        ax.set_ylabel("Weight value")

        def init():
            line.set_data([], [])
            return line,

        def update(frame):
            line.set_data(np.arange(n_pre), incoming_weights[:, frame])
            ax.set_title(f"time {frame}")
            return line,


        ani = FuncAnimation(fig, update, frames=n_days, init_func=init, blit=True)
        ani.save(save_loc + "4. incoming_weights_animation.gif", writer='pillow', fps=5)
            

        # also plot preferred orientation over time
        plt.figure(figsize=(6, 4))
        post_idxs = [50, 100, 150, 200, 250, 300, 350, 400, 450]
        for post_idx in post_idxs:
            plt.plot(np.arange(n_days), POs[:, post_idx], label=f"Post {post_idx}", alpha=0.3)
        plt.xlabel("Time (days)")
        plt.ylabel("Preferred orientation (radians)")
        plt.title("Preferred orientation over time")
        plt.xticks(np.arange(0, n_days, 20), np.arange(0, n_days, 20))
        plt.tight_layout()
        plt.savefig(save_loc + "5. preferred_orientation_over_time.svg")