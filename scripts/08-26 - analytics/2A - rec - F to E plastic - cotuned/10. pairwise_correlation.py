
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
             J_EE, J_EI, J_IE, J_II,
             n_theta, N, n_days):
    N_inh = 100
    tuning_curves_over_days_E = np.zeros((len(cno_inh_input_scales), n_days, N, n_theta))
    tuning_curves_over_days_I = np.zeros((len(cno_inh_input_scales), n_days, N_inh, n_theta))
    
    for input_idx, cno_inh_input_scale in enumerate(cno_inh_input_scales):
        network = BaselineNetwork(inh_type='co-tuned', E_to_E="on", E_to_I="on", I_to_I="on",
                        plasticity_E_to_E="off", plasticity_E_to_I="off", plasticity_I_to_E="off", plasticity_I_to_I="off",
                        norm=True, set_seed=True, seed=int(seed),
                        inh_scale=1.0, train_sigma=25,
                        J_EE=J_EE, J_EI=J_EI, J_IE=J_IE, J_II=J_II,
                        inh_mod_type="hyperpolarizing", inh_input_scale=cno_inh_input_scale,
                        hebb_scaling=hebb_k, rand_scaling=eta_k, n_days=2,
                        save_location=save_loc)
        network.run_analysis(save_results=False, save_anims=False,
                             save_weights=False, save_tuning=False, plot_metrics=False)
        
        tuning_curves_over_days_E[input_idx] = network.tuning_curves_over_days
        tuning_curves_over_days_I[input_idx] = network.tuning_curves_over_days_I


    with h5py.File(save_loc + "tuning_curves_over_days.hdf5", "w") as f:
        f.create_dataset("tuning_curves_over_days_E", data=tuning_curves_over_days_E)
        f.create_dataset("tuning_curves_over_days_I", data=tuning_curves_over_days_I)
        f.create_dataset("cno_inh_input_scales", data=cno_inh_input_scales)

    # save hyperparameters as a json file
    params = {
        "seed": seed,
        "hebb_k": hebb_k,
        "eta_k": eta_k,
        "n_theta": n_theta,
        "n_days": n_days,
        "N": N,
        "cno_inh_input_scales": cno_inh_input_scales
    }

    with open(save_loc + "hyperparameters.json", 'w') as f:
        json.dump(params, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FF - cotuned: run simulations or generate plots")
    parser.add_argument("--stage", choices=["generate", "plot", "complete"], default="plot")

    args = parser.parse_args()

    save_loc = "../../../results/08-26 - analytics/2A - rec - F to E plastic - cotuned/10. pairwise_correlation/"

    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
        
    cno_inh_input_scales = [0.0, 0.5, 1.0, 1.5, 2.0]  # simulate different strengths of the inhibitory manipulation
    seed = 10
    hebb_k = 0.3
    eta_k = 1.0
    n_theta = 500
    N = 400
    n_days = 2
    J_EE = 1.2
    J_EI = 1.2
    J_IE = 1.2
    J_II = 1.0


    if args.stage == "generate":
        generate(save_loc, cno_inh_input_scales, seed, hebb_k, eta_k,
                  J_EE, J_EI, J_IE, J_II,
                  n_theta, N, n_days)

    if args.stage == "plot":

        with h5py.File(save_loc + "tuning_curves_over_days.hdf5", "r") as f:
            tuning_curves_over_days_E = f["tuning_curves_over_days_E"][:]
            tuning_curves_over_days_I = f["tuning_curves_over_days_I"][:]
            cno_inh_input_scales = f["cno_inh_input_scales"][:]

        scaled_cno_inh_input_scales = 0.1 * cno_inh_input_scales

        # compute pairwise pearson correlation coefficients for each day and each input scale for all neurons
        pairwise_corrs_E = np.zeros((len(cno_inh_input_scales), n_days, N, N))
        N_inh = tuning_curves_over_days_I.shape[2]
        pairwise_corrs_I = np.zeros((len(cno_inh_input_scales), n_days, N_inh, N_inh))

        day = 1
        for input_idx, cno_inh_input_scale in enumerate(cno_inh_input_scales):
            pairwise_corrs_E[input_idx, day] = np.corrcoef(tuning_curves_over_days_E[input_idx, day])
            pairwise_corrs_I[input_idx, day] = np.corrcoef(tuning_curves_over_days_I[input_idx, day])

        fisher_z_baseline = np.arctanh(pairwise_corrs_E[0, day])
        fisher_z_cno = np.arctanh(pairwise_corrs_E[1, day])

        # avg of the upper triangle of the pairwise correlation matrices for each input scale
        avg_baseline = np.nanmean(pairwise_corrs_E[0, day][np.triu_indices(N, k=1)])
        avg_cno = np.nanmean(pairwise_corrs_E[1, day][np.triu_indices(N, k=1)])

        print(f"Avg pairwise correlation - baseline: {avg_baseline:.4f}")
        print(f"Avg pairwise correlation - CNO: {avg_cno:.4f}")

        fig, axs = plt.subplots(1, 3, figsize=(10, 4))
        im = axs[0].imshow(pairwise_corrs_E[0, day], vmin=-1, vmax=1, cmap='bwr')
        axs[0].set_title(f"Pairwise Correlation - E neurons\nCNO input scale: {scaled_cno_inh_input_scales[0]:.2f}")
        axs[0].set_xlabel("Neuron index")
        axs[0].set_ylabel("Neuron index")
        fig.colorbar(im, ax=axs[0])

        im = axs[1].imshow(pairwise_corrs_E[1, day], vmin=-1, vmax=1, cmap='bwr')
        axs[1].set_title(f"Pairwise Correlation - E neurons\nCNO input scale: {scaled_cno_inh_input_scales[1]:.2f}")
        axs[1].set_xlabel("Neuron index")
        axs[1].set_ylabel("Neuron index")
        fig.colorbar(im, ax=axs[1])

        im = axs[2].bar([0, 1],
                            [avg_baseline, avg_cno],
                            yerr=[np.std(pairwise_corrs_E[0, day][np.triu_indices(N, k=1)]), np.std(pairwise_corrs_E[1, day][np.triu_indices(N, k=1)])],
                            capsize=5)
        axs[2].set_title("Mean Pairwise Correlation - E neurons")
        axs[2].set_xlabel("CNO input scale")
        axs[2].set_ylabel("Mean Pairwise Correlation")
        axs[2].set_xticks([0, 1])
        axs[2].set_xticklabels([f"{scaled_cno_inh_input_scales[0]:.2f}", f"{scaled_cno_inh_input_scales[1]:.2f}"])

        plt.tight_layout()
        plt.savefig(save_loc + f"pairwise_correlations_baseline_cno.png")


    elif args.stage == "complete":
        generate(save_loc, cno_inh_input_scales, seed, hebb_k, eta_k,
                 J_EE, J_EI, J_IE, J_II,
                   n_theta, N, n_days)

        with h5py.File(save_loc + "tuning_curves_over_days.hdf5", "r") as f:
            tuning_curves_over_days_E = f["tuning_curves_over_days_E"][:]
            tuning_curves_over_days_I = f["tuning_curves_over_days_I"][:]
            cno_inh_input_scales = f["cno_inh_input_scales"][:]

        scaled_cno_inh_input_scales = 0.1 * cno_inh_input_scales
        
        # compute pairwise pearson correlation coefficients for each day and each input scale for all neurons
        pairwise_corrs_E = np.zeros((len(cno_inh_input_scales), n_days, N, N))
        N_inh = tuning_curves_over_days_I.shape[2]
        pairwise_corrs_I = np.zeros((len(cno_inh_input_scales), n_days, N_inh, N_inh))

        day = 1
        for input_idx, cno_inh_input_scale in enumerate(cno_inh_input_scales):
            pairwise_corrs_E[input_idx, day] = np.corrcoef(tuning_curves_over_days_E[input_idx, day])
            pairwise_corrs_I[input_idx, day] = np.corrcoef(tuning_curves_over_days_I[input_idx, day])

        fisher_z_baseline = np.arctanh(pairwise_corrs_E[0, day])
        fisher_z_cno = np.arctanh(pairwise_corrs_E[1, day])

        # avg of the upper triangle of the pairwise correlation matrices for each input scale
        avg_baseline = np.nanmean(pairwise_corrs_E[0, day][np.triu_indices(N, k=1)])
        avg_cno = np.nanmean(pairwise_corrs_E[1, day][np.triu_indices(N, k=1)])

        print(f"Avg pairwise correlation - baseline: {avg_baseline:.4f}")
        print(f"Avg pairwise correlation - CNO: {avg_cno:.4f}")

        fig, axs = plt.subplots(1, 3, figsize=(10, 4))
        im = axs[0].imshow(pairwise_corrs_E[0, day], vmin=-1, vmax=1, cmap='bwr')
        axs[0].set_title(f"Pairwise Correlation - E neurons\nCNO input scale: {scaled_cno_inh_input_scales[0]:.2f}")
        axs[0].set_xlabel("Neuron index")
        axs[0].set_ylabel("Neuron index")
        fig.colorbar(im, ax=axs[0])

        im = axs[1].imshow(pairwise_corrs_E[1, day], vmin=-1, vmax=1, cmap='bwr')
        axs[1].set_title(f"Pairwise Correlation - E neurons\nCNO input scale: {scaled_cno_inh_input_scales[1]:.2f}")
        axs[1].set_xlabel("Neuron index")
        axs[1].set_ylabel("Neuron index")
        fig.colorbar(im, ax=axs[1])

        im = axs[2].bar([0, 1],
                            [avg_baseline, avg_cno],
                            yerr=[np.std(pairwise_corrs_E[0, day][np.triu_indices(N, k=1)]), np.std(pairwise_corrs_E[1, day][np.triu_indices(N, k=1)])],
                            capsize=5)
        axs[2].set_title("Mean Pairwise Correlation - E neurons")
        axs[2].set_xlabel("CNO input scale")
        axs[2].set_ylabel("Mean Pairwise Correlation")
        axs[2].set_xticks([0, 1])
        axs[2].set_xticklabels([f"{scaled_cno_inh_input_scales[0]:.2f}", f"{scaled_cno_inh_input_scales[1]:.2f}"])

        plt.tight_layout()
        plt.savefig(save_loc + f"pairwise_correlations_baseline_cno.png")
