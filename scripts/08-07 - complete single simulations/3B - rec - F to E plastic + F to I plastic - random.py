## The point of this script is to 
## 1) run a single network configuration for many seeds
## 2) save the results for each seed
## 3) plot conclusive drift metrics across seeds

import h5py
import numpy as np
from matplotlib import pyplot as plt
import sys, os
sys.path.append("../../src/")

import argparse
from plot_complete import plot_all, load_data
from tqdm import tqdm
from baseline_network import BaselineNetwork

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams['font.size'] = 12


def generate(seeds, hebb_k, eta_k,
                n_days, N, cno_inh_input_scale, baseline_inh_input_scale,
                save_loc_sim):
    for seed_idx, seed in enumerate(seeds):

        save_loc_seed = save_loc_sim + f"data/seed_{seed}/"

        save_loc_baseline = save_loc_seed + "baseline/"
        if not os.path.exists(save_loc_baseline):
            os.makedirs(save_loc_baseline)
        save_loc_cno = save_loc_seed + "cno/"
        if not os.path.exists(save_loc_cno):
            os.makedirs(save_loc_cno)

        if os.path.exists(save_loc_baseline + "results.hdf5") and os.path.exists(save_loc_cno + "results.hdf5"):
            print(f"Seed {seed} results already exist. Skipping simulation.")
            continue

        network = BaselineNetwork(inh_type='random', E_to_E="on", E_to_I="on", I_to_I="on",
                                plasticity_E_to_E="off", plasticity_E_to_I="off", plasticity_I_to_E="off", plasticity_I_to_I="off",
                                plasticity_F_to_I="on",
                                norm=True, set_seed=True, seed=int(seed),
                                inh_scale=1.0, train_sigma=25,
                                inh_mod_type="hyperpolarizing", inh_input_scale=baseline_inh_input_scale,
                                hebb_scaling=hebb_k, rand_scaling=eta_k, n_days=n_days, 
                                save_location=save_loc_baseline)
        network.run_analysis(save_results=True, save_anims=False,
                            save_weights=False, save_tuning=True)
        
        network_cno = BaselineNetwork(inh_type='random', E_to_E="on", E_to_I="on", I_to_I="on",
                                plasticity_E_to_E="off", plasticity_E_to_I="off", plasticity_I_to_E="off", plasticity_I_to_I="off",
                                plasticity_F_to_I="on",
                                norm=True, set_seed=True, seed=int(seed),
                                inh_scale=1.0, train_sigma=25,
                                inh_mod_type="hyperpolarizing", inh_input_scale=cno_inh_input_scale,
                            hebb_scaling=hebb_k, rand_scaling=eta_k, n_days=n_days,
                            save_location=save_loc_cno)
        network_cno.run_analysis(save_results=True, save_anims=False,
                                save_weights=False, save_tuning=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FF - random: run simulations or generate plots")
    parser.add_argument("--stage", choices=["generate", "plot"], default="plot")

    args = parser.parse_args()

    n_seeds = 10
    seeds = np.arange(1, n_seeds + 1)  # 10 seeds
    hebb_k = 0.6
    eta_k = 0.5
    n_days = 50
    N = 400

    cno_inh_input_scale = 1.0
    baseline_inh_input_scale = 0.0

    n_theta = 500

    save_loc_general = "../../results/08-07 - complete single simulations/3B - rec - F to E plastic + F to I plastic - random/"
    sim_loc = save_loc_general + "hebb_{:.1f}_eta_{:.1f}/".format(hebb_k, eta_k)
    plots_loc = sim_loc + "plots/"
    if not os.path.exists(plots_loc):
        os.makedirs(plots_loc)

    if args.stage == "generate":
        generate(seeds, hebb_k, eta_k, n_days, N, cno_inh_input_scale, baseline_inh_input_scale, sim_loc)

    elif args.stage == "plot":
        (baseline_drift_mag, baseline_drift_rate,
         cno_drift_mag, cno_drift_rate,
         baseline_POs, cno_POs,
         baseline_tuning_over_days, cno_tuning_over_days) = load_data(sim_loc, seeds, n_seeds, n_days, N, n_theta,
                                                                      filename="results.hdf5")

        plot_all(baseline_drift_mag, baseline_drift_rate,
                 cno_drift_mag, cno_drift_rate,
                 baseline_POs, cno_POs,
                 baseline_tuning_over_days, cno_tuning_over_days,
                 n_seeds, n_days, N, n_theta,
                 seeds,
                 hebb_k, eta_k, plots_loc)
