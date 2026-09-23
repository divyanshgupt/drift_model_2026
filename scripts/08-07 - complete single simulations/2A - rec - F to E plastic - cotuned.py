## The point of this script is to 
## 1) run a single network configuration for many seeds
## 2) save the results for each seed
## 3) plot conclusive drift metrics across seeds

import h5py
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import sys, os
sys.path.append("../../src/")

import argparse
from joblib import Parallel, delayed
from plot_complete import plot_all, load_data
from tqdm import tqdm
from baseline_network import BaselineNetwork

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams['font.size'] = 12


def generate(seeds, hebb_k, eta_k,
                n_days, N, weight_clip,
                  cno_inh_input_scale, baseline_inh_input_scale,
                save_loc_sim):
    
    def run_one(seed_idx, seed):

        save_loc_seed = save_loc_sim + f"data/seed_{seed}/"

        save_loc_baseline = save_loc_seed + "baseline/"
        if not os.path.exists(save_loc_baseline):
            os.makedirs(save_loc_baseline)
        save_loc_cno = save_loc_seed + "cno/"
        if not os.path.exists(save_loc_cno):
            os.makedirs(save_loc_cno)

        if os.path.exists(save_loc_baseline + "results.hdf5") and os.path.exists(save_loc_cno + "results.hdf5"):
            print(f"Seed {seed} results already exist. Skipping simulation.")
            return

        network = BaselineNetwork(inh_type='co-tuned', E_to_E="on", E_to_I="on", I_to_I="on",
                                plasticity_E_to_E="off", plasticity_E_to_I="off", plasticity_I_to_E="off", plasticity_I_to_I="off",
                                norm=True, set_seed=True, seed=int(seed),
                                inh_scale=1.0, train_sigma=25,
                                weight_clipping=weight_clip, inh_mod_type="hyperpolarizing", inh_input_scale=baseline_inh_input_scale,
                                hebb_scaling=hebb_k, rand_scaling=eta_k, n_days=n_days, 
                                save_location=save_loc_baseline)
        network.run_analysis(save_results=True, save_anims=False,
                            save_weights=True, save_tuning=True)
        
        network_cno = BaselineNetwork(inh_type='co-tuned', E_to_E="on", E_to_I="on", I_to_I="on",
                                plasticity_E_to_E="off", plasticity_E_to_I="off", plasticity_I_to_E="off", plasticity_I_to_I="off",
                                norm=True, set_seed=True, seed=int(seed),
                                inh_scale=1.0, train_sigma=25,
                                weight_clipping=weight_clip, inh_mod_type="hyperpolarizing", inh_input_scale=cno_inh_input_scale,
                            hebb_scaling=hebb_k, rand_scaling=eta_k, n_days=n_days,
                            save_location=save_loc_cno)
        network_cno.run_analysis(save_results=True, save_anims=False,
                                save_weights=True, save_tuning=True)

    Parallel(n_jobs=24, batch_size=1, verbose=10)(
        delayed(run_one)(seed_idx, seed) for seed_idx, seed in enumerate(seeds)
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FF - random: run simulations or generate plots")
    parser.add_argument("--stage", choices=["generate", "plot", "complete"], default="plot")

    args = parser.parse_args()

    n_seeds = 10
    seeds = np.arange(1, n_seeds + 1)  # 10 seeds
    hebb_k = 0.6
    eta_k = 0.5
    n_days = 50
    N = 400
    weight_clip = False
    cno_inh_input_scale = 1.0
    baseline_inh_input_scale = 0.0

    n_theta = 500

    hebb_k_list = [0.5]
    eta_k_list = [1.5]
    hebb_eta_zip = zip(hebb_k_list, eta_k_list)


    if args.stage == "generate":

        for hebb_k, eta_k in hebb_eta_zip:
            save_loc_general = "../../results/08-07 - complete single simulations/2A - rec - F to E plastic - cotuned/"
            if weight_clip:
                save_loc_general += "weight_clipping/"
            sim_loc = save_loc_general + "hebb_{:.2f}_eta_{:.2f}/".format(hebb_k, eta_k)
            plots_loc = sim_loc + "plots/"
            if not os.path.exists(plots_loc):
                os.makedirs(plots_loc)

            generate(seeds, hebb_k, eta_k, n_days, N, weight_clip,
                      cno_inh_input_scale, baseline_inh_input_scale, sim_loc)

    elif args.stage == "plot":

        for hebb_k, eta_k in hebb_eta_zip:
            save_loc_general = "../../results/08-07 - complete single simulations/2A - rec - F to E plastic - cotuned/"
            if weight_clip:
                save_loc_general += "weight_clipping/"
            sim_loc = save_loc_general + "hebb_{:.2f}_eta_{:.2f}/".format(hebb_k, eta_k)
            plots_loc = sim_loc + "plots/"
            if not os.path.exists(plots_loc):
                os.makedirs(plots_loc)


            (baseline_drift_mag, baseline_drift_rate,
            cno_drift_mag, cno_drift_rate,
            baseline_POs, cno_POs,
            baseline_tuning_over_days, cno_tuning_over_days,
            baseline_tuning_widths_over_days, cno_tuning_widths_over_days,
            W_baseline, W_cno) = load_data(sim_loc, seeds, n_seeds, n_days, N, n_theta,
                                                                        filename="results.hdf5")

            plot_all(baseline_drift_mag, baseline_drift_rate,
                    cno_drift_mag, cno_drift_rate,
                    baseline_POs, cno_POs,
                    baseline_tuning_over_days, cno_tuning_over_days,
                    baseline_tuning_widths_over_days, cno_tuning_widths_over_days,
                    W_baseline, W_cno,
                    n_seeds, n_days, N, n_theta,
                    seeds,
                    hebb_k, eta_k, plots_loc)

    elif args.stage == "complete":
    
        for hebb_k, eta_k in hebb_eta_zip:
            save_loc_general = "../../results/08-07 - complete single simulations/2A - rec - F to E plastic - cotuned/"
            if weight_clip:
                save_loc_general += "weight_clipping/"
            sim_loc = save_loc_general + "hebb_{:.2f}_eta_{:.2f}/".format(hebb_k, eta_k)
            plots_loc = sim_loc + "plots/"
            if not os.path.exists(plots_loc):
                os.makedirs(plots_loc)

            generate(seeds, hebb_k, eta_k, n_days, N, weight_clip,
                      cno_inh_input_scale, baseline_inh_input_scale, sim_loc)

            (baseline_drift_mag, baseline_drift_rate,
            cno_drift_mag, cno_drift_rate,
            baseline_POs, cno_POs,
            baseline_tuning_over_days, cno_tuning_over_days,
            baseline_tuning_widths_over_days, cno_tuning_widths_over_days,
            W_baseline, W_cno) = load_data(sim_loc, seeds, n_seeds, n_days, N, n_theta,
                                                                        filename="results.hdf5")

            plot_all(baseline_drift_mag, baseline_drift_rate,
                    cno_drift_mag, cno_drift_rate,
                    baseline_POs, cno_POs,
                    baseline_tuning_over_days, cno_tuning_over_days,
                    baseline_tuning_widths_over_days, cno_tuning_widths_over_days,
                    W_baseline, W_cno,
                    n_seeds, n_days+1, N, n_theta,
                    seeds,
                    hebb_k, eta_k, plots_loc)
