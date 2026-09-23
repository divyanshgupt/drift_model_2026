# 1. generate simulations for hebb vs stochastic grid for many seed values
# 2. save results for each seed
# 3. plot the drift difference grid plot

import sys, os
sys.path.append("../../src/")

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for plotting
from matplotlib import pyplot as plt

from joblib import Parallel, delayed
import argparse
from tqdm import tqdm
from network import FeedForward
from load_and_plot import plot_complete_hebb_vs_stoch_grid, load_data

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams['font.size'] = 12

def generate(seeds, hebb_k_range, eta_k_range, n_days, N, weight_clipping,
             baseline_inh_input_scale, cno_inh_input_scale,
             save_loc_general):
    
    tasks = [(seed_idx, seed, hebb_k, eta_k, cond)
             for seed_idx, seed in enumerate(seeds)
             for hebb_k in hebb_k_range
             for eta_k in eta_k_range
             for cond in ["baseline", "cno"]]

    def run_one(seed_idx, seed, hebb_k, eta_k, cond):
        sim_dir = save_loc_general + f"data/seed_{seed}/hebb_{hebb_k:.2f}_eta_{eta_k:.2f}/{cond}/"
        if os.path.exists(sim_dir + "results.hdf5"):
            return  # Skip if results already exist
        os.makedirs(sim_dir, exist_ok=True)
        net = FeedForward(inh="on", inh_type="blanket",
                          inh_mod_type="hyperpolarizing", inh_input_scale=baseline_inh_input_scale if cond == "baseline" else cno_inh_input_scale,
                          weight_clipping=weight_clipping, hebb_scaling=hebb_k, rand_scaling=eta_k, n_days=n_days, N=N,
                          seed=int(seed))
        net.run_analysis(saveloc=sim_dir, save_results=True, 
                         save_anims=False, save_weights=False,
                         save_tuning=False)

    Parallel(n_jobs=12, batch_size=1, verbose=10)(
        delayed(run_one)(*t) for t in tasks
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FF - random: run simulations or generate plots")
    parser.add_argument("--stage", choices=["generate", "plot"], default="plot")

    args = parser.parse_args()

    hebb_k_range = np.linspace(0, 2, 20)
    eta_k_range = np.linspace(0, 2, 20)
    seeds = np.arange(1, 10)
    n_days = 50
    weight_clipping = True
    N = 500
    baseline_inh_input_scale = 0
    cno_inh_input_scale = 1.0

    save_loc_general = "../../results/08-25 - new - Hebb vs stochastic parameterization/1B - FF - random/"
    if weight_clipping:
        save_loc_general += "weight_clipping/"

    if not os.path.exists(save_loc_general):
        os.makedirs(save_loc_general)

    if args.stage == "generate":
        generate(seeds, hebb_k_range, eta_k_range, n_days, N, weight_clipping,
                 baseline_inh_input_scale, cno_inh_input_scale,
                 save_loc_general)

    elif args.stage == "plot":
        drift_mag_baseline_all, drift_mag_cno_all = load_data(save_loc_general, hebb_k_range, eta_k_range, seeds,
                                                              n_days, N)
        plot_complete_hebb_vs_stoch_grid(drift_mag_baseline_all, drift_mag_cno_all,
                                         hebb_k_range, eta_k_range,
                                         save_loc_general)
