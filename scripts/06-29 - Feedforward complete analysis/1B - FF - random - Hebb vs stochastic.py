
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
matplotlib.use("Agg")

import sys, os
sys.path.append("../../src/")
from joblib import Parallel, delayed, parallel_config

import h5py
import scipy.stats as stats
from tqdm import tqdm
from baseline_network import BaselineNetwork
from network import FeedForward

import plot_config


save_loc_general = "../../results/06-29 - Feedforward complete analysis/1B - FF - random - Hebb vs stochastic - inh mod/"
save_loc_data = save_loc_general + "data/"

if not os.path.exists(save_loc_data):
    os.makedirs(save_loc_data)

# What are the parameters for the baseline network?

seeds = np.arange(0, 5)  # 4 seeds
hebb_k_range = np.linspace(0, 5, 10)  # 20 values from 0 to 3
eta_k_range = np.linspace(0, 5, 10)  # 20 values from 0 to 3

N = 500
cno_inh_input_scale = 1
baseline_inh_input_scale = 0
n_days = 50

drift_baseline = np.full((len(hebb_k_range), len(eta_k_range), len(seeds), N), np.nan)
drift_cno = np.full_like(drift_baseline, np.nan)
drift_diff_matrix = np.zeros((len(hebb_k_range), len(eta_k_range), len(seeds)))  # (n_hebb_k, n_eta_k, n_seeds)


def run_cell(task):

    seed_idx, seed, i, hebb_k, j, eta_k = task

    baseline_dir = save_loc_data + f"baseline hebb_{hebb_k:.1f}_eta_{eta_k:.1f}_seed_{seed}/"
    cno_dir = save_loc_data + f"cno hebb_{hebb_k:.1f}_eta_{eta_k:.1f}_seed_{seed}/"

    if os.path.exists(baseline_dir + "results.hdf5") and os.path.exists(cno_dir + "results.hdf5"):
        with h5py.File(baseline_dir + "results.hdf5", "r") as f:
            drift_base_final = f["drift_mag"][-1]  # get final day drift magnitude
        with h5py.File(cno_dir + "results.hdf5", "r") as f:
            drift_cno_final = f["drift_mag"][-1]
        return seed_idx, i, j, drift_base_final, drift_cno_final

    # Simulate the feedforward network for a single value of k
    ## Baseline
    net = FeedForward(inh='on', inh_type='blanket', 
                      inh_mod_type="hyperpolarizing", inh_input_scale=baseline_inh_input_scale,
                        hebb_scaling=hebb_k, rand_scaling=eta_k, n_days=n_days,
                        seed=seed)
    net.run_analysis(saveloc=baseline_dir, save_results=True,
                     save_anims=False)

    drift_mag, _, _ = net.get_metrics(net.N, net.n_days, net.theta_stim, net.POs)
    drift_base_final = drift_mag[-1]  # store final day drift magnitude

    del net; plt.close('all')  
    
    ## CNO
    net_CNO = FeedForward(inh='on', inh_type='blanket', 
                          inh_mod_type="hyperpolarizing", inh_input_scale=cno_inh_input_scale,
                            hebb_scaling=hebb_k, rand_scaling=eta_k, n_days=n_days,
                            seed=seed)
    net_CNO.run_analysis(saveloc=cno_dir, save_results=True,
                         save_anims=False)

    drift_mag_CNO, _, _ = net_CNO.get_metrics(net_CNO.N, net_CNO.n_days, net_CNO.theta_stim, net_CNO.POs)
    drift_cno_final = drift_mag_CNO[-1] # store final day drift magnitude

    del net_CNO; plt.close('all') 

    return seed_idx, i, j, drift_base_final, drift_cno_final



tasks = [(s, int(seed), i, float(h), j, float(e))
         for s, seed in enumerate(seeds)
         for i, h in enumerate(hebb_k_range)
         for j, e in enumerate(eta_k_range)]


with parallel_config(backend="loky", inner_max_num_threads=1):
    results = Parallel(n_jobs=4, return_as="generator_unordered")(
        delayed(run_cell)(task) for task in tasks)
    for seed_idx, i, j, drift_base_final, drift_cno_final in tqdm(results, total=len(tasks), desc="sims"):
        drift_baseline[i, j, seed_idx] = drift_base_final
        drift_cno[i, j, seed_idx] = drift_cno_final
    
drift_diff_matrix = drift_cno.mean(axis=3) - drift_baseline.mean(axis=3)  # shape: (len(hebb_k_range), len(eta_k_range), len(seeds))

np.savez(save_loc_data + "drift_diff_matrix.npz", 
         drift_baseline=drift_baseline, drift_cno=drift_cno,
         drift_diff_matrix=drift_diff_matrix, 
         hebb_k_range=hebb_k_range, eta_k_range=eta_k_range, seeds=seeds)


mean_drift_diff_matrix = drift_diff_matrix.mean(axis=2)  # shape: (len(hebb_k_range), len(eta_k_range))
## Collective heatmap
plt.figure(figsize=(6, 5))
plt.imshow(mean_drift_diff_matrix, aspect='auto', cmap='bwr', vmin=-np.abs(mean_drift_diff_matrix).max(), vmax=np.abs(mean_drift_diff_matrix).max())
plt.colorbar(label='CNO - Baseline drift (deg)')
plt.xticks(ticks=np.arange(len(eta_k_range)), labels=[f"{v:.1f}" for v in eta_k_range])
plt.yticks(ticks=np.arange(len(hebb_k_range)), labels=[f"{v:.1f}" for v in hebb_k_range])
plt.xlabel('Eta_k')
plt.ylabel('Hebb_k')
plt.title('Drift Difference (CNO - Baseline)')
plt.tight_layout()
plt.savefig(save_loc_general + "drift_diff_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()

