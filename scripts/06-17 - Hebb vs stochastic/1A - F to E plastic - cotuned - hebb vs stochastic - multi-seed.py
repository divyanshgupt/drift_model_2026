import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # Limit number of threads for numpy/scipy to avoid oversubscription
import matplotlib; matplotlib.use("Agg")  # Use non-interactive backend for plotting
from joblib import Parallel, delayed

import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append("../../src/")

import h5py
import scipy.stats as stats
from tqdm import tqdm
import baseline_network
from baseline_network import BaselineNetwork

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams['font.size'] = 12

save_loc_general = "../../results/06-17 - hebb vs stochastic/1A. F to E plastic - co-tuned inh - hebb vs stochastic - multi-seed/"
if not os.path.exists(save_loc_general):
    os.makedirs(save_loc_general)

hebb_k_range = np.arange(0, 1.01, 0.1) 
eta_k_range = np.arange(0, 1.01, 0.1)
seeds = np.arange(1, 10)  # 9 seeds
cno_inh_input_scale = 1
baseline_inh_input_scale = 0

shape = (len(seeds), len(hebb_k_range), len(eta_k_range), 400)  # (n_seeds, n_hebb_k, n_eta_k, n_cells)
drift_mag_baseline_all = np.zeros(shape) 
drift_mag_cno_all = np.zeros(shape) 


def run_one(s, i, j, seed, hebb_k, eta_k, cond):
    # baseline_network.tqdm = lambda x, **kwargs: x  # silence nested bars from N workers
    d = save_loc_general + f"{cond} hebb_{hebb_k:.1f}_eta_{eta_k:.1f}_seed_{seed}/"
    if os.path.exists(d + "results.h5"):
        with h5py.File(d + "results.h5", "r") as f:
            drift_mag = f["drift_mag"][:]
        return s, i, j, cond, drift_mag[-1]  # return final day drift magnitude
    os.makedirs(d, exist_ok=True)
    net = BaselineNetwork(inh_type="co-tuned", E_to_E="on", E_to_I="on", I_to_I="on",
                          plasticity_E_to_E="off", plasticity_E_to_I="off", plasticity_I_to_E="off", plasticity_I_to_I="off",
                          norm=True, set_seed=True, seed=seed,
                          inh_scale=1.0, train_sigma=25,
                          inh_mod_type="hyperpolarizing",
                          hebb_scaling=hebb_k, rand_scaling=eta_k, n_days=100,
                          inh_input_scale=baseline_inh_input_scale if cond == "baseline" else cno_inh_input_scale,
                          save_location=d)

    net.run_analysis(save_results=True)
    drift_mag, _, _ = net.get_drift_metrics()
    return s, i, j, cond, drift_mag[-1]

tasks = [(s, i, j, int(seed), h, e, c)
         for s, seed in enumerate(seeds)
         for i, h in enumerate(hebb_k_range)
         for j, e in enumerate(eta_k_range)
         for c in ["baseline", "cno"]]

if __name__ == "__main__":
    
    results = Parallel(n_jobs=8, batch_size=1, verbose=10)(
        delayed(run_one)(*t) for t in tasks
    )

    for s, i, j, cond, dm in results:
        (drift_mag_baseline_all if cond == "baseline" else drift_mag_cno_all)[s, i, j] = dm


    drift_mag_baseline_all = np.array(drift_mag_baseline_all)
    print("Baseline Drift Mag Array shape:", drift_mag_baseline_all.shape)
    drift_mag_cno_all = np.array(drift_mag_cno_all)
    print("CNO Drift Mag Array shape:", drift_mag_cno_all.shape)

    drift_mag_baseline_means = np.nanmean(drift_mag_baseline_all, axis=(0,3)) # avg over seeds and cells
    drift_mag_cno_means = np.nanmean(drift_mag_cno_all, axis=(0,3)) # avg over seeds and cells

    drift_diff_matrix = drift_mag_cno_means - drift_mag_baseline_means # (len(hebb_k_range), len(eta_k_range))

    plt.figure(figsize=(6, 4))
    plt.imshow(drift_diff_matrix, aspect='auto', cmap='bwr', vmin=-np.nanmax(np.abs(drift_diff_matrix)), vmax=np.nanmax(np.abs(drift_diff_matrix)))
    plt.colorbar()
    plt.xlabel('Eta_k')
    plt.ylabel('Hebb_k')
    plt.title('Drift Difference (CNO - Baseline)')
    plt.savefig(save_loc_general + "drift_diff_heatmap.png", dpi=300)
    # plt.show()