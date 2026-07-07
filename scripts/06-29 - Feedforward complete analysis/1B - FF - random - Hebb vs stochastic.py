#%%
import numpy as np
from matplotlib import pyplot as plt
import sys, os
sys.path.append("../../src/")

import h5py
import scipy.stats as stats
from tqdm import tqdm
from baseline_network import BaselineNetwork
from network import FeedForward

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams['font.size'] = 12


save_loc_general = "../../results/06-29 - Feedforward complete analysis/1B - FF - random - Hebb vs stochastic/"
if not os.path.exists(save_loc_general):
    os.makedirs(save_loc_general)

# What are the parameters for the baseline network?

hebb_k_range = np.arange(0, 1.01, 0.1)
eta_k_range = np.arange(0, 1.01, 0.1)

drift_mag_baseline_all = np.zeros((len(hebb_k_range), len(eta_k_range), 500))  # (n_hebb_k, n_eta_k, n_cells)
drift_mag_cno_all = np.zeros((len(hebb_k_range), len(eta_k_range), 500))  # (n_hebb_k, n_eta_k, n_cells)


#%%
for i, hebb_k in enumerate(tqdm(hebb_k_range)):
    for j, eta_k in enumerate(tqdm(eta_k_range)):

        baseline_dir = save_loc_general + f"baseline hebb_{hebb_k:.1f}_eta_{eta_k:.1f}/"
        cno_dir = save_loc_general + f"cno hebb_{hebb_k:.1f}_eta_{eta_k:.1f}/"
        
        if os.path.exists(baseline_dir + "results.h5") and os.path.exists(cno_dir + "results.h5"):
            with h5py.File(baseline_dir + "results.h5", "r") as f:
                drift_mag_baseline_all[i, j] = f["drift_mag"][:][-1]  # get final day drift magnitude
            with h5py.File(cno_dir + "results.h5", "r") as f:
                drift_mag_cno_all[i, j] = f["drift_mag"][:][-1]
            continue  # skip to next k value
            
        # Simulate the feedforward network for a single value of k
        ## Baseline
        net = FeedForward(inh='on', inh_type='blanket', inh_scale=1.0, hebb_scaling=hebb_k, rand_scaling=1, n_days=100)
        net.run_analysis(saveloc=baseline_dir, save_results=True)
        drift_mag, _, _ = net.get_drift_metrics()
        drift_mag_baseline_all[i, j] = drift_mag[-1]  # store final day drift magnitude
        
        ## CNO
        net_CNO = FeedForward(inh='on', inh_type='blanket', inh_scale=0.6, hebb_scaling=hebb_k, rand_scaling=1, n_days=100)
        net_CNO.run_analysis(saveloc=cno_dir, save_results=True)
        drift_mag_CNO, _, _ = net_CNO.get_drift_metrics()
        drift_mag_cno_all[i, j] = drift_mag_CNO[-1] # store final day drift magnitude
# %%
print("Baseline drift magnitude shape:", drift_mag_baseline_all.shape)
print("CNO drift magnitude shape:", drift_mag_cno_all.shape)

drift_mag_baseline_means = drift_mag_baseline_all.mean(axis=2)
drift_mag_cno_means = drift_mag_cno_all.mean(axis=2)

drift_diff_matrix = drift_mag_cno_means - drift_mag_baseline_means # shape: (len(hebb_k_range), len(eta_k_range))

## Collective heatmap
plt.figure(figsize=(6, 5))
plt.imshow(drift_diff_matrix, aspect='auto', cmap='bwr', vmin=-np.abs(drift_diff_matrix).max(), vmax=np.abs(drift_diff_matrix).max())
plt.colorbar(label='CNO - Baseline drift (deg)')
plt.xticks(ticks=np.arange(len(eta_k_range)), labels=[f"{v:.1f}" for v in eta_k_range])
plt.yticks(ticks=np.arange(len(hebb_k_range)), labels=[f"{v:.1f}" for v in hebb_k_range])
plt.xlabel('Eta_k')
plt.ylabel('Hebb_k')
plt.title('Drift Difference (CNO - Baseline)')
plt.savefig(save_loc_general + "drift_diff_heatmap.png", dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()


