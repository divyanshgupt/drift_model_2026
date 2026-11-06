import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import h5py as h5
import sys, os
sys.path.append("../../src/")
import analysis_plots

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'Arial'

save_loc_general = "../../results/06-17 - hebb vs stochastic/"
specific = '1A. F to E plastic - co-tuned inh - hebb vs stochastic/'
k_range = np.arange(0, 1.01, 0.1)
drift_mag_baseline_all = []
drift_mag_cno_all = []

for k in tqdm(k_range):
    baseline_path = save_loc_general + specific + f"baseline k_{k:.1f}/"
    cno_path = save_loc_general + specific + f"cno k_{k:.1f}/"

    with h5.File(baseline_path + "results.h5", "r") as f:
        drift_mag_baseline_all.append(f["drift_mag"][:])

    with h5.File(cno_path + "results.h5", "r") as f:
        drift_mag_cno_all.append(f["drift_mag"][:])

drift_mag_baseline_all = np.array(drift_mag_baseline_all)
print("Baseline Drift Mag Array shape:", drift_mag_baseline_all.shape)
drift_mag_cno_all = np.array(drift_mag_cno_all)
print("CNO Drift Mag Array shape:", drift_mag_cno_all.shape)

drift_mag_baseline_means = np.nanmean(drift_mag_baseline_all, axis=2) # avg over cells
drift_mag_cno_means = np.nanmean(drift_mag_cno_all, axis=2)

drift_diff_matrix = np.nanmean(drift_mag_cno_all - drift_mag_baseline_all, axis=2).T  # (n_days, n_k)

plt.figure(figsize=(6, 4))
plt.imshow(drift_diff_matrix, aspect='auto', origin='lower', extent=[k_range])

plt.colorbar(label='Mean Drift Magnitude Difference (CNO - Baseline)')
plt.xlabel('Hebbian Scaling (k)')
plt.ylabel('Day')
plt.title('Drift Magnitude Difference Heatmap')
plt.tight_layout()
plt.savefig(save_loc_general + specific + "drift_diff_heatmap.png", dpi=300)


# Plot a line of drift magnitude for each k value


## CNO summary plot
plt.figure()
for k_idx, k_val in enumerate(k_range):
    plt.plot(drift_mag_cno_means[k_idx], label=f'k={k_val:.1f}', alpha=0.8*k_idx/len(k_range))
plt.savefig(save_loc_general + specific + "cno_drift_magnitude.png", dpi=300)


## Baseline summary plot
plt.figure()
for k_idx, k_val in enumerate(k_range):
    plt.plot(drift_mag_baseline_means[k_idx], label=f'k={k_val:.1f}', alpha=0.8*k_idx/len(k_range))
plt.savefig(save_loc_general + specific + "baseline_drift_magnitude.png", dpi=300)