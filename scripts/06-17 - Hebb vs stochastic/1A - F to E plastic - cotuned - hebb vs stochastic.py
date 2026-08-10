import numpy as np
from matplotlib import pyplot as plt
import sys, os
sys.path.append("../../src/")

import h5py
import scipy.stats as stats
from tqdm import tqdm
from baseline_network import BaselineNetwork

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams['font.size'] = 12
# plt.rcParams['font.family'] = 'Arial'

save_loc_general = "../../results/06-17 - hebb vs stochastic/1A. F to E plastic - co-tuned inh - hebb vs stochastic/"
if not os.path.exists(save_loc_general):
    os.makedirs(save_loc_general)

hebb_k_range = np.arange(0, 1.01, 0.1) 
eta_k_range = np.arange(0, 1.01, 0.1)

drift_mag_baseline_all = np.zeros((len(hebb_k_range), len(eta_k_range), 400))  # (n_hebb_k, n_eta_k, n_cells)
drift_mag_cno_all = np.zeros((len(hebb_k_range), len(eta_k_range), 400))  # (n_hebb_k, n_eta_k, n_cells)

for i, hebb_k in enumerate(tqdm(hebb_k_range)):
    for j, eta_k in enumerate(tqdm(eta_k_range)):
    # check if simulation results already exist for this k value
        baseline_dir = save_loc_general + f"baseline hebb_{hebb_k:.1f}_eta_{eta_k:.1f}/"
        cno_dir = save_loc_general + f"cno hebb_{hebb_k:.1f}_eta_{eta_k:.1f}/"

        if os.path.exists(baseline_dir + "results.h5") and os.path.exists(cno_dir + "results.h5"):
            # load existing results
            with h5py.File(baseline_dir + "results.h5", "r") as f:
                drift_mag_baseline_all[i, j] = f["drift_mag"][:][-1]  # get final day drift magnitude
            with h5py.File(cno_dir + "results.h5", "r") as f:
                drift_mag_cno_all[i, j] = f["drift_mag"][:][-1]
            continue  # skip to next k value

        net_baseline = BaselineNetwork(inh_type="co-tuned", E_to_E="on", E_to_I="on", I_to_I="on",
                            plasticity_E_to_E="off", plasticity_E_to_I="off", plasticity_I_to_E="off", plasticity_I_to_I="off",
                            norm=True, set_seed=True, seed=1,
                            inh_scale=1.0, train_sigma=25,
                            inh_mod_type="hyperpolarizing",
                            hebb_scaling=hebb_k, rand_scaling=eta_k, n_days=100,
                            inh_input_scale=0,
                            save_location = save_loc_general + f"baseline hebb_{hebb_k:.1f}_eta_{eta_k:.1f}/")
        net_baseline.run_analysis(save_results=True)
        drift_mag, _, _ = net_baseline.get_drift_metrics()
        drift_mag_baseline_all[i, j] = drift_mag[-1]  # store final day drift magnitude

        net_cno = BaselineNetwork(inh_type="co-tuned", E_to_E="on", E_to_I="on", I_to_I="on",
                            plasticity_E_to_E="off", plasticity_E_to_I="off", plasticity_I_to_E="off", plasticity_I_to_I="off",
                                norm=True, set_seed=True, seed=1,
                                inh_scale=1.0, train_sigma=25,
                                inh_mod_type="hyperpolarizing",
                                hebb_scaling=hebb_k, rand_scaling=eta_k, n_days=100,
                                inh_input_scale=1,
                                save_location = save_loc_general + f"cno hebb_{hebb_k:.1f}_eta_{eta_k:.1f}/")
        net_cno.run_analysis(save_results=True)
        drift_mag, _, _ = net_cno.get_drift_metrics()
        drift_mag_cno_all[i, j] = drift_mag[-1]  # store final day drift magnitude

drift_mag_baseline_all = np.array(drift_mag_baseline_all)
print("Baseline Drift Mag Array shape:", drift_mag_baseline_all.shape)
drift_mag_cno_all = np.array(drift_mag_cno_all)
print("CNO Drift Mag Array shape:", drift_mag_cno_all.shape)


drift_mag_baseline_means = np.mean(drift_mag_baseline_all, axis=2) # avg over cells
drift_mag_cno_means = np.mean(drift_mag_cno_all, axis=2)

drift_diff_matrix = drift_mag_cno_means - drift_mag_baseline_means # (len(hebb_k_range), len(eta_k_range))


plt.figure(figsize=(6, 4))
plt.imshow(drift_diff_matrix, aspect='auto', cmap='bwr', vmin=-np.max(np.abs(drift_diff_matrix)), vmax=np.max(np.abs(drift_diff_matrix)))
plt.colorbar()
plt.xlabel('Eta_k')
plt.ylabel('Hebb_k')
plt.title('Drift Difference (CNO - Baseline)')
plt.savefig(save_loc_general + "drift_diff_heatmap.png", dpi=300)
plt.show()
