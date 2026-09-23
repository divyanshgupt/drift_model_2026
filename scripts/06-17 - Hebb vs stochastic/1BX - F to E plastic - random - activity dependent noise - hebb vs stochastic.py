import numpy as np
from matplotlib import pyplot as plt
import sys, os
sys.path.append("../../src/")

import h5py
import scipy.stats as stats
from tqdm import tqdm
from baseline_network import BaselineNetwork
import analysis_plots

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'Arial'

save_loc_general = "../../results/06-17 - hebb vs stochastic/1BX. F to E plastic - random - activity dependent noise - hebb vs stochastic/"
if not os.path.exists(save_loc_general):
    os.makedirs(save_loc_general)

k_range = np.arange(0, 1.01, 0.1) 
drift_mag_baseline_all = []
drift_mag_cno_all = []

for k in tqdm(k_range):
    net_baseline = BaselineNetwork(inh_type="random", E_to_E="on", E_to_I="on", I_to_I="on",
                          plasticity_E_to_E="off", plasticity_E_to_I="off", plasticity_I_to_E="off", plasticity_I_to_I="off",
                          norm=True, set_seed=True, seed=1,
                          inh_scale=1.0, train_sigma=25,
                          inh_mod_type="hyperpolarizing",
                          hebb_scaling=k, rand_scaling=1, n_days=100,
                          activity_dependent_noise=True,
                          inh_input_scale=0,
                          save_location = save_loc_general + f"baseline k_{k:.1f}/")
    net_baseline.run_analysis(save_results=True)
    drift_mag, _, _ = net_baseline.get_drift_metrics()
    drift_mag_baseline_all.append(drift_mag)

    net_cno = BaselineNetwork(inh_type="random", E_to_E="on", E_to_I="on", I_to_I="on",
                          plasticity_E_to_E="off", plasticity_E_to_I="off", plasticity_I_to_E="off", plasticity_I_to_I="off",
                            norm=True, set_seed=True, seed=1,
                            inh_scale=1.0, train_sigma=25,
                            inh_mod_type="hyperpolarizing",
                            hebb_scaling=k, rand_scaling=1, n_days=100,
                            activity_dependent_noise=True,
                            inh_input_scale=1,
                            save_location = save_loc_general + f"cno k_{k:.1f}/")
    net_cno.run_analysis(save_results=True)
    drift_mag, _, _ = net_cno.get_drift_metrics()
    drift_mag_cno_all.append(drift_mag)

drift_mag_baseline_all = np.array(drift_mag_baseline_all)
print("Baseline Drift Mag Array shape:", drift_mag_baseline_all.shape)
drift_mag_cno_all = np.array(drift_mag_cno_all)
print("CNO Drift Mag Array shape:", drift_mag_cno_all.shape)


drift_mag_baseline_means = np.nanmean(drift_mag_baseline_all, axis=2) # avg over cells
drift_mag_cno_means = np.nanmean(drift_mag_cno_all, axis=2)

drift_diff_matrix = np.nanmean(drift_mag_cno_all - drift_mag_baseline_all, axis=2).T  # (n_days, n_k)

plt.figure(figsize=(6, 4))
plt.imshow(drift_diff_matrix, aspect='auto', cmap='bwr', vmin=-np.nanmax(np.abs(drift_diff_matrix)), vmax=np.nanmax(np.abs(drift_diff_matrix)))
plt.colorbar()
plt.xlabel('k')
plt.ylabel('Day')
plt.title('Drift Difference (CNO - Baseline)')
plt.savefig(save_loc_general + "drift_diff_heatmap.png", dpi=300)
plt.show()
