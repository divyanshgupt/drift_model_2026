#%%
import numpy as np
from matplotlib import pyplot as plt
import sys, os
sys.path.append("../../src/")

import h5py
import scipy.stats as stats
from tqdm import tqdm
from network import FeedForward
import analysis_plots

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'Arial'

save_loc_general = "../../results/06-29 - Feedforward complete analysis/3 - FF - cotuned + increasing width/"
if not os.path.exists(save_loc_general):
    os.makedirs(save_loc_general)

#%%
net_inh_1 = FeedForward(inh='on', inh_type='co-tuned', 
                        input_sigma=25, inh_scale=0.6,
                        vars_ef_mean=2)
net_inh_1.run_analysis(save_results=True,
                       saveloc=save_loc_general + "vars_ef_mean_2/")

net_inh_2 = FeedForward(inh='on', inh_type='co-tuned',
                        input_sigma=25, inh_scale=0.6,
                        vars_ef_mean=2.2)
net_inh_2.run_analysis(save_results=True,
                       saveloc=save_loc_general + "vars_ef_mean_2.2/")

net_inh_3 = FeedForward(inh='on', inh_type='co-tuned',
                        input_sigma=25, inh_scale=0.6,
                        vars_ef_mean=2.4)
net_inh_3.run_analysis(save_results=True,
                       saveloc=save_loc_general + "vars_ef_mean_2.4/")

# load hdf5 files for the three networks and extract drift metrics
file_1 = save_loc_general + "vars_ef_mean_2/" + "results.h5"
file_2 = save_loc_general + "vars_ef_mean_2.2/" + "results.h5"
file_3 = save_loc_general + "vars_ef_mean_2.4/" + "results.h5"

# open the files and extract the drift metrics

with h5py.File(file_1, "r") as f:
    drift_mag_1 = f["drift_mag"][:]
    drift_rate_1 = f["drift_rate"][:]
    convergence_1 = f["convergence"][:]
    tuning_curve_1 = f["tuning_curves_over_days"][:]
    vars_ef_1 = f["vars_ef"][:]
    POs_1 = f["POs"][:]

with h5py.File(file_2, "r") as f:
    drift_mag_2 = f["drift_mag"][:]
    drift_rate_2 = f["drift_rate"][:]
    convergence_2 = f["convergence"][:]
    tuning_curve_2 = f["tuning_curves_over_days"][:]
    vars_ef_2 = f["vars_ef"][:]
    POs_2 = f["POs"][:]

with h5py.File(file_3, "r") as f:
    drift_mag_3 = f["drift_mag"][:]
    drift_rate_3 = f["drift_rate"][:]
    convergence_3 = f["convergence"][:]
    tuning_curve_3 = f["tuning_curves_over_days"][:]
    vars_ef_3 = f["vars_ef"][:]
    POs_3 = f["POs"][:]


drift_mags = [drift_mag_1, drift_mag_2, drift_mag_3]
drift_rates = [drift_rate_1, drift_rate_2, drift_rate_3]
convergences = [convergence_1, convergence_2, convergence_3]
tuning_curves_list = [tuning_curve_1, tuning_curve_2, tuning_curve_3]
vars_efs = [vars_ef_1, vars_ef_2, vars_ef_3]
POs = [POs_1, POs_2, POs_3]


labels = ["vars_ef_mean_2", "vars_ef_mean_2.2", "vars_ef_mean_2.4"]

#%%
# Plots
analysis_plots.plot_drift_comparison(drift_mags, drift_rates, convergences, labels, save_loc_general)

analysis_plots.plot_drift_cdf_comparison(drift_mags, labels, save_loc_general)

analysis_plots.plot_tuning_curves_comparison(tuning_curves_list, labels, cell_idx=200, day=0, save_loc=save_loc_general)

analysis_plots.plot_drift_vs_vars_ef_comparison(drift_mags, vars_efs, labels, save_loc_general)

analysis_plots.animation_tuning_curve_comparison_single_cell(tuning_curves_list, POs, labels, cell_idx=200, plot_initial=True, save_loc=save_loc_general)

analysis_plots.plot_POs_initial_vs_final_comparison(POs, labels, save_loc_general)

analysis_plots.plot_POs_spread_comparison(POs, labels, save_loc_general)

