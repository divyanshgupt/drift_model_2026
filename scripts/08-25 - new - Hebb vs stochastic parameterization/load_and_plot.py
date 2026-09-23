


import os 
import numpy as np
import h5py
from matplotlib import pyplot as plt

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams['font.size'] = 12


def load_data(data_loc, hebb_k_range, eta_k_range, seeds,
              n_days, N):
    shape = (len(seeds), len(hebb_k_range), len(eta_k_range), n_days, N)  # (n_seeds, n_hebb_k, n_eta_k)
    drift_mag_baseline_all = np.full(shape, np.nan)
    drift_mag_cno_all = np.full(shape, np.nan)

    for seed_idx, seed in enumerate(seeds):
        for i, hebb_k in enumerate(hebb_k_range):
            for j, eta_k in enumerate(eta_k_range):
                for cond in ["baseline", "cno"]:
                    data_dir = data_loc + f"data/seed_{seed}/hebb_{hebb_k:.2f}_eta_{eta_k:.2f}/{cond}/"
                    if os.path.exists(data_dir + "results.hdf5"):
                        with h5py.File(data_dir + "results.hdf5", "r") as f:
                            drift_mag = f["drift_mag"][:]
                        if cond == "baseline":
                            drift_mag_baseline_all[seed_idx, i, j] = drift_mag
                        else:
                            drift_mag_cno_all[seed_idx, i, j] = drift_mag
    return drift_mag_baseline_all, drift_mag_cno_all


def plot_complete_hebb_vs_stoch_grid(drift_mag_baseline_all, drift_mag_cno_all,
                                      hebb_k_range, eta_k_range,
                                      save_loc_general):

    plots_loc = save_loc_general + "plots/"
    os.makedirs(plots_loc, exist_ok=True)
    median_plots_loc = plots_loc + "median/"
    os.makedirs(median_plots_loc, exist_ok=True)
    # mean drift across seeds and cells
    mean_drift_baseline = np.nanmean(drift_mag_baseline_all, axis=(0, 4))
    mean_drift_cno = np.nanmean(drift_mag_cno_all, axis=(0, 4))

    final_drift_baseline = mean_drift_baseline[:, :, -1]  # final day drift
    final_drift_cno = mean_drift_cno[:, :, -1]  # final day drift

    # Plot grid of baseline drift magnitudes

    cmap = 'RdBu_r'  # colormap for drift magnitude

    # show only 5 xticks
    xticks = np.linspace(0, len(eta_k_range) - 1, 5, dtype=int)
    yticks = np.linspace(0, len(hebb_k_range) - 1, 5, dtype=int)

    v_baseline = np.nanmax(np.abs(final_drift_baseline))  # max value for color scale
    plt.figure(figsize=(6, 5))
    plt.imshow(final_drift_baseline, origin='lower', aspect='auto',
                cmap=cmap, vmin=-v_baseline, vmax=v_baseline)
    # set x and y ticks to hebb_k_range and eta_k_range
    plt.xticks(ticks=xticks, labels=[f"{x:.1f}" for x in eta_k_range[xticks]])
    plt.yticks(ticks=yticks, labels=[f"{x:.1f}" for x in hebb_k_range[yticks]])
    plt.colorbar(label='Drift Magnitude (Baseline)')
    plt.xlabel(r'$K_{\eta}$ (Stochastic Scaling)')
    plt.ylabel(r'$K_{H}$ (Hebbian Scaling)')
    plt.title('Baseline Drift Magnitude')
    plt.tight_layout()
    plt.savefig(plots_loc + "baseline_drift_grid_hebb_vs_stoch.svg", dpi=300)
    plt.close()

    # Plot grid of cno drift magnitudes
    v_cno = np.nanmax(np.abs(final_drift_cno))  # max value for color scale
    plt.figure(figsize=(6, 5))
    plt.imshow(final_drift_cno, origin='lower', aspect='auto',
             cmap=cmap, vmin=-v_cno, vmax=v_cno)
    plt.xticks(ticks=xticks, labels=[f"{x:.1f}" for x in eta_k_range[xticks]])
    plt.yticks(ticks=yticks, labels=[f"{x:.1f}" for x in hebb_k_range[yticks]])
    plt.colorbar(label='Drift Magnitude (CNO)')
    plt.xlabel(r'$K_{\eta}$ (Stochastic Scaling)')
    plt.ylabel(r'$K_{H}$ (Hebbian Scaling)')
    plt.title('CNO Drift Magnitude')
    plt.tight_layout()
    plt.savefig(plots_loc + "cno_drift_grid_hebb_vs_stoch.svg", dpi=300)
    plt.close()

    # Plot grid of drift differences (cno - baseline)
    plt.figure(figsize=(6, 5))
    final_drift_diff = final_drift_cno - final_drift_baseline
    v_diff = np.nanmax(np.abs(final_drift_diff))  # max value for color scale
    plt.imshow(final_drift_diff, origin='lower', aspect='auto', 
               cmap=cmap, vmin=-v_diff, vmax=v_diff)
    plt.xticks(ticks=xticks, labels=[f"{x:.1f}" for x in eta_k_range[xticks]])
    plt.yticks(ticks=yticks, labels=[f"{x:.1f}" for x in hebb_k_range[yticks]])
    plt.colorbar(label='Drift Difference (CNO - Baseline)')
    plt.xlabel(r'$K_{\eta}$ (Stochastic Scaling)')
    plt.ylabel(r'$K_{H}$ (Hebbian Scaling)')
    plt.title('Drift Difference')
    plt.tight_layout()
    plt.savefig(plots_loc + "drift_difference_grid_hebb_vs_stoch.svg", dpi=300)
    plt.close()


    median_drift_baseline = np.nanmedian(drift_mag_baseline_all, axis=(0, 4))
    median_drift_cno = np.nanmedian(drift_mag_cno_all, axis=(0, 4))
    final_median_drift_baseline = median_drift_baseline[:, :, -1]  # final day median drift
    final_median_drift_cno = median_drift_cno[:, :, -1]  # final day median drift

    # Plot grid of baseline median drift magnitudes
    v_median_baseline = np.nanmax(np.abs(final_median_drift_baseline))  # max value for color scale
    plt.figure(figsize=(6, 5))
    plt.imshow(final_median_drift_baseline, origin='lower', aspect='auto',
                cmap=cmap, vmin=-v_median_baseline, vmax=v_median_baseline)
    plt.xticks(ticks=xticks, labels=[f"{x:.1f}" for x in eta_k_range[xticks]])
    plt.yticks(ticks=yticks, labels=[f"{x:.1f}" for x in hebb_k_range[yticks]])
    plt.colorbar(label='Median Drift Magnitude (Baseline)')
    plt.xlabel(r'$K_{\eta}$ (Stochastic Scaling)')
    plt.ylabel(r'$K_{H}$ (Hebbian Scaling)')
    plt.title('Baseline Median Drift Magnitude')
    plt.tight_layout()
    plt.savefig(median_plots_loc + "baseline_median_drift_grid_hebb_vs_stoch.svg", dpi=300)
    plt.close()

    # Plot gird of cno median drift magnitudes
    v_median_cno = np.nanmax(np.abs(final_median_drift_cno))  # max value for color scale
    plt.figure(figsize=(6, 5))
    plt.imshow(final_median_drift_cno, origin='lower', aspect='auto',
             cmap=cmap, vmin=-v_median_cno, vmax=v_median_cno)
    plt.xticks(ticks=xticks, labels=[f"{x:.1f}" for x in eta_k_range[xticks]])
    plt.yticks(ticks=yticks, labels=[f"{x:.1f}" for x in hebb_k_range[yticks]])
    plt.colorbar(label='Median Drift Magnitude (CNO)')
    plt.xlabel(r'$K_{\eta}$ (Stochastic Scaling)')
    plt.ylabel(r'$K_{H}$ (Hebbian Scaling)')
    plt.title('CNO Median Drift Magnitude')
    plt.tight_layout()
    plt.savefig(median_plots_loc + "cno_median_drift_grid_hebb_vs_stoch.svg", dpi=300)
    plt.close()

    # Plot grid of median drift differences (cno - baseline)
    plt.figure(figsize=(6, 5))
    final_median_drift_diff = final_median_drift_cno - final_median_drift_baseline
    v_median_diff = np.nanmax(np.abs(final_median_drift_diff))  # max value for color scale
    plt.imshow(final_median_drift_diff, origin='lower', aspect='auto', 
               cmap=cmap, vmin=-v_median_diff, vmax=v_median_diff)
    plt.xticks(ticks=xticks, labels=[f"{x:.1f}" for x in eta_k_range[xticks]])
    plt.yticks(ticks=yticks, labels=[f"{x:.1f}" for x in hebb_k_range[yticks]])
    plt.colorbar(label='Median Drift Difference (CNO - Baseline)')
    plt.xlabel(r'$K_{\eta}$ (Stochastic Scaling)')
    plt.ylabel(r'$K_{H}$ (Hebbian Scaling)')
    plt.title('Median Drift Difference')
    plt.tight_layout()
    plt.savefig(median_plots_loc + "drift_median_difference_grid_hebb_vs_stoch.svg", dpi=300)
    plt.close()