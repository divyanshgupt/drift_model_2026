

from matplotlib import pyplot as plt
import numpy as np
import h5py
import os
import sys
sys.path.append("../../src/")

baseline_color = 'blue'
cno_color = 'orange'

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams['font.size'] = 13
plt.rcParams['lines.linewidth'] = 2


def load_data(sim_loc, seeds, n_seeds, n_days, N, n_theta,
              filename="results.hdf5"):

    # baseline_drift_mag = np.full((n_seeds, n_days, N), np.nan)
    # baseline_drift_rate = np.full((n_seeds, n_days-1, N), np.nan)
    # cno_drift_mag = np.full((n_seeds, n_days, N), np.nan)
    # cno_drift_rate = np.full((n_seeds, n_days-1, N), np.nan)

    # baseline_POs = np.full((n_seeds, n_days, N), np.nan)
    # baseline_tuning_over_days = np.full((n_seeds, n_days, N, n_theta), np.nan)

    # cno_POs = np.full((n_seeds, n_days, N), np.nan)
    # cno_tuning_over_days = np.full((n_seeds, n_days, N, n_theta), np.nan)

    # Shapes come from the files themselves — the day axis differs between models
    # (BaselineNetwork records an extra pre-training day) as does n_theta.
    baseline_drift_mag, baseline_drift_rate = [], []
    cno_drift_mag, cno_drift_rate = [], []
    baseline_POs, cno_POs = [], []
    baseline_tuning_over_days, cno_tuning_over_days = [], []
    baseline_tuning_widths_over_days, cno_tuning_widths_over_days = [], []

    W_baseline = []
    W_cno = []
    
    for seed in seeds:
        save_loc_seed = sim_loc + f"data/seed_{seed}/"
        save_loc_baseline = save_loc_seed + "baseline/"
        save_loc_cno = save_loc_seed + "cno/"

        if not os.path.exists(save_loc_baseline + filename) or not os.path.exists(save_loc_cno + filename):
            print(f"Seed {seed} is missing baseline or cno data. Skipping.")
            continue

        with h5py.File(save_loc_baseline + filename, "r") as f:
            baseline_drift_mag.append(f["drift_mag"][:])
            baseline_drift_rate.append(f["drift_rate"][:])
            baseline_POs.append(f["POs"][:])
            baseline_tuning_over_days.append(f["tuning_curves_over_days"][:])
            baseline_tuning_widths_over_days.append(f["tuning_widths_over_days"][:])
            try:
                W_baseline.append(f["W"][:])
            except:
                pass

        with h5py.File(save_loc_cno + filename, "r") as f:
            cno_drift_mag.append(f["drift_mag"][:])
            cno_drift_rate.append(f["drift_rate"][:])
            cno_POs.append(f["POs"][:])
            cno_tuning_over_days.append(f["tuning_curves_over_days"][:])
            cno_tuning_widths_over_days.append(f["tuning_widths_over_days"][:])
            try:
                W_cno.append(f["W"][:])
            except:
                pass

    return (np.array(baseline_drift_mag), np.array(baseline_drift_rate), 
            np.array(cno_drift_mag), np.array(cno_drift_rate),
            np.array(baseline_POs), np.array(cno_POs),
            np.array(baseline_tuning_over_days), np.array(cno_tuning_over_days),
            np.array(baseline_tuning_widths_over_days), np.array(cno_tuning_widths_over_days),
            np.array(W_baseline), np.array(W_cno))


def plot_all(baseline_drift_mag, baseline_drift_rate,
              cno_drift_mag, cno_drift_rate,
              baseline_POs, cno_POs,
              baseline_tuning_over_days, cno_tuning_over_days,
              baseline_tuning_widths_over_days, cno_tuning_widths_over_days,
              W_baseline, W_cno,
              n_seeds, n_days, N, n_theta,
              seeds,
              hebb_k, eta_k, plots_loc):


    n_mag_days = baseline_drift_mag.shape[1]
    n_rate_days = baseline_drift_rate.shape[1]
    n_tc_days = baseline_tuning_over_days.shape[1]
    
    # Plot baseline drift magnitude and rate
    print("Plotting baseline drift magnitude and rate...")
    fig, axs = plt.subplots(1, 2, figsize=(8, 3), dpi=300)

    for seed_idx in range(n_seeds):
        axs[0].plot(np.arange(n_mag_days), np.nanmean(baseline_drift_mag[seed_idx], axis=1), color=baseline_color, alpha=0.3)
        axs[1].plot(np.arange(n_rate_days), np.nanmean(baseline_drift_rate[seed_idx], axis=1), color=baseline_color, alpha=0.3)

    axs[0].set_title("Drift Magnitude")
    axs[0].set_xlabel("Days")
    axs[0].set_ylabel(r"$\Delta\theta$ (°)")
    axs[0].set_ylim(-0.1, np.nanmax(np.nanmean(baseline_drift_mag, axis=2)) + 0.5)
    axs[1].set_title("Drift Rate")
    axs[1].set_xlabel("Days")
    axs[1].set_ylabel(r"$\Delta\theta/\Delta t$ (°/day)")
    axs[1].set_ylim(0, np.nanmax(np.nanmean(baseline_drift_rate, axis=2)) + 1.0)

    fig.suptitle(rf"$K_{{hebb}}$ = {hebb_k}, $K_{{eta}}$ = {eta_k}")
    fig.tight_layout()
    fig.savefig(plots_loc + "baseline_drift_metrics.svg")

    print("Plotting CNO drift magnitude and rate...")
    # Plot baseline drift distributions (combined across seeds)
    fig, axs = plt.subplots(1, 2, figsize=(8, 3), dpi=300)
    final_baseline_drift_mag = baseline_drift_mag[:, -1, :].flatten()
    final_baseline_drift_rate = baseline_drift_rate[:, -1, :].flatten()

    axs[0].hist(final_baseline_drift_mag, bins=30, color=baseline_color, alpha=0.7)
    axs[1].hist(final_baseline_drift_rate, bins=30, color=baseline_color, alpha=0.7)
    axs[0].set_title("Final Drift Magnitude ")
    axs[0].set_xlabel(r"$\Delta\theta$ (°)")
    axs[0].set_ylabel("Count")
    axs[1].set_title("Final Drift Rate")
    axs[1].set_xlabel(r"$\Delta\theta/\Delta t$ (°/day)")
    axs[1].set_ylabel("Count")

    axs[0].axvline(np.nanmean(final_baseline_drift_mag), color=baseline_color, linestyle='--', label='mean = {:.2f} °'.format(np.nanmean(final_baseline_drift_mag)))
    axs[1].axvline(np.nanmean(final_baseline_drift_rate), color=baseline_color, linestyle='--', label='mean = {:.2f} °/day'.format(np.nanmean(final_baseline_drift_rate)))

    axs[0].legend(frameon=False)
    axs[1].legend(frameon=False)
    axs[0].set_yscale('log')
    axs[1].set_yscale('log')

    fig.suptitle(rf"$K_{{hebb}}$ = {hebb_k}, $K_{{eta}}$ = {eta_k}")
    fig.tight_layout()
    fig.savefig(plots_loc + "baseline_drift_distributions.svg")


    # Plot initial and final tuning widths (combined across seeds)
    print("Plotting baseline initial and final tuning widths...")
    fig, axs = plt.subplots(1, 2, figsize=(6, 3), dpi=300)
    initial_tuning_widths = baseline_tuning_widths_over_days[:, 0, :].flatten()
    final_tuning_widths = baseline_tuning_widths_over_days[:, -1, :].flatten()

    axs[0].hist(initial_tuning_widths, bins=30, color=baseline_color, alpha=0.7)
    axs[0].axvline(np.mean(initial_tuning_widths), color=baseline_color, linestyle='--', label='mean = {:.2f} °'.format(np.mean(initial_tuning_widths)))
    axs[1].hist(final_tuning_widths, bins=30, color=baseline_color, alpha=0.7)
    axs[1].axvline(np.mean(final_tuning_widths), color=baseline_color, linestyle='--', label='mean = {:.2f} °'.format(np.mean(final_tuning_widths)))
    axs[0].set_xlabel("Tuning Width (°)")
    axs[0].set_ylabel("Count")
    axs[0].set_title("Initial")
    axs[1].set_xlabel("Tuning Width (°)")
    axs[1].set_ylabel("Count")
    axs[1].set_title("Final")
    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    axs[0].legend(frameon=False)
    axs[1].legend(frameon=False)
    plt.xticks([0, 45, 90, 135, 180])

    fig.suptitle(rf"$K_{{hebb}}$ = {hebb_k}, $K_{{eta}}$ = {eta_k}")
    fig.tight_layout()
    fig.savefig(plots_loc + "baseline_initial_vs_final_tuning_widths.svg")

    # Plot final tuning width distributions (combined across seeds)
    print("Plotting baseline vs CNO final tuning widths...")
    fig, axs = plt.subplots(1, 1, figsize=(4.5, 4), dpi=300)
    final_tuning_widths_baseline = baseline_tuning_widths_over_days[:, -1, :].flatten()
    final_tuning_widths_cno = cno_tuning_widths_over_days[:, -1, :].flatten()
    axs.hist(final_tuning_widths, bins=30, color=baseline_color, alpha=0.7)
    axs.hist(final_tuning_widths_cno, bins=30, color=cno_color, alpha=0.7)
    axs.set_xlabel("Final Tuning Width (°)")
    axs.set_ylabel("Count")
    axs.axvline(np.mean(final_tuning_widths_baseline), color=baseline_color, linestyle='--', label='baseline mean = {:.2f} °'.format(np.mean(final_tuning_widths_baseline)))
    axs.axvline(np.mean(final_tuning_widths_cno), color=cno_color, linestyle='--', label='cno mean = {:.2f} °'.format(np.mean(final_tuning_widths_cno)))
    axs.legend(frameon=False)
    axs.set_yscale('log')
    fig.suptitle(rf"$K_{{hebb}}$ = {hebb_k}, $K_{{eta}}$ = {eta_k}")
    fig.tight_layout()
    fig.savefig(plots_loc + "baseline_vs_cno_final_tuning_widths.svg")
    axs.set_title("Final Tuning Widths")

    # Baseline initial vs final POs (combined across seeds)
    print("Plotting baseline initial vs final POs...")
    fig, axs = plt.subplots(1, 1, figsize=(4.5, 4), dpi=300)
    initial_POs = baseline_POs[:, 0, :].flatten()
    final_POs = baseline_POs[:, -1, :].flatten()

    axs.scatter(initial_POs, final_POs, color=baseline_color, alpha=0.2)
    axs.set_title("Baseline")
    axs.set_xlabel("Initial POs (°)")
    axs.set_ylabel("Final POs (°)")
    axs.plot([0, 180], [0, 180], color='black', linestyle='--')
    axs.set_xticks([0, 45, 90, 135, 180])
    axs.set_yticks([0, 45, 90, 135, 180])

    fig.suptitle(rf"$K_{{hebb}}$ = {hebb_k}, $K_{{eta}}$ = {eta_k}")
    fig.tight_layout()
    fig.savefig(plots_loc + "baseline_initial_vs_final_POs.svg")

    # CNO initial vs final POs (combined across seeds)
    print("Plotting CNO initial vs final POs...")
    fig, axs = plt.subplots(1, 1, figsize=(4.5, 4), dpi=300)
    initial_POs = cno_POs[:, 0, :].flatten()
    final_POs = cno_POs[:, -1, :].flatten()

    axs.scatter(initial_POs, final_POs, color=cno_color, alpha=0.3)
    axs.set_title("CNO")
    axs.set_xlabel("Initial POs (°)")
    axs.set_ylabel("Final POs (°)")
    axs.plot([0, 180], [0, 180], color='black', linestyle='--')
    axs.set_xticks([0, 45, 90, 135, 180])
    axs.set_yticks([0, 45, 90, 135, 180])
    fig.suptitle(rf"$K_{{hebb}}$ = {hebb_k}, $K_{{eta}}$ = {eta_k}")
    fig.tight_layout()
    fig.savefig(plots_loc + "cno_initial_vs_final_POs.svg")


    # Plot baseline vs cno drift magnitude and rate
    print("Plotting baseline vs CNO drift magnitude and rate...")
    fig, axs = plt.subplots(1, 2, figsize=(8, 3), dpi=300)
    for seed_idx in range(n_seeds):
        axs[0].plot(np.arange(n_mag_days), np.nanmean(baseline_drift_mag[seed_idx], axis=1), color=baseline_color, alpha=0.3)
        axs[0].plot(np.arange(n_mag_days), np.nanmean(cno_drift_mag[seed_idx], axis=1), color=cno_color, alpha=0.3)
        axs[1].plot(np.arange(n_rate_days), np.nanmean(baseline_drift_rate[seed_idx], axis=1), color=baseline_color, alpha=0.3, label='baseline' if seed_idx == 0 else "")
        axs[1].plot(np.arange(n_rate_days), np.nanmean(cno_drift_rate[seed_idx], axis=1), color=cno_color, alpha=0.3, label='cno' if seed_idx == 0 else "")

    axs[0].set_title("Drift Magnitude")
    axs[0].set_xlabel("Days")
    axs[0].set_ylabel(r"$\Delta\theta$ (°)")
    # ylim as max of baseline or cno mag
    axs[0].set_ylim(-0.1, max(np.nanmax(np.nanmean(baseline_drift_mag, axis=2)), np.nanmax(np.nanmean(cno_drift_mag, axis=2))) + 0.5)
    axs[1].set_title("Drift Rate")
    axs[1].set_xlabel("Days")
    axs[1].set_ylabel(r"$\Delta\theta/\Delta t$ (°/day)")
    axs[1].set_ylim(0, max(np.nanmax(np.nanmean(baseline_drift_rate, axis=2)), np.nanmax(np.nanmean(cno_drift_rate, axis=2))) + 1.0)
    axs[1].legend(frameon=False)

    fig.suptitle(rf"$K_{{hebb}}$ = {hebb_k}, $K_{{eta}}$ = {eta_k}")
    fig.tight_layout()
    fig.savefig(plots_loc + "baseline_vs_cno_drift_metrics.svg")




    # Plot baseline vs cno drift distributions (combined across seeds)
    print("Plotting baseline vs CNO drift distributions...")
    fig, axs = plt.subplots(1, 2, figsize=(8, 3), dpi=300)

    final_baseline_drift_mag = baseline_drift_mag[:, -1, :].flatten()
    final_cno_drift_mag = cno_drift_mag[:, -1, :].flatten()
    final_baseline_drift_rate = baseline_drift_rate[:, -1, :].flatten()
    final_cno_drift_rate = cno_drift_rate[:, -1, :].flatten()

    axs[0].hist(final_baseline_drift_mag, bins=30, color=baseline_color, alpha=0.7)
    axs[0].hist(final_cno_drift_mag, bins=30, color=cno_color, alpha=0.7)
    axs[1].hist(final_baseline_drift_rate, bins=30, color=baseline_color, alpha=0.7)
    axs[1].hist(final_cno_drift_rate, bins=30, color=cno_color, alpha=0.7)
    axs[0].set_title("Drift Magnitude")
    axs[0].set_xlabel(r"$\Delta\theta$ (°)")
    axs[0].set_ylabel("Count")
    axs[1].set_title("Drift Rate")
    axs[1].set_xlabel(r"$\Delta\theta/\Delta t$ (°/day)")
    axs[1].set_ylabel("Count")

    axs[0].axvline(np.nanmean(final_baseline_drift_mag), color=baseline_color, linestyle='--', label='baseline mean = {:.2f} °'.format(np.nanmean(final_baseline_drift_mag)))
    axs[0].axvline(np.nanmean(final_cno_drift_mag), color=cno_color, linestyle='--', label='cno mean = {:.2f} °'.format(np.nanmean(final_cno_drift_mag)))
    axs[1].axvline(np.nanmean(final_baseline_drift_rate), color=baseline_color, linestyle='--', label='baseline mean = {:.2f} °/day'.format(np.nanmean(final_baseline_drift_rate)))
    axs[1].axvline(np.nanmean(final_cno_drift_rate), color=cno_color, linestyle='--', label='cno mean = {:.2f} °/day'.format(np.nanmean(final_cno_drift_rate)))


    axs[0].legend(frameon=False) 
    axs[1].legend(frameon=False)


    axs[0].set_yscale('log')
    axs[1].set_yscale('log')

    fig.suptitle(rf"$K_{{hebb}}$ = {hebb_k}, $K_{{eta}}$ = {eta_k}")
    fig.tight_layout()
    fig.savefig(plots_loc + "baseline_vs_cno_drift_distributions.svg")

    # tuning curve plots

    ## initial vs final tuning curves for baseline

    print("Plotting baseline initial vs final tuning curves...")
    seed_idx = 0  # example seed index
    initial_tuning_curves_baseline = baseline_tuning_over_days[seed_idx, 0, :, :].reshape(-1, n_theta)
    final_tuning_curves_baseline = baseline_tuning_over_days[seed_idx, -1, :, :].reshape(-1, n_theta)

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.5), dpi=300)
    offset = 0.01
    for cell_idx in range(0, initial_tuning_curves_baseline.shape[0], 20):
        axs[0].plot(np.linspace(0, 180, n_theta), initial_tuning_curves_baseline[cell_idx] + cell_idx * offset, color=baseline_color, alpha=0.8)
        axs[1].plot(np.linspace(0, 180, n_theta), final_tuning_curves_baseline[cell_idx] + cell_idx * offset, color=baseline_color, alpha=0.8)

    axs[0].set_xlabel(r"$\theta$ (°)")
    axs[1].set_xlabel(r"$\theta$ (°)")

    axs[0].set_yticks([]); axs[1].set_yticks([])

    fig.suptitle(rf"$K_{{hebb}}$ = {hebb_k}, $K_{{eta}}$ = {eta_k} - baseline tuning curve seed {seeds[seed_idx]}")
    fig.savefig(plots_loc + "baseline_initial_vs_final_tuning_curves.svg")


    # tuning curve animation for baseline - single cell
    print("Creating baseline tuning curve animation for single cell...")
    seed_idx = 0  # example seed index
    cell_idx = int(N/2)  # example cell index
    baseline_cell_tuning_curve = baseline_tuning_over_days[seed_idx, :, cell_idx, :]

    import matplotlib.animation as animation

    fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
    ax.set_xlim(0, 180)
    ax.set_ylim(0, np.max(baseline_cell_tuning_curve) * 1.1)
    ax.set_xlabel(r"$\theta$ (°)")
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_ylabel("Firing Rate")
    line, = ax.plot([], [], color=baseline_color)
    fig.suptitle(rf"$K_{{hebb}}$ = {hebb_k}, $K_{{eta}}$ = {eta_k} - baseline")

    fig.tight_layout()
    # plot the initial tuning curve
    theta_list = np.linspace(0, 180, n_theta, endpoint=False)
    ax.plot(theta_list, baseline_cell_tuning_curve[0, :], alpha=0.4)

    def init():
        line.set_data([], [])
        return line,

    def animate(day):
        theta_list = np.linspace(0, 180, n_theta, endpoint=False)
        line.set_data(theta_list, baseline_cell_tuning_curve[day, :])
        ax.set_title(f"Day {day + 1}, cell {cell_idx}")
        return line,

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=n_tc_days, interval=600, blit=True)
    save_path = plots_loc + f"baseline_cell_{cell_idx}_tuning_curve_animation.gif"
    anim.save(save_path, writer='imagemagick')


    # tuning curve animation for baseline vs cno - single cell
    print("Creating baseline vs CNO tuning curve animation for single cell...")
    seed_idx = 0  # example seed index
    cell_idx = int(N/2)  # example cell index
    baseline_cell_tuning_curve = baseline_tuning_over_days[seed_idx, :, cell_idx, :]
    cno_cell_tuning_curve = cno_tuning_over_days[seed_idx, :, cell_idx, :]

    fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
    # ax.set_xlim(0, 180)
    ax.set_ylim(0, max(np.max(baseline_cell_tuning_curve), np.max(cno_cell_tuning_curve)) * 1.1)
    ax.set_xlabel(r"$\theta$ (°)")
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_ylabel("Firing Rate")
    baseline_line, = ax.plot([], [], color=baseline_color, label="baseline")
    cno_line, = ax.plot([], [], color=cno_color, label="cno")
    ax.legend(frameon=False)

    ax.plot(np.linspace(0, 180, n_theta), baseline_cell_tuning_curve[0, :], alpha=0.4, color=baseline_color)
    ax.plot(np.linspace(0, 180, n_theta), cno_cell_tuning_curve[0, :], alpha=0.4, color=cno_color)
    fig.suptitle(rf"$K_{{hebb}}$ = {hebb_k}, $K_{{eta}}$ = {eta_k} - baseline vs cno")
    fig.tight_layout()

    def init():
        baseline_line.set_data([], [])
        cno_line.set_data([], [])
        return baseline_line, cno_line

    def animate(day):
        theta_list = np.linspace(0, 180, n_theta, endpoint=False)
        baseline_line.set_data(theta_list, baseline_cell_tuning_curve[day, :])
        cno_line.set_data(theta_list, cno_cell_tuning_curve[day, :])
        ax.set_title(f"Day {day + 1}, cell {cell_idx}")
        return baseline_line, cno_line

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=n_tc_days, interval=600, blit=True)
    save_path = plots_loc + f"baseline_vs_cno_cell_{cell_idx}_tuning_curve_animation.gif"
    anim.save(save_path, writer='imagemagick')

    # Static plot of tuning curves for baseline vs cno - single cell
    print("Creating static plot of tuning curves for baseline vs CNO - single cell...")
    seed_idx = 0  # example seed index
    cell_idx = int(N/2)  # example cell index
    day_idx = 0  # example day index
    baseline_cell_tuning_curve = baseline_tuning_over_days[seed_idx, day_idx, cell_idx, :]
    cno_cell_tuning_curve = cno_tuning_over_days[seed_idx, day_idx, cell_idx, :]

    fig, axs = plt.subplots(1, 1, figsize=(5, 4), dpi=200)
    theta_list = np.linspace(0, 180, n_theta, endpoint=False)
    axs.plot(theta_list, baseline_cell_tuning_curve, color=baseline_color, label="baseline")
    axs.plot(theta_list, cno_cell_tuning_curve, color=cno_color, label="cno")
    axs.set_xlabel(r"$\theta$ (°)")
    axs.set_xticks([0, 45, 90, 135, 180])
    axs.set_ylabel("Firing Rate")
    axs.set_title(f"Day {day_idx + 1}, cell {cell_idx}")
    axs.legend(frameon=False)
    fig.suptitle(rf"$K_{{hebb}}$ = {hebb_k}, $K_{{eta}}$ = {eta_k} - baseline vs cno")
    fig.tight_layout()
    save_path = plots_loc + f"baseline_vs_cno_cell_{cell_idx}_day_{day_idx + 1}_tuning_curve.svg"
    fig.savefig(save_path, dpi=300)


    # Population tuning curve animation for baseline
    print("Creating population tuning curve animation for baseline...")
    seed_idx = 0  # example seed index
    baseline_population_tuning_curves = baseline_tuning_over_days[seed_idx, :, :, :]
    skip_freq = 20
    offset = 0.001  # vertical offset for plotting multiple curves

    fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
    ax.set_xlim(0, 180)
    ax.set_ylim(0, np.max(baseline_population_tuning_curves) + (offset * N))  
    ax.set_xlabel(r"$\theta$ (°)")
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_ylabel("Firing Rate")
    fig.suptitle(rf"$K_{{hebb}}$ = {hebb_k}, $K_{{eta}}$ = {eta_k} - baseline")
    fig.tight_layout()

    lines = [ax.plot([], [], color=baseline_color, alpha=0.3)[0] for _ in range(N // skip_freq)]

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def animate(day):
        theta_list = np.linspace(0, 180, n_theta, endpoint=False)
        for idx, line in enumerate(lines):
            cell_idx = idx * skip_freq
            line.set_data(theta_list, baseline_population_tuning_curves[day, cell_idx, :] + cell_idx * offset)
        ax.set_title(f"Day {day + 1}")
        return lines

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=n_tc_days, interval=600, blit=True)
    save_path = plots_loc + f"baseline_population_tuning_curves_animation.gif"
    anim.save(save_path, writer='imagemagick')


    # if w_baseline and w_cno are not empty, plot the weight matrices
    if W_baseline.size > 0 and W_cno.size > 0:

        # 1. weights over time for single post neuron
        seed_idx = 0
        cell_idx = int(N/2)  # example post neuron index
        incoming_weights_baseline = W_baseline[seed_idx, :, cell_idx, :]
        incoming_weights_cno = W_cno[seed_idx, :, cell_idx, :]
        time_array = np.linspace(0, n_days, incoming_weights_baseline.shape[1])
        fig, axs = plt.subplots(1, 2, figsize=(8, 3), dpi=300)

        for pre_idx in range(0, incoming_weights_baseline.shape[0], 5):
            axs[0].plot(time_array, incoming_weights_baseline[pre_idx, :], color=baseline_color, alpha=0.3)
            axs[1].plot(time_array, incoming_weights_cno[pre_idx, :], color=cno_color, alpha=0.3)

        plt.xlabel("Time (days)")
        plt.ylabel("Weight Value")
        plt.title("Incoming Weights for Post Neuron {}".format(cell_idx))
        axs[0].set_title("Baseline")
        axs[1].set_title("CNO")
        fig.suptitle(rf"$K_{{hebb}}$ = {hebb_k}, $K_{{eta}}$ = {eta_k}")
        fig.tight_layout()
        fig.savefig(plots_loc + "incoming_weights_post_neuron_{}_seed_{}.svg".format(cell_idx, seed_idx))

        # 2. row of the weight matrix over pre-synaptic neurons (different time snapshots as different lines)
        cell_idx = int(N/2)  # example post neuron index
        fig, axs = plt.subplots(1, 2, figsize=(8, 3), dpi=300)
        for time_idx in range(0, incoming_weights_baseline.shape[1], 100):
            axs[0].plot(np.arange(N), incoming_weights_baseline[:, time_idx], color=baseline_color, alpha=0.3)
            axs[1].plot(np.arange(N), incoming_weights_cno[:, time_idx], color=cno_color, alpha=0.3)
        plt.xlabel("Pre-synaptic Neuron Index")
        plt.ylabel("Weight Value")
        axs[0].set_title("Baseline")
        axs[1].set_title("CNO")
        fig.suptitle(rf"$K_{{hebb}}$ = {hebb_k}, $K_{{eta}}$ = {eta_k}")
        fig.tight_layout()
        fig.savefig(plots_loc + "weight_matrix_row_post_neuron_{}_seed_{}.svg".format(cell_idx, seed_idx))
        

