
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
import matplotlib.animation as animation

def plot_drift_comparison(drift_mag_list, drift_rate_list, convergence_list, labels, save_loc=None):

    fig, axs = plt.subplots(1, 3, figsize=(12, 3), dpi=300)
    for i in range(len(drift_mag_list)):
        axs[0].plot(np.nanmean(drift_mag_list[i], axis=1), marker='o', label=labels[i])
        axs[1].plot(np.nanmean(drift_rate_list[i], axis=1), marker='o', label=labels[i])
        axs[2].plot(np.nanmean(convergence_list[i], axis=1), marker='o', label=labels[i])
    axs[0].set_title("Drift magnitude")
    axs[1].set_title("Drift rate")
    axs[2].set_title("Convergence")

    for ax in axs:
        ax.legend(frameon=False)
        ax.set_ylim([-1, 6])
    plt.tight_layout()
    if save_loc is not None:
        plt.savefig(save_loc + "drift_comparison_plot.png", dpi=300)

    return None

def plot_drift_cdf_comparison(drift_mag_list, labels, save_loc=None):

    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
    for i in range(len(drift_mag_list)):
        ax.hist(drift_mag_list[i].flatten(), bins=50, density=True, cumulative=True, histtype='step', label=labels[i])
    ax.set_title("CDF of Drift Magnitude")
    ax.set_xlabel("Drift Magnitude")
    ax.set_ylabel("Cumulative density")
    ax.legend(frameon=False)
    plt.tight_layout()
    if save_loc is not None:
        plt.savefig(save_loc + "drift_magnitude_cdf_comparison.png", dpi=300)

    return None

def plot_tuning_curves_comparison(tuning_curves_list, labels, cell_idx=0, day=0, save_loc=None):

    theta_list = np.linspace(0, 180, 500, endpoint=False)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.5), dpi=300)

    for i in range(len(tuning_curves_list)):
        tuning_curve = tuning_curves_list[i][day, cell_idx, :]
        normalized_tuning_curve = tuning_curve / np.max(tuning_curve)
        axs[0].plot(theta_list, tuning_curve, label=labels[i])
        axs[1].plot(theta_list, normalized_tuning_curve, label=labels[i])

    axs[0].set_title(f"Tuning curve")
    axs[1].set_title(f"Normalized tuning curve")

    for ax in axs:
        ax.set_xlabel("Theta (degrees)")
        ax.set_ylabel("Response")
        ax.legend(frameon=False)

    fig.suptitle(f"Cell {cell_idx}, Day {day}", y=1.05)
    plt.tight_layout()

    if save_loc is not None:
        plt.savefig(save_loc + f"tuning_curves_comparison_cell{cell_idx}_day{day}.png", dpi=300)

    return None

def plot_drift_vs_vars_ef_comparison(drift_mag_list, var_ef_list, labels, save_loc=None):

    fig, axs = plt.subplots(1, len(labels), figsize=(10, 3), dpi=300, sharex=True, sharey=True)
    for i in range(len(drift_mag_list)):
        axs[i].scatter(var_ef_list[i], drift_mag_list[i][-1], alpha=0.5, color=f'C{i}')
        axs[i].set_title(f"{labels[i]}")
        axs[i].set_xlabel("Weight width F to E")
        axs[i].set_ylabel("Drift magnitude")
        # axs[i].legend(frameon=False)
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")

    # fit a line to the log-log data for each subplot
    for i in range(len(drift_mag_list)):
        x = var_ef_list[i]
        y = drift_mag_list[i][-1]
        log_x = np.log(x)
        log_y = np.log(y)
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
        x_fit = np.linspace(np.min(x), np.max(x), 100)
        y_fit = np.exp(intercept) * x_fit ** slope
        axs[i].plot(x_fit, y_fit, color='red', label=f"Slope: {slope:.2f}")
        axs[i].legend(frameon=False)
    plt.tight_layout()
    if save_loc is not None:
        plt.savefig(save_loc + "drift_vs_var_ef_comparison.png", dpi=300)

    return None

def animation_tuning_curve_comparison_single_cell(tuning_curves_list, POs, labels, cell_idx=200, plot_initial=False, save_loc=None):

    theta_list = np.linspace(0, 180, 500, endpoint=False)

    fig, ax = plt.subplots(figsize=(4, 3.5), dpi=300)
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 3)
    ax.set_xlabel("Theta (degrees)")
    ax.set_ylabel("Firing rate")
    ax.set_title("Tuning curve comparison")

    if plot_initial:
        for i in range(len(labels)):
            tuning_curve = tuning_curves_list[i][0, cell_idx, :]
            ax.plot(theta_list, tuning_curve, alpha=0.5, color=f'C{i}')
        fig.tight_layout()

    lines = []
    for i in range(len(labels)):
        line, = ax.plot([], [], label=labels[i], color=f'C{i}')
        lines.append(line)
    ax.legend(frameon=False)
    fig.suptitle(f"Tuning curves for cell {cell_idx}", y=1.05)
    fig.tight_layout()

    points = []
    for i in range(len(labels)):
        point = ax.scatter([], [], color=f'C{i}')
        points.append(point)

    def init():
        for line in lines:
            line.set_data([], [])
        for point in points:
            point.set_offsets(np.empty((0, 2)))
        return [*lines, *points]

    def animate(day):
        for i in range(len(labels)):
            tuning_curve = tuning_curves_list[i][day, cell_idx, :]
            lines[i].set_data(theta_list, tuning_curve)
            points[i].set_offsets([[POs[i][day+1, cell_idx], tuning_curve.max()]])
        ax.set_title(f"Day {day}")

        return [*lines, *points]

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=tuning_curves_list[0].shape[0], interval=500, blit=True)
    if save_loc is not None:
        anim.save(save_loc + f"tuning_curve_comparison_cell-{cell_idx}.gif", writer='imagemagick')


def plot_POs_initial_vs_final_comparison(POs_list, labels, save_loc=None):

    fig, axs = plt.subplots(1, len(labels), figsize=(10, 3.5), dpi=300, sharex=True, sharey=True)
    for i in range(len(labels)):
        axs[i].scatter(POs_list[i][0], POs_list[i][-1], alpha=0.5, color=f'C{i}')
        axs[i].set_title(f"{labels[i]}")
        axs[i].set_xlabel("Initial PO (degrees)")
        axs[i].set_ylabel("Final PO (degrees)")
    
    fig.suptitle("Initial vs final POs", y=1.05)
    plt.tight_layout()

    if save_loc is not None:
        plt.savefig(save_loc + "POs_initial_vs_final_comparison.png", dpi=300)

    return None


def plot_POs_spread_comparison(POs_list, labels, save_loc=None):

    fig, axs = plt.subplots(1, len(labels), figsize=(12, 3.5), dpi=200, sharex=True, sharey=True)
    for i in range(len(labels)):
        for cell_idx in range(POs_list[i].shape[1]):
            axs[i].plot(POs_list[i][:, cell_idx] - POs_list[i][0, cell_idx], alpha=0.3, color=f'C{i}')
        axs[i].set_title(f"{labels[i]}")
        axs[i].set_xlabel("Day")
        axs[i].set_ylabel("Change in PO (degrees)")
        axs[i].set_ylim([-180, 180])
    fig.suptitle("Spread of POs over time", y=1.05)
    plt.tight_layout()
    if save_loc is not None:
        plt.savefig(save_loc + "POs_spread_comparison.png", dpi=300)
    
    return None