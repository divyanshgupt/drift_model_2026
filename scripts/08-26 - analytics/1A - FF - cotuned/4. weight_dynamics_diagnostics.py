# Diagnostics for the F->E weight matrix over time.
#
# Answers three questions that the PO-based drift metrics cannot:
#   A. Is a single synapse oscillating, random-walking, or dying?
#   B. Does the weight vector of one cell keep moving, or does it settle?
#   C. Is the "drift" we measure real motion, or measurement jitter?
#
# Note on the learning rule: dw = lr * (K_H*H + K_eta*eta) * tanh(a*w), and with
# a*w ~ 0.02 the propensity is linear, so dw ~ w.  The dynamics are therefore
# MULTIPLICATIVE: log|w| performs a random walk with per-step variance
# (lr*a*K_eta)^2, whose median decays at -(lr*a*K_eta)^2 / 2 per step.  Panels
# C/D below measure that decay directly - it is what kills the network.

import json
import os
import sys

import h5py
import numpy as np
from matplotlib import pyplot as plt

sys.path.append("../../../src/")

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["font.size"] = 11
plt.rcParams["lines.linewidth"] = 1.5


def circular_dist(x, y):
    return np.minimum(np.abs(x - y), 180 - np.abs(x - y))


def circular_com(w, N):
    """Circular centre of mass (in deg, 0-180) of a weight vector over pre-synaptic index.
    Rectified, because negative weights have no meaningful 'position'."""
    ang = 2 * np.linspace(0, np.pi, N, endpoint=False)
    wp = np.maximum(w, 0)
    z = (wp * np.exp(1j * ang)[:, None]).sum(axis=0)
    return np.degrees(np.angle(z)) / 2 % 180


def participation_ratio(w):
    """Effective number of synapses carrying a column: (sum w^2)^-1 * (sum w)^2 form.
    PR = N for a flat column, ~1 when one synapse carries everything."""
    wp = np.maximum(w, 0)
    return wp.sum(axis=0) ** 2 / (np.sum(wp**2, axis=0) + 1e-300)


def load(save_loc, cell_idx, stride):
    """W is stored as (pre, post, time) and is several GB - never do f['W'][:].
    One hyperslab read for the population stats, one row read for the single cell."""
    with h5py.File(save_loc + "results.hdf5", "r") as f:
        n_pre, n_post, n_t = f["W"].shape
        Wsub = f["W"][:, :, ::stride]          # (pre, post, n_snap) - snapshots
        row_full = f["W"][:, cell_idx, :]      # (pre, time)  - every timestep, one cell
    return Wsub, row_full, n_t


if __name__ == "__main__":

    save_loc = ("../../../results/08-26 - analytics/1A - FF - cotuned/"
                "3. weights_over_time/hebb_0.3_eta_1.0_inh_input_0/")

    with open(save_loc + "hyperparameters.json") as f:
        hp = json.load(f)
    N = hp["N"]
    nspn = hp["n_steps_per_norm"]          # steps per normalisation == steps per day here
    n_days = hp["n_days"]
    lr, a, K_eta = hp["learning_rate"], hp["a"], hp["rand_scaling"]

    cell_idx = N // 2
    Wsub, row_full, n_t = load(save_loc, cell_idx, stride=nspn)
    n_snap = Wsub.shape[2]
    days = np.arange(n_snap)
    print(f"W: (pre={N}, post={N}, t={n_t});  snapshots every {nspn} steps -> {n_snap}")

    row_day = row_full[:, ::nspn]                 # (pre, days) - fixed phase of the norm cycle
    plots = save_loc + "diagnostics/"
    os.makedirs(plots, exist_ok=True)

    # ---------------------------------------------------------------- figure 1
    # A single cell: what do its incoming weights actually do?
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5))

    # A. heatmap of the row over time, log scale - the ONLY scale that shows the
    #    dynamic range these weights develop.
    ax = axs[0, 0]
    m = np.log10(np.maximum(row_day, 1e-30))
    im = ax.imshow(m, origin="lower", aspect="auto", cmap="magma",
                   extent=[0, n_snap, 0, N], vmin=np.percentile(m, 2))
    fig.colorbar(im, ax=ax, label=r"$\log_{10} w$")
    ax.set(xlabel="Day", ylabel="Pre-synaptic neuron", title=f"A. Incoming weights, cell {cell_idx}")

    # B. individual synapses on a log axis. Straight-ish lines with equal scatter
    #    = geometric random walk. Flat lines = frozen. Wiggles about a level = stable.
    ax = axs[0, 1]
    strongest = np.argsort(row_day[:, 0])[-6:]
    for p in strongest:
        ax.plot(days, np.maximum(row_day[p], 1e-30), alpha=0.8, label=f"pre {p}")
    ax.set(xlabel="Day", ylabel="w (log)", yscale="log",
           title="B. Six strongest synapses onto that cell")
    ax.legend(frameon=False, fontsize=7, ncol=2)

    # C. the weight distribution collapses: quantiles of |w| in this column.
    #    The dashed line is the predicted median decay from the Ito term of the
    #    multiplicative noise; if data tracks it, drift is being driven by
    #    synapse death, not by the Hebbian term.
    ax = axs[1, 0]
    aw = np.abs(row_day)
    for q, st in [(50, "-"), (90, "--"), (99, ":")]:
        ax.plot(days, np.percentile(aw, q, axis=0), st, label=f"{q}th pct")
    ax.plot(days, np.abs(row_day).max(axis=0), "k-", label="max")
    sig2 = (lr * a * K_eta) ** 2
    ax.plot(days, np.percentile(aw, 50, axis=0)[0] * np.exp(-0.5 * sig2 * days * nspn),
            "r--", lw=1, label=r"$e^{-\sigma^2 t/2}$ prediction")
    ax.set(xlabel="Day", ylabel="|w|", yscale="log", title="C. Column weight distribution collapses")
    ax.legend(frameon=False, fontsize=8)

    # D. participation ratio: how many synapses are actually carrying the cell.
    #    This is the cleanest "has the network settled?" readout.
    ax = axs[1, 1]
    pr = participation_ratio(Wsub.reshape(N, -1)).reshape(N, n_snap)
    ax.plot(days, pr[cell_idx], "k-", label=f"cell {cell_idx}")
    ax.fill_between(days, np.percentile(pr, 25, axis=0), np.percentile(pr, 75, axis=0),
                    alpha=0.25, label="population IQR")
    ax.plot(days, np.median(pr, axis=0), label="population median")
    ax.set(xlabel="Day", ylabel="Effective # synapses (PR)", yscale="log",
           title="D. Synaptic participation ratio")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(plots + "1_single_cell_weight_dynamics.png", dpi=200)

    # ---------------------------------------------------------------- figure 2
    # Is the motion real drift, or jitter?  And is normalisation aliasing us?
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5))

    # E. the normalisation sawtooth.  Weights are renormalised every n_steps_per_norm
    #    steps, so anything sampled at the wrong phase mixes drift with the sawtooth.
    #    NB in evolve_W the norm is applied when t % nspn == 0, i.e. to W[..., t+1],
    #    while POs are read off W_old - one step BEFORE the norm, the worst phase.
    ax = axs[0, 0]
    cs = row_full.sum(axis=0)
    k = min(4 * nspn, n_t)
    ax.plot(np.arange(n_t - k, n_t), cs[-k:], "k-", lw=1)
    for t in range(n_t - k, n_t):
        if t % nspn == 0:
            ax.axvline(t, color="r", ls=":", lw=0.8)
    ax.set(xlabel="Timestep", ylabel=r"$\sum_{pre} w$",
           title="E. Column sum: normalisation sawtooth (red = norm step)")

    # F. centre of mass of the row, per timestep vs per day.  If the per-timestep
    #    trace rattles inside a band that itself does not move, you have jitter.
    ax = axs[0, 1]
    com_full = circular_com(row_full, N)
    ax.plot(np.arange(n_t) / nspn, com_full, lw=0.6, alpha=0.6, label="every timestep")
    ax.plot(days, circular_com(row_day, N), "k.-", ms=4, label="once per day")
    ax.set(xlabel="Day", ylabel="Weight COM (deg)", title="F. Centre of mass of the row")
    ax.legend(frameon=False, fontsize=8)

    # G. MSD vs lag for the population COM.  Slope 0.5 on log-log = free diffusion.
    #    A plateau = confined / settled.  A flat line from lag 1 = pure jitter,
    #    i.e. the day-99-vs-day-0 "drift" is just the single-step measurement noise.
    ax = axs[1, 0]
    com_pop = circular_com(Wsub.reshape(N, -1), N).reshape(N, n_snap)
    with h5py.File(save_loc + "results.hdf5", "r") as f:
        POs = f["POs"][:].T                                    # (cells, days)
    # POs holds one entry per day, Wsub one per day plus the t=0 snapshot
    max_lag = min(n_snap, POs.shape[1]) - 1
    lags = np.unique(np.round(np.logspace(0, np.log10(max_lag), 15)).astype(int))
    msd = np.array([circular_dist(com_pop[:, L:], com_pop[:, :-L]).mean() for L in lags])
    ax.plot(lags, msd, "o-", label="weight COM")
    msd_po = np.array([circular_dist(POs[:, L:], POs[:, :-L]).mean() for L in lags])
    ax.plot(lags, msd_po, "s-", label="preferred orientation")
    ax.plot(lags, msd[0] * np.sqrt(lags), "k--", lw=1, label=r"$\sqrt{lag}$ (free diffusion)")
    ax.set(xlabel="Lag (days)", ylabel="Mean |displacement| (deg)",
           xscale="log", yscale="log", title="G. MSD vs lag: diffusing or settled?")
    ax.legend(frameon=False, fontsize=8)

    # H. autocorrelation of the weight vector itself.  r(lag)->0 means the cell
    #    genuinely reorganises; r(lag) pinned at 1 means it froze.
    ax = axs[1, 1]
    z = row_day - row_day.mean(axis=0, keepdims=True)
    z /= np.linalg.norm(z, axis=0, keepdims=True) + 1e-300
    ac = np.array([(z[:, L:] * z[:, :-L]).sum(axis=0).mean() for L in lags])
    ax.plot(lags, ac, "o-", label=f"cell {cell_idx}")
    zz = Wsub - Wsub.mean(axis=0, keepdims=True)
    zz /= np.linalg.norm(zz, axis=0, keepdims=True) + 1e-300
    ac_pop = np.array([(zz[:, :, L:] * zz[:, :, :-L]).sum(axis=0).mean() for L in lags])
    ax.plot(lags, ac_pop, "s-", label="population mean")
    ax.axhline(0, color="k", lw=0.8)
    ax.set(xlabel="Lag (days)", ylabel="Weight-vector correlation", xscale="log",
           ylim=(-0.1, 1.05), title="H. Does the weight vector renew itself?")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(plots + "2_drift_vs_jitter.png", dpi=200)

    # ---------------------------------------------------------------- summary
    print(f"\nhebb={hp['hebb_scaling']}  eta={K_eta}  lr={lr}  a={a}")
    print(f"predicted log-median decay per day: {-0.5 * sig2 * nspn:.3f} nats")
    print(f"median |w| in column: day 0 {np.median(aw[:, 0]):.3g} "
          f"-> day {n_snap-1} {np.median(aw[:, -1]):.3g}")
    print(f"participation ratio (pop median): day 0 {np.median(pr[:, 0]):.1f} "
          f"-> day {n_snap-1} {np.median(pr[:, -1]):.1f}   (N = {N})")
    print(f"fraction of weights < 0 at end: {(Wsub[:, :, -1] < 0).mean():.3f}")
    print(f"MSD(1 day) = {msd[0]:.2f} deg,  MSD({lags[-1]} days) = {msd[-1]:.2f} deg,"
          f"  ratio {msd[-1]/msd[0]:.2f}  (sqrt law would give {np.sqrt(lags[-1]):.2f})")
    print(f"weight autocorrelation at {lags[-1]} days: {ac_pop[-1]:.3f}")
    print(f"\nsaved -> {plots}")
