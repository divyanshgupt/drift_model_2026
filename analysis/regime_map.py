import os, glob, json
import numpy as np, h5py

ROOT = "/home/go76jav/drift_model_2026/results/08-25 - new - Hebb vs stochastic parameterization/1A - FF - cotuned"
hk = np.linspace(0, 5, 20); ek = np.linspace(0, 5, 20)

def fwhm(tc, x):
    """circular FWHM of each row of tc (n_cells, n_angles)"""
    out = np.full(tc.shape[0], np.nan)
    c = tc.shape[1]//2
    for i, y in enumerate(tc):
        if y.max() <= 0: continue
        y = np.roll(y, c - np.argmax(y))
        idx = np.where(y >= y.max()/2)[0]
        if len(idx) > 1: out[i] = (idx[-1]-idx[0]) * (x[1]-x[0])
    return out

n_h, n_e = len(hk), len(ek)
W0 = np.full((n_h, n_e), np.nan)   # day-0 median FWHM
W1 = np.full((n_h, n_e), np.nan)   # final-day median FWHM
DR = np.full((n_h, n_e), np.nan)   # drift rate, deg/day (late)
SL = np.full((n_h, n_e), np.nan)   # log-log exponent
x = np.linspace(0, 180, 100, endpoint=False)

for i, h_ in enumerate(hk):
    for j, e_ in enumerate(ek):
        p = f"{ROOT}/data/seed_1/hebb_{h_:.1f}_eta_{e_:.1f}/baseline/results.hdf5"
        if not os.path.exists(p): continue
        with h5py.File(p) as f:
            dm = f["drift_mag"][:]
            t0 = f["tuning_curves_over_days"][0]
            t1 = f["tuning_curves_over_days"][-1]
        W0[i, j] = np.nanmedian(fwhm(t0, x))
        W1[i, j] = np.nanmedian(fwhm(t1, x))
        m = np.nanmean(dm, axis=1)
        DR[i, j] = m[-1] - m[-2]
        d = np.arange(1, len(m))
        ok = m[1:] > 1e-9
        if ok.sum() > 5:
            SL[i, j] = np.polyfit(np.log(d[ok]), np.log(m[1:][ok]), 1)[0]

np.set_printoptions(precision=2, suppress=True, linewidth=250)
cov = np.mean(~np.isnan(W0))
print(f"grid coverage (seed 1, baseline): {cov:.2f}\n")

step = 3
def show(A, name, fmt="{:6.1f}"):
    print(f"--- {name} ---")
    print("        K_eta:", " ".join(f"{v:6.1f}" for v in ek[::step]))
    for i in range(0, n_h, step):
        print(f"K_H={hk[i]:4.1f}  ", " ".join(
            (fmt.format(v) if np.isfinite(v) else "     .") for v in A[i, ::step]))
    print()

show(W0, "median tuning FWHM at day 0 (deg)")
show(W1, "median tuning FWHM at day 100 (deg)")
show(W1 - W0, "FWHM change over 100 days (deg)  <- want ~0", "{:+6.1f}")
show(DR, "drift rate on final day (deg/day)", "{:6.2f}")
show(SL, "log-log exponent  <- want 0.5", "{:6.2f}")

# where is the self-consistency locus (width neither sharpens nor broadens)?
d = W1 - W0
print("cells with |FWHM change| < 2 deg over 100 days:")
for i in range(n_h):
    for j in range(n_e):
        if np.isfinite(d[i, j]) and abs(d[i, j]) < 2:
            print(f"   K_H={hk[i]:.2f}  K_eta={ek[j]:.2f}   FWHM {W0[i,j]:.1f}->{W1[i,j]:.1f} deg"
                  f"   drift/day {DR[i,j]:.3f}  exponent {SL[i,j]:.2f}")
