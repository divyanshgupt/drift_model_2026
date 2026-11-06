import sys, os
sys.path.append("/home/go76jav/drift_model_2026/src")
import numpy as np, h5py
from network import FeedForward

ROOT = "/home/go76jav/drift_model_2026/results/08-25 - new - Hebb vs stochastic parameterization/1A - FF - cotuned"

def po_vec(w, ff, n_ang=720):
    """population-vector PO of every column of w"""
    th = np.linspace(0, 180, n_ang, endpoint=False)
    R = np.array([ff.circular_gaussian(ff.N, t, amp=0.62, sigma=25) for t in th])
    rI = np.maximum(0, R @ ff.w_if + ff.inh_input)
    rE = np.maximum(0, R @ w - rI @ ff.w_ei)
    z = (rE * np.exp(2j*np.deg2rad(th))[:, None]).sum(0) / (rE.sum(0) + 1e-12)
    po = np.rad2deg(np.angle(z))/2
    return np.mod(po, 180)

print("="*76)
print("A. IS THERE A RESTORING FORCE ON THE PO?  (deterministic, K_eta = 0)")
print("="*76)
print("   translate one equilibrated column by delta, run noise-free dynamics,")
print("   and see whether its PO returns toward where it started.\n")

ff = FeedForward(inh='on', inh_type='co-tuned', inh_mod_type='hyperpolarizing',
                 inh_input_scale=0, hebb_scaling=0.6, rand_scaling=0.5, seed=1)
w_eq = ff.w_ef_baseline.copy()
N = ff.N
shift_bins = 20                      # 20 * 180/500 = 7.2 deg
j = N//2

for label, wstart in [("undisplaced", w_eq.copy()),
                      ("displaced +7.2 deg", None)]:
    w = w_eq.copy()
    if wstart is None:
        w[:, j] = np.roll(w_eq[:, j], shift_bins)
    po0 = po_vec(w, ff)[j]
    traj = [po0]
    ffd = FeedForward(inh='on', inh_type='co-tuned', inh_mod_type='hyperpolarizing',
                      inh_input_scale=0, hebb_scaling=0.6, rand_scaling=0.0, seed=1)
    ffd.w_if, ffd.w_ei, ffd.inh_input = ff.w_if, ff.w_ei, ff.inh_input
    for t in range(900):
        H, rf, re = ffd.hebbian_component(N, w, ffd.w_if, ffd.w_ei, 90, 'baseline')
        w = w + (ffd.hebb_scaling * H * ffd.propensity(w, ffd.a)) * ffd.learning_rate
        if t % 30 == 0:
            w = w/(w.sum(0) + 1e-10)
            traj.append(po_vec(w, ffd)[j])
    traj = np.array(traj)
    print(f"  {label:20s} PO: start {traj[0]:7.2f}  ->  after 900 steps {traj[-1]:7.2f}"
          f"   (net move {traj[-1]-traj[0]:+.2f} deg)")

eq_po = po_vec(w_eq, ff)[j]
print(f"\n  undisplaced equilibrium PO = {eq_po:.2f} deg")
print("  if the displaced run returns toward the undisplaced value -> restoring force (OU)")
print("  if it stays put                                          -> free zero mode")

print()
print("="*76)
print("B. DOES drift(t) FIT AN OU CURVE BETTER THAN A POWER LAW?")
print("="*76)
print("   OU:        drift(t) = M * sqrt(1 - exp(-2*k*t))     (plateaus at M)")
print("   power law: drift(t) = A * t**p\n")
from scipy.optimize import curve_fit

def ou(t, M, k):  return M*np.sqrt(np.maximum(1e-12, 1-np.exp(-2*k*t)))
def pl(t, A, p):  return A*t**p

print(f"  {'K_H':>5s} {'K_eta':>6s} | {'OU M':>7s} {'OU k':>8s} {'OU rms':>8s} | {'PL p':>6s} {'PL rms':>8s} | better")
for h_ in [1.58]:
    for e_ in [0.26, 0.79, 1.58, 3.16, 5.00]:
        p_ = f"{ROOT}/data/seed_1/hebb_{h_:.1f}_eta_{e_:.1f}/baseline/results.hdf5"
        if not os.path.exists(p_): continue
        with h5py.File(p_) as f: dm = np.nanmean(f["drift_mag"][:], axis=1)
        t = np.arange(1, len(dm)); y = dm[1:]
        try:
            po_, _ = curve_fit(ou, t, y, p0=[y[-1]*1.2, .05], maxfev=20000)
            r_ou = np.sqrt(np.mean((ou(t, *po_)-y)**2))
        except Exception: po_, r_ou = (np.nan, np.nan), np.inf
        try:
            pp, _ = curve_fit(pl, t, y, p0=[y[0], .5], maxfev=20000)
            r_pl = np.sqrt(np.mean((pl(t, *pp)-y)**2))
        except Exception: pp, r_pl = (np.nan, np.nan), np.inf
        best = "OU" if r_ou < r_pl else "power law"
        print(f"  {h_:5.2f} {e_:6.2f} | {po_[0]:7.2f} {po_[1]:8.4f} {r_ou:8.4f} | "
              f"{pp[1]:6.3f} {r_pl:8.4f} | {best}")

print()
print("="*76)
print("C. CHECK THE ANALYTIC CONSTANTS")
print("="*76)
A_, sig, L = 0.62, 25.0, 180.0
mu_pred = A_*sig*np.sqrt(2*np.pi)/L
th = np.linspace(0, 180, 4000, endpoint=False)
R = np.array([ff.circular_gaussian(ff.N, t, amp=A_, sigma=sig) for t in th])
print(f"  mu_F   predicted A*sigma*sqrt(2pi)/L = {mu_pred:.5f}   measured = {R.mean():.5f}")
C_meas = (R.T @ R)/len(th)
u = np.arange(ff.N)*L/ff.N; u = np.minimum(u, L-u)
C_pred = (A_**2*sig*np.sqrt(np.pi)/L)*np.exp(-u**2/(4*sig**2))
k = ff.N//2
print(f"  C(0)   predicted A^2*sigma*sqrt(pi)/L = {C_pred[0]:.5f}   measured = {C_meas[k,k]:.5f}")
print(f"  C(18deg)                  predicted = {C_pred[50]:.5f}   measured = {C_meas[k,k+50]:.5f}")
print(f"  C width = sigma*sqrt(2)             = {sig*np.sqrt(2):.2f} deg")
