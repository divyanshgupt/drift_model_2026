import sys, os
sys.path.append("/home/go76jav/drift_model_2026/src")
import numpy as np, h5py
from baseline_network import BaselineNetwork
from network import FeedForward
from helper_functions import circular_gaussian

OUT = "/home/go76jav/drift_model_2026/analysis/_tmp/"
ROOT = "/home/go76jav/drift_model_2026/results/08-25 - new - Hebb vs stochastic parameterization/1A - FF - cotuned"

print("="*78)
print("I. IS THE SATURATION CONFINEMENT, OR TRUE RANDOMISATION?")
print("="*78)
print("  ceiling for uniformly-randomised PO on [0,180) = 45 deg\n")
for h_, e_ in [(1.58, 0.26), (1.58, 1.58), (1.58, 3.16), (1.58, 5.00)]:
    p = f"{ROOT}/data/seed_1/hebb_{h_:.1f}_eta_{e_:.1f}/baseline/results.hdf5"
    if not os.path.exists(p): print("missing", p); continue
    with h5py.File(p) as f:
        PO = f["POs"][:]; dm = f["drift_mag"][:]
    p0, p1 = PO[0], PO[-1]
    # circular correlation on doubled angles
    z0, z1 = np.exp(2j*np.deg2rad(p0)), np.exp(2j*np.deg2rad(p1))
    r = np.abs(np.mean(z1*np.conj(z0)))
    print(f"  K_H={h_:.2f} K_eta={e_:.2f}: final drift {np.nanmean(dm[-1]):5.2f} deg  "
          f"| circ.corr(PO_0, PO_100) = {r:.3f}  | PO range {p1.min():.0f}-{p1.max():.0f}")
print("\n  high circ.corr with drift << 45 deg  =>  confined diffusion, not randomisation")

print()
print("="*78)
print("II. RATE STATISTICS ACROSS THE SIX SCENARIOS (baseline, no CNO)")
print("="*78)

def stats_rec(**kw):
    n = BaselineNetwork(E_to_E="on", E_to_I="on", I_to_I="on", set_seed=True, seed=1,
                        train_sigma=25, inh_mod_type="hyperpolarizing", inh_input_scale=0.0,
                        hebb_scaling=0.6, rand_scaling=0.5, n_days=3, if_pre_run=False,
                        save_location=OUT, **kw)
    rF = np.array([circular_gaussian(n.N, a, amp=0.62, sigma=25)
                   for a in np.linspace(0, 180, 60, endpoint=False)])
    rE, rI = n._settle_batch(rF, n.w_ef, n.w_ee, n.w_ei, n.w_if, n.w_ie, n.w_ii, 1.0)
    return rE, rI, n

def stats_ff(**kw):
    n = FeedForward(inh='on', inh_mod_type="hyperpolarizing", inh_input_scale=0,
                    hebb_scaling=0.6, rand_scaling=0.5, n_days=3, seed=1, pre_run=False, **kw)
    rE, rI = [], []
    for a in np.linspace(0, 180, 60, endpoint=False):
        rf = n.circular_gaussian(n.N, a, amp=0.62, sigma=25)
        ri = np.maximum(0, n.w_if.T@rf + n.inh_input)
        re = np.maximum(0, n.w_ef_baseline.T@rf - n.w_ei.T@ri)
        rE.append(re); rI.append(ri)
    return np.array(rE), np.array(rI), n

rows = []
for name, fn, kw in [
    ("1A  FF cotuned",            stats_ff,  dict(inh_type='co-tuned')),
    ("1B  FF random(blanket)",    stats_ff,  dict(inh_type='blanket')),
    ("2A  rec cotuned",           stats_rec, dict(inh_type='co-tuned')),
    ("2B  rec random",            stats_rec, dict(inh_type='random')),
    ("3A  rec cotuned +F->I pl",  stats_rec, dict(inh_type='co-tuned', plasticity_F_to_I='on')),
    ("3B  rec random  +F->I pl",  stats_rec, dict(inh_type='random',  plasticity_F_to_I='on')),
]:
    rE, rI, n = fn(**kw)
    rows.append((name, rE.mean(), np.mean(rE > 0), rI.mean(), np.mean(rI > 0),
                 rI.mean()/max(rE.mean(), 1e-9), rE.max()))

print(f"{'scenario':26s} {'mean rE':>8s} {'act.E':>6s} {'mean rI':>8s} {'act.I':>6s} {'rI/rE':>7s} {'max rE':>7s}")
for r in rows:
    print(f"{r[0]:26s} {r[1]:8.4f} {r[2]:6.3f} {r[3]:8.4f} {r[4]:6.3f} {r[5]:7.2f} {r[6]:7.3f}")
print("\n  1A/1B vs 2A/2B differ ~5x in mean rE at nominally identical settings.")

print()
print("="*78)
print("III. CALIBRATING A SHARED OPERATING POINT")
print("="*78)
print("  Two knobs suffice: g_ee (E->E loop gain) and g_ei (I->E strength).")
print("  Targets here: mean rE = 0.20 and fraction of active E cells = 0.50\n")

def rec_rates(g_ee, g_ei, inh_type='co-tuned', beta=0.0):
    n = BaselineNetwork(inh_type=inh_type, E_to_E="on", E_to_I="on", I_to_I="on",
                        set_seed=True, seed=1, train_sigma=25,
                        inh_mod_type="hyperpolarizing", inh_input_scale=0.0,
                        hebb_scaling=0.6, rand_scaling=0.5, n_days=3, if_pre_run=False,
                        save_location=OUT)
    n.w_ee = n.w_ee*g_ee; n.w_ei = n.w_ei*g_ei
    n.inh_input = -beta*np.ones(n.N_inh)
    rF = np.array([circular_gaussian(n.N, a, amp=0.62, sigma=25)
                   for a in np.linspace(0, 180, 60, endpoint=False)])
    rE, rI = n._settle_batch(rF, n.w_ef, n.w_ee, n.w_ei, n.w_if, n.w_ie, n.w_ii, 1.0)
    return rE.mean(), np.mean(rE > 0)

# coarse 2-D solve on a small grid, then report the closest
best = None
for g_ee in np.linspace(0.0, 0.9, 10):
    for g_ei in np.linspace(0.2, 2.5, 24):
        m, a = rec_rates(g_ee, g_ei)
        err = ((m-0.20)/0.20)**2 + ((a-0.50)/0.50)**2
        if best is None or err < best[0]: best = (err, g_ee, g_ei, m, a)
print(f"  recurrent, co-tuned  -> g_ee={best[1]:.2f}, g_ei={best[2]:.2f}"
      f"   gives mean rE={best[3]:.4f}, active={best[4]:.3f}")
print(f"  (current setting is g_ee=1.00, g_ei=1.00 -> mean rE=0.2270, active=0.556)")

print()
print("  Matched-effect beta: solve for 'beta that raises mean rE by 25%'")
for tag, it in [("recurrent co-tuned", 'co-tuned')]:
    base, _ = rec_rates(best[1], best[2], it, beta=0.0)
    lo, hi = 0.0, 3.0
    for _ in range(22):
        mid = (lo+hi)/2
        m, _ = rec_rates(best[1], best[2], it, beta=mid)
        if m < 1.25*base: lo = mid
        else: hi = mid
    m, a = rec_rates(best[1], best[2], it, beta=(lo+hi)/2)
    print(f"    {tag:20s} beta = {(lo+hi)/2:.4f}  -> mean rE {base:.4f} -> {m:.4f}  (+{100*(m/base-1):.0f}%)")
print(f"    current hard-coded beta = 0.75 raises mean rE by +216% (0.227 -> 0.718)")
