import sys
sys.path.append("/home/go76jav/drift_model_2026/src")
import numpy as np
from network import FeedForward
from baseline_network import BaselineNetwork
from helper_functions import circular_gaussian

OUT = "/home/go76jav/drift_model_2026/analysis/_tmp/"

print("="*70); print("D. FROZEN SYNAPSES: exact zeros in the weight matrix")
ff = FeedForward(inh='on', inh_type='co-tuned', inh_mod_type="hyperpolarizing",
                 inh_input_scale=0, hebb_scaling=0.6, rand_scaling=0.5, n_days=5, seed=1,
                 pre_run=False)
w0 = ff.w_ef_init
print("FF  w_ef_init: frac exactly 0 = %.4f ; frac < 1e-12 = %.4f" % (np.mean(w0==0), np.mean(w0<1e-12)))
print("    tanh(10*w) at those entries =", np.tanh(10*w0[w0==0][:3]) if (w0==0).any() else "n/a")
# per-column support width in degrees
supp = (w0 > 0).sum(0) * 180.0/ff.N
print("    per-column non-zero support: median %.1f deg, min %.1f, max %.1f" %
      (np.median(supp), supp.min(), supp.max()))

net = BaselineNetwork(inh_type="co-tuned", E_to_E="on", E_to_I="on", I_to_I="on",
                      set_seed=True, seed=1, train_sigma=25, inh_mod_type="hyperpolarizing",
                      inh_input_scale=0.0, hebb_scaling=0.6, rand_scaling=0.5, n_days=3,
                      if_pre_run=False, save_location=OUT)
we = net.w_ef
supp = (we > 0).sum(0) * 180.0/net.N
print("REC w_ef     : frac exactly 0 = %.4f ; per-column support: median %.1f deg, min %.1f, max %.1f"
      % (np.mean(we==0), np.median(supp), supp.min(), supp.max()))

print()
print("="*70); print("E. PO estimator: argmax vs vector-average (recurrent net, day 0)")
for meth in ["argmax", "circular_mean"]:
    try:
        po = net.get_preferred_orientations(method=meth)
        print("  %-14s ok, n valid %d, first 5: %s" % (meth, np.sum(~np.isnan(po)), np.round(po[:5],3)))
    except Exception as e:
        print("  %-14s FAILED: %s: %s" % (meth, type(e).__name__, e))

print()
print("="*70); print("F. selection threshold differs between conditions?")
for scale, name in [(0.0,"baseline"), (1.0,"CNO")]:
    n2 = BaselineNetwork(inh_type="co-tuned", E_to_E="on", E_to_I="on", I_to_I="on",
                         set_seed=True, seed=1, train_sigma=25, inh_mod_type="hyperpolarizing",
                         inh_input_scale=scale, hebb_scaling=0.6, rand_scaling=0.5, n_days=3,
                         if_pre_run=False, save_location=OUT)
    rF = np.array([circular_gaussian(n2.N, a, amp=0.62, sigma=n2.probe_sigma) for a in
                   np.linspace(0,180,n2.n_test_angles,endpoint=False)])
    rE,_ = n2._settle_batch(rF, n2.w_ef, n2.w_ee, n2.w_ei, n2.w_if, n2.w_ie, n2.w_ii, 1.0)
    tc = rE.T; peak = tc.max(1); thr = 0.05*peak.max()
    print("  %-9s abs threshold used = %.4f ; n cells kept = %d/%d" %
          (name, thr, np.sum(peak>thr), n2.N))

print()
print("="*70); print("G. analytic check: is CNO equivalent to a DC term in the Hebbian drive?")
# theory: <r_E,j r_F> = C_FF w - C_FI v_j ; with r_I = w_if^T r_F + b (linear), b = -beta*1,
# and 1^T v_j = 1 (column-normalised w_ei) ->  extra term = + beta * mean(r_F) * 1
ff2 = FeedForward(inh='on', inh_type='co-tuned', inh_mod_type="hyperpolarizing",
                  inh_input_scale=0, hebb_scaling=0.6, rand_scaling=0.5, n_days=5, seed=1, pre_run=False)
thetas = np.linspace(0,180,2000,endpoint=False)
RF = np.array([ff2.circular_gaussian(ff2.N, t, amp=0.62, sigma=25) for t in thetas])
muF = RF.mean()
beta = 0.1
print("  mean(r_F) over stimuli   = %.4f" % muF)
print("  predicted DC Hebbian bump per synapse, beta*mu_F = %.4f" % (beta*muF))
# tuned part magnitude for comparison
CFF = RF.T@RF/len(thetas)
w = ff2.w_ef_init[:, ff2.N//2]; v = ff2.w_ei[:, ff2.N//2]
tuned = CFF@(w - ff2.w_if@v)
print("  tuned Hebbian drive C_FF(w - w_if v): mean %.4f, max %.4f, std %.4f" %
      (tuned.mean(), tuned.max(), tuned.std()))
print("  -> DC/tuned-std ratio = %.2f" % (beta*muF/tuned.std()))
print("  1^T v_j (should be 1 by column norm) = %.4f" % v.sum())

print()
print("="*70); print("H. shape check for 08-25 load_and_plot.load_data")
print("  FeedForward  drift_mag shape -> (n_days,   N)")
print("  BaselineNet  drift_mag shape -> (n_days+1, N)   <-- load_data allocates (n_days, N)")
net2 = BaselineNetwork(inh_type="co-tuned", E_to_E="on", E_to_I="on", I_to_I="on",
                       set_seed=True, seed=1, train_sigma=25, inh_mod_type="hyperpolarizing",
                       inh_input_scale=0.0, hebb_scaling=0.6, rand_scaling=0.5, n_days=2,
                       if_pre_run=False, n_pre_run_stimuli=2, save_location=OUT)
net2.run()
dm, dr, cv = net2.get_drift_metrics()
print("  measured (n_days=2): drift_mag %s, drift_rate %s, len(POs) %d" % (dm.shape, dr.shape, len(net2.POs)))
