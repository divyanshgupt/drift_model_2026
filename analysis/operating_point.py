import sys, os
sys.path.append("/home/go76jav/drift_model_2026/src")
import numpy as np
from baseline_network import BaselineNetwork
from network import FeedForward
from helper_functions import circular_gaussian

np.set_printoptions(precision=4, suppress=True)

print("="*70)
print("A. RECURRENT NET OPERATING REGIME (2A config)")
print("="*70)
net = BaselineNetwork(inh_type="co-tuned", E_to_E="on", E_to_I="on", I_to_I="on",
                      norm=True, set_seed=True, seed=1, inh_scale=1.0, train_sigma=25,
                      inh_mod_type="hyperpolarizing", inh_input_scale=0.0,
                      hebb_scaling=0.6, rand_scaling=0.5, n_days=3,
                      if_pre_run=False,
                      save_location="/home/go76jav/drift_model_2026/analysis/_tmp/")

print("col sums w_ee :", net.w_ee.sum(0)[:4], "  spec.radius(w_ee) =", np.abs(np.linalg.eigvals(net.w_ee)).max())
print("col sums w_ei :", net.w_ei.sum(0)[:4])
print("col sums w_ie :", net.w_ie.sum(0)[:4])
print("col sums w_ii :", net.w_ii.sum(0)[:4], " spec.radius(w_ii) =", np.abs(np.linalg.eigvals(net.w_ii)).max())
print("col sums w_if :", net.w_if.sum(0)[:4])

# Full linear Jacobian of the (E,I) system assuming all units above threshold
N, M = net.N, net.N_inh
J = np.zeros((N+M, N+M))
J[:N, :N] = net.w_ee.T
J[:N, N:] = -net.w_ei.T
J[N:, :N] = net.w_ie.T
J[N:, N:] = -net.w_ii.T
ev = np.linalg.eigvals(J)
print("\nfull linearised recurrent matrix J (all-active):")
print("   max Re(eig) =", ev.real.max(), "  spectral radius =", np.abs(ev).max())
print("   -> stability of rate dynamics needs max Re(eig(J)) < 1 (since dU = (-U + J r)/tau)")

# E-only loop gain (I clamped)
print("   E-subnetwork alone spectral radius (w_ee.T) =", np.abs(np.linalg.eigvals(net.w_ee.T)).max())

print("\n--- steady-state rates, baseline vs CNO ---")
for scale, name in [(0.0, "baseline"), (1.0, "CNO")]:
    n2 = BaselineNetwork(inh_type="co-tuned", E_to_E="on", E_to_I="on", I_to_I="on",
                         norm=True, set_seed=True, seed=1, inh_scale=1.0, train_sigma=25,
                         inh_mod_type="hyperpolarizing", inh_input_scale=scale,
                         hebb_scaling=0.6, rand_scaling=0.5, n_days=3, if_pre_run=False,
                         save_location="/home/go76jav/drift_model_2026/analysis/_tmp/")
    rF = np.array([circular_gaussian(n2.N, a, amp=0.62, sigma=25) for a in np.linspace(0,180,50,endpoint=False)])
    rE, rI = n2._settle_batch(rF, n2.w_ef, n2.w_ee, n2.w_ei, n2.w_if, n2.w_ie, n2.w_ii, 1.0)
    print(f"{name:9s} inh_input={n2.inh_input[0]:+.3f} | r_F max {rF.max():.3f} | "
          f"r_E mean {rE.mean():.3f} max {rE.max():.3f} | r_I mean {rI.mean():.3f} max {rI.max():.3f} | "
          f"frac I silent {np.mean(rI==0):.3f} | frac E silent {np.mean(rE==0):.3f}")
    # F->I drive alone, for comparison with inh_input magnitude
    print(f"           raw w_if^T r_F : mean {(rF@n2.w_if).mean():.4f}  max {(rF@n2.w_if).max():.4f}")

print()
print("="*70)
print("B. FEEDFORWARD NET (1A config)")
print("="*70)
for scale, name in [(0, "baseline"), (1, "CNO")]:
    ff = FeedForward(inh='on', inh_type='co-tuned', inh_mod_type="hyperpolarizing",
                     inh_input_scale=scale, hebb_scaling=0.6, rand_scaling=0.5,
                     n_days=5, seed=1, pre_run=False)
    rf = ff.circular_gaussian(ff.N, 90., amp=0.62, sigma=25)
    ri = np.maximum(0, ff.w_if.T @ rf + ff.inh_input)
    re = np.maximum(0, ff.w_ef_baseline.T @ rf - ff.w_ei.T @ ri)
    print(f"{name:9s} inh_input={ff.inh_input[0]:+.3f} | raw w_if^T r_F mean {(ff.w_if.T@rf).mean():.4f} "
          f"max {(ff.w_if.T@rf).max():.4f} | r_I mean {ri.mean():.4f} | frac I silent {np.mean(ri==0):.3f} "
          f"| r_E mean {re.mean():.4f} | frac E silent {np.mean(re==0):.3f}")

print()
print("="*70)
print("C. DO PLASTIC WEIGHTS GO NEGATIVE?  (FF, short run)")
print("="*70)
ff = FeedForward(inh='on', inh_type='co-tuned', inh_mod_type="hyperpolarizing",
                 inh_input_scale=0, hebb_scaling=0.6, rand_scaling=0.5, n_days=10, seed=1)
print("after pre_run: frac w_ef < 0 =", np.mean(ff.w_ef_baseline < 0),
      " min =", ff.w_ef_baseline.min(), " colsum range", ff.w_ef_baseline.sum(0).min(), ff.w_ef_baseline.sum(0).max())
POs = ff.get_POs_over_trials(ff.w_ef_baseline, ff.n_steps, type="baseline")
Wf = ff.W[:, :, -1]
print("after 10 days : frac w_ef < 0 =", np.mean(Wf < 0), " min =", Wf.min(),
      " colsum range", Wf.sum(0).min(), Wf.sum(0).max())
POs = np.array(POs)
print("PO grid spacing (FF) = %.2f deg (n_test_angles=%d)" % (180/ff.n_test_angles, ff.n_test_angles))
d = np.minimum(np.abs(POs[-1]-POs[0]), 180-np.abs(POs[-1]-POs[0]))
print("day-10 drift magnitude: mean %.3f deg, frac cells with exactly 0 drift = %.3f"
      % (d.mean(), np.mean(d == 0)))
print("unique drift values:", np.unique(d)[:10])
