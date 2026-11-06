import sys
sys.path.append("/home/go76jav/drift_model_2026/analysis")
sys.path.append("/home/go76jav/drift_model_2026/src")
import numpy as np
from scipy.stats import norm
from network import FeedForward
from hebbian_variance import predict, simulate, po_and_grad, NS, EPS

def exp_abs(mu, s):
    """E|X| for X ~ N(mu, s^2)"""
    s = np.maximum(s, 1e-12)
    return s*np.sqrt(2/np.pi)*np.exp(-mu**2/(2*s**2)) + mu*(1-2*norm.cdf(-mu/s))

def kernel_width(W, x):
    """circular s.d. of each weight column, deg"""
    psi = 2*np.pi*x/180
    Z = (W*np.exp(1j*psi)[:, None]).sum(0)/W.sum(0)
    R = np.abs(Z)
    return np.sqrt(-2*np.log(np.clip(R, 1e-12, 1)))*180/(2*np.pi), R

np.set_printoptions(precision=3, suppress=True)
for K_H, K_eta, pre in [(0.6, 0.0, 300), (0.6, 0.0, 3000), (1.58, 0.0, 300), (1.58, 0.0, 3000),
                        (0.6, 0.5, 300)]:
    print("="*100)
    print(f"K_H={K_H} K_eta={K_eta}  pre-run steps={pre}")
    for name, s in [("baseline", 0), ("CNO", 1)]:
        ff = FeedForward(inh='on', inh_type='co-tuned', inh_mod_type='hyperpolarizing',
                         inh_input_scale=s, hebb_scaling=K_H, rand_scaling=K_eta,
                         n_days=20, seed=1, init_steps=pre)
        x = np.linspace(0, 180, ff.N)
        W0 = ff.w_ef_baseline.copy()
        D_H, D_eta, v0, _ = predict(ff, W0, K_H, K_eta)
        traj = simulate(ff, W0, K_H, K_eta, 20, seed=101)
        d20 = traj[-1]
        # re-evaluate systematic velocity from the weights at day 20
        # (re-simulate cheaply: last weights are not returned, so estimate v from late slope)
        late_v = (traj[-1]-traj[10])/10
        early_v = (traj[10]-traj[0])/10
        pred_abs = exp_abs(v0*20, np.sqrt(2*(D_H+D_eta)*20))
        wid0, R0 = kernel_width(W0, x)
        big = np.abs(d20) > np.percentile(np.abs(d20), 98)
        print(f" {name:8s} |d20| mean {np.abs(d20).mean():6.3f}  median {np.median(np.abs(d20)):6.3f}  "
              f"p98 {np.percentile(np.abs(d20),98):6.2f}  max {np.abs(d20).max():6.2f} | "
              f"pred E|d20| mean {pred_abs.mean():6.3f} median {np.median(pred_abs):6.3f}")
        print(f"          median ratio meas/pred {np.median(np.abs(d20))/np.median(pred_abs):5.2f} | "
              f"corr(|d20|, pred) {np.corrcoef(np.abs(d20), pred_abs)[0,1]:5.2f} | "
              f"corr(early v, late v) {np.corrcoef(early_v, late_v)[0,1]:5.2f} | "
              f"corr(v0, early slope) {np.corrcoef(v0, early_v)[0,1]:5.2f}")
        print(f"          top-2% movers: R0 {R0[big].mean():.3f} vs rest {R0[~big].mean():.3f}; "
              f"width0 {wid0[big].mean():5.1f} vs {wid0[~big].mean():5.1f} deg; "
              f"sign-consistent motion (early&late same sign) {np.mean(np.sign(early_v[big])==np.sign(late_v[big])):.2f}")
