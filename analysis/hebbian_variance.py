"""
Predict PO diffusion from the stimulus variance of the Hebbian term and compare
with simulation. Feedforward model, current src/network.py update rule.

  per presentation:  d(theta*)_j = eps*K_H*q_j(theta) + eps*K_eta*sum_i g_ij phi_ij xi_ij
  q_j(theta)  = r_E,j(theta) * p_j(theta),   p_j(theta) = sum_i g_ij phi(w_ij) r_F,i(theta)
  D_H   = 0.5 * n_s * eps^2 * K_H^2  * Var_theta[q_j]
  D_eta = 0.5 * n_s * eps^2 * K_eta^2 * sum_i g_ij^2 phi_ij^2
  drift velocity (systematic) = n_s * eps * K_H * <q_j>_theta   (deg/day)
"""
import sys
sys.path.append("/home/go76jav/drift_model_2026/src")
import numpy as np
from network import FeedForward

L, EPS, NS = 180.0, 0.01, 30


def po_and_grad(W, x):
    """population-vector PO of each column (deg) and exact gradient d(theta*)/dW"""
    psi = 2*np.pi*x/L
    E = np.exp(1j*psi)[:, None]                      # (N,1)
    Z = (W*E).sum(0)                                 # (N,) unnormalised resultant
    theta = np.mod(np.angle(Z)*L/(2*np.pi), L)
    G = (L/(2*np.pi))*np.imag(E/Z[None, :])          # d arg(Z)/dW_ij, exact
    return theta, G


def responses(ff, W, thetas):
    RF = np.array([ff.circular_gaussian(ff.N, t, amp=0.62, sigma=ff.input_sigma) for t in thetas])
    RI = np.maximum(0, RF @ ff.w_if + ff.inh_input)
    RE = np.maximum(0, RF @ W - RI @ ff.w_ei)
    return RF, RE


def predict(ff, W, K_H, K_eta):
    x = np.linspace(0, 180, ff.N)                    # matches circular_gaussian
    _, G = po_and_grad(W, x)
    PHI = np.tanh(ff.a*W)
    thetas = np.linspace(0, 180, 720, endpoint=False)
    RF, RE = responses(ff, W, thetas)
    P = RF @ (G*PHI)                                 # (n_theta, N)  p_j(theta)
    Q = RE*P                                         # (n_theta, N)  q_j(theta)
    D_H = 0.5*NS*EPS**2*K_H**2*Q.var(0)
    D_eta = 0.5*NS*EPS**2*K_eta**2*((G*PHI)**2).sum(0)
    v = NS*EPS*K_H*Q.mean(0)                          # deg/day, signed
    return D_H, D_eta, v, RE.mean()


def simulate(ff, W0, K_H, K_eta, n_days, seed):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 180, ff.N)
    W = W0.copy()
    th0, _ = po_and_grad(W, x)
    traj = [np.zeros(ff.N)]
    for t in range(n_days*NS):
        th = rng.uniform(0, 180)
        rf = ff.circular_gaussian(ff.N, th, amp=0.62, sigma=ff.input_sigma)
        ri = np.maximum(0, ff.w_if.T @ rf + ff.inh_input)
        re = np.maximum(0, W.T @ rf - ff.w_ei.T @ ri)
        phi = np.tanh(ff.a*W)
        W = W + EPS*(K_H*np.outer(rf, re) + K_eta*rng.standard_normal(W.shape))*phi
        if t % NS == 0:
            W = W/(W.sum(0) + 1e-10)
        if (t+1) % NS == 0:
            th_t, _ = po_and_grad(W, x)
            traj.append(((th_t - th0 + 90) % 180) - 90)
    return np.array(traj)                            # (n_days+1, N) signed displacement


def fit_D(traj):
    """MSD(t) = 2 D t through the origin, and mean signed displacement slope"""
    t = np.arange(traj.shape[0])
    msd = (traj**2).mean(1)
    D = (t*msd).sum()/(2*(t*t).sum())
    return D, msd


def run(K_H, K_eta, n_days=20, seed=1):
    out = {}
    for name, s in [("baseline", 0), ("CNO", 1)]:
        ff = FeedForward(inh='on', inh_type='co-tuned', inh_mod_type='hyperpolarizing',
                         inh_input_scale=s, hebb_scaling=K_H, rand_scaling=K_eta,
                         n_days=n_days, seed=seed)
        W0 = ff.w_ef_baseline.copy()
        D_H, D_eta, v, rE = predict(ff, W0, K_H, K_eta)
        traj = simulate(ff, W0, K_H, K_eta, n_days, seed=seed+100)
        # remove each cell's predicted systematic drift before measuring diffusion
        t = np.arange(traj.shape[0])[:, None]
        D_meas_raw, _ = fit_D(traj)
        D_meas_res, _ = fit_D(traj - v[None, :]*t)
        out[name] = dict(D_H=D_H.mean(), D_eta=D_eta.mean(), v_rms=np.sqrt((v**2).mean()),
                         D_meas_raw=D_meas_raw, D_meas_res=D_meas_res, rE=rE,
                         drift_day=np.abs(traj[-1]).mean())
    return out


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    for K_H, K_eta in [(0.6, 0.0), (1.58, 0.0), (0.6, 0.5)]:
        print("="*96)
        print(f"K_H = {K_H}, K_eta = {K_eta}   (20 days, 500 cells, seed 1)")
        print(f"{'':9s} {'mean rE':>8s} {'D_H pred':>10s} {'D_eta pred':>10s} {'D total':>10s} "
              f"{'D meas':>10s} {'meas/pred':>9s} {'Hebb share':>10s} {'v_rms':>8s} {'|drift| d20':>11s}")
        res = run(K_H, K_eta)
        for name in ["baseline", "CNO"]:
            r = res[name]
            tot = r['D_H'] + r['D_eta']
            print(f"{name:9s} {r['rE']:8.4f} {r['D_H']:10.5f} {r['D_eta']:10.5f} {tot:10.5f} "
                  f"{r['D_meas_res']:10.5f} {r['D_meas_res']/tot:9.2f} {r['D_H']/tot:10.2f} "
                  f"{r['v_rms']:8.4f} {r['drift_day']:11.3f}")
        b, c = res["baseline"], res["CNO"]
        pred_ratio = np.sqrt((c['D_H']+c['D_eta'])/(b['D_H']+b['D_eta']))
        meas_ratio = c['drift_day']/b['drift_day']
        print(f"  CNO/baseline: rE ratio {c['rE']/b['rE']:.2f} | predicted drift ratio "
              f"sqrt(D ratio) = {pred_ratio:.2f} | measured |drift| ratio day 20 = {meas_ratio:.2f}")
        print(f"  D_meas without removing systematic drift: baseline {b['D_meas_raw']:.5f}, CNO {c['D_meas_raw']:.5f}")
