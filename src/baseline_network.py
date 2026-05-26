
import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from helper_functions import circular_gaussian
import scipy.stats as stats
import h5py as h5
import json
"""
Setup an all to all recurrent network with:
F -> E
F -> I
E -> E
E -> I
I -> E
I -> I

connections that can be turned on or off

all connections could be co-tuned, flat, or counter-tuned

all connections could follow plasticity rules or be fixed

Use instantiations of this network to test the contribution
 of each connection type to drift and stability of tuning curves
"""

class BaselineNetwork():

    def __init__(self, inh_type="co-tuned", E_to_E="off",
                  E_to_I="off", I_to_I="off",
                  plasticity_E_to_E="off", plasticity_E_to_I="off",
                  plasticity_I_to_E="off", plasticity_I_to_I="off",
                  plasticity_F_to_I="off",
                  inh_time_varying="off", inh_mod_start=7, inh_mod_end=14, inh_mod_scale=0.5,
                  inh_mod_type="weight_mod", # hyperpolarizing or weight_mod
                  inh_input_scale = 0,
                  norm=True, set_seed=False, seed=42, inh_scale=1, E_to_I_scale=1,
                  if_pre_run=True, n_pre_run_stimuli=300,
                  train_sigma=25, probe_sigma=5,
                  save_location="../results/recurrent_complete/feedforward_subset/"):

        self.inh_type = inh_type
        self.inh_scale = inh_scale
        self.E_to_I_scale = E_to_I_scale

        self.E_to_E = E_to_E
        self.E_to_I = E_to_I
        self.I_to_I = I_to_I
        self.plasticity_E_to_E = plasticity_E_to_E
        self.plasticity_E_to_I = plasticity_E_to_I
        self.plasticity_I_to_E = plasticity_I_to_E
        self.plasticity_I_to_I = plasticity_I_to_I
        self.plasticity_F_to_I = plasticity_F_to_I

    
        self.inh_mod_type = inh_mod_type
        self.inh_input_scale = inh_input_scale # for input based inhibition manipulation

        self.inh_time_varying = inh_time_varying
        self.inh_mod_start = inh_mod_start
        self.inh_mod_end = inh_mod_end
        self.inh_mod_scale = inh_mod_scale

        self.train_sigma = train_sigma
        self.probe_sigma = probe_sigma
        self.if_pre_run = if_pre_run
        self.n_pre_run_stimuli = n_pre_run_stimuli

        self.save_location = save_location
        os.makedirs(save_location, exist_ok=True)

        self.norm = norm
        self.set_seed = set_seed
        self.seed = seed

        if self.set_seed:
            np.random.seed(self.seed)

        self.setup_params()
        self.setup_weights()
        self.setup_inh_input()
        # self.plot_initial_weights()

        self.setup_inh_timeline()
    
    def setup_params(self):

        self.N = 400
        self.N_inh = 100

        self.tau_E = 10
        self.tau_I = 10

        self.dt = 1

        self.prop_a = 10
        self.prop_shift = 0

        self.learning_rate = 0.01
        self.hebb_scaling = 0.3
        self.rand_scaling = 1.0

        self.input_sigma = self.train_sigma
        
        self.vars_ef_mean = 2 # mean of lognormal distribution of widths of F to E weights
        self.vars_if_mean = 3 # mean of lognormal distribution of widths of F to I weights
        self.vars_ei_mean = 3 # mean of lognormal distribution of widths of I to E weights
        self.vars_ee_mean = 2 # mean of lognormal distribution of widths of E to E weights
        self.vars_ie_mean = 2 # mean of lognormal distribution of widths of E to I weights

        if self.I_to_I == "on":
            self.vars_ii_mean = 2

        self.n_days = 28
        self.n_test_angles = 500

        self.T_seq = 200 # dt steps per stimulus (should be >> tau to allow settling)
        self.n_stim_per_day = 30
        self.n_stim_per_norm = 30
        self.n_stim_total = self.n_stim_per_day * self.n_days
        
        return None
    
    def setup_variables(self):
        self.u_E = np.zeros(self.N)
        self.u_I = np.zeros(self.N_inh)

        self.r_E = np.zeros(self.N)
        self.r_I = np.zeros(self.N_inh)
    
        return None

    def setup_inh_timeline(self):

        self.inh_scale_timeline = np.ones(self.n_days)
        if self.inh_time_varying == "on":
            self.inh_scale_timeline[self.inh_mod_start:self.inh_mod_end] = self.inh_mod_scale

        return None
    
    def setup_inh_input(self):

        if self.inh_mod_type == "hyperpolarizing":
            self.inh_input = -0.5 * self.inh_input_scale * np.ones(self.N_inh)

        elif self.inh_mod_type == "weight_mod":
            self.inh_input = np.zeros(self.N_inh)
            if self.inh_time_varying == "off":
                self.w_ei *= self.inh_scale

    
    def setup_weights(self):

        # feedforward connection widths
        self.vars_ef = np.random.lognormal(mean=self.vars_ef_mean, sigma=0.6, size=self.N)
        self.vars_if = np.random.lognormal(mean=self.vars_if_mean, sigma=0.6, size=self.N_inh)
        self.vars_ei = np.random.lognormal(mean=self.vars_ei_mean, sigma=0.6, size=self.N)

        # recurrent connection widths (if present)
        self.vars_ee = np.random.lognormal(mean=self.vars_ee_mean, sigma=0.6, size=self.N)
        self.vars_ie = np.random.lognormal(mean=self.vars_ie_mean, sigma=0.6, size=self.N_inh)

        #    feedforward weights
        self.w_ef = self.gaussian_weights(self.N, self.N, self.vars_ef) # feedforward weights
        self.w_if = self.gaussian_weights(self.N, self.N_inh, self.vars_if) # feedforward input to inhibition
        if self.inh_type == "co-tuned":
            self.w_ei = self.gaussian_weights(self.N_inh, self.N, self.vars_ei) # I to E weights
        elif self.inh_type == "random":
            self.w_ei = self.random_weights(self.N_inh, self.N)
        # self.w_ei *= self.inh_scale

        # recurrent weights (if present)
        if self.E_to_E == "on":
            self.w_ee = self.gaussian_weights(self.N, self.N, self.vars_ee) # E to E weights
            np.fill_diagonal(self.w_ee, 0)
        else:
            self.w_ee = np.zeros((self.N, self.N))

        if self.E_to_I == "on":
            self.w_ie = self.gaussian_weights(self.N, self.N_inh, self.vars_ie) # E to I weights
            self.w_ie *= self.E_to_I_scale
        else:
            self.w_ie = np.zeros((self.N, self.N_inh))

        if self.I_to_I == "on":
            self.vars_ii = np.random.lognormal(mean=self.vars_ii_mean, sigma=0.6, size=self.N_inh)
            self.w_ii = self.gaussian_weights(self.N_inh, self.N_inh, self.vars_ii) # I to I weights
            np.fill_diagonal(self.w_ii, 0)
        else:
            self.w_ii = np.zeros((self.N_inh, self.N_inh))
        
        return None

    def setup_weight_histories(self):
        
        # Feedforward weights
        self.W_ef = np.zeros((self.N, self.N, self.n_stim_total + 1))
        self.W_ef[:, :, 0] = self.w_ef.copy()
        self.W_if = np.zeros((self.N, self.N_inh, self.n_stim_total + 1))
        self.W_if[:, :, 0] = self.w_if.copy()
        self.W_ei = np.zeros((self.N_inh, self.N, self.n_stim_total + 1))
        self.W_ei[:, :, 0] = self.w_ei.copy()

        # Recurrent weights
        self.W_ee = np.zeros((self.N, self.N, self.n_stim_total + 1))
        if self.E_to_E == "on":
            self.W_ee[:, :, 0] = self.w_ee.copy()
        self.W_ie = np.zeros((self.N, self.N_inh, self.n_stim_total + 1))
        if self.E_to_I == "on":
            self.W_ie[:, :, 0] = self.w_ie.copy()
        self.W_ii = np.zeros((self.N_inh, self.N_inh, self.n_stim_total + 1))
        if self.I_to_I == "on":
            self.W_ii[:, :, 0] = self.w_ii.copy()

        return None
    
    def gaussian_weights(self, N_pre, N_post, vars_list):
        x_pre = np.linspace(0, 180, N_pre, endpoint=False) # pre synaptic neuron tuning peaks
        x_post = np.linspace(0, 180, N_post, endpoint=False) # post synaptic neuron tuning peaks (in presynaptic neuron space)
        matrix = np.zeros((N_pre, N_post))
        for i in range(N_post): # for each post syn neuron
            matrix[:, i] = (stats.norm.pdf(x_pre, loc=x_post[i], scale=vars_list[i])
                            + stats.norm.pdf(x_pre, loc=x_post[i]+180, scale=vars_list[i])
                            + stats.norm.pdf(x_pre, loc=x_post[i]-180, scale=vars_list[i]))
        weights = matrix/N_pre # divide by number of inputs
        weights /= np.sum(weights, axis=0) # normalize each column to sum to 1
        return weights
    
    def random_weights(self, N_pre, N_post):

        weights = np.random.rand(N_pre, N_post)
        weights /= N_pre # divide by number of inputs
        weights /= np.sum(weights, axis=0) # normalize each column to sum to 1

        return weights

    def transfer_E(self, u):
        return np.maximum(0, u)

    def transfer_I(self, u):
        return np.maximum(0, u)

    def propensity(self, w, a):
        """
        tanh function
        """
        return np.tanh(a*w + self.prop_shift)
    
    def circular_dist(self, x, y):
        """
        a circle over 0 to 180
        """
        return np.minimum(np.abs(x - y), 180 - np.abs(x - y))

    def normalisation(self, w, if_recurrent=False):

        w_normed = w/(np.sum(w, axis=0) + 1e-10)

        if self.norm == True:
            return w_normed
        else:
            return w
        
    def circular_gaussian(self, N, theta_stim, amp=2, sigma=20, baseline=0):
        """
        Generate presynaptic activity based on theta stimulus 
        """
        theta_y = np.linspace(0, 180, N, endpoint=False)
        d = np.abs(theta_stim - theta_y)
        d_plus = d + 180
        d_minus = d - 180
        y = amp * (np.exp(-(d**2) / (2*sigma**2)) + np.exp(-(d_plus**2)/(2*sigma**2)) + np.exp(-(d_minus**2)/(2*sigma**2))) + baseline

        return y
    
    def pre_run(self):

        for stim_idx in tqdm(range(self.n_pre_run_stimuli)):
            self.theta_stim = np.random.uniform(0, 180)
            self.r_F = circular_gaussian(self.N, self.theta_stim, amp=0.62, sigma=self.train_sigma, baseline=0)

            for t in range(self.T_seq):
                self.step()
            self.evolve_weights()

            # normalize plastic weights every n_stim_per_norm presentations
            if stim_idx % self.n_stim_per_norm == 0:
                self.w_ef = self.normalisation(self.w_ef)
                # if self.plasticity_E_to_E == "on":
                #     self.w_ee = self.normalisation(self.w_ee)
                if self.plasticity_E_to_I == "on":
                    self.w_ie = self.normalisation(self.w_ie)
                if self.plasticity_I_to_E == "on":
                    self.w_ei = self.normalisation(self.w_ei)
                if self.plasticity_I_to_I == "on":
                    self.w_ii = self.normalisation(self.w_ii)
                if self.plasticity_F_to_I == "on":
                    self.w_if = self.normalisation(self.w_if)
    
    def run(self):
        
        self.setup_variables()
        self.setup_weight_histories()

        if self.if_pre_run:
            self.pre_run()

        self.POs = [self.get_preferred_orientations(inh_mod_scale=1.0)]

        for stim_idx in tqdm(range(self.n_stim_total)):

            # new stimulus every presentation
            self.theta_stim = np.random.uniform(0, 180)
            self.r_F = circular_gaussian(self.N, self.theta_stim, amp=0.62, sigma=self.train_sigma, baseline=0)

            #estimate day from stim idx to apply time varying inhibition if applicable
            day = stim_idx // self.n_stim_per_day
            inh_mod_scale = self.inh_scale_timeline[day]

            for t in range(self.T_seq):
                self.step(inh_scale=inh_mod_scale)

            self.evolve_weights()

            # normalize plastic weights every n_stim_per_norm presentations
            if stim_idx % self.n_stim_per_norm == 0:
                self.w_ef = self.normalisation(self.w_ef)
                # if self.plasticity_E_to_E == "on":
                #     self.w_ee = self.normalisation(self.w_ee)
                if self.plasticity_E_to_I == "on":
                    self.w_ie = self.normalisation(self.w_ie)
                if self.plasticity_I_to_E == "on":
                    self.w_ei = self.normalisation(self.w_ei)
                if self.plasticity_I_to_I == "on":
                    self.w_ii = self.normalisation(self.w_ii)
                if self.plasticity_F_to_I == "on":
                    self.w_if = self.normalisation(self.w_if)

            # estimate preferred orientations at end of each day
            if stim_idx % self.n_stim_per_day == 0:
                PO = self.get_preferred_orientations(inh_mod_scale=inh_mod_scale)
                self.POs.append(PO)

            self.record_weights(stim_idx + 1)

        return None

    def step(self, inh_scale=1.0):

        dU_E = (1/self.tau_E) * (-self.u_E + self.w_ef.T @ self.r_F + self.w_ee.T @ self.r_E - inh_scale * self.w_ei.T @ self.r_I)
        dU_I = (1/self.tau_I) * (-self.u_I + self.w_if.T @ self.r_F + self.w_ie.T @ self.r_E - inh_scale * self.w_ii.T @ self.r_I + self.inh_input)

        self.u_E += dU_E * self.dt
        self.u_I += dU_I * self.dt

        self.r_E = self.transfer_E(self.u_E)
        self.r_I = self.transfer_I(self.u_I)

        return None

    def _settle_batch(self, r_F_batch, w_ef, w_ee, w_ei, w_if, w_ie, w_ii, inh_scale=1.0):
        """Run all stimuli in r_F_batch simultaneously to steady state.
        r_F_batch: (n_stim, N). Returns r_E: (n_stim, N), r_I: (n_stim, N_inh).
        ~10-50x faster than looping over stimuli: one batched matmul replaces
        n_stim serial matvecs, and weight arrays are only accessed once.
        """
        n = r_F_batch.shape[0]
        u_E = np.zeros((n, self.N))
        u_I = np.zeros((n, self.N_inh))
        for _ in range(self.T_seq):
            r_E = np.maximum(0, u_E)
            r_I = np.maximum(0, u_I)
            dU_E = (1/self.tau_E) * (-u_E + r_F_batch @ w_ef + r_E @ w_ee - inh_scale * r_I @ w_ei)
            dU_I = (1/self.tau_I) * (-u_I + r_F_batch @ w_if + r_E @ w_ie - inh_scale * r_I @ w_ii)
            u_E += dU_E * self.dt
            u_I += dU_I * self.dt
        return np.maximum(0, u_E), np.maximum(0, u_I)

    def hebbian_component(self, r_pre, r_post):

        # simple hebbian component based on pre and post activity
        H = np.outer(r_pre, r_post)

        return H 

    def hebbian_plasticity_rule(self, w_old, r_pre, r_post, if_recurrent=False, intrinsic=True):
        """
        dW = learning_rate * (kH + eta) * propensity
        """
        H = self.hebbian_component(r_pre, r_post)
        eta = np.random.randn(*w_old.shape)
        prop_function = self.propensity(w_old, self.prop_a)
        hebb = self.hebb_scaling * H * prop_function
        if intrinsic:
            rand = self.rand_scaling * eta*prop_function
        else:
            rand = 0
        w_new = w_old + (hebb + rand) * self.learning_rate

        # set diagonal to 0 for recurrent weights
        if if_recurrent:
            np.fill_diagonal(w_new, 0)

        return w_new

    def evolve_weights(self):

        # Plasticity of feedforward weights
        self.w_ef = self.hebbian_plasticity_rule(self.w_ef, self.r_F, self.r_E)
        # Plasticity of F->I weights
        if self.plasticity_F_to_I == "on":
            self.w_if = self.hebbian_plasticity_rule(self.w_if, self.r_F, self.r_I)

        # Plasticity of recurrent E->E weights
        if self.plasticity_E_to_E == "on":
            self.w_ee = self.hebbian_plasticity_rule(self.w_ee, self.r_E, self.r_E, 
                                                     if_recurrent=True, intrinsic=False)
            self.w_ee = self.normalisation(self.w_ee, if_recurrent=True)
        # Plasticity of E->I weights
        if self.plasticity_E_to_I == "on":
            self.w_ie = self.hebbian_plasticity_rule(self.w_ie, self.r_E, self.r_I,
                                                     intrinsic=False)
        # Plasticity of I->E weights
        if self.plasticity_I_to_E == "on":
            self.w_ei = self.hebbian_plasticity_rule(self.w_ei, self.r_I, self.r_E,
                                                     intrinsic=False)
        # Plasticity of I->I weights
        if self.plasticity_I_to_I == "on":
            self.w_ii = self.hebbian_plasticity_rule(self.w_ii, self.r_I, self.r_I, 
                                                     if_recurrent=True, intrinsic=False)


    def record_weights(self, stim_idx):

        self.W_ef[:, :, stim_idx] = self.w_ef.copy()
        self.W_ee[:, :, stim_idx] = self.w_ee.copy()
        self.W_ei[:, :, stim_idx] = self.w_ei.copy()
        self.W_if[:, :, stim_idx] = self.w_if.copy()
        self.W_ie[:, :, stim_idx] = self.w_ie.copy()
        self.W_ii[:, :, stim_idx] = self.w_ii.copy()

        return None


    def get_preferred_orientations(self, inh_mod_scale=1.0):

        theta_list = np.linspace(0, 180, self.n_test_angles, endpoint=False)

        r_F_batch = np.array([
            circular_gaussian(self.N, a, amp=0.62, sigma=self.probe_sigma, baseline=0)
            for a in theta_list
        ])

        r_E_batch, _ = self._settle_batch(
            r_F_batch,
            self.w_ef, self.w_ee, self.w_ei,
            self.w_if, self.w_ie, self.w_ii,
            inh_mod_scale
        )
        tuning_curves = r_E_batch.T  # (N, n_angles)

        peak_activity = np.max(tuning_curves, axis=1)
        threshold = 0.05 * np.max(peak_activity)
        PO_estimate = np.where(
            peak_activity > threshold,
            theta_list[np.argmax(tuning_curves, axis=1)],
            np.nan
        )

        return PO_estimate

    def get_drift_metrics(self, theta_stim=90):
        """
        
        """
        preferences = np.array(self.POs).T # shape (N, n_days)
        initial_preferences = preferences[:, 0]
        # alternate measure of initial preferences
        # initial_preferences = np.linspace(0, 180, self.N, endpoint=False)
        
        drift_mag = np.array([self.circular_dist(preferences[:, day], initial_preferences) for day in range(len(self.POs))])
        drift_rate = np.array([self.circular_dist(preferences[:, day], preferences[:, day-1]) for day in range(1, len(self.POs))])

        initial_distances = np.abs(initial_preferences - theta_stim)
        distances = np.abs(preferences - theta_stim)
        convergence = np.array([initial_distances - distances[:, day] for day in range(1, len(self.POs))])
    
        return drift_mag, drift_rate, convergence

    def estimate_activity_at_day(self, theta, day, sigma=None):
        """
        
        """
        if sigma is None:
            sigma = self.input_sigma

        r_F = circular_gaussian(self.N, theta, amp=0.62, sigma=sigma, baseline=0)

        stim_idx = day * self.n_stim_per_day
        inh_mod_scale = self.inh_scale_timeline[day]

        w_ef = self.W_ef[:, :, stim_idx].copy()
        w_ee = self.W_ee[:, :, stim_idx].copy()
        w_ei = self.W_ei[:, :, stim_idx].copy()
        w_if = self.W_if[:, :, stim_idx].copy()
        w_ie = self.W_ie[:, :, stim_idx].copy()
        w_ii = self.W_ii[:, :, stim_idx].copy()

        u_E = np.zeros(self.N)
        u_I = np.zeros(self.N_inh)
        r_E = np.zeros(self.N)
        r_I = np.zeros(self.N_inh)

        for t in range(self.T_seq):
            dU_E = (1/self.tau_E) * (-u_E + w_ef.T @ r_F + w_ee.T @ r_E - inh_mod_scale * w_ei.T @ r_I)
            dU_I = (1/self.tau_I) * (-u_I + w_if.T @ r_F + w_ie.T @ r_E - inh_mod_scale * w_ii.T @ r_I)

            u_E += dU_E * self.dt
            u_I += dU_I * self.dt

            r_E = self.transfer_E(u_E)
            r_I = self.transfer_I(u_I)

        return r_E, r_I
    
    def estimate_tuning_curves_at_day(self, day, sigma=None, width_method='circular'):
        
        if sigma is None:
            sigma = self.input_sigma

        theta_list = np.linspace(0, 180, self.n_test_angles, endpoint=False)

        # Build all probe inputs once — (n_angles, N)
        r_F_batch = np.array([
            circular_gaussian(self.N, a, amp=0.62, sigma=sigma, baseline=0)
            for a in theta_list
        ])

        stim_idx = day * self.n_stim_per_day
        inh_scale = self.inh_scale_timeline[day]

        # Fetch weight snapshot for this day once (copy → contiguous for fast BLAS)
        w_ef = self.W_ef[:, :, stim_idx].copy()
        w_ee = self.W_ee[:, :, stim_idx].copy()
        w_ei = self.W_ei[:, :, stim_idx].copy()
        w_if = self.W_if[:, :, stim_idx].copy()
        w_ie = self.W_ie[:, :, stim_idx].copy()
        w_ii = self.W_ii[:, :, stim_idx].copy()

        r_E_batch, _ = self._settle_batch(
            r_F_batch, w_ef, w_ee, w_ei, w_if, w_ie, w_ii, inh_scale
        )
        tuning_curves_E = r_E_batch.T  # (N, n_angles)
        tuning_widths = np.zeros(self.N)

        for neuron_idx in range(self.N):
            if width_method == 'circular':
                tuning_widths[neuron_idx] = self.circular_FWHM(
                    theta_list, tuning_curves_E[neuron_idx, :]
                )
            elif width_method == 'standard':
                tuning_widths[neuron_idx] = self.standard_FWHM(
                    theta_list, tuning_curves_E[neuron_idx, :]
                )

        return tuning_curves_E, tuning_widths
    
    def estimate_tuning_curves_over_days(self, sigma=None, width_method='circular'):

        if sigma is None:
            sigma = self.input_sigma

        theta_list = np.linspace(0, 180, self.n_test_angles, endpoint=False)

        # Build all probe inputs once — (n_angles, N)
        r_F_batch = np.array([
            circular_gaussian(self.N, a, amp=0.62, sigma=sigma, baseline=0)
            for a in theta_list
        ])

        self.tuning_curves_over_days = np.zeros((self.n_days, self.N, self.n_test_angles))
        self.tuning_widths_over_days = np.zeros((self.n_days, self.N))

        for day in tqdm(range(self.n_days), desc="days"):
            stim_idx = day * self.n_stim_per_day
            inh_scale = self.inh_scale_timeline[day]

            # Fetch weight snapshot for this day once (copy → contiguous for fast BLAS)
            w_ef = self.W_ef[:, :, stim_idx].copy()
            w_ee = self.W_ee[:, :, stim_idx].copy()
            w_ei = self.W_ei[:, :, stim_idx].copy()
            w_if = self.W_if[:, :, stim_idx].copy()
            w_ie = self.W_ie[:, :, stim_idx].copy()
            w_ii = self.W_ii[:, :, stim_idx].copy()

            r_E_batch, _ = self._settle_batch(
                r_F_batch, w_ef, w_ee, w_ei, w_if, w_ie, w_ii, inh_scale
            )
            self.tuning_curves_over_days[day] = r_E_batch.T  # (N, n_angles)

            for neuron_idx in range(self.N):
                if width_method == 'circular':
                    self.tuning_widths_over_days[day, neuron_idx] = self.circular_FWHM(
                        theta_list, self.tuning_curves_over_days[day, neuron_idx, :]
                    )
                elif width_method == 'standard':
                    self.tuning_widths_over_days[day, neuron_idx] = self.standard_FWHM(
                        theta_list, self.tuning_curves_over_days[day, neuron_idx, :]
                    )

        return self.tuning_curves_over_days, self.tuning_widths_over_days

    def standard_FWHM(self, x, y):
        half_max = np.max(y) / 2
        indices = np.where(y >= half_max)[0]
        if len(indices) < 2:
            return np.nan  # Not enough points above half max to calculate FWHM
        left_idx = indices[0]
        right_idx = indices[-1]
        fwhm = x[right_idx] - x[left_idx]
        return fwhm
    

    def circular_FWHM(self, x, y):
        """
        Calculate FWHM for circular data by rolling the tuning curve so the peak is in the center,
        then finding the width at half max as usual.
        """
        peak_idx = np.argmax(y)
        center = len(y) // 2
        y_rolled = np.roll(y, center - peak_idx)   # shift peak to middle

        half_max = np.max(y_rolled) / 2
        indices = np.where(y_rolled >= half_max)[0]
        if len(indices) < 2:
            return np.nan

        dx = x[1] - x[0]   # degrees per bin
        return (indices[-1] - indices[0]) * dx


    def plot_population_tuning_curves(self, day, offset=0.028, sigma=None, savefig=False):
        if sigma is None:
            sigma = self.input_sigma

        tuning_curves, _ = self.estimate_tuning_curves_at_day(day, sigma=sigma)
        theta_list = np.linspace(0, 180, self.n_test_angles, endpoint=False)

        fig, axs = plt.subplots(figsize=(6, 4), dpi=300)
        for neuron_idx in range(0, self.N, 15): # plot every 30th neuron for visibility
            axs.plot(theta_list, tuning_curves[neuron_idx, :] + offset*neuron_idx, color='black')
        axs.set_title(f"Population Tuning Curves at Day {day}")
        axs.set_xlabel("Stimulus Angle")
        axs.set_ylabel("Firing Rate")
        fig.tight_layout()
        if savefig:
            fig.savefig(self.save_location+f"population_tuning_curves_day_{day}.png", dpi=300)
        fig.show()
    

    def plot_cell_tuning_curve(self, cell_idx, day, sigma=None, savefig=False):
        if sigma is None:
            sigma = self.input_sigma
        
        tuning_curves, _ = self.estimate_tuning_curves_at_day(day, sigma=sigma)
        theta_list = np.linspace(0, 180, self.n_test_angles, endpoint=False)

        fig, axs = plt.subplots(figsize=(6, 4))
        axs.plot(theta_list, tuning_curves[cell_idx, :], marker='o', ms=4, clip_on=False)
        axs.set_title(f"Tuning Curve of Cell {cell_idx} at Day {day}")
        axs.set_xlabel("Stimulus Angle")
        axs.set_ylabel("Firing Rate")
        fig.tight_layout()
        if savefig:
            fig.savefig(self.save_location+f"tuning_curve_cell_{cell_idx}_day_{day}.png", dpi=300)
        fig.show()

    def plot_initial_weights(self, savefig=False):

        fig, axs = plt.subplots(2, 3, figsize=(12, 6))

        panels = [
            (axs[0, 0], self.w_ef.T, "F to E weights"),
            (axs[0, 1], self.w_ee.T, "E to E weights"),
            (axs[0, 2], self.w_ei.T, "I to E weights"),
            (axs[1, 0], self.w_if.T, "F to I weights"),
            (axs[1, 1], self.w_ie.T, "E to I weights"),
            (axs[1, 2], self.w_ii.T, "I to I weights"),
        ]
        for ax, data, title in panels:
            im = ax.imshow(data, aspect='auto')
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.tight_layout()
        if savefig:
            fig.savefig(self.save_location+"initial_weights.png", dpi=300)
        fig.show()   

    def plot_weights(self, savefig=False):

        fig, axs = plt.subplots(2, 3, figsize=(12, 6))

        panels = [
            (axs[0, 0], self.w_ef.T, "F to E weights"),
            (axs[0, 1], self.w_ee.T, "E to E weights"),
            (axs[0, 2], self.w_ei.T, "I to E weights"),
            (axs[1, 0], self.w_if.T, "F to I weights"),
            (axs[1, 1], self.w_ie.T, "E to I weights"),
            (axs[1, 2], self.w_ii.T, "I to I weights"),
        ]
        for ax, data, title in panels:
            im = ax.imshow(data, aspect='auto')
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.tight_layout()
        if savefig:
            fig.savefig(self.save_location+"final_weights.png", dpi=300)
        fig.show()
          

    def plot_drift_metrics(self, drift_mag, drift_rate, convergence, savefig=False,
                           figsize=(10, 3)):

        fig, axs = plt.subplots(1, 3, figsize=figsize)

        drift_mag_mean = np.nanmean(drift_mag, axis=1)
        drift_mag_std = np.nanstd(drift_mag, axis=1)/np.sqrt(drift_mag.shape[1])
        axs[0].plot(drift_mag_mean, marker='o', ms=4, clip_on=False)
        axs[0].fill_between(range(len(drift_mag_mean)), drift_mag_mean - drift_mag_std, drift_mag_mean + drift_mag_std, alpha=0.2)
        axs[0].set_title("Drift Magnitude")
        axs[0].set_xlabel("Day")
        axs[0].set_ylabel("Degrees")
        axs[0].set_ylim([-1, 5])

        drift_rate_mean = np.nanmean(drift_rate, axis=1)
        drift_rate_std = np.nanstd(drift_rate, axis=1)/np.sqrt(drift_rate.shape[1])
        axs[1].plot(drift_rate_mean, marker='o', ms=4, clip_on=False)
        axs[1].fill_between(range(len(drift_rate_mean)), drift_rate_mean - drift_rate_std, drift_rate_mean + drift_rate_std, alpha=0.2)
        axs[1].set_title("Drift Rate")
        axs[1].set_xlabel("Day")
        axs[1].set_ylabel("Degrees/day")
        axs[1].set_ylim([-1, 5])

        convergence_mean = np.nanmean(convergence, axis=1)
        convergence_std = np.nanstd(convergence, axis=1)/np.sqrt(convergence.shape[1])
        axs[2].plot(convergence_mean, marker='o', ms=4, clip_on=False)
        axs[2].fill_between(range(len(convergence_mean)), convergence_mean - convergence_std, convergence_mean + convergence_std, alpha=0.2)
        axs[2].set_title("Convergence")
        axs[2].set_xlabel("Day")
        axs[2].set_ylabel("Degrees")
        axs[2].set_ylim([-1, 5])

        fig.tight_layout()
        if savefig:
            fig.savefig(self.save_location+"drift_metrics.png", dpi=300)
        fig.show()

    def plot_drift_metric_distributions(self, drift_mag, drift_rate, convergence, savefig=False, figsize=(10, 3)):

        fig, axs = plt.subplots(1, 3, figsize=figsize)

        axs[0].hist(drift_mag[-1, :], bins='fd', alpha=0.7)
        axs[0].set_title("Drift Magnitude Distribution")
        axs[0].set_xlabel("Degrees")
        axs[0].set_ylabel("Frequency")

        axs[1].hist(drift_rate[-1, :], bins='fd', alpha=0.7)
        axs[1].set_title("Drift Rate Distribution")
        axs[1].set_xlabel("Degrees/day")
        axs[1].set_ylabel("Frequency")

        axs[2].hist(convergence[-1, :], bins='fd', alpha=0.7)
        axs[2].set_title("Convergence Distribution")
        axs[2].set_xlabel("Degrees")
        axs[2].set_ylabel("Frequency")

        fig.tight_layout()
        if savefig:
            fig.savefig(self.save_location+"drift_metric_distributions.png", dpi=300)
        fig.show()

    def plot_activity_at_day(self, day, theta, sigma=None, savefig=False):

        if sigma is None:
            sigma = self.input_sigma

        r_F = circular_gaussian(self.N, theta, amp=0.62, sigma=sigma, baseline=0)
        r_E, r_I = self.estimate_activity_at_day(theta, day, sigma=sigma)

        fig, axs = plt.subplots(1, 3, figsize=(11, 4))

        axs[0].plot(r_F, marker='.', ms=4, clip_on=False)
        axs[0].set_title(f"Input pop")
        axs[0].set_xlabel("Neuron Index")
        axs[0].set_ylabel("Firing Rate")

        axs[1].plot(r_E, marker='.', ms=4, clip_on=False)
        axs[1].set_title(f"E Pop")
        axs[1].set_xlabel("Neuron Index")
        axs[1].set_ylabel("Firing Rate")

        axs[2].plot(r_I, marker='.', ms=4, clip_on=False)
        axs[2].set_title(f"I Pop")
        axs[2].set_xlabel("Neuron Index")
        axs[2].set_ylabel("Firing Rate")

        fig.suptitle(f"Activity at Day {day} for Stimulus {theta:.1f}")

        fig.tight_layout()
        if savefig:
            fig.savefig(self.save_location+f"activity_day_{day}_theta_{theta:.1f}.png", dpi=300)
        fig.show()

    def plot_drift_against_tuning(self, drift_mag, tuning_widths, savefig=False):
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(tuning_widths, drift_mag, alpha=0.5)
        ax.set_xlabel("Tuning Width")
        ax.set_ylabel("Drift Magnitude")
        ax.set_title("Drift Magnitude vs Tuning Width")
        ax.set_xscale('log')
        ax.set_yscale('log')
        fig.tight_layout()
        if savefig:
            fig.savefig(self.save_location+"drift_against_tuning.png", dpi=300)
        fig.show()

    def plot_initial_vs_final_tuning_curves(self, sigma=None):

        if sigma is None:
            sigma = self.input_sigma

        initial_tuning_curves, _ = self.estimate_tuning_curves_at_day(0, sigma=sigma)
        final_tuning_curves, _ = self.estimate_tuning_curves_at_day(self.n_days-1, sigma=sigma)

        theta_list = np.linspace(0, 180, self.n_test_angles, endpoint=False)

        fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
        for neuron_idx in range(0, self.N, 15): # plot every 30th neuron for visibility
            axs[0].plot(theta_list, initial_tuning_curves[neuron_idx, :] + 0.028*neuron_idx, color='black')
            axs[1].plot(theta_list, final_tuning_curves[neuron_idx, :] + 0.028*neuron_idx, color='black')
        axs[0].set_title(f"Initial Tuning Curves")
        axs[1].set_title(f"Final Tuning Curves")
        for ax in axs:
            ax.set_xlabel("Stimulus Angle")
            ax.set_ylabel("Firing Rate")
        fig.tight_layout()
        fig.savefig(self.save_location+"initial_vs_final_tuning_curves.png", dpi=300)
        fig.show()

    def create_tuning_curve_animation(self, skip_freq=15, sigma=None):

        import matplotlib.animation as animation

        offset = 0.028
        neuron_indices = list(range(0, self.N, skip_freq))
        theta_list = np.linspace(0, 180, self.n_test_angles, endpoint=False)
        max_ylim = offset * neuron_indices[-1] + 2

        fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        ax.set_xlim(0, 180)
        ax.set_ylim(0, max_ylim)
        ax.set_xlabel("Stimulus Angle")
        ax.set_ylabel("Firing Rate")
        fig.suptitle("Tuning Curve Evolution")

        lines = []

        def init():
            for l in lines:
                l.remove()
            lines.clear()
            tuning_curves, _ = self.estimate_tuning_curves_at_day(0, sigma=sigma)
            for nrn_idx in neuron_indices:
                l, = ax.plot(theta_list, tuning_curves[nrn_idx, :] + offset*nrn_idx, color='black')
                lines.append(l)
            return lines

        def animate(day):
            tuning_curves, _ = self.estimate_tuning_curves_at_day(day, sigma=sigma)
            for l, nrn_idx in zip(lines, neuron_indices):
                l.set_data(theta_list, tuning_curves[nrn_idx, :] + offset*nrn_idx)
            ax.set_title(f"Day {day}")
            return lines

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=self.n_days, interval=500, blit=True)

        save_path = self.save_location + "tuning_curve_evolution.gif"
        anim.save(save_path, writer='imagemagick')

    def create_single_cell_tuning_curve_animation(self, tuning_curve_over_days, cell_idx):
        """
        tuning_curve_over_days: (n_days, n_angles)
        """
        import matplotlib.animation as animation

        fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        ax.set_xlim(0, 180)
        ax.set_ylim(0, 4)
        ax.set_xlabel("Stimulus Angle")
        ax.set_ylabel("Firing Rate")
        fig.suptitle(f"Tuning Curve Evolution of Cell {cell_idx}")

        line, = ax.plot([], [], color='blue')

        def init():
            line.set_data([], [])
            return line,

        def animate(day):
            theta_list = np.linspace(0, 180, self.n_test_angles, endpoint=False)
            line.set_data(theta_list, tuning_curve_over_days[day, :])
            ax.set_title(f"Day {day}")
            return line,

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=self.n_days, interval=500, blit=True)

        save_path = self.save_location + f"tuning_curve_evolution_cell_{cell_idx}.gif"
        anim.save(save_path, writer='imagemagick')

    def create_pop_activity_animation(self, theta, sigma=None):
        """
        Animate population activity over days for a given stimulus.
        """
        if sigma is None:
            sigma = self.input_sigma

        import matplotlib.animation as animation

        fig, ax = plt.subplots(2, 1, figsize=(6, 4), dpi=200)

        ax[0].set_xlim(0, self.N)
        ax[0].set_ylim(0, 4)
        ax[0].set_xlabel("Neuron Index")
        ax[0].set_ylabel("Firing Rate")
        ax[1].set_xlim(0, self.N_inh)
        ax[1].set_ylim(0, 4)
        ax[1].set_xlabel("Neuron Index")
        ax[1].set_ylabel("Firing Rate")

        fig.suptitle(f"Population Activity Evolution for Stimulus {theta:.1f}")

        line_E, = ax[0].plot([], [], marker='.', ms=4, color='blue')
        line_I, = ax[1].plot([], [], marker='.', ms=4, color='red')

        def init():
            line_E.set_data([], [])
            line_I.set_data([], [])
            return line_E, line_I,

        def animate(day):
            r_E, r_I = self.estimate_activity_at_day(theta, day, sigma=sigma)
            line_E.set_data(range(self.N), r_E)
            line_I.set_data(range(self.N_inh), r_I)
            ax[0].set_title(f"Day {day}")
            return line_E, line_I,

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=self.n_days, interval=500, blit=True)

        save_path = self.save_location + f"population_activity_evolution_theta_{theta:.1f}.gif"
        anim.save(save_path, writer='imagemagick')

    def save_results(self, drift_mag, drift_rate, convergence):

        with h5.File(self.save_location + "results.h5", "w") as f:
            f.create_dataset("W_ef", data=self.W_ef)
            f.create_dataset("W_ee", data=self.W_ee)
            f.create_dataset("W_ei", data=self.W_ei)
            f.create_dataset("W_if", data=self.W_if)
            f.create_dataset("W_ie", data=self.W_ie)
            f.create_dataset("W_ii", data=self.W_ii)

            f.create_dataset("POs", data=self.POs)
            f.create_dataset("tuning_widths_over_days", data=self.tuning_widths_over_days)
            f.create_dataset("tuning_curves_over_days", data=self.tuning_curves_over_days)

            f.create_dataset("drift_mag", data=drift_mag)
            f.create_dataset("drift_rate", data=drift_rate)
            f.create_dataset("convergence", data=convergence)


        # save hyperparameters in json
        params = {
            "inh_scale": self.inh_scale,
            "inh_type": self.inh_type,
            "E_to_E": self.E_to_E,
            "E_to_I": self.E_to_I,
            # "I_to_E": self.I_to_E,
            "I_to_I": self.I_to_I,
            "plasticity_E_to_E": self.plasticity_E_to_E,
            "plasticity_E_to_I": self.plasticity_E_to_I,
            "plasticity_I_to_E": self.plasticity_I_to_E,
            "plasticity_I_to_I": self.plasticity_I_to_I,
            "plasticity_F_to_I": self.plasticity_F_to_I,
            "inh_time_varying": self.inh_time_varying,
            "inh_mod_start": self.inh_mod_start,
            "inh_mod_end": self.inh_mod_end,
            "inh_mod_scale": self.inh_mod_scale,
            "inh_mod_type": self.inh_mod_type,
            "tau_E": self.tau_E,
            "tau_I": self.tau_I,
            "dt": self.dt,
            "hebb_scaling": self.hebb_scaling,
            "rand_scaling": self.rand_scaling,
            "learning_rate": self.learning_rate,
            "prop_a": self.prop_a,
            "prop_shift": self.prop_shift,
            "vars_ef_mean": self.vars_ef_mean,
            "vars_if_mean": self.vars_if_mean,
            "vars_ei_mean": self.vars_ei_mean,
            "vars_ee_mean": self.vars_ee_mean,
            "vars_ie_mean": self.vars_ie_mean,
            "N": self.N,
            "N_inh": self.N_inh,
            "n_days": self.n_days,
            "n_stim_per_day": self.n_stim_per_day,
            "n_pre_run_stimuli": self.n_pre_run_stimuli,
            "n_test_angles": self.n_test_angles,
            "train_sigma": self.train_sigma,
            "probe_sigma": self.probe_sigma,
            "set_seed": self.set_seed,
            "seed": self.seed
        }

        if self.I_to_I == "on":
            params["vars_ii_mean"] = self.vars_ii_mean

        with open(self.save_location + "hyperparameters.json", "w") as f:
            json.dump(params, f, indent=4)
        
        pass
    
    def run_analysis(self, save_results=True):

        self.plot_initial_weights(savefig=True)
        self.run()
        self.plot_weights(savefig=True)
        drift_mag, drift_rate, convergence = self.get_drift_metrics()
        self.plot_drift_metrics(drift_mag, drift_rate, convergence, savefig=True)
        self.plot_drift_metric_distributions(drift_mag, drift_rate, convergence, savefig=True)
        # estimate tuning widths at initial day
        # tuning_curves_initial = self.estimate_tuning_curves_at_day(0)
        tuning_widths_assigned = self.vars_ef.copy() # assigned tuning widths based on feedforward weights

        self.tuning_curves_over_days, self.tuning_widths_over_days = self.estimate_tuning_curves_over_days(sigma=self.train_sigma)

        self.plot_drift_against_tuning(drift_mag[-1], tuning_widths_assigned, savefig=True)

        self.plot_initial_vs_final_tuning_curves(sigma=self.train_sigma)

        self.create_tuning_curve_animation(sigma=self.train_sigma)

        self.create_single_cell_tuning_curve_animation(self.tuning_curves_over_days[:, self.N//2, :], cell_idx=self.N//2)
        self.create_pop_activity_animation(theta=90)
        if save_results:
            self.save_results(drift_mag, drift_rate, convergence)

        return None
    
def load_data(save_location):

    with h5.File(save_location + "results.h5", "r") as f:
        W_ef = f["W_ef"][:]
        W_ee = f["W_ee"][:]
        W_ei = f["W_ei"][:]
        W_if = f["W_if"][:]
        W_ie = f["W_ie"][:]
        W_ii = f["W_ii"][:]

        POs = f["POs"][:]
        tuning_widths_over_days = f["tuning_widths_over_days"][:]
        tuning_curves_over_days = f["tuning_curves_over_days"][:]

        drift_mag = f["drift_mag"][:]
        drift_rate = f["drift_rate"][:]
        convergence = f["convergence"][:]

    with open(save_location + "hyperparameters.json", "r") as f:
        params = json.load(f)

    return {
        "weights": {
            "W_ef": W_ef,
            "W_ee": W_ee,
            "W_ei": W_ei,
            "W_if": W_if,
            "W_ie": W_ie,
            "W_ii": W_ii
        },
        "metrics": {
            "POs": POs,
            "tuning_widths_over_days": tuning_widths_over_days,
            "tuning_curves_over_days": tuning_curves_over_days,
            "drift_mag": drift_mag,
            "drift_rate": drift_rate,
            "convergence": convergence
        },
        "hyperparameters": params
    }
