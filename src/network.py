import json
import os

import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
from tqdm import tqdm
import h5py as h5

from helper_functions import circular_gaussian


class FeedForward():

    def __init__(self, N=500, N_inh = 500, a = 10, prop_shift=0, theta_stim=90, n_test_angles=100,
                 learning_rate = 0.01, n_days = 28, n_norm_per_day = 1,
                 n_steps_per_norm = 30, init_steps = 300, hebb_scaling = 0.3,
                 input_sigma = 25,
                 rand_scaling = 1, inh_scale = 1, 
                 vars_if_mean=3, vars_ei_mean=3, vars_ef_mean=2,
                 activity_dependent_noise = False,
                 inh="off", inh_type="co-tuned", 
                 norm = True, pre_run=True, seed = 100, set_seed=True):

        self.set_seed = set_seed
        self.seed = seed
        if self.set_seed:
            print(f"setting seed: {seed}")
            np.random.seed(self.seed)
        self.N = N
        self.N_inh = N_inh
        self.a = a
        self.prop_shift = prop_shift
        self.theta_stim = theta_stim
        self.n_test_angles = n_test_angles
        self.input_sigma = input_sigma  # Default value, can be overridden

        self.inh_type = inh_type
        self.inh_scale = inh_scale
        self.vars_ef = np.random.lognormal(vars_ef_mean, 0.6, N)
        # print(self.vars_ef)
        self.vars_if = np.random.lognormal(vars_if_mean, 0.6, N)
        self.vars_ei = np.random.lognormal(vars_ei_mean, 0.6, N)
        self.learning_rate = learning_rate


        self.norm = norm
        self.n_days = n_days
        self.n_norm_per_day = n_norm_per_day
        self.n_steps_per_norm = n_steps_per_norm
        self.n_steps = n_steps_per_norm * n_norm_per_day * n_days
        self.init_steps = init_steps

        self.hebb_scaling = hebb_scaling
        self.rand_scaling = rand_scaling
        self.activity_dependent_noise = activity_dependent_noise

        self.w_ef_init = self.initialise_w_ef(N, self.vars_ef)

        if inh == "on":
            self.w_if = inh_scale * self.initialise_w_if(N_inh, self.vars_if, inh_type)
            self.w_ei = inh_scale * self.initialise_w_ei(N_inh, self.vars_ei, inh_type)
        else:
            self.w_if = np.zeros((N, N_inh))
            self.w_ei = np.zeros((N_inh, N))

        if pre_run:
            self.w_ef_baseline = self.pre_run(self.w_ef_init, self.init_steps)
        else:
            self.w_ef_baseline = self.w_ef_init

    def propensity(self, w, a):
        """
        tanh function
        """

        return np.tanh(a*w + self.prop_shift)

    def circular_dist(self, x, y):
        """
        a circle over 0 to 180
        """
        return np.minimum(np.abs(x-y), 180-np.abs(x-y))

    def normalisation(self, w):
        """
        divisive normalisation and rectification
        """
        w_normed = w/(np.sum(w, axis=0) + 1e-10)

        if self.norm == True:
            return w_normed
        else:
            return w

    def initialise_w_ef(self, N, vars_ef):
        """
        stimulus pop (F) to exc pop (E) weights
        initialised as gaussian tuning curves with varyings widths
        """

        x = np.linspace(0, 180, N)
        matrix = np.zeros((N, N))
        for i in range(N):
            matrix[:, i] = stats.norm.pdf(x, x[i], vars_ef[i]) + stats.norm.pdf(x, x[i] + 180, vars_ef[i]) + stats.norm.pdf(x, x[i]-180, vars_ef[i])
        w_ef = matrix/N
        w_ef /= np.sum(w_ef, axis=0)

        return w_ef

    def initialise_w_if(self, N_inh, vars_if, inh_type="co-tuned"):
        """
        stimulus pop (F) to inh pop (I) weights
        initialised as gaussian tuning curves with varyings widths
        """
        if inh_type == "co-tuned":
            x = np.linspace(0, 180, N_inh)  # peaks of tuning curves of I neurons
            matrix = np.zeros((self.N, N_inh))
            for i in range(N_inh):
                matrix[:, i] = stats.norm.pdf(x, x[i], vars_if[i]) + stats.norm.pdf(x, x[i] + 180, vars_if[i]) + stats.norm.pdf(x, x[i]-180, vars_if[i])
            w_if = matrix/self.N
            w_if /= np.sum(w_if, axis=0)

        elif inh_type == "tuned_blanket":
            x = np.linspace(0, 180, N_inh)  # peaks of tuning curves of I neurons
            matrix = np.zeros((self.N, N_inh))
            for i in range(N_inh):
                matrix[:, i] = stats.norm.pdf(x, x[i], vars_if[i]) + stats.norm.pdf(x, x[i] + 180, vars_if[i]) + stats.norm.pdf(x, x[i]-180, vars_if[i])
            w_if = matrix/self.N
            w_if /= np.sum(w_if, axis=0)

        elif inh_type == "blanket":
            matrix = np.random.rand(self.N, N_inh)
            w_if = matrix/self.N
            w_if /= np.sum(w_if, axis=0)

        return w_if

    def initialise_w_ei(self, N_inh, vars_ei, inh_type="co-tuned"):
        """
        inh pop (I) to exc pop (E) weights
        initialised as gaussian tuning curves with varyings widths
        """
        if inh_type == "co-tuned":
            x = np.linspace(0, 180, self.N)
            matrix = np.zeros((N_inh, self.N))
            for i in range(self.N):
                matrix[:, i] = stats.norm.pdf(x, x[i], vars_ei[i]) + stats.norm.pdf(x, x[i] + 180, vars_ei[i]) + stats.norm.pdf(x, x[i]-180, vars_ei[i])
            w_ei = matrix/N_inh
            w_ei /= np.sum(w_ei, axis=0)  

        elif inh_type == "tuned_blanket":
            matrix = np.random.rand(N_inh, self.N)
            w_ei = matrix/N_inh
            w_ei /= np.sum(w_ei, axis=0)

        elif inh_type == "blanket":
            matrix = np.random.rand(N_inh, self.N)
            w_ei = matrix/N_inh
            w_ei /= np.sum(w_ei, axis=0)

        return w_ei     

    def circular_gaussian(self, N, theta, amp=2, sigma=25, baseline=0):
        """
        Generate pre-synaptic activity based on theta stimulus
        """
        theta_y = np.linspace(0, 180, N)  # center of tuning curves
        d = np.abs(theta - theta_y)    # distance to input theta
        d_plus = d + 180
        d_minus = d - 180
        y = amp * (np.exp(-(d**2)/(2*sigma**2)) + np.exp(-(d_plus**2)/(2*sigma**2)) + np.exp(-(d_minus**2)/(2*sigma**2))) + baseline

        return y

    def hebbian_component(self, N, w_ef, w_if, w_ei, theta_stim, type):
        """
        computes post-synaptic activity in the E & I population and 
        returns the hebbian outer product for E & F 
        """
        if type == "baseline" or type == "test" : theta = np.random.uniform(0, 180)
        elif type == "stripe_rearing": theta = theta_stim

        r_f = self.circular_gaussian(N, theta, amp=0.62, sigma=self.input_sigma, baseline=0)
        r_i = w_if.T.dot(r_f)

        r_e = w_ef.T.dot(r_f) - w_ei.T.dot(r_i)
        r_e[r_e < 0] = 0

        return np.outer(r_f, r_e), r_f, r_e

    def pre_run(self, w_init, init_steps):
        """
        initial evolution of weights
        """

        w = w_init
        # self.plot_weights(w, "F->E weights at initialisation")
        for t in range(init_steps):
            H, r_f, r_e = self.hebbian_component(self.N, w, self.w_if, self.w_ei, self.theta_stim, type='baseline')
            eta = np.random.randn(self.N, self.N)
            if self.activity_dependent_noise:
                eta_activity = eta * r_e
            else:
                eta_activity = 0

            prop_function = self.propensity(w, self.a)
            dw = (self.hebb_scaling * H * prop_function + self.rand_scaling * (eta + eta_activity) * prop_function) * self.learning_rate
            w += dw
            if t % self.n_steps_per_norm == 0:
                w = self.normalisation(w)
        # self.plot_weights(w, f"F->I weights after {self.init_steps} prerun steps")
        return w

    def get_preferred_orientations(self, N, w, n_angles):
        """
        
        """
        posts = np.zeros((N, n_angles))
        for i, angle in enumerate(np.linspace(0, 180, n_angles)):
            y = self.circular_gaussian(N, angle, amp=1, sigma=self.input_sigma, baseline=0)
            inh = self.w_if.T.dot(y)
            posts[:, i] = w.T.dot(y) - self.w_ei.T.dot(inh)
            posts[posts < 0] = 0
            # posts[:, i] = w.T.dot(y)

        return 180 * np.argmax(posts, axis=1) / n_angles

    def evolve_W(self, W_old, t, type):
        """
        
        """
        # print(f"timestep: {t}, w ={W_old}")
        H, r_f, r_e = self.hebbian_component(self.N, W_old, self.w_if, self.w_ei, self.theta_stim, type=type)
        eta = np.random.randn(self.N, self.N)

        if self.activity_dependent_noise:
            eta_activity = eta * r_e
        else:
            eta_activity = 0

        prop_function = self.propensity(W_old, self.a)
        hebb = self.hebb_scaling * H * prop_function
        rand = self.rand_scaling * (eta + eta_activity) * prop_function
        w_new = W_old + (hebb + rand) * self.learning_rate

        if t % self.n_steps_per_norm == 0:
            w_new = self.normalisation(w_new)

            if t % (self.n_steps_per_norm * self.n_norm_per_day) == 0:
                PO = self.get_preferred_orientations(self.N, W_old, n_angles=self.n_test_angles)
                # print(f"timestep: {t}, POs: {PO}")
                self.POs.append(PO)
        return w_new

    def get_POs_over_trials(self, w_baseline, n_steps, type):
        """
        
        """
        if self.set_seed:
            np.random.seed(self.seed)
            
        self.POs = []
        self.W = np.zeros((self.N, self.N, n_steps+1))
        self.W[:, :, 0] = w_baseline

        for t in tqdm(range(n_steps)):
            self.W[:, :, t+1] = self.evolve_W(self.W[:, :, t], t, type)

        return self.POs


    def get_metrics(self, N, n_days, theta_stim, POs):
        """
        computes drift magnitude, drift rate and convergence
        based on given preferred orientation array over time
        """
        preferences = np.array(POs).T
        initial_preferences = np.linspace(0, 180, N)
        # initial_preferences = preferences[:, 0]

        drift_magnitude = np.array([self.circular_dist(preferences[:, day], initial_preferences) for day in range(n_days)])
        drift_rate = np.array([self.circular_dist(preferences[:, day+1], preferences[:, day]) for day in range(n_days-1)])
        
        initial_distances = np.abs(initial_preferences - theta_stim)
        distances = np.abs(preferences - theta_stim)
        convergence = np.array([initial_distances - distances[:, day] for day in range(n_days - 1)])

        return drift_magnitude, drift_rate, convergence
    

    def get_correlations(self, w_ef):
        """
        computes activity correlations between neurons based on given preferred orientation array over time
        """

        theta_list = np.linspace(0, 180, 100)
        corr = np.zeros((self.N, self.N, len(theta_list)))

        # take a theta value and compute the activity of each neuron based on its preferred orientation and the circular gaussian function
        for stim_num, theta in enumerate(theta_list):
            u = self.circular_gaussian(self.N, theta, sigma=self.input_sigma)
            i = self.w_if.T.dot(u)
            e = w_ef.T.dot(u) - self.w_ei.T.dot(i)
            e[e < 0] = 0
            corr[:, :, stim_num] = self.corr_function(e)
        return corr
    
    def corr_function(self, rates):
        """
        Compute correlation matrix of rates
        """
        # Compute correlation matrix
        corr = np.outer(rates, rates) / (np.linalg.norm(rates) * np.linalg.norm(rates))

        return corr
    
    def get_correlations_new(self, w_ef):
        """
        uses np.corrcoef to compute correlations across neurons based on their responses to different stimuli,
        which is more efficient than computing pairwise correlations for each stimulus and then averaging.
        """
        theta_list = np.linspace(0, 180, 100)
        responses = np.zeros((self.N, len(theta_list)))
        
        for stim_num, theta in enumerate(theta_list):
            u = self.circular_gaussian(self.N, theta, sigma=self.input_sigma)
            i = self.w_if.T.dot(u)
            e = w_ef.T.dot(u) - self.w_ei.T.dot(i)
            e[e < 0] = 0
            responses[:, stim_num] = e
        
        # Compute pairwise correlations across stimuli
        corr = np.corrcoef(responses)  # N x N correlation matrix
        return corr

    def summed_corr_over_time(self, W, sigma=None):
        """
        computes activity correlations between neurons based on given preferred orientation array over time
        """
        if sigma is None:
            sigma = self.input_sigma

        theta_list = np.linspace(0, 180, 100)
        corr_over_time = np.zeros((W.shape[2]))

        # take a theta value and compute the activity of each neuron based on its preferred orientation and the circular gaussian function
        for t in tqdm(range(W.shape[2])):
            corr_over_stim = np.zeros((self.N, self.N, len(theta_list)))    
            for stim_num, theta in enumerate(theta_list):
                u = self.circular_gaussian(self.N, theta, sigma=sigma)
                i = self.w_if.T.dot(u)
                e = W[:, :, t].T.dot(u) - self.w_ei.T.dot(i)
                e[e < 0] = 0
                corr_over_stim[:, :, stim_num] = self.corr_function(e)

            corr_avg = np.mean(corr_over_stim, axis=2)
            corr_upper_triangle = corr_avg[np.triu_indices(self.N, k=1)]
            corr_over_time[t] = np.mean(corr_upper_triangle)

        return corr_over_time



    def plot_drift_magnitude(self, drift_mag_baseline, title, eo=2):
        """
        Plot drift magnitude over time
        """
        fig, ax = plt.subplots(1, 1, figsize=(3, 2), dpi=180)
        ax.plot(np.arange(1, self.n_days)[::eo], np.median(drift_mag_baseline, axis=1)[:-1][::eo],
                 c='black', ls='-', marker='o', ms=4, label='Baseline')
        
        ax.set_ylim([0, 5])
        ax.set_yticks([0, 5])
        ax.set_xlabel('time since start [days]')
        ax.set_ylabel(r'drift magnitude $ \; [\degree]$')
        ax.set_xlim(0, 30)
        ax.legend(frameon=False, fontsize=8)
        ax.set_title(title)
        fig.tight_layout()
        fig.show()
        return fig

    def plot_weights(self, weights, title):
        """
        Plot weight matrix
        """
        fig, axs = plt.subplots(1, 1, figsize=(5, 5), dpi=180)
        im = axs.imshow(weights)
        axs.set_ylabel("post")
        axs.set_xlabel("pre")
        axs.set_title(title)

        fig.colorbar(im, ax = axs)
        fig.show()
        
        return fig
    

    def estimate_initial_tuning_width(self):
        """
        Returns:
            tuning_curves: array of shape (N, n_test_angles) 
                    containing the tuning curves of each neuron to the test angles
            tuning_widths : array of shape (N,) containing the estimated tuning width of each neuron based on its tuning curve
        """
        tuning_curves = np.zeros((self.N, self.n_test_angles))
        theta_list = np.linspace(0, 180, self.n_test_angles)

        for theta_idx, theta in enumerate(theta_list):

            r_f = self.circular_gaussian(self.N, theta, amp=0.62, sigma=self.input_sigma, baseline=0)
            r_i = self.w_if.T.dot(r_f)
            r_e = self.w_ef_baseline.T.dot(r_f) - self.w_ei.T.dot(r_i)
            r_e[r_e < 0] = 0
            tuning_curves[:, theta_idx] = r_e

        tuning_widths = np.zeros(self.N)
        for i in range(self.N):
            curve = tuning_curves[i, :]
            curve = curve / np.max(curve)  # normalize the curve to have a max of 1
            half_max = 0.5
            indices_above_half_max = np.where(curve >= half_max)[0]
            if len(indices_above_half_max) > 1:
                tuning_widths[i] = theta_list[indices_above_half_max[-1]] - theta_list[indices_above_half_max[0]]
            else:
                tuning_widths[i] = np.nan  # if the curve does not reach half max, set width to NaN
        
        return tuning_curves, tuning_widths


    def estimate_initial_activity(self, probe_angle=60, sigma=None):
        """
        Returns the activity of each neuron to the probe stimulus at a given day
        """
        if sigma is None:
            sigma = self.input_sigma

        r_f = self.circular_gaussian(self.N, probe_angle, amp=0.62, sigma=sigma, baseline=0)
        r_i = self.w_if.T.dot(r_f)
        r_e = self.w_ef_baseline.T.dot(r_f) - self.w_ei.T.dot(r_i)
        r_e[r_e < 0] = 0

        return r_e, r_i, r_f
    
    def estimate_initial_tuning_inh(self, sigma=None):
        """
        Returns the tuning curves of the inhibitory population to the test angles
        """
        if sigma is None:
            sigma = self.input_sigma

        tuning_curves_inh = np.zeros((self.N_inh, self.n_test_angles))
        theta_list = np.linspace(0, 180, self.n_test_angles)

        for theta_idx, theta in enumerate(theta_list):

            r_f = self.circular_gaussian(self.N, theta, amp=0.62, sigma=sigma, baseline=0)
            r_i = self.w_if.T.dot(r_f)
            tuning_curves_inh[:, theta_idx] = r_i

        return tuning_curves_inh
    
    def estimate_activity_at_day(self, theta, day, sigma=None):
        if sigma is None:
            sigma = self.input_sigma
        
        stim_idx = day * self.n_norm_per_day * self.n_steps_per_norm
        w_ef = self.W[:, :, stim_idx]

        r_F = self.circular_gaussian(self.N, theta, amp=0.62, sigma=sigma, baseline=0)
        r_I = self.w_if.T.dot(r_F)
        r_E = w_ef.T.dot(r_F) - self.w_ei.T.dot(r_I)
        r_E[r_E < 0] = 0

        return r_E, r_I

    def estimate_tuning_curves_at_day(self, day, sigma=None, width_method='circular'):
        if sigma is None:
            sigma = self.input_sigma

        tuning_curves = np.zeros((self.N, self.n_test_angles))
        theta_list = np.linspace(0, 180, self.n_test_angles)

        stim_idx = day * self.n_norm_per_day * self.n_steps_per_norm
        w_ef = self.W[:, :, stim_idx]
        for theta_idx, theta in enumerate(theta_list):
            r_f = self.circular_gaussian(self.N, theta, amp=0.62, sigma=sigma, baseline=0)
            r_i = self.w_if.T.dot(r_f)
            r_e = w_ef.T.dot(r_f) - self.w_ei.T.dot(r_i)
            r_e[r_e < 0] = 0
            tuning_curves[:, theta_idx] = r_e
        return tuning_curves
    
    def estimate_tuning_curves_over_days(self, sigma=None):

        if sigma is None:
            sigma = self.input_sigma

        theta_list = np.linspace(0, 180, self.n_test_angles)
        
        tuning_curves_over_days = []

        for day in tqdm(range(self.n_days), desc='days'):
            tuning_curves = self.estimate_tuning_curves_at_day(day, sigma=sigma)
            tuning_curves_over_days.append(tuning_curves)

        return np.array(tuning_curves_over_days)


    def plot_weights_complete(self, savefig=False):

        fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=180)
        # plot F->E weights
        im1 = axs[0].imshow(self.w_ef_baseline)
        axs[0].set_title("F->E weights at baseline")
        axs[0].set_ylabel("post")
        axs[0].set_xlabel("pre")
        fig.colorbar(im1, ax=axs[0])

        # plot F->I weights
        im2 = axs[1].imshow(self.w_if)
        axs[1].set_title("F->I weights")
        axs[1].set_ylabel("post")
        axs[1].set_xlabel("pre")
        fig.colorbar(im2, ax=axs[1])

        # plot I->E weights
        im3 = axs[2].imshow(self.w_ei)
        axs[2].set_title("I->E weights")
        axs[2].set_ylabel("post")
        axs[2].set_xlabel("pre")
        fig.colorbar(im3, ax=axs[2])
        fig.tight_layout()

        if savefig:
            fig.savefig(self.save_location + "weights_complete.svg")

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

    def plot_initial_vs_final_tuning_curves(self, sigma=None):

        if sigma is None:
            sigma = self.input_sigma

        initial_tuning_curves = self.estimate_tuning_curves_at_day(0, sigma=sigma)
        final_tuning_curves = self.estimate_tuning_curves_at_day(self.n_days-1, sigma=sigma)

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

    def plot_POs_initial_vs_final(self):
        initial_POs = self.POs[0]
        final_POs = self.POs[-1]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(initial_POs, final_POs, alpha=0.5)
        ax.plot([0, 180], [0, 180], 'r--')  # reference line
        ax.set_xlabel("Initial Preferred Orientation")
        ax.set_ylabel("Final Preferred Orientation")
        ax.set_title("Initial vs Final Preferred Orientations")
        fig.tight_layout()
        fig.savefig(self.save_location+"POs_initial_vs_final.png", dpi=300)
        fig.show()

    def create_tuning_curves_animation(self, skip_freq=15, sigma=None):

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
            tuning_curves = self.estimate_tuning_curves_at_day(0, sigma=sigma)
            for nrn_idx in neuron_indices:
                l, = ax.plot(theta_list, tuning_curves[nrn_idx, :] + offset*nrn_idx, color='black')
                lines.append(l)
            return lines

        def animate(day):
            tuning_curves = self.estimate_tuning_curves_at_day(day, sigma=sigma)
            for l, nrn_idx in zip(lines, neuron_indices):
                l.set_data(theta_list, tuning_curves[nrn_idx, :] + offset*nrn_idx)
            ax.set_title(f"Day {day}")
            return lines

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=self.n_days, interval=500, blit=True)

        save_path = self.save_location + "tuning_curve_evolution.gif"
        anim.save(save_path, writer='imagemagick')

    def create_single_cell_tuning_curve_animation(self, tuning_curve_over_days, cell_idx):

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
        
    def run_analysis(self, saveloc=None, save_results=False):

        # work on this
        POs = self.get_POs_over_trials(self.w_ef_baseline, self.n_steps, type="baseline")
        drift_mag, drift_rate, convergence = self.get_metrics(self.N, self.n_days, self.theta_stim, POs)

        self.save_location = saveloc
        os.makedirs(saveloc, exist_ok=True)
        
        self.plot_weights_complete(savefig=True)
        self.plot_drift_metrics(drift_mag, drift_rate, convergence, savefig=True)
        self.plot_drift_metric_distributions(drift_mag, drift_rate, convergence, savefig=True)
        tuning_widths_assigned = self.vars_ef
        self.plot_drift_against_tuning(drift_mag[-1], tuning_widths_assigned, savefig=True)
        self.plot_initial_vs_final_tuning_curves()
        tuning_curves_over_days = self.estimate_tuning_curves_over_days()
        self.create_tuning_curves_animation()
        self.create_single_cell_tuning_curve_animation(tuning_curves_over_days[:, self.N//2, :], cell_idx=self.N//2)
        self.plot_POs_initial_vs_final()
        # self.create_pop_activity_animation()

        if save_results:
            self.save_results(drift_mag, drift_rate, convergence, save_weights=False)
        
        # [x] initial weights
        # [x] drift metrics over time
        # [x] final drift metric distributions
        # [x] drift against initial tuning width assigned (vars_ef)
        
        # [x] tuning curves initial and final
        # [x] POs initial vs final
        # [x] tuning curves animation
        # [x] single cell tuning curve animation
        # [x] population activity animation

        return None

    def save_results(self, drift_mag, drift_rate, convergence, save_weights=False):

        with h5.File(self.save_location + 'results.hdf5', 'w') as f:

            if save_weights:
                f.create_dataset("W", data=self.W)
                f.create_dataset("w_ef_baseline", data=self.w_ef_baseline)
                f.create_dataset("w_if", data=self.w_if)
                f.create_dataset("w_ei", data=self.w_ei)

            f.create_dataset("POs", data=self.POs)
            f.create_dataset("vars_ef", data=self.vars_ef)
            f.create_dataset("drift_mag", data=drift_mag)
            f.create_dataset("drift_rate", data=drift_rate)
            f.create_dataset("convergence", data=convergence)

        params = {
            "N": self.N,
            "N_inh": self.N_inh,
            "a": self.a,
            "prop_shift": self.prop_shift,
            "theta_stim": self.theta_stim,
            "n_test_angles": self.n_test_angles,
            "learning_rate": self.learning_rate,
            "n_days": self.n_days,
            "n_norm_per_day": self.n_norm_per_day,
            "n_steps_per_norm": self.n_steps_per_norm,
            "init_steps": self.init_steps,
            "hebb_scaling": self.hebb_scaling,
            "rand_scaling": self.rand_scaling,
            "inh_scale": self.inh_scale,
            "inh_type": self.inh_type,
            "vars_if_mean": self.vars_if.mean(),
            "vars_ei_mean": self.vars_ei.mean(),
            "vars_ef_mean": self.vars_ef.mean(),
            "input_sigma": self.input_sigma,
            "norm": self.norm,
            "seed": self.seed
        }

        with open(self.save_location + "hyperparameters.json", 'w') as f:
            json.dump(params, f, indent=4)
        pass