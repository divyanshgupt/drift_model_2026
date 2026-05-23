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
                 rand_scaling = 1, inh_scale = 1, vars_if_mean=3, vars_ei_mean=3, inh="off", inh_type="co-tuned", 
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
        
        self.inh_type = inh_type
        self.inh_scale = inh_scale
        self.vars_ef = np.random.lognormal(2, 0.6, N)
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

    def circular_gaussian(self, N, theta, amp=2, sigma=20, baseline=0):
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

        r_f = self.circular_gaussian(N, theta, amp=0.62, sigma=60, baseline=0)
        r_i = w_if.T.dot(r_f)

        r_e = w_ef.T.dot(r_f) - w_ei.T.dot(r_i)
        r_e[r_e < 0] = 0

        return np.outer(r_f, r_e)

    def pre_run(self, w_init, init_steps):
        """
        initial evolution of weights
        """

        w = w_init
        # self.plot_weights(w, "F->E weights at initialisation")
        for t in range(init_steps):
            H = self.hebbian_component(self.N, w, self.w_if, self.w_ei, self.theta_stim, type='baseline')
            eta = np.random.randn(self.N, self.N)
            prop_function = self.propensity(w, self.a)
            dw = (self.hebb_scaling * H * prop_function + self.rand_scaling * eta * prop_function) * self.learning_rate
            w += dw
            if t % self.n_steps_per_norm == 0:
                w = self.normalisation(w)
        # self.plot_weights(w, f"F->I weights after {self.init_steps} prerun steps")
        return w

    def get_preferred_orientations(self, N, w, n_angles):
        """
        
        """
        posts = np.zeros((N, n_angles))
        for i, angle in enumerate(np.linspace(0, 181, n_angles)):
            y = self.circular_gaussian(N, angle, amp=1, sigma=5, baseline=0)
            inh = self.w_if.T.dot(y)
            posts[:, i] = w.T.dot(y) - self.w_ei.T.dot(inh)
            posts[posts < 0] = 0
            # posts[:, i] = w.T.dot(y)

        return 180 * np.argmax(posts, axis=1) / n_angles

    def evolve_W(self, W_old, t, type):
        """
        
        """
        # print(f"timestep: {t}, w ={W_old}")
        H = self.hebbian_component(self.N, W_old, self.w_if, self.w_ei, self.theta_stim, type=type)
        eta = np.random.randn(self.N, self.N)
        prop_function = self.propensity(W_old, self.a)
        hebb = self.hebb_scaling * H * prop_function
        rand = self.rand_scaling * eta * prop_function
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
            u = circular_gaussian(self.N, theta)
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
            u = circular_gaussian(self.N, theta)
            i = self.w_if.T.dot(u)
            e = w_ef.T.dot(u) - self.w_ei.T.dot(i)
            e[e < 0] = 0
            responses[:, stim_num] = e
        
        # Compute pairwise correlations across stimuli
        corr = np.corrcoef(responses)  # N x N correlation matrix
        return corr

    def summed_corr_over_time(self, W):
        """
        computes activity correlations between neurons based on given preferred orientation array over time
        """

        theta_list = np.linspace(0, 180, 100)
        corr_over_time = np.zeros((W.shape[2]))

        # take a theta value and compute the activity of each neuron based on its preferred orientation and the circular gaussian function
        for t in tqdm(range(W.shape[2])):
            corr_over_stim = np.zeros((self.N, self.N, len(theta_list)))    
            for stim_num, theta in enumerate(theta_list):
                u = circular_gaussian(self.N, theta)
                i = self.w_if.T.dot(u)
                e = W[:, :, t].T.dot(u) - self.w_ei.T.dot(i)
                e[e < 0] = 0
                corr_over_stim[:, :, stim_num] = self.corr_function(e)

            corr_avg = np.mean(corr_over_stim, axis=2)
            corr_upper_triangle = corr_avg[np.triu_indices(self.N, k=1)]
            corr_over_time[t] = np.mean(corr_upper_triangle)

        return corr_over_time

    def save_hdf5(self, POs, location):

        drift_mag, drift_rate, convergence = self.get_metrics(self.N, self.n_days, self.theta_stim, POs)
        f = h5.File(location+'_drift_simulations.hdf5', 'a')
        i = len(f.keys())
        sim_group = f.create_group(f"sim_{i}")

        weights = sim_group.create_group("weights")
        weights.create_dataset("w_ef_init", data=self.w_ef_init)
        weights.create_dataset("w_if", data=self.w_if)
        weights.create_dataset("w_ei", data=self.w_ei)
        weights.create_dataset("W", data=self.W)

        drift_metrics = sim_group.create_group("drift_metrics")
        drift_metrics.create_dataset("drift_mag", data=drift_mag)
        drift_metrics.create_dataset("drift_rate", data=drift_rate)
        drift_metrics.create_dataset("convergence", data=convergence)
        
        sim_group.attrs['i_scale'] = self.inh_scale
        sim_group.attrs['hebb_scale'] = self.hebb_scaling
        sim_group.attrs['rand_scale'] = self.rand_scaling
        sim_group.attrs['inh_type'] = self.inh_type

        f.close()

    def params_to_string(self):

        params_dict = {'a': self.a,
                       'learning_rate': self.learning_rate,
                       'n_steps': self.n_steps,
                       'N_inh': self.N_inh,
                       'n_days': self.n_days,
                       }


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

            r_f = self.circular_gaussian(self.N, theta, amp=0.62, sigma=5, baseline=0)
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


    def estimate_initial_activity(self, probe_angle=60):
        """
        Returns the activity of each neuron to the probe stimulus at a given day
        """

        r_f = self.circular_gaussian(self.N, probe_angle, amp=0.62, sigma=5, baseline=0)
        r_i = self.w_if.T.dot(r_f)
        r_e = self.w_ef_baseline.T.dot(r_f) - self.w_ei.T.dot(r_i)
        r_e[r_e < 0] = 0

        return r_e, r_i, r_f
    
    def estimate_initial_tuning_inh(self):
        """
        Returns the tuning curves of the inhibitory population to the test angles
        """
        tuning_curves_inh = np.zeros((self.N_inh, self.n_test_angles))
        theta_list = np.linspace(0, 180, self.n_test_angles)

        for theta_idx, theta in enumerate(theta_list):

            r_f = self.circular_gaussian(self.N, theta, amp=0.62, sigma=5, baseline=0)
            r_i = self.w_if.T.dot(r_f)
            tuning_curves_inh[:, theta_idx] = r_i

        return tuning_curves_inh