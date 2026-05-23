import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from helper_functions import circular_gaussian
from network import FeedForward

from matplotlib.animation import FuncAnimation


class TimeVaryingFeedForward(FeedForward):
    def __init__(self, N=500, N_inh = 500, a = 10, prop_shift=0, theta_stim=90, n_test_angles=1000,
                 learning_rate = 0.01, n_days = 28, n_norm_per_day = 1,
                 n_steps_per_norm = 30, init_steps = 300, hebb_scaling = 0.3,
                 rand_scaling = 1, inh_scale = 1, vars_if_mean=3, vars_ei_mean=3, inh="on", inh_type="co-tuned", 
                 norm = True, seed = 100, set_seed=True, time_varying=False, partial_modulation=False, mod_frac=0.3,
                 inh_modulation="biased", inh_mod_scale=0.7, homeo="off", inh_start=7, inh_end=14, input_sigma=15,
                 pre_run=False, num_clusters=2):
        
        super().__init__(N, N_inh, a, prop_shift, theta_stim, n_test_angles, learning_rate, n_days,
                         n_norm_per_day, n_steps_per_norm, init_steps,
                         hebb_scaling, rand_scaling, inh_scale, vars_if_mean, vars_ei_mean,
                         inh, inh_type, norm, seed, set_seed, pre_run)
        
        self.time_varying = time_varying
        self.partial_modulation = partial_modulation
        self.inh_modulation = inh_modulation
        self.inh_modulation_scale = inh_mod_scale
        self.mod_frac = mod_frac
        self.num_clusters = num_clusters
        self.input_sigma = input_sigma


        if self.partial_modulation: # if modulating a subset of neurons
            self.inh_modulation_scale = 1 - ((1-self.inh_modulation_scale) / self.mod_frac) # scale up the modulation for the subset of neurons so that the overall inhibition is reduced by 0.6
            print(f"Scaling up modulation for subset of neurons to {self.inh_modulation_scale} to achieve overall modulation of {inh_mod_scale} with mod_frac of {self.mod_frac}")
            self.setup_partial_modulation()

        self.setup_inhibition_timeline(homeo=homeo, inh_start = inh_start, inh_end = inh_end)

        print(f"Inh scale timeline: {self.inh_scale_timeline}")

    def setup_inhibition_timeline(self, homeo="off", inh_start = 1, inh_end = 7):

        if homeo == "off":
            # step current inhibition

            self.inh_scale_timeline = np.ones(self.n_days)
            if self.time_varying:
                self.inh_scale_timeline[inh_start:inh_end] = self.inh_modulation_scale

        # Not tested yet:
        # elif homeo == "on": 
        #     # homeostatic inhibition that exponentially decays back to baseline
        #     self.inh_scale_timeline = np.ones(self.n_days)
        #     for t in range(inh_start, inh_end):
        #         self.inh_scale_timeline[t] = self.inh_scale_timeline[t-1] * 0.5

        return None

    def setup_partial_modulation(self):

        if self.inh_modulation == 'random':
            self.inh_mask = np.random.binomial(1, self.mod_frac, size=self.N_inh) # mod_frac% of inhibitory neurons are modulated
        elif self.inh_modulation == 'biased':
            # modulate 
            self.inh_mask = np.zeros(self.N_inh)
            self.inh_mask[:int(self.mod_frac*self.N_inh)] = 1 # first mod_frac% of inhibitory neurons are modulated
        elif self.inh_modulation == 'biased_structured':
            self.inh_mask = np.zeros(self.N_inh)
            band_size = int(self.N_inh * self.mod_frac / self.num_clusters)
            spacing = self.N_inh // self.num_clusters
            for k in range(self.num_clusters):
                start = k * spacing
                self.inh_mask[start:start + band_size] = 1
            print(self.inh_mask)

        return None

    def run(self, type='baseline'):

        if self.set_seed:
            np.random.seed(self.seed)

        self.POs = []
        self.W = np.zeros((self.N, self.N, self.n_steps+1))
        self.W[:, :, 0] = self.w_ef_baseline
        self.w_ei_initial = self.w_ei.copy()

        for t in tqdm(range(self.n_steps)):
            # set inhibition based on day
            if self.time_varying:
                day = t // (self.n_norm_per_day * self.n_steps_per_norm)
                self.inh_scale = self.inh_scale_timeline[day]
                # self.w_ei = self.inh_scale * self.w_ei_initial

            self.W[:, :, t+1] = self.evolve_W_time_varying_inh(self.W[:, :, t], t, type)

        return None
    
    def evolve_W_time_varying_inh(self, W_old, t, type):
        "evolve weights based on hebbian component, propensity function and noise, with inhibition that changes over time"
        H = self.hebbian_component_time_varying_inh(self.N, W_old, self.w_if, self.w_ei, self.theta_stim, type=type)
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
    
    def hebbian_component_time_varying_inh(self, N, w_ef, w_if, w_ei, theta_stim, type):
        """
        computes post-synaptic activity in the E & I population
        scales inhibition by inh_scale which changes over time to simulate CNO effects
        returns the hebbian outer product for E & F 

        """
        if type == "baseline" or type == "test" : theta = np.random.uniform(0, 180)
        elif type == "stripe_rearing": theta = theta_stim

        # r_f = self.circular_gaussian(N, theta, amp=0.62, sigma=60, baseline=0)
        r_f = circular_gaussian(N, theta, amp=0.62, sigma=self.input_sigma, baseline=0)

        if self.partial_modulation:
            r_i = np.zeros(self.N_inh)
            r_i[self.inh_mask == 1] = self.inh_scale * w_if.T.dot(r_f)[self.inh_mask == 1] # this is the only different part from the original hebbian component function - inhibition is scaled by inh_scale which changes over time, and modulated by inh_mask which determines which inhibitory neurons are modulated
            r_i[self.inh_mask == 0] = w_if.T.dot(r_f)[self.inh_mask == 0] # unmodulated inhibitory neurons have normal activity
        else:
            r_i = self.inh_scale * w_if.T.dot(r_f) # this is the only different part from the original hebbian component function - inhibition is scaled by inh_scale which changes over time

        r_e = w_ef.T.dot(r_f) - w_ei.T.dot(r_i)
        r_e[r_e < 0] = 0

        return np.outer(r_f, r_e)

    # def activity_for_theta(self, theta):

    #     r_f = self.circular_gaussian(self.N, theta, amp=0.62, sigma=60, baseline=0)
    #     if self.partial_modulation:
    #         r_i = np.zeros(self.N_inh)
    #         r_i[self.inh_mask == 1] = self.inh_scale * self.w_if.T.dot(r_f)[self.inh_mask == 1] # this is the only different part from the original hebbian component function - inhibition is scaled by inh_scale which changes over time, and modulated by inh_mask which determines which inhibitory neurons are modulated
    #         r_i[self.inh_mask == 0] = self.w_if.T.dot(r_f)[self.inh_mask == 0] # unmodulated inhibitory neurons have normal activity
    #     else:
    #         r_i = self.inh_scale * self.w_if.T.dot(r_f) # this is the only different part from the original hebbian component function - inhibition is scaled by inh_scale which changes over time       

    def estimate_tuning_curves_at_day(self, day, probe_sigma=None):

        if probe_sigma is None:
            probe_sigma = self.input_sigma

        tuning_curves = np.zeros((self.N, self.n_test_angles))
        theta_list = np.linspace(0, 180, self.n_test_angles)

        day_idx = day * self.n_norm_per_day * self.n_steps_per_norm
        inh_scale = self.inh_scale_timeline[day]

        for theta_idx, theta in enumerate(theta_list):
            u = circular_gaussian(self.N, theta, sigma=probe_sigma)

            if self.partial_modulation:
                i = np.zeros(self.N_inh)
                i[self.inh_mask == 1] = inh_scale * self.w_if.T.dot(u)[self.inh_mask == 1] # this is the only different part from the original hebbian component function - inhibition is scaled by inh_scale which changes over time, and modulated by inh_mask which determines which inhibitory neurons are modulated
                i[self.inh_mask == 0] = self.w_if.T.dot(u)[self.inh_mask == 0] # unmodulated inhibitory neurons have normal activity
            else:
                i = inh_scale * self.w_if.T.dot(u)

            e = self.W[:, :, day_idx].T.dot(u) - self.w_ei.T.dot(i)
            e[e < 0] = 0
            tuning_curves[:, theta_idx] = e

        return tuning_curves
    
    def estimate_tuning_widths_at_day(self, day, probe_sigma=None):

        tuning_curves = self.estimate_tuning_curves_at_day(day, probe_sigma=probe_sigma)
        theta_list = np.linspace(0, 180, self.n_test_angles)

        widths = np.zeros(self.N)
        
        for i in range(self.N):
            curve = tuning_curves[i, :]
            curve /= np.max(curve)
            half_max = 0.5
            indices_above_half_max = np.where(curve >= half_max)[0]
            if len(indices_above_half_max) > 0:
                widths[i] = len(indices_above_half_max) * (180 / self.n_test_angles)


        # interpolate to get more precise estimate of tuning width
        # for i in range(self.N):
        #     curve = tuning_curves[i, :] / (np.max(tuning_curves[i, :]) + 1e-10)
        #     crossings = np.where(np.diff(np.sign(curve - 0.5)))[0]
        #     if len(crossings) >= 2:
        #         # linear interpolation of each crossing
        #         def interp_crossing(idx):
        #             x0, x1 = theta_list[idx], theta_list[idx + 1]
        #             y0, y1 = curve[idx] - 0.5, curve[idx + 1] - 0.5
        #             return x0 - y0 * (x1 - x0) / (y1 - y0)
        #         left = interp_crossing(crossings[0])
        #         right = interp_crossing(crossings[-1])
        #         widths[i] = right - left

        return widths


    def estimate_avg_exc_activity_at_day(self, day):

        "to show change in exc. activity with CNO"

        theta_list = np.linspace(0, 180, self.n_test_angles)
        day_idx = day * self.n_norm_per_day * self.n_steps_per_norm
        activity = np.zeros((self.N, self.n_test_angles))
        avg_activity = np.zeros(self.N) # across angles
        inh_scale = self.inh_scale_timeline[day]

        for theta_idx, theta in enumerate(theta_list):

            u = circular_gaussian(self.N, theta, sigma=self.input_sigma)

            if self.partial_modulation:
                i = np.zeros(self.N_inh)
                i[self.inh_mask == 1] = inh_scale * self.w_if.T.dot(u)[self.inh_mask == 1] # this is the only different part from the original hebbian component function - inhibition is scaled by inh_scale which changes over time, and modulated by inh_mask which determines which inhibitory neurons are modulated
                i[self.inh_mask == 0] = self.w_if.T.dot(u)[self.inh_mask == 0] # unmodulated inhibitory neurons have normal activity
            else:
                i = inh_scale * self.w_if.T.dot(u)

            e = self.W[:, :, day_idx].T.dot(u) - self.w_ei.T.dot(i)
            e[e < 0] = 0
            activity[:, theta_idx] = e
        avg_activity[:] = np.mean(activity, axis=1)
    

        return avg_activity
        

    def estimate_avg_inh_activity_at_day(self, day):
        
        "to show change in inh. activity with CNO"

        theta_list = np.linspace(0, 180, self.n_test_angles)
        day_idx = day * self.n_norm_per_day * self.n_steps_per_norm
        activity = np.zeros((self.N_inh, self.n_test_angles))
        avg_inh_activity = np.zeros(self.N_inh) # across angles

        inh_scale = self.inh_scale_timeline[day]
        
        for theta_idx, theta in enumerate(theta_list):
            u = circular_gaussian(self.N, theta, sigma=self.input_sigma)
            if self.partial_modulation:
                i = np.zeros(self.N_inh)
                i[self.inh_mask == 1] = inh_scale * self.w_if.T.dot(u)[self.inh_mask == 1] # this is the only different part from the original hebbian component function - inhibition is scaled by inh_scale which changes over time, and modulated by inh_mask which determines which inhibitory neurons are modulated
                i[self.inh_mask == 0] = self.w_if.T.dot(u)[self.inh_mask == 0] # unmodulated inhibitory neurons have normal activity
            else:
                i = inh_scale * self.w_if.T.dot(u)
            e = self.W[:, :, day_idx].T.dot(u) - self.w_ei.T.dot(i)
            e[e < 0] = 0
            activity[:, theta_idx] = i
        avg_inh_activity[:] = np.mean(activity, axis=1)
    
        return avg_inh_activity
    

    # def estimate_POs_at_day(self, day):

    def estimate_pop_activity_at_day(self, day, theta_stim):

        exc_activity = np.zeros(self.N)
        inh_activity = np.zeros(self.N_inh)

        day_idx = day * self.n_norm_per_day * self.n_steps_per_norm

        u = circular_gaussian(self.N, theta_stim, sigma=self.input_sigma)
        
        if self.partial_modulation:
            i = np.zeros(self.N_inh)
            i[self.inh_mask == 1] = self.inh_scale_timeline[day] * self.w_if.T.dot(u)[self.inh_mask == 1] # this is the only different part from the original hebbian component function - inhibition is scaled by inh_scale which changes over time, and modulated by inh_mask which determines which inhibitory neurons are modulated
            i[self.inh_mask == 0] = self.w_if.T.dot(u)[self.inh_mask == 0] # unmodulated inhibitory neurons have normal activity
        else:
            i = self.inh_scale_timeline[day] * self.w_if.T.dot(u)

        e = self.W[:, :, day_idx].T.dot(u) - self.w_ei.T.dot(i)
        e[e < 0] = 0

        exc_activity[:] = e
        inh_activity[:] = i

        return exc_activity, inh_activity


    def plot_inh_scale_timeline(self, color='black', CNO=True):

        day_range = np.arange(self.n_days)
        plt.figure(figsize=(4, 2), dpi=300)
        plt.plot(day_range, self.inh_scale_timeline, c=color, ls='-')
        # draw a vertical shaded region to indicate when inhibition is reduced
        inh_start_day = 6
        inh_end_day = 14
        if CNO:
            plt.axvspan(inh_start_day, inh_end_day, color='yellow', alpha=0.3, label='CNO')
            plt.legend(frameon=False)

        plt.xlabel('Day')
        plt.ylabel('Inhibition scale')
        plt.title('Inhibition scale timeline')
        plt.ylim(0, 1.5)
        plt.tight_layout()
        plt.show()


    def estimate_pairwise_correlations_at_day(self, day):
        
        """
        to show change in pairwise correlations with CNO"

        uses np.corrcoef to compute pairwise correlations between excitatory neurons based on their activity across different stimulus angles
        """


        theta_list = np.linspace(0, 180, self.n_test_angles)
        day_idx = day * self.n_norm_per_day * self.n_steps_per_norm
        activity = np.zeros((self.N, self.n_test_angles))

        for theta_idx, theta in enumerate(theta_list):
            u = circular_gaussian(self.N, theta, sigma=self.input_sigma)
            i = self.w_if.T.dot(u)
            e = self.W[:, :, day_idx].T.dot(u) - self.w_ei.T.dot(i)
            e[e < 0] = 0
            activity[:, theta_idx] = e

        corr_matrix = np.corrcoef(activity)
        upper_tri_indices = np.triu_indices(self.N, k=1)
        pairwise_correlations = corr_matrix[upper_tri_indices]

        return pairwise_correlations
    
    def population_activity_for_theta_across_time(self):

        activity = np.zeros((self.N, self.n_steps))
        theta_stim = self.theta_stim

        for t in tqdm(range(self.n_steps)):
            u = circular_gaussian(self.N, theta_stim, sigma=self.input_sigma)
            i = self.w_if.T.dot(u)
            e = self.W[:, :, t].T.dot(u) - self.w_ei.T.dot(i)
            e[e < 0] = 0
            activity[:, t] = e

        return activity
    
    def plot_animation_pop_activity(self, nrn_idx = 249, title=""):

        """
        panels:
        - preferred orientation of one neuron over time
        - tuning curve of that neuron at each time point
        - distribution of preferred orientations across the population at each time point
        - population tuning curves at each day
        """

        fig, axs = plt.subplots(2, 2, figsize=(8, 6), dpi=300)

        PO_over_time_line, = axs[0, 0].plot([], [], ls='-', marker='o', ms=4)
        axs[0, 0].set_title(f"Neuron: {nrn_idx}")
        axs[0, 0].set_xlim(0, self.n_days)
        axs[0, 0].set_ylim(0, 180)
        axs[0, 0].set_ylabel("PO")

        tuning_curve_line, = axs[0, 1].plot([], [], ls='-', marker='o', ms=4)
        axs[0, 1].set_title("Tuning curve")
        axs[0, 1].set_xlim(0, 180)
        axs[0, 1].set_ylim(0, 2)
        axs[0, 1].set_xlabel("Stimulus angle")
        axs[0, 1].set_ylabel("Activity")

        distribution_line = axs[1, 0].hist([])
        axs[1, 0].set_title("PO distribution")
        axs[1, 0].set_xlim(0, 180)
        axs[1, 0].set_ylim(0, 50)
        axs[1, 0].set_xlabel("Preferred orientation")
        axs[1, 0].set_ylabel("Frequency")

        tuning_curve_tiles = axs[1, 1].plot([], [], ls='-', marker='o', ms=4)
        axs[1, 1].set_title("Population tuning curves")
        # axs[1, 1].set_xlim(0, 180)
        # axs[1, 1].set_ylim(0, self.N*0.025 + 2)
        axs[1, 1].set_xlabel("Stimulus angle")

        fig.suptitle("Day: 0")
        fig.tight_layout()

        def update(day):

            frame = day * self.n_norm_per_day * self.n_steps_per_norm
            weights = self.W[:, :, frame]
            PO = self.POs[day]
            tuning_curves = self.estimate_tuning_curves_at_day(day)

            PO_over_time_line.set_data(np.arange(day+1), [self.POs[d][nrn_idx] for d in range(day+1)])
            tuning_curve_line.set_data(np.linspace(0, 180, self.n_test_angles), tuning_curves[nrn_idx, :])

            axs[1, 0].cla()
            distribution_line = axs[1, 0].hist(PO, bins=20, range=(0, 180), color='blue', alpha=0.7)
            axs[1, 0].set_title("PO distribution")
            axs[1, 0].set_xlim(0, 180)
            axs[1, 0].set_ylim(0, 50)
            axs[1, 0].set_xlabel("Preferred orientation")
            axs[1, 0].set_ylabel("Frequency")

            axs[1, 1].cla()
            for i in range(0, self.N, 30):
                shift = 0.025
                tuning_curve_tiles = axs[1, 1].plot(np.linspace(0, 180, self.n_test_angles), (shift*i) + tuning_curves[i, :], ms=2, color='black')
            axs[1, 1].set_title("Population tuning curves")
            axs[1, 1].set_xlim(0, 180)
            axs[1, 1].set_ylim(0, self.N*shift + 2)
            axs[1, 1].set_xlabel("Stimulus angle")

            fig.suptitle(f"Day {day+1}")

        ani = FuncAnimation(fig, update, frames=self.n_days, repeat=False)

        ani.save(f'{title}_population_activity_animation.gif', writer='pillow', fps=3)
        return None

