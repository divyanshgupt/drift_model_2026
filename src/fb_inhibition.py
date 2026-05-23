import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import scipy.stats as stats
from helper_functions import circular_gaussian
from network import FeedForward

from matplotlib.animation import FuncAnimation

class FeedbackInhibition(FeedForward):
    def __init__(self, inh="on", vars_ie_mean=3, vars_ee_mean=2, EE_conn=False, seed=100, input_sigma = 15):

        super().__init__(N=500, N_inh = 500, a = 10, prop_shift=0, theta_stim=90, n_test_angles=100,
                 learning_rate = 0.01, n_days = 28, n_norm_per_day = 1,
                 n_steps_per_norm = 30, init_steps = 300, hebb_scaling = 0.3,
                 rand_scaling = 1, inh_scale = 1, vars_if_mean=3, vars_ei_mean=3, inh=inh, inh_type="co-tuned", 
                 norm = True, pre_run=False, seed = seed, set_seed=True)

        self.vars_ie_mean = vars_ie_mean
        self.vars_ee_mean = vars_ee_mean
        self.input_sigma = input_sigma
        self.EE_conn = EE_conn

        self.setup_params()
        self.setup_variables()
        self.setup_weights()

        self.w_ef_baseline = self.w_ef_init
        

    def setup_params(self):
        self.tau_E = 10
        self.tau_I = 10

        self.dt = 1
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

    def setup_weights(self):

        self.vars_ie = np.random.lognormal(mean=self.vars_ie_mean, sigma=0.6, size=self.N)
        self.vars_ee = np.random.lognormal(mean=self.vars_ee_mean, sigma=0.6, size=self.N)

        self.w_ef_init = self.initialise_w_ef(self.N, self.vars_ef)

        if self.EE_conn:
            self.w_ee = self.initialise_w_ee(self.N, self.vars_ee)
        else:
            self.w_ee = np.zeros((self.N, self.N))
        self.w_if = self.initialise_w_if(self.N_inh, self.vars_if, self.inh_type)
        self.w_ei = self.initialise_w_ei(self.N_inh, self.vars_ei, self.inh_type)
        self.w_ie = self.initialise_w_ie(self.N, self.vars_ie, self.inh_type)

        return None
    
    def transfer_E(self, u):
        return np.maximum(0, u)

    def transfer_I(self, u):
        return np.maximum(0, u)


    def initialise_w_ie(self, N_exc, vars_ie, inh_type="co-tuned"):
        """
        excitatory pop (E) to inh pop (I) weights
        initialised as gaussian tuning curves with varyings widths
        """

        if inh_type == "co-tuned":
            x = np.linspace(0, 180, self.N_inh, endpoint=False) # post synaptic neuron tuning peaks (in presynaptic neuron space)
            matrix = np.zeros((self.N, self.N_inh))
            for i in range(self.N_inh):
                matrix[:, i] = stats.norm.pdf(x, loc=x[i], scale=vars_ie[i]) + stats.norm.pdf(x, loc=x[i]+180, scale=vars_ie[i]) + stats.norm.pdf(x, loc=x[i]-180, scale=vars_ie[i])
            w_ie = matrix/self.N
            w_ie /= np.sum(w_ie, axis=0)
        
        return w_ie
    
    def initialise_w_ee(self, N_exc, vars_ee):
        """
        setup e to e connections
        
        """
        w_ee = np.zeros((self.N, self.N))
        x = np.linspace(0, 180, self.N, endpoint=False) # post synaptic neuron tuning peaks (in presynaptic neuron space)
        for i in range(self.N): # for each post syn neuron
            w_ee[:, i] = stats.norm.pdf(x, loc=x[i], scale=vars_ee[i]) + stats.norm.pdf(x, loc=x[i]+180, scale=vars_ee[i]) + stats.norm.pdf(x, loc=x[i]-180, scale=vars_ee[i])
        w_ee = w_ee/self.N # divide by number of inputs
        w_ee /= np.sum(w_ee, axis=0) # normalize each column to sum to 1

        return w_ee

    # def pre_run_fb_inh():
    #     self.w_ef_baseline = 

    #     return None

    def run(self):

        if self.set_seed:
            np.random.seed(self.seed)
        
        self.POs = []
        # self.W = np.zeros((self.N, self.N, self.n_steps+1))
        self.W = np.zeros((self.N, self.N, self.n_stim_total+1))
        self.W[:, :, 0] = self.w_ef_baseline
        self.w_ef = self.w_ef_baseline.copy()

        self.W_ee = np.zeros((self.N, self.N, self.n_stim_total+1))
        self.W_ee[:, :, 0] = self.w_ee.copy()
        
        for stim_idx in tqdm(range(self.n_stim_total)):

            # new stimulus every presentation
            self.theta_stim = np.random.uniform(0, 180)
            self.r_F = circular_gaussian(self.N, self.theta_stim, amp=0.62, sigma=self.input_sigma, baseline=0)
            
            # run dynamics for T_seq steps until activity settles
            for t in range(self.T_seq):
                self.step()

            # single weight update based on settled activity
            self.w_ef = self.evolve_W_fb_inh(self.w_ef, stim_idx)
            self.W[:, :, stim_idx+1] = self.w_ef.copy()

            # self.w_ee = self.evolve_W_ee(self.w_ee, stim_idx)
            # self.W_ee[:, :, stim_idx+1] = self.w_ee.copy()
        # for t in tqdm(range(self.n_steps)):
            
        #     # switch input every T_seq steps
        #     if t % self.T_seq == 0:
        #         self.theta_stim = np.random.uniform(0, 180)
        #         self.r_F = circular_gaussian(self.N, self.theta_stim, amp=0.62, sigma=self.input_sigma, baseline=0)
        #     self.step()
        #     self.w_ef = self.evolve_W_fb_inh(self.w_ef, t)
        #     self.W[:, :, t+1] = self.w_ef.copy()

        return None

    def step(self):

        du_E = (1/self.tau_E)*(-self.u_E + self.w_ef.T@self.r_F + self.w_ee.T@self.r_E - self.w_ei.T@self.r_I)
        du_I = (1/self.tau_I)*(-self.u_I + self.w_if.T@self.r_F + self.w_ie.T@self.r_E)

        self.u_E += du_E * self.dt
        self.u_I += du_I * self.dt

        self.r_E = self.transfer_E(self.u_E)
        self.r_I = self.transfer_I(self.u_I)
        return None

    def evolve_W_fb_inh(self, w_old, stim_idx):
        """
        Evolve the feedforward weights according to hebbian plasticity rule for feedback inhibition model
        """

        H = self.hebbian_component_fb_inh(self.r_F, self.r_E)
        eta = np.random.randn(self.N, self.N)
        prop_function = self.propensity(w_old, self.a)
        hebb = self.hebb_scaling * H * prop_function
        rand = self.rand_scaling * eta * prop_function
        w_new = w_old + self.learning_rate * (hebb + rand)


        # normalize every n_stim_per_norm stimulus presentations
        if stim_idx % self.n_stim_per_norm == 0:
            w_new = self.normalisation(w_new)

        # record POs once per day
        if stim_idx % self.n_stim_per_day == 0:
            PO = self.get_preferred_orientations_recurrent(self.N, w_new)
            self.POs.append(PO)

        # if t % self.n_steps_per_norm == 0:
        #     w_new = self.normalisation(w_new)

        #     if t % (self.n_steps_per_norm * self.n_norm_per_day) == 0:
        #         PO = self.get_preferred_orientations(self.N, w_old, n_angles=self.n_test_angles)
        #         self.POs.append(PO)

        return w_new
    
    def evolve_W_ee(self, w_old, stim_idx):
        """
        Evolve the recurrent excitatory weights according to hebbian plasticity rule for feedback inhibition model
        """

        H = self.hebbian_component_fb_inh(self.r_E, self.r_E, type="ee")
        eta = np.random.randn(self.N, self.N)
        prop_function = self.propensity(w_old, self.a)
        hebb = self.hebb_scaling * H * prop_function
        rand = self.rand_scaling * eta * prop_function
        w_new = w_old + self.learning_rate * (hebb + rand)


        # normalize every n_stim_per_norm stimulus presentations
        if stim_idx % self.n_stim_per_norm == 0:
            w_new = self.normalisation(w_new)

        return w_new

    def get_preferred_orientations_recurrent(self, N, w):
        """
        Estimate the preferred orientation of each neuron given the feedforward weights w and recurrent weights w_ee
        """

        activity = np.zeros((N, self.n_test_angles))
        theta_list = np.linspace(0, 180, self.n_test_angles, endpoint=False)

        for theta_idx, theta in enumerate(theta_list):
            r_f = circular_gaussian(self.N, theta, amp=0.62, sigma=self.input_sigma, baseline=0)

            u_E = np.zeros(self.N)
            u_I = np.zeros(self.N_inh)
            r_E = np.zeros(self.N)
            r_I = np.zeros(self.N_inh)

            for t in range(self.T_seq):
                du_E = (1/self.tau_E)*(-u_E + w.T@r_f + self.w_ee.T@r_E - self.w_ei.T@r_I)
                du_I = (1/self.tau_I)*(-u_I + self.w_if.T@r_f + self.w_ie.T@r_E)

                u_E += du_E * self.dt
                u_I += du_I * self.dt

                r_E = self.transfer_E(u_E)
                r_I = self.transfer_I(u_I)

            activity[:, theta_idx] = r_E
        PO_estimate = theta_list[np.argmax(activity, axis=1)]
        return PO_estimate
        


    def hebbian_component_fb_inh(self, r_F, r_E, type="baseline"):
        """
        Calculate the hebbian component of the weight change for feedback inhibition model
        """
        # if type == "baseline":  
        #     theta = np.random.uniform(0, 180)
        
        H = np.outer(r_F, r_E)

        return H
    
    def estimate_tuning_curves_at_day(self, day):
        """
        Estimate the tuning curves for each neuron at a given day
        """

        tuning_curves = np.zeros((self.N, self.n_test_angles))
        theta_list = np.linspace(0, 180, self.n_test_angles, endpoint=False)

        # time_idx = day * self.n_steps_per_norm * self.n_norm_per_day
        time_idx = day * self.n_stim_per_day
        w_ef = self.W[:, :, time_idx]
        w_ee = self.W_ee[:, :, time_idx]

        for theta_idx, theta in enumerate(theta_list):
            r_f = circular_gaussian(self.N, theta, amp=0.62, sigma=self.input_sigma, baseline=0)

            u_E = np.zeros(self.N)
            u_I = np.zeros(self.N_inh)
            r_E = np.zeros(self.N)
            r_I = np.zeros(self.N_inh)

            for t in range(self.T_seq):
                du_E = (1/self.tau_E)*(-u_E + w_ef.T@r_f + w_ee.T@r_E - self.w_ei.T@r_I)
                du_I = (1/self.tau_I)*(-u_I + self.w_if.T@r_f + self.w_ie.T@r_E)

                u_E += du_E * self.dt
                u_I += du_I * self.dt

                r_E = self.transfer_E(u_E)
                r_I = self.transfer_I(u_I)

            tuning_curves[:, theta_idx] = r_E
        
        return tuning_curves

    def estimate_activity_at_day(self, day, theta):
        
        time_idx = day * self.n_stim_per_day
        w_ef = self.W[:, :, time_idx]
        w_ee = self.W_ee[:, :, time_idx]

        r_f = self.circular_gaussian(self.N, theta, amp=0.62, sigma=self.input_sigma, baseline=0)

        u_E = np.zeros(self.N)
        u_I = np.zeros(self.N_inh)
        r_E = np.zeros(self.N)
        r_I = np.zeros(self.N_inh)

        for t in range(self.T_seq):
            du_E = (1/self.tau_E)*(-u_E + w_ef.T@r_f + w_ee.T@r_E - self.w_ei.T@r_I)
            du_I = (1/self.tau_I)*(-u_I + self.w_if.T@r_f + self.w_ie.T@r_E)

            u_E += du_E * self.dt
            u_I += du_I * self.dt

            r_E = self.transfer_E(u_E)
            r_I = self.transfer_I(u_I)
        
        return r_E, r_I
    
    def plot_initial_weight_matrices(self):
        
        fig, axs = plt.subplots(1, 4, figsize=(15, 5))
        im0 = axs[0].imshow(self.w_ef_baseline, aspect='auto', origin='lower')
        axs[0].set_title('E-F Feedforward Weights')
        plt.colorbar(im0, ax=axs[0])

        im1 = axs[1].imshow(self.w_if, aspect='auto', origin='lower')
        axs[1].set_title('I-F Feedforward Weights')
        plt.colorbar(im1, ax=axs[1])

        im2 = axs[2].imshow(self.w_ie, aspect='auto', origin='lower')
        axs[2].set_title('I-E Feedback Weights')
        plt.colorbar(im2, ax=axs[2])

        im3 = axs[3].imshow(self.w_ei, aspect='auto', origin='lower')
        axs[3].set_title('E-I Feedback Weights')
        plt.colorbar(im3, ax=axs[3])

        plt.tight_layout()
        plt.show()
        return None
    
    def plot_pop_activity_at_day(self, day, theta):
        r_E, r_I = self.estimate_activity_at_day(day, theta)

        fig, axs = plt.subplots(1, 3, figsize=(10, 3), dpi=200)

        r_F = self.circular_gaussian(self.N, theta, amp=0.62, sigma=self.input_sigma, baseline=0)

        axs[0].plot(r_F)
        axs[0].set_title('Feedforward Input')
        axs[0].set_xlabel('Neuron Index')

        axs[1].plot(r_E)
        axs[1].set_title('E Pop. Activity')
        axs[1].set_xlabel('Neuron Index')

        axs[2].plot(r_I)
        axs[2].set_title('I Pop. Activity')
        axs[2].set_xlabel('Neuron Index')


        fig.tight_layout()
        fig.show()

        return None
    
