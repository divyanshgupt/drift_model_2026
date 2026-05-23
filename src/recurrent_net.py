import numpy as np
from matplotlib import pyplot as plt
import h5py as h5
from tqdm import tqdm


class recurrent_network():
    def __init__(self, normalization=True, plot_weights_each_step=False, seed=0):

        self.setup_params()
        self.setup_state()
        self.setup_record_vars()
        self.setup_weights()
        self.setup_input()
        self.normalization = normalization
        self.plot_weights_each_step = plot_weights_each_step

    def setup_params(self):
        self.tau_e = 20
        self.tau_i = 17
        self.dt = 10

        self.N_E = 10
        self.N_I = 10
        self.N_F = 40

        self.a = 0.04
        self.b = 0
        self.n = 2

        self.c = 1.0
        self.A_F = 35
        self.sigma_F = 12 * np.pi / 180  # radians

        self.mu_W = 0.2
        self.sigma_W = 0.1
        
        self.W_EE_norm = 2
        self.W_EI_norm = 0.8
        self.W_IE_norm = 2
        self.W_II_norm = 0.8

        self.epsilon_EE = 2 * 1e-9 # ms-1
        self.epsilon_EF = 2 * 1e-9 # ms-1
        self.epsilon_EI = 4 * 1e-9 # ms-1
        self.epsilon_IF = 3 * 1e-9 # ms-1
        self.epsilon_IE = 3 * 1e-9 # ms-1
        self.epsilon_II = 5 * 1e-9 # ms-1

        self.input_presentation_time = 200  # ms
        self.duration = 100 * 1000  # ms

        self.num_timesteps = self.duration // self.dt
        self.record_freq = 1
        return None

    def setup_state(self): 

        self.u_e = np.zeros(self.N_E)
        self.u_i = np.zeros(self.N_I)
        self.r_E = np.zeros(self.N_E)
        self.r_I = np.zeros(self.N_I)
        self.r_F = np.zeros(self.N_F)

        return None
    
    def setup_record_vars(self):

        self.u_e_record = []
        self.u_i_record = []
        self.r_E_record = []
        self.r_I_record = []
        self.r_F_record = []

        self.W_EE_record = []
        self.W_EI_record = []
        self.W_IE_record = []
        self.W_II_record = []
        self.W_EF_record = []
        self.W_IF_record = []

        return None

    def setup_weights(self):

        # Feedforward weights
        self.W_EF = np.abs(np.random.normal(self.mu_W, self.sigma_W, (self.N_E, self.N_F)))
        self.W_EF_norm = 1
        self.W_IF = np.abs(np.random.normal(self.mu_W, self.sigma_W, (self.N_I, self.N_F)))
        self.W_IF_norm = 1

        # Recurrent excitatory weights
        self.W_EE = np.zeros((self.N_E, self.N_E))
        self.W_IE = np.zeros((self.N_I, self.N_E))
        # self.W_EE = 1e-10 * np.abs(np.random.normal(self.mu_W, self.sigma_W, (self.N_E, self.N_E)))
        # self.W_IE = 1e-10 * np.abs(np.random.normal(self.mu_W, self.sigma_W, (self.N_I, self.N_E)))

        # recurrent inhibitory weights

        # self.W_EI = np.zeros((self.N_E, self.N_I))
        # self.W_II = np.zeros((self.N_I, self.N_I))
        self.W_EI = 1e-10 * np.abs(np.random.normal(self.mu_W, self.sigma_W, (self.N_E, self.N_I)))
        self.W_II = 1e-10 * np.abs(np.random.normal(self.mu_W, self.sigma_W, (self.N_I, self.N_I)))
        # set self-connections to 0 for II
        np.fill_diagonal(self.W_II, 0)
        

        fig, axs = plt.subplots(2, 3, figsize=(10, 8))

        im0 = axs[0, 0].imshow(self.W_EF, aspect='auto', cmap='Greys')
        axs[0, 0].set_title('W_EF')
        plt.colorbar(im0, ax=axs[0, 0])
        im1 = axs[0, 1].imshow(self.W_EE, aspect='auto', cmap='Greys')
        axs[0, 1].set_title('W_EE')
        plt.colorbar(im1, ax=axs[0, 1])
        im2 = axs[0, 2].imshow(self.W_EI, aspect='auto', cmap='Greys')
        axs[0, 2].set_title('W_EI')
        plt.colorbar(im2, ax=axs[0, 2])
        im3 = axs[1, 0].imshow(self.W_IF, aspect='auto', cmap='Greys')
        axs[1, 0].set_title('W_IF')
        plt.colorbar(im3, ax=axs[1, 0])
        im4 = axs[1, 1].imshow(self.W_IE, aspect='auto', cmap='Greys')
        axs[1, 1].set_title('W_IE')
        plt.colorbar(im4, ax=axs[1, 1])
        im5 = axs[1, 2].imshow(self.W_II, aspect='auto', cmap='Greys')
        axs[1, 2].set_title('W_II')
        plt.colorbar(im5, ax=axs[1, 2])


        fig.tight_layout()
        return None
    
    def angular_dist(self, angle_1, angle_2):
        "on a circle of circumference pi"
        diff = np.abs(angle_1 - angle_2)
        return np.minimum(diff, np.pi - diff)
    
    def setup_input(self):

        # create an array of angles, with 200 ms presentation time each
        num_inputs = self.duration // self.input_presentation_time
        self.angles = np.random.random(num_inputs) * np.pi        
        self.preffered_angles_F = np.linspace(0, np.pi, self.N_F)

        return None

    def relu(self, x):
        return np.maximum(0, x)

    def transfer_F(self, u):
        return self.a * self.relu(u - self.b)**self.n
    
    def input(self, angle):
        """
        Setup activity of the pre-synaptic stimulus locked neurons
        based on the presented angle
        """
        self.r_F = self.c * self.A_F * np.exp(-self.angular_dist(angle, self.preffered_angles_F)**2 / (2 * self.sigma_F**2))
        return self.r_F

    def step(self):

        du_e = (1/self.tau_e) * (-self.u_e + self.W_EF@self.r_F + self.W_EE@self.r_E - self.W_EI@self.r_I)
        du_i = (1/self.tau_i) * (-self.u_i + self.W_IF@self.r_F + self.W_IE@self.r_E - self.W_II@self.r_I)

        self.u_e += du_e * self.dt
        self.u_i += du_i * self.dt

        self.r_E = self.transfer_F(self.u_e)
        self.r_I = self.transfer_F(self.u_i)
        return None


    def weight_update(self):

        dW_EF = self.epsilon_EF * np.outer(self.r_E, self.r_F)
        dW_EE = self.epsilon_EE * np.outer(self.r_E, self.r_E)
        np.fill_diagonal(dW_EE, 0)
        dW_EI = self.epsilon_EI * np.outer(self.r_E, self.r_I)

        dW_IF = self.epsilon_IF * np.outer(self.r_I, self.r_F)
        dW_IE = self.epsilon_IE * np.outer(self.r_I, self.r_E)
        dW_II = self.epsilon_II * np.outer(self.r_I, self.r_I)
        np.fill_diagonal(dW_II, 0)

        self.W_EF += self.dt*dW_EF
        self.W_EE += self.dt*dW_EE
        self.W_EI += self.dt*dW_EI

        self.W_IF += self.dt*dW_IF
        self.W_IE += self.dt*dW_IE
        self.W_II += self.dt*dW_II

        return None


    def normalize_weights(self):

        sum_E_exc = np.sum(self.W_EE, axis=1, keepdims=True) + np.sum(self.W_EF, axis=1, keepdims=True) + 1e-10
        sum_I_exc = np.sum(self.W_IE, axis=1, keepdims=True) + np.sum(self.W_IF, axis=1, keepdims=True) + 1e-10
        sum_EI = np.sum(self.W_EI, axis=1, keepdims=True) + 1e-10
        sum_II = np.sum(self.W_II, axis=1, keepdims=True) + 1e-10       

        self.W_EF = self.W_EF_norm * (self.W_EF / sum_E_exc)
        self.W_EE = self.W_EE_norm * (self.W_EE / sum_E_exc)

        self.W_IE = self.W_IE_norm * (self.W_IE / sum_I_exc)
        self.W_IF = self.W_IF_norm * (self.W_IF / sum_I_exc)

        self.W_EI = self.W_EI_norm * (self.W_EI / sum_EI)
        self.W_II = self.W_II_norm * (self.W_II / sum_II)

        return None

    def run(self):

        for t in tqdm(range(self.num_timesteps)):

            # present new input every -- timesteps
            if t % (self.input_presentation_time // self.dt) == 0:
                input_index = t // (self.input_presentation_time // self.dt)
                angle = self.angles[input_index]
                # set r_F based on angle
                self.r_F = self.input(angle)

            # simulate next step
            self.step()
            self.weight_update()
            if self.normalization:
                self.normalize_weights()

            if self.plot_weights_each_step:
                self.plot_weights(timestep=-1)

            # record every -- timesteps
            if t % self.record_freq == 0:
                self.record_state()


        return None
    

    def record_state(self):

        self.u_e_record.append(self.u_e.copy())
        self.u_i_record.append(self.u_i.copy())
        self.r_E_record.append(self.r_E.copy())
        self.r_I_record.append(self.r_I.copy())
        self.r_F_record.append(self.r_F.copy())

        self.W_EE_record.append(self.W_EE.copy())
        self.W_EI_record.append(self.W_EI.copy())
        self.W_IE_record.append(self.W_IE.copy())
        self.W_II_record.append(self.W_II.copy())
        self.W_EF_record.append(self.W_EF.copy())
        self.W_IF_record.append(self.W_IF.copy())
        self.r_F_record.append(self.r_F.copy())

        return None

    def estimate_tuning_curves(self):
        return None


    def plot_weights(self, timestep=0):
        
        fig, axs = plt.subplots(2, 3, figsize=(10, 8))

        im0 = axs[0, 0].imshow(self.W_EF_record[timestep], aspect='auto', cmap='Greys')
        axs[0, 0].set_title('Feedforward to Excitatory Weights')
        plt.colorbar(im0, ax=axs[0, 0])

        im1 = axs[0, 1].imshow(self.W_EE_record[timestep], aspect='auto', cmap='Greys')
        axs[0, 1].set_title('Excitatory to Excitatory Weights')
        plt.colorbar(im1, ax=axs[0, 1])

        im2 = axs[0, 2].imshow(self.W_EI_record[timestep], aspect='auto', cmap='Greys')
        axs[0, 2].set_title('Excitatory to Inhibitory Weights')
        plt.colorbar(im2, ax=axs[0, 2])

        im3 = axs[1, 0].imshow(self.W_IF_record[timestep], aspect='auto', cmap='Greys')
        axs[1, 0].set_title('Feedforward to Inhibitory Weights')
        plt.colorbar(im3, ax=axs[1, 0])

        im4 = axs[1, 1].imshow(self.W_IE_record[timestep], aspect='auto', cmap='Greys')
        axs[1, 1].set_title('Inhibitory to Excitatory Weights')
        plt.colorbar(im4, ax=axs[1, 1])

        im5 = axs[1, 2].imshow(self.W_II_record[timestep], aspect='auto', cmap='Greys')
        axs[1, 2].set_title('Inhibitory to Inhibitory Weights')
        plt.colorbar(im5, ax=axs[1, 2])

        fig.tight_layout()

        return None
    
    def save_data(self):

        # Save weights and firing rates using h5py

        return None
    
    def plot_activity(self, timestep=10):

        fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=False)

        # Activity at last timestep
        r_E_array = np.array(self.r_E_record[timestep])
        r_I_array = np.array(self.r_I_record[timestep])
        r_F_array = np.array(self.r_F_record[timestep])

        axs[0].plot(r_F_array, label='Feedforward Neurons', linestyle='--', color='gray')

        axs[1].plot(np.arange(self.N_E), r_E_array)
        axs[1].set_title('Excitatory')
        axs[1].set_ylabel('Firing rate (r_e)')
        axs[1].legend()

        axs[2].plot(np.arange(self.N_I), r_I_array)
        axs[2].set_title('Inhibitory')
        axs[2].set_xlabel('Neuron Index')
        axs[2].set_ylabel('Firing rate (r_i)')
        axs[2].legend()

        plt.tight_layout()
        plt.show()
        return None
    
    def estimate_tuning_curves(self):

        all_inputs = np.linspace(0, np.pi, 50)
        tuning_curves_E = np.zeros((self.N_E, len(all_inputs)))
        tuning_curves_I = np.zeros((self.N_I, len(all_inputs)))
        # Estimate tuning curves
        for input_idx, input in enumerate(all_inputs):
            # simulate for a few seconds to reach steady state
            r_F = self.input(input)

            u_E = np.zeros(self.N_E)
            u_I = np.zeros(self.N_I)
            r_E = np.zeros(self.N_E)
            r_I = np.zeros(self.N_I)

            for _ in range(100):

                du_E = (1/self.tau_e) * (-u_E + self.W_EF@r_F + self.W_EE@r_E - self.W_EI@r_I)
                du_I = (1/self.tau_i) * (-u_I + self.W_IF@r_F + self.W_IE@r_E - self.W_II@r_I)
                u_E += du_E * self.dt
                u_I += du_I * self.dt

                r_E = self.transfer_F(u_E)
                r_I = self.transfer_F(u_I)

            tuning_curves_E[:, input_idx] = r_E
            tuning_curves_I[:, input_idx] = r_I

        self.tuning_curves_E = tuning_curves_E
        self.tuning_curves_I = tuning_curves_I

        return None
    

    # create an animation for changes in weights over time
    def animate_weights(self):

        

        return None
