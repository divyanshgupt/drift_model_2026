import numpy as np
from tqdm import tqdm

def circular_gaussian(N, theta, amp=2, sigma=20, baseline=0):
    """
    Generate pre-synaptic activity based on theta stimulus
    """
    theta_y = np.linspace(0, 180, N)  # center of tuning curves
    d = np.abs(theta - theta_y)    # distance to input theta
    d_plus = d + 180
    d_minus = d - 180
    y = amp * (np.exp(-(d**2)/(2*sigma**2)) + np.exp(-(d_plus**2)/(2*sigma**2)) + np.exp(-(d_minus**2)/(2*sigma**2))) + baseline
    return y

def estimate_tuning_curves(W, w_if, w_ei, N=500, num_stimuli=500):

    num_stimuli = num_stimuli
    theta_list = np.linspace(0, 180, num_stimuli)
    tuning_curves =  np.empty((N, num_stimuli))

    for stim_num, theta in tqdm(enumerate(theta_list)):
        u = circular_gaussian(N, theta)
        inh_activity = w_if.T.dot(u)
        exc_activity = W.T.dot(u) - w_ei.T.dot(inh_activity)
        exc_activity[exc_activity < 0] = 0
        tuning_curves[:, stim_num] = exc_activity

    return tuning_curves

def estimate_tuning_curves_over_time(W, w_if, w_ei, n_steps):

    tuning_curves_over_time = np.empty((N, num_stimuli, n_steps))
    
    for t in tqdm(range(n_steps)):
        tuning_curves_over_time[:, :, t] = estimate(estimate_tuning_curves(W, w_if, w_ei))
    
    return tuning_curves_over_time


def params_to_string(params):
    '''incomplete'''

    string = f"inh_scale: {inh_scale}"