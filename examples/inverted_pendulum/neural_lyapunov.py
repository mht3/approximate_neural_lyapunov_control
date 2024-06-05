'''
See gymnasium cartpole environment: https://gymnasium.farama.org/environments/classic_control/cart_pole/
'''
# NOTE run with: python examples/cartpole/cartpole_neural_lyapunov.py

import sys, os
# add module to system path
cur_path = os.path.dirname(os.path.realpath(__file__))
module_path = os.path.join(cur_path, '..', '..')
sys.path.insert(0, module_path)
from matplotlib import pyplot as plt

import gymnasium as gym
import numpy as np
import torch
from lqr import LQR, plot_multiple_trajectories

from lyapunov_policy_optimization.models.neural_lyapunov_model import NeuralLyapunovController

class NeuralLyaponovControl():
    '''
    Neural Lyaponov Control For CartPole Environment
    '''
    def __init__(self, env, controller):
        # Gymnasium environment
        self.env = env
        # input to controller x has 2 dimensions: theta, thetadot
        self.controller = controller

    def get_input(self, observation):
        # x is by default a vector of length 3 (x, y, theta_dot)
        # convert x and y to theta with inverse tangent
        x_val, y_val, theta_dot = observation
        # subtract 90 degrees because 0 degrees is at top
        theta = np.arctan2(x_val, -y_val) - (np.pi / 2)
        print(theta, self.env.unwrapped.state[0])
        x_tensor = torch.Tensor([theta, theta_dot])
        # pass through trained model
        V, u_tensor = self.controller(x_tensor)
        u_numpy = u_tensor.cpu().detach().numpy()
        return u_numpy


    def control(self, noisy_observer=False):
        initial_state = np.random.uniform([-np.pi/2, -0.5], [np.pi/2, 0.5], size=2)
        observation = self.reset_env(initial_state)
        self.env.unwrapped.state = initial_state
        for _ in range(1000):
            # get action and input (force) required 
            action = self.get_input(observation)
            observation, reward, terminated, truncated, info = self.env.step(action)

            # add small noise to observer (position velocity, pole angle, pole angular velocity)
            if noisy_observer:
                observation += np.random.normal(0, 0.15, 3)
            if terminated or truncated:
                initial_state = np.random.uniform([-np.pi/2, -0.5], [np.pi/2, 0.5], size=2)
                observation = self.reset_env(initial_state)

        self.env.close()
    def normalize_angle(self, angle):
        '''
        Normalize the angle to constrain from -pi to pi
        '''
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def simulate(self, initial_state, time):
        observation = self.reset_env(initial_state)
        theta, theta_dot = initial_state
        trajectory = {'label': "Angle={:0.3f}, Angular Velocity={:0.3f}".format(theta, theta_dot), 'theta': [], 'theta_dot': []}
        for t in range(len(time)):
            action = self.get_input(observation)
            observation, reward, terminated, truncated, info = self.env.step(action)
            theta, theta_dot = self.env.unwrapped.state
            trajectory['theta'].append(self.normalize_angle(theta))
            trajectory['theta_dot'].append(theta_dot)

        return trajectory
    
    def reset_env(self, initial_state):
        observation, info = self.env.reset()
        theta, theta_dot = initial_state
        observation = np.array([np.cos(theta), np.sin(theta), theta_dot])
        self.env.unwrapped.state = initial_state
        return observation

def load_model(pt_path):
    # Load neural lyapunov controller with trained weights
    lqr = LQR()
    K = lqr.K
    # initial weight for controller
    lqr_val = -torch.Tensor(K)
    d_in, n_hidden, d_out = 2, 6, 1
    controller = NeuralLyapunovController(d_in, n_hidden, d_out, lqr_val)
    state_dict = torch.load(pt_path)
    controller.load_state_dict(state_dict)
    return controller


if __name__ == '__main__':
    show_gui = True
    noisy_observer = False
    if show_gui:
        env = gym.make('Pendulum-v1', render_mode="human", g=9.81)
    else: 
        env = gym.make('Pendulum-v1', g=9.81)
    controller = load_model('examples/inverted_pendulum/models/pendulum_lyapunov_model_1.pt')

    nlc = NeuralLyaponovControl(env, controller)
    if show_gui:
        nlc.control(noisy_observer=noisy_observer)

    # Example data for multiple trajectories
    time = np.arange(0, 5.05, 0.05)
    trajectories = []
    initial_states = np.array([[np.pi/8, 0], [np.pi/4, 0], [-np.pi/8, 0]])
    for initial_state in initial_states:
        trajectory= nlc.simulate(initial_state, time)
        trajectories.append(trajectory)
    # Call the function with the example data
    plot_multiple_trajectories(time, trajectories,'examples/inverted_pendulum/results/time_convergence_lpo.png')
