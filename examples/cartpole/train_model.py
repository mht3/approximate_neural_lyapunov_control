# NOTE Must run python examples/cartpole/train_model.py

import sys, os
# add module to system path
cur_path = os.path.dirname(os.path.realpath(__file__))
module_path = os.path.join(cur_path, '..', '..')
sys.path.insert(0, module_path)

from matplotlib import pyplot as plt
import torch
import numpy as np
from lqr import LQR
from lyapunov_policy_optimization.models.neural_lyapunov_model import NeuralLyapunovController
from lyapunov_policy_optimization.loss import LyapunovRisk, CircleTuningLoss
from lyapunov_policy_optimization.falsifier import Falsifier
from lyapunov_policy_optimization.trainer import Trainer

import gymnasium as gym 

class CartPoleTrainer():
    def __init__(self, model, lr, optimizer, loss_fn, dt=0.02, circle_tuning_loss_fn=None, falsifier=None, loss_mode='true'):
        super().__init__(model, lr, optimizer, loss_fn, dt, circle_tuning_loss_fn, falsifier, loss_mode)
        if self.loss_mode == 'approx_dynamics' or self.loss_mode == 'approx_lie':
            # use env to get s, a, s' pairs and use finite difference approximation
            self.env = gym.make('CartPole-v1')
            
    def step(self, X, u):
        '''
        Generates all X_primes needed given current state and current action
        X: current position, velocity, pole angle, and pole angular velocity
        u: input for cartpole 
        '''
        # take step in environment based upon current state and action
        N = X.shape[0]
        u = torch.clip(u, -10, 10)

        X_prime = torch.empty_like(X)
        observation, info = self.env.reset()
        for i in range(N):
            x_i = X[i, :].detach().numpy()
            # set environment as x_i
            observation, info = self.env.reset()
            self.env.unwrapped.state = x_i

            # get current action to take 
            u_i = u[i][0].detach().numpy()
            action = 0
            if u_i > 0:
                # move cart right
                action = 1
            else:
                # move cart left
                action = 0

            # set magnitude of force as input
            self.env.unwrapped.force_mag = abs(u_i)
            # take step in environment
            observation, reward, terminated, truncated, info = self.env.step(action)
            # add sample to X_prime
            X_prime[i, :] = torch.tensor(observation)

        return X_prime

    def f_value_linearized(self, X, u):
        y = []
        N = X.shape[0]
        # Get system dynamics for cartpole 
        lqr = LQR()
        A, B, Q, R, K = lqr.get_system()
        u = torch.clip(u, -10, 10)
        for i in range(0, N): 
            x_i = X[i, :].detach().numpy()

            u_i = u[i].detach().numpy()
            # xdot = Ax + Bu
            f = A@x_i + B@u_i
            
            y.append(f.tolist()) 

        y = torch.tensor(y)
        return y
    
def load_model():
    lqr = LQR()
    K = lqr.K
    lqr_val = -torch.Tensor(K)
    # 4 states in CartPole.
    d_in, n_hidden, d_out = 4, 6, 1
    controller = NeuralLyapunovController(d_in, n_hidden, d_out, lqr_val)
    return controller

def plot_losses(approx_dynamics_loss, approx_lie_loss):
    fig = plt.figure(figsize=(8, 6))
    x1 = range(len(approx_dynamics_loss))
    x2 = range(len(approx_lie_loss))

    # plt.plot(x, true_loss, label='True Loss')
    plt.plot(x1, approx_dynamics_loss, label='Approximate Dynamics Loss')
    plt.plot(x2, approx_lie_loss, label='Approximate Lie Derivative Loss')

    plt.ylabel('Lyapunov Risk', size=16)
    plt.xlabel('Epochs', size=16)
    plt.grid()
    plt.legend()
    plt.savefig('examples/cartpole/results/loss_comparison.png')

def load_state(state_min, state_max, N=500):
    # X: Nxlen(state_min) tensor of initial states
    X = torch.empty(N, 0)
    for i in range(len(state_min)):
        s_min = state_min[i]
        s_max = state_max[i]
        x = torch.Tensor(N, 1).uniform_(s_min, s_max)
        X = torch.cat([X, x], dim=1)
    return X

if __name__ == '__main__':
    ### Generate random training data ###
    # number of samples
    N = 500
    '''
    bounds for position, velocity, angle, and angular velocity
    position: -2.4 to 2.4
    velocity: -2 to 2
    theta: -12 to 12 degrees
    thata_dot: -2 to 2
    '''
    state_min = [-2.4, -2., -0.2094, -2]
    state_max = [2.4, 2., 0.2094, 2]

    # load 500 length 4 vectors of the state at random
    X = load_state(state_min, state_max, N=500)
    # stable conditions (used for V(x_0) = 0)
    # note that X_p is a free variable and can be at any position
    x_p_eq, x_v_eq, x_theta_eq, x_theta_dot_eq = 0., 0., 0., 0.
    X_0 = torch.Tensor([x_p_eq, x_v_eq, x_theta_eq, x_theta_dot_eq])

    ### Start training process ##
    loss_fn = LyapunovRisk(lyapunov_factor=1., lie_factor=1., equilibrium_factor=1.)
    lr = 0.01
    print("Training with approx dynamics loss...")
    model_2 = load_model()
    optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=lr)
    trainer_2 = CartPoleTrainer(model_2, lr, optimizer_2, loss_fn, loss_mode='approx_dynamics')
    # calculate lie derivative when system dynamics are unknown (this model compares the approximate f to the ground truth)
    approx_dynamics_loss = trainer_2.train(X, X_0, epochs=200, verbose=True)
    print("Training with approx lie derivative loss...")
    model_3 = load_model()
    optimizer_3 = torch.optim.Adam(model_3.parameters(), lr=lr)
    trainer_3 = CartPoleTrainer(model_3, lr, optimizer_3, loss_fn, loss_mode='approx_lie')
    # calculate lie derivative when system dynamics are unknown (this model compares the approximate f to the ground truth)
    approx_lie_loss = trainer_3.train(X, X_0, epochs=200, verbose=True)
    plot_losses(approx_dynamics_loss, approx_lie_loss)