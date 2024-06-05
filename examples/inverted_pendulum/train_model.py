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

import gymnasium as gym 

class Trainer():
    def __init__(self, model, lr, optimizer, loss_fn, circle_tuning_loss_fn=None, falsifier=None, loss_mode='true'):
        self.model = model
        self.lr = lr
        self.optimizer = optimizer
        self.lyapunov_loss = loss_fn
        self.circle_tuning_loss = circle_tuning_loss_fn
        self.loss_mode = loss_mode
        # use env to get s, a, s' pairs and use finite difference approximation
        self.env = gym.make('Pendulum-v1', g=9.81)
        # initialize falsifier class with epsilon (constraint on lie derivative falsification)
        self.falsifier = falsifier
    
    def get_lie_derivative(self, X, V_candidate, f):
        '''
        Calculates L_V = ∑∂V/∂xᵢ*fᵢ
        '''
        w1 = self.model.layer1.weight
        b1 = self.model.layer1.bias
        w2 = self.model.layer2.weight
        b2 = self.model.layer2.bias
        # running through model again 
        z1 = X @ w1.t() + b1
        a1 = torch.tanh(z1)
        z2 = a1 @ w2.t() + b2
        d_z2 = 1. - V_candidate**2
        partial_z2_a1 = w2
        partial_a1_z1 = 1 - torch.tanh(z1)**2
        partial_z1_x = w1

        d_a1 = (d_z2 @ partial_z2_a1)
        d_z1 = d_a1 * partial_a1_z1

        # gets final ∂V/∂x
        d_x = d_z1 @ partial_z1_x

        lie_derivative = torch.diagonal((d_x @ f.t()), 0)
        return lie_derivative
    
    def get_approx_lie_derivative(self, V_candidate, V_candidate_next, dt):
        '''
        Calculates L_V = ∑∂V/∂xᵢ*fᵢ by forward finite difference
                    L_V = (V' - V) / dt
        '''
        return (V_candidate_next - V_candidate) / dt

    def adjust_learning_rate(self, decay_rate=.9):
        # new_lr = lr * decay_rate
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] * decay_rate
    
    def reset_learning_rate(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def train(self, X, x_0, epochs=2000, verbose=False, every_n_epochs=10, step_size=100, decay_rate=1.):
        self.model.train()
        loss_list = []
        original_size = len(X)
        for epoch in range(1, epochs+1):
            # lr scheduler
            if (epoch + 1) % step_size == 0:
                self.adjust_learning_rate(decay_rate)
            # zero gradients
            self.optimizer.zero_grad()

            # get lyapunov function and input from model
            V_candidate, u = self.model(X)
            # get lyapunov function evaluated at equilibrium point
            V_X0, u_X0 = self.model(x_0)

            # get loss
            if self.loss_mode == 'true':
                # Compute lie derivative of V : L_V = ∑∂V/∂xᵢ*fᵢ
                f = f_value(X, u)
                L_V = self.get_lie_derivative(X, V_candidate, f)
            
            elif self.loss_mode == 'approx_dynamics':
                # compute approximate f_dot and compare to true f
                X_prime = step(X, u, self.env)
                f = f_value(X, u)
                f_approx = approx_f_value(X, X_prime, dt=0.05)
                # print(f[0:5])
                # print(f_approx[0:5])
                # print('--')
                L_V = self.get_lie_derivative(X, V_candidate, f_approx)

            elif self.loss_mode == 'approx_lie':
                # compute approximate f_dot and compare to true f
                X_prime = step(X, u, self.env)
                V_candidate_prime, u = self.model(X_prime)
                L_V = self.get_approx_lie_derivative(V_candidate, V_candidate_prime, dt=0.05)
            
            loss = self.lyapunov_loss(V_candidate, L_V, V_X0)
            if self.circle_tuning_loss is not None:
                loss += self.circle_tuning_loss(X, V_candidate)
        
            loss_list.append(loss.item())
            loss.backward()
            self.optimizer.step() 
            if verbose and (epoch % every_n_epochs == 0):
                print('Epoch:\t{}\tLyapunov Risk: {:.4f}'.format(epoch, loss.item()))

            # run falsifier every falsifier_frequency epochs
            if (self.falsifier is not None) and epoch % (self.falsifier.get_frequency())== 0:
                counterexamples = self.falsifier.check_lyapunov(X, V_candidate, L_V)
                if (not (counterexamples is None)): 
                    print("Not a Lyapunov function. Found {} counterexamples.".format(counterexamples.shape[0]))
                    # add new counterexamples sampled from old ones
                    if self.falsifier.counterexamples_added + len(counterexamples) > original_size:
                        print("Too many previous counterexamples. Pruning...")
                        # keep 1/5 of random elements of counterexamples
                        num_keep = original_size // 5
                        counterexample_keep_idx = torch.randperm(len(counterexamples))[:num_keep]
                        cur_counterexamples = X[counterexample_keep_idx]
                        X = X[:original_size]
                        X = torch.cat([X, cur_counterexamples], dim=0)
                        # reset counter
                        self.falsifier.counterexamples_added = num_keep

                    X = self.falsifier.add_counterexamples(X, counterexamples)

                else:  
                    if verbose:
                        print('No counterexamples found!')
                    
        return loss_list
    

    
def plot_loss(true_loss, filename):
    fig = plt.figure(figsize=(8, 6))
    plt.plot(range(len(true_loss)), true_loss, label='True Loss')

    plt.ylabel('Lyapunov Risk', size=16)
    plt.xlabel('Epochs', size=16)
    plt.grid()
    plt.legend()
    plt.savefig(filename)



def load_model():
    lqr = LQR()
    K = lqr.K
    lqr_val = -torch.Tensor(K)
    d_in, n_hidden, d_out = 2, 6, 1
    controller = NeuralLyapunovController(d_in, n_hidden, d_out, lqr_val)
    return controller

@torch.no_grad()
def step(X, u, env):
    '''
    Generates all X_primes needed given current state and current action
    X: current angle and angular velocity
    u: input torque for inverted pendulum
    '''
    # take step in environment based upon current state and action
    N = X.shape[0]
    u_numpy = u.cpu().detach().numpy()
    X_prime = torch.empty_like(X)
    for i in range(N):
        x_i = X[i, :].detach().numpy()
        # set environment as x_i
        observation, info = env.reset()
        # get current action to take 
        u_i = u_numpy[i, :]
        env.unwrapped.state = x_i
        # take step in environment
        observation, reward, terminated, truncated, info = env.step(u_i)
        # add sample to X_prime
        X_prime[i, :] = torch.from_numpy(env.unwrapped.state)

    X_prime[:, 0] = normalize_angle(X_prime[:, 0])

    return X_prime

def approx_f_value(X, X_prime, dt):
    # Approximate f value with S, a, S'
    y = (X_prime - X) / dt
    return y

def normalize_angle(angle):
    '''
    Normalize the angle to constrain from -pi to pi
    '''
    return (angle + np.pi) % (2 * np.pi) - np.pi

def f_value(X, u):
    y = []
    N = X.shape[0]
    # Get linearized system dynamics for cartpole 
    lqr = LQR()
    A, B, Q, R, K = lqr.get_system()
    for i in range(0, N): 
        x_i = X[i, :].detach().numpy()

        u_i = u[i].detach().numpy()

        # (linearized) xdot = Ax + Bu
        f = A@x_i + B@u_i
        
        y.append(f.tolist()) 

    y = torch.tensor(y)
    return y

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
    torch.random.manual_seed(42)

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
    # make samples closer to equilibrium
    state_min = [-np.pi, -np.pi]
    state_max = [np.pi, np.pi]
    # load 500 length 2 vectors of the state at random
    X = load_state(state_min, state_max, N=500)
    # stable conditions (used for V(x_0) = 0)
    theta_eq, theta_dot_eq = 0., 0.
    X_0 = torch.Tensor([theta_eq, theta_dot_eq])

    ### Start training process ##
    loss_fn = LyapunovRisk(lyapunov_factor=1., lie_factor=1.5, equilibrium_factor=1.)
    circle_tuning_loss_fn = CircleTuningLoss(state_max=np.mean(state_max), tuning_factor=0.1)
    lr = 0.01
    ### Load falsifier
    falsifier = Falsifier(state_min, state_max, epsilon=0., scale=0.02, frequency=50, num_samples=5)
    ### load model and training pipeline with initialized LQR weights ###
    model_1 = load_model()
    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=lr)
    trainer_1 = Trainer(model_1, lr, optimizer_1, loss_fn, circle_tuning_loss_fn=circle_tuning_loss_fn,
                        falsifier=falsifier, loss_mode='true')
    true_loss = trainer_1.train(X, X_0, epochs=1000, verbose=True, step_size=50, decay_rate=1.0)
    # save model corresponding to true loss
    torch.save(model_1.state_dict(), 'examples/inverted_pendulum/models/pendulum_lyapunov_model_1.pt')

    plot_loss(true_loss, 'examples/inverted_pendulum/results/true_loss.png')