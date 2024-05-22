# NOTE Must run python examples/cartpole/train_model.py

import sys, os
# add module to system path
cur_path = os.path.dirname(os.path.realpath(__file__))
module_path = os.path.join(cur_path, '..', '..')
sys.path.insert(0, module_path)

import torch
from lqr import LQR
from lyapunov_policy_optimization.models.neural_lyapunov_model import NeuralLyapunovController
from lyapunov_policy_optimization.loss import LyapunovRisk
import gymnasium as gym 
from matplotlib import pyplot as plt

class Trainer():
    def __init__(self, model, lr, optimizer, loss_fn, loss_mode='true'):
        self.model = model
        self.lr = lr
        self.optimizer = optimizer
        self.lyapunov_loss = loss_fn
        self.loss_mode = loss_mode
        if self.loss_mode == 'approx_dynamics' or self.loss_mode == 'approx_lie':
            # use env to get s, a, s' pairs and use finite difference approximation
            self.env = gym.make('CartPole-v1')
    
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
    
    def get_approx_lie_derivative(self, V_candidate, V_candidate_next, dt=0.02):
        '''
        Calculates L_V = ∑∂V/∂xᵢ*fᵢ by forward finite difference
                    L_V = (V' - V) / dt
        '''
        return (V_candidate_next - V_candidate) / dt

    def train(self, X, x_0, epochs=2000, verbose=False, every_n_epochs=10, approx_loss=False):
        self.model.train()
        valid = False
        loss_list = []

        for epoch in range(1, epochs+1):
            if valid == True:
                if verbose:
                    print('Found valid solution.')
                break
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
                loss = self.lyapunov_loss(V_candidate, L_V, V_X0)
            
            elif self.loss_mode == 'approx_dynamics':
                # compute approximate f_dot and compare to true f
                X_prime = step(X, u, self.env)
                f_approx = approx_f_value(X, X_prime, dt=0.02)
                # check dx/dt estimates are close
                # epsilon for x_dot. cart velocity and angular velocity are easier to approximate than accelerations.
                # TODO is there a better way to approximate without running throught the simulator multiple times?
                epsilon = torch.tensor([1e-4, 10., 1e-4, 10.])
                # assert(torch.all(abs(f - f_approx) < epsilon))
                # could replace loss function 
                L_V_approx = self.get_lie_derivative(X, V_candidate, f_approx)
                loss = self.lyapunov_loss(V_candidate, L_V_approx, V_X0)
            elif self.loss_mode == 'approx_lie':
                # compute approximate f_dot and compare to true f
                X_prime = step(X, u, self.env)
                V_candidate_prime, u = self.model(X_prime)
                L_V_approx = self.get_approx_lie_derivative(V_candidate, V_candidate_prime, dt=0.02)
                loss = self.lyapunov_loss(V_candidate, L_V_approx, V_X0)

            loss_list.append(loss.item())
            loss.backward()
            self.optimizer.step() 
            if verbose and (epoch % every_n_epochs == 0):
                print('Epoch:\t{}\tLyapunov Risk: {:.4f}'.format(epoch, loss.item()))

            # TODO Add in falsifier here
            # add counterexamples

        return loss_list
    
def load_model():
    lqr = LQR()
    K = lqr.K
    lqr_val = -torch.Tensor(K)
    d_in, n_hidden, d_out = 4, 6, 1
    controller = NeuralLyapunovController(d_in, n_hidden, d_out, lqr_val)
    return controller

def step(X, u, env):
    '''
    Generates all X_primes needed given current state and current action
    X: current position, velocity, pole angle, and pole angular velocity
    u: input for cartpole 
    '''
    # take step in environment based upon current state and action
    N = X.shape[0]
    u = torch.clip(u, -10, 10)

    X_prime = torch.empty_like(X)
    observation, info = env.reset()
    for i in range(N):
        x_i = X[i, :].detach().numpy()
        # set environment as x_i
        observation, info = env.reset()
        env.unwrapped.state = x_i

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
        env.unwrapped.force_mag = abs(u_i)
        # take step in environment
        observation, reward, terminated, truncated, info = env.step(action)
        # add sample to X_prime
        X_prime[i, :] = torch.tensor(observation)

    return X_prime

def approx_f_value(X, X_prime, dt=0.02):
    # Approximate f value with S, a, S'
    y = (X_prime - X) / dt
    return y

def f_value(X, u):
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

def plot_losses(true_loss, approx_dynamics_loss, approx_lie_loss):
    fig = plt.figure(figsize=(8, 6))
    x = range(len(true_loss))
    plt.plot(x, true_loss, label='True Loss')
    plt.plot(x, approx_dynamics_loss, label='Approximate Dynamics Loss')
    plt.plot(x, approx_lie_loss, label='Approximate Lie Derivative Loss')

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
    print("Training with true loss...")
    ### load model and training pipeline with initialized LQR weights ###
    model_1 = load_model()
    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=lr)
    trainer_1 = Trainer(model_1, lr, optimizer_1, loss_fn, loss_mode='true')
    true_loss = trainer_1.train(X, X_0, epochs=200, verbose=True)
    # save model corresponding to true loss
    torch.save(model_1.state_dict(), 'examples/cartpole/models/cartpole_lyapunov_model_1.pt')
    print("Training with approx dynamics loss...")
    model_2 = load_model()
    optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=lr)
    trainer_2 = Trainer(model_2, lr, optimizer_2, loss_fn, loss_mode='approx_dynamics')
    # calculate lie derivative when system dynamics are unknown (this model compares the approximate f to the ground truth)
    approx_dynamics_loss = trainer_2.train(X, X_0, epochs=200, verbose=True)
    print("Training with approx lie derivative loss...")
    model_3 = load_model()
    optimizer_3 = torch.optim.Adam(model_3.parameters(), lr=lr)
    trainer_3 = Trainer(model_3, lr, optimizer_3, loss_fn, loss_mode='approx_lie')
    # calculate lie derivative when system dynamics are unknown (this model compares the approximate f to the ground truth)
    approx_lie_loss = trainer_3.train(X, X_0, epochs=200, verbose=True)
    plot_losses(true_loss, approx_dynamics_loss, approx_lie_loss)