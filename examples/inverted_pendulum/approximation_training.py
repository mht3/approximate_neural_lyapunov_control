import sys, os
# add module to system path
cur_path = os.path.dirname(os.path.realpath(__file__))
module_path = os.path.join(cur_path, '..', '..')
sys.path.insert(0, module_path)
import torch
from train_model import Trainer, load_model, load_state
from lyapunov_policy_optimization.loss import LyapunovRisk
from matplotlib import pyplot as plt
from lyapunov_policy_optimization.falsifier import Falsifier

def plot_losses(true_loss, approx_dynamics_loss, approx_lie_loss):
    fig = plt.figure(figsize=(8, 6))
    plt.plot(range(len(true_loss)), true_loss, label='True Loss')
    plt.plot(range(len(approx_dynamics_loss)), approx_dynamics_loss, label='Approximate Dynamics Loss')
    plt.plot(range(len(approx_lie_loss)), approx_lie_loss, label='Approximate Lie Derivative Loss')

    plt.ylabel('Lyapunov Risk', size=16)
    plt.xlabel('Epochs', size=16)
    plt.grid()
    plt.legend()
    plt.savefig('examples/inverted_pendulum/results/loss_comparison.png')


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
    state_min = [-6., -6.]
    state_max = [6., 6.]
    # load 500 length 2 vectors of the state at random
    X = load_state(state_min, state_max, N=500)
    # stable conditions (used for V(x_0) = 0)
    theta_eq, theta_dot_eq = 0., 0.
    X_0 = torch.Tensor([theta_eq, theta_dot_eq])
    ### Load falsifier
    falsifier = Falsifier(state_min, state_max, epsilon=0., scale=0.02, frequency=150, num_samples=10)
    ### Start training process ##
    loss_fn = LyapunovRisk(lyapunov_factor=1., lie_factor=1., equilibrium_factor=1.)
    lr = 0.01
    print("Training with true loss...")
    ### load model and training pipeline with initialized LQR weights ###
    model_1 = load_model()
    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=lr)
    trainer_1 = Trainer(model_1, lr, optimizer_1, loss_fn, falsifier=falsifier, loss_mode='true')
    true_loss = trainer_1.train(X, X_0, epochs=1000, verbose=True)

    print("Training with approx dynamics loss...")
    ### Load falsifier with different frequency to give approximate loss more time to converge
    falsifier = Falsifier(state_min, state_max, epsilon=0., scale=0.02, frequency=400, num_samples=1)
    model_2 = load_model()
    optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=lr)
    trainer_2 = Trainer(model_2, lr, optimizer_2, loss_fn, falsifier=falsifier, loss_mode='approx_dynamics')
    # calculate lie derivative when system dynamics are unknown (this model compares the approximate f to the ground truth)
    approx_dynamics_loss = trainer_2.train(X, X_0, epochs=1000, verbose=True)
    print("Training with approx lie derivative loss...")
    model_3 = load_model()
    optimizer_3 = torch.optim.Adam(model_3.parameters(), lr=lr)
    trainer_3 = Trainer(model_3, lr, optimizer_3, loss_fn, falsifier=falsifier, loss_mode='approx_lie')
    # calculate lie derivative when system dynamics are unknown (this model compares the approximate f to the ground truth)
    approx_lie_loss = trainer_3.train(X, X_0, epochs=1000, verbose=True)
    plot_losses(true_loss, approx_dynamics_loss, approx_lie_loss)