import sys, os
# add module to system path
cur_path = os.path.dirname(os.path.realpath(__file__))
module_path = os.path.join(cur_path, '..', '..')
sys.path.insert(0, module_path)

import torch
from train_model import InvertedPendulumTrainer, load_model, load_state
from lyapunov_policy_optimization.loss import LyapunovRisk, CircleTuningLoss
from matplotlib import pyplot as plt
from lyapunov_policy_optimization.falsifier import Falsifier
import numpy as np

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
    torch.random.manual_seed(43)

    ### Generate random training data ###
    # number of samples
    N = 500
    # make samples closer to equilibrium
    state_min = [-np.pi/4, -np.pi/4]
    state_max = [np.pi/4, np.pi/4]
    # load 500 length 2 vectors of the state at random
    X = load_state(state_min, state_max, N=500)
    # stable conditions (used for V(x_0) = 0)
    theta_eq, theta_dot_eq = 0., 0.
    X_0 = torch.Tensor([theta_eq, theta_dot_eq])
    ### Load falsifier
    falsifier = Falsifier(state_min, state_max, epsilon=0., scale=0.05, frequency=100, num_samples=5)
    ### Start training process ##
    loss_fn = LyapunovRisk(lyapunov_factor=1., lie_factor=1., equilibrium_factor=1.)
    circle_tuning_loss_fn = CircleTuningLoss(state_max=np.mean(state_max), tuning_factor=0.1)
    lr = 0.01
    ### Load falsifier
    print("Training with true loss...")
    ## load model and training pipeline with initialized LQR weights ###
    model_1 = load_model()
    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=lr)
    trainer_1 = InvertedPendulumTrainer(model_1, lr, optimizer_1, loss_fn, circle_tuning_loss_fn=circle_tuning_loss_fn,
                        falsifier=falsifier, loss_mode='true')
    true_loss = trainer_1.train(X, X_0, epochs=1250, verbose=True)
    # torch.save(model_1.state_dict(), 'examples/inverted_pendulum/models/pendulum_lyapunov_model_true.pt')

    print("Training with approx dynamics loss...")
    alpha = 0.7
    lr = 0.01
    ### Load falsifier with different frequency to give approximate loss more time to converge
    loss_fn = LyapunovRisk(lyapunov_factor=1., lie_factor=1.5, equilibrium_factor=1.)

    falsifier = Falsifier(state_min, state_max, epsilon=0., scale=0.05, frequency=150, num_samples=5)
    circle_tuning_loss_fn = CircleTuningLoss(state_max=np.max(state_max), tuning_factor=0.1)
    # falsifier = None
    model_2 = load_model()
    optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=lr)
    trainer_2 = InvertedPendulumTrainer(model_2, lr, optimizer_2, loss_fn, circle_tuning_loss_fn=circle_tuning_loss_fn,
                        falsifier=falsifier, loss_mode='approx_dynamics')
    # calculate lie derivative when system dynamics are unknown (this model compares the approximate f to the ground truth)
    approx_dynamics_loss = trainer_2.train(X, X_0, alpha=alpha, epochs=1200, verbose=True)
    torch.save(model_2.state_dict(), 'examples/inverted_pendulum/models/pendulum_lyapunov_model_appx_dynamics_{}.pt'.format(int(alpha*10)))

    print("Training with approx lie derivative loss...")
    model_3 = load_model()
    optimizer_3 = torch.optim.Adam(model_3.parameters(), lr=lr)
    trainer_3 = InvertedPendulumTrainer(model_3, lr, optimizer_3, loss_fn, circle_tuning_loss_fn=circle_tuning_loss_fn,
                        falsifier=falsifier, loss_mode='approx_lie')
    # calculate lie derivative when system dynamics are unknown (this model compares the approximate f to the ground truth)
    approx_lie_loss = trainer_3.train(X, X_0, alpha=alpha, epochs=1200, verbose=True)
    torch.save(model_3.state_dict(), 'examples/inverted_pendulum/models/pendulum_lyapunov_model_appx_lie_{}.pt'.format(int(alpha*10)))

    plot_losses(true_loss, approx_dynamics_loss, approx_lie_loss)