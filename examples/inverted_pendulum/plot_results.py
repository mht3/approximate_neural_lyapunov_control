import sys, os
# add module to system path
cur_path = os.path.dirname(os.path.realpath(__file__))
module_path = os.path.join(cur_path, '..', '..')
sys.path.insert(0, module_path)
from lyapunov_policy_optimization.utils import plot_2D_roa, Plot3D, plot_2D_roa_lie_overlay
from lqr import LQR

from matplotlib import pyplot as plt
from neural_lyapunov import load_model
import numpy as np
import torch

def plot_trajectory(time, theta_lqr, theta_dot_lqr, theta_lpo, theta_dot_lpo,
                    filename='trajectory.png'):
    fig = plt.figure()

    plt.plot(time, theta_lqr, label=r'$\theta_{LQR}$')
    plt.plot(time, theta_dot_lqr, label=r'$\dot{\theta}_{LQR}$')
    plt.plot(time, theta_lpo, label=r'$\theta_{LPO}$')
    plt.plot(time, theta_dot_lpo, label=r'$\dot{\theta}_{LPO}$')
    plt.legend()
    plt.xlabel('time', size=14)
    
    plt.savefig(filename)

def get_nn_solution(model, x):
    V_est, u = model(x)
    # convert to 100x100 numpy array
    V_est = V_est.detach().numpy().reshape(100, 100)
    return V_est

def get_lie_derivative(model, X, V_candidate, f):
    '''
    Calculates L_V = ∑∂V/∂xᵢ*fᵢ
    '''
    w1 = model.layer1.weight
    b1 = model.layer1.bias
    w2 = model.layer2.weight
    b2 = model.layer2.bias
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

def f_value(X, u) :
    lqr = LQR()
    m, g, l, b = lqr.m, lqr.g, lqr.l, lqr.b
    # theta and theta_dot
    y = []
    N = X.shape[0]
    for i in range(0, N): 
        x_i = X[i, :].detach().numpy()
        u_i = u[i].detach().numpy()[0]
        x1, x2 = x_i
        # get torque using LQR solution
        # system dynamics as list
        x1_dot = x2
        x2_dot = (m*g*l*np.sin(x1) + u_i - b*x2) / (m * l**2)
        x_dot = [x1_dot, x2_dot]
        y.append(x_dot)
    y = torch.tensor(y, dtype=X.dtype)
    return y

def find_lie_counterexamples(model, X, epsilon=0.04):
    V_candidate, u = model(X) 
    # Compute lie derivative of V : L_V = ∑∂V/∂xᵢ*fᵢ
    f = f_value(X, u)
    L_V = get_lie_derivative(model, X, V_candidate, f)
    # Separate the Lie derivatives
    positives = L_V < 0.
    negatives = L_V > epsilon
    borderline = (L_V <= epsilon).logical_and(L_V >= 0)

    return positives, negatives, borderline


if __name__ == '__main__':
    lqr_shift, nn_true_shift, nn_dynamics_shift, nn_lie_shift  = 0.085, 0.47, 0.11, 0.23
    r = np.pi/4
    X = np.linspace(-r, r, 100) 
    Y = np.linspace(-r, r, 100)
    #### PLOTTING LQR SOLUTION
    # get equations for inverted pendulum
    lqr = LQR()
    # algebraic ricatti equation gives p and V = x^TPx is lyapunov function
    P = lqr.get_are()
 
    x1, x2 = np.meshgrid(X, Y)
    x = np.vstack([x1.flatten(), x2.flatten()])

    # Equivalent to X^TPX: P[0, 0]*x1**2 + 2*P[0, 1]*x1*x2 + P[1, 1]*x2**2
    V_lqr = np.einsum('ij,ji->i', x.T, np.dot(P, x)).reshape(100, 100)- lqr_shift
    print('LQR Controller weights:         \t', -lqr.K)
    # Plot results for original LQR solution
    Plot3D(X, Y, V_lqr, r, filename='examples/inverted_pendulum/results/v_lqr_3d.png')

    #### PLOTTING TRAINED NN SOlUTIONS
    x_torch = torch.Tensor(x.transpose(1, 0))
    # True loss
    model_true = load_model('examples/inverted_pendulum/models/pendulum_lyapunov_model_true.pt')
    V_est = get_nn_solution(model_true, x_torch) - nn_true_shift
    Plot3D(X, Y, V_est, r, filename='examples/inverted_pendulum/results/v_lpo_3d.png')
    # find lie derivative counterexamples
    positives, negatives, borderline = find_lie_counterexamples(model_true, x_torch)
    plot_2D_roa_lie_overlay(X, Y, V_est , lqr.f, r, filename='examples/inverted_pendulum/results/lie_overlay_true.png',
                            positive_examples=positives, negative_examples=negatives, borderline_examples=borderline)
    print('NN True Controller weights:         \t', model_true.control.weight.detach().numpy())
    ##### Dynamics Approximation####
    model_dynamics = load_model('examples/inverted_pendulum/models/pendulum_lyapunov_model_appx_dynamics_7.pt')
    V_est_dynamics = get_nn_solution(model_dynamics, x_torch) - nn_dynamics_shift
    Plot3D(X, Y, V_est_dynamics, r, filename='examples/inverted_pendulum/results/v_lpo_dynamics_3d.png')
    positives, negatives, borderline = find_lie_counterexamples(model_dynamics, x_torch)
    plot_2D_roa_lie_overlay(X, Y, V_est_dynamics, lqr.f, r, filename='examples/inverted_pendulum/results/lie_overlay_dynamics.png',
                            positive_examples=positives, negative_examples=negatives, borderline_examples=borderline)
    print('NN Appx Dynamics Controller weights:\t', model_dynamics.control.weight.detach().numpy())

    # # lie approximation
    model_lie = load_model('examples/inverted_pendulum/models/pendulum_lyapunov_model_appx_lie_7.pt')
    V_est_lie = get_nn_solution(model_lie, x_torch) - nn_lie_shift
    Plot3D(X, Y, V_est_lie, r, filename='examples/inverted_pendulum/results/v_lpo_lie_3d.png')
    positives, negatives, borderline = find_lie_counterexamples(model_lie, x_torch)
    plot_2D_roa_lie_overlay(X, Y, V_est_lie, lqr.f, r, filename='examples/inverted_pendulum/results/lie_overlay_lie.png',
                            positive_examples=positives, negative_examples=negatives, borderline_examples=borderline)
    print('NN Appx Lie Controller weights:     \t', model_lie.control.weight.detach().numpy())

    # #### Plot both together in 2D
    plot_2D_roa(X, Y, V_lqr, V_est, lqr.f, r, filename='examples/inverted_pendulum/results/2d_roa_circle_tune.png',
                V_nn_est2=V_est_dynamics, V_nn_est3=V_est_lie)