import sys, os
# add module to system path
cur_path = os.path.dirname(os.path.realpath(__file__))
module_path = os.path.join(cur_path, '..', '..')
sys.path.insert(0, module_path)
from lyapunov_policy_optimization.utils import plot_2D_roa, Plot3D
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

def get_nn_solution(pt_file, x):
    controller = load_model(pt_file)
    V_est, u = controller(x)
    # convert to 100x100 numpy array
    V_est = V_est.detach().numpy().reshape(100, 100) - 0.02
    return V_est

if __name__ == '__main__':
    r = np.pi/4
    X = np.linspace(-r, r, 100) 
    Y = np.linspace(-r, r, 100)
    #### PLOTTING LQR SOLUTION
    # get equations for inverted pendulum
    lqr = LQR()
    # algebraic ricatti equation gives p and V = x^TPx is lyapunov function
    P = lqr.get_are()
 
    x1, x2 = np.meshgrid(X,Y)
    x = np.vstack([x1.flatten(), x2.flatten()])

    # Equivalent to X^TPX: P[0, 0]*x1**2 + 2*P[0, 1]*x1*x2 + P[1, 1]*x2**2
    V_lqr = np.einsum('ij,ji->i', x.T, np.dot(P, x)).reshape(100, 100) - 0.08
    # Plot results for original LQR solution
    Plot3D(X, Y, V_lqr, r, filename='examples/inverted_pendulum/results/v_lqr_3d.png')

    #### PLOTTING TRAINED NN SOlUTIONS
    x_torch = torch.Tensor(x.transpose(1, 0))
    # True loss
    V_est = get_nn_solution('examples/inverted_pendulum/models/pendulum_lyapunov_model_true.pt', x_torch) - 0.25
    Plot3D(X, Y, V_est, r, filename='examples/inverted_pendulum/results/v_lpo_3d.png')

    # Dynamics Approximation
    V_est_dynamics = get_nn_solution('examples/inverted_pendulum/models/pendulum_lyapunov_model_appx_dynamics.pt', x_torch) - 0.18
    Plot3D(X, Y, V_est_dynamics, r, filename='examples/inverted_pendulum/results/v_lpo_dynamics_3d.png')

    
    # lie approximation
    V_est_lie = get_nn_solution('examples/inverted_pendulum/models/pendulum_lyapunov_model_appx_lie.pt', x_torch) - 0.3
    Plot3D(X, Y, V_est_lie, r, filename='examples/inverted_pendulum/results/v_lpo_lie_3d.png')


    #### Plot both together in 2D
    plot_2D_roa(X, Y, V_lqr, V_est, lqr.f, r, filename='examples/inverted_pendulum/results/2d_roa.png',
                V_nn_est2=V_est_dynamics, V_nn_est3=V_est_lie)