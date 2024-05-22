'''
See gymnasium cartpole environment: https://gymnasium.farama.org/environments/classic_control/cart_pole/
'''

import gymnasium as gym
import numpy as np
from scipy import linalg

import sys, os
# add module to system path
cur_path = os.path.dirname(os.path.realpath(__file__))
module_path = os.path.join(cur_path, '..', '..')
sys.path.insert(0, module_path)
from lyapunov_policy_optimization.utils import plot_2D_roa

class LQR():
    '''
    LQR For Inverted Pendulum Environment
    '''
    def __init__(self, env=None):
        # Gymnasium environment
        self.env = env
        # Initial parameter values: see source code
        # https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py#L95
        self.g = 9.81  # gravity
        self.l = 1.   # length of the pole 
        self.m = 1.  # ball mass
        self.b = 0.  # friction
        
        self.A = np.array([[0., 1.],
                           [self.g / self.l, -self.b / (self.m * self.l**2)]])

        # input matrix
        self.B = np.array([[0.], [1. / (self.m * self.l**2)]])

        # penalties for lqr
        self.R = 0.1 * np.eye(1)
        self.Q = np.diag([1., 1.])
        self.K = LQR.lqr(self.A, self.B, self.Q, self.R)

    def f(self, x, t) :
        # theta and theta_dot
        x1, x2 = x
        # observation is (x, y, theta_dot)
        # observation = np.array([np.cos(x1), np.sin(x1), x2])
        # get torque using LQR solution
        # K is a numpy array with weights [[20.11708979  7.08760747]]
        u = -(self.K[0, 0]*x1 + self.K[0, 1]*x2)
        # system dynamics as list
        x1_dot = x2
        x2_dot = (self.m*self.g*self.l*np.sin(x1) + u - self.b*x2) / (self.m * self.l**2)
        x_dot = [x1_dot, x2_dot]
        return x_dot
    
    def get_system(self):
        return self.A, self.B, self.Q, self.R, self.K

    def get_are(self):
        '''
        Get solution for coninuous algebraic ricatti equation
        '''
        P=linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
        return P
    
    def lyapunov_function(self, x):
        # algebraic ricatti equation
        P=linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
        return x.T @ P @ x

    @staticmethod
    def lqr(A,B,Q,R):
        '''
        Method taked from my undergrad class (AE 353 at UIUC with Tim Bretl)
        https://tbretl.github.io/ae353-sp22/reference#lqr
        '''
        # algebraic ricatti equation
        P=linalg.solve_continuous_are(A,B,Q,R)
        # side note: v = xTPx is the lyapunov function
        K=linalg.inv(R) @ B.T @ P
        return K

    @staticmethod
    def get_input(K, observation):
        # x is by default a vector of length 3 (x, y, theta_dot)
        # convert x and y to theta with inverse tangent
        # u = -Kx
        x_val, y_val, theta_dot = observation
        # subtract 90 degrees because 0 degrees is at top
        theta = np.arctan2(x_val, -y_val) - (np.pi / 2)
        x = np.array([theta, theta_dot])
        u = -np.dot(K, x)
        # set min and max bounds for torque
        # input is scalar
        return u

    def control(self, noisy_observer=False):
        observation, info = self.env.reset()
        initial_state = np.random.uniform([-np.pi/2, -0.5], [np.pi/2, 0.5], size=2)
        self.env.unwrapped.state = initial_state
        for _ in range(1000):
            # get action and input (force) required 
            action = LQR.get_input(self.K, observation)

            observation, reward, terminated, truncated, info = self.env.step(action)

            # add small noise to observer (position velocity, pole angle, pole angular velocity)
            if noisy_observer:
                observation += np.random.normal(0, 0.15, 3)
            if terminated or truncated:
                observation, info = self.env.reset()
                initial_state = np.random.uniform([-np.pi/2, -0.5], [np.pi/2, 0.5], size=2)
                self.env.unwrapped.state = initial_state

        self.env.close()

def checkStability(A, B, K):
    '''
    Prove closed loop stability
    '''
    s = A - B@K
    eigenvalues, eigenvectors = np.linalg.eig(s)
    assert(np.all(eigenvalues.real < 0))

if __name__ == '__main__':
    show_gui = False
    noisy_observer = False
    env = gym.make('Pendulum-v1', render_mode="human", g=9.81)

    lqr = LQR(env)
    A, B, Q, R, K = lqr.get_system()

    checkStability(A, B, K)

    if show_gui:
        lqr.control(noisy_observer=noisy_observer)

    # plotting
    # algebraic ricatti equation gives p and V = x^TPx is lyapunov function
    P = lqr.get_are()
    # print(V_lqr.shape)
    f = lqr.f
    # # TODO Plot results
    plot_2D_roa(P, f)


