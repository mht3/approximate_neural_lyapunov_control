'''
See gymnasium cartpole environment: https://gymnasium.farama.org/environments/classic_control/cart_pole/
'''

import gymnasium as gym
import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt
import sys, os
# add module to system path
cur_path = os.path.dirname(os.path.realpath(__file__))
module_path = os.path.join(cur_path, '..', '..')
sys.path.insert(0, module_path)

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
            action = LQR.get_input(self.K, observation)
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
    
    def control(self, noisy_observer=False):
        initial_state = np.random.uniform([-np.pi/2, -0.5], [np.pi/2, 0.5], size=2)
        observation = self.reset_env(initial_state)
        for _ in range(1000):
            # get action and input (force) required 
            action = LQR.get_input(self.K, observation)

            observation, reward, terminated, truncated, info = self.env.step(action)
            # add small noise to observer (position velocity, pole angle, pole angular velocity)
            if noisy_observer:
                observation += np.random.normal(0, 0.15, 3)
            if terminated or truncated:
                initial_state = np.random.uniform([-np.pi/2, -0.5], [np.pi/2, 0.5], size=2)
                observation = self.reset_env(initial_state)


        self.env.close()

def checkStability(A, B, K):
    '''
    Prove closed loop stability
    '''
    s = A - B@K
    eigenvalues, eigenvectors = np.linalg.eig(s)
    assert(np.all(eigenvalues.real < 0))

def plot_multiple_trajectories(time, trajectories, filename='tmp.png'):
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    axs[0].axhline(0, time[0], time[-1], linestyle='--', color='red', label='Goal')
    axs[1].axhline(0, time[0], time[-1], linestyle='--', color='red', label='Goal')

    for traj in trajectories:
        label = traj['label']
        theta = traj['theta']
        theta_dot = traj['theta_dot']
        axs[0].plot(time, theta, label='{}'.format(label))

        axs[1].plot(time, theta_dot, label='{}'.format(label))
    
    axs[0].set_title(r'$\Theta$ Over Time')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel(r'$\Theta$ (rad)')
    axs[0].legend()
    axs[0].grid(True)
    
    axs[1].set_title(r'$\dot{\Theta}(t)$ Over Time')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel(r'$\dot{\Theta}$ (rad/s)')
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)

if __name__ == '__main__':
    show_gui = False
    noisy_observer = False
    if show_gui:
        env = gym.make('Pendulum-v1', render_mode="human", g=9.81)
    else: 
        env = gym.make('Pendulum-v1', g=9.81)

    lqr = LQR(env)
    A, B, Q, R, K = lqr.get_system()

    checkStability(A, B, K)

    if show_gui:
        lqr.control(noisy_observer=noisy_observer)

    # Example data for multiple trajectories
    time = np.arange(0, 5.05, 0.05)
    trajectories = []
    initial_states = np.array([[np.pi/8, 0], [np.pi/4, 0], [-np.pi/8, 0]])
    for initial_state in initial_states:
        trajectory= lqr.simulate(initial_state, time)
        trajectories.append(trajectory)
    # Call the function with the example data
    plot_multiple_trajectories(time, trajectories,'examples/inverted_pendulum/results/time_convergence_lqr.png')


