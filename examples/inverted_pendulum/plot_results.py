import sys, os
# add module to system path
cur_path = os.path.dirname(os.path.realpath(__file__))
module_path = os.path.join(cur_path, '..', '..')
sys.path.insert(0, module_path)
from lyapunov_policy_optimization.utils import plot_2D_roa, Plot3D
from lqr import LQR

if __name__ == '__main__':
    # get equations for inverted pendulum
    lqr = LQR()

    # algebraic ricatti equation gives p and V = x^TPx is lyapunov function
    P = lqr.get_are()
    # print(V_lqr.shape)
    f = lqr.f
    # # TODO Plot results
    Plot3D(P, r=6)
    plot_2D_roa(P, f)