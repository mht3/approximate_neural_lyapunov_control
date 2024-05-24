import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

def Plot3D(P, r=6):
    X = np.linspace(-6, 6, 100) 
    Y = np.linspace(-6, 6, 100)
    x1, x2 = np.meshgrid(X,Y)
    V = P[0, 0]*x1**2 + 2*P[0, 1]*x1*x2 + P[1, 1]*x2**2        

    # Plot Lyapunov functions  
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, V-2.6, rstride=5, cstride=5, alpha=0.5, cmap=cm.coolwarm)
    ax.contour(X, Y, V-2.6, 10, zdir='z', offset=0, cmap=cm.coolwarm)
    
    # Plot Valid region computed by dReal
    theta = np.linspace(0,2*np.pi,50)
    xc = r*np.cos(theta)
    yc = r*np.sin(theta)
    ax.plot(xc[:],yc[:],'r',linestyle='--', linewidth=2 ,label='Valid region')
    plt.legend(loc='upper right')

    ax.set_xlabel('$\Theta$')
    ax.set_ylabel('$\dot{\Theta}$')
    ax.set_zlabel('V')
    plt.title('Lyapunov Function')
    plt.show()

def Plotflow(Xd, Yd, t, f):
    # Plot phase plane 
    DX, DY = f([Xd, Yd],t)
    DX=DX/np.linalg.norm(DX, ord=2, axis=1, keepdims=True)
    DY=DY/np.linalg.norm(DY, ord=2, axis=1, keepdims=True)
    plt.streamplot(Xd,Yd,DX,DY, color=('gray'), linewidth=0.5,
                  density=0.5, arrowstyle='-|>', arrowsize=1.5)
    
def plot_2D_roa(P, f):
    '''
    Plot Region of attraction for systems with 2 state variables
    '''
    fig = plt.figure(figsize=(8,6))
    X = np.linspace(-6, 6, 100) 
    Y = np.linspace(-6, 6, 100)
    x1, x2 = np.meshgrid(X, Y)

    ax = plt.gca()
    # Vaild Region
    C = plt.Circle((0, 0),6, color='r', linewidth=1.5, fill=False)
    ax.add_artist(C)

    # plot direction field
    xd = np.linspace(-6, 6, 10) 
    yd = np.linspace(-6, 6, 10)
    Xd, Yd = np.meshgrid(xd,yd)
    t = np.linspace(0,2,100)
    Plotflow(Xd, Yd, t, f) 
    # get lyapunov function (x^T P x) for each pair of x,y coordinates
    V_lqr = P[0, 0]*x1**2 + 2*P[0, 1]*x1*x2 + P[1, 1]*x2**2
    # why subtract 2.6?
    # plot contour of lyapunov wehere function equals 2.6 V(x) = 2.6
    ax.contour(X, Y, V_lqr-2.6, 0, linewidths=2, colors='m', linestyles='--', label='lqr')
    plt.title('Region of Attraction')
    plt.legend([plt.Rectangle((0,0),1,2,color='k',fill=False,linewidth = 2),
                plt.Rectangle((0,0),1,2,color='m',fill=False,linewidth = 2,linestyle='--'), C]\
               ,['NN','LQR', 'Valid Region'],loc='upper right')
    plt.xlabel(r'Angle, $\theta$ (rad)')
    plt.ylabel(r'Angular velocity $\dot{\theta}$')
    plt.show()