import numpy as np
from main import ActionMinimiser
import matplotlib.pyplot as plt
pi = np.pi

def generate_lagrangian(refractive_index):
    '''
    A function to generate the lagrangian for an optical path given a refractive index function n(x,y)
    '''
    def lagrangian(t, q, qdot):
        x, y = q
        xdot, ydot = qdot
        return refractive_index(x,y) * np.sqrt(xdot**2 + ydot**2)
    return lagrangian

def sigmoid(x):
    return 1/(1+np.exp(-x))

if __name__=="__main__":

    # Initialise the lagrangian for a given refractive index n(x,y)
    # Choose any (sensible differentiable) refractive index function and see how the algorithm works
    n = lambda x, y: 1 + .4*sigmoid(200*(y-2.5))
    lagrangian = generate_lagrangian(refractive_index=n)
    
    # Boundary conditions
    initial_pos = np.array([0, 0])
    final_pos = np.array([5, 5])

    # Bezier parameters and initial comtrol points guess
    degree = 4
    initial_guess = (final_pos-initial_pos)*np.random.rand(degree-1, 2) + initial_pos

    # Create an instance of ActionMinimiser and minimize the action for the optics problem
    opticalPath = ActionMinimiser(lagrangian=lagrangian, degree=degree, initial_pos=initial_pos, final_pos=final_pos)
    opticalPath.minimise(initial_guess=initial_guess)
    print(f"Total action: {opticalPath.S}")

    # Plot solution with refractive index contour plot
    x = np.linspace(min(opticalPath.control_points[:,0]), max(opticalPath.control_points[:,0]), 100)
    y = np.linspace(min(opticalPath.control_points[:,1]), max(opticalPath.control_points[:,1]), 100)
    X, Y = np.meshgrid(x,y)
    Z = n(X, Y)
    plt.contourf(X, Y, Z, 50)
    plt.colorbar(label="Refractive index")

    opticalPath.plot_bezier_curve()

    plt.show()