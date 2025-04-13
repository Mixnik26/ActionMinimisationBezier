import numpy as np
from main import ActionMinimiser

# Lagrangian for the Brachistochrone problem
def Lagrangian(t, q, qdot):
    x, y = q
    xdot, ydot = qdot
    return np.sqrt((xdot**2 + ydot**2)/(-2*y))

# Initial conditions
initial_pos = np.array([0, 0])
final_pos = np.array([3, -1])
degree = 4
initial_guess = [[.5, -.5] for _ in range(degree-1)]

# Create an instance of ActionMinimiser and minimize the action for the Brachistochrone problem
Brachistochrone = ActionMinimiser(Lagrangian, degree, initial_pos, final_pos)
Brachistochrone.minimise(initial_guess)
Brachistochrone.plot_bezier_curve()