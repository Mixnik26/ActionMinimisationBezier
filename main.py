import numpy as np
import matplotlib.pyplot as plt
from math import comb as nCr
from scipy.integrate import quad
from scipy.optimize import minimize

class BezierCurve:
    '''
    Class to represent a Bezier curve defined by control points.
    '''
    def __init__(self, control_points: np.ndarray[np.ndarray]):
        # Validate control points and define degree
        self.control_points = np.array(control_points)
        self.degree = len(self.control_points) - 1

    def curve(self, t: float):
        '''
        Calculate the Bezier curve point at parameter t as a Bernstein polynomial.
        '''
        # Calculate the Bezier curve point at parameter t
        terms = [nCr(self.degree, i) * (1-t)**(self.degree-i) * t**i * self.control_points[i] for i in range(self.degree + 1)]
        return sum(terms)
    
    def curve_deriv(self, t: float):
        '''
        Calculate the derivative of the Bezier curve at parameter t.
        '''
        # Calculate the derivative of the Bezier curve at parameter t
        terms = [nCr(self.degree-1, i) * (1-t)**(self.degree-i-1) * t**i * (self.control_points[i+1]-self.control_points[i]) for i in range(self.degree)]
        return self.degree*sum(terms)

class ActionMinimiser:
    '''
    Class to handle the minimization of the action functional for a given Lagrangian.
    The action is defined as the integral of the Lagrangian over the path defined by a Bezier curve.
    '''
    def __init__(self, lagrangian, degree, initial_pos, final_pos):
        # Initialize the minimization parameters
        self.lagrangian = lagrangian
        self.degree = degree
        self.initial_pos = initial_pos
        self.final_pos = final_pos

        # Define the action function to be minimized
        def func_to_minimise(control_points: np.ndarray):
            # Reshape control points to 2D array and introduce boundariy conditions
            control_points = control_points.reshape(degree-1, len(control_points)//(degree-1))
            control_points = np.vstack((initial_pos, control_points, final_pos))

            # Define the Bezier curve and compute the action
            bezier = BezierCurve(control_points)
            action = quad(lambda t: lagrangian(t, bezier.curve(t), bezier.curve_deriv(t)), 0, 1)[0]

            # Return the action to be minimized
            return action
        self.action = func_to_minimise

    def minimise(self, initial_guess, **kwargs):
        '''
        Function to handle the minimization of the action function.
        Inputs:
        - initial_guess: Initial guess for the control points (2D array)
        - kwargs: Additional arguments for the minimization function
        Outputs (if successful minimization):
        - control_points: Optimized control points (2D array)
        - action: Optimized action value (float)
        '''
        # Reshape the initial guess for scipy's minimize function
        initial_guess = np.ndarray.flatten(np.array(initial_guess))
        # Minimize the action using scipy's minimize function
        minimize_result = minimize(self.action, initial_guess, **kwargs)
        if minimize_result.success:
            print("Minimisation successful!")
            control_points = minimize_result.x.reshape(self.degree-1, len(minimize_result.x)//(self.degree-1))
            control_points = np.vstack((self.initial_pos, control_points, self.final_pos))
            self.control_points = control_points
            self.min_action = minimize_result.fun
            return control_points, minimize_result.fun
        else:
            print("Minimisation failed.")
    
    def plot_bezier_curve(self, num_points=100):
        '''
        Function to plot the Bezier curve defined by the optimized control points.
        Inputs:
        - num_points: Number of points to plot on the curve (int)
        '''
        bezier = BezierCurve(self.control_points)
        t_vals = np.linspace(0, 1, num_points)
        curve_points = np.array([bezier.curve(t) for t in t_vals])
        
        plt.plot(curve_points[:, 0], curve_points[:, 1], label='Bezier Curve')
        plt.scatter(self.control_points[:, 0], self.control_points[:, 1], color='red', label='Control Points')
        plt.title('Bezier curve solution')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid()
        plt.show()

def plot_2d_parameter_space(action, n=20):
    x = np.linspace(0, 1, n)
    y = np.linspace(-.1, -5, n)
    z = np.array([action(np.array([i,j])) for j in y for i in x])

    X, Y = np.meshgrid(x, y)
    Z = z.reshape(n, n)

    plt.contourf(X, Y, Z)
    plt.colorbar(label='Action')
    plt.title('Parameter Space')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

if __name__ == "__main__":
    # Example Lagrangian
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