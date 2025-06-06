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
    def __init__(self, lagrangian, degree, initial_pos, final_pos, t_bounds = None):
        # Initialize the minimization parameters
        self.lagrangian = lagrangian
        self.degree = degree
        self.initial_pos = initial_pos
        self.final_pos = final_pos


        # Define the action function to be minimized
        # If t_bounds is not provided, assume the lagrangian is the special case where t_i and t_f are not required
        if t_bounds is None:
            def _func_to_minimise(control_points: np.ndarray):
                '''
                From a 1D array of control points (for scipy's minimise), compute the action along the Bezier curve described by said control points using the formula:
                S = ∫ L[B(s), dB(s)/ds] ds
                '''
                # Reshape control points to 2D array and introduce boundary conditions
                control_points = control_points.reshape(degree-1, len(control_points)//(degree-1))
                control_points = np.vstack((initial_pos, control_points, final_pos))

                # Define the Bezier curve and compute the action
                bezier = BezierCurve(control_points)
                action = quad(lambda t: lagrangian(t, bezier.curve(t), bezier.curve_deriv(t)), 0, 1)[0]

                # Return the action to be minimized
                return action
                
        else: # Otherwise, use t_bounds to define the action
            delta_t = t_bounds[1] - t_bounds[0]
            def _func_to_minimise(control_points: np.ndarray):
                '''
                From a 1D array of control points (for scipy's minimise), compute the action along the Bezier curve described by said control points using delta_t as per the formula:
                S = ∫ delta_t * L[t_i + delta_t*s, B(s), (1/delta_t) * dB(s)/ds] ds
                '''
                # Reshape control points to 2D array and introduce boundary conditions
                control_points = control_points.reshape(degree-1, len(control_points)//(degree-1))
                control_points = np.vstack((initial_pos, control_points, final_pos))

                # Define the Bezier curve and compute the action
                bezier = BezierCurve(control_points)
                action = quad(lambda t: delta_t*lagrangian(t[0] + delta_t*t, bezier.curve(t), (1/delta_t)*bezier.curve_deriv(t)), 0, 1)[0]

                # Return the action to be minimized
                return action
        self._action_func = _func_to_minimise

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
        if self.degree == 1:
            print("Bezier curve is of first degree and thus cannot be optimized.")
            control_points = np.vstack((self.initial_pos, self.final_pos))
            self.control_points = control_points
            # Define the Bezier curve and compute the action along it
            bezier = BezierCurve(control_points)
            self.action = quad(lambda t: self.lagrangian(t, bezier.curve(t), bezier.curve_deriv(t)), 0, 1)[0]
            self.S = self.action
            return control_points, self.action

        # Reshape the initial guess for scipy's minimize function
        initial_guess = np.ndarray.flatten(np.array(initial_guess))
        # Minimize the action using scipy's minimize function
        minimize_result = minimize(self._action_func, initial_guess, **kwargs)

        if minimize_result.success:
            print("Minimisation successful!")
            # Reshape control points for readability and store the result
            control_points = minimize_result.x.reshape(self.degree-1, len(minimize_result.x)//(self.degree-1))
            control_points = np.vstack((self.initial_pos, control_points, self.final_pos))
            self.control_points = control_points
            self.action = minimize_result.fun
            self.S = self.action
            return control_points, minimize_result.fun
        else:
            # Otherwise return a failure
            print("Minimisation failed: " + minimize_result.message)
            self.control_points = None
            self.action = None
            self.S = self.action
    
    def plot_bezier_curve(self, num_points=100):
        '''
        Function to plot the Bezier curve defined by the optimized control points.
        Inputs:
        - num_points: Number of points to plot on the curve (int)
        '''
        bezier = BezierCurve(self.control_points)
        t_vals = np.linspace(0, 1, num_points)
        curve_points = np.array([bezier.curve(t) for t in t_vals])
        
        plt.plot(curve_points[:, 0], curve_points[:, 1], label='Bezier Curve', color="red")
        plt.scatter(self.control_points[:, 0], self.control_points[:, 1], color='black', label='Control Points')
        plt.title('Bezier curve solution')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid()