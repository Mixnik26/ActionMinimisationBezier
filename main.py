import numpy as np
import matplotlib.pyplot as plt
from math import comb as nCr
from scipy.integrate import quad
from scipy.optimize import minimize
from sympy import plot

class BezierCurve:
    def __init__(self, control_points):
        self.control_points = np.array(control_points)
        self.degree = len(self.control_points) - 1

    def curve(self, t):
        terms = [nCr(self.degree, i) * (1-t)**(self.degree-i) * t**i * self.control_points[i] for i in range(self.degree + 1)]
        return sum(terms)
    
    def curve_deriv(self, t):
        terms = [nCr(self.degree-1, i) * (1-t)**(self.degree-i-1) * t**i * (self.control_points[i+1] - self.control_points[i]) for i in range(self.degree)]
        return self.degree*sum(terms)

def Lagrangian(t, q, qdot):
    x, y = q
    xdot, ydot = qdot
    return np.sqrt(xdot**2 + ydot**2) + y**2

def generate_func_to_minimise(qi, qf, degree):

    def func_to_minimise(control_points):
        control_points = control_points.reshape(degree-1, len(control_points)//(degree-1))
        control_points = np.vstack((qi, control_points, qf))
        bezier = BezierCurve(control_points)
        action = quad(lambda t: Lagrangian(t, bezier.curve(t), bezier.curve_deriv(t)), 0, 1)[0]
        return action

    return func_to_minimise

def plot_bezier_curve(control_points, num_points=100):
    bezier = BezierCurve(control_points)
    t_vals = np.linspace(0, 1, num_points)
    curve_points = np.array([bezier.curve(t) for t in t_vals])
    
    plt.plot(curve_points[:, 0], curve_points[:, 1], label='Bezier Curve')
    plt.scatter(control_points[:, 0], control_points[:, 1], color='red', label='Control Points')
    plt.title('Bezier Curve')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.show()

def plot_parameter_space(action):
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    z = np.array([action(np.array([i,j])) for j in y for i in x])

    X, Y = np.meshgrid(x, y)
    Z = z.reshape(10, 10)

    plt.contourf(X, Y, Z)
    plt.colorbar(label='Action')
    plt.title('Parameter Space')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

if __name__ == "__main__":
    qi = np.array([0, 0])
    qf = np.array([1, -1])

    action = generate_func_to_minimise(qi, qf, degree=2)
    minimize_result = minimize(action, np.random.rand(2))
    
    if minimize_result.success:
        print("Optimization successful!")
        print("Action:", minimize_result.fun)
        control_points = minimize_result.x.reshape(1, 2)
        control_points = np.vstack((qi, control_points, qf))
        plot_bezier_curve(control_points)
    else:
        print("Optimization failed.")
    