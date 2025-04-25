import numpy as np
from main import ActionMinimiser
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve

# Lagrangian for the Brachistochrone problem using g=1
def Lagrangian(t, q, qdot):
    x, y = q
    xdot, ydot = qdot
    return np.sqrt((xdot**2 + ydot**2)/(-2*y))

# Analytic solution for some parameters of the Brachistochrone problem assuming the particle starts at the origin
def analytic_parameters(final_pos):
    '''
    Analytic solution for the Brachistochrone problem from the "Indirect method" section on wikipedia: https://en.wikipedia.org/wiki/Brachistochrone_curve
    '''
    # The final position is given by the coordinates of the final point
    x_f = final_pos[0]
    y_f = -final_pos[1]

    # The final angular parameter is given by the solution of the equation
    def f(phi_f):
        return (phi_f - np.sin(phi_f)) - (x_f/y_f)*(1 - np.cos(phi_f))

    # Solve the equation for phi_f using fsolve
    # the 1-norm of final_pos seems to be a good initial guess
    phi_f = fsolve(f, np.abs(x_f) + np.abs(y_f), xtol = 1e-10)[0]

    # Calculate the final time using the other parameter r and substitute into t_f = sqrt(r/g)*phi_f
    # Assuming g=1 for simplicity
    r = y_f/(1 - np.cos(phi_f))
    t_f = phi_f * np.sqrt(r)
    return t_f, r, phi_f

def analytic_solution(phi, r):
    '''
    Analytic solution for the Brachistochrone problem from the "Indirect method" section on wikipedia: https://en.wikipedia.org/wiki/Brachistochrone_curve
    '''
    return np.array([r * (phi -  np.sin(phi)), -r * (1 - np.cos(phi))])


if __name__ == "__main__":
    # Initial conditions
    initial_pos = np.array([0, 0])
    final_pos = np.array([3, -1])

    # Calculate the analytic final time
    t_f, r, phi_f = analytic_parameters(final_pos)

    # Define the range of degrees for the Bezier curve and collect the total time for the particle to travel along the curve under a gravity of g=1
    degrees = np.arange(1, 6)
    final_times_error = []
    for d in degrees:
        initial_guess = [[.5, -.5] for _ in range(d-1)]
        # Create an instance of ActionMinimiser and minimize the action for the Brachistochrone problem
        Brachistochrone = ActionMinimiser(Lagrangian, d, initial_pos, final_pos)
        Brachistochrone.minimise(initial_guess)
        # Append the deviation of final time to the analytical solution to the list of final time errors
        final_times_error.append(Brachistochrone.S - t_f)

    # Plot the final degree bezier curve and the analytic solution
    Brachistochrone.plot_bezier_curve()
    phi = np.linspace(0, phi_f, 100)
    plt.plot(analytic_solution(phi, r)[0], analytic_solution(phi, r)[1], ls="--", color="red", label='Analytic solution')
    plt.legend()
    plt.show()

    # Fit the final time errors to a line in log space
    # This is done to see if the error scales with the degree of the Bezier curve
    fit = curve_fit(lambda x, a, b: a*x + b, degrees, np.log(final_times_error))
    a, b = fit[0]

    # Plot the final time errors and the fitted line
    plt.title('Action errors vs degree of Bezier curve')
    plt.scatter(degrees, np.log(final_times_error), label='Final time errors')
    plt.plot(degrees, a*np.array(degrees) + b, label=f'Fitted line with slope {a:.2f}')
    plt.xlabel('Degree of Bezier curve')
    plt.ylabel('Log of action error')
    plt.legend()
    plt.show()