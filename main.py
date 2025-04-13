import numpy as np
import matplotlib.pyplot as plt
from math import comb as nCr

class BezierCurve:
    def __init__(self, control_points):
        self.control_points = np.array(control_points)
        self.degree = len(self.control_points) - 1

    def curve(self, t):
        terms = [nCr(self.degree, i) * (1-t)**(self.degree-i) * t**i * self.control_points[i] for i in range(self.degree + 1)]
        return sum(terms)
    
    def curve_deriv(self, t):
        terms = 