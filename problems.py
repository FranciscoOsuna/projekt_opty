import numpy as np

def rosenbrock(x):
    a, b = 1, 100
    return (a - x[0])**2 + b*(x[1] - x[0]**2)**2

def booth(x):
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2

def three_hump_camel(x):
    return 2*x[0]**2 - 1.05*x[0]**4 + x[0]**6/6 + x[0]*x[1] + x[1]**2