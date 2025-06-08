import numpy as np

def rosenbrock(x):
    a, b = 1, 100
    return (a - x[0])**2 + b*(x[1] - x[0]**2)**2

def booth(x):
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2

def three_hump_camel(x):
    return 2*x[0]**2 - 1.05*x[0]**4 + x[0]**6/6 + x[0]*x[1] + x[1]**2

def penalty_function(constraints, x, penalty_factor=1e6):
    """
    constraints: list of constraint functions, each returning a signed value:
        - <= 0 means constraint is satisfied
        - > 0 means constraint is violated
    x: point to evaluate
    penalty_factor: scaling factor for penalty term
    """
    penalty = 0.0
    for g in constraints:
        violation = max(0.0, g(x))  # for inequality constraints
        penalty += violation ** 2   # squared penalty
    return penalty_factor * penalty

def constrained_objective(original_func, constraints, penalty_factor=1e6):
    """
    Returns a new objective function that includes penalty terms for constraints.
    """
    def wrapped_func(x):
        return original_func(x) + penalty_function(constraints, x, penalty_factor)
    return wrapped_func