
def rosenbrock_constraint(x):
    """
    g1(x1, x2) = 1.5 - 0.5*x1 - x2 = 0
    Ponieważ jest to równanie równościowe, implementujemy je jako nierówność:
    |g1(x)| <= epsilon (np. 1e-6)
    """
    return abs(1.5 - 0.5*x[0] - x[1])

def booth_constraint(x):
    """
    g2(x1, x2) = x1^2 + 2*x1 - x2 <= 0
    """
    return x[0]**2 + 2*x[0] - x[1]

def three_hump_camel_constraint(x):
    """
    x1^2 + x2^2 <= 1
    """
    return x[0]**2 + x[1]**2 - 1



def penalty_function(constraint, x, penalty_factor=1e6):
    """
    constraint: funkcja ograniczenia zwracająca wartość:
        - <= 0 oznacza, że punkt spełnia ograniczenie
        - > 0 oznacza, że punkt narusza ograniczenie
    x: punkt do oceny
    penalty_factor: współczynnik kary
    """
    violation = max(0.0, constraint(x))
    penalty = violation ** 2
    return penalty_factor * penalty

def constrained_objective(original_func, constraint, penalty_factor=1e6):
    """
    Zwraca funkcję celu z karą za naruszenie ograniczenia.
    """
    def wrapped_func(x):
        return original_func(x) + penalty_function(constraint, x, penalty_factor)
    return wrapped_func