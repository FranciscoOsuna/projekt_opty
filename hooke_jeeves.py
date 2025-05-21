import numpy as np

class HookeJeeves:
    def __init__(self, func, x0, step_size=0.5, alpha=2.0):
        """
        func: callable, function to minimize, takes numpy array x
        x0: initial point (list or array)
        step_size: initial exploratory step
        alpha: pattern step multiplier
        """
        self.func = func
        self.x = np.array(x0, dtype=float)
        self.step_size = step_size
        self.alpha = alpha
        self.history = []

    def explore(self, base_point, delta):
        """
        Exploratory search around base_point with step delta
        Returns improved point
        """
        x_new = base_point.copy()
        f_base = self.func(x_new)
        for i in range(len(x_new)):
            for direction in [+1, -1]:
                x_trial = x_new.copy()
                x_trial[i] += direction * delta
                f_trial = self.func(x_trial)
                if f_trial < f_base:
                    x_base, f_base = x_trial, f_trial
                    x_new = x_trial
                    break
        return x_new

    def optimize(self, max_iter=100, tol=1e-6):
        """
        Perform Hooke-Jeeves optimization
        max_iter: maximum number of iterations
        tol: tolerance for step_size
        Returns best found x
        """
        base_point = self.x.copy()
        self.history.append(base_point.copy())
        delta = self.step_size

        for iteration in range(max_iter):
            # Exploratory search
            new_point = self.explore(base_point, delta)
            # Pattern move
            if self.func(new_point) < self.func(base_point):
                pattern_point = new_point + self.alpha * (new_point - base_point)
                # Further exploration around pattern_point
                base_point = self.explore(pattern_point, delta)
            else:
                # Reduce step size
                delta *= 0.5
                base_point = new_point

            self.history.append(base_point.copy())

            if delta < tol:
                break

        self.x = base_point
        return self.x