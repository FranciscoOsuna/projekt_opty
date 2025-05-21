import numpy as np

class GeneralizedPatternSearch:
    def __init__(self, func, x0, step_size=0.5):
        """
        func: callable, function to minimize, takes numpy array x
        x0: initial point (list or array)
        step_size: initial exploratory step size
        """
        self.func = func
        self.x = np.array(x0, dtype=float)
        self.step_size = step_size
        self.history = []

    def explore(self, base_point, delta):
        """
        Evaluates function in all positive and negative coordinate directions.
        Returns best point found and its function value.
        """
        best_point = base_point.copy()
        best_value = self.func(base_point)

        for i in range(len(base_point)):
            for direction in [+1, -1]:
                x_trial = base_point.copy()
                x_trial[i] += direction * delta
                f_trial = self.func(x_trial)

                if f_trial < best_value:
                    best_point = x_trial
                    best_value = f_trial

        return best_point, best_value

    def optimize(self, max_iter=100, tol=1e-6):
        """
        Perform Generalized Pattern Search optimization.
        Returns best found x.
        """
        current_point = self.x.copy()
        delta = self.step_size
        self.history.append(current_point.copy())

        for iteration in range(max_iter):
            new_point, new_value = self.explore(current_point, delta)
            current_value = self.func(current_point)

            if new_value < current_value:
                current_point = new_point
            else:
                delta *= 0.5  # reduce step size

            self.history.append(current_point.copy())

            if delta < tol:
                break

        self.x = current_point
        return self.x
