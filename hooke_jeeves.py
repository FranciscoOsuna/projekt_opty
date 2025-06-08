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
        Exploratory search around base_point with step delta.
        Returns an improved point or the base_point if no improvement found.
        """
        x_new = base_point.copy()
        f_base = self.func(x_new)

        for i in range(len(x_new)):
            improved = False
            for direction in [+1, -1]:
                x_trial = x_new.copy()
                x_trial[i] += direction * delta
                f_trial = self.func(x_trial)
                if f_trial < f_base:
                    x_new = x_trial
                    f_base = f_trial
                    improved = True
                    break  # Only take one successful direction per axis
            # Continue to next dimension regardless of improvement
        return x_new

    def optimize(self, max_iter=100, tol=1e-6):
        """
        Perform Hooke-Jeeves optimization.
        max_iter: maximum number of iterations
        tol: tolerance for step_size
        Returns best found x
        """
        base_point = self.x.copy()
        delta = self.step_size
        self.history.append(base_point.copy())

        for iteration in range(max_iter):
            # Exploratory search
            new_point = self.explore(base_point, delta)
            f_base = self.func(base_point)
            f_new = self.func(new_point)

            if f_new < f_base:
                # Pattern move attempt
                pattern_point = new_point + self.alpha * (new_point - base_point)
                f_pattern = self.func(pattern_point)

                if f_pattern < f_new:
                    # Move along pattern direction if it helps
                    base_point = self.explore(pattern_point, delta)
                else:
                    base_point = new_point
            else:
                # No improvement, reduce step size
                delta *= 0.5

            self.history.append(base_point.copy())

            if delta < tol:
                break

        self.x = base_point
        return self.x