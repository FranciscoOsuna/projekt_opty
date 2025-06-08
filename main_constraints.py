import numpy as np
import hooke_jeeves as hj
import problems
import graph
import constraints


penalized_booth = constraints.constrained_objective(problems.booth, constraints.booth_constraint)


optimizer = hj.HookeJeeves(penalized_booth, x0=[-10, 7.5], step_size=0.5)
solution = optimizer.optimize()

print(solution)
graph.visualise(problems.booth, point=solution)
graph.animate(problems.booth, optimizer.history)