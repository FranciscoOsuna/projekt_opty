import numpy as np
import hooke_jeeves as hj
import problems
import graph

optimizer = hj.HookeJeeves(problems.booth, x0 = [-5, -5], step_size=0.5)
solution = optimizer.optimize()

print(solution)
graph.visualise(problems.booth, point=solution)
graph.animate(problems.booth, optimizer.history)
